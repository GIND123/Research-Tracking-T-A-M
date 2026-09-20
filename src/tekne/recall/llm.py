"""Open-vocabulary recall from a language model.

This is the only recaller that can name a technology no lexicon has heard of,
which is the whole reason it is here -- and it is also the only one that can
invent one.  The asymmetry is handled at the boundary: whatever the model
returns is immediately re-grounded against the document, and anything that
cannot be located is dropped before it becomes a :class:`Candidate`.  The count
of dropped items is kept, because "fraction of model proposals that were not in
the text" is the hallucination rate we report.

Cost discipline: the recaller is tier 2, runs only over windows that cheaper
recallers left thin, and caps itself on the budget.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..agents.prompts import PROMPT_VERSIONS, PROPOSER_SYSTEM, PROPOSER_USER
from ..guard.grounding import GroundingStatus, ground
from ..guard.injection import wrap_untrusted
from ..guard.structured import StructuredCaller
from ..schema import Candidate, SectionKind
from .base import RecallContext, Recaller


class ProposedMention(BaseModel):
    surface: str = Field(description="exact span copied from the document")
    evidence: str = Field(description="sentence copied from the document containing the surface")
    type: str = Field(default="artifact")


class ProposerOutput(BaseModel):
    mentions: list[ProposedMention] = Field(default_factory=list)


class LLMRecaller(Recaller):
    name = "llm"
    version = PROMPT_VERSIONS["proposer"]
    cost_tier = 2

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-5",
        max_windows: int = 3,
        window_chars: int = 2800,
        max_input_chars: int = 9000,
        allow_fuzzy_grounding: bool = False,
    ) -> None:
        self.model = model
        self.max_windows = max_windows
        self.window_chars = window_chars
        # Hard ceiling on what one document may send, independent of how many
        # windows survive selection. Long specifications are the case where
        # window selection alone is not enough of a brake.
        self.max_input_chars = max_input_chars
        self.allow_fuzzy_grounding = allow_fuzzy_grounding
        self.stats: dict[str, int] = {
            "proposed": 0,
            "grounded_exact": 0,
            "grounded_relaxed": 0,
            "ungrounded": 0,
            "windows": 0,
            "chars_sent": 0,
            "windows_skipped_budget": 0,
        }

    def propose(self, ctx: RecallContext) -> list[Candidate]:
        backend = ctx.llm
        if backend is None or not backend.available:
            return []
        budget = ctx.budget
        caller = StructuredCaller(backend, max_repairs=1)
        doc = ctx.document
        out: list[Candidate] = []

        sent_chars = 0
        for section_kind, lo, hi in self._windows(ctx):
            if budget is not None and not budget.can_spend_llm_call():
                self.stats["windows_skipped_budget"] += 1
                break
            if sent_chars + (hi - lo) > self.max_input_chars:
                self.stats["windows_skipped_budget"] += 1
                break
            self.stats["windows"] += 1
            window_text = doc.text[lo:hi]
            sent_chars += hi - lo
            self.stats["chars_sent"] += hi - lo
            user = PROPOSER_USER.format(
                genre=doc.genre.value,
                section=section_kind.value,
                document=wrap_untrusted(window_text),
            )
            result = caller.call(
                ProposerOutput,
                system=PROPOSER_SYSTEM,
                user=user,
                model=self.model,
                max_tokens=2048,
            )
            if budget is not None:
                for response in result.responses:
                    budget.record_llm_call(response.usage, self.model)
            if not result.ok or result.value is None:
                if budget is not None:
                    budget.record_schema_failure()
                continue
            if budget is not None:
                budget.record_success()

            for proposal in result.value.mentions:  # type: ignore[union-attr]
                self.stats["proposed"] += 1
                cand = self._ground_proposal(ctx, proposal, lo, hi)
                if cand is not None:
                    out.append(cand)
        return out

    # -- grounding ----------------------------------------------------------

    def _ground_proposal(
        self, ctx: RecallContext, proposal: ProposedMention, lo: int, hi: int
    ) -> Candidate | None:
        text = ctx.document.text

        # Ground the evidence sentence first, then search for the surface inside
        # it. Locating the span within its cited sentence rules out the failure
        # where a real term is attributed to a sentence that does not contain it.
        evidence_hit = ground(text, proposal.evidence, window=(lo, hi))
        if evidence_hit.grounded and evidence_hit.span is not None:
            search_window = (evidence_hit.span.start, evidence_hit.span.end)
        else:
            search_window = (lo, hi)

        hit = ground(
            text,
            proposal.surface,
            window=search_window,
            allow_fuzzy=self.allow_fuzzy_grounding,
        )
        if not hit.grounded and search_window != (lo, hi):
            hit = ground(
                text, proposal.surface, window=(lo, hi), allow_fuzzy=self.allow_fuzzy_grounding
            )

        if not hit.grounded or hit.span is None:
            self.stats["ungrounded"] += 1
            return None
        if hit.status is GroundingStatus.EXACT:
            self.stats["grounded_exact"] += 1
        else:
            self.stats["grounded_relaxed"] += 1

        notes: dict[str, Any] = {
            "path": "llm",
            "llm_type": proposal.type,
            "grounding": hit.status.value,
            "evidence_grounded": bool(evidence_hit.grounded),
        }
        return self._candidate(
            ctx.document,
            hit.span.start,
            hit.span.end,
            features={
                "llm_proposed": 1.0,
                "grounding_similarity": hit.similarity,
                "grounding_exact": 1.0 if hit.status is GroundingStatus.EXACT else 0.0,
            },
            notes=notes,
        )

    # -- window selection ---------------------------------------------------

    def _windows(self, ctx: RecallContext) -> list[tuple[SectionKind, int, int]]:
        """Pick the regions worth spending a model call on.

        Sections are ranked by how information-dense they are for this task --
        title and abstract first, then the zones that carry the contribution --
        and then thinned by how much the cheap recallers already found there.
        """
        doc = ctx.document
        priority = {
            SectionKind.TITLE: 0,
            SectionKind.ABSTRACT: 1,
            SectionKind.CLAIM_INDEP: 2,
            SectionKind.PATENT_SUMMARY: 3,
            SectionKind.METHOD: 3,
            SectionKind.INTRODUCTION: 4,
            SectionKind.CLAIM_DEP: 5,
            SectionKind.EXPERIMENT: 5,
            SectionKind.RESULTS: 6,
            SectionKind.PATENT_DETAIL: 7,
            SectionKind.RELATED_WORK: 7,
            SectionKind.PATENT_BACKGROUND: 8,
        }
        sections = doc.sections or []
        if not sections:
            return [(SectionKind.OTHER, 0, min(len(doc.text), self.window_chars))]

        ranked = sorted(sections, key=lambda s: (priority.get(s.kind, 9), s.span.start))
        windows: list[tuple[SectionKind, int, int]] = []
        for sec in ranked:
            lo, hi = sec.span.start, sec.span.end
            covered = sum(1 for sp in ctx.covered if sp.start >= lo and sp.end <= hi)
            span_len = hi - lo
            # Skip sections the cheap tiers already covered densely: roughly one
            # candidate per 120 characters is saturation in practice.
            if span_len > 400 and covered >= span_len / 120:
                continue
            while lo < hi and len(windows) < self.max_windows:
                end = min(lo + self.window_chars, hi)
                if end < hi:
                    cut = doc.text.rfind(". ", lo + self.window_chars // 2, end)
                    if cut > lo:
                        end = cut + 1
                windows.append((sec.kind, lo, end))
                lo = end
            if len(windows) >= self.max_windows:
                break
        return windows[: self.max_windows]
