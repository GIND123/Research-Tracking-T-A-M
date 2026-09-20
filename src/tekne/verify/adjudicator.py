"""Third-opinion adjudication for proposer/verifier disagreements.

Runs on the strongest model and sees the widest context, but only on the residue
where a cheap proposal met a cheap rejection.  That population is small -- single
-digit percentages of candidates in our runs -- which is what makes it
affordable to spend the expensive model on it.

The adjudicator is allowed to abstain, and the prompt pushes it to do so.  In a
corpus that will be aggregated into trend lines, moving a genuinely ambiguous
mention into a review queue costs one annotator-minute; guessing costs a data
point that no one will ever revisit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ..agents.prompts import ADJUDICATOR_SYSTEM, ADJUDICATOR_USER, PROMPT_VERSIONS
from ..guard.injection import wrap_untrusted
from ..guard.structured import StructuredCaller
from ..schema import Document, TechType, Verdict
from .verifier import VerifierDecision


class AdjudicatorOutput(BaseModel):
    decision: str = Field(default="abstain")
    type: str | None = Field(default=None)
    reason: str = Field(default="")


@dataclass
class Adjudication:
    verdict: Verdict
    reason: str = ""
    retyped: TechType | None = None
    source: str = "adjudicator"


class Adjudicator:
    name = "adjudicator"
    version = PROMPT_VERSIONS["adjudicator"]

    def __init__(
        self,
        backend,
        *,
        model: str = "claude-opus-5",
        context_radius: int = 900,
        budget: Any = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.context_radius = context_radius
        self.budget = budget
        self.caller = StructuredCaller(backend, max_repairs=1) if backend else None
        self.stats = {"calls": 0, "accept": 0, "reject": 0, "abstain": 0, "retyped": 0}

    @property
    def available(self) -> bool:
        return bool(self.backend and self.backend.available and self.caller)

    def adjudicate(self, mention, doc: Document, verifier: VerifierDecision) -> Adjudication:
        if not self.available:
            return Adjudication(Verdict.ABSTAIN, "adjudicator unavailable")
        if self.budget is not None and not self.budget.can_spend_llm_call():
            return Adjudication(Verdict.ABSTAIN, "budget exhausted")

        span = mention.span
        lo = max(0, span.start - self.context_radius)
        hi = min(len(doc.text), span.end + self.context_radius)
        user = ADJUDICATOR_USER.format(
            context=wrap_untrusted(doc.text[lo:hi], label="context"),
            surface=span.surface,
            type=mention.type.value,
            verifier_reason=verifier.reason or "not confirmed",
        )
        result = self.caller.call(
            AdjudicatorOutput,
            system=ADJUDICATOR_SYSTEM,
            user=user,
            model=self.model,
            max_tokens=512,
        )
        self.stats["calls"] += 1
        if self.budget is not None:
            for response in result.responses:
                self.budget.record_llm_call(response.usage, self.model)

        if not result.ok or result.value is None:
            return Adjudication(Verdict.ABSTAIN, f"adjudicator call failed: {result.error}")

        out: AdjudicatorOutput = result.value  # type: ignore[assignment]
        retyped: TechType | None = None
        if out.type:
            try:
                candidate_type = TechType(out.type)
                if candidate_type is not mention.type:
                    retyped = candidate_type
                    self.stats["retyped"] += 1
            except ValueError:
                retyped = None

        decision = (out.decision or "abstain").strip().lower()
        if decision == "accept":
            self.stats["accept"] += 1
            return Adjudication(Verdict.PASS, out.reason[:200], retyped)
        if decision == "reject":
            self.stats["reject"] += 1
            return Adjudication(Verdict.REJECT, out.reason[:200], retyped)
        self.stats["abstain"] += 1
        return Adjudication(Verdict.ABSTAIN, out.reason[:200], retyped)
