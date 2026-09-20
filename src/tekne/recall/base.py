"""Recaller interface.

Recallers are deliberately cheap to write and deliberately over-eager: the
design puts *all* precision downstream, in typing, verification and the guard
stack.  The only thing a recaller owes the rest of the system is that every span
it emits is a true slice of ``Document.text`` at the offsets it reports.

``cost_tier`` drives the escalation cascade in the orchestrator:

    0  deterministic, microseconds        -- always run
    1  neural, milliseconds               -- run unless the budget is exhausted
    2  LLM, hundreds of milliseconds + $  -- run only on the uncertain residue
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..nlp import Analysis
from ..schema import Candidate, Document, Span

if TYPE_CHECKING:  # pragma: no cover
    from ..agents.budget import Budget
    from ..llm.backend import LLMBackend


@dataclass
class RecallContext:
    document: Document
    analysis: Analysis
    #: Corpus-level term statistics, when a corpus pass has been made.
    corpus_stats: Any = None
    gazetteer: Any = None
    llm: LLMBackend | None = None
    budget: Budget | None = None
    #: Spans already proposed by cheaper tiers; tier-2 recallers use this to
    #: avoid paying for what is already covered.
    covered: list[Span] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


class Recaller(ABC):
    name: str = "recaller"
    version: str = "0"
    cost_tier: int = 0

    @abstractmethod
    def propose(self, ctx: RecallContext) -> list[Candidate]:
        """Return candidate spans. Duplicates across recallers are expected."""

    def _candidate(
        self,
        doc: Document,
        start: int,
        end: int,
        *,
        features: dict[str, float] | None = None,
        notes: dict[str, Any] | None = None,
    ) -> Candidate | None:
        """Build a candidate, returning ``None`` if the offsets are unusable.

        Every recaller funnels through here so that an off-by-one in a regex
        cannot leak a span whose surface disagrees with the document.
        """
        start, end = _trim(doc.text, start, end)
        if end - start < 2:
            return None
        return Candidate(
            span=Span(start=start, end=end, surface=doc.text[start:end]),
            proposers=(self.name,),
            features=features or {},
            notes=notes or {},
        )


_TRIM_LEFT = " \t\n([{\"'“‘,;:"
_TRIM_RIGHT = " \t\n)]}\"'”’,;:.!?"


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))
    while start < end and text[start] in _TRIM_LEFT:
        start += 1
    while end > start and text[end - 1] in _TRIM_RIGHT:
        end -= 1
    # Rebalance brackets that survived trimming: "(CNN" -> "CNN".
    frag = text[start:end]
    if frag.count("(") > frag.count(")") and "(" in frag:
        idx = frag.rfind("(")
        if idx > len(frag) // 2:
            end = start + idx
    return start, end


def merge_candidates(groups: list[list[Candidate]]) -> list[Candidate]:
    """Union candidates from several recallers, keyed on exact offsets."""
    by_span: dict[tuple[int, int], Candidate] = {}
    for group in groups:
        for cand in group:
            key = (cand.span.start, cand.span.end)
            existing = by_span.get(key)
            by_span[key] = cand.merged_with(existing) if existing else cand
    return sorted(by_span.values(), key=lambda c: (c.span.start, -c.span.end))
