"""Dictionary recall against the compiled knowledge base.

High precision, structurally incapable of finding anything new -- which is why
it is one recaller among several rather than the system.  Its real value is as a
*consensus witness*: a span that both the parser path and the KB propose is
almost always a genuine technology, and the confidence model leans on that.
"""

from __future__ import annotations

from ..schema import Candidate
from .base import RecallContext, Recaller


class GazetteerRecaller(Recaller):
    name = "gazetteer"
    version = "1"
    cost_tier = 0

    def __init__(self, *, min_chars: int = 4) -> None:
        self.min_chars = min_chars

    def propose(self, ctx: RecallContext) -> list[Candidate]:
        gaz = ctx.gazetteer
        if not gaz or len(gaz) == 0:
            return []
        doc = ctx.document
        # Case-folding is length-preserving for the character classes we see
        # after normalisation, so offsets carry over. Guarded by an assertion
        # because a mismatch here would silently corrupt every span.
        lowered = doc.text.lower()
        if len(lowered) != len(doc.text):  # pragma: no cover - defensive
            return []

        out: list[Candidate] = []
        for start, end, entry in gaz.scan(lowered):
            if end - start < self.min_chars:
                continue
            cand = self._candidate(
                doc,
                start,
                end,
                features={
                    "kb_hit": 1.0,
                    "kb_depth": float(entry.depth),
                },
                notes={
                    "kb": entry.kb,
                    "entry_id": entry.entry_id,
                    "kb_label": entry.label,
                    "path": "gazetteer",
                },
            )
            if cand:
                out.append(cand)
        return out
