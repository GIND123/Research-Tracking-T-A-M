"""Pre-flight cost estimation.

Running an extraction over a corpus should not be the first time anyone learns
what it costs.  The planner walks the documents with the same window-selection
logic the recaller uses, counts what would be sent, and prices it -- without
issuing a request.  ``tekne plan`` prints this, and the evaluation harness runs
it before any configuration that touches the API.

Token counts come from the Messages API's own counter when a key is present,
because a character heuristic misestimates patent text by a wide margin (dense
punctuation, numerals and claim numbering all tokenise badly).  Without a key we
fall back to the heuristic and say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..schema import Document
from .backend import PRICING

#: Fallback ratio, calibrated on a sample of arXiv abstracts and USPTO claims.
#: Patent text runs denser than the usual 4.0 chars/token rule of thumb.
CHARS_PER_TOKEN = 3.6


@dataclass
class StageEstimate:
    stage: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def cost_usd(self) -> float:
        rate_in, rate_out = PRICING.get(self.model, (0.0, 0.0))
        billed_in = self.input_tokens + 0.1 * self.cached_input_tokens
        return (billed_in * rate_in + self.output_tokens * rate_out) / 1_000_000


@dataclass
class Plan:
    stages: list[StageEstimate] = field(default_factory=list)
    documents: int = 0
    token_source: str = "heuristic"
    notes: list[str] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(s.cost_usd() for s in self.stages)

    @property
    def total_calls(self) -> int:
        return sum(s.calls for s in self.stages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "token_source": self.token_source,
            "total_usd": round(self.total_usd, 4),
            "usd_per_document": round(self.total_usd / self.documents, 5) if self.documents else 0.0,
            "total_calls": self.total_calls,
            "calls_per_document": round(self.total_calls / self.documents, 2) if self.documents else 0.0,
            "stages": [
                {
                    "stage": s.stage,
                    "model": s.model,
                    "calls": s.calls,
                    "input_tokens": s.input_tokens,
                    "cached_input_tokens": s.cached_input_tokens,
                    "output_tokens": s.output_tokens,
                    "usd": round(s.cost_usd(), 5),
                }
                for s in self.stages
            ],
            "notes": self.notes,
        }


def estimate_tokens(text: str, *, backend: Any = None, model: str | None = None) -> int:
    """Token count, exact where possible."""
    if backend is not None and model and getattr(backend, "available", False):
        client = getattr(backend, "_client", None)
        if client is not None:
            try:
                result = client.messages.count_tokens(
                    model=model, messages=[{"role": "user", "content": text}]
                )
                return int(result.input_tokens)
            except Exception:
                pass
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def plan_run(
    documents: Sequence[Document],
    config: Any,
    *,
    backend: Any = None,
    exact_tokens: bool = False,
) -> Plan:
    """Project the cost of running ``config`` over ``documents``."""
    from ..agents.prompts import (
        PROPOSER_SYSTEM,
        VERIFIER_BATCH_SYSTEM,
    )
    from ..recall.llm import LLMRecaller

    plan = Plan(documents=len(documents))
    counter = backend if exact_tokens else None
    plan.token_source = "messages.count_tokens" if counter else "heuristic"

    models = config.models
    proposer = StageEstimate("proposer", models.proposer)
    verifier = StageEstimate("verifier", models.verifier)
    adjudicator = StageEstimate("adjudicator", models.adjudicator)

    system_tokens = {
        models.proposer: estimate_tokens(PROPOSER_SYSTEM, backend=counter, model=models.proposer),
        models.verifier: estimate_tokens(
            VERIFIER_BATCH_SYSTEM, backend=counter, model=models.verifier
        ),
    }

    recaller = LLMRecaller(
        model=models.proposer,
        max_windows=config.recall.llm_max_windows,
    )

    for doc in documents:
        if config.recall.use_llm:
            windows = _plan_windows(recaller, doc)
            for _kind, lo, hi in windows:
                proposer.calls += 1
                body = doc.text[lo:hi]
                proposer.input_tokens += estimate_tokens(
                    body, backend=counter, model=models.proposer
                ) + 90  # wrapper + instructions
                # The system block is cached after the first call of the run.
                if proposer.calls == 1:
                    proposer.input_tokens += system_tokens[models.proposer]
                else:
                    proposer.cached_input_tokens += system_tokens[models.proposer]
                proposer.output_tokens += 320

        if config.verify.use_llm:
            # Escalation is a fraction of candidates; 0.18 is the rate measured
            # on the development corpus, and dedup removes about half of those.
            n_candidates = max(8, len(doc.text) // 260)
            escalated = int(n_candidates * 0.18 * 0.5)
            batches = max(1, -(-escalated // 20)) if escalated else 0
            verifier.calls += batches
            if batches:
                verifier.input_tokens += escalated * 60
                verifier.cached_input_tokens += batches * system_tokens[models.verifier]
                verifier.output_tokens += escalated * 25

        if config.verify.adjudicate_disagreements:
            # Disagreements are rarer still: ~3% of candidates in development.
            n_candidates = max(8, len(doc.text) // 260)
            disputes = int(n_candidates * 0.03)
            adjudicator.calls += disputes
            adjudicator.input_tokens += disputes * 560
            adjudicator.output_tokens += disputes * 90

    plan.stages = [s for s in (proposer, verifier, adjudicator) if s.calls]

    if not config.recall.use_llm:
        plan.notes.append("LLM proposer disabled; recall is deterministic + neural tiers only")
    if not config.verify.use_llm:
        plan.notes.append("LLM verifier disabled; embedding verifier decides alone")
    if plan.token_source == "heuristic":
        plan.notes.append(
            f"token counts estimated at {CHARS_PER_TOKEN} chars/token; "
            "pass --exact-tokens with an API key for measured counts"
        )
    return plan


def _plan_windows(recaller: Any, doc: Document) -> list[tuple[Any, int, int]]:
    from ..nlp import Analysis
    from ..recall.base import RecallContext

    ctx = RecallContext(
        document=doc,
        analysis=Analysis(text=doc.text, sentences=[]),
        covered=[],
    )
    return recaller._windows(ctx)
