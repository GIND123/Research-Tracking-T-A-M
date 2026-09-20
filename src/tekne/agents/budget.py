"""Per-document resource accounting and the circuit breaker.

Corpus-scale extraction fails in a characteristic way: one pathological document
-- a 400-page continuation-in-part, or a document that makes the model loop on a
schema it will not satisfy -- absorbs an unbounded share of the run.  The budget
makes that a bounded, *recorded* event: the document is finished with whatever
tiers it could afford and the shortfall appears in its trace, rather than the
run stalling or the document being silently dropped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..llm.backend import Usage


@dataclass
class BudgetSnapshot:
    llm_calls: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    exhausted: str | None = None
    tripped: bool = False


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(
        self,
        *,
        max_llm_calls: int = 12,
        max_usd: float = 0.25,
        max_seconds: float = 180.0,
        circuit_breaker_failures: int = 5,
    ) -> None:
        self.max_llm_calls = max_llm_calls
        self.max_usd = max_usd
        self.max_seconds = max_seconds
        self.circuit_breaker_failures = circuit_breaker_failures

        self.llm_calls = 0
        self.usage = Usage()
        self.usd = 0.0
        self.consecutive_failures = 0
        self.tripped = False
        self.exhausted: str | None = None
        self._start = time.monotonic()
        self.per_model: dict[str, dict[str, float]] = {}

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        self.llm_calls = 0
        self.usage = Usage()
        self.usd = 0.0
        self.consecutive_failures = 0
        self.tripped = False
        self.exhausted = None
        self._start = time.monotonic()
        self.per_model = {}

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    # -- admission ----------------------------------------------------------

    def can_spend_llm_call(self) -> bool:
        if self.tripped:
            self.exhausted = self.exhausted or "circuit_breaker"
            return False
        if self.llm_calls >= self.max_llm_calls:
            self.exhausted = self.exhausted or "llm_calls"
            return False
        if self.usd >= self.max_usd:
            self.exhausted = self.exhausted or "usd"
            return False
        if self.elapsed >= self.max_seconds:
            self.exhausted = self.exhausted or "seconds"
            return False
        return True

    # -- accounting ---------------------------------------------------------

    def record_llm_call(self, usage: Usage, model: str) -> None:
        self.llm_calls += 1
        self.usage = self.usage + usage
        cost = usage.cost_usd(model)
        self.usd += cost
        bucket = self.per_model.setdefault(
            model, {"calls": 0.0, "usd": 0.0, "input": 0.0, "output": 0.0}
        )
        bucket["calls"] += 1
        bucket["usd"] += cost
        bucket["input"] += usage.input_tokens
        bucket["output"] += usage.output_tokens

    def record_schema_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.circuit_breaker_failures:
            self.tripped = True

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            llm_calls=self.llm_calls,
            usd=round(self.usd, 6),
            seconds=round(self.elapsed, 3),
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            cache_read_tokens=self.usage.cache_read_tokens,
            exhausted=self.exhausted,
            tripped=self.tripped,
        )


@dataclass
class RunLedger:
    """Aggregate of per-document budgets across a run."""

    documents: int = 0
    llm_calls: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    exhausted: dict[str, int] = field(default_factory=dict)
    per_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, budget: Budget) -> None:
        snap = budget.snapshot()
        self.documents += 1
        self.llm_calls += snap.llm_calls
        self.usd += snap.usd
        self.seconds += snap.seconds
        if snap.exhausted:
            self.exhausted[snap.exhausted] = self.exhausted.get(snap.exhausted, 0) + 1
        for model, bucket in budget.per_model.items():
            target = self.per_model.setdefault(
                model, {"calls": 0.0, "usd": 0.0, "input": 0.0, "output": 0.0}
            )
            for key, value in bucket.items():
                target[key] += value

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "llm_calls": self.llm_calls,
            "usd_total": round(self.usd, 6),
            "usd_per_doc": round(self.usd / self.documents, 6) if self.documents else 0.0,
            "seconds_total": round(self.seconds, 2),
            "seconds_per_doc": round(self.seconds / self.documents, 3) if self.documents else 0.0,
            "budget_exhausted": self.exhausted,
            "per_model": {
                m: {k: round(v, 6) for k, v in bucket.items()}
                for m, bucket in self.per_model.items()
            },
        }
