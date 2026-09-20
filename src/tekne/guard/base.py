"""Guard interface and the stack that runs them.

A guard is a unary predicate over a candidate or a finished mention that returns
PASS, REJECT or ABSTAIN together with a reason.  Guards never repair, never
rewrite and never silently drop: a REJECT is recorded in
``ExtractionResult.rejected`` with its reason, an ABSTAIN goes to the review
queue.  Everything the pipeline decided not to emit stays inspectable, which is
what makes the ablations in the evaluation cheap to run and the output auditable.

Ordering matters.  Cheap structural guards run before anything that costs a
model call, so a fabricated span is thrown away before it can be argued about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..schema import GuardVerdict, Verdict


@dataclass
class GuardContext:
    document: Any
    analysis: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class Guard(ABC):
    name: str = "guard"
    #: "candidate" guards run before typing; "mention" guards run on the final record.
    stage: str = "candidate"
    #: A blocking guard's REJECT removes the item. A non-blocking guard only
    #: annotates, letting the confidence model weigh the signal instead.
    blocking: bool = True

    @abstractmethod
    def check(self, item: Any, ctx: GuardContext) -> GuardVerdict:
        ...

    def _pass(self, reason: str = "", score: float | None = None) -> GuardVerdict:
        return GuardVerdict(guard=self.name, verdict=Verdict.PASS, reason=reason, score=score)

    def _reject(self, reason: str, score: float | None = None) -> GuardVerdict:
        return GuardVerdict(guard=self.name, verdict=Verdict.REJECT, reason=reason, score=score)

    def _abstain(self, reason: str, score: float | None = None) -> GuardVerdict:
        return GuardVerdict(guard=self.name, verdict=Verdict.ABSTAIN, reason=reason, score=score)


@dataclass
class GuardOutcome:
    verdict: Verdict
    verdicts: list[GuardVerdict] = field(default_factory=list)
    #: Name of the guard that decided the outcome, if not PASS.
    decided_by: str | None = None

    @property
    def reason(self) -> str:
        for v in self.verdicts:
            if v.verdict is not Verdict.PASS:
                return f"{v.guard}: {v.reason}"
        return ""


class GuardStack:
    def __init__(self, guards: list[Guard]) -> None:
        self.guards = guards

    def for_stage(self, stage: str) -> GuardStack:
        return GuardStack([g for g in self.guards if g.stage == stage])

    def run(self, item: Any, ctx: GuardContext) -> GuardOutcome:
        verdicts: list[GuardVerdict] = []
        outcome = Verdict.PASS
        decided_by: str | None = None
        for guard in self.guards:
            verdict = guard.check(item, ctx)
            verdicts.append(verdict)
            if verdict.verdict is Verdict.PASS:
                continue
            if not guard.blocking:
                continue
            if verdict.verdict is Verdict.REJECT:
                # Short-circuit: no point paying for later guards.
                return GuardOutcome(Verdict.REJECT, verdicts, guard.name)
            if outcome is Verdict.PASS:
                outcome = Verdict.ABSTAIN
                decided_by = guard.name
        return GuardOutcome(outcome, verdicts, decided_by)
