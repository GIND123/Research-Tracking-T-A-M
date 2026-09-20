"""Batched verification.

Per-candidate model calls are the obvious way to build a verifier and the wrong
one.  A single patent yields on the order of 10^2 candidates; at one call each
the request overhead and the re-sent system prompt dominate the bill, and the
useful payload -- a term, a type and one sentence -- is a few dozen tokens.

Three reductions are applied before anything is sent, in this order:

1. **Deduplicate.** Verification is a function of (surface, type, sentence).  A
   term repeated forty times in a specification is one question, and its answer
   is reused across all forty occurrences.
2. **Skip the decided.** Only the embedding verifier's uncertainty band is
   escalated; confident accepts and rejects never reach a model.
3. **Batch.** The survivors are packed into groups sharing one system prompt,
   which is itself marked for caching, so the marginal cost of an item is close
   to its own tokens.

Together these took our development corpus from ~120 verifier calls per document
to between one and three.  The trade is that one malformed response can affect a
whole group, so the parser is strict about ids and any item missing from the
response is treated as an abstention rather than being inferred.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..agents.prompts import (
    PROMPT_VERSIONS,
    VERIFIER_BATCH_ITEM,
    VERIFIER_BATCH_SYSTEM,
    VERIFIER_BATCH_USER,
)
from ..guard.injection import wrap_untrusted
from ..guard.structured import StructuredCaller
from ..schema import Document, TechType, Verdict
from .verifier import TECHNOLOGY_TYPES, VerifierDecision

#: Items per request. Chosen so a group's user message stays near 1.5k tokens,
#: which keeps the response well inside a small max_tokens and keeps a single
#: malformed response from costing much.
DEFAULT_BATCH_SIZE = 20

#: Evidence longer than this is truncated around the mention; verification only
#: ever needs the local clause, and long claim sentences are mostly boilerplate.
MAX_EVIDENCE_CHARS = 320


class BatchVerifierItem(BaseModel):
    id: int
    present: bool = False
    is_tech: str = "uncertain"
    type_ok: str = "uncertain"
    reason: str = ""


class BatchVerifierOutput(BaseModel):
    items: list[BatchVerifierItem] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class VerificationQuestion:
    """The deduplication key: what actually determines the answer."""

    surface: str
    type: TechType
    evidence: str

    @property
    def key(self) -> str:
        blob = f"{self.surface}\x00{self.type.value}\x00{self.evidence}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class BatchStats:
    questions: int = 0
    unique_questions: int = 0
    calls: int = 0
    items_sent: int = 0
    missing_ids: int = 0
    schema_failures: int = 0
    reused: int = 0
    per_model: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        dedup = 1.0 - (self.unique_questions / self.questions) if self.questions else 0.0
        packing = self.items_sent / self.calls if self.calls else 0.0
        return {
            "questions": self.questions,
            "unique_questions": self.unique_questions,
            "dedup_rate": round(dedup, 4),
            "calls": self.calls,
            "items_per_call": round(packing, 2),
            "missing_ids": self.missing_ids,
            "schema_failures": self.schema_failures,
        }


class BatchedLLMVerifier:
    name = "llm_batch"
    version = PROMPT_VERSIONS["verifier_batch"]

    def __init__(
        self,
        backend,
        *,
        model: str = "claude-haiku-4-5",
        batch_size: int = DEFAULT_BATCH_SIZE,
        budget: Any = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.batch_size = max(1, batch_size)
        self.budget = budget
        self.caller = StructuredCaller(backend, max_repairs=1) if backend else None
        self.stats = BatchStats()
        #: Answers persist for the life of the verifier, so a term that recurs
        #: across documents in one run is asked about once.
        self._answers: dict[str, VerifierDecision] = {}

    @property
    def available(self) -> bool:
        return bool(self.backend and self.backend.available and self.caller)

    def verify(
        self, mentions: Sequence[Any], doc: Document
    ) -> dict[tuple[int, int], VerifierDecision]:
        """Return a decision per mention span, asking about each question once."""
        if not mentions:
            return {}
        if not self.available:
            return {
                _span_key(m): VerifierDecision(Verdict.ABSTAIN, "no LLM backend", source=self.name)
                for m in mentions
            }

        questions: dict[str, VerificationQuestion] = {}
        mention_to_question: dict[tuple[int, int], str] = {}

        for mention in mentions:
            question = VerificationQuestion(
                surface=mention.span.surface,
                type=mention.type,
                evidence=_trim_evidence(mention),
            )
            questions.setdefault(question.key, question)
            mention_to_question[_span_key(mention)] = question.key

        self.stats.questions += len(mentions)
        pending = [q for key, q in questions.items() if key not in self._answers]
        self.stats.reused += len(questions) - len(pending)
        self.stats.unique_questions += len(questions)

        for group in _chunks(pending, self.batch_size):
            if self.budget is not None and not self.budget.can_spend_llm_call():
                break
            self._ask(group)

        out: dict[tuple[int, int], VerifierDecision] = {}
        for span_key, question_key in mention_to_question.items():
            decision = self._answers.get(question_key)
            if decision is None:
                decision = VerifierDecision(
                    Verdict.ABSTAIN, "not verified (budget or batch failure)", source=self.name
                )
            out[span_key] = decision
        return out

    # -- request ------------------------------------------------------------

    def _ask(self, group: list[VerificationQuestion]) -> None:
        index = {i + 1: q for i, q in enumerate(group)}
        body = "\n\n".join(
            VERIFIER_BATCH_ITEM.format(
                id=i, surface=q.surface, type=q.type.value, evidence=q.evidence
            )
            for i, q in index.items()
        )
        user = VERIFIER_BATCH_USER.format(items=wrap_untrusted(body, label="items"))

        result = self.caller.call(
            BatchVerifierOutput,
            system=VERIFIER_BATCH_SYSTEM,
            user=user,
            model=self.model,
            # ~25 output tokens per item plus JSON scaffolding.
            max_tokens=min(4096, 160 + 40 * len(group)),
        )
        self.stats.calls += 1
        self.stats.items_sent += len(group)
        if self.budget is not None:
            for response in result.responses:
                self.budget.record_llm_call(response.usage, self.model)

        if not result.ok or result.value is None:
            self.stats.schema_failures += 1
            if self.budget is not None:
                self.budget.record_schema_failure()
            return
        if self.budget is not None:
            self.budget.record_success()

        seen: set[int] = set()
        for item in result.value.items:  # type: ignore[union-attr]
            question = index.get(item.id)
            if question is None or item.id in seen:
                continue
            seen.add(item.id)
            self._answers[question.key] = _decide(question, item, self.name)

        missing = set(index) - seen
        self.stats.missing_ids += len(missing)
        for item_id in missing:
            question = index[item_id]
            self._answers[question.key] = VerifierDecision(
                Verdict.ABSTAIN, "omitted from batch response", source=self.name
            )


def _decide(question: VerificationQuestion, item: BatchVerifierItem, source: str) -> VerifierDecision:
    claims_tech = question.type in TECHNOLOGY_TYPES
    if item.is_tech == "no":
        verdict = Verdict.REJECT if claims_tech else Verdict.PASS
    elif item.is_tech == "yes":
        verdict = Verdict.PASS if claims_tech else Verdict.ABSTAIN
    else:
        verdict = Verdict.ABSTAIN

    if verdict is Verdict.PASS and item.type_ok == "no":
        verdict = Verdict.ABSTAIN

    return VerifierDecision(
        verdict=verdict,
        reason=(item.reason or "")[:120],
        score=1.0 if verdict is Verdict.PASS else 0.0,
        source=source,
        evidence_seen=True,
        details={"present": item.present, "is_tech": item.is_tech, "type_ok": item.type_ok},
    )


def _trim_evidence(mention: Any) -> str:
    """Keep the clause around the mention when the sentence is oversized."""
    evidence = mention.evidence.span
    text = evidence.surface
    if len(text) <= MAX_EVIDENCE_CHARS:
        return text
    rel_start = mention.span.start - evidence.start
    rel_end = mention.span.end - evidence.start
    pad = (MAX_EVIDENCE_CHARS - (rel_end - rel_start)) // 2
    lo = max(0, rel_start - pad)
    hi = min(len(text), rel_end + pad)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{text[lo:hi]}{suffix}"


def _span_key(mention: Any) -> tuple[int, int]:
    return (mention.span.start, mention.span.end)


def _chunks(items: list[VerificationQuestion], size: int) -> list[list[VerificationQuestion]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
