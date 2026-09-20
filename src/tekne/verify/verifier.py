"""Adversarial verification of extracted mentions.

Three verifiers of increasing cost, run as a cascade:

``StructuralVerifier``
    Re-derives the grounding and evidence relations from the document rather
    than trusting the record.  Cannot be fooled by anything upstream; catches
    plumbing bugs and, in the guard test suite, catches a backend that fabricates
    offsets to go with its fabricated spans.

``EmbeddingVerifier``
    Distributional second opinion (see :mod:`tekne.verify.embed`).  Decides the
    easy cases in both directions and leaves an uncertainty band.

``LLMVerifier``
    Blind judge over the evidence sentence alone.  It is shown the sentence, the
    term and the proposed type, and nothing about how any of them were produced:
    a verifier given the proposer's reasoning ratifies it.

Only candidates that fall in the uncertainty band reach the LLM, which is what
keeps per-document cost bounded while still putting a model on the genuinely
ambiguous cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..agents.prompts import PROMPT_VERSIONS, VERIFIER_SYSTEM, VERIFIER_USER
from ..guard.injection import wrap_untrusted
from ..guard.structured import StructuredCaller
from ..schema import Document, TechType, Verdict

TECHNOLOGY_TYPES = frozenset({TechType.ARTIFACT, TechType.METHOD, TechType.MATERIAL, TechType.TOOL})


@dataclass
class VerifierDecision:
    verdict: Verdict
    reason: str = ""
    score: float = 0.0
    source: str = ""
    evidence_seen: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class VerifierOutput(BaseModel):
    present: bool = Field(default=False)
    is_tech: str = Field(default="uncertain")
    type_ok: str = Field(default="uncertain")
    reason: str = Field(default="")


class StructuralVerifier:
    name = "structural"
    version = "1"

    def check(self, mention, doc: Document) -> VerifierDecision:
        span = mention.span
        if span.end > len(doc.text):
            return VerifierDecision(Verdict.REJECT, "span outside document", source=self.name)
        if doc.text[span.start : span.end] != span.surface:
            return VerifierDecision(Verdict.REJECT, "surface/offset mismatch", source=self.name)
        ev = mention.evidence.span
        if doc.text[ev.start : ev.end] != ev.surface:
            return VerifierDecision(Verdict.REJECT, "evidence/offset mismatch", source=self.name)
        if not ev.contains(span):
            return VerifierDecision(Verdict.REJECT, "evidence excludes mention", source=self.name)
        return VerifierDecision(Verdict.PASS, "structurally sound", 1.0, self.name, True)


class EmbeddingVerifier:
    name = "embedding"
    version = "1"

    def __init__(self, scorer, *, low: float = 0.35, high: float = 0.65) -> None:
        self.scorer = scorer
        self.low = low
        self.high = high

    @property
    def available(self) -> bool:
        return bool(self.scorer and self.scorer.available)

    def check(self, mention, doc: Document) -> VerifierDecision:
        if not self.available:
            return VerifierDecision(Verdict.ABSTAIN, "embedding scorer unavailable", source=self.name)
        score = self.scorer.score(mention.span.surface)
        if not score.available:
            return VerifierDecision(Verdict.ABSTAIN, "no embedding", source=self.name)

        # Map the raw cosine margin onto [0, 1]; the band edges were set on a
        # development sample and are exposed in VerifyConfig.
        confidence = _sigmoid(score.tech_margin * 12.0)
        claims_tech = mention.type in TECHNOLOGY_TYPES
        details = {
            "tech_margin": round(score.tech_margin, 4),
            "nearest": score.best_type.value if score.best_type else None,
        }

        if claims_tech:
            if confidence >= self.high:
                return VerifierDecision(
                    Verdict.PASS, "distributionally technological", confidence, self.name, True, details
                )
            # Rejection requires the mention to look like *furniture*, not merely
            # like a different technology-adjacent class. Nearest-centroid on
            # short phrases confuses methods with the tasks they solve often
            # enough that treating "closer to TASK than to ARTIFACT" as a
            # rejection removed real technologies ("mixture-of-experts" was the
            # case that caught this). A type disagreement is an abstention and a
            # job for the typer; only NOT_TECH proximity is evidence against
            # technology-hood at all.
            if confidence <= self.low and score.best_type is TechType.NOT_TECH:
                return VerifierDecision(
                    Verdict.REJECT,
                    "nearest prototype class is not_tech",
                    confidence,
                    self.name,
                    True,
                    details,
                )
            if confidence <= self.low:
                return VerifierDecision(
                    Verdict.ABSTAIN,
                    f"type disputed: nearest prototype is {details['nearest']}",
                    confidence,
                    self.name,
                    True,
                    details,
                )
        else:
            # The mention claims to be a field/task/metric. Confirm only when the
            # embedding agrees it is not an artefact.
            if confidence <= self.high:
                return VerifierDecision(
                    Verdict.PASS, "consistent with non-artefact type", 1.0 - confidence, self.name, True, details
                )
        return VerifierDecision(
            Verdict.ABSTAIN, "inside uncertainty band", confidence, self.name, True, details
        )


class LLMVerifier:
    name = "llm"
    version = PROMPT_VERSIONS["verifier"]

    def __init__(self, backend, *, model: str = "claude-haiku-4-5", budget=None) -> None:
        self.backend = backend
        self.model = model
        self.budget = budget
        self.caller = StructuredCaller(backend, max_repairs=1) if backend else None
        self.stats = {"calls": 0, "schema_failures": 0, "absent_spans": 0}

    @property
    def available(self) -> bool:
        return bool(self.backend and self.backend.available and self.caller)

    def check(self, mention, doc: Document) -> VerifierDecision:
        if not self.available:
            return VerifierDecision(Verdict.ABSTAIN, "no LLM backend", source=self.name)
        if self.budget is not None and not self.budget.can_spend_llm_call():
            return VerifierDecision(Verdict.ABSTAIN, "budget exhausted", source=self.name)

        evidence = mention.evidence.span.surface
        user = VERIFIER_USER.format(
            evidence=wrap_untrusted(evidence, label="sentence"),
            surface=mention.span.surface,
            type=mention.type.value,
        )
        result = self.caller.call(
            VerifierOutput,
            system=VERIFIER_SYSTEM,
            user=user,
            model=self.model,
            max_tokens=512,
        )
        self.stats["calls"] += 1
        if self.budget is not None:
            for response in result.responses:
                self.budget.record_llm_call(response.usage, self.model)

        if not result.ok or result.value is None:
            self.stats["schema_failures"] += 1
            return VerifierDecision(
                Verdict.ABSTAIN, f"verifier call failed: {result.error}", source=self.name
            )

        out: VerifierOutput = result.value  # type: ignore[assignment]

        # The model claiming the term is absent from a sentence we already proved
        # contains it is itself a signal -- record it, but trust the document.
        if not out.present:
            self.stats["absent_spans"] += 1

        claims_tech = mention.type in TECHNOLOGY_TYPES
        if out.is_tech == "no":
            verdict = Verdict.REJECT if claims_tech else Verdict.PASS
        elif out.is_tech == "yes":
            verdict = Verdict.PASS if claims_tech else Verdict.ABSTAIN
        else:
            verdict = Verdict.ABSTAIN

        if verdict is Verdict.PASS and out.type_ok == "no":
            verdict = Verdict.ABSTAIN

        return VerifierDecision(
            verdict=verdict,
            reason=(out.reason or "")[:200],
            score=1.0 if verdict is Verdict.PASS else 0.0,
            source=self.name,
            evidence_seen=True,
            details={"present": out.present, "is_tech": out.is_tech, "type_ok": out.type_ok},
        )


class VerifierCascade:
    """Run verifiers cheapest-first and stop as soon as one is decisive."""

    def __init__(
        self,
        *,
        structural: StructuralVerifier | None = None,
        embedding: EmbeddingVerifier | None = None,
        llm: LLMVerifier | None = None,
    ) -> None:
        self.structural = structural or StructuralVerifier()
        self.embedding = embedding
        self.llm = llm
        self.stats = {"structural": 0, "embedding": 0, "llm": 0, "escalated": 0}

    def check(self, mention, doc: Document) -> VerifierDecision:
        decision = self.structural.check(mention, doc)
        self.stats["structural"] += 1
        if decision.verdict is Verdict.REJECT:
            return decision

        if self.embedding is not None and self.embedding.available:
            decision = self.embedding.check(mention, doc)
            self.stats["embedding"] += 1
            if decision.verdict is not Verdict.ABSTAIN:
                return decision

        if self.llm is not None and self.llm.available:
            self.stats["escalated"] += 1
            llm_decision = self.llm.check(mention, doc)
            self.stats["llm"] += 1
            if llm_decision.verdict is not Verdict.ABSTAIN:
                return llm_decision
            return llm_decision

        return decision


def _sigmoid(x: float) -> float:
    import math

    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
