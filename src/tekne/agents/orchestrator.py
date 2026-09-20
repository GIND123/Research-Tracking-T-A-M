"""The pipeline: a blackboard, a fixed stage order, and a trace of what happened.

The "agentic" structure here is not a model deciding what to do next.  It is a
set of specialised components writing to a shared typed state under contracts,
with an escalation policy that decides *which* components run on which items and
a budget that can stop any of them.  For an extraction task with a known
decomposition, a learned controller would add cost and non-determinism and buy
nothing: the interesting autonomy is in the per-item routing, not in the plan.

What the orchestrator owns:

* **Stage order and the blackboard.** Each stage declares what it reads and
  writes; the blackboard is checked against those declarations, so a stage
  cannot quietly depend on something that has not been produced.
* **Escalation.** Cheap recallers run on everything; the model-backed recaller
  runs on the regions they left thin; the model-backed verifier runs on the
  band the embedding verifier could not settle; the adjudicator runs on the
  residue where proposer and verifier disagree.
* **The trace.** Every stage appends a record with its inputs, outputs, timings
  and any degradation. A run that skipped a tier says so rather than silently
  producing fewer mentions.

Corpus extraction is three passes: analyse and collect term statistics, extract
per document, then canonicalise across documents and re-apply the guards that
need corpus-level knowledge (the temporal one).
"""

from __future__ import annotations

import bisect
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..calib.selective import ConfidenceInputs, ConfidenceModel
from ..classify.roler import RoleClassifier
from ..classify.typer import TypeClassifier
from ..config import Config
from ..guard.base import GuardContext, GuardStack
from ..guard.checks import (
    ConsensusGuard,
    EvidenceGuard,
    ProvenanceGuard,
    TemporalGuard,
    VerifierGuard,
)
from ..guard.grounding import SpanGroundingGuard
from ..guard.injection import InjectionGuard, document_risk
from ..guard.negative import NegativeLexiconGuard
from ..kb import Gazetteer
from ..link.canonical import Canonicalizer
from ..llm.backend import LLMBackend, NullBackend
from ..llm.cache import CallCache
from ..nlp import analyse, pipeline_available
from ..recall.abbrev import AbbreviationRecaller
from ..recall.base import RecallContext, merge_candidates
from ..recall.cvalue import TermStatRecaller, build_corpus_stats
from ..recall.gazetteer import GazetteerRecaller
from ..recall.llm import LLMRecaller
from ..recall.patterns import PatternRecaller
from ..recall.select import DISTINCTIVE as DISTINCTIVE_PROPOSERS
from ..recall.select import candidate_score
from ..schema import (
    Candidate,
    Document,
    Evidence,
    ExtractionResult,
    KBLink,
    Provenance,
    Role,
    SectionKind,
    Span,
    TechMention,
    TechType,
    Verdict,
    content_hash,
)
from ..verify.adjudicator import Adjudicator
from ..verify.batch import BatchedLLMVerifier
from ..verify.embed import EmbeddingScorer
from ..verify.verifier import EmbeddingVerifier, StructuralVerifier
from .budget import Budget, RunLedger

#: Zones whose mentions are weighted up in the confidence model. A technology
#: named in a title or an independent claim is being asserted, not surveyed.
SECTION_WEIGHT: dict[SectionKind, float] = {
    SectionKind.TITLE: 1.0,
    SectionKind.ABSTRACT: 0.85,
    SectionKind.CLAIM_INDEP: 0.9,
    SectionKind.PATENT_SUMMARY: 0.7,
    SectionKind.METHOD: 0.7,
    SectionKind.CLAIM_DEP: 0.55,
    SectionKind.INTRODUCTION: 0.5,
    SectionKind.EXPERIMENT: 0.45,
    SectionKind.RESULTS: 0.4,
    SectionKind.PATENT_DETAIL: 0.35,
    SectionKind.RELATED_WORK: 0.25,
    SectionKind.PATENT_BACKGROUND: 0.2,
    SectionKind.CONCLUSION: 0.5,
    SectionKind.OTHER: 0.3,
}


class _IntervalIndex:
    """Overlap queries over accepted spans, keeping insertion-order semantics.

    The decoder needs the *first accepted* overlapping span -- first meaning
    highest confidence, since acceptance proceeds in descending confidence --
    not merely some overlapping span, so the query returns the minimum insertion
    index among the matches rather than the first one it happens to find.
    """

    __slots__ = ("_starts", "_entries", "_max_width")

    def __init__(self) -> None:
        self._starts: list[int] = []
        self._entries: list[tuple[int, int, int]] = []  # (start, end, insertion index)
        self._max_width = 0

    def add(self, span: Span, order: int) -> None:
        position = bisect.bisect_left(self._starts, span.start)
        self._starts.insert(position, span.start)
        self._entries.insert(position, (span.start, span.end, order))
        self._max_width = max(self._max_width, span.end - span.start)

    def first_overlapping(self, span: Span, kept: list[TechMention]) -> TechMention | None:
        if not self._entries:
            return None
        # Any overlapping interval must start before `span.end` and, since no
        # accepted interval is wider than `_max_width`, no earlier than
        # `span.start - _max_width`.
        hi = bisect.bisect_left(self._starts, span.end)
        lo = bisect.bisect_left(self._starts, span.start - self._max_width)
        best: int | None = None
        for start, end, order in self._entries[lo:hi]:
            if start < span.end and span.start < end and (best is None or order < best):
                best = order
        return kept[best] if best is not None else None


@dataclass
class Blackboard:
    """Shared, declared state for one document."""

    document: Document
    analysis: Any = None
    candidates: list[Candidate] = field(default_factory=list)
    mentions: list[TechMention] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    abstained: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def note(self, stage: str, **payload: Any) -> None:
        self.trace.append({"stage": stage, **payload})


class Pipeline:
    def __init__(self, config: Config | None = None, *, backend: LLMBackend | None = None) -> None:
        self.config = config or Config()
        self.backend = backend or self._make_backend()
        self.gazetteer = self._load_gazetteer()
        self.embedding = (
            EmbeddingScorer(self.config.verify.embedding_model)
            if self.config.verify.use_embedding
            else None
        )
        self.typer = TypeClassifier(embedding_scorer=self.embedding)
        self.roler = RoleClassifier()
        self.confidence = ConfidenceModel()
        self.canonicalizer = Canonicalizer(embedding_scorer=self.embedding)
        self.ledger = RunLedger()
        self.corpus_stats: Any = None
        self._llm_stats: dict[str, Any] = {}

        low, high = self.config.verify.escalation_band
        self.structural_verifier = StructuralVerifier()
        self.embedding_verifier = (
            EmbeddingVerifier(self.embedding, low=low, high=high) if self.embedding else None
        )
        self.batch_verifier: BatchedLLMVerifier | None = None
        self.adjudicator: Adjudicator | None = None

        self.candidate_guards = self._build_candidate_guards()
        self.mention_guards = self._build_mention_guards()

    # -- construction -------------------------------------------------------

    def _make_backend(self) -> LLMBackend:
        cfg = self.config
        wants_llm = cfg.recall.use_llm or cfg.verify.use_llm or cfg.verify.adjudicate_disagreements
        if not wants_llm or not cfg.has_api_key():
            return NullBackend()
        from ..llm.anthropic_backend import AnthropicBackend

        cache = CallCache(cfg.cache_path) if cfg.cache_path else None
        return AnthropicBackend(cache=cache, effort=cfg.models.effort)

    def _load_gazetteer(self) -> Gazetteer:
        path = self.config.kb_path
        if path and Path(path).is_file():
            return Gazetteer.load(Path(path))
        return Gazetteer.empty()

    def _build_candidate_guards(self) -> GuardStack:
        g = self.config.guards
        guards = []
        if g.grounding:
            guards.append(SpanGroundingGuard())
        if g.negative_lexicon:
            guards.append(NegativeLexiconGuard(max_tokens=self.config.recall.max_chunk_tokens + 2))
        if g.injection:
            guards.append(InjectionGuard())
        if g.consensus:
            guards.append(ConsensusGuard(min_proposers=g.consensus_min_proposers))
        return GuardStack(guards)

    def _build_mention_guards(self) -> GuardStack:
        g = self.config.guards
        guards = []
        if g.evidence:
            guards.append(EvidenceGuard())
        if g.verifier:
            guards.append(VerifierGuard())
        if g.temporal:
            guards.append(TemporalGuard())
        if g.provenance:
            guards.append(ProvenanceGuard())
        return GuardStack(guards)

    def _ensure_llm_components(self, budget: Budget) -> None:
        if not self.backend.available:
            return
        if self.config.verify.use_llm and self.batch_verifier is None:
            self.batch_verifier = BatchedLLMVerifier(
                self.backend,
                model=self.config.models.verifier,
                batch_size=self.config.verify.batch_size,
                budget=budget,
            )
        elif self.batch_verifier is not None:
            self.batch_verifier.budget = budget

        if self.config.verify.adjudicate_disagreements and self.adjudicator is None:
            self.adjudicator = Adjudicator(
                self.backend, model=self.config.models.adjudicator, budget=budget
            )
        elif self.adjudicator is not None:
            self.adjudicator.budget = budget

    # -- corpus -------------------------------------------------------------

    def run(
        self,
        documents: Sequence[Document],
        *,
        analyses: dict[str, Any] | None = None,
    ) -> list[ExtractionResult]:
        """Extract over a corpus, including the passes that need all documents.

        ``analyses`` lets a caller supply parses it already has. The evaluation
        harness runs the same corpus through a dozen configurations, and parsing
        is both the single most expensive stage and completely independent of
        configuration, so re-parsing per condition is pure waste.
        """
        docs = list(documents)
        started = time.monotonic()

        analyses = dict(analyses) if analyses else {}
        for doc in docs:
            if doc.doc_id not in analyses:
                analyses[doc.doc_id] = analyse(doc)
        if self.config.recall.use_cvalue:
            self.corpus_stats = build_corpus_stats(docs)

        results = [self.extract(doc, analysis=analyses[doc.doc_id]) for doc in docs]

        all_mentions = [m for r in results for m in r.mentions]
        years = {doc.doc_id: doc.year() for doc in docs}
        self.canonicalizer.add_abbreviation_pairs(
            pair for r in results for pair in r.stats.get("abbreviation_pairs", [])
        )
        self.canonicalizer.fit(all_mentions, years)

        attestation = self.canonicalizer.attestation()
        for result in results:
            result.mentions = [
                self._apply_canonical(m, attestation) for m in result.mentions
            ]
            result.stats["canonical"] = self.canonicalizer.summary()

        elapsed = time.monotonic() - started
        for result in results:
            result.stats["run_seconds_total"] = round(elapsed, 2)
        return results

    def _apply_canonical(self, mention: TechMention, attestation: dict[str, int]) -> TechMention:
        mention = self.canonicalizer.assign(mention)
        canonical_id = mention.canonical_id
        first_seen = attestation.get(canonical_id) if canonical_id else None
        if first_seen is not None and mention.kb_link is not None and mention.kb_link.attested_from is None:
            mention = mention.model_copy(
                update={"kb_link": mention.kb_link.model_copy(update={"attested_from": first_seen})}
            )
        return mention

    # -- single document ----------------------------------------------------

    def extract(self, doc: Document, *, analysis: Any = None) -> ExtractionResult:
        budget = Budget(
            max_llm_calls=self.config.budget.max_llm_calls_per_doc,
            max_usd=self.config.budget.max_usd_per_doc,
            max_seconds=self.config.budget.max_seconds_per_doc,
            circuit_breaker_failures=self.config.budget.circuit_breaker_failures,
        )
        self._ensure_llm_components(budget)
        board = Blackboard(document=doc)

        self._stage_analyse(board, analysis)
        self._stage_recall_cheap(board, budget)
        self._stage_guard_candidates(board, phase="cheap")
        self._stage_recall_llm(board, budget)
        self._stage_guard_candidates(board, phase="llm")
        self._stage_prune(board)
        self._stage_build_mentions(board)
        self._stage_verify(board, budget)
        self._stage_guard_mentions(board)
        self._stage_score(board)
        self._stage_decode(board)

        self.ledger.add(budget)
        return self._finalise(board, budget)

    # -- stages -------------------------------------------------------------

    def _stage_analyse(self, board: Blackboard, analysis: Any) -> None:
        t0 = time.monotonic()
        board.analysis = analysis if analysis is not None else analyse(board.document)
        risk = document_risk(board.document.text)
        board.note(
            "analyse",
            seconds=round(time.monotonic() - t0, 3),
            sentences=len(board.analysis.sentences),
            chunks=len(board.analysis.chunks),
            parser=pipeline_available(),
            injection_signals=risk,
        )

    def _stage_recall_cheap(self, board: Blackboard, budget: Budget) -> None:
        t0 = time.monotonic()
        cfg = self.config.recall
        ctx = RecallContext(
            document=board.document,
            analysis=board.analysis,
            corpus_stats=self.corpus_stats,
            gazetteer=self.gazetteer,
            llm=self.backend,
            budget=budget,
        )
        groups: list[list[Candidate]] = []
        fired: dict[str, int] = {}

        recallers = []
        if cfg.use_patterns:
            recallers.append(
                PatternRecaller(
                    max_chunk_tokens=cfg.max_chunk_tokens,
                    require_tech_head=cfg.require_tech_head,
                )
            )
        if cfg.use_abbrev:
            recallers.append(AbbreviationRecaller())
        if cfg.use_cvalue:
            recallers.append(TermStatRecaller(threshold=cfg.cvalue_threshold))
        if cfg.use_gazetteer:
            recallers.append(GazetteerRecaller())

        for recaller in recallers:
            produced = recaller.propose(ctx)
            groups.append(produced)
            fired[recaller.name] = len(produced)

        board.candidates = merge_candidates(groups)
        board.extras["recall_ctx"] = ctx
        board.extras["abbreviation_pairs"] = ctx.options.get("abbreviation_pairs", [])
        board.note(
            "recall_cheap",
            seconds=round(time.monotonic() - t0, 3),
            per_recaller=fired,
            merged=len(board.candidates),
        )

    def _stage_recall_llm(self, board: Blackboard, budget: Budget) -> None:
        if not self.config.recall.use_llm or not self.backend.available:
            board.note("recall_llm", skipped=True, reason="disabled or no backend")
            return
        t0 = time.monotonic()
        ctx: RecallContext = board.extras["recall_ctx"]
        ctx.covered = [c.span for c in board.candidates]
        ctx.budget = budget
        recaller = LLMRecaller(
            model=self.config.models.proposer,
            max_windows=self.config.recall.llm_max_windows,
            max_input_chars=self.config.recall.llm_max_input_chars,
            allow_fuzzy_grounding=self.config.guards.allow_fuzzy_grounding,
        )
        produced = recaller.propose(ctx)
        board.candidates = merge_candidates([board.candidates, produced])
        self._llm_stats = dict(recaller.stats)
        board.note(
            "recall_llm",
            seconds=round(time.monotonic() - t0, 3),
            proposed=recaller.stats["proposed"],
            grounded=recaller.stats["grounded_exact"] + recaller.stats["grounded_relaxed"],
            ungrounded=recaller.stats["ungrounded"],
            windows=recaller.stats["windows"],
            chars_sent=recaller.stats["chars_sent"],
        )

    def _stage_guard_candidates(self, board: Blackboard, *, phase: str) -> None:
        t0 = time.monotonic()
        ctx = GuardContext(document=board.document, analysis=board.analysis)
        survivors: list[Candidate] = []
        counts: dict[str, int] = {}

        for cand in board.candidates:
            outcome = self.candidate_guards.run(cand, ctx)
            if outcome.verdict is Verdict.REJECT:
                counts[outcome.decided_by or "?"] = counts.get(outcome.decided_by or "?", 0) + 1
                if self.config.output.keep_rejected:
                    board.rejected.append(_record(board.document, cand, outcome.reason))
                continue
            if outcome.verdict is Verdict.ABSTAIN:
                # An abstaining candidate guard is a soft signal, not a drop:
                # consensus failure lowers confidence rather than removing the
                # candidate, so the calibration curve can price it.
                cand = cand.model_copy(
                    update={"features": {**cand.features, "guard_abstain": 1.0}}
                )
            survivors.append(cand)

        board.candidates = survivors
        board.note(
            f"guard_candidates:{phase}",
            seconds=round(time.monotonic() - t0, 3),
            kept=len(survivors),
            rejected_by=counts,
        )

    def _stage_prune(self, board: Blackboard) -> None:
        """Bound the work, without deciding boundaries.

        Overlap resolution used to happen here, and it was the wrong place: the
        only evidence available before typing is how many recallers fired and how
        long the span is, which picks "superficial string similarity" over
        "string similarity" about as often as not. Choosing between overlapping
        spans is now the decoder's job (``_stage_decode``), after each span has a
        type, a verifier verdict and a calibrated confidence. All this stage does
        is cap the candidate set when a document produces an unreasonable number,
        keeping the highest-support spans.
        """
        t0 = time.monotonic()
        cap = self.config.output.max_mentions_per_doc
        before = len(board.candidates)
        if before > cap:
            ranked = sorted(board.candidates, key=lambda c: -candidate_score(c))[:cap]
            board.candidates = sorted(ranked, key=lambda c: (c.span.start, c.span.end))
        board.note(
            "prune",
            seconds=round(time.monotonic() - t0, 3),
            proposed=before,
            kept=len(board.candidates),
            capped=before > cap,
        )

    def _stage_decode(self, board: Blackboard) -> None:
        """Pick a non-overlapping set of mentions, highest confidence first.

        A nested span survives only when it carries evidence its container does
        not -- an abbreviation pair or a knowledge-base hit -- which is what keeps
        "CNN" alive inside "CNN encoder" while discarding the bare "network"
        inside "neural network".
        """
        t0 = time.monotonic()
        kept: list[TechMention] = []
        dropped = 0
        nested = 0
        # Interval index over what has been kept, so the overlap test is a binary
        # search rather than a scan. The naive version is quadratic in the number
        # of surviving mentions, which is invisible at the default threshold and
        # dominates the run once abstention is disabled (the ablation emits every
        # candidate, ~2k per patent).
        index = _IntervalIndex()

        for mention in sorted(board.mentions, key=lambda m: (-m.confidence, m.span.start)):
            container = index.first_overlapping(mention.span, kept)
            if container is None:
                kept.append(mention)
                index.add(mention.span, len(kept) - 1)
                continue
            distinctive = set(mention.provenance.proposers) & DISTINCTIVE_PROPOSERS
            if (
                container.span.contains(mention.span)
                and distinctive
                and not distinctive <= set(container.provenance.proposers)
            ):
                kept.append(mention)
                index.add(mention.span, len(kept) - 1)
                nested += 1
                continue
            dropped += 1
            board.abstained.append(
                _mention_record(mention, f"overlapped by {container.span.surface!r}")
            )

        board.mentions = sorted(kept, key=lambda m: (m.span.start, m.span.end))
        board.note(
            "decode",
            seconds=round(time.monotonic() - t0, 3),
            kept=len(kept),
            dropped_overlap=dropped,
            kept_nested=nested,
        )

    def _stage_build_mentions(self, board: Blackboard) -> None:
        t0 = time.monotonic()
        doc = board.document
        # Warm the embedding cache in one batch. The typer and the embedding
        # verifier both ask for a surface's prototype similarities, and asking
        # one string at a time costs ~8ms against ~0.8ms batched -- a tenfold
        # difference that, at a couple of thousand candidates per patent,
        # dominated the whole run.
        self._warm_embeddings(board)
        mentions: list[TechMention] = []
        type_counts: dict[str, int] = {}

        for cand in board.candidates[: self.config.output.max_mentions_per_doc]:
            section = doc.section_at(cand.span.start)
            kind = section.kind if section else SectionKind.OTHER
            prediction = self.typer.predict(cand, doc, kind)
            role = self.roler.predict(doc.text, cand.span.start, cand.span.end, kind)
            evidence = self._evidence_for(board, cand)
            if evidence is None:
                continue

            kb_link = self._kb_link(cand)
            digest = content_hash(doc.doc_id, cand.span.start, cand.span.end, cand.span.surface)
            mention = TechMention(
                mention_id=f"{doc.doc_id}:{digest}",
                doc_id=doc.doc_id,
                span=cand.span,
                normalized=_normalized(cand.span.surface),
                type=prediction.type,
                granularity=prediction.granularity,
                role=role.role,
                section=kind,
                evidence=evidence,
                confidence=0.0,
                kb_link=kb_link,
                provenance=Provenance(
                    doc_id=doc.doc_id,
                    proposers=tuple(sorted(cand.proposers)),
                    stage_versions={
                        "typer": self.typer.version,
                        "roler": self.roler.version,
                        "confidence": self.confidence.version,
                    },
                    config_digest=self.config.digest(),
                    content_hash=digest,
                ),
            )
            mentions.append(mention)
            type_counts[prediction.type.value] = type_counts.get(prediction.type.value, 0) + 1
            board.extras.setdefault("predictions", {})[(cand.span.start, cand.span.end)] = (
                prediction,
                role,
                cand,
            )

        board.mentions = mentions
        board.note(
            "build_mentions",
            seconds=round(time.monotonic() - t0, 3),
            mentions=len(mentions),
            types=type_counts,
        )

    def _stage_verify(self, board: Blackboard, budget: Budget) -> None:
        t0 = time.monotonic()
        doc = board.document
        decisions: dict[tuple[int, int], Any] = {}
        counts = {"structural": 0, "embedding": 0, "llm": 0, "adjudicated": 0}

        residue = []
        for mention in board.mentions:
            key = (mention.span.start, mention.span.end)
            decision = self.structural_verifier.check(mention, doc)
            counts["structural"] += 1
            if decision.verdict is Verdict.REJECT:
                decisions[key] = decision
                continue
            if self.embedding_verifier is not None and self.embedding_verifier.available:
                decision = self.embedding_verifier.check(mention, doc)
                counts["embedding"] += 1
            if decision.verdict is Verdict.ABSTAIN:
                residue.append(mention)
            decisions[key] = decision

        if residue and self.batch_verifier is not None and self.batch_verifier.available:
            llm_decisions = self.batch_verifier.verify(residue, doc)
            counts["llm"] = len(llm_decisions)
            for key, decision in llm_decisions.items():
                if decision.verdict is not Verdict.ABSTAIN or decisions.get(key) is None:
                    decisions[key] = decision

        if self.adjudicator is not None and self.adjudicator.available:
            for mention in board.mentions:
                key = (mention.span.start, mention.span.end)
                decision = decisions.get(key)
                if decision is None or decision.verdict is not Verdict.REJECT:
                    continue
                # Only disputes worth the strongest model: the proposer had real
                # support and the verifier still said no.
                if len(mention.provenance.proposers) < 2:
                    continue
                verdict = self.adjudicator.adjudicate(mention, doc, decision)
                counts["adjudicated"] += 1
                if verdict.verdict is Verdict.PASS:
                    decisions[key] = decision.__class__(
                        Verdict.PASS, f"adjudicated: {verdict.reason}", 0.8, "adjudicator", True
                    )
                elif verdict.verdict is Verdict.ABSTAIN:
                    decisions[key] = decision.__class__(
                        Verdict.ABSTAIN, f"adjudicated: {verdict.reason}", 0.0, "adjudicator", True
                    )

        board.extras["verifier_decisions"] = decisions
        board.note(
            "verify",
            seconds=round(time.monotonic() - t0, 3),
            checked=counts,
            escalated=len(residue),
            batch=self.batch_verifier.stats.as_dict() if self.batch_verifier else None,
        )

    def _stage_guard_mentions(self, board: Blackboard) -> None:
        t0 = time.monotonic()
        ctx = GuardContext(
            document=board.document,
            analysis=board.analysis,
            extras={"verifier_decisions": board.extras.get("verifier_decisions", {})},
        )
        kept: list[TechMention] = []
        counts: dict[str, int] = {}
        flagged = 0

        for mention in board.mentions:
            outcome = self.mention_guards.run(mention, ctx)
            mention = mention.model_copy(update={"guards": outcome.verdicts})
            if outcome.verdict is Verdict.REJECT:
                counts[outcome.decided_by or "?"] = counts.get(outcome.decided_by or "?", 0) + 1
                if self.config.output.keep_rejected:
                    board.rejected.append(_mention_record(mention, outcome.reason))
                continue
            # A guard that could not decide does not get to delete the mention.
            # The system has exactly one abstention mechanism -- the calibrated
            # confidence threshold in _stage_score -- and every soft signal feeds
            # it. Letting each guard abstain independently produced a pipeline
            # whose recall depended on guard ordering and whose review queue was
            # full of items no one had scored.
            if outcome.verdict is Verdict.ABSTAIN:
                n_flags = sum(1 for v in outcome.verdicts if v.verdict is Verdict.ABSTAIN)
                board.extras.setdefault("guard_flags", {})[
                    (mention.span.start, mention.span.end)
                ] = n_flags
                flagged += 1
            kept.append(mention)

        board.mentions = kept
        board.note(
            "guard_mentions",
            seconds=round(time.monotonic() - t0, 3),
            kept=len(kept),
            rejected_by=counts,
            flagged=flagged,
        )

    def _stage_score(self, board: Blackboard) -> None:
        t0 = time.monotonic()
        predictions = board.extras.get("predictions", {})
        decisions = board.extras.get("verifier_decisions", {})
        guard_flags = board.extras.get("guard_flags", {})
        scored: list[TechMention] = []
        withheld = 0
        not_tech = 0

        for mention in board.mentions:
            key = (mention.span.start, mention.span.end)
            # A mention the type classifier called NOT_TECH is the classifier
            # saying the span is not a technology at all. Emitting it and leaving
            # the filtering to a downstream type whitelist worked, but it meant
            # the review queue and the per-document output were mostly furniture.
            if mention.type is TechType.NOT_TECH:
                board.rejected.append(_mention_record(mention, "typed not_tech"))
                not_tech += 1
                continue
            prediction, role_pred, cand = predictions.get(key, (None, None, None))
            decision = decisions.get(key)
            inputs = ConfidenceInputs(
                n_proposers=len(mention.provenance.proposers),
                trusted_proposer=bool(
                    set(mention.provenance.proposers) & {"abbrev", "gazetteer"}
                ),
                kb_hit=mention.kb_link is not None,
                type_margin=prediction.margin if prediction else 0.0,
                type_prob=prediction.score if prediction else 0.0,
                verifier_score=decision.score if decision else 0.0,
                verifier_passed=bool(decision and decision.verdict is Verdict.PASS),
                verifier_source=decision.source if decision else "",
                grounding_exact=bool(
                    cand is None or cand.features.get("grounding_exact", 1.0) >= 1.0
                ),
                section_weight=SECTION_WEIGHT.get(mention.section, 0.3),
                corpus_df=cand.features.get("corpus_df", 0.0) if cand else 0.0,
                llm_proposed="llm" in mention.provenance.proposers,
                verifier_abstained=bool(decision and decision.verdict is Verdict.ABSTAIN),
                guard_flags=guard_flags.get(key, 0)
                + int(bool(cand and cand.features.get("guard_abstain"))),
                distinctive_form=cand.features.get("orthographic", 0.0) if cand else 0.0,
                asserted_contribution=bool(
                    role_pred
                    and role_pred.role in (Role.PROPOSED, Role.CLAIMED)
                    and role_pred.margin >= 1.0
                ),
            )
            confidence = self.confidence.score(inputs)
            mention = mention.model_copy(update={"confidence": confidence})
            if confidence < self.config.output.abstain_below:
                board.abstained.append(
                    _mention_record(mention, f"confidence {confidence:.3f} below threshold")
                )
                withheld += 1
                continue
            scored.append(mention)

        board.mentions = sorted(scored, key=lambda m: (m.span.start, m.span.end))
        board.note(
            "score",
            seconds=round(time.monotonic() - t0, 3),
            emitted=len(scored),
            withheld_low_confidence=withheld,
            dropped_not_tech=not_tech,
        )

    # -- helpers ------------------------------------------------------------

    def _warm_embeddings(self, board: Blackboard) -> None:
        if self.embedding is None or not self.embedding.available:
            return
        surfaces = sorted({c.span.surface for c in board.candidates})
        if surfaces:
            self.embedding.encode(surfaces)

    def _evidence_for(self, board: Blackboard, cand: Candidate) -> Evidence | None:
        idx, start, end = board.analysis.sentence_containing(cand.span.start)
        if end <= start:
            return None
        # A mention straddling a sentence boundary means the splitter was wrong;
        # widen to cover it rather than emit evidence that excludes the mention.
        if cand.span.end > end:
            end = min(len(board.document.text), cand.span.end)
        text = board.document.text
        return Evidence(
            span=Span(start=start, end=end, surface=text[start:end]), sentence_index=idx
        )

    def _kb_link(self, cand: Candidate) -> KBLink | None:
        entry_id = cand.notes.get("entry_id")
        if not entry_id:
            entry = self.gazetteer.lookup(cand.span.surface)
            if entry is None:
                return None
            entry_id = entry.entry_id
            label = entry.label
            kb = entry.kb
            attested = entry.attested_from
        else:
            entry = self.gazetteer.entries.get(entry_id)
            label = cand.notes.get("kb_label", entry.label if entry else entry_id)
            kb = cand.notes.get("kb", entry.kb if entry else "cso")
            attested = entry.attested_from if entry else None
        return KBLink(
            kb=kb, entry_id=entry_id, label=label, score=1.0, attested_from=attested
        )

    def _finalise(self, board: Blackboard, budget: Budget) -> ExtractionResult:
        stats: dict[str, Any] = {
            "budget": budget.snapshot().__dict__,
            "n_candidates": len(board.candidates),
            "n_mentions": len(board.mentions),
            "n_rejected": len(board.rejected),
            "n_abstained": len(board.abstained),
            "abbreviation_pairs": board.extras.get("abbreviation_pairs", []),
            "backend": self.backend.name,
            "llm_available": self.backend.available,
        }
        if self._llm_stats:
            stats["llm_recall"] = dict(self._llm_stats)
        if self.batch_verifier is not None:
            stats["llm_verify"] = self.batch_verifier.stats.as_dict()
        if self.adjudicator is not None:
            stats["adjudicator"] = dict(self.adjudicator.stats)
        if self.embedding is not None and not self.embedding.available:
            stats["embedding_unavailable"] = self.embedding.unavailable_reason

        return ExtractionResult(
            doc_id=board.document.doc_id,
            genre=board.document.genre,
            mentions=board.mentions,
            rejected=board.rejected,
            abstained=board.abstained,
            trace=board.trace,
            stats=stats,
        )


def _record(doc: Document, cand: Candidate, reason: str) -> dict[str, Any]:
    return {
        "doc_id": doc.doc_id,
        "start": cand.span.start,
        "end": cand.span.end,
        "surface": cand.span.surface,
        "proposers": list(cand.proposers),
        "reason": reason,
    }


def _mention_record(mention: TechMention, reason: str) -> dict[str, Any]:
    return {
        "doc_id": mention.doc_id,
        "start": mention.span.start,
        "end": mention.span.end,
        "surface": mention.span.surface,
        "type": mention.type.value,
        "role": mention.role.value,
        "confidence": round(mention.confidence, 4),
        "proposers": list(mention.provenance.proposers),
        "reason": reason,
    }


def _normalized(surface: str) -> str:
    from ..lexicons import lemma_key

    return lemma_key(surface)
