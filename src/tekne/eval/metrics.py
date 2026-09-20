"""Scoring.

Three families of number, answering three different questions.

**Did we find the right spans?**  Strict and partial span F1.  Strict requires
exact offsets; partial credits an overlap.  We report both because the gap
between them is informative on its own: a system with high partial and low
strict F1 is finding the technologies and disagreeing about boundaries, which is
a much better failure than missing them.

**Did we say the right things about them?**  Type and role accuracy, measured on
correctly-found spans only, so that recall failures are not double-counted.

**Did we make anything up?**  The hallucination rate: the fraction of emitted
surfaces that are not verbatim substrings of the source document.  The design
intends this to be exactly zero, so it is reported as a check on the system
rather than as a comparison between systems -- a non-zero value is a bug report.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..schema import Document, Role, TechMention, TechType
from .gold import GoldDocument, GoldMention


@dataclass
class PRF:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> PRF:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return cls(precision, recall, f1, tp, fp, fn)

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class EvalReport:
    strict: PRF = field(default_factory=PRF)
    partial: PRF = field(default_factory=PRF)
    type_accuracy: float = 0.0
    role_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    n_system: int = 0
    n_gold: int = 0
    n_documents: int = 0
    per_genre: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    coverage: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strict": self.strict.as_dict(),
            "partial": self.partial.as_dict(),
            "type_accuracy": round(self.type_accuracy, 4),
            "role_accuracy": round(self.role_accuracy, 4),
            "hallucination_rate": round(self.hallucination_rate, 6),
            "n_system": self.n_system,
            "n_gold": self.n_gold,
            "n_documents": self.n_documents,
            "coverage": round(self.coverage, 4),
            "per_genre": self.per_genre,
            "per_type": self.per_type,
            "confusion": self.confusion,
            "notes": self.notes,
        }


#: Types the evaluation scores as technology. Gold mentions of other types are
#: kept in the file (they are useful for the type-confusion table) but a system
#: is neither rewarded nor penalised for emitting them.
DEFAULT_SCORED_TYPES = frozenset({TechType.ARTIFACT, TechType.METHOD, TechType.MATERIAL})


def evaluate(
    system: dict[str, Sequence[TechMention]],
    gold: dict[str, GoldDocument],
    documents: dict[str, Document],
    *,
    scored_types: frozenset[TechType] = DEFAULT_SCORED_TYPES,
    match_types: bool = False,
) -> EvalReport:
    """Score system output against gold.

    ``match_types`` controls whether a span match must also agree on type to
    count as a true positive.  We report the type-agnostic figure as the headline
    -- finding the technology is the task, and typing is measured separately --
    and the typed figure in the appendix table.
    """
    report = EvalReport(n_documents=len(gold))
    totals = Counter()
    type_hits = type_total = 0
    role_hits = role_total = 0
    hallucinated = emitted = 0
    confusion: dict[str, Counter[str]] = {}
    genre_counts: dict[str, Counter] = {}
    per_type_counts: dict[str, Counter] = {}

    for doc_id, gold_doc in gold.items():
        doc = documents.get(doc_id)
        if doc is None:
            report.notes.append(f"{doc_id}: document missing, skipped")
            continue

        gold_mentions = [m for m in gold_doc.mentions if m.type in scored_types]
        predicted = [
            m
            for m in system.get(doc_id, ())
            if m.type in scored_types and gold_doc.in_zone(m.span.start, m.span.end)
        ]

        for mention in system.get(doc_id, ()):
            emitted += 1
            if doc.text[mention.span.start : mention.span.end] != mention.span.surface:
                hallucinated += 1

        pairs, unmatched_sys, unmatched_gold = align(predicted, gold_mentions, strict=True)
        totals["tp"] += len(pairs)
        totals["fp"] += len(unmatched_sys)
        totals["fn"] += len(unmatched_gold)

        p_pairs, p_sys, p_gold = align(predicted, gold_mentions, strict=False)
        totals["ptp"] += len(p_pairs)
        totals["pfp"] += len(p_sys)
        totals["pfn"] += len(p_gold)

        genre = gold_doc.genre.value
        g = genre_counts.setdefault(genre, Counter())
        g["tp"] += len(pairs)
        g["fp"] += len(unmatched_sys)
        g["fn"] += len(unmatched_gold)

        for sys_mention, gold_mention in p_pairs:
            type_total += 1
            if sys_mention.type is gold_mention.type:
                type_hits += 1
            confusion.setdefault(gold_mention.type.value, Counter())[sys_mention.type.value] += 1

            t = per_type_counts.setdefault(gold_mention.type.value, Counter())
            t["matched"] += 1

            if gold_mention.role is not Role.UNKNOWN:
                role_total += 1
                if sys_mention.role is gold_mention.role:
                    role_hits += 1

        for gold_mention in p_gold:
            per_type_counts.setdefault(gold_mention.type.value, Counter())["missed"] += 1

        report.n_system += len(predicted)
        report.n_gold += len(gold_mentions)

    if match_types:
        report.notes.append("strict figures require type agreement")

    report.strict = PRF.from_counts(totals["tp"], totals["fp"], totals["fn"])
    report.partial = PRF.from_counts(totals["ptp"], totals["pfp"], totals["pfn"])
    report.type_accuracy = type_hits / type_total if type_total else 0.0
    report.role_accuracy = role_hits / role_total if role_total else 0.0
    report.hallucination_rate = hallucinated / emitted if emitted else 0.0
    report.coverage = report.n_system / report.n_gold if report.n_gold else 0.0
    report.per_genre = {
        genre: PRF.from_counts(c["tp"], c["fp"], c["fn"]).as_dict()
        for genre, c in genre_counts.items()
    }
    report.per_type = {
        name: {
            "matched": c["matched"],
            "missed": c["missed"],
            "recall": round(c["matched"] / (c["matched"] + c["missed"]), 4)
            if (c["matched"] + c["missed"])
            else 0.0,
        }
        for name, c in per_type_counts.items()
    }
    report.confusion = {k: dict(v) for k, v in confusion.items()}
    return report


def align(
    predicted: Sequence[TechMention],
    gold: Sequence[GoldMention],
    *,
    strict: bool,
) -> tuple[
    list[tuple[TechMention, GoldMention]], list[TechMention], list[GoldMention]
]:
    """One-to-one greedy alignment, best overlap first.

    Greedy rather than optimal (Hungarian) matching: with nested technology terms
    the two differ on a handful of cases per corpus, and greedy is easier to
    explain when someone disputes a score.
    """
    if strict:
        by_offsets: dict[tuple[int, int], GoldMention] = {(m.start, m.end): m for m in gold}
        pairs = []
        used: set[tuple[int, int]] = set()
        leftover_sys = []
        for mention in predicted:
            key = (mention.span.start, mention.span.end)
            match = by_offsets.get(key)
            if match is not None and key not in used:
                used.add(key)
                pairs.append((mention, match))
            else:
                leftover_sys.append(mention)
        leftover_gold = [m for m in gold if (m.start, m.end) not in used]
        return pairs, leftover_sys, leftover_gold

    scored: list[tuple[float, int, int]] = []
    for i, mention in enumerate(predicted):
        for j, gold_mention in enumerate(gold):
            overlap = _overlap(
                mention.span.start, mention.span.end, gold_mention.start, gold_mention.end
            )
            if overlap > 0:
                union = max(mention.span.end, gold_mention.end) - min(
                    mention.span.start, gold_mention.start
                )
                scored.append((overlap / union if union else 0.0, i, j))

    scored.sort(key=lambda t: -t[0])
    used_sys: set[int] = set()
    used_gold: set[int] = set()
    pairs = []
    for _score, i, j in scored:
        if i in used_sys or j in used_gold:
            continue
        used_sys.add(i)
        used_gold.add(j)
        pairs.append((predicted[i], gold[j]))
    return (
        pairs,
        [m for i, m in enumerate(predicted) if i not in used_sys],
        [m for j, m in enumerate(gold) if j not in used_gold],
    )


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def labelled_scores(
    system: dict[str, Sequence[TechMention]],
    gold: dict[str, GoldDocument],
    *,
    scored_types: frozenset[TechType] = DEFAULT_SCORED_TYPES,
) -> tuple[list[float], list[int]]:
    """Confidence scores paired with correctness, for the risk-coverage curve."""
    scores: list[float] = []
    labels: list[int] = []
    for doc_id, gold_doc in gold.items():
        gold_mentions = [m for m in gold_doc.mentions if m.type in scored_types]
        predicted = [
            m
            for m in system.get(doc_id, ())
            if m.type in scored_types and gold_doc.in_zone(m.span.start, m.span.end)
        ]
        pairs, unmatched, _ = align(predicted, gold_mentions, strict=False)
        matched = {id(m) for m, _g in pairs}
        for mention in predicted:
            scores.append(mention.confidence)
            labels.append(1 if id(mention) in matched else 0)
        del unmatched
    return scores, labels
