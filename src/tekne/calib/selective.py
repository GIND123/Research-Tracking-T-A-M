"""Confidence, calibration and selective prediction.

A technology-tracking corpus does not need every mention; it needs the mentions
it keeps to be right, and it needs to know which ones it threw away.  That is
the selective-prediction setting: the system may abstain on a fraction of inputs
and is judged by the risk it carries at each level of coverage, rather than by a
single F1 at full coverage.

Two things live here.  :class:`ConfidenceModel` turns the evidence a mention
accumulated into a score -- deliberately a small, legible linear form, because
the score is shown to whoever works the review queue.  :func:`risk_coverage`
turns scores plus labels into the curve we report, whose summary statistic
(area under the risk-coverage curve, lower is better) is the number that
actually distinguishes the configurations in the evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ConfidenceInputs:
    n_proposers: int = 0
    trusted_proposer: bool = False
    kb_hit: bool = False
    type_margin: float = 0.0
    type_prob: float = 0.0
    verifier_score: float = 0.0
    verifier_passed: bool = False
    verifier_source: str = ""
    grounding_exact: bool = True
    section_weight: float = 0.0
    corpus_df: float = 0.0
    llm_proposed: bool = False
    #: A verifier that could not settle the case is evidence, just weaker and of
    #: the opposite sign to a pass. Folding it in here rather than letting it
    #: drop the mention is what keeps a single abstention mechanism in the system.
    verifier_abstained: bool = False
    #: Number of non-blocking guards that flagged the mention (consensus
    #: shortfall, injection-like context, withheld KB link).
    guard_flags: int = 0
    #: Orthographic distinctiveness: CamelCase, an acronym, a version number.
    #: A form like "SwitchRoute" or "LLaMA-2" is almost never ordinary
    #: vocabulary, and this is the main evidence available for a technology too
    #: new to be in any lexicon -- precisely the population we must not lose.
    distinctive_form: float = 0.0
    #: The document frames this mention as its own contribution, with a cue
    #: strong enough to be unambiguous ("we propose X", an independent claim).
    asserted_contribution: bool = False


DEFAULT_WEIGHTS: dict[str, float] = {
    "bias": -1.55,
    "n_proposers": 0.62,
    "trusted_proposer": 0.85,
    "kb_hit": 0.70,
    "type_margin": 0.30,
    "type_prob": 1.25,
    "verifier_score": 1.10,
    "verifier_passed": 1.35,
    "verifier_llm": 0.45,
    "grounding_exact": 0.55,
    "section_weight": 0.40,
    "corpus_df": 0.22,
    "llm_proposed": 0.30,
    "verifier_abstained": -0.75,
    "guard_flags": -0.60,
    "distinctive_form": 0.80,
    "asserted_contribution": 0.65,
}


class ConfidenceModel:
    """Logistic combination of the evidence a mention accumulated."""

    version = "2"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    def features(self, inputs: ConfidenceInputs) -> dict[str, float]:
        return {
            "bias": 1.0,
            "n_proposers": min(inputs.n_proposers, 4) / 4.0,
            "trusted_proposer": 1.0 if inputs.trusted_proposer else 0.0,
            "kb_hit": 1.0 if inputs.kb_hit else 0.0,
            "type_margin": math.tanh(inputs.type_margin / 3.0),
            "type_prob": inputs.type_prob,
            "verifier_score": inputs.verifier_score,
            "verifier_passed": 1.0 if inputs.verifier_passed else 0.0,
            "verifier_llm": 1.0 if inputs.verifier_source == "llm" else 0.0,
            "grounding_exact": 1.0 if inputs.grounding_exact else 0.0,
            "section_weight": inputs.section_weight,
            "corpus_df": math.tanh(inputs.corpus_df / 5.0),
            "llm_proposed": 1.0 if inputs.llm_proposed else 0.0,
            "verifier_abstained": 1.0 if inputs.verifier_abstained else 0.0,
            "guard_flags": min(inputs.guard_flags, 3) / 3.0,
            "distinctive_form": min(inputs.distinctive_form / 2.5, 1.0),
            "asserted_contribution": 1.0 if inputs.asserted_contribution else 0.0,
        }

    def score(self, inputs: ConfidenceInputs) -> float:
        feats = self.features(inputs)
        z = sum(self.weights.get(name, 0.0) * value for name, value in feats.items())
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, examples: Sequence[tuple[ConfidenceInputs, int]], *, C: float = 1.0) -> dict[str, Any]:
        """Re-estimate the weights from labelled mentions (development split only)."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        if len(examples) < 40 or len({y for _x, y in examples}) < 2:
            return {"fitted": False, "reason": "insufficient labelled data", "n": len(examples)}

        names = [n for n in DEFAULT_WEIGHTS if n != "bias"]
        X = np.asarray([[self.features(x)[n] for n in names] for x, _y in examples])
        y = np.asarray([label for _x, label in examples])
        model = LogisticRegression(C=C, max_iter=2000)
        model.fit(X, y)
        self.weights = {"bias": float(model.intercept_[0])}
        self.weights.update({n: float(w) for n, w in zip(names, model.coef_[0])})
        return {"fitted": True, "n": len(examples), "weights": dict(self.weights)}


# --- selective prediction --------------------------------------------------


@dataclass
class RiskCoveragePoint:
    threshold: float
    coverage: float
    precision: float
    risk: float
    n_kept: int


@dataclass
class RiskCoverageCurve:
    points: list[RiskCoveragePoint] = field(default_factory=list)
    aurc: float = 0.0
    n_total: int = 0

    def precision_at_coverage(self, target: float) -> float | None:
        best: RiskCoveragePoint | None = None
        for point in self.points:
            if point.coverage >= target and (best is None or point.coverage < best.coverage):
                best = point
        return best.precision if best else None

    def coverage_at_precision(self, target: float) -> float:
        """Largest coverage whose precision still meets ``target``."""
        eligible = [p for p in self.points if p.precision >= target]
        return max((p.coverage for p in eligible), default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "aurc": round(self.aurc, 5),
            "n_total": self.n_total,
            "points": [
                {
                    "threshold": round(p.threshold, 4),
                    "coverage": round(p.coverage, 4),
                    "precision": round(p.precision, 4),
                    "risk": round(p.risk, 4),
                    "n_kept": p.n_kept,
                }
                for p in self.points
            ],
        }


def risk_coverage(scores: Sequence[float], labels: Sequence[int]) -> RiskCoverageCurve:
    """Risk-coverage curve for a selective predictor.

    Items are sorted by descending confidence; at each prefix we report the
    coverage (fraction kept) and the risk (error rate among kept items).  The
    area under this curve is the headline selective-prediction metric: it rewards
    a model whose errors are concentrated in its low-confidence tail, which is
    exactly the property a review queue needs.
    """
    if not scores or len(scores) != len(labels):
        return RiskCoverageCurve(n_total=len(scores))

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    total = len(order)
    correct = 0
    points: list[RiskCoveragePoint] = []
    risks: list[tuple[float, float]] = []

    for rank, idx in enumerate(order, start=1):
        correct += int(labels[idx])
        coverage = rank / total
        precision = correct / rank
        risk = 1.0 - precision
        points.append(RiskCoveragePoint(scores[idx], coverage, precision, risk, rank))
        risks.append((coverage, risk))

    aurc = 0.0
    for i in range(1, len(risks)):
        x0, r0 = risks[i - 1]
        x1, r1 = risks[i]
        aurc += (x1 - x0) * (r0 + r1) / 2.0
    if risks:
        aurc += risks[0][0] * risks[0][1]

    return RiskCoverageCurve(points=points, aurc=aurc, n_total=total)


def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> float:
    """Standard binned ECE. Reported alongside AURC so that a well-ranked but
    badly-scaled confidence is not mistaken for a calibrated one."""
    if not scores:
        return 0.0
    buckets: list[list[int]] = [[] for _ in range(bins)]
    bucket_scores: list[list[float]] = [[] for _ in range(bins)]
    for score, label in zip(scores, labels):
        idx = min(int(score * bins), bins - 1)
        buckets[idx].append(label)
        bucket_scores[idx].append(score)

    total = len(scores)
    ece = 0.0
    for labels_in_bin, scores_in_bin in zip(buckets, bucket_scores):
        if not labels_in_bin:
            continue
        accuracy = sum(labels_in_bin) / len(labels_in_bin)
        confidence = sum(scores_in_bin) / len(scores_in_bin)
        ece += (len(labels_in_bin) / total) * abs(accuracy - confidence)
    return ece


def choose_threshold(curve: RiskCoverageCurve, *, target_precision: float = 0.90) -> float:
    """Lowest confidence threshold whose kept set still meets a precision target."""
    eligible = [p for p in curve.points if p.precision >= target_precision]
    if not eligible:
        return 1.0
    return min(eligible, key=lambda p: p.threshold).threshold
