"""Type assignment over the extraction ontology.

The interesting decision here is not artefact-vs-method, it is
**technology-vs-not**: the recall stage hands over a stream of well-formed noun
phrases of which a large minority are tasks, fields, metrics or ordinary
vocabulary.  Getting that boundary right is worth more to a tracking corpus than
any refinement within the technology types, because a field admitted as an
artefact becomes a permanent, always-rising trend line.

The classifier is a linear model over interpretable features.  Defaults are set
by hand from a development sample; :meth:`TypeClassifier.fit` replaces them with
weights estimated from annotated data when any is available.  Keeping it linear
is a deliberate choice -- the decision has to be explainable in an audit, and
with a few hundred annotated mentions a linear model is also simply the better
estimator.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..lexicons import (
    head_to_type,
    lemma_key,
    orthographic_patterns,
    tech_heads,
    type_cues,
    umbrella_terms,
)
from ..schema import Candidate, Document, SectionKind, TechType

TYPE_ORDER: tuple[TechType, ...] = (
    TechType.ARTIFACT,
    TechType.METHOD,
    TechType.MATERIAL,
    TechType.TASK,
    TechType.FIELD,
    TechType.DATASET,
    TechType.METRIC,
    TechType.TOOL,
    TechType.NOT_TECH,
)

FEATURES: tuple[str, ...] = (
    "bias",
    "head_type_match",
    "orthographic",
    "umbrella",
    "kb_hit",
    "kb_depth_shallow",
    "kb_depth_deep",
    "n_tokens",
    "has_digit",
    "has_upper",
    "all_lower",
    "hyphenated",
    "abbrev_pair",
    "claim_zone",
    "title_zone",
    "llm_agrees",
    "ncvalue",
    "head_is_tech",
    "suffix_ing",
    "suffix_ion",
    "exact_head_match",
    "proto_tech",
    "proto_self",
)


@dataclass
class TypePrediction:
    type: TechType
    granularity: int
    score: float
    margin: float
    scores: dict[TechType, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)


def _default_weights() -> dict[TechType, dict[str, float]]:
    """Hand-set starting weights.

    Read them as log-odds contributions. The two that do most of the work are
    ``head_type_match`` (the candidate's head noun is listed under this type)
    and ``orthographic`` (it looks like a proper name), which is why artefact and
    method differ mainly along the second.
    """
    w: dict[TechType, dict[str, float]] = {t: dict.fromkeys(FEATURES, 0.0) for t in TYPE_ORDER}

    # ``exact_head_match`` fires when the whole candidate is a single token that
    # the lexicon lists under this type. For "BLEU" or "dataset" that is close to
    # decisive, and it has to outweigh the orthographic evidence that would
    # otherwise make every all-caps token an artefact.
    w[TechType.ARTIFACT].update(
        bias=-0.4, head_type_match=2.0, exact_head_match=0.2, orthographic=1.1,
        kb_hit=0.5, kb_depth_deep=0.8, has_digit=0.5, has_upper=0.7,
        hyphenated=0.3, abbrev_pair=1.0, llm_agrees=1.4, head_is_tech=0.6,
        n_tokens=0.15,
    )
    w[TechType.METHOD].update(
        bias=0.1, head_type_match=2.0, exact_head_match=0.6, orthographic=-0.3,
        kb_hit=0.4, kb_depth_deep=0.4, all_lower=0.6, llm_agrees=1.4,
        head_is_tech=0.5, n_tokens=0.2, ncvalue=0.15, suffix_ion=0.3,
    )
    w[TechType.MATERIAL].update(
        bias=-0.8, head_type_match=2.2, exact_head_match=0.8, orthographic=0.4,
        has_digit=0.3, llm_agrees=1.4, head_is_tech=0.3, n_tokens=0.1,
    )
    w[TechType.TASK].update(
        bias=-0.6, head_type_match=2.2, exact_head_match=1.0, orthographic=-0.5,
        all_lower=0.4, llm_agrees=1.4, suffix_ing=0.4, suffix_ion=0.8, n_tokens=0.1,
    )
    w[TechType.FIELD].update(
        bias=-1.2, head_type_match=2.0, exact_head_match=0.8, umbrella=3.2,
        kb_depth_shallow=1.4, all_lower=0.3, llm_agrees=1.4, orthographic=-0.6,
    )
    w[TechType.DATASET].update(
        bias=-1.6, head_type_match=2.6, exact_head_match=1.6, orthographic=0.8,
        has_upper=0.4, llm_agrees=1.4,
    )
    w[TechType.METRIC].update(
        bias=-1.6, head_type_match=2.4, exact_head_match=1.6, orthographic=0.3,
        llm_agrees=1.4,
    )
    w[TechType.TOOL].update(
        bias=-1.8, head_type_match=1.8, exact_head_match=1.2, orthographic=1.0,
        has_upper=0.6, llm_agrees=1.4,
    )
    # NOT_TECH used to carry a large positive bias, which was correct while the
    # chunker only proposed lexicon-headed phrases. With open-vocabulary recall
    # it made every unlisted head ("highback", "elision") not-a-technology and
    # cost 22% of recall. The bias is now slightly negative: the type classifier
    # is not the precision gate, the guard stack and the confidence threshold
    # are, and a wrong NOT_TECH is unrecoverable whereas a wrong METHOD is not.
    w[TechType.NOT_TECH].update(
        bias=-0.15, head_is_tech=-1.6, head_type_match=-1.0, orthographic=-0.8,
        kb_hit=-1.2, abbrev_pair=-0.8, umbrella=-1.0, llm_agrees=-0.6, ncvalue=-0.2,
        proto_tech=-1.9, proto_self=0.9,
    )
    for tech_type in (TechType.ARTIFACT, TechType.METHOD, TechType.MATERIAL, TechType.TOOL):
        w[tech_type]["proto_tech"] = 0.8
    for tech_type in TYPE_ORDER:
        w[tech_type]["proto_self"] = 1.1
    return w


class TypeClassifier:
    version = "2"

    def __init__(
        self,
        weights: dict[TechType, dict[str, float]] | None = None,
        *,
        embedding_scorer: Any = None,
    ) -> None:
        self.weights = weights or _default_weights()
        # Distributional back-off for heads the lexicon does not list. Note that
        # in the offline configuration this is the same signal the embedding
        # verifier uses, so that verifier degrades from an independent check to a
        # consistency check; full independence needs the LLM verifier tier.
        self.embedding = embedding_scorer
        self._heads = tech_heads()
        self._head_type = head_to_type()
        self._umbrella = umbrella_terms()
        self._ortho = orthographic_patterns()
        self._granularity = {
            k: int(v) for k, v in type_cues()["granularity"].items() if not k.startswith("_")
        }

    # -- features -----------------------------------------------------------

    def features(self, cand: Candidate, doc: Document, section: SectionKind) -> dict[str, float]:
        surface = cand.span.surface
        key = lemma_key(surface)
        tokens = key.split()
        head = tokens[-1] if tokens else ""
        notes = cand.notes or {}

        ortho = 0.0
        for _name, pattern, weight in self._ortho:
            if pattern.search(surface):
                ortho = max(ortho, weight)

        kb_depth = float(cand.features.get("kb_depth", -1.0))
        return {
            "bias": 1.0,
            # Filled per-type in ``score``; present here for the fitted model.
            "head_type_match": 0.0,
            "orthographic": min(ortho / 2.5, 1.0),
            "umbrella": 1.0 if key in self._umbrella else 0.0,
            "kb_hit": float(cand.features.get("kb_hit", 0.0)),
            "kb_depth_shallow": 1.0 if 0 <= kb_depth <= 1 else 0.0,
            "kb_depth_deep": 1.0 if kb_depth >= 3 else 0.0,
            "n_tokens": min(len(tokens), 5) / 5.0,
            "has_digit": 1.0 if any(c.isdigit() for c in surface) else 0.0,
            "has_upper": 1.0 if any(c.isupper() for c in surface[1:]) else 0.0,
            "all_lower": 1.0 if surface.islower() else 0.0,
            "hyphenated": 1.0 if "-" in surface else 0.0,
            "abbrev_pair": float(cand.features.get("abbrev_pair", 0.0)),
            "claim_zone": 1.0 if section in (SectionKind.CLAIM_INDEP, SectionKind.CLAIM_DEP) else 0.0,
            "title_zone": 1.0 if section is SectionKind.TITLE else 0.0,
            "llm_agrees": 0.0,  # per-type, see ``score``
            "ncvalue": _squash(cand.features.get("ncvalue", 0.0)),
            "head_is_tech": 1.0 if head in self._heads else 0.0,
            "suffix_ing": 1.0 if head.endswith("ing") else 0.0,
            "suffix_ion": 1.0 if head.endswith(("tion", "sion")) else 0.0,
            "exact_head_match": 0.0,  # per-type, see ``predict``
            "proto_tech": 0.0,  # filled in ``predict`` from the embedding scorer
            "proto_self": 0.0,
            "_head": head,  # type: ignore[dict-item]
            "_llm_type": notes.get("llm_type", ""),  # type: ignore[dict-item]
        }

    # -- scoring ------------------------------------------------------------

    def predict(self, cand: Candidate, doc: Document, section: SectionKind) -> TypePrediction:
        feats = self.features(cand, doc, section)
        head = str(feats.pop("_head", ""))
        llm_type = str(feats.pop("_llm_type", ""))
        head_type = self._head_type.get(head)
        single_token = len(lemma_key(cand.span.surface).split()) == 1
        proto = self._prototype_scores(cand.span.surface)

        scores: dict[TechType, float] = {}
        for tech_type in TYPE_ORDER:
            local = dict(feats)
            head_match = head_type == tech_type.value
            local["head_type_match"] = 1.0 if head_match else 0.0
            local["exact_head_match"] = 1.0 if (head_match and single_token) else 0.0
            local["llm_agrees"] = 1.0 if llm_type == tech_type.value else 0.0
            if proto is not None:
                local["proto_tech"] = proto["margin"]
                local["proto_self"] = proto["per_type"].get(tech_type, 0.0)
            weights = self.weights[tech_type]
            scores[tech_type] = sum(weights.get(name, 0.0) * value for name, value in local.items())

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, best_score = ranked[0]
        margin = best_score - ranked[1][1] if len(ranked) > 1 else best_score
        probs = _softmax([s for _t, s in ranked])
        return TypePrediction(
            type=best,
            granularity=self._granularity.get(best.value, 2),
            score=probs[0],
            margin=margin,
            scores=scores,
            features=feats,
        )

    # -- optional supervised refit ------------------------------------------

    def fit(
        self,
        examples: Sequence[tuple[Candidate, Document, SectionKind, TechType]],
        *,
        C: float = 1.0,
    ) -> dict[str, Any]:
        """Re-estimate the weights by multinomial logistic regression.

        Called from the evaluation driver on the development split only.  With
        fewer than ~40 examples per class the hand-set weights are usually
        better, so we refuse rather than overfit quietly.
        """
        from collections import Counter

        import numpy as np
        from sklearn.linear_model import LogisticRegression

        labels = [t for _c, _d, _s, t in examples]
        counts = Counter(labels)
        usable = {t for t, n in counts.items() if n >= 8}
        if len(usable) < 2:
            return {"fitted": False, "reason": "insufficient labelled data", "counts": dict(counts)}

        rows: list[list[float]] = []
        targets: list[str] = []
        for cand, doc, section, label in examples:
            if label not in usable:
                continue
            rows.append(self._design_row(cand, doc, section))
            targets.append(label.value)

        X = np.asarray(rows, dtype=float)
        y = np.asarray(targets)
        model = LogisticRegression(C=C, max_iter=2000, multi_class="multinomial")
        model.fit(X, y)

        design = list(FEATURES) + ["head_type_match_self", "llm_agrees_self"]
        for idx, class_name in enumerate(model.classes_):
            tech_type = TechType(class_name)
            coefs = model.coef_[idx] if model.coef_.ndim > 1 else model.coef_[0]
            table = dict.fromkeys(FEATURES, 0.0)
            for name, value in zip(design, coefs, strict=False):
                if name in table:
                    table[name] = float(value)
            table["head_type_match"] = float(coefs[design.index("head_type_match_self")])
            table["llm_agrees"] = float(coefs[design.index("llm_agrees_self")])
            intercept = model.intercept_[idx] if model.intercept_.ndim else model.intercept_
            table["bias"] = float(intercept)
            self.weights[tech_type] = table

        return {"fitted": True, "classes": list(model.classes_), "n": len(rows)}

    def _design_row(self, cand: Candidate, doc: Document, section: SectionKind) -> list[float]:
        feats = self.features(cand, doc, section)
        head = str(feats.pop("_head", ""))
        llm_type = str(feats.pop("_llm_type", ""))
        head_type = self._head_type.get(head)
        row = [feats.get(name, 0.0) for name in FEATURES]
        row.append(1.0 if head_type else 0.0)
        row.append(1.0 if llm_type else 0.0)
        return row


    def _prototype_scores(self, surface: str) -> dict[str, Any] | None:
        if self.embedding is None or not getattr(self.embedding, "available", False):
            return None
        score = self.embedding.score(surface)
        if not score.available:
            return None
        # Centre the per-type similarities so they act as relative evidence:
        # raw cosines against MiniLM centroids sit in a narrow band and would
        # otherwise just add a constant to every class.
        values = list(score.per_type.values())
        mean = sum(values) / len(values) if values else 0.0
        return {
            "margin": math.tanh(score.tech_margin * 6.0),
            "per_type": {t: (v - mean) * 6.0 for t, v in score.per_type.items()},
        }


def _softmax(values: Iterable[float]) -> list[float]:
    vals = list(values)
    if not vals:
        return []
    top = max(vals)
    exps = [math.exp(v - top) for v in vals]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _squash(value: float) -> float:
    return math.tanh(float(value) / 8.0)
