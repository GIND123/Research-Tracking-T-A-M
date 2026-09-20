"""Distributional type scoring against seeded prototypes.

The verifier needs a signal the proposer did not already use.  Head-noun
lexicons, orthography and the KB are all shared upstream, so re-reading them
would produce a verifier that agrees with the proposer by construction -- the
standard failure of self-consistency checks.  Sentence embeddings are the
cheapest genuinely independent view available: the proposer never sees them.

Prototypes come from ``resources/type_prototypes.json``, a small hand-written
seed set per class.  Nearest-centroid over seeds is a weak classifier on its own;
it is used here for what it is good at, which is the coarse
technology-vs-furniture separation where the seeds are unambiguous.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

from ..schema import TechType

#: Classes whose centroid counts as evidence *for* being a technology.
TECHNOLOGY_CLASSES = (TechType.ARTIFACT, TechType.METHOD, TechType.MATERIAL, TechType.TOOL)
NON_TECHNOLOGY_CLASSES = (TechType.NOT_TECH, TechType.FIELD, TechType.TASK, TechType.METRIC)


@dataclass
class EmbeddingScore:
    per_type: dict[TechType, float] = field(default_factory=dict)
    tech_margin: float = 0.0
    best_type: TechType | None = None
    available: bool = False


@lru_cache(maxsize=1)
def _prototypes() -> dict[str, list[str]]:
    payload = json.loads(
        resources.files("tekne.resources").joinpath("type_prototypes.json").read_text("utf-8")
    )
    return {k: v for k, v in payload.items() if not k.startswith("_")}


class EmbeddingScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._centroids: Any = None
        self._cache: dict[str, Any] = {}
        self._unavailable_reason: str | None = None

    # -- lifecycle ----------------------------------------------------------

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            self._unavailable_reason = f"sentence-transformers unavailable: {exc}"
            return False
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # pragma: no cover - model download/load failure
            self._unavailable_reason = f"could not load {self.model_name}: {exc}"
            return False

        protos = _prototypes()
        names = list(protos)
        flat = [term for name in names for term in protos[name]]
        vectors = self._model.encode(flat, normalize_embeddings=True, show_progress_bar=False)
        centroids: dict[str, Any] = {}
        cursor = 0
        for name in names:
            n = len(protos[name])
            block = vectors[cursor : cursor + n]
            centroid = block.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            centroids[name] = centroid
            cursor += n
        self._centroids = centroids
        return True

    @property
    def available(self) -> bool:
        return self._ensure()

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    # -- scoring ------------------------------------------------------------

    def encode(self, texts: list[str]) -> Any:
        if not self._ensure():
            return None
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vectors = self._model.encode(  # type: ignore[union-attr]
                missing, normalize_embeddings=True, show_progress_bar=False
            )
            for text, vector in zip(missing, vectors):
                self._cache[text] = vector
        import numpy as np

        return np.stack([self._cache[t] for t in texts])

    def score(self, surface: str) -> EmbeddingScore:
        return self.score_batch([surface])[0]

    def score_batch(self, surfaces: list[str]) -> list[EmbeddingScore]:
        if not surfaces:
            return []
        vectors = self.encode(surfaces)
        if vectors is None:
            return [EmbeddingScore(available=False) for _ in surfaces]

        out: list[EmbeddingScore] = []
        for vector in vectors:
            sims: dict[TechType, float] = {}
            for name, centroid in self._centroids.items():  # type: ignore[union-attr]
                try:
                    sims[TechType(name)] = float(vector @ centroid)
                except ValueError:
                    continue
            tech = max((sims.get(t, -1.0) for t in TECHNOLOGY_CLASSES), default=0.0)
            other = max((sims.get(t, -1.0) for t in NON_TECHNOLOGY_CLASSES), default=0.0)
            best = max(sims, key=lambda t: sims[t]) if sims else None
            out.append(
                EmbeddingScore(
                    per_type=sims, tech_margin=tech - other, best_type=best, available=True
                )
            )
        return out
