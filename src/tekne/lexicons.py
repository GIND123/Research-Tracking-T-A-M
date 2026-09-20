"""Loaders for the hand-curated lexical resources under ``tekne/resources``.

Everything is cached at module scope; the files are small and read-only.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any

_DET = re.compile(r"^(?:the|a|an|this|that|these|those|its|their|our|such|said)\s+", re.I)
_PLURAL = re.compile(r"(?<=[a-z])(?:ies|es|s)$")


def _read_text(name: str) -> str:
    return resources.files("tekne.resources").joinpath(name).read_text(encoding="utf-8")


def _read_json(name: str) -> dict[str, Any]:
    return json.loads(_read_text(name))


def _lines(name: str) -> list[str]:
    out = []
    for raw in _read_text(name).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.lower())
    return out


@lru_cache(maxsize=1)
def tech_heads() -> frozenset[str]:
    return frozenset(_lines("tech_heads.txt"))


@lru_cache(maxsize=1)
def negative_lexicon() -> frozenset[str]:
    return frozenset(_lines("negative_lexicon.txt"))


@lru_cache(maxsize=1)
def role_cues() -> dict[str, Any]:
    return _read_json("role_cues.json")


@lru_cache(maxsize=1)
def type_cues() -> dict[str, Any]:
    return _read_json("type_cues.json")


@lru_cache(maxsize=1)
def head_to_type() -> dict[str, str]:
    """Invert the per-type head-noun lists into a single lookup.

    Collisions are real ("framework" is both a tool and a method head,
    "catalyst" both material and artefact).  We resolve them by taking the
    higher-weighted type, which keeps the mapping deterministic.
    """
    cues = type_cues()
    weights = cues["head_weights"]
    out: dict[str, str] = {}
    for type_name, heads in cues["head_nouns"].items():
        for head in heads:
            prev = out.get(head)
            if prev is None or weights.get(type_name, 0) > weights.get(prev, 0):
                out[head] = type_name
    return out


@lru_cache(maxsize=1)
def umbrella_terms() -> frozenset[str]:
    return frozenset(t.lower() for t in type_cues()["umbrella"])


@lru_cache(maxsize=1)
def orthographic_patterns() -> list[tuple[str, re.Pattern[str], float]]:
    return [
        (name, re.compile(pat), float(weight))
        for name, pat, weight in type_cues()["orthographic"]["patterns"]
    ]


def strip_determiner(surface: str) -> str:
    return _DET.sub("", surface.strip()).strip()


def lemma_key(surface: str) -> str:
    """Cheap normalisation used for lexicon lookup and within-doc matching.

    Not a real lemmatiser: it lowercases, drops a leading determiner, collapses
    whitespace and strips a plural suffix from the final token only.  That is
    enough for gazetteer hits and deliberately leaves hyphenation and internal
    morphology alone, because those distinctions are often the whole point
    ("self-attention" vs "attention").
    """
    s = strip_determiner(surface.lower())
    s = re.sub(r"\s+", " ", s).strip(" \t\n.,;:()[]{}\"'")
    if not s:
        return s
    parts = s.split(" ")
    parts[-1] = _depluralize(parts[-1])
    return " ".join(parts)


#: Singular words that a naive "strip the final s" rule mangles. The -us/-is/-ss
#: families are handled by suffix below; these are the ones that are not, and
#: "bias" in particular appears constantly in this domain.
_SINGULAR_IN_S = frozenset(
    """
    bias gas atlas canvas alias lens news series species means
    physics mathematics statistics optics acoustics electronics mechanics
    dynamics kinetics robotics photonics informatics genomics proteomics
    ceramics logistics diagnostics analytics semantics ethics
    """.split()
)


def _depluralize(word: str) -> str:
    if len(word) <= 3 or not word.isalpha():
        return word
    if word in _SINGULAR_IN_S:
        return word
    # Irregulars we actually see: "analyses", "indices", "matrices", "bases".
    irregular = {
        "analyses": "analysis",
        "bases": "basis",
        "indices": "index",
        "matrices": "matrix",
        "vertices": "vertex",
        "media": "medium",
        "data": "data",
        "criteria": "criterion",
        "phenomena": "phenomenon",
    }
    if word in irregular:
        return irregular[word]
    if word.endswith("ss") or word.endswith("us") or word.endswith("is"):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and word[-3:-2] in ("s", "x", "z", "h"):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word
