"""Span grounding: the guard the rest of the design is organised around.

A generative extractor can return a technology name that appears nowhere in the
document.  Downstream, that fabrication is indistinguishable from a real find --
it has a plausible name, a plausible type, and it will be counted in a trend.
Filtering such cases *statistically* (confidence thresholds, self-consistency)
lowers the rate but cannot drive it to zero.

So we do not filter them.  We make them unrepresentable: an emitted mention is a
pair of character offsets into the source, and its surface form is defined as the
slice at those offsets.  A model that returns a string we cannot locate has
returned nothing.  The cost is that a model which correctly identifies a
technology but paraphrases its name is also discarded; :func:`ground` therefore
offers a bounded ladder of relocation attempts -- whitespace-insensitive, then
case-insensitive, then (optionally) a similarity snap -- each of which returns
*the document's* characters, never the model's.

The ladder is instrumented because the paper reports where on it each accepted
span landed; ``fuzzy`` is off by default precisely because it is the only rung
that can silently shift a span onto a different entity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..schema import Span
from .base import Guard, GuardContext
from ..schema import GuardVerdict


class GroundingStatus(str, Enum):
    EXACT = "exact"
    WHITESPACE = "whitespace"
    CASE = "case"
    FUZZY = "fuzzy"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"


@dataclass(slots=True)
class GroundingResult:
    status: GroundingStatus
    span: Span | None = None
    similarity: float = 0.0
    n_matches: int = 0

    @property
    def grounded(self) -> bool:
        return self.span is not None


_WS = re.compile(r"\s+")


def ground(
    text: str,
    claim: str,
    *,
    anchor: int | None = None,
    window: tuple[int, int] | None = None,
    allow_fuzzy: bool = False,
    fuzzy_threshold: float = 0.90,
) -> GroundingResult:
    """Locate ``claim`` inside ``text`` and return the document's own span.

    ``window`` restricts the search to a region (typically the evidence sentence
    the model cited, which is itself grounded first).  ``anchor`` disambiguates
    repeated surfaces by preferring the occurrence nearest that offset.
    """
    claim = claim.strip()
    if not claim or not text:
        return GroundingResult(GroundingStatus.UNMATCHED)

    lo, hi = window if window else (0, len(text))
    lo = max(0, min(lo, len(text)))
    hi = max(lo, min(hi, len(text)))
    region = text[lo:hi]

    exact = _all_occurrences(region, claim)
    if exact:
        start = _pick(exact, anchor, lo)
        return GroundingResult(
            GroundingStatus.EXACT,
            _span(text, lo + start, lo + start + len(claim)),
            1.0,
            len(exact),
        )

    ws = _whitespace_insensitive(region, claim)
    if ws:
        start, end = _pick_pair(ws, anchor, lo)
        return GroundingResult(
            GroundingStatus.WHITESPACE, _span(text, lo + start, lo + end), 1.0, len(ws)
        )

    ci = _all_occurrences(region.lower(), claim.lower())
    if ci:
        start = _pick(ci, anchor, lo)
        return GroundingResult(
            GroundingStatus.CASE, _span(text, lo + start, lo + start + len(claim)), 1.0, len(ci)
        )

    if allow_fuzzy:
        hit = _fuzzy(region, claim, fuzzy_threshold)
        if hit:
            start, end, score = hit
            return GroundingResult(
                GroundingStatus.FUZZY, _span(text, lo + start, lo + end), score, 1
            )

    return GroundingResult(GroundingStatus.UNMATCHED)


def _span(text: str, start: int, end: int) -> Span | None:
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))
    if end <= start:
        return None
    return Span(start=start, end=end, surface=text[start:end])


def _all_occurrences(haystack: str, needle: str) -> list[int]:
    out: list[int] = []
    pos = haystack.find(needle)
    while pos != -1 and len(out) < 64:
        out.append(pos)
        pos = haystack.find(needle, pos + 1)
    return out


def _pick(positions: list[int], anchor: int | None, offset: int) -> int:
    if anchor is None or len(positions) == 1:
        return positions[0]
    return min(positions, key=lambda p: abs(p + offset - anchor))


def _pick_pair(pairs: list[tuple[int, int]], anchor: int | None, offset: int) -> tuple[int, int]:
    if anchor is None or len(pairs) == 1:
        return pairs[0]
    return min(pairs, key=lambda p: abs(p[0] + offset - anchor))


def _whitespace_insensitive(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Match ignoring how whitespace is distributed, keeping real offsets."""
    collapsed = _WS.sub(" ", needle).strip()
    if not collapsed:
        return []
    pattern = re.compile(r"\s+".join(re.escape(tok) for tok in collapsed.split(" ")))
    return [(m.start(), m.end()) for m in pattern.finditer(haystack)][:64]


def _fuzzy(haystack: str, needle: str, threshold: float) -> tuple[int, int, float] | None:
    """Best character-level alignment of ``needle`` over token-aligned windows.

    Deliberately restricted: windows must begin and end on token boundaries and
    be within 40% of the claim's length, so the result is still a well-formed
    phrase from the document rather than an arbitrary character range.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:  # pragma: no cover - optional dependency
        return None

    tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", haystack)]
    if not tokens:
        return None
    target_len = len(needle)
    best: tuple[int, int, float] | None = None

    for i in range(len(tokens)):
        for j in range(i, min(i + 8, len(tokens))):
            start, end = tokens[i][0], tokens[j][1]
            width = end - start
            if width < target_len * 0.6 or width > target_len * 1.4:
                continue
            score = fuzz.ratio(haystack[start:end].lower(), needle.lower()) / 100.0
            if score >= threshold and (best is None or score > best[2]):
                best = (start, end, score)
    return best


class SpanGroundingGuard(Guard):
    """Assert that a candidate's surface is literally the document's own text.

    Everything upstream is *supposed* to produce grounded spans; this guard is
    the assertion that makes that a checked property rather than a convention,
    and it is the one guard that never abstains.
    """

    name = "grounding"
    stage = "candidate"
    blocking = True

    def check(self, item, ctx: GuardContext) -> GuardVerdict:
        span = getattr(item, "span", None)
        if span is None:
            return self._reject("candidate carries no span")
        text = ctx.document.text
        if span.end > len(text):
            return self._reject(f"span end {span.end} beyond document length {len(text)}")
        actual = text[span.start : span.end]
        if actual != span.surface:
            return self._reject(f"surface {span.surface!r} != document slice {actual!r}")
        if not any(ch.isalnum() for ch in actual):
            return self._reject("span contains no alphanumeric character")
        return self._pass("verbatim")
