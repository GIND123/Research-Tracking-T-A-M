"""Acronym / expansion mining, after Schwartz and Hearst (2003).

Worth having as a first-class recaller rather than a post-processing trick:
in both genres the *defining* mention of a technology is very often the
parenthetical one ("a convolutional neural network (CNN)"), so this recaller
recovers the canonical long form and the short form together, with no model and
no error bar.  The pairs it finds are also the highest-quality within-document
coreference links we get anywhere in the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schema import Candidate
from .base import RecallContext, Recaller

# "long form (SHORT)" and "SHORT (long form)".
_PAREN = re.compile(r"\(([^()]{2,120})\)")

_MAX_LONG_TOKENS = 10


@dataclass(frozen=True, slots=True)
class AbbreviationPair:
    short_start: int
    short_end: int
    short: str
    long_start: int
    long_end: int
    long: str


class AbbreviationRecaller(Recaller):
    name = "abbrev"
    version = "1"
    cost_tier = 0

    def propose(self, ctx: RecallContext) -> list[Candidate]:
        doc = ctx.document
        out: list[Candidate] = []
        pairs = find_pairs(doc.text)
        for pair in pairs:
            for start, end, kind in (
                (pair.long_start, pair.long_end, "long"),
                (pair.short_start, pair.short_end, "short"),
            ):
                cand = self._candidate(
                    doc,
                    start,
                    end,
                    features={"abbrev_pair": 1.0},
                    notes={
                        "form": kind,
                        "short": pair.short,
                        "long": pair.long,
                        "path": "schwartz_hearst",
                    },
                )
                if cand:
                    out.append(cand)
        ctx.options.setdefault("abbreviation_pairs", []).extend(pairs)
        return out


def find_pairs(text: str) -> list[AbbreviationPair]:
    pairs: list[AbbreviationPair] = []
    for m in _PAREN.finditer(text):
        inner = m.group(1).strip()
        inner_start = m.start(1) + (len(m.group(1)) - len(m.group(1).lstrip()))

        # Case A: parenthetical is the short form, long form precedes it.
        if _is_short_form(inner):
            left_end = m.start()
            window_start = max(0, left_end - _window_chars(inner))
            long_span = _match_long_form(text, window_start, left_end, inner)
            if long_span:
                pairs.append(
                    AbbreviationPair(
                        short_start=inner_start,
                        short_end=inner_start + len(inner),
                        short=inner,
                        long_start=long_span[0],
                        long_end=long_span[1],
                        long=text[long_span[0] : long_span[1]],
                    )
                )
            continue

        # Case B: parenthetical is the long form, short form precedes it.
        preceding = text[max(0, m.start() - 20) : m.start()].strip()
        token = preceding.split()[-1] if preceding.split() else ""
        if token and _is_short_form(token) and len(inner.split()) <= _MAX_LONG_TOKENS:
            long_span = _match_long_form(text, inner_start, inner_start + len(inner), token)
            if long_span:
                short_start = m.start() - len(preceding) + preceding.rfind(token)
                pairs.append(
                    AbbreviationPair(
                        short_start=short_start,
                        short_end=short_start + len(token),
                        short=token,
                        long_start=long_span[0],
                        long_end=long_span[1],
                        long=text[long_span[0] : long_span[1]],
                    )
                )
    return pairs


def _is_short_form(candidate: str) -> bool:
    """Schwartz-Hearst short-form conditions, slightly tightened.

    The original accepts anything 2-10 characters whose first character is
    alphanumeric.  We additionally require at least one upper-case letter, which
    removes a large class of false pairs in patents where parentheses hold
    reference numerals and units ("(10)", "(mm)").
    """
    if not (2 <= len(candidate) <= 10):
        return False
    if " " in candidate.strip():
        return False
    if not candidate[0].isalnum():
        return False
    if not any(c.isupper() for c in candidate):
        return False
    if not any(c.isalpha() for c in candidate):
        return False
    return True


def _window_chars(short: str) -> int:
    # Schwartz-Hearst: search min(|A| + 5, |A| * 2) words back. We use a
    # character proxy so we can stay in offset space.
    return min(len(short) + 5, len(short) * 2) * 9


def _match_long_form(text: str, lo: int, hi: int, short: str) -> tuple[int, int] | None:
    """Right-to-left character matching between the short form and the window."""
    window = text[lo:hi]
    letters = [c for c in short if c.isalnum()]
    if not letters:
        return None

    s_idx = len(letters) - 1
    w_idx = len(window) - 1
    first_char = letters[0].lower()

    while s_idx >= 0:
        target = letters[s_idx].lower()
        matched = False
        while w_idx >= 0:
            ch = window[w_idx].lower()
            if ch == target:
                # The first character of the abbreviation must align with the
                # start of a word in the long form.
                if s_idx == 0:
                    if w_idx == 0 or not window[w_idx - 1].isalnum():
                        matched = True
                        break
                else:
                    matched = True
                    break
            w_idx -= 1
        if not matched:
            return None
        if s_idx == 0:
            break
        w_idx -= 1
        s_idx -= 1

    if w_idx < 0:
        return None
    start = lo + w_idx
    end = hi
    # Trim trailing whitespace/punctuation left by the parenthesis boundary.
    while end > start and not text[end - 1].isalnum():
        end -= 1
    long_form = text[start:end]
    if not long_form or len(long_form.split()) > _MAX_LONG_TOKENS:
        return None
    if long_form[0].lower() != first_char:
        return None
    return (start, end)
