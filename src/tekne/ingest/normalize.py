"""Offset-preserving text normalisation.

Everything downstream addresses text by character offset, so normalisation has
to be invertible: we keep a per-character map back into the raw string.  That
costs a list of ints per document and buys the ability to quote a mention out of
the *original* PDF text or patent XML when a human audits it.

The transformations are deliberately conservative.  Anything that would change
token identity (case folding, stemming, stopword removal) belongs in
:mod:`tekne.link.canonical`, not here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Ligatures and typographic characters that survive PDF extraction and break
# naive string matching against a gazetteer.
_CHAR_MAP = {
    # Ligatures. Written as escapes rather than literals because several of the
    # characters below are invisible in an editor, and a char map you cannot
    # read is a char map nobody will ever correct.
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    # Dashes and minus signs, all folded to ASCII hyphen.
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    # Quotation marks.
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    # Spaces that are not U+0020.
    "\u00a0": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    # Zero-width and formatting characters, dropped entirely.
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
    "\u00ad": "",  # soft hyphen
}

# "convolu-\ntional" -> "convolutional".  Only fires when the fragment either
# side is lowercase, so "end-to-end\nlearning" and "3GPP-\nLTE" survive intact.
_LINEBREAK_HYPHEN = re.compile(r"([a-z])-\s*\n\s*([a-z])")

# Reference markers: "[12]", "[3, 4]", "[1]-[5]".  Masked to spaces so that a
# term abutting a citation still yields a clean span.
_CITATION_BRACKET = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")

# "(Devlin et al., 2019)" / "(see Smith 2020)" author-year citations.
_CITATION_PAREN = re.compile(
    r"\((?:[A-Z][A-Za-z'`-]+(?:\s+(?:et\s+al\.?|and|&|,)\s*)*)+,?\s*\d{4}[a-z]?"
    r"(?:\s*;\s*[^()]{0,80}?\d{4}[a-z]?)*\)"
)


@dataclass(slots=True)
class NormalizedText:
    """Normalised text plus the map from its offsets back to the raw string."""

    text: str
    #: ``raw_index[i]`` is the offset in ``raw`` that produced ``text[i]``.
    raw_index: list[int]
    raw: str

    def to_raw_span(self, start: int, end: int) -> tuple[int, int]:
        """Project a normalised span back onto raw-string offsets."""
        if not self.raw_index:
            return (0, 0)
        lo = self.raw_index[min(start, len(self.raw_index) - 1)]
        hi = self.raw_index[min(end - 1, len(self.raw_index) - 1)] + 1
        return (lo, max(hi, lo + 1))

    def raw_quote(self, start: int, end: int) -> str:
        lo, hi = self.to_raw_span(start, end)
        return self.raw[lo:hi]


def normalize(raw: str, *, mask_citations: bool = True) -> NormalizedText:
    """Normalise ``raw`` while tracking where every output character came from."""
    # Pass 1: NFKC-safe character substitution, char by char, so the index map
    # stays exact. We avoid unicodedata.normalize() on the whole string because
    # it silently changes lengths in ways we cannot attribute.
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(raw):
        repl = _CHAR_MAP.get(ch)
        if repl is None:
            cat = unicodedata.category(ch)
            # Drop control characters other than newline/tab.
            if cat == "Cc" and ch not in "\n\t":
                continue
            repl = ch
        for out_ch in repl:
            chars.append(out_ch)
            index.append(i)

    stage1 = "".join(chars)

    # Pass 2: regex-driven edits. Each one is expressed as (start, end, replacement)
    # on stage1 and applied together so index bookkeeping happens once.
    edits: list[tuple[int, int, str]] = []
    for m in _LINEBREAK_HYPHEN.finditer(stage1):
        edits.append((m.start(), m.end(), m.group(1) + m.group(2)))
    if mask_citations:
        for pattern in (_CITATION_BRACKET, _CITATION_PAREN):
            for m in pattern.finditer(stage1):
                edits.append((m.start(), m.end(), " " * (m.end() - m.start())))

    stage2, index2 = _apply_edits(stage1, index, edits)

    # Pass 3: whitespace collapse. Newlines become spaces; runs collapse to one.
    out_chars: list[str] = []
    out_index: list[int] = []
    prev_ws = False
    for ch, src in zip(stage2, index2, strict=True):
        if ch.isspace():
            if prev_ws:
                continue
            out_chars.append(" ")
            out_index.append(src)
            prev_ws = True
        else:
            out_chars.append(ch)
            out_index.append(src)
            prev_ws = False

    # Trim leading/trailing whitespace without losing the map.
    lo, hi = 0, len(out_chars)
    while lo < hi and out_chars[lo] == " ":
        lo += 1
    while hi > lo and out_chars[hi - 1] == " ":
        hi -= 1

    return NormalizedText(text="".join(out_chars[lo:hi]), raw_index=out_index[lo:hi], raw=raw)


def _apply_edits(
    text: str, index: list[int], edits: list[tuple[int, int, str]]
) -> tuple[str, list[int]]:
    if not edits:
        return text, index
    edits.sort(key=lambda e: e[0])
    out_chars: list[str] = []
    out_index: list[int] = []
    cursor = 0
    for start, end, repl in edits:
        if start < cursor:  # overlapping match, keep the first
            continue
        out_chars.extend(text[cursor:start])
        out_index.extend(index[cursor:start])
        # Replacement characters inherit the source offset of the region start.
        anchor = index[start] if start < len(index) else (index[-1] if index else 0)
        for j, ch in enumerate(repl):
            out_chars.append(ch)
            out_index.append(index[start + j] if start + j < end else anchor)
        cursor = end
    out_chars.extend(text[cursor:])
    out_index.extend(index[cursor:])
    return "".join(out_chars), out_index


# --- sentence segmentation -------------------------------------------------

# Abbreviations whose trailing period does not end a sentence. The patent side
# of this list matters more than the paper side: "U.S. Pat. No. 5,123,456" alone
# would otherwise produce four spurious sentence breaks.
_ABBREV = frozenset(
    """
    e.g i.e cf vs etc al fig figs eq eqs ref refs sec secs tab tabs approx resp
    no nos pat pats u.s us ser inc ltd corp co dept univ dr prof mr mrs ms jr sr
    vol pp ch app appl pub publ int nat proc trans j
    """.split()
)

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split into sentences, returning ``(start, end)`` offsets into ``text``.

    A hand-rolled splitter rather than a dependency: patent claim text is a
    pathological case for off-the-shelf models (single 400-word "sentences"
    punctuated by semicolons) and we need offsets, which several popular
    splitters do not expose reliably.
    """
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_BOUNDARY.finditer(text):
        end = m.start()
        if _is_false_break(text, end):
            continue
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _is_false_break(text: str, end: int) -> bool:
    """True if the period at ``end - 1`` is an abbreviation or a decimal point."""
    if end == 0 or text[end - 1] != ".":
        return False
    # Walk back over the token preceding the period.
    j = end - 1
    k = j
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] in ".-"):
        k -= 1
    token = text[k:j].lower().strip(".")
    if token in _ABBREV:
        return True
    # Single initial ("J. Smith") or a decimal fragment ("0.").
    if len(token) == 1 and token.isalpha():
        return True
    if token.isdigit():
        return True
    return False
