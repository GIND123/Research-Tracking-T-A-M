"""Lexical rejection of things that look like terms but carry no technology.

Patents are the hard case.  A head-noun recaller run over claim text proposes
"said apparatus", "the present invention" and "a plurality of elements" many
times per document, and every one of them is a well-formed noun phrase with a
technology-ish head.  Filtering them with a learned classifier is possible but
wasteful: the set is closed, conventional, and stable across decades of drafting
practice, so a curated lexicon is both cheaper and more auditable.

The non-obvious rule here is the *bare-generic* one.  "apparatus" alone is
furniture; "heat exchanger apparatus" is a technology.  Rejecting the head noun
outright would lose the second, so the lexicon entry only fires when it accounts
for the entire candidate.
"""

from __future__ import annotations

import re

from ..lexicons import lemma_key, negative_lexicon, strip_determiner
from ..schema import GuardVerdict
from .base import Guard, GuardContext

# Organisation and person markers: assignee and author names get proposed
# constantly because they sit next to technology terms.
_ORG_MARKERS = re.compile(
    r"\b(?:inc|llc|ltd|gmbh|corp|corporation|company|co|university|universit[ae]|"
    r"institute|laborator(?:y|ies)|foundation|college|school|department|"
    r"technologies|holdings|group|ag|sa|nv|kk|plc)\b\.?$",
    re.I,
)
_PERSON = re.compile(r"^[A-Z][a-z]+\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+)$")
_PURE_NUMBER = re.compile(r"^[\d\s.,%/+-]+$")
_UNIT = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:nm|um|mm|cm|m|km|mg|g|kg|ms|s|min|h|hz|khz|mhz|ghz|"
    r"v|mv|kv|a|ma|w|kw|mw|j|kj|k|c|f|pa|kpa|mpa|bar|psi|%|wt%|mol|ppm)$",
    re.I,
)

# Function words: a candidate made only of these is never a term.
_FUNCTION_WORDS = frozenset(
    """
    a an the of in on at to for with by from as is are was were be been being
    and or nor but so yet if then than that which who whom whose this these those
    it its their there here when where while during after before above below
    into onto upon within without between among through across over under
    said such any all each both more most less least other another same
    """.split()
)


class NegativeLexiconGuard(Guard):
    name = "negative_lexicon"
    stage = "candidate"
    blocking = True

    def __init__(self, *, min_chars: int = 3, max_tokens: int = 8) -> None:
        self.min_chars = min_chars
        self.max_tokens = max_tokens
        self._negative = negative_lexicon()

    def check(self, item, ctx: GuardContext) -> GuardVerdict:
        span = getattr(item, "span", None)
        if span is None:
            return self._reject("no span")
        surface = span.surface.strip()
        stripped = strip_determiner(surface)

        if len(stripped) < self.min_chars:
            return self._reject(f"too short: {surface!r}")

        tokens = stripped.split()
        if len(tokens) > self.max_tokens:
            return self._reject(f"too long: {len(tokens)} tokens")

        if all(t.lower().strip(".,;:") in _FUNCTION_WORDS for t in tokens):
            return self._reject("function words only")

        key = lemma_key(surface)
        if key in self._negative:
            return self._reject(f"negative lexicon: {key!r}")

        # Also test the determiner-stripped raw form, so "the present invention"
        # is caught even when lemmatisation moves the key.
        if stripped.lower() in self._negative:
            return self._reject(f"negative lexicon: {stripped.lower()!r}")

        if _PURE_NUMBER.match(stripped) or _UNIT.match(stripped):
            return self._reject("numeric or unit expression")

        if _ORG_MARKERS.search(stripped):
            return self._reject("organisation name")

        if _PERSON.match(stripped):
            return self._reject("person name")

        # A candidate ending in a preposition or conjunction is a chunking error.
        if tokens[-1].lower() in _FUNCTION_WORDS:
            return self._reject(f"dangling function word: {tokens[-1]!r}")

        return self._pass()
