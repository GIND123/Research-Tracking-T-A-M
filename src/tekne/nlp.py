"""Shared linguistic analysis, computed once per document and passed around.

Recallers and classifiers all want the same things (tokens, POS tags, noun
chunks, sentence offsets).  Recomputing them per stage was the single largest
cost in an early profile, so the orchestrator builds one :class:`Analysis` and
puts it on the blackboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from .ingest.normalize import sentence_spans

if TYPE_CHECKING:  # pragma: no cover
    from .schema import Document

_MODEL_NAME = "en_core_web_sm"


@lru_cache(maxsize=1)
def _pipeline() -> Any:
    """Load spaCy lazily; the parser is needed for ``noun_chunks``.

    NER is disabled: its ontology (PERSON/ORG/GPE) is close to useless here and
    it costs roughly a third of the runtime.
    """
    import spacy

    try:
        return spacy.load(_MODEL_NAME, exclude=["ner", "lemmatizer"])
    except OSError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            f"spaCy model {_MODEL_NAME!r} is missing. Run: python -m spacy download {_MODEL_NAME}"
        ) from exc


def pipeline_available() -> bool:
    try:
        _pipeline()
        return True
    except Exception:
        return False


@dataclass(slots=True)
class Chunk:
    """A candidate noun phrase with its head, in document character offsets."""

    start: int
    end: int
    text: str
    head_start: int
    head_end: int
    head_text: str
    head_pos: str


@dataclass(slots=True)
class Analysis:
    text: str
    sentences: list[tuple[int, int]]
    chunks: list[Chunk] = field(default_factory=list)
    #: (start, end, text, pos, tag) per token, in document offsets.
    tokens: list[tuple[int, int, str, str, str]] = field(default_factory=list)
    spacy_doc: Any = None

    def sentence_containing(self, offset: int) -> tuple[int, int, int]:
        """Return ``(index, start, end)`` of the sentence covering ``offset``."""
        for i, (s, e) in enumerate(self.sentences):
            if s <= offset < e:
                return i, s, e
        if self.sentences:
            return len(self.sentences) - 1, *self.sentences[-1]
        return -1, 0, len(self.text)

    def window(self, start: int, end: int, radius: int) -> str:
        return self.text[max(0, start - radius) : min(len(self.text), end + radius)]


#: spaCy's default max_length is 1e6 chars; patents can exceed it in pathological
#: cases, so we chunk the text and offset the results.
_MAX_CHARS = 200_000


def analyse(doc: Document) -> Analysis:
    text = doc.text
    sents = sentence_spans(text)
    analysis = Analysis(text=text, sentences=sents)

    try:
        nlp = _pipeline()
    except RuntimeError:
        # Degraded mode: the pattern recaller falls back to its regex chunker and
        # the orchestrator records that the parse-dependent recallers were skipped.
        return analysis

    for base, segment in _segments(text):
        sdoc = nlp(segment)
        if analysis.spacy_doc is None:
            analysis.spacy_doc = sdoc
        for tok in sdoc:
            analysis.tokens.append(
                (base + tok.idx, base + tok.idx + len(tok.text), tok.text, tok.pos_, tok.tag_)
            )
        for nc in sdoc.noun_chunks:
            head = nc.root
            analysis.chunks.append(
                Chunk(
                    start=base + nc.start_char,
                    end=base + nc.end_char,
                    text=nc.text,
                    head_start=base + head.idx,
                    head_end=base + head.idx + len(head.text),
                    head_text=head.text,
                    head_pos=head.pos_,
                )
            )
    return analysis


def _segments(text: str) -> list[tuple[int, str]]:
    if len(text) <= _MAX_CHARS:
        return [(0, text)]
    out: list[tuple[int, str]] = []
    pos = 0
    while pos < len(text):
        end = min(pos + _MAX_CHARS, len(text))
        if end < len(text):
            # Back off to the nearest space so we do not split a token.
            cut = text.rfind(" ", pos + _MAX_CHARS // 2, end)
            if cut > pos:
                end = cut
        out.append((pos, text[pos:end]))
        pos = end
    return out
