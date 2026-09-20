"""Genre-aware structural segmentation.

Both segmenters map native headings onto the closed :class:`SectionKind` set so
that every later stage can reason about zones without knowing which genre it is
looking at.  This is what lets one role classifier serve papers and patents.
"""

from __future__ import annotations

import re

from ..schema import Section, SectionKind, Span

# --- paper headings --------------------------------------------------------

_PAPER_HEADING_MAP: list[tuple[re.Pattern[str], SectionKind]] = [
    (re.compile(r"^\s*abstract\b", re.I), SectionKind.ABSTRACT),
    (re.compile(r"^\s*(?:\d+\.?\s*)?introduction\b", re.I), SectionKind.INTRODUCTION),
    (
        re.compile(r"^\s*(?:\d+\.?\s*)?(?:related work|background|prior art|literature)\b", re.I),
        SectionKind.RELATED_WORK,
    ),
    (
        re.compile(
            r"^\s*(?:\d+\.?\s*)?(?:method|methodology|approach|model|architecture|"
            r"proposed|our |system design|framework)\b",
            re.I,
        ),
        SectionKind.METHOD,
    ),
    (
        re.compile(
            r"^\s*(?:\d+\.?\s*)?(?:experiment|experimental setup|evaluation|setup|dataset)\b", re.I
        ),
        SectionKind.EXPERIMENT,
    ),
    (
        re.compile(r"^\s*(?:\d+\.?\s*)?(?:result|finding|analysis|discussion|ablation)\b", re.I),
        SectionKind.RESULTS,
    ),
    (
        re.compile(r"^\s*(?:\d+\.?\s*)?(?:conclusion|future work|summary)\b", re.I),
        SectionKind.CONCLUSION,
    ),
]

# A heading line: short, starts with an optional section number, no terminal period.
_HEADING_LINE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][^.!?]{2,60}$")


def segment_paper(text: str, *, title_len: int = 0, abstract_len: int = 0) -> list[Section]:
    """Segment a paper.

    ``title_len``/``abstract_len`` let callers that assembled the text from
    structured metadata (the arXiv API path) declare the leading zones exactly
    instead of re-detecting them.
    """
    sections: list[Section] = []
    cursor = 0
    ordinal = 0

    if title_len:
        sections.append(
            Section(
                kind=SectionKind.TITLE,
                span=_span(text, 0, title_len),
                ordinal=ordinal,
            )
        )
        ordinal += 1
        cursor = title_len
    if abstract_len:
        sections.append(
            Section(
                kind=SectionKind.ABSTRACT,
                span=_span(text, cursor, cursor + abstract_len),
                ordinal=ordinal,
            )
        )
        ordinal += 1
        cursor += abstract_len

    body = text[cursor:]
    if not body.strip():
        return _close(sections, len(text))

    # Heading detection over the remaining body. Normalisation has already
    # collapsed newlines, so we look for heading-like phrases at sentence starts.
    marks = _find_paper_headings(body, offset=cursor)
    if not marks:
        sections.append(
            Section(kind=SectionKind.OTHER, span=_span(text, cursor, len(text)), ordinal=ordinal)
        )
        return _close(sections, len(text))

    for i, (pos, kind, heading) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        if end <= pos:
            continue
        sections.append(
            Section(kind=kind, span=_span(text, pos, end), ordinal=ordinal, heading=heading)
        )
        ordinal += 1
    return _close(sections, len(text))


def _find_paper_headings(body: str, offset: int) -> list[tuple[int, SectionKind, str]]:
    marks: list[tuple[int, SectionKind, str]] = []
    # Candidate positions: start of body, or after a period followed by a capital.
    for m in re.finditer(r"(?:^|(?<=[.\s]))(\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Za-z ]{2,40})", body):
        frag = m.group(0).strip()
        for pattern, kind in _PAPER_HEADING_MAP:
            if pattern.match(frag):
                marks.append((offset + m.start(), kind, frag))
                break
    # Deduplicate by position, keep ascending order.
    seen: set[int] = set()
    out = []
    for pos, kind, head in sorted(marks):
        if pos in seen:
            continue
        seen.add(pos)
        out.append((pos, kind, head))
    return out


# --- patent structure ------------------------------------------------------

_PATENT_HEADING_MAP: list[tuple[re.Pattern[str], SectionKind]] = [
    (
        re.compile(
            r"^(?:technical\s+)?field(?:\s+of\s+(?:the\s+)?(?:invention|disclosure|technology))?$",
            re.I,
        ),
        SectionKind.PATENT_FIELD,
    ),
    (
        re.compile(r"^(?:background|description of (?:the )?(?:related|prior) art|prior art)", re.I),
        SectionKind.PATENT_BACKGROUND,
    ),
    (
        re.compile(r"^(?:brief\s+)?summary(?:\s+of\s+(?:the\s+)?invention)?", re.I),
        SectionKind.PATENT_SUMMARY,
    ),
    (
        re.compile(r"^(?:detailed\s+description|description of (?:the )?(?:preferred|embodiment))", re.I),
        SectionKind.PATENT_DETAIL,
    ),
]

# Claim openers: "1. A method ..." / "2 . The system of claim 1 ...".
_CLAIM_START = re.compile(r"(?:^|\s)(\d{1,3})\s*\.\s+(?=[A-Z(])")
_CLAIM_DEPENDENCY = re.compile(
    r"\b(?:of|in|according\s+to|as\s+(?:claimed|recited|set\s+forth)\s+in)\s+"
    r"(?:any\s+(?:one\s+)?of\s+)?claims?\s+(\d{1,3})",
    re.I,
)


def segment_patent(
    text: str,
    *,
    title_len: int = 0,
    abstract_len: int = 0,
    claims_range: tuple[int, int] | None = None,
    description_range: tuple[int, int] | None = None,
) -> list[Section]:
    """Segment a patent document.

    The caller supplies the coarse field boundaries (they come for free from the
    source markup); this function resolves the two structures that actually
    require parsing: individual claims with their dependency status, and the
    conventional headings inside the description.
    """
    sections: list[Section] = []
    ordinal = 0
    cursor = 0

    if title_len:
        sections.append(Section(kind=SectionKind.TITLE, span=_span(text, 0, title_len), ordinal=0))
        ordinal += 1
        cursor = title_len
    if abstract_len:
        sections.append(
            Section(
                kind=SectionKind.ABSTRACT,
                span=_span(text, cursor, cursor + abstract_len),
                ordinal=ordinal,
            )
        )
        ordinal += 1

    if claims_range:
        for sec in _segment_claims(text, *claims_range, start_ordinal=ordinal):
            sections.append(sec)
            ordinal += 1

    if description_range:
        for sec in _segment_description(text, *description_range, start_ordinal=ordinal):
            sections.append(sec)
            ordinal += 1

    if not sections:
        sections.append(Section(kind=SectionKind.OTHER, span=_span(text, 0, len(text)), ordinal=0))
    return sorted(sections, key=lambda s: s.span.start)


def _segment_claims(text: str, lo: int, hi: int, start_ordinal: int) -> list[Section]:
    body = text[lo:hi]
    starts = [(lo + m.start(1), int(m.group(1))) for m in _CLAIM_START.finditer(body)]
    if not starts:
        return [
            Section(kind=SectionKind.CLAIM_INDEP, span=_span(text, lo, hi), ordinal=start_ordinal)
        ]

    out: list[Section] = []
    for i, (pos, number) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else hi
        if end <= pos:
            continue
        claim_text = text[pos:end]
        dep = _CLAIM_DEPENDENCY.search(claim_text)
        kind = SectionKind.CLAIM_DEP if dep else SectionKind.CLAIM_INDEP
        attrs = {"claim_number": number}
        if dep:
            attrs["depends_on"] = int(dep.group(1))
        out.append(
            Section(
                kind=kind,
                span=_span(text, pos, end),
                ordinal=start_ordinal + i,
                attrs=attrs,
            )
        )
    return out


def _segment_description(text: str, lo: int, hi: int, start_ordinal: int) -> list[Section]:
    body = text[lo:hi]
    marks: list[tuple[int, SectionKind]] = []
    # Patent headings are conventionally upper case; after whitespace collapse
    # they appear as runs of capitals mid-string.
    for m in re.finditer(r"(?:^|\s)((?:[A-Z][A-Z()/'-]*\s+){0,6}[A-Z][A-Z()/'-]*)(?=\s|$)", body):
        frag = m.group(1).strip()
        if len(frag) < 5 or len(frag.split()) > 7:
            continue
        for pattern, kind in _PATENT_HEADING_MAP:
            if pattern.match(frag):
                marks.append((lo + m.start(1), kind))
                break

    if not marks:
        return [
            Section(kind=SectionKind.PATENT_DETAIL, span=_span(text, lo, hi), ordinal=start_ordinal)
        ]

    out: list[Section] = []
    if marks[0][0] > lo:
        out.append(
            Section(kind=SectionKind.PATENT_DETAIL, span=_span(text, lo, marks[0][0]), ordinal=start_ordinal)
        )
    for i, (pos, kind) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else hi
        if end <= pos:
            continue
        out.append(Section(kind=kind, span=_span(text, pos, end), ordinal=start_ordinal + i + 1))
    return out


def _span(text: str, start: int, end: int) -> Span:
    start = max(0, min(start, len(text)))
    end = max(start + 1, min(end, len(text)))
    return Span(start=start, end=end, surface=text[start:end])


def _close(sections: list[Section], total_len: int) -> list[Section]:
    return [s for s in sections if s.span.end <= total_len]
