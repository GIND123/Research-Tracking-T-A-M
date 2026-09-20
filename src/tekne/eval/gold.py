"""Gold annotation format and the annotation unit.

Annotation covers a *zone* of each document rather than the whole thing:

* papers -- title and abstract;
* patents -- title, abstract and the independent claims.

That is not a shortcut around the hard parts, it is where the technology content
is.  A patent specification runs to tens of thousands of characters of which the
overwhelming majority is boilerplate and embodiment enumeration; annotating it
exhaustively would spend the entire budget on text that contributes almost
nothing to a technology trend, and would make inter-annotator agreement
meaningless.  Restricting the zone keeps the sample honest: the system is scored
on exactly the span of text a human read.

The evaluation zone is stored on each gold record so that system output can be
filtered to the same region -- a system is not penalised for finding real
mentions outside the annotated zone, and gets no credit for them either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..schema import Document, Genre, Role, SectionKind, Span, TechType


@dataclass(frozen=True, slots=True)
class GoldMention:
    start: int
    end: int
    surface: str
    type: TechType
    role: Role = Role.UNKNOWN
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = {
            "start": self.start,
            "end": self.end,
            "surface": self.surface,
            "type": self.type.value,
            "role": self.role.value,
        }
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> GoldMention:
        return cls(
            start=int(row["start"]),
            end=int(row["end"]),
            surface=row["surface"],
            type=TechType(row.get("type", "artifact")),
            role=Role(row.get("role", "unknown")),
            note=row.get("note", ""),
        )


@dataclass
class GoldDocument:
    doc_id: str
    genre: Genre
    #: Half-open character ranges that were actually read and annotated.
    zones: list[tuple[int, int]]
    mentions: list[GoldMention] = field(default_factory=list)
    annotator: str = ""
    notes: str = ""

    def in_zone(self, start: int, end: int) -> bool:
        return any(lo <= start and end <= hi for lo, hi in self.zones)

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "genre": self.genre.value,
            "zones": [list(z) for z in self.zones],
            "annotator": self.annotator,
            "notes": self.notes,
            "mentions": [m.as_dict() for m in self.mentions],
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> GoldDocument:
        return cls(
            doc_id=row["doc_id"],
            genre=Genre(row["genre"]),
            zones=[tuple(z) for z in row.get("zones", [])],
            mentions=[GoldMention.from_dict(m) for m in row.get("mentions", [])],
            annotator=row.get("annotator", ""),
            notes=row.get("notes", ""),
        )


#: Sections that constitute the annotation zone, per genre.
ZONE_SECTIONS: dict[Genre, tuple[SectionKind, ...]] = {
    Genre.PAPER: (SectionKind.TITLE, SectionKind.ABSTRACT),
    Genre.PATENT: (SectionKind.TITLE, SectionKind.ABSTRACT, SectionKind.CLAIM_INDEP),
}


def evaluation_zones(doc: Document, *, max_claims: int = 1) -> list[tuple[int, int]]:
    """Character ranges to annotate and to score against, for one document."""
    wanted = ZONE_SECTIONS.get(doc.genre, (SectionKind.TITLE, SectionKind.ABSTRACT))
    zones: list[tuple[int, int]] = []
    claims_taken = 0
    for section in sorted(doc.sections, key=lambda s: s.span.start):
        if section.kind not in wanted:
            continue
        if section.kind is SectionKind.CLAIM_INDEP:
            if claims_taken >= max_claims:
                continue
            claims_taken += 1
        zones.append((section.span.start, section.span.end))
    if not zones:
        zones.append((0, min(len(doc.text), 2000)))
    return _merge_adjacent(zones)


def zone_text(doc: Document, zones: Sequence[tuple[int, int]]) -> str:
    return "\n\n".join(doc.text[lo:hi] for lo, hi in zones)


def _merge_adjacent(zones: list[tuple[int, int]]) -> list[tuple[int, int]]:
    zones = sorted(zones)
    merged: list[tuple[int, int]] = []
    for lo, hi in zones:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


# --- persistence -----------------------------------------------------------


def write_gold(records: Iterable[GoldDocument], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


def read_gold(path: str | Path) -> dict[str, GoldDocument]:
    out: dict[str, GoldDocument] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            record = GoldDocument.from_dict(json.loads(line))
            out[record.doc_id] = record
    return out


def validate_gold(records: dict[str, GoldDocument], documents: dict[str, Document]) -> list[str]:
    """Check that every gold span still matches the document it points into.

    Offsets are brittle across a re-fetch or a normalisation change, and a gold
    file that has silently drifted produces a confidently wrong evaluation. This
    runs in the test suite.
    """
    problems: list[str] = []
    for doc_id, record in records.items():
        doc = documents.get(doc_id)
        if doc is None:
            problems.append(f"{doc_id}: no such document")
            continue
        for mention in record.mentions:
            if mention.end > len(doc.text):
                problems.append(f"{doc_id}: span [{mention.start},{mention.end}) past end")
                continue
            actual = doc.text[mention.start : mention.end]
            if actual != mention.surface:
                problems.append(
                    f"{doc_id}: gold surface {mention.surface!r} != document {actual!r}"
                )
            if not record.in_zone(mention.start, mention.end):
                problems.append(f"{doc_id}: gold mention {mention.surface!r} outside zone")
    return problems


def spans_of(record: GoldDocument) -> list[Span]:
    return [
        Span(start=m.start, end=m.end, surface=m.surface) for m in record.mentions
    ]
