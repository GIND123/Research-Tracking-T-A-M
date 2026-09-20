#!/usr/bin/env python3
"""Turn the annotation table into offset-anchored gold records.

    python scripts/make_gold.py --check      # validate without writing
    python scripts/make_gold.py              # write data/gold/gold.jsonl

Annotators work in ``data/gold/annotations.tsv``, which lists one row per
*technology per document*:

    doc_id <TAB> surface <TAB> type <TAB> role [<TAB> note]

not one row per occurrence.  This script expands each row to every occurrence of
the surface inside that document's evaluation zone, matching case-insensitively
on token boundaries and taking the span text from the document rather than from
the annotation.  Two consequences worth stating plainly:

* annotating a term is a claim about the whole document, which is the level at
  which a human can actually be consistent -- deciding per-occurrence whether the
  fourteenth "grinding tool" is a technology is not a judgement anyone makes
  reliably;
* a shorter term nested inside a longer annotated one is suppressed at that
  position, so annotating both "carbon nanotube" and "carbon nanotube dispersion
  liquid" yields the longer span where they overlap and the shorter one
  everywhere else.

Rows whose surface does not occur in the zone are reported as errors rather than
skipped: they almost always mean a typo or a stale offset, and silently dropping
them would quietly shrink the gold set.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tekne.eval.gold import (  # noqa: E402
    GoldDocument,
    GoldMention,
    evaluation_zones,
    validate_gold,
    write_gold,
)
from tekne.ingest.sources import read_jsonl  # noqa: E402
from tekne.schema import Document, Role, TechType  # noqa: E402

CORPORA = ["data/raw/papers_eval.jsonl", "data/raw/patents_eval.jsonl"]


def load_documents(paths: list[str]) -> dict[str, Document]:
    docs: dict[str, Document] = {}
    for path in paths:
        if not Path(path).is_file():
            print(f"warning: {path} not found, skipping", file=sys.stderr)
            continue
        for doc in read_jsonl(path):
            docs[doc.doc_id] = doc
    return docs


def read_annotations(path: Path) -> dict[str, list[tuple[str, str, str, str]]]:
    rows: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for lineno, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#") or not row[0].strip():
                continue
            if len(row) < 4:
                raise SystemExit(f"{path}:{lineno}: expected 4+ columns, got {len(row)}")
            doc_id, surface, type_name, role_name = (c.strip() for c in row[:4])
            note = row[4].strip() if len(row) > 4 else ""
            rows[doc_id].append((surface, type_name, role_name, note))
    return rows


def expand(
    doc: Document, annotations: list[tuple[str, str, str, str]], zones: list[tuple[int, int]]
) -> tuple[list[GoldMention], list[str]]:
    problems: list[str] = []
    # Longest surfaces first so nested terms are suppressed where they overlap.
    ordered = sorted(annotations, key=lambda a: -len(a[0]))
    taken: list[tuple[int, int]] = []
    mentions: list[GoldMention] = []

    for surface, type_name, role_name, note in ordered:
        try:
            tech_type = TechType(type_name)
            role = Role(role_name)
        except ValueError as exc:
            problems.append(f"{doc.doc_id}: {surface!r}: {exc}")
            continue

        hits = 0
        pattern = re.compile(_boundary_pattern(surface), re.IGNORECASE)
        for lo, hi in zones:
            for match in pattern.finditer(doc.text, lo, hi):
                start, end = match.start(), match.end()
                if any(start < t_end and t_start < end for t_start, t_end in taken):
                    continue
                taken.append((start, end))
                mentions.append(
                    GoldMention(
                        start=start,
                        end=end,
                        surface=doc.text[start:end],
                        type=tech_type,
                        role=role,
                        note=note,
                    )
                )
                hits += 1
        if hits == 0:
            problems.append(f"{doc.doc_id}: {surface!r} does not occur in the evaluation zone")

    mentions.sort(key=lambda m: (m.start, m.end))
    return mentions, problems


def _boundary_pattern(surface: str) -> str:
    """Whitespace-tolerant, token-bounded literal match."""
    parts = [re.escape(tok) for tok in surface.split()]
    body = r"\s+".join(parts)
    left = r"(?<![A-Za-z0-9])" if surface[:1].isalnum() else ""
    right = r"(?![A-Za-z0-9])" if surface[-1:].isalnum() else ""
    return left + body + right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/gold/annotations.tsv")
    parser.add_argument("--out", default="data/gold/gold.jsonl")
    parser.add_argument("--corpora", nargs="*", default=CORPORA)
    parser.add_argument("--annotator", default="a1")
    parser.add_argument("--max-claim-chars", type=int, default=1400)
    parser.add_argument("--check", action="store_true", help="validate only, do not write")
    args = parser.parse_args()

    documents = load_documents(args.corpora)
    annotations = read_annotations(Path(args.annotations))
    print(f"{len(documents)} documents, {len(annotations)} annotated")

    records: list[GoldDocument] = []
    all_problems: list[str] = []
    for doc_id, rows in annotations.items():
        doc = documents.get(doc_id)
        if doc is None:
            all_problems.append(f"{doc_id}: no such document in the corpora")
            continue
        zones = _capped_zones(doc, args.max_claim_chars)
        mentions, problems = expand(doc, rows, zones)
        all_problems.extend(problems)
        records.append(
            GoldDocument(
                doc_id=doc_id,
                genre=doc.genre,
                zones=zones,
                mentions=mentions,
                annotator=args.annotator,
            )
        )

    by_id = {r.doc_id: r for r in records}
    all_problems.extend(validate_gold(by_id, documents))

    total = sum(len(r.mentions) for r in records)
    per_type: dict[str, int] = defaultdict(int)
    per_role: dict[str, int] = defaultdict(int)
    for record in records:
        for mention in record.mentions:
            per_type[mention.type.value] += 1
            per_role[mention.role.value] += 1

    print(f"{total} gold mentions over {len(records)} documents")
    print("  by type:", dict(sorted(per_type.items(), key=lambda kv: -kv[1])))
    print("  by role:", dict(sorted(per_role.items(), key=lambda kv: -kv[1])))

    if all_problems:
        print(f"\n{len(all_problems)} problems:", file=sys.stderr)
        for problem in all_problems[:40]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if not args.check:
        write_gold(records, args.out)
        print(f"wrote {args.out}")
    return 0


def _capped_zones(doc: Document, max_claim_chars: int) -> list[tuple[int, int]]:
    """Evaluation zones with the claim truncated.

    Independent claims in chemical cases are Markush structures running to
    thousands of characters of variable definitions ("RA2 is selected from the
    group consisting of ..."). That text names no technology and annotating it
    exhaustively would swamp the sample, so the claim zone is capped. The cap is
    recorded in the zone, so system output is filtered to the same region.
    """
    from tekne.schema import SectionKind

    zones = []
    for lo, hi in evaluation_zones(doc):
        section = doc.section_at(lo)
        if section and section.kind is SectionKind.CLAIM_INDEP:
            hi = min(hi, lo + max_claim_chars)
        zones.append((lo, hi))
    return zones


if __name__ == "__main__":
    raise SystemExit(main())
