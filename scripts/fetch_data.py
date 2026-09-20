#!/usr/bin/env python3
"""Assemble the demonstration corpora.

    python scripts/fetch_data.py eval     # the annotated evaluation sample
    python scripts/fetch_data.py trend    # the larger time-sliced paper corpus
    python scripts/fetch_data.py patents  # patent full text via the HUPD frame

Sampling is deliberately spread across fields rather than concentrated in NLP:
the claim under test is that one pipeline handles both genres and several
domains, and a corpus drawn only from cs.CL would not test it.  Everything is
written to ``data/raw`` as JSONL and is safe to re-run -- existing documents are
kept unless ``--refresh`` is given.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tekne.ingest.sources import (  # noqa: E402
    ArxivThrottled,
    arxiv_to_document,
    fetch_arxiv,
    fetch_google_patent,
    load_hupd_frame,
    patent_to_document,
    read_jsonl,
    write_jsonl,
)

# Six fields, chosen so the evaluation covers vocabularies that share almost no
# surface forms: transformer-era NLP, classical ML, signal processing, materials,
# energy and bio. A pipeline tuned to one of these will visibly fail on others.
EVAL_QUERIES = [
    ("cat:cs.CL", 6),
    ("cat:cs.LG", 5),
    ("cat:eess.SP", 5),
    ("cat:cond-mat.mtrl-sci", 5),
    ("cat:physics.app-ph", 4),
    ("cat:q-bio.QM", 3),
]

# For the evolution demonstration we need the same field over many years.
TREND_QUERY = "cat:cs.CL OR cat:cs.LG OR cat:cs.CV"


def cmd_eval(args: argparse.Namespace) -> int:
    out = Path(args.out) / "papers_eval.jsonl"
    if out.is_file() and not args.refresh:
        print(f"{out} exists ({len(read_jsonl(out))} docs); pass --refresh to rebuild")
        return 0

    docs = []
    for query, n in EVAL_QUERIES:
        print(f"arxiv: {query} (n={n})")
        try:
            records = fetch_arxiv(query, max_results=n * 3, delay=args.delay)
        except ArxivThrottled as exc:
            # One throttled field should not cost the whole corpus; the sample
            # is reported per field, so a short row is visible rather than silent.
            print(f"  SKIPPED: {exc}")
            continue
        # Skip very short abstracts: they carry too few mentions to be worth an
        # annotation slot.
        usable = [r for r in records if len(r["abstract"]) > 600][:n]
        docs.extend(arxiv_to_document(r) for r in usable)
        print(f"  kept {len(usable)}")

    write_jsonl(docs, out)
    print(f"wrote {len(docs)} papers -> {out}")
    return 0


def cmd_trend(args: argparse.Namespace) -> int:
    out = Path(args.out) / "papers_trend.jsonl"
    existing = {d.doc_id for d in read_jsonl(out)} if out.is_file() else set()
    docs = read_jsonl(out) if out.is_file() and not args.refresh else []

    for year in range(args.start_year, args.end_year + 1):
        query = f"({TREND_QUERY}) AND submittedDate:[{year}01010000 TO {year}12312359]"
        print(f"arxiv {year}: requesting {args.per_year}", flush=True)
        fetched = 0
        for offset in range(0, args.per_year, 100):
            try:
                batch = fetch_arxiv(
                    query,
                    max_results=min(100, args.per_year - offset),
                    start=offset,
                    delay=args.delay,
                )
            except ArxivThrottled as exc:
                print(f"  SKIPPED offset {offset}: {exc}")
                break
            if not batch:
                break
            for record in batch:
                doc = arxiv_to_document(record)
                if doc.doc_id in existing or len(record["abstract"]) < 400:
                    continue
                existing.add(doc.doc_id)
                docs.append(doc)
                fetched += 1
        # Write after every year: a throttled run should keep what it fetched
        # rather than losing an hour of polite waiting to one 406.
        write_jsonl(docs, out)
        print(f"  +{fetched} (total {len(docs)})", flush=True)

    write_jsonl(docs, out)
    print(f"wrote {len(docs)} papers -> {out}")
    return 0


def cmd_patents(args: argparse.Namespace) -> int:
    out = Path(args.out) / "patents_eval.jsonl"
    if out.is_file() and not args.refresh:
        print(f"{out} exists ({len(read_jsonl(out))} docs); pass --refresh to rebuild")
        return 0

    frame_path = Path(args.frame)
    if not frame_path.is_file():
        print(f"missing HUPD frame at {frame_path}", file=sys.stderr)
        print("download: https://huggingface.co/datasets/HUPD/hupd/resolve/main/"
              "hupd_metadata_jan16_2022-02-22.feather", file=sys.stderr)
        return 1

    rows = load_hupd_frame(frame_path)
    print(f"frame: {len(rows)} applications")

    # Stratify by CPC section so the sample is not dominated by one technology
    # area. Sections: A human necessities, B operations, C chemistry, D textiles,
    # E construction, F engineering, G physics, H electricity.
    by_section: dict[str, list[dict]] = {}
    for row in rows:
        label = (row.get("main_cpc_label") or "").strip()
        if not label:
            continue
        by_section.setdefault(label[0].upper(), []).append(row)

    rng = random.Random(args.seed)
    wanted = [s for s in args.sections if s in by_section]
    per_section = max(1, args.n // max(len(wanted), 1))
    selected: list[dict] = []
    for section in wanted:
        pool = by_section[section]
        rng.shuffle(pool)
        selected.extend(pool[:per_section])
    print(f"selected {len(selected)} across sections {wanted}")

    docs = []
    failures = 0
    for i, row in enumerate(selected, start=1):
        pub = str(row.get("earliest_pgpub_number") or "").strip()
        if not pub:
            continue
        record = fetch_google_patent(pub, delay=args.delay)
        if record is None:
            failures += 1
            print(f"  [{i}/{len(selected)}] {pub}: no full text")
            continue
        cpc = [c.strip() for c in str(row.get("cpc_labels") or "").split(",") if c.strip()]
        filing = str(row.get("filing_date") or "")[:10] or None
        docs.append(patent_to_document(record, filing_date=filing, cpc=cpc))
        print(f"  [{i}/{len(selected)}] {pub}: {len(record.get('claims') or '')} claim chars")
        if len(docs) >= args.n:
            break

    write_jsonl(docs, out)
    print(f"wrote {len(docs)} patents ({failures} unavailable) -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--refresh", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("eval").set_defaults(func=cmd_eval)

    trend = sub.add_parser("trend")
    trend.add_argument("--start-year", type=int, default=2015)
    trend.add_argument("--end-year", type=int, default=2025)
    trend.add_argument("--per-year", type=int, default=60)
    trend.set_defaults(func=cmd_trend)

    patents = sub.add_parser("patents")
    patents.add_argument("--frame", default="data/raw/hupd_metadata_jan16.feather")
    patents.add_argument("--n", type=int, default=24)
    patents.add_argument(
        "--sections",
        default="ACGHB",
        help="CPC section letters to stratify over, e.g. ACGHB",
    )
    patents.add_argument("--seed", type=int, default=20260101)
    patents.set_defaults(func=cmd_patents, delay=1.5)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
