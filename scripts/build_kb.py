#!/usr/bin/env python3
"""Compile the Computer Science Ontology into the gazetteer TEKNE loads.

    python scripts/build_kb.py --out data/kb/cso.json

Downloads the CSO triple dump on first run and caches it under ``data/kb``.
Surface forms shorter than three characters and those that collide with the
negative lexicon are dropped here rather than at match time, because the
automaton is built once and queried millions of times.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tekne.kb import add_tsv_gazetteer, build_from_cso  # noqa: E402
from tekne.lexicons import lemma_key, negative_lexicon  # noqa: E402

CSO_URL = "https://cso.kmi.open.ac.uk/download/version-3.4/CSO.3.4.csv.zip"


def fetch_cso(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "CSO.3.4.csv"
    if csv_path.is_file():
        return csv_path

    import httpx

    zip_path = cache_dir / "CSO.3.4.csv.zip"
    print(f"downloading {CSO_URL}")
    with httpx.stream("GET", CSO_URL, follow_redirects=True, timeout=180.0) as response:
        response.raise_for_status()
        with zip_path.open("wb") as fh:
            for chunk in response.iter_bytes(65536):
                fh.write(chunk)
    with ZipFile(zip_path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        with archive.open(name) as src, csv_path.open("wb") as dst:
            dst.write(src.read())
    zip_path.unlink(missing_ok=True)
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/kb/cso.json")
    parser.add_argument("--cache-dir", default="data/kb")
    parser.add_argument("--csv", default=None, help="use a local CSO dump instead of downloading")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KB:PATH",
        help="merge a TSV gazetteer, e.g. cpc:data/kb/cpc_titles.tsv",
    )
    parser.add_argument("--min-chars", type=int, default=3)
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else fetch_cso(Path(args.cache_dir))
    print(f"parsing {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
    gaz = build_from_cso(csv_path)
    print(f"  {len(gaz)} entries, {len(gaz.surface_index)} surface forms")

    for spec in args.extra:
        kb_name, _, path = spec.partition(":")
        add_tsv_gazetteer(gaz, Path(path), kb=kb_name)
        print(f"  merged {kb_name} from {path}")

    negative = negative_lexicon()
    before = len(gaz.surface_index)
    gaz.surface_index = {
        key: entry
        for key, entry in gaz.surface_index.items()
        if len(key) >= args.min_chars and lemma_key(key) not in negative
    }
    print(f"  pruned {before - len(gaz.surface_index)} surface forms (short or in negative lexicon)")

    gaz.build_automaton()
    out = Path(args.out)
    gaz.save(out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
