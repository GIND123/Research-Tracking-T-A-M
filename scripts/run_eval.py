#!/usr/bin/env python3
"""Run the evaluation suite and write the report the paper reads from.

    python scripts/run_eval.py --suite baselines
    python scripts/run_eval.py --suite ablations --config configs/default.yaml
    python scripts/run_eval.py --suite all --out runs/eval

With no API key present the LLM tiers are disabled automatically and the run is
labelled accordingly, so an offline reproduction produces the deterministic and
neural rows of every table without further flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tekne.config import Config  # noqa: E402
from tekne.eval.gold import read_gold  # noqa: E402
from tekne.eval.runner import (  # noqa: E402
    ablation_conditions,
    baseline_conditions,
    run_suite,
    write_report,
)
from tekne.ingest.sources import read_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--gold", default="data/gold/gold.jsonl")
    parser.add_argument(
        "--corpora",
        nargs="*",
        default=["data/raw/papers_eval.jsonl", "data/raw/patents_eval.jsonl"],
    )
    parser.add_argument("--suite", choices=["baselines", "ablations", "all"], default="all")
    parser.add_argument("--out", default="runs/eval")
    parser.add_argument("--no-llm", action="store_true", help="force the offline tier")
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.no_llm or not config.has_api_key():
        config = config.with_overrides(
            **{
                "recall.use_llm": False,
                "verify.use_llm": False,
                "verify.adjudicate_disagreements": False,
                "label": "offline",
            }
        )
        print("running without the LLM tier (no API key set or --no-llm given)")
    else:
        config = config.with_overrides(**{"verify.use_llm": True, "label": "full"})
        print(f"running with models: {config.models}")

    gold = read_gold(args.gold)
    documents = [d for path in args.corpora for d in read_jsonl(path)]
    documents = [d for d in documents if d.doc_id in gold]
    print(f"{len(documents)} documents, {sum(len(g.mentions) for g in gold.values())} gold mentions")

    conditions = []
    if args.suite in ("baselines", "all"):
        conditions += baseline_conditions(config)
    if args.suite in ("ablations", "all"):
        conditions += ablation_conditions(config)

    results = run_suite(conditions, documents, gold, out_dir=args.out)
    write_report(results, args.out)
    print(f"\nwrote {args.out}/results.json and results.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
