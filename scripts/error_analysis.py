#!/usr/bin/env python3
"""Where gold mentions are lost, stage by stage.

    python scripts/error_analysis.py --limit 20

Span F1 says how much is lost; this says where. Each gold mention in a scored
type is attributed to exactly one outcome:

  emitted            found, with the right offsets and a scored type
  wrong_type         found, but typed outside the scored set
  withheld           found, but below the confidence threshold
  rejected           removed by a guard, with the guard named
  lost_at_decode     proposed, but an overlapping span was selected instead
  never_proposed     no recaller produced a span overlapping it

The distinction that matters for reading the results table is between
`lost_at_decode` (boundary disagreement -- partial F1 still credits it) and
`never_proposed` (a genuine recall failure).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tekne.agents.orchestrator import Pipeline  # noqa: E402
from tekne.config import Config  # noqa: E402
from tekne.eval.gold import read_gold  # noqa: E402
from tekne.eval.metrics import DEFAULT_SCORED_TYPES  # noqa: E402
from tekne.ingest.sources import read_jsonl  # noqa: E402
from tekne.nlp import analyse  # noqa: E402
from tekne.recall.base import RecallContext, merge_candidates  # noqa: E402


def recall_stage_spans(pipeline: Pipeline, doc, analysis) -> set[tuple[int, int]]:
    """Every span the deterministic recallers proposed, before any filtering."""
    from tekne.recall.abbrev import AbbreviationRecaller
    from tekne.recall.cvalue import TermStatRecaller
    from tekne.recall.gazetteer import GazetteerRecaller
    from tekne.recall.patterns import PatternRecaller

    ctx = RecallContext(
        document=doc,
        analysis=analysis,
        corpus_stats=pipeline.corpus_stats,
        gazetteer=pipeline.gazetteer,
    )
    recallers = [
        PatternRecaller(max_chunk_tokens=pipeline.config.recall.max_chunk_tokens),
        AbbreviationRecaller(),
        TermStatRecaller(threshold=pipeline.config.recall.cvalue_threshold),
        GazetteerRecaller(),
    ]
    merged = merge_candidates([r.propose(ctx) for r in recallers])
    return {(c.span.start, c.span.end) for c in merged}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/gold/gold.jsonl")
    parser.add_argument(
        "--corpora",
        nargs="*",
        default=["data/raw/papers_eval.jsonl", "data/raw/patents_eval.jsonl"],
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="runs/eval/error_analysis.json")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    gold = read_gold(args.gold)
    documents = [d for path in args.corpora for d in read_jsonl(path) if d.doc_id in gold]
    if args.limit:
        documents = documents[: args.limit]

    config = Config.load(args.config).with_overrides(
        **{
            "recall.use_llm": False,
            "verify.use_llm": False,
            "verify.adjudicate_disagreements": False,
        }
    )
    pipeline = Pipeline(config)
    analyses = {d.doc_id: analyse(d) for d in documents}

    outcomes: Counter[str] = Counter()
    by_guard: Counter[str] = Counter()
    type_errors: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    for doc in documents:
        record = gold[doc.doc_id]
        targets = [m for m in record.mentions if m.type in DEFAULT_SCORED_TYPES]
        if not targets:
            continue

        proposed = recall_stage_spans(pipeline, doc, analyses[doc.doc_id])
        result = pipeline.extract(doc, analysis=analyses[doc.doc_id])
        emitted = {(m.span.start, m.span.end): m for m in result.mentions}
        withheld = {(r["start"], r["end"]) for r in result.abstained}
        rejected = {(r["start"], r["end"]): r for r in result.rejected}

        for mention in targets:
            key = (mention.start, mention.end)
            if key in emitted:
                found = emitted[key]
                if found.type in DEFAULT_SCORED_TYPES:
                    outcomes["emitted"] += 1
                else:
                    outcomes["wrong_type"] += 1
                    type_errors[f"{mention.type.value}->{found.type.value}"] += 1
                    _example(examples, "wrong_type", mention.surface, args.examples)
            elif key in withheld:
                outcomes["withheld"] += 1
                _example(examples, "withheld", mention.surface, args.examples)
            elif key in rejected:
                outcomes["rejected"] += 1
                by_guard[rejected[key]["reason"].split(":")[0]] += 1
                _example(examples, "rejected", mention.surface, args.examples)
            elif key in proposed:
                outcomes["lost_at_decode"] += 1
                _example(examples, "lost_at_decode", mention.surface, args.examples)
            else:
                outcomes["never_proposed"] += 1
                _example(examples, "never_proposed", mention.surface, args.examples)

    total = sum(outcomes.values()) or 1
    payload = {
        "documents": len(documents),
        "gold_scored": total,
        "outcomes": {k: {"n": v, "share": round(v / total, 4)} for k, v in outcomes.most_common()},
        "rejected_by_guard": dict(by_guard.most_common()),
        "type_confusions": dict(type_errors.most_common(10)),
        "examples": examples,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{len(documents)} documents, {total} scored gold mentions")
    for name, row in payload["outcomes"].items():
        print(f"  {name:18} {row['n']:5}  {row['share']:6.1%}")
    if by_guard:
        print("  rejected by:", dict(by_guard.most_common(5)))
    if type_errors:
        print("  type errors:", dict(type_errors.most_common(5)))
    print(f"wrote {out}")
    return 0


def _example(store: dict[str, list[str]], bucket: str, surface: str, cap: int) -> None:
    rows = store.setdefault(bucket, [])
    if len(rows) < cap and surface not in rows:
        rows.append(surface)


if __name__ == "__main__":
    raise SystemExit(main())
