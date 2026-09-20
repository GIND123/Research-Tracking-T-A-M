"""Experiment driver: baselines, ablations, and the report they produce.

Every row of every table in the paper is a :class:`Condition` here, so the
tables cannot drift from the code that generated them.  A condition is a name, a
:class:`~tekne.config.Config` and nothing else -- ablations are configuration
changes rather than code paths, which is the only way to keep an ablation honest
once a pipeline has this many interacting stages.

The baselines are deliberately *components of this system* rather than external
tools.  Running the gazetteer alone, or the chunker alone, answers the question
a reviewer actually has -- what does the machinery buy over its own parts -- and
avoids the usual apples-to-oranges comparison against a tool built for a
different ontology.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.orchestrator import Pipeline
from ..calib.selective import (
    RiskCoverageCurve,
    expected_calibration_error,
    risk_coverage,
)
from ..config import Config
from ..schema import Document, TechMention
from .gold import GoldDocument
from .metrics import EvalReport, evaluate, labelled_scores


@dataclass
class Condition:
    name: str
    config: Config
    description: str = ""


@dataclass
class ConditionResult:
    name: str
    description: str
    report: EvalReport
    curve: RiskCoverageCurve
    ece: float
    seconds: float
    ledger: dict[str, Any]
    config_digest: str
    per_doc: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "config_digest": self.config_digest,
            "seconds": round(self.seconds, 2),
            "metrics": self.report.as_dict(),
            "selective": {
                "aurc": round(self.curve.aurc, 5),
                "ece": round(self.ece, 5),
                "precision_at_80_coverage": _round(self.curve.precision_at_coverage(0.8)),
                "precision_at_50_coverage": _round(self.curve.precision_at_coverage(0.5)),
                "coverage_at_90_precision": round(self.curve.coverage_at_precision(0.9), 4),
            },
            "cost": self.ledger,
        }


def baseline_conditions(base: Config) -> list[Condition]:
    """Single-recaller baselines: what each component achieves alone.

    Each disables the guard stack down to grounding, because a baseline dressed
    in our guards is not a baseline; the point is to show the raw precision of
    a dictionary, a chunker and a terminology statistic on this task.
    """
    def strip(**over: Any) -> Config:
        cfg = base.with_overrides(
            **{
                "recall.use_patterns": False,
                "recall.use_abbrev": False,
                "recall.use_cvalue": False,
                "recall.use_gazetteer": False,
                "recall.use_llm": False,
                "guards.negative_lexicon": False,
                "guards.consensus": False,
                "guards.verifier": False,
                "guards.temporal": False,
                "verify.use_embedding": False,
                "verify.use_llm": False,
                "verify.adjudicate_disagreements": False,
                "output.abstain_below": 0.0,
                **over,
            }
        )
        return cfg

    return [
        Condition(
            "gazetteer",
            strip(**{"recall.use_gazetteer": True}),
            "CSO dictionary match only",
        ),
        Condition(
            "chunker",
            strip(**{"recall.use_patterns": True}),
            "noun-phrase chunker, open vocabulary",
        ),
        Condition(
            "chunker-gated",
            strip(**{"recall.use_patterns": True, "recall.require_tech_head": True}),
            "noun-phrase chunker gated on the technology head-noun lexicon",
        ),
        Condition(
            "cvalue",
            strip(**{"recall.use_cvalue": True}),
            "C-value / NC-value terminology extraction",
        ),
        Condition(
            "union",
            strip(
                **{
                    "recall.use_patterns": True,
                    "recall.use_abbrev": True,
                    "recall.use_cvalue": True,
                    "recall.use_gazetteer": True,
                }
            ),
            "union of all deterministic recallers, no guards or verification",
        ),
    ]


def ablation_conditions(base: Config) -> list[Condition]:
    """One knob off at a time, from the full configuration."""
    return [
        Condition(
            "full", base, "complete pipeline at the configured tier"
        ),
        Condition(
            "-negative",
            base.with_overrides(**{"guards.negative_lexicon": False}),
            "without the negative lexicon guard",
        ),
        Condition(
            "-consensus",
            base.with_overrides(**{"guards.consensus": False}),
            "without cross-recaller agreement",
        ),
        Condition(
            "-verifier",
            base.with_overrides(
                **{
                    "guards.verifier": False,
                    "verify.use_embedding": False,
                    "verify.use_llm": False,
                }
            ),
            "without adversarial verification",
        ),
        Condition(
            "-abstention",
            base.with_overrides(**{"output.abstain_below": 0.0}),
            "emit every surviving mention regardless of confidence",
        ),
        Condition(
            "-structure",
            base.with_overrides(**{"recall.use_gazetteer": False, "recall.use_abbrev": False}),
            "without the knowledge base and abbreviation miner",
        ),
    ]


def run_condition(
    condition: Condition,
    documents: Sequence[Document],
    gold: dict[str, GoldDocument],
) -> ConditionResult:
    started = time.monotonic()
    pipeline = Pipeline(condition.config)
    results = pipeline.run(list(documents))
    elapsed = time.monotonic() - started

    system: dict[str, list[TechMention]] = {r.doc_id: list(r.mentions) for r in results}
    doc_index = {d.doc_id: d for d in documents}
    report = evaluate(system, gold, doc_index)
    scores, labels = labelled_scores(system, gold)
    curve = risk_coverage(scores, labels)
    ece = expected_calibration_error(scores, labels)

    return ConditionResult(
        name=condition.name,
        description=condition.description,
        report=report,
        curve=curve,
        ece=ece,
        seconds=elapsed,
        ledger=pipeline.ledger.as_dict(),
        config_digest=condition.config.digest(),
        per_doc={doc_id: len(ms) for doc_id, ms in system.items()},
    )


def run_suite(
    conditions: Sequence[Condition],
    documents: Sequence[Document],
    gold: dict[str, GoldDocument],
    *,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> list[ConditionResult]:
    results: list[ConditionResult] = []
    for condition in conditions:
        if verbose:
            print(f"[{condition.name}] {condition.description}")
        result = run_condition(condition, documents, gold)
        results.append(result)
        if verbose:
            m = result.report
            print(
                f"    strict P/R/F1 {m.strict.precision:.3f}/{m.strict.recall:.3f}/{m.strict.f1:.3f}"
                f"  partial F1 {m.partial.f1:.3f}"
                f"  type {m.type_accuracy:.3f}  halluc {m.hallucination_rate:.4f}"
                f"  AURC {result.curve.aurc:.4f}  {result.seconds:.1f}s"
            )
    if out_dir:
        write_report(results, out_dir)
    return results


def write_report(results: Sequence[ConditionResult], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "conditions": [r.as_dict() for r in results],
    }
    path = out / "results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    curves = {
        r.name: r.curve.as_dict() for r in results
    }
    (out / "risk_coverage.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    (out / "results.tex").write_text(latex_table(results), encoding="utf-8")
    return path


def latex_table(results: Sequence[ConditionResult]) -> str:
    """Emit the main results table, ready to \\input into the paper."""
    lines = [
        "% generated by tekne.eval.runner -- do not edit by hand",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "System & P & R & F$_1$ & F$_1^{\\text{p}}$ & Type & Halluc. & AURC \\\\",
        "\\midrule",
    ]
    for result in results:
        m = result.report
        lines.append(
            f"{_escape(result.name)} & {m.strict.precision:.3f} & {m.strict.recall:.3f} & "
            f"{m.strict.f1:.3f} & {m.partial.f1:.3f} & {m.type_accuracy:.3f} & "
            f"{m.hallucination_rate:.3f} & {result.curve.aurc:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _escape(text: str) -> str:
    return text.replace("_", "\\_").replace("-", "--")


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
