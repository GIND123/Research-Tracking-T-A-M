"""Command line interface.

    tekne extract data/raw/papers_eval.jsonl --out runs/extract
    tekne inspect arxiv:2609.20800v1 --corpus data/raw/papers_eval.jsonl
    tekne plan data/raw/patents_eval.jsonl        # cost projection, no API calls
    tekne trends runs/extract/mentions.jsonl
    tekne evaluate --gold data/gold/gold.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Config
from .ingest.sources import read_jsonl
from .schema import ExtractionResult

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@app.command()
def extract(
    corpus: list[Path] = typer.Argument(..., help="JSONL document files"),
    out: Path = typer.Option(Path("runs/extract"), help="output directory"),
    config: Path | None = typer.Option(None, help="YAML config"),
    limit: int = typer.Option(0, help="process only the first N documents"),
    no_llm: bool = typer.Option(False, "--no-llm", help="force the offline tier"),
    trace: bool = typer.Option(False, help="write the full per-document trace"),
) -> None:
    """Run the pipeline over one or more corpora."""
    from .agents.orchestrator import Pipeline

    cfg = _resolve_config(config, no_llm)
    documents = [d for path in corpus for d in read_jsonl(path)]
    if limit:
        documents = documents[:limit]
    console.print(
        f"[bold]{len(documents)}[/bold] documents, config [cyan]{cfg.digest()}[/cyan], "
        f"backend {'anthropic' if cfg.has_api_key() and not no_llm else 'offline'}"
    )

    pipeline = Pipeline(cfg)
    with console.status("extracting..."):
        results = pipeline.run(documents)

    out.mkdir(parents=True, exist_ok=True)
    _write_results(results, out, include_trace=trace)

    total = sum(len(r.mentions) for r in results)
    withheld = sum(len(r.abstained) for r in results)
    rejected = sum(len(r.rejected) for r in results)
    console.print(
        f"[green]{total}[/green] mentions emitted, "
        f"[yellow]{withheld}[/yellow] withheld for review, "
        f"[dim]{rejected}[/dim] rejected"
    )
    console.print(f"cost: {json.dumps(pipeline.ledger.as_dict()['per_model'])}")
    console.print(f"wrote {out}/mentions.jsonl")


@app.command()
def inspect(
    doc_id: str = typer.Argument(..., help="document id"),
    corpus: list[Path] = typer.Option(..., help="JSONL document files"),
    config: Path | None = typer.Option(None),
    no_llm: bool = typer.Option(False, "--no-llm"),
    show_rejected: bool = typer.Option(False, help="also list rejected candidates"),
) -> None:
    """Extract one document and show every decision the pipeline made."""
    from .agents.orchestrator import Pipeline

    cfg = _resolve_config(config, no_llm)
    documents = {d.doc_id: d for path in corpus for d in read_jsonl(path)}
    doc = documents.get(doc_id)
    if doc is None:
        console.print(f"[red]no document {doc_id}[/red] in {len(documents)} loaded")
        raise typer.Exit(1)

    result = Pipeline(cfg).extract(doc)

    console.print(f"\n[bold]{doc.metadata.title or doc.doc_id}[/bold]")
    console.print(f"[dim]{doc.genre.value} | {doc.metadata.date} | {len(doc.text)} chars[/dim]\n")

    table = Table(show_header=True, header_style="bold")
    for column in ("conf", "surface", "type", "role", "section", "proposers"):
        table.add_column(column)
    for mention in sorted(result.mentions, key=lambda m: -m.confidence):
        table.add_row(
            f"{mention.confidence:.3f}",
            mention.span.surface[:44],
            mention.type.value,
            mention.role.value,
            mention.section.value,
            ",".join(mention.provenance.proposers),
        )
    console.print(table)

    if result.abstained:
        console.print(f"\n[yellow]withheld for review ({len(result.abstained)})[/yellow]")
        for row in sorted(result.abstained, key=lambda r: -r["confidence"])[:12]:
            console.print(f"  {row['confidence']:.3f}  {row['surface'][:44]:46} {row['reason'][:50]}")

    if show_rejected and result.rejected:
        console.print(f"\n[dim]rejected ({len(result.rejected)})[/dim]")
        for row in result.rejected[:20]:
            console.print(f"  {row['surface'][:44]:46} {row['reason'][:60]}")

    console.print("\n[bold]trace[/bold]")
    for step in result.trace:
        stage = step.pop("stage")
        console.print(f"  {stage:26} {step}")


@app.command()
def plan(
    corpus: list[Path] = typer.Argument(...),
    config: Path | None = typer.Option(None),
    exact_tokens: bool = typer.Option(False, help="count tokens via the API instead of estimating"),
) -> None:
    """Project what a run would cost, without issuing a single model call."""
    from .llm.planner import plan_run

    cfg = Config.load(config)
    documents = [d for path in corpus for d in read_jsonl(path)]
    backend = None
    if exact_tokens and cfg.has_api_key():
        from .llm.anthropic_backend import AnthropicBackend

        backend = AnthropicBackend()

    projection = plan_run(documents, cfg, backend=backend, exact_tokens=exact_tokens)
    payload = projection.as_dict()

    table = Table(title=f"projected cost for {payload['documents']} documents")
    for column in ("stage", "model", "calls", "input", "cached in", "output", "usd"):
        table.add_column(column, justify="right" if column != "stage" else "left")
    for stage in payload["stages"]:
        table.add_row(
            stage["stage"],
            stage["model"],
            str(stage["calls"]),
            f"{stage['input_tokens']:,}",
            f"{stage['cached_input_tokens']:,}",
            f"{stage['output_tokens']:,}",
            f"${stage['usd']:.4f}",
        )
    console.print(table)
    console.print(
        f"[bold]total ${payload['total_usd']:.4f}[/bold] "
        f"({payload['usd_per_document']:.5f}/doc, {payload['calls_per_document']} calls/doc)"
    )
    for note in payload["notes"]:
        console.print(f"[dim]note: {note}[/dim]")


@app.command()
def trends(
    mentions: Path = typer.Argument(..., help="mentions.jsonl from `tekne extract`"),
    corpus: list[Path] = typer.Option(..., help="the documents those mentions came from"),
    top: int = typer.Option(20),
    min_confidence: float = typer.Option(0.5),
) -> None:
    """Aggregate extracted mentions into technology trend series."""
    from .track.trends import build_trends, emerging, role_transitions

    documents = [d for path in corpus for d in read_jsonl(path)]
    results = _read_results(mentions)
    table_data = build_trends(results, documents, min_confidence=min_confidence)

    console.print(f"[dim]{json.dumps(table_data.as_dict())}[/dim]\n")

    table = Table(title=f"most frequent technologies (n={len(table_data.series)} series)")
    for column in ("technology", "docs", "first seen"):
        table.add_column(column)
    for series in table_data.top(top):
        table.add_row(series.label[:50], str(series.total_docs), str(series.first_year))
    console.print(table)

    rising = emerging(table_data)
    if rising:
        growth = Table(title="fastest rising")
        for column in ("technology", "growth", "first seen", "docs"):
            growth.add_column(column)
        for row in rising[:top]:
            growth.add_row(row.label[:50], f"{row.growth:.2f}x", str(row.first_year), str(row.n_docs))
        console.print(growth)

    shifts = role_transitions(table_data)
    if shifts:
        adoption = Table(title="adoption shift (proposed -> used/compared)")
        for column in ("technology", "early proposed", "late used", "shift"):
            adoption.add_column(column)
        for row in shifts[:10]:
            adoption.add_row(
                row["label"][:50],
                f"{row['early_proposed']:.2f}",
                f"{row['late_used']:.2f}",
                f"{row['adoption_shift']:+.2f}",
            )
        console.print(adoption)


@app.command()
def evaluate(
    gold: Path = typer.Option(Path("data/gold/gold.jsonl")),
    corpus: list[Path] = typer.Option(
        [Path("data/raw/papers_eval.jsonl"), Path("data/raw/patents_eval.jsonl")]
    ),
    config: Path | None = typer.Option(None),
    suite: str = typer.Option("all", help="baselines | ablations | all"),
    out: Path = typer.Option(Path("runs/eval")),
    no_llm: bool = typer.Option(False, "--no-llm"),
) -> None:
    """Score the pipeline against the gold annotations."""
    from .eval.gold import read_gold
    from .eval.runner import ablation_conditions, baseline_conditions, run_suite

    cfg = _resolve_config(config, no_llm)
    gold_data = read_gold(gold)
    documents = [d for path in corpus for d in read_jsonl(path) if d.doc_id in gold_data]

    conditions = []
    if suite in ("baselines", "all"):
        conditions += baseline_conditions(cfg)
    if suite in ("ablations", "all"):
        conditions += ablation_conditions(cfg)

    run_suite(conditions, documents, gold_data, out_dir=out)
    console.print(f"wrote {out}/results.json")


@app.command()
def guard_report(
    corpus: list[Path] = typer.Argument(...),
    config: Path | None = typer.Option(None),
    limit: int = typer.Option(0),
) -> None:
    """Summarise what the guard stack removed and why, across a corpus."""
    from collections import Counter

    from .agents.orchestrator import Pipeline

    cfg = _resolve_config(config, no_llm=True)
    documents = [d for path in corpus for d in read_jsonl(path)]
    if limit:
        documents = documents[:limit]
    results = Pipeline(cfg).run(documents)

    reasons: Counter[str] = Counter()
    for result in results:
        for row in result.rejected:
            reasons[row["reason"].split(":")[0]] += 1

    table = Table(title="rejections by guard")
    table.add_column("guard")
    table.add_column("count", justify="right")
    table.add_column("share", justify="right")
    total = sum(reasons.values()) or 1
    for guard, count in reasons.most_common():
        table.add_row(guard, str(count), f"{count / total:.1%}")
    console.print(table)

    emitted = sum(len(r.mentions) for r in results)
    withheld = sum(len(r.abstained) for r in results)
    console.print(
        f"{emitted} emitted, {withheld} withheld, {total} rejected "
        f"({total / max(emitted + withheld + total, 1):.1%} of candidates)"
    )


# --- helpers ---------------------------------------------------------------


def _resolve_config(path: Path | None, no_llm: bool) -> Config:
    cfg = Config.load(path)
    if no_llm or not cfg.has_api_key():
        cfg = cfg.with_overrides(
            **{
                "recall.use_llm": False,
                "verify.use_llm": False,
                "verify.adjudicate_disagreements": False,
            }
        )
    return cfg


def _write_results(results: list[ExtractionResult], out: Path, *, include_trace: bool) -> None:
    with (out / "mentions.jsonl").open("w", encoding="utf-8") as fh:
        for result in results:
            payload = result.model_dump(mode="json")
            if not include_trace:
                payload.pop("trace", None)
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    flat = out / "mentions.tsv"
    with flat.open("w", encoding="utf-8") as fh:
        fh.write("doc_id\tstart\tend\tsurface\ttype\trole\tsection\tconfidence\tcanonical_id\n")
        for result in results:
            for m in result.mentions:
                fh.write(
                    f"{m.doc_id}\t{m.span.start}\t{m.span.end}\t{m.span.surface}\t"
                    f"{m.type.value}\t{m.role.value}\t{m.section.value}\t"
                    f"{m.confidence:.4f}\t{m.canonical_id or ''}\n"
                )


def _read_results(path: Path) -> list[ExtractionResult]:
    out: list[ExtractionResult] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(ExtractionResult.model_validate_json(line))
    return out


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
