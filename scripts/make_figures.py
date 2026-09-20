#!/usr/bin/env python3
"""Figures for the report.

    python scripts/make_figures.py

Writes into report/figures/:

  landscape.pdf   two panels, full text width: risk-coverage curves and the
                  precision/recall landscape across all conditions
  breakdown.pdf   two panels, one column: where gold mentions are lost, and
                  per-type recall

Design constraints, in rough order of how often they were violated by the first
attempt: ACL papers get printed in greyscale, so every colour distinction is
backed by a line style or a marker shape; the physical size is fixed at the
column and text widths so nothing is scaled at \\includegraphics time and the
type stays at the paper's own size; and no series is identified by colour alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Categorical slots 1-4 of the validated default palette. Adjacent-pair CVD
# separation 9.2 (deutan), normal-vision 27.6; the aqua slot sits below 3:1 on a
# white surface, which the direct labels on every series discharge.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID = "#dedcd6"

#: Single hue for magnitude panels, from the sequential blue ramp.
SEQ_FILL, SEQ_EDGE = "#86b6ef", "#2a78d6"

CURVES = [
    ("full", "TEKNE", BLUE, "-"),
    ("-verifier", "$-$verifier", ORANGE, "--"),
    ("-negative", "$-$negative lex.", AQUA, "-."),
    ("union", "union, no guards", VIOLET, (0, (1, 1.2))),
]

OUTCOME_LABELS = {
    "emitted": "emitted correctly",
    "withheld": "withheld: low confidence",
    "lost_at_decode": "lost: overlap decoding",
    "rejected": "rejected by a guard",
    "never_proposed": "never proposed",
    "wrong_type": "found, wrong type",
}

BASELINES = {"gazetteer", "chunker", "chunker-gated", "cvalue", "union"}


def style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "axes.edgecolor": INK_MUTED,
            "axes.linewidth": 0.5,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def recede(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


# --- figure 1 --------------------------------------------------------------


def landscape(results: dict, curves: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(6.9, 1.62))

    # (a) risk-coverage
    for name, label, colour, dash in CURVES:
        curve = curves.get(name)
        if not curve or not curve["points"]:
            continue
        xs = [p["coverage"] for p in curve["points"]]
        ys = [p["risk"] for p in curve["points"]]
        left.plot(
            xs, ys,
            color=colour, linestyle=dash, linewidth=1.1, zorder=3,
            label=f"{label}   {curve['aurc']:.2f}",
        )

    left.set_xlabel("coverage")
    left.set_ylabel("risk  (1 $-$ precision)")
    left.set_xlim(0, 1.02)
    left.set_ylim(0, 0.8)
    left.set_title("(a) risk–coverage", loc="left", color=INK_SOFT)
    # A legend rather than end-of-line labels: two of the four curves very
    # nearly coincide, which is the finding, and stacked end labels were
    # illegible because of it.
    legend = left.legend(
        frameon=False, loc="lower right", handlelength=2.4,
        handletextpad=0.5, labelspacing=0.25, borderpad=0.2,
        title="AURC (lower is better)", title_fontsize=6,
    )
    legend.get_title().set_color(INK_SOFT)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    recede(left)

    # (b) precision-recall landscape
    marks = {"baseline": ("o", ORANGE), "ablation": ("s", BLUE), "full": ("D", AQUA)}
    # Hand-placed label offsets: the ablations cluster inside a 0.06 x 0.06 box,
    # so automatic placement stacks them on top of one another.
    offsets = {
        "full": (-8, 5), "-structure": (4, 3), "-consensus": (7, 0),
        "-abstention": (6, -2), "-negative": (6, -2), "-verifier": (6, -2),
        "gazetteer": (6, -2), "chunker-gated": (6, -2), "chunker": (-4, -8),
        "union": (6, -2), "cvalue": (6, -1),
    }
    seen: set[str] = set()
    for name, row in results.items():
        metrics = row["metrics"]["strict"]
        group = "full" if name == "full" else ("baseline" if name in BASELINES else "ablation")
        marker, colour = marks[group]
        right.scatter(
            metrics["recall"],
            metrics["precision"],
            s=44 if group == "full" else 26,
            marker=marker,
            facecolor=colour if group == "full" else "none",
            edgecolor=colour,
            linewidth=1.0,
            zorder=4,
            label=group if group not in seen else None,
        )
        seen.add(group)
        right.annotate(
            "TEKNE" if name == "full" else name,
            xy=(metrics["recall"], metrics["precision"]),
            xytext=offsets.get(name, (5, -2)),
            textcoords="offset points",
            ha="right" if name == "full" else "left",
            fontsize=5.4,
            fontweight="bold" if group == "full" else "normal",
            color=INK if group == "full" else INK_SOFT,
        )

    # Iso-F1 contours, recessive: they are the reading aid, not the data.
    for f1 in (0.1, 0.2, 0.3, 0.4):
        # p = f1*r / (2r - f1); undefined at r = f1/2, so start above it.
        rs = [r / 400 for r in range(1, 400) if 2 * (r / 400) - f1 > 1e-6]
        ps = [(f1 * r) / (2 * r - f1) for r in rs]
        pts = [(r, p) for r, p in zip(rs, ps, strict=True) if 0 < p <= 0.44 and r <= 0.57]
        if pts:
            right.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                color=GRID, linewidth=0.5, zorder=1,
            )
            right.annotate(
                f"F$_1$={f1:g}",
                xy=pts[-1], xytext=(1, 1), textcoords="offset points",
                fontsize=5, color=INK_MUTED,
            )

    right.set_xlabel("strict recall")
    right.set_ylabel("strict precision")
    right.set_xlim(0, 0.58)
    right.set_ylim(0, 0.46)
    right.set_title("(b) the precision/recall landscape", loc="left", color=INK_SOFT)
    legend = right.legend(
        frameon=False, loc="lower left", handletextpad=0.2,
        labelspacing=0.25, borderpad=0.2, scatterpoints=1,
    )
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    recede(right)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    fig.savefig(out)
    print(f"wrote {out}")


# --- figure 2 --------------------------------------------------------------


def breakdown(results: dict, errors: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(3.3, 1.85), gridspec_kw={"height_ratios": [6, 2.6]}
    )

    # (a) where gold mentions go
    rows = [(OUTCOME_LABELS.get(k, k), v["share"], v["n"]) for k, v in errors["outcomes"].items()]
    rows.sort(key=lambda r: r[1])
    ys = range(len(rows))
    top.barh(
        list(ys),
        [r[1] for r in rows],
        height=0.62,
        color=SEQ_FILL,
        edgecolor=SEQ_EDGE,
        linewidth=0.6,
        zorder=3,
    )
    for y, (_label, share, n) in zip(ys, rows, strict=True):
        top.annotate(
            f"{share:.1%}  ({n})",
            xy=(share, y),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=5.8,
            color=INK_SOFT,
        )
    top.set_yticks(list(ys))
    top.set_yticklabels([r[0] for r in rows])
    top.set_xlim(0, 0.52)
    top.set_xlabel(f"share of {errors['gold_scored']} scored gold mentions")
    top.set_title("(a) where gold mentions end up", loc="left", color=INK_SOFT)
    recede(top)
    top.grid(axis="y", visible=False)

    # (b) per-type recall
    per_type = results["full"]["metrics"]["per_type"]
    order = [t for t in ("artifact", "material", "method") if t in per_type]
    values = [per_type[t]["recall"] for t in order]
    counts = [per_type[t]["matched"] + per_type[t]["missed"] for t in order]
    ys2 = range(len(order))
    bottom.barh(
        list(ys2), values, height=0.48,
        color=SEQ_FILL, edgecolor=SEQ_EDGE, linewidth=0.6, zorder=3,
    )
    for y, value, n in zip(ys2, values, counts, strict=True):
        bottom.annotate(
            f"{value:.2f}  (n={n})",
            xy=(value, y), xytext=(3, 0), textcoords="offset points",
            va="center", fontsize=5.8, color=INK_SOFT,
        )
    bottom.set_yticks(list(ys2))
    bottom.set_yticklabels(order)
    bottom.set_xlim(0, 0.95)
    bottom.set_xlabel("overlap recall")
    bottom.set_title("(b) recall by gold type", loc="left", color=INK_SOFT)
    recede(bottom)
    bottom.grid(axis="y", visible=False)

    fig.tight_layout(pad=0.4, h_pad=1.4)
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="runs/eval/results.json")
    parser.add_argument("--curves", default="runs/eval/risk_coverage.json")
    parser.add_argument("--errors", default="runs/eval/error_analysis.json")
    parser.add_argument("--out", default="report/figures")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    results = {c["name"]: c for c in payload["conditions"]}
    curves = json.loads(Path(args.curves).read_text(encoding="utf-8"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    landscape(results, curves, out / "landscape.pdf")

    errors_path = Path(args.errors)
    if errors_path.is_file():
        breakdown(results, json.loads(errors_path.read_text(encoding="utf-8")), out / "breakdown.pdf")
    else:
        print(f"skipping breakdown: {errors_path} not found (run scripts/error_analysis.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
