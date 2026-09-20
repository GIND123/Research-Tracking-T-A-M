#!/usr/bin/env python3
"""Plot the figures the report uses.

    python scripts/make_figures.py --curves runs/eval/risk_coverage.json

Produces report/figures/risk_coverage.pdf: risk against coverage for each
condition. This is the figure that carries the selective-prediction argument --
the systems that look similar on F1 separate clearly on where their errors sit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Conditions worth plotting; anything else clutters the panel.
SERIES = [
    ("full", "TEKNE", "-", 2.0),
    ("-verifier", "$-$ verifier", "--", 1.4),
    ("-negative", "$-$ negative lexicon", "-.", 1.4),
    ("union", "union of recallers", ":", 1.4),
    ("chunker", "chunker only", (0, (3, 1, 1, 1)), 1.2),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", default="runs/eval/risk_coverage.json")
    parser.add_argument("--out", default="report/figures/risk_coverage.pdf")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = json.loads(Path(args.curves).read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for name, label, style, width in SERIES:
        curve = curves.get(name)
        if not curve or not curve["points"]:
            continue
        points = curve["points"]
        xs = [p["coverage"] for p in points]
        ys = [p["risk"] for p in points]
        ax.plot(xs, ys, linestyle=style, linewidth=width, label=f"{label} ({curve['aurc']:.2f})")

    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (1 $-$ precision)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=6, loc="lower right", frameon=False, title="AURC", title_fontsize=6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
