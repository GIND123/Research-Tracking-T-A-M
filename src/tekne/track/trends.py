"""Technology evolution from extracted mentions.

This module exists to make the extraction task's requirements concrete rather
than to be a contribution in itself.  Three properties of the extractor turn out
to matter here, and none of them is visible in an F1 score:

**Canonical identity.**  A trend line is per technology, not per string.  If
"convolutional neural network" and "CNN" are separate series, both look half as
important as the technology is, and a technology that changes its preferred name
appears to die and be replaced.

**Precision, asymmetrically.**  A false mention is not averaged away by scale.
Because trend detection looks for *deviation from a baseline*, a term that is
spuriously extracted at a low constant rate produces a flat line (harmless), but
one whose spurious rate tracks corpus growth produces a rising line
indistinguishable from a real emerging technology.  Recall errors, by contrast,
mostly rescale a series without changing its shape.

**Role.**  Raw counts conflate a technology being *introduced* with it being
*used* and with it being *beaten*.  The role distribution separates them, and the
transition from PROPOSED-heavy to USED-heavy is the clearest single signal of a
technology having been adopted.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..schema import Document, ExtractionResult, Role, TechMention, TechType


@dataclass
class Series:
    """One technology's trajectory."""

    canonical_id: str
    label: str
    #: year -> number of *documents* mentioning it (not raw mentions)
    doc_counts: dict[int, int] = field(default_factory=dict)
    roles: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    first_year: int | None = None
    total_docs: int = 0

    def share(self, totals: dict[int, int]) -> dict[int, float]:
        """Document frequency normalised by corpus size, which is what makes
        years comparable when the corpus grows."""
        return {
            year: self.doc_counts.get(year, 0) / totals[year]
            for year in sorted(totals)
            if totals[year]
        }

    def role_mix(self, year: int) -> dict[str, float]:
        counts = self.roles.get(year)
        if not counts:
            return {}
        total = sum(counts.values()) or 1
        return {role: n / total for role, n in counts.items()}


@dataclass
class TrendTable:
    series: dict[str, Series] = field(default_factory=dict)
    year_totals: dict[int, int] = field(default_factory=dict)
    n_documents: int = 0
    n_mentions: int = 0
    undated: int = 0

    def top(self, n: int = 20) -> list[Series]:
        return sorted(self.series.values(), key=lambda s: -s.total_docs)[:n]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_documents": self.n_documents,
            "n_mentions": self.n_mentions,
            "undated": self.undated,
            "year_totals": dict(sorted(self.year_totals.items())),
            "n_series": len(self.series),
        }


def build_trends(
    results: Sequence[ExtractionResult],
    documents: Sequence[Document],
    *,
    types: frozenset[TechType] | None = None,
    min_confidence: float = 0.5,
) -> TrendTable:
    """Aggregate per-document extractions into per-year, per-technology series.

    Counting is by *document frequency* rather than mention frequency. A patent
    that repeats "grinding tool" forty times is one document's worth of evidence
    that grinding tools matter, not forty, and mention frequency is dominated by
    genre conventions rather than by technological significance.
    """
    types = types or frozenset({TechType.ARTIFACT, TechType.METHOD, TechType.MATERIAL})
    years = {doc.doc_id: doc.year() for doc in documents}
    table = TrendTable(n_documents=len(documents))

    for result in results:
        year = years.get(result.doc_id)
        if year is None:
            table.undated += 1
            continue
        table.year_totals[year] = table.year_totals.get(year, 0) + 1

        seen: dict[str, list[TechMention]] = defaultdict(list)
        for mention in result.mentions:
            if mention.type not in types or mention.confidence < min_confidence:
                continue
            key = mention.canonical_id or mention.normalized
            seen[key].append(mention)
            table.n_mentions += 1

        for key, mentions in seen.items():
            series = table.series.get(key)
            if series is None:
                series = Series(canonical_id=key, label=_label(mentions))
                table.series[key] = series
            series.doc_counts[year] = series.doc_counts.get(year, 0) + 1
            series.total_docs += 1
            series.first_year = year if series.first_year is None else min(series.first_year, year)
            for mention in mentions:
                series.roles[year][mention.role.value] += 1

    return table


def _label(mentions: Sequence[TechMention]) -> str:
    """Most frequent surface, tie-broken towards the longer form."""
    counts = Counter(m.normalized for m in mentions)
    return max(counts, key=lambda s: (counts[s], len(s)))


# --- emergence -------------------------------------------------------------


@dataclass
class Emergence:
    canonical_id: str
    label: str
    first_year: int
    slope: float
    recent_share: float
    earlier_share: float
    n_docs: int
    #: Ratio of recent to earlier share, capped for display.
    growth: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "label": self.label,
            "first_year": self.first_year,
            "slope": round(self.slope, 6),
            "growth": round(self.growth, 3),
            "recent_share": round(self.recent_share, 5),
            "earlier_share": round(self.earlier_share, 5),
            "n_docs": self.n_docs,
        }


def emerging(
    table: TrendTable,
    *,
    window: int = 3,
    min_docs: int = 4,
    top_n: int = 25,
) -> list[Emergence]:
    """Rank technologies by recent growth in normalised document frequency.

    Deliberately simple: a least-squares slope over the share series plus the
    ratio of the last ``window`` years to everything before them.  Anything more
    elaborate (Bass diffusion, change-point detection) would be fitting a shape
    to a handful of noisy points, and the point of the exercise is to show that
    the extractor's output supports the analysis, not to advance trend detection.
    """
    years = sorted(table.year_totals)
    if len(years) < 2:
        return []
    recent = set(years[-window:])
    earlier = set(years[:-window])

    out: list[Emergence] = []
    for series in table.series.values():
        if series.total_docs < min_docs:
            continue
        shares = series.share(table.year_totals)
        slope = _slope([(y, shares.get(y, 0.0)) for y in years])

        recent_share = _mean([shares.get(y, 0.0) for y in recent]) if recent else 0.0
        earlier_share = _mean([shares.get(y, 0.0) for y in earlier]) if earlier else 0.0
        # Laplace-style floor so a technology absent from the earlier window has
        # a large but finite growth rather than an infinite one.
        floor = 1.0 / max(sum(table.year_totals.values()), 1)
        growth = (recent_share + floor) / (earlier_share + floor)

        out.append(
            Emergence(
                canonical_id=series.canonical_id,
                label=series.label,
                first_year=series.first_year or years[0],
                slope=slope,
                recent_share=recent_share,
                earlier_share=earlier_share,
                n_docs=series.total_docs,
                growth=growth,
            )
        )

    out.sort(key=lambda e: (-e.growth, -e.slope))
    return out[:top_n]


def declining(table: TrendTable, **kwargs: Any) -> list[Emergence]:
    rows = emerging(table, top_n=10**6, **kwargs)
    rows.sort(key=lambda e: (e.growth, e.slope))
    return rows[: kwargs.get("top_n", 25)]


def role_transitions(table: TrendTable, *, min_docs: int = 6) -> list[dict[str, Any]]:
    """Technologies whose stance mix has shifted from proposed towards used.

    The adoption signature: a technology starts life as something papers claim
    to have invented and ends as something they merely employ.
    """
    out: list[dict[str, Any]] = []
    for series in table.series.values():
        if series.total_docs < min_docs:
            continue
        years = sorted(series.roles)
        if len(years) < 2:
            continue
        half = max(1, len(years) // 2)
        early = Counter()
        late = Counter()
        for year in years[:half]:
            early.update(series.roles[year])
        for year in years[half:]:
            late.update(series.roles[year])

        early_prop = _fraction(early, Role.PROPOSED.value)
        late_used = _fraction(late, Role.USED.value)
        late_compared = _fraction(late, Role.COMPARED.value)
        shift = (late_used + late_compared) - early_prop
        out.append(
            {
                "canonical_id": series.canonical_id,
                "label": series.label,
                "early_proposed": round(early_prop, 3),
                "late_used": round(late_used, 3),
                "late_compared": round(late_compared, 3),
                "adoption_shift": round(shift, 3),
                "n_docs": series.total_docs,
            }
        )
    out.sort(key=lambda r: -r["adoption_shift"])
    return out


def _fraction(counter: Counter, key: str) -> float:
    total = sum(counter.values())
    return counter.get(key, 0) / total if total else 0.0


def _slope(points: Sequence[tuple[int, float]]) -> float:
    if len(points) < 2:
        return 0.0
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in points)
    den = sum((x - mean_x) ** 2 for x, _y in points)
    return num / den if den else 0.0


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def cooccurrence(
    results: Sequence[ExtractionResult], *, min_docs: int = 3, top_n: int = 40
) -> list[tuple[str, str, float]]:
    """Pointwise mutual information between technologies over documents.

    A cheap view of which technologies travel together, which is the raw
    material for the successor/complement analysis sketched in the paper's
    future work.
    """
    doc_sets: list[set[str]] = []
    counts: Counter[str] = Counter()
    for result in results:
        keys = {m.canonical_id or m.normalized for m in result.mentions}
        if keys:
            doc_sets.append(keys)
            counts.update(keys)

    n = len(doc_sets) or 1
    pair_counts: Counter[tuple[str, str]] = Counter()
    for keys in doc_sets:
        eligible = sorted(k for k in keys if counts[k] >= min_docs)
        for i, a in enumerate(eligible):
            for b in eligible[i + 1 :]:
                pair_counts[(a, b)] += 1

    out: list[tuple[str, str, float]] = []
    for (a, b), joint in pair_counts.items():
        if joint < min_docs:
            continue
        pmi = math.log((joint / n) / ((counts[a] / n) * (counts[b] / n)))
        out.append((a, b, pmi))
    out.sort(key=lambda t: -t[2])
    return out[:top_n]
