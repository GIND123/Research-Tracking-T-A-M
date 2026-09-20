"""Resolving overlapping and nested candidates into a coherent mention set.

Five recallers running over the same text produce heavy overlap, and the overlap
is not noise -- it is the granularity question in disguise.  Given
"graph convolutional network", the chunker proposes all of "graph convolutional
network", "convolutional network" and (via the KB) "network"; picking wrongly
either fragments one technology into three or collapses three into one.

The rule used here is: prefer the span with the most independent support, break
ties towards the longer span, and let a strictly-nested span survive only when it
has support the container lacks.  That last clause is what keeps "CNN" alive
inside "CNN encoder" when the abbreviation miner found it, while discarding the
bare "network" inside "neural network" that only the chunker proposed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema import Candidate

#: Recallers whose independent support is strong enough to rescue a nested span.
DISTINCTIVE = frozenset({"abbrev", "gazetteer", "llm"})


@dataclass
class SelectionStats:
    proposed: int = 0
    kept: int = 0
    dropped_overlap: int = 0
    kept_nested: int = 0


def candidate_score(cand: Candidate) -> float:
    """Ranking signal for overlap resolution (not a confidence)."""
    feats = cand.features
    score = 1.6 * len(set(cand.proposers))
    score += 1.4 * feats.get("kb_hit", 0.0)
    score += 1.2 * feats.get("abbrev_pair", 0.0)
    score += 1.0 * feats.get("llm_proposed", 0.0)
    score += 0.8 * feats.get("claim_frame", 0.0)
    score += 0.6 * min(feats.get("ncvalue", 0.0) / 8.0, 1.0)
    score += 0.5 * feats.get("head_is_tech", 0.0)
    score += 0.35 * min(feats.get("orthographic", 0.0) / 2.5, 1.0)
    # Mild length preference, saturating: a four-token term is specific, an
    # eight-token one is usually a chunking accident.
    n_tokens = feats.get("n_tokens") or float(len(cand.span.surface.split()))
    score += 0.30 * min(n_tokens, 4.0)
    return score


def resolve_overlaps(
    candidates: list[Candidate], *, keep_distinctive_nested: bool = True
) -> tuple[list[Candidate], SelectionStats]:
    stats = SelectionStats(proposed=len(candidates))
    if not candidates:
        return [], stats

    ranked = sorted(candidates, key=lambda c: (-candidate_score(c), c.span.start, -len(c.span)))
    kept: list[Candidate] = []

    for cand in ranked:
        conflict = None
        for existing in kept:
            if cand.span.overlaps(existing.span):
                conflict = existing
                break
        if conflict is None:
            kept.append(cand)
            continue

        if (
            keep_distinctive_nested
            and conflict.span.contains(cand.span)
            and set(cand.proposers) & DISTINCTIVE
            and not (set(cand.proposers) <= set(conflict.proposers))
        ):
            kept.append(cand)
            stats.kept_nested += 1
            continue

        stats.dropped_overlap += 1

    kept.sort(key=lambda c: (c.span.start, c.span.end))
    stats.kept = len(kept)
    return kept, stats
