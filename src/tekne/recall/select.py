"""Ranking signal for overlapping candidates.

Five recallers over the same text produce heavy overlap, and the overlap is the
granularity question in disguise: given "graph convolutional network", the
chunker proposes that, "convolutional network" and (via the KB) "network", and
picking wrongly either fragments one technology into three or collapses three
into one.

Choosing between them is the decoder's job in
:mod:`tekne.agents.orchestrator`, after each span has a type and a calibrated
confidence.  What lives here is the cheap ranking used when a document produces
more candidates than the budget allows and the set has to be capped before any
of that is known.
"""

from __future__ import annotations

from ..schema import Candidate

#: Recallers whose independent support is strong enough to rescue a nested span.
DISTINCTIVE = frozenset({"abbrev", "gazetteer", "llm"})


def candidate_score(cand: Candidate) -> float:
    """How much independent support a candidate has, before it is typed.

    Used only to cap an over-large candidate set: it is a proxy for "how likely
    is this to matter", not a confidence, and it deliberately ignores everything
    that needs a classifier to know.
    """
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
