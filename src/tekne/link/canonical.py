"""Normalisation and canonical identity for extracted mentions.

Counting surface forms is not tracking technologies.  "CNN", "convolutional
neural network" and "ConvNets" are one technology with three trajectories unless
they are merged, and merging them by string similarity alone also merges
"convolutional neural network" with "recurrent neural network", which is worse.

The approach is a cascade of increasingly permissive, increasingly expensive
evidence, stopping at the first that fires:

1. **In-document abbreviation pairs.** Free, near-exact, and the only signal that
   reliably links an acronym to its expansion. A pair observed in one document is
   promoted to a corpus-wide alias.
2. **Knowledge-base identity.** Two mentions that hit the same KB entry are the
   same technology by fiat.
3. **Lemma-key identity.** Morphological and determiner variation.
4. **Embedding agglomeration over the NIL residue.** Everything the KB has never
   heard of -- which is where emerging technologies live by definition -- is
   clustered on its own, with a deliberately conservative threshold and a
   head-noun compatibility constraint that stops "graph neural network" and
   "spiking neural network" collapsing.

Clusters that never touch the KB get a stable synthetic id derived from their
canonical form, so a technology can be tracked from the moment it is first named
and reconciled with an ontology later.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..lexicons import lemma_key
from ..schema import TechMention


@dataclass
class CanonicalEntry:
    canonical_id: str
    label: str
    aliases: set[str] = field(default_factory=set)
    kb_id: str | None = None
    #: Earliest document year in which the cluster was observed.
    first_seen: int | None = None
    n_mentions: int = 0
    n_documents: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "label": self.label,
            "aliases": sorted(self.aliases),
            "kb_id": self.kb_id,
            "first_seen": self.first_seen,
            "n_mentions": self.n_mentions,
            "n_documents": self.n_documents,
        }


class Canonicalizer:
    version = "2"

    def __init__(
        self,
        *,
        embedding_scorer: Any = None,
        similarity_threshold: float = 0.86,
        require_head_match: bool = True,
    ) -> None:
        self.scorer = embedding_scorer
        self.similarity_threshold = similarity_threshold
        self.require_head_match = require_head_match
        self.entries: dict[str, CanonicalEntry] = {}
        self._key_to_id: dict[str, str] = {}
        self._alias_pairs: dict[str, str] = {}

    # -- alias harvesting ---------------------------------------------------

    def add_abbreviation_pairs(self, pairs: Iterable[Any]) -> None:
        """Register short/long pairs found in documents as corpus-wide aliases."""
        for pair in pairs:
            short = lemma_key(getattr(pair, "short", "") or "")
            long = lemma_key(getattr(pair, "long", "") or "")
            if not short or not long or short == long:
                continue
            # The expansion is the canonical side: it is the form that carries
            # meaning across documents, and acronyms collide across domains.
            self._alias_pairs[short] = long

    # -- fitting ------------------------------------------------------------

    def fit(self, mentions: Sequence[TechMention], years: dict[str, int | None] | None = None) -> None:
        years = years or {}
        groups: dict[str, list[TechMention]] = defaultdict(list)

        for mention in mentions:
            key = self._resolve_key(mention)
            groups[key].append(mention)

        # KB identity merges groups that the lexical key kept apart.
        by_kb: dict[str, list[str]] = defaultdict(list)
        for key, items in groups.items():
            kb_ids = {m.kb_link.entry_id for m in items if m.kb_link}
            if len(kb_ids) == 1:
                by_kb[next(iter(kb_ids))].append(key)

        merged: dict[str, str] = {}
        for kb_id, keys in by_kb.items():
            if len(keys) < 2:
                continue
            anchor = min(keys, key=lambda k: (-len(groups[k]), k))
            for key in keys:
                merged[key] = anchor

        nil_keys = [k for k in groups if not any(m.kb_link for m in groups[k])]
        for key, anchor in self._cluster_nil(nil_keys, groups).items():
            merged.setdefault(key, anchor)

        # Materialise entries.
        for key, items in groups.items():
            anchor = merged.get(key, key)
            entry = self.entries.get(anchor)
            if entry is None:
                entry = CanonicalEntry(canonical_id="", label=anchor)
                self.entries[anchor] = entry
            entry.aliases.update(lemma_key(m.span.surface) for m in items)
            entry.n_mentions += len(items)
            entry.n_documents += len({m.doc_id for m in items})
            for m in items:
                if m.kb_link and not entry.kb_id:
                    entry.kb_id = m.kb_link.entry_id
                year = years.get(m.doc_id)
                if year and (entry.first_seen is None or year < entry.first_seen):
                    entry.first_seen = year
            self._key_to_id[key] = anchor

        for anchor, entry in self.entries.items():
            entry.label = self._pick_label(anchor, entry)
            entry.canonical_id = entry.kb_id or f"tekne:{_stable_id(anchor)}"

    def _resolve_key(self, mention: TechMention) -> str:
        key = lemma_key(mention.span.surface)
        return self._alias_pairs.get(key, key)

    def _cluster_nil(
        self, keys: list[str], groups: dict[str, list[TechMention]]
    ) -> dict[str, str]:
        """Single-link agglomeration over the KB-unknown residue."""
        if len(keys) < 2 or self.scorer is None or not getattr(self.scorer, "available", False):
            return {}
        vectors = self.scorer.encode(keys)
        if vectors is None:
            return {}

        import numpy as np

        sims = np.asarray(vectors) @ np.asarray(vectors).T
        parent = {k: k for k in keys}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        order = sorted(range(len(keys)), key=lambda i: -len(groups[keys[i]]))
        for ai in order:
            for bi in order:
                if ai >= bi:
                    continue
                if sims[ai, bi] < self.similarity_threshold:
                    continue
                a, b = keys[ai], keys[bi]
                if self.require_head_match and not _heads_compatible(a, b):
                    continue
                ra, rb = find(a), find(b)
                if ra != rb:
                    # Keep the more frequent form as the cluster anchor.
                    if len(groups[ra]) >= len(groups[rb]):
                        parent[rb] = ra
                    else:
                        parent[ra] = rb
        return {k: find(k) for k in keys if find(k) != k}

    def _pick_label(self, anchor: str, entry: CanonicalEntry) -> str:
        """Prefer the longest alias: expansions beat acronyms as display forms."""
        candidates = entry.aliases or {anchor}
        return max(candidates, key=lambda a: (len(a.split()), len(a)))

    # -- application --------------------------------------------------------

    def assign(self, mention: TechMention) -> TechMention:
        key = self._resolve_key(mention)
        anchor = self._key_to_id.get(key, key)
        entry = self.entries.get(anchor)
        canonical_id = entry.canonical_id if entry else f"tekne:{_stable_id(key)}"
        return mention.model_copy(update={"canonical_id": canonical_id})

    def attestation(self) -> dict[str, int]:
        """Earliest corpus year per canonical id, for the temporal guard."""
        return {
            e.canonical_id: e.first_seen for e in self.entries.values() if e.first_seen is not None
        }

    def summary(self) -> dict[str, Any]:
        kb_linked = sum(1 for e in self.entries.values() if e.kb_id)
        return {
            "clusters": len(self.entries),
            "kb_linked": kb_linked,
            "nil_clusters": len(self.entries) - kb_linked,
            "alias_pairs": len(self._alias_pairs),
        }


def _heads_compatible(a: str, b: str) -> bool:
    """Two terms may only merge if they share a head noun or one contains the other."""
    ha, hb = a.split()[-1], b.split()[-1]
    if ha == hb:
        return True
    return a in b or b in a


def _stable_id(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def within_document_coref(mentions: Sequence[TechMention]) -> dict[str, list[str]]:
    """Group a document's mentions by normalised form.

    Kept separate from corpus canonicalisation because the two answer different
    questions: this one is "how many distinct technologies does this document
    discuss", which is the quantity a per-document report shows.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for mention in mentions:
        groups[lemma_key(mention.span.surface)].append(mention.mention_id)
    return dict(groups)


def frequency_table(mentions: Sequence[TechMention]) -> Counter[str]:
    return Counter(m.canonical_id or lemma_key(m.span.surface) for m in mentions)
