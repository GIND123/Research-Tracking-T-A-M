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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

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
        max_anchors_per_block: int = 40,
    ) -> None:
        self.scorer = embedding_scorer
        self.similarity_threshold = similarity_threshold
        self.require_head_match = require_head_match
        self.max_anchors_per_block = max_anchors_per_block
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
        for _kb_id, keys in by_kb.items():
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
        """Single-link agglomeration over the KB-unknown residue.

        Candidate pairs are generated by blocking rather than by comparing every
        pair. The head-compatibility constraint already restricts merges to terms
        that share a head noun or where one contains the other, so those two
        relations *are* the blocking keys: bucket by head, and look up each term's
        contiguous sub-phrases. That turns an all-pairs similarity matrix into a
        few thousand lookups.

        The difference is not academic. Without a knowledge base every term is
        NIL, so the residue is the whole vocabulary; on a 44-document corpus the
        all-pairs version needed a 9000x9000 matrix and had not finished after
        fifteen minutes, which is also what it would do to a corpus run.
        """
        if len(keys) < 2 or self.scorer is None or not getattr(self.scorer, "available", False):
            return {}

        pairs = self._candidate_pairs(keys, groups)
        if not pairs:
            return {}

        index = {key: i for i, key in enumerate(keys)}
        vectors = self.scorer.encode(keys)
        if vectors is None:
            return {}

        import numpy as np

        matrix = np.asarray(vectors)
        left = np.fromiter((index[a] for a, _b in pairs), dtype=int, count=len(pairs))
        right = np.fromiter((index[b] for _a, b in pairs), dtype=int, count=len(pairs))
        sims = np.einsum("ij,ij->i", matrix[left], matrix[right])

        parent = {k: k for k in keys}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Merge the most frequent terms first so the cluster anchor is stable.
        order = sorted(
            range(len(pairs)),
            key=lambda i: -(len(groups[pairs[i][0]]) + len(groups[pairs[i][1]])),
        )
        for i in order:
            if sims[i] < self.similarity_threshold:
                continue
            ra, rb = find(pairs[i][0]), find(pairs[i][1])
            if ra == rb:
                continue
            if len(groups[ra]) >= len(groups[rb]):
                parent[rb] = ra
            else:
                parent[ra] = rb
        return {k: find(k) for k in keys if find(k) != k}

    def _candidate_pairs(
        self, keys: list[str], groups: dict[str, list[TechMention]] | None = None
    ) -> list[tuple[str, str]]:
        """Pairs that could pass ``_heads_compatible``, found without all-pairs.

        Within a head bucket, each term is compared against the most frequent
        members rather than against all of them. Head frequency is Zipfian --
        "system", "device" and "member" collect hundreds of terms each -- so an
        unbounded bucket reintroduces the quadratic term this method exists to
        remove. Restricting to frequent anchors loses nothing that single-link
        agglomeration would have kept, because a cluster's anchor is its most
        frequent form by construction.
        """
        counts = {k: len(groups[k]) if groups and k in groups else 1 for k in keys}
        by_head: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            by_head[key.split()[-1]].append(key)

        pairs: set[tuple[str, str]] = set()
        for bucket in by_head.values():
            if len(bucket) < 2:
                continue
            bucket.sort(key=lambda k: (-counts[k], k))
            anchors = bucket[: self.max_anchors_per_block]
            for key in bucket:
                for anchor in anchors:
                    if key != anchor:
                        pairs.add(tuple(sorted((key, anchor))))

        if not self.require_head_match:
            return sorted(pairs)

        # The containment half of the constraint: a term and any of its
        # contiguous sub-phrases, looked up rather than searched for.
        present = set(keys)
        for key in keys:
            tokens = key.split()
            if len(tokens) < 2:
                continue
            for size in range(1, len(tokens)):
                for start in range(len(tokens) - size + 1):
                    sub = " ".join(tokens[start : start + size])
                    if sub != key and sub in present:
                        pairs.add(tuple(sorted((key, sub))))
        return sorted(pairs)


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
