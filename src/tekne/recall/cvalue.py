"""C-value / NC-value terminology extraction (Frantzi et al., 2000).

The point of keeping a 25-year-old statistical method in a modern pipeline is
that it is the only recaller here that is sensitive to *nesting*.  Patents are
full of terms that contain other terms ("lithium iron phosphate cathode material"
contains "iron phosphate cathode" and "cathode material"), and picking the right
level of the nest is exactly the granularity problem that breaks naive keyword
tracking.  C-value's nesting penalty is a principled, cheap answer.

We extend the standard formulation in two small ways:

* NC-value context weights are computed from the role-cue lexicon rather than
  from an unsupervised context-word pass, because our corpora are small.
* A background document frequency, if supplied, down-weights terms that are
  common in general English -- the "termhood" half of Kageura and Umino's
  unithood/termhood split.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..lexicons import lemma_key, tech_heads
from ..schema import Candidate, Document
from .base import RecallContext, Recaller

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9/+-]*")

# The classic (Adj|Noun)+Noun filter needs a way to reject everything else
# without a POS tagger, because this recaller is meant to stay parser-independent
# for the ablation. A closed-class stoplist is the standard substitute: no term
# contains a pronoun, an auxiliary, a preposition or a finite verb.
_CLOSED_CLASS = frozenset(
    """
    i we you he she it they them us him her me my our your their its his
    this that these those which who whom whose what where when why how
    a an the and or nor but so yet if then than because although while
    of in on at to for with by from as into onto upon within without
    between among through across over under above below after before during
    is are was were be been being am do does did done have has had having
    can could may might shall should will would must ought
    not no nor none any all both each every some many few more most less least
    very much such same other another here there also however therefore thus
    where whereas wherein whereby herein hereof thereof
    """.split()
)

_CONTEXT_CUES = frozenset(
    """
    propose present introduce develop use employ apply implement based novel
    method technique approach system device apparatus comprise include utilize
    known conventional improve enable perform provide
    """.split()
)


@dataclass
class CorpusStats:
    """Term frequencies over a corpus, plus the nesting relation."""

    term_freq: Counter[str] = field(default_factory=Counter)
    doc_freq: Counter[str] = field(default_factory=Counter)
    #: longer term -> nested shorter terms it contains
    nested_in: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    context_freq: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    n_docs: int = 0
    background_df: dict[str, float] | None = None

    def cvalue(self, term: str) -> float:
        n_words = len(term.split())
        if n_words < 1:
            return 0.0
        freq = self.term_freq.get(term, 0)
        if freq == 0:
            return 0.0
        length_factor = math.log2(n_words + 1)
        containers = [t for t, nested in self.nested_in.items() if term in nested and t != term]
        if not containers:
            return length_factor * freq
        nested_total = sum(self.term_freq.get(t, 0) for t in containers)
        return length_factor * (freq - nested_total / len(containers))

    def ncvalue(self, term: str, *, alpha: float = 0.8) -> float:
        """C-value blended with a context-cue weight and a termhood penalty."""
        base = self.cvalue(term)
        if base <= 0:
            return base
        ctx = self.context_freq.get(term)
        ctx_weight = 0.0
        if ctx:
            total = sum(ctx.values()) or 1
            hits = sum(c for w, c in ctx.items() if w in _CONTEXT_CUES)
            ctx_weight = hits / total
        score = alpha * base + (1 - alpha) * base * ctx_weight

        if self.background_df:
            # Penalise terms whose head is ubiquitous in general English.
            head = term.split()[-1]
            bg = self.background_df.get(head, 0.0)
            score *= 1.0 / (1.0 + 4.0 * bg)
        return score


def build_corpus_stats(
    docs: list[Document],
    analyses: dict[str, object] | None = None,
    *,
    min_tokens: int = 1,
    max_tokens: int = 5,
    background_df: dict[str, float] | None = None,
) -> CorpusStats:
    """Collect n-gram statistics under a linguistic filter.

    The filter is the classic ``(Adj|Noun)+ Noun`` approximation: we accept an
    n-gram whose final token is a known technology head and whose other tokens
    are alphabetic.  Running it over raw n-grams rather than parsed chunks keeps
    this recaller independent of the parser, which matters for the ablation.
    """
    stats = CorpusStats(n_docs=len(docs), background_df=background_df)
    heads = tech_heads()

    for doc in docs:
        seen_in_doc: set[str] = set()
        tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(doc.text)]
        words = [t[0] for t in tokens]
        lowered = [w.lower() for w in words]

        for n in range(min_tokens, max_tokens + 1):
            for i in range(len(words) - n + 1):
                gram_words = words[i : i + n]
                if lemma_key(gram_words[-1]) not in heads:
                    continue
                if not _is_termlike(gram_words):
                    continue
                term = lemma_key(" ".join(gram_words))
                if not term or len(term) < 3:
                    continue
                stats.term_freq[term] += 1
                seen_in_doc.add(term)
                # Context words: one token either side.
                if i > 0:
                    stats.context_freq[term][lemma_key(lowered[i - 1])] += 1
                if i + n < len(words):
                    stats.context_freq[term][lemma_key(lowered[i + n])] += 1

        for term in seen_in_doc:
            stats.doc_freq[term] += 1

    _index_nesting(stats)
    return stats


def _is_termlike(words: list[str]) -> bool:
    """Approximate the (Adj|Noun)+ Noun filter without a POS tagger.

    A single closed-class token anywhere in the n-gram disqualifies it. That is
    enough to drop "we compare against Switch Transformer" and "based on
    Transformer architectures" while keeping "learned sparse representation",
    because verbs in running text are nearly always adjacent to a preposition,
    auxiliary or pronoun that this list covers.
    """
    return all(len(w) >= 2 and w.lower() not in _CLOSED_CLASS for w in words)


def _index_nesting(stats: CorpusStats) -> None:
    by_length: dict[int, list[str]] = defaultdict(list)
    for term in stats.term_freq:
        by_length[len(term.split())].append(term)
    max_len = max(by_length, default=0)
    for length in range(1, max_len + 1):
        shorter = set(by_length.get(length, ()))
        if not shorter:
            continue
        for longer_len in range(length + 1, max_len + 1):
            for longer in by_length.get(longer_len, ()):
                words = longer.split()
                for i in range(len(words) - length + 1):
                    sub = " ".join(words[i : i + length])
                    if sub in shorter:
                        stats.nested_in[longer].add(sub)


class TermStatRecaller(Recaller):
    """Propose spans whose lemma key scores above an NC-value threshold."""

    name = "cvalue"
    version = "1"
    cost_tier = 0

    def __init__(self, *, threshold: float = 2.0, max_terms: int = 400) -> None:
        self.threshold = threshold
        self.max_terms = max_terms

    def propose(self, ctx: RecallContext) -> list[Candidate]:
        stats: CorpusStats | None = ctx.corpus_stats
        if stats is None:
            # Single-document mode: compute statistics over this document alone.
            stats = build_corpus_stats([ctx.document])
        heads = tech_heads()
        doc = ctx.document

        scored = sorted(
            ((term, stats.ncvalue(term)) for term in stats.term_freq),
            key=lambda kv: -kv[1],
        )
        accepted = {t for t, s in scored[: self.max_terms] if s >= self.threshold}
        if not accepted:
            return []

        out: list[Candidate] = []
        tokens = [(m.start(), m.end()) for m in _TOKEN.finditer(doc.text)]
        for n in range(1, 6):
            for i in range(len(tokens) - n + 1):
                start = tokens[i][0]
                end = tokens[i + n - 1][1]
                surface = doc.text[start:end]
                if lemma_key(surface.split()[-1]) not in heads:
                    continue
                key = lemma_key(surface)
                if key not in accepted:
                    continue
                cand = self._candidate(
                    doc,
                    start,
                    end,
                    features={
                        "ncvalue": float(stats.ncvalue(key)),
                        "corpus_df": float(stats.doc_freq.get(key, 0)),
                    },
                    notes={"term_key": key, "path": "cvalue"},
                )
                if cand:
                    out.append(cand)
        return out
