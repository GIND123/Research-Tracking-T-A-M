"""Lexico-syntactic recall: noun-phrase chunking plus patent claim patterns.

This carries most of the recall in practice.  Two paths exist: a dependency-parse
path over spaCy noun chunks, and a regex fallback for environments without the
model.  The fallback is not a toy -- patents are formulaic enough that a POS-free
chunker gets within a few points of the parsed path on claim text -- but it does
worse on paper prose, so the orchestrator records which path ran.
"""

from __future__ import annotations

import re

from ..lexicons import lemma_key, orthographic_patterns, tech_heads
from ..schema import Candidate, SectionKind
from .base import RecallContext, Recaller

# Modifiers that carry no technological content; stripping them from the left of
# a chunk turns "several recent convolutional architectures" into the term.
_VAGUE_MODIFIERS = frozenset(
    """
    a an the this that these those such said its their our your his her my
    one two three four five six seven eight nine ten first second third fourth
    another other others certain various several many few some all both each
    every any no more most least less further additional respective
    present instant subject aforementioned above below following preceding
    exemplary illustrative preferred alternative optional suitable appropriate
    conventional typical usual ordinary common general generic particular
    specific corresponding associated related given known new novel recent
    current existing proposed described disclosed claimed
    different same similar various number plurality
    """.split()
)

# Patent claim frames. Group 1 is the technology-bearing phrase in each.
_CLAIM_FRAMES = [
    # "means for cooling the substrate" -- functional claim element
    re.compile(r"\bmeans\s+for\s+([a-z][a-z-]*ing(?:\s+(?:the|a|an)?\s*[a-z][a-z-]*){0,3})", re.I),
    # "a photodetector configured to ..." -- the NP before the participle
    re.compile(
        r"\b(?:a|an|the|said)\s+((?:[a-z][a-z-]*\s+){0,3}[a-z][a-z-]*)\s+"
        r"(?:configured|adapted|arranged|operable|designed)\s+to\b",
        re.I,
    ),
    # "wherein the beamforming module comprises"
    re.compile(
        r"\bwherein\s+(?:the|said|a|an)\s+((?:[a-z][a-z-]*\s+){0,3}[a-z][a-z-]*)\s+"
        r"(?:comprises|includes|consists|is|are)\b",
        re.I,
    ),
    # "applying a chemical vapor deposition process to ..."
    re.compile(
        r"\b(?:applying|performing|using|employing|providing|forming|depositing|generating)\s+"
        r"(?:a|an|the|said)?\s*((?:[a-z][a-z-]*\s+){0,3}[a-z][a-z-]*)\b",
        re.I,
    ),
]

#: Caps on the span lattice a single chunk may contribute.
_MAX_VARIANTS = 10
_MAX_RIGHT_ALTERNATIVES = 2

# Coordination inside a chunk: "convolutional and recurrent neural networks".
_COORD = re.compile(r"\s+(?:and|or|and/or)\s+")

# POS-free chunker for the fallback path: a run of capitalised or lowercase
# word-like tokens ending in a known technology head.
_FALLBACK_NP = re.compile(
    r"\b((?:[A-Za-z][A-Za-z0-9/+.-]*\s+){0,4}[A-Za-z][A-Za-z0-9/+-]*)\b"
)


class PatternRecaller(Recaller):
    name = "pattern"
    version = "2"
    cost_tier = 0

    def __init__(self, *, max_chunk_tokens: int = 6, require_tech_head: bool = False) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        # Gating on the head-noun lexicon looks attractive and costs most of the
        # recall: measured on the development corpus it drops 53% of gold
        # mentions before any classifier sees them, because the lexicon cannot
        # anticipate "highback", "scapegoating", "rule-based elision" or the
        # mechanical-component vocabulary that patents are built from. The gate
        # is kept as an option only so the ablation can quantify that.
        self.require_tech_head = require_tech_head
        self._heads = tech_heads()
        self._ortho = orthographic_patterns()

    def propose(self, ctx: RecallContext) -> list[Candidate]:
        out: list[Candidate] = []
        if ctx.analysis.chunks:
            out.extend(self._from_chunks(ctx))
        else:
            out.extend(self._from_regex(ctx))
        out.extend(self._from_claim_frames(ctx))
        return [c for c in out if c is not None]

    # -- parsed path --------------------------------------------------------

    def _from_chunks(self, ctx: RecallContext) -> list[Candidate]:
        doc = ctx.document
        out: list[Candidate] = []
        noun_ends = {
            end for start, end, _text, pos, _tag in ctx.analysis.tokens if pos in ("NOUN", "PROPN")
        }
        for chunk in ctx.analysis.chunks:
            if chunk.head_pos not in ("NOUN", "PROPN"):
                continue
            head_key = lemma_key(chunk.head_text)
            head_is_tech = head_key in self._heads
            for start, end in self._variants(
                doc.text, chunk.start, chunk.end, chunk.head_end, noun_ends
            ):
                surface = doc.text[start:end]
                if len(surface.split()) > self.max_chunk_tokens:
                    continue
                ortho = self._ortho_score(surface)
                if self.require_tech_head and not head_is_tech:
                    if ortho < 1.0 and not self._kb_hit(ctx, surface):
                        continue
                cand = self._candidate(
                    doc,
                    start,
                    end,
                    features={
                        "head_is_tech": 1.0 if head_is_tech else 0.0,
                        "orthographic": ortho,
                        "n_tokens": float(len(surface.split())),
                    },
                    notes={"head": chunk.head_text, "path": "parsed"},
                )
                if cand:
                    out.append(cand)
        return out

    def _variants(
        self,
        text: str,
        start: int,
        end: int,
        head_end: int,
        noun_ends: set[int] | None = None,
    ) -> list[tuple[int, int]]:
        """Yield the plausible term spans inside one noun chunk.

        Boundaries are enumerated in both directions, which is a deliberate
        change from the obvious "emit the chunk" design. Two failure modes forced
        it, both measured on the development corpus:

        * **Left.** Whether the term is "string similarity" or "superficial
          string similarity" depends on information the recaller does not have.
        * **Right.** The parser attaches a following word to the chunk often
          enough to matter ("Obstacle-aware route planning grounds", where
          "grounds" is a verb tagged as a noun), and no amount of left truncation
          recovers the real term.

        So the chunk contributes a small lattice -- every left truncation crossed
        with every noun-final right boundary -- and the confidence-ranked decoder
        downstream picks one. The cap keeps that lattice at a few spans per
        chunk rather than quadratic in chunk length.
        """
        end = min(end, max(head_end, end))
        span_text = text[start:end]
        if not span_text.strip():
            return []

        token_offsets = _token_offsets(text, start, end)
        if not token_offsets:
            return []

        variants: list[tuple[int, int]] = []
        idx = 0
        while idx < len(token_offsets):
            tok_start, tok_end = token_offsets[idx]
            word = text[tok_start:tok_end].lower().strip(".,;:()")
            if word in _VAGUE_MODIFIERS or word.isdigit():
                idx += 1
                continue
            break
        if idx >= len(token_offsets):
            return []

        core_start = token_offsets[idx][0]
        ends = [end]
        if noun_ends:
            # Alternative right boundaries: any noun-final token in the last few
            # positions of the chunk, nearest the head first.
            for _tok_start, tok_end in reversed(token_offsets[idx:-1][-_MAX_RIGHT_ALTERNATIVES:]):
                if tok_end in noun_ends and tok_end < end:
                    ends.append(tok_end)
        for right in ends:
            for offset in range(idx, len(token_offsets)):
                left = token_offsets[offset][0]
                if left < right:
                    variants.append((left, right))
            if len(variants) >= _MAX_VARIANTS:
                break

        # Coordination split: propose each conjunct with the shared head.
        for coord in _COORD.finditer(span_text):
            left_end = start + coord.start()
            if left_end > core_start + 1:
                variants.append((core_start, left_end))
            right_start = start + coord.end()
            if right_start < end - 1:
                variants.append((right_start, end))

        seen: set[tuple[int, int]] = set()
        return [v for v in variants if v not in seen and not seen.add(v)]

    # -- fallback path ------------------------------------------------------

    def _from_regex(self, ctx: RecallContext) -> list[Candidate]:
        doc = ctx.document
        out: list[Candidate] = []
        for m in _FALLBACK_NP.finditer(doc.text):
            phrase = m.group(1)
            tokens = phrase.split()
            if not tokens:
                continue
            head_is_tech = lemma_key(tokens[-1]) in self._heads
            if self.require_tech_head and not head_is_tech:
                continue
            start = m.start(1)
            # Strip vague modifiers from the left.
            offset = 0
            while offset < len(tokens) - 1 and tokens[offset].lower() in _VAGUE_MODIFIERS:
                start += len(tokens[offset]) + 1
                offset += 1
            cand = self._candidate(
                doc,
                start,
                m.end(1),
                features={
                    "head_is_tech": 1.0 if head_is_tech else 0.0,
                    "n_tokens": float(len(tokens) - offset),
                },
                notes={"head": tokens[-1], "path": "regex"},
            )
            if cand:
                out.append(cand)
        return out

    # -- patent claim frames ------------------------------------------------

    def _from_claim_frames(self, ctx: RecallContext) -> list[Candidate]:
        doc = ctx.document
        claim_zones = [
            s.span
            for s in doc.sections
            if s.kind in (SectionKind.CLAIM_INDEP, SectionKind.CLAIM_DEP)
        ]
        if not claim_zones:
            return []
        out: list[Candidate] = []
        for zone in claim_zones:
            region = doc.text[zone.start : zone.end]
            for frame in _CLAIM_FRAMES:
                for m in frame.finditer(region):
                    start = zone.start + m.start(1)
                    end = zone.start + m.end(1)
                    surface = doc.text[start:end]
                    if len(surface.split()) > self.max_chunk_tokens:
                        continue
                    cand = self._candidate(
                        doc,
                        start,
                        end,
                        features={"claim_frame": 1.0, "n_tokens": float(len(surface.split()))},
                        notes={"path": "claim_frame"},
                    )
                    if cand:
                        out.append(cand)
        return out

    # -- helpers ------------------------------------------------------------

    def _ortho_score(self, surface: str) -> float:
        best = 0.0
        for _name, pattern, weight in self._ortho:
            if pattern.search(surface):
                best = max(best, weight)
        return best

    @staticmethod
    def _kb_hit(ctx: RecallContext, surface: str) -> bool:
        gaz = ctx.gazetteer
        return bool(gaz and gaz.contains(lemma_key(surface)))


_TOKEN = re.compile(r"\S+")


def _token_offsets(text: str, start: int, end: int) -> list[tuple[int, int]]:
    return [(start + m.start(), start + m.end()) for m in _TOKEN.finditer(text[start:end])]
