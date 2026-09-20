"""Typed contracts shared by every stage of the pipeline.

Two invariants hold everywhere below and are enforced rather than assumed:

1. A ``Span`` is a half-open character interval into ``Document.text`` and its
   ``surface`` field is *always* the literal slice at those offsets.  Nothing
   downstream is allowed to carry a string that did not come out of the source
   document (see :mod:`tekne.guard.grounding`).
2. Anything that reaches :class:`TechMention` carries a complete
   :class:`Provenance` record.  Emitting an unattributable mention is a bug, not
   a degraded mode.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Genre(str, Enum):
    PAPER = "paper"
    PATENT = "patent"


class SectionKind(str, Enum):
    """Structural zones we care about, unified across the two genres.

    The zone is the single strongest prior we have on what a mention *means*:
    a term in ``RELATED_WORK`` is prior art, the same term in ``CLAIM_INDEP`` is
    the thing being monopolised.  Genre-specific segmenters map their native
    headings onto this closed set.
    """

    TITLE = "title"
    ABSTRACT = "abstract"
    # Paper zones
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    EXPERIMENT = "experiment"
    RESULTS = "results"
    CONCLUSION = "conclusion"
    # Patent zones
    PATENT_FIELD = "patent_field"
    PATENT_BACKGROUND = "patent_background"
    PATENT_SUMMARY = "patent_summary"
    PATENT_DETAIL = "patent_detail"
    CLAIM_INDEP = "claim_independent"
    CLAIM_DEP = "claim_dependent"
    # Fallback
    OTHER = "other"


PRIOR_ART_ZONES = frozenset(
    {SectionKind.RELATED_WORK, SectionKind.PATENT_BACKGROUND, SectionKind.PATENT_FIELD}
)
CONTRIBUTION_ZONES = frozenset(
    {
        SectionKind.METHOD,
        SectionKind.PATENT_SUMMARY,
        SectionKind.CLAIM_INDEP,
        SectionKind.CLAIM_DEP,
    }
)


class TechType(str, Enum):
    """Extraction target ontology.

    ``FIELD`` and ``TASK`` are extracted but flagged, not discarded: trend
    analysis needs them as the backdrop against which artefacts are placed, yet
    mixing them into the artefact stream is what makes naive keyword trackers
    report that "machine learning" is the fastest-growing technology of every
    year since 1995.
    """

    ARTIFACT = "artifact"  # named, specific: BERT, LoRA, LiDAR, CRISPR-Cas9
    METHOD = "method"  # generic technique: attention mechanism, CVD
    FIELD = "field"  # umbrella area: machine learning, wireless communication
    TASK = "task"  # problem being solved: object detection, state of charge estimation
    MATERIAL = "material"  # substance-as-technology: LiFePO4 cathode, graphene
    DATASET = "dataset"
    METRIC = "metric"
    TOOL = "tool"  # implementation vehicle: PyTorch, CUDA
    NOT_TECH = "not_tech"


#: Types that participate in technology-evolution tracking by default.
TRACKED_TYPES = frozenset({TechType.ARTIFACT, TechType.METHOD, TechType.MATERIAL})


class Role(str, Enum):
    """Stance of the document towards the mention.

    This is the field that turns a bag of terms into an evolution signal: a
    technology's life cycle is visible as its role distribution shifting from
    ``PROPOSED`` to ``COMPARED`` to ``USED`` over time.
    """

    PROPOSED = "proposed"  # the document's own contribution
    USED = "used"  # employed as a component
    COMPARED = "compared"  # baseline / prior system contrasted against
    BACKGROUND = "background"  # cited context, no commitment
    CLAIMED = "claimed"  # patent-specific: inside an independent claim
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    PASS = "pass"
    REJECT = "reject"
    ABSTAIN = "abstain"


class Span(BaseModel):
    """Half-open ``[start, end)`` character interval into ``Document.text``."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    surface: str

    @model_validator(mode="after")
    def _check_interval(self) -> Span:
        if self.end <= self.start:
            raise ValueError(f"empty or inverted span [{self.start}, {self.end})")
        if len(self.surface) != self.end - self.start:
            raise ValueError(
                f"surface length {len(self.surface)} != interval width {self.end - self.start}"
            )
        return self

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, other: Span) -> bool:
        return self.start <= other.start and other.end <= self.end

    def __len__(self) -> int:
        return self.end - self.start


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: SectionKind
    span: Span
    ordinal: int = 0
    heading: str | None = None
    # Claim number for patent claims, dependency parent if any.
    attrs: dict[str, Any] = Field(default_factory=dict)


class DocMetadata(BaseModel):
    source: str = "unknown"  # arxiv | google_patents | file
    source_id: str | None = None
    url: str | None = None
    title: str | None = None
    # Publication or filing date, ISO-8601. Drives the temporal guard.
    date: str | None = None
    categories: list[str] = Field(default_factory=list)  # arXiv cats or CPC labels
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A normalised document. ``text`` is the single source of truth for offsets."""

    doc_id: str
    genre: Genre
    text: str
    sections: list[Section] = Field(default_factory=list)
    metadata: DocMetadata = Field(default_factory=DocMetadata)
    #: sha256 of the raw text this was normalised from, for reproducibility.
    raw_sha256: str | None = None

    def slice(self, span: Span) -> str:
        return self.text[span.start : span.end]

    def section_at(self, offset: int) -> Section | None:
        # Sections do not overlap; linear scan is fine at document scale.
        for sec in self.sections:
            if sec.span.start <= offset < sec.span.end:
                return sec
        return None

    def year(self) -> int | None:
        if self.metadata.date and len(self.metadata.date) >= 4:
            try:
                return int(self.metadata.date[:4])
            except ValueError:
                return None
        return None


class Evidence(BaseModel):
    """The sentence a mention was read out of, plus its offsets."""

    model_config = ConfigDict(frozen=True)

    span: Span
    sentence_index: int = -1


class Candidate(BaseModel):
    """A pre-verification proposal. Recallers over-generate on purpose."""

    span: Span
    #: Recaller ids that independently proposed this span (drives consensus).
    proposers: tuple[str, ...] = ()
    features: dict[str, float] = Field(default_factory=dict)
    notes: dict[str, Any] = Field(default_factory=dict)

    def merged_with(self, other: Candidate) -> Candidate:
        props = tuple(sorted(set(self.proposers) | set(other.proposers)))
        feats = {**other.features, **self.features}
        return Candidate(
            span=self.span,
            proposers=props,
            features=feats,
            notes={**other.notes, **self.notes},
        )


class GuardVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    guard: str
    verdict: Verdict
    reason: str = ""
    score: float | None = None


class KBLink(BaseModel):
    model_config = ConfigDict(frozen=True)

    kb: str  # "cso" | "cpc" | ...
    entry_id: str
    label: str
    score: float = 0.0
    #: Earliest year the KB (or corpus) attests this entry; used by the temporal guard.
    attested_from: int | None = None


class Provenance(BaseModel):
    """Everything needed to re-derive, audit or diff a single emitted mention."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    proposers: tuple[str, ...]
    stage_versions: dict[str, str] = Field(default_factory=dict)
    config_digest: str = ""
    llm_calls: int = 0
    #: sha256 over (doc_id, offsets, surface) -- stable mention identity.
    content_hash: str = ""


class TechMention(BaseModel):
    """A technology mention that survived the full pipeline."""

    mention_id: str
    doc_id: str
    span: Span
    normalized: str
    type: TechType
    granularity: int = Field(default=2, ge=0, le=2)
    role: Role = Role.UNKNOWN
    section: SectionKind = SectionKind.OTHER
    evidence: Evidence
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    canonical_id: str | None = None
    kb_link: KBLink | None = None
    guards: list[GuardVerdict] = Field(default_factory=list)
    provenance: Provenance

    @property
    def surface(self) -> str:
        return self.span.surface

    def is_tracked(self) -> bool:
        return self.type in TRACKED_TYPES


class ExtractionResult(BaseModel):
    """Per-document pipeline output, including what was deliberately withheld."""

    doc_id: str
    genre: Genre
    mentions: list[TechMention] = Field(default_factory=list)
    #: Candidates dropped by a guard, retained so ablations and audits are cheap.
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    #: Candidates the pipeline declined to decide on -> human review queue.
    abstained: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


def content_hash(doc_id: str, start: int, end: int, surface: str) -> str:
    h = hashlib.sha256()
    h.update(f"{doc_id}\x00{start}\x00{end}\x00{surface}".encode())
    return h.hexdigest()[:16]
