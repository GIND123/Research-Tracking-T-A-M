"""The invariants that must hold no matter how the pipeline is configured.

The guards in :mod:`tekne.guard` are filters and most of them are ablatable --
that is the point of having an ablation table. These tests cover the separate,
smaller set of properties that are *not* filters and must not be ablatable:

* an emitted mention's surface is the document's own characters at its offsets;
* its evidence span is the document's own characters and contains it;
* nothing --- including a model with a wider context window and a strong opinion
  --- can override either of those.

Each test here was written against a hole that existed. The pipeline passed its
whole suite with all three open, because the deterministic recallers never
actually produce a malformed span; the holes were only reachable through a
misconfiguration or a hostile model, which is exactly the case the design claims
to cover.
"""

from __future__ import annotations

import pytest

from tekne.agents.orchestrator import Pipeline
from tekne.config import Config
from tekne.schema import (
    DocMetadata,
    Document,
    Evidence,
    Genre,
    Provenance,
    Span,
    TechMention,
    TechType,
    Verdict,
)
from tekne.verify.verifier import StructuralVerifier

TEXT = (
    "A wheel bearing device with a constant velocity universal joint. "
    "The hub wheel is press-fitted to the outer joint member."
)


def doc(text: str = TEXT) -> Document:
    return Document(
        doc_id="t:1",
        genre=Genre.PATENT,
        text=text,
        metadata=DocMetadata(source="inline", date="2016-01-01"),
    )


def offline(**over) -> Config:
    base = {
        "recall.use_llm": False,
        "verify.use_llm": False,
        "verify.adjudicate_disagreements": False,
        "kb_path": None,
    }
    base.update(over)
    return Config().with_overrides(**base)


def mention(text: str, surface: str, *, evidence: str | None = None, **kwargs) -> TechMention:
    start = text.index(surface)
    span = Span(start=start, end=start + len(surface), surface=surface)
    ev_text = evidence if evidence is not None else text
    ev_start = text.index(ev_text)
    ev = Span(start=ev_start, end=ev_start + len(ev_text), surface=ev_text)
    return TechMention(
        mention_id="m1",
        doc_id="t:1",
        span=span,
        normalized=surface.lower(),
        type=kwargs.pop("type", TechType.ARTIFACT),
        evidence=Evidence(span=ev, sentence_index=0),
        confidence=kwargs.pop("confidence", 0.9),
        provenance=Provenance(
            doc_id="t:1",
            proposers=kwargs.pop("proposers", ("pattern", "gazetteer")),
            content_hash="abc",
        ),
        **kwargs,
    )


def corrupt(m: TechMention, *, shift: int = 3) -> TechMention:
    """A mention whose offsets no longer address its surface.

    Stands in for every way a span can come loose: an off-by-one in a recaller,
    a document re-fetched with different normalisation, or a model that returns
    offsets alongside a fabricated string.
    """
    return m.model_copy(
        update={
            "span": Span(
                start=m.span.start + shift,
                end=m.span.end + shift,
                surface=m.span.surface,
            )
        }
    )


# --- the invariant itself --------------------------------------------------


def test_structural_verifier_rejects_offset_drift():
    d = doc()
    bad = corrupt(mention(TEXT, "hub wheel"))
    assert StructuralVerifier().check(bad, d).verdict is Verdict.REJECT


def test_structural_verifier_rejects_evidence_that_excludes_the_mention():
    d = doc()
    detached = mention(TEXT, "hub wheel", evidence="A wheel bearing device")
    assert StructuralVerifier().check(detached, d).verdict is Verdict.REJECT


# --- the invariant must survive every configuration ------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"guards.verifier": False},
        {"guards.grounding": False},
        {"guards.evidence": False},
        {"guards.provenance": False},
        {"guards.negative_lexicon": False, "guards.consensus": False},
        {
            "guards.grounding": False,
            "guards.evidence": False,
            "guards.verifier": False,
            "guards.negative_lexicon": False,
            "guards.consensus": False,
            "guards.injection": False,
            "guards.temporal": False,
            "guards.provenance": False,
            "output.abstain_below": 0.0,
        },
    ],
    ids=["default", "no-verifier", "no-grounding", "no-evidence", "no-provenance",
         "no-filters", "everything-off"],
)
def test_emitted_spans_are_document_slices_under_every_configuration(overrides):
    """Turning every guard off must not let a mis-addressed span out.

    The guard flags select *filters*; they do not select whether the output is
    allowed to lie about where it came from.
    """
    d = doc()
    result = Pipeline(offline(**overrides)).extract(d)
    for m in result.mentions:
        assert d.text[m.span.start : m.span.end] == m.span.surface
        ev = m.evidence.span
        assert d.text[ev.start : ev.end] == ev.surface
        assert ev.contains(m.span)


def test_pipeline_records_zero_integrity_violations_on_a_clean_run():
    result = Pipeline(offline()).extract(doc())
    assert result.stats["integrity_violations"] == 0


# --- nothing may override the invariant ------------------------------------


def test_adjudicator_cannot_resurrect_a_structural_rejection():
    """A model with a wider window and a strong opinion is still not allowed to
    vouch for a span whose offsets do not address its own surface."""

    class AlwaysAccept:
        available = True
        stats: dict[str, int] = {}

        def adjudicate(self, _mention, _doc, _decision):
            from tekne.verify.adjudicator import Adjudication

            return Adjudication(Verdict.PASS, "looks fine to me")

    d = doc()
    pipeline = Pipeline(offline())
    pipeline.adjudicator = AlwaysAccept()

    from tekne.agents.budget import Budget
    from tekne.agents.orchestrator import Blackboard

    board = Blackboard(document=d)
    board.mentions = [corrupt(mention(TEXT, "hub wheel"))]
    pipeline._stage_verify(board, Budget())

    key = (board.mentions[0].span.start, board.mentions[0].span.end)
    assert board.extras["verifier_decisions"][key].verdict is Verdict.REJECT


def test_corrupted_mention_is_dropped_and_counted_at_emission():
    """The last line of defence: whatever went wrong upstream, a mention that
    fails the structural check does not reach the output, and the fact that one
    was seen is recorded rather than swallowed."""
    d = doc()
    pipeline = Pipeline(offline(**{"guards.verifier": False, "guards.evidence": False}))

    from tekne.agents.orchestrator import Blackboard

    board = Blackboard(document=d)
    board.mentions = [
        mention(TEXT, "hub wheel"),
        corrupt(mention(TEXT, "outer joint member")),
    ]
    kept = pipeline._enforce_integrity(board)

    assert [m.span.surface for m in kept] == ["hub wheel"]
    assert board.extras["integrity_violations"] == 1
    assert any("integrity" in r["reason"] for r in board.rejected)
