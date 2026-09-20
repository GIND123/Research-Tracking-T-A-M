"""Trend aggregation.

Built on synthetic extraction results so the properties under test are the
aggregation rules themselves -- document-frequency counting, corpus-size
normalisation, adoption signature -- rather than the extractor's behaviour on
any particular corpus.
"""

from __future__ import annotations

from tekne.schema import (
    DocMetadata,
    Document,
    Evidence,
    ExtractionResult,
    Genre,
    Provenance,
    Role,
    Span,
    TechMention,
    TechType,
)
from tekne.track.trends import build_trends, cooccurrence, emerging, role_transitions


def make(doc_id: str, year: int, mentions: list[tuple[str, Role]], *, repeats: int = 1):
    """One document and its extraction, with each mention repeated ``repeats`` times."""
    text = " ".join(surface for surface, _role in mentions for _ in range(repeats)) or "empty"
    doc = Document(
        doc_id=doc_id,
        genre=Genre.PAPER,
        text=text,
        metadata=DocMetadata(source="inline", date=f"{year}-06-01"),
    )
    out: list[TechMention] = []
    cursor = 0
    for surface, role in mentions:
        for _ in range(repeats):
            start = text.index(surface, cursor)
            end = start + len(surface)
            cursor = end
            span = Span(start=start, end=end, surface=surface)
            out.append(
                TechMention(
                    mention_id=f"{doc_id}:{start}",
                    doc_id=doc_id,
                    span=span,
                    normalized=surface.lower(),
                    type=TechType.ARTIFACT,
                    role=role,
                    evidence=Evidence(span=Span(start=0, end=len(text), surface=text)),
                    confidence=0.9,
                    canonical_id=f"tekne:{surface.lower()}",
                    provenance=Provenance(doc_id=doc_id, proposers=("pattern",), content_hash="h"),
                )
            )
    return doc, ExtractionResult(doc_id=doc_id, genre=Genre.PAPER, mentions=out)


def test_counting_is_by_document_not_mention():
    """A patent repeating a component forty times is one document of evidence."""
    doc_a, res_a = make("d1", 2020, [("alpha", Role.USED)], repeats=40)
    doc_b, res_b = make("d2", 2020, [("alpha", Role.USED)], repeats=1)
    table = build_trends([res_a, res_b], [doc_a, doc_b])
    assert table.series["tekne:alpha"].doc_counts[2020] == 2


def test_share_normalises_by_corpus_size():
    docs, results = [], []
    for i in range(2):
        d, r = make(f"a{i}", 2020, [("alpha", Role.USED)])
        docs.append(d)
        results.append(r)
    for i in range(8):
        d, r = make(f"b{i}", 2021, [("alpha", Role.USED)] if i < 4 else [("beta", Role.USED)])
        docs.append(d)
        results.append(r)

    table = build_trends(results, docs)
    shares = table.series["tekne:alpha"].share(table.year_totals)
    # Present in every 2020 document, half of 2021's -- a raw count would call
    # that growth (2 -> 4); the share correctly calls it decline.
    assert shares[2020] == 1.0
    assert shares[2021] == 0.5


def test_emerging_ranks_a_newcomer_above_an_incumbent():
    docs, results = [], []
    for year in (2016, 2017, 2018, 2019, 2020, 2021):
        for i in range(5):
            mentions = [("incumbent", Role.USED)]
            if year >= 2020:
                mentions.append(("newcomer", Role.PROPOSED))
            d, r = make(f"{year}-{i}", year, mentions)
            docs.append(d)
            results.append(r)

    table = build_trends(results, docs)
    ranked = emerging(table, window=2, min_docs=3)
    assert ranked, "expected at least one emerging technology"
    assert ranked[0].label == "newcomer"
    assert ranked[0].first_year == 2020
    assert ranked[0].growth > 1.0


def test_adoption_shift_detects_proposed_to_used():
    docs, results = [], []
    for year, role in [(2017, Role.PROPOSED), (2018, Role.PROPOSED),
                       (2019, Role.USED), (2020, Role.USED)]:
        for i in range(3):
            d, r = make(f"{year}-{i}", year, [("gadget", role)])
            docs.append(d)
            results.append(r)

    table = build_trends(results, docs)
    rows = {row["label"]: row for row in role_transitions(table, min_docs=4)}
    assert rows["gadget"]["early_proposed"] == 1.0
    assert rows["gadget"]["late_used"] == 1.0
    assert rows["gadget"]["adoption_shift"] > 0


def test_undated_documents_are_counted_not_dropped_silently():
    doc, res = make("d1", 2020, [("alpha", Role.USED)])
    doc = doc.model_copy(update={"metadata": DocMetadata(source="inline", date=None)})
    table = build_trends([res], [doc])
    assert table.undated == 1
    assert table.series == {}


def test_low_confidence_mentions_are_excluded():
    doc, res = make("d1", 2020, [("alpha", Role.USED)])
    res.mentions = [m.model_copy(update={"confidence": 0.1}) for m in res.mentions]
    table = build_trends([res], [doc], min_confidence=0.5)
    assert table.series == {}


def test_cooccurrence_pairs_technologies_that_travel_together():
    docs, results = [], []
    for i in range(6):
        d, r = make(f"d{i}", 2020, [("alpha", Role.USED), ("beta", Role.USED)])
        docs.append(d)
        results.append(r)
    for i in range(6):
        d, r = make(f"e{i}", 2020, [("gamma", Role.USED)])
        docs.append(d)
        results.append(r)

    pairs = cooccurrence(results, min_docs=3)
    assert pairs
    top_a, top_b, pmi = pairs[0]
    assert {top_a, top_b} == {"tekne:alpha", "tekne:beta"}
    assert pmi > 0
