"""Normalisation, offset mapping and structural segmentation.

Offsets are the load-bearing abstraction in this system, so most of these tests
are about them rather than about the text that comes out.
"""

from __future__ import annotations

import pytest

from tekne.ingest.normalize import normalize, sentence_spans
from tekne.ingest.segment import segment_patent
from tekne.schema import SectionKind


def test_offsets_map_back_to_raw():
    raw = "The convolu-\ntional  network  (CNN) works."
    norm = normalize(raw)
    assert "convolutional network" in norm.text
    start = norm.text.index("convolutional")
    end = start + len("convolutional network")
    quoted = norm.raw_quote(start, end)
    assert "convolu" in quoted and "network" in quoted


def test_index_map_has_one_entry_per_character():
    raw = "ﬁne-tuning a  model with ligatures"
    norm = normalize(raw)
    assert len(norm.raw_index) == len(norm.text)
    assert all(0 <= i < len(raw) for i in norm.raw_index)


def test_ligatures_expand():
    assert normalize("ﬁeld eﬀect transistor").text == "field effect transistor"


def test_linebreak_hyphen_joins_only_lowercase():
    assert "convolutional" in normalize("convolu-\ntional").text
    # A genuine compound must survive a line break intact.
    assert "end-to-end" in normalize("end-to-end\nlearning").text


def test_citation_masking_preserves_length_and_offsets():
    raw = "We build on BERT [12] and RoBERTa (Liu et al., 2019) here."
    norm = normalize(raw)
    assert "12" not in norm.text
    assert "Liu et al." not in norm.text
    assert "BERT" in norm.text and "RoBERTa" in norm.text
    assert len(norm.raw_index) == len(norm.text)


def test_normalization_is_idempotent_on_its_own_output():
    raw = "A  method—with  ﬂow control\n\nand dashes–here."
    once = normalize(raw).text
    assert normalize(once).text == once


@pytest.mark.parametrize(
    "text,expected",
    [
        ("One sentence only", 1),
        ("First. Second. Third.", 3),
        ("See e.g. Smith. Then more.", 2),
        ("U.S. Pat. No. 5,123,456 is cited. Next sentence.", 2),
        ("Value is 3.14 in this case. Done.", 2),
    ],
)
def test_sentence_splitting(text: str, expected: int):
    assert len(sentence_spans(text)) == expected


def test_sentence_spans_tile_the_text():
    text = "First sentence here. Second one follows! Third? Yes."
    spans = sentence_spans(text)
    for start, end in spans:
        assert text[start:end].strip()
    assert spans[0][0] == 0


# --- patent structure ------------------------------------------------------

CLAIMS = (
    "1. A bearing device comprising an outer member and an inner member. "
    "2. The bearing device of claim 1, wherein the outer member is steel. "
    "3. A method of assembling the bearing device according to claim 2."
)


def test_claim_dependency_detection():
    text = "Wheel bearing " + CLAIMS
    sections = segment_patent(
        text, title_len=len("Wheel bearing"), claims_range=(len("Wheel bearing ") - 1, len(text))
    )
    claims = [s for s in sections if s.kind.value.startswith("claim")]
    assert len(claims) == 3
    assert claims[0].kind is SectionKind.CLAIM_INDEP
    assert claims[1].kind is SectionKind.CLAIM_DEP
    assert claims[1].attrs["depends_on"] == 1
    assert claims[2].kind is SectionKind.CLAIM_DEP
    assert claims[2].attrs["depends_on"] == 2


def test_claim_numbers_are_recorded():
    text = "T " + CLAIMS
    sections = segment_patent(text, title_len=1, claims_range=(2, len(text)))
    numbers = [s.attrs.get("claim_number") for s in sections if "claim" in s.kind.value]
    assert numbers == [1, 2, 3]


def test_sections_do_not_overlap():
    text = "T " + CLAIMS
    sections = sorted(
        segment_patent(text, title_len=1, claims_range=(2, len(text))),
        key=lambda s: s.span.start,
    )
    for earlier, later in zip(sections, sections[1:]):
        assert earlier.span.end <= later.span.start + 1


def test_section_lookup_by_offset():
    from tekne.schema import Document, Genre

    text = "Wheel bearing " + CLAIMS
    sections = segment_patent(
        text, title_len=len("Wheel bearing"), claims_range=(len("Wheel bearing ") - 1, len(text))
    )
    doc = Document(doc_id="p", genre=Genre.PATENT, text=text, sections=sections)
    assert doc.section_at(0).kind is SectionKind.TITLE
    inside_claim_1 = text.index("outer member")
    assert doc.section_at(inside_claim_1).kind is SectionKind.CLAIM_INDEP
