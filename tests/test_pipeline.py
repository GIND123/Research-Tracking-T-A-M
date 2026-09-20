"""Schema invariants, guards, canonicalisation, metrics and the gold file."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tekne.calib.selective import (
    ConfidenceInputs,
    ConfidenceModel,
    choose_threshold,
    expected_calibration_error,
    risk_coverage,
)
from tekne.config import Config
from tekne.eval.gold import GoldDocument, GoldMention, read_gold, validate_gold
from tekne.eval.metrics import align, evaluate
from tekne.guard.base import GuardContext, GuardStack
from tekne.guard.checks import ConsensusGuard, EvidenceGuard, TemporalGuard
from tekne.guard.grounding import SpanGroundingGuard
from tekne.guard.negative import NegativeLexiconGuard
from tekne.lexicons import lemma_key
from tekne.link.canonical import Canonicalizer
from tekne.schema import (
    Candidate,
    DocMetadata,
    Document,
    Evidence,
    Genre,
    KBLink,
    Provenance,
    Role,
    Span,
    TechMention,
    TechType,
    Verdict,
)

TEXT = "A wheel bearing device with a constant velocity universal joint and a hub wheel."


def doc(text: str = TEXT, **kwargs) -> Document:
    return Document(
        doc_id="t:1",
        genre=Genre.PATENT,
        text=text,
        metadata=DocMetadata(source="inline", date=kwargs.pop("date", "2016-01-01")),
        **kwargs,
    )


def span_for(text: str, needle: str) -> Span:
    start = text.index(needle)
    return Span(start=start, end=start + len(needle), surface=needle)


# --- schema ----------------------------------------------------------------


def test_span_rejects_surface_mismatch():
    with pytest.raises(ValidationError):
        Span(start=0, end=5, surface="too long a surface")


def test_span_rejects_inverted_interval():
    with pytest.raises(ValidationError):
        Span(start=10, end=3, surface="")


def test_span_containment_and_overlap():
    outer = Span(start=0, end=10, surface="0123456789")
    inner = Span(start=2, end=5, surface="234")
    assert outer.contains(inner) and outer.overlaps(inner)
    assert not inner.contains(outer)


def test_lemma_key_normalisation():
    assert lemma_key("The Neural Networks") == "neural network"
    assert lemma_key("analyses") == "analysis"
    assert lemma_key("self-attention") == "self-attention"
    # A trailing "ss"/"us"/"is" is not a plural.
    assert lemma_key("bias") == "bias"


# --- guards ----------------------------------------------------------------


def test_grounding_guard_rejects_offset_drift():
    d = doc()
    bad = Candidate(span=Span(start=2, end=7, surface="wrong"), proposers=("x",))
    verdict = SpanGroundingGuard().check(bad, GuardContext(document=d))
    assert verdict.verdict is Verdict.REJECT


def test_grounding_guard_accepts_real_span():
    d = doc()
    good = Candidate(span=span_for(TEXT, "hub wheel"), proposers=("x",))
    assert SpanGroundingGuard().check(good, GuardContext(document=d)).verdict is Verdict.PASS


@pytest.mark.parametrize(
    "surface",
    ["the present invention", "apparatus", "a plurality", "figure", "model", "1.4 mm", "of the"],
)
def test_negative_lexicon_rejects(surface: str):
    text = f"Some text with {surface} inside it."
    d = doc(text)
    cand = Candidate(span=span_for(text, surface), proposers=("pattern",))
    assert NegativeLexiconGuard().check(cand, GuardContext(document=d)).verdict is Verdict.REJECT


@pytest.mark.parametrize(
    "surface",
    ["heat exchanger apparatus", "graph neural network", "hub wheel", "lithium iron phosphate"],
)
def test_negative_lexicon_keeps_modified_terms(surface: str):
    text = f"Some text with {surface} inside it."
    d = doc(text)
    cand = Candidate(span=span_for(text, surface), proposers=("pattern",))
    assert NegativeLexiconGuard().check(cand, GuardContext(document=d)).verdict is Verdict.PASS


def test_consensus_guard_trusts_deterministic_recallers_alone():
    d = doc()
    cand = Candidate(span=span_for(TEXT, "hub wheel"), proposers=("gazetteer",))
    assert ConsensusGuard().check(cand, GuardContext(document=d)).verdict is Verdict.PASS

    lonely = Candidate(span=span_for(TEXT, "hub wheel"), proposers=("pattern",))
    assert ConsensusGuard().check(lonely, GuardContext(document=d)).verdict is Verdict.ABSTAIN


def make_mention(text: str, surface: str, *, evidence: str | None = None, **kwargs) -> TechMention:
    span = span_for(text, surface)
    ev_text = evidence if evidence is not None else text
    ev = span_for(text, ev_text)
    return TechMention(
        mention_id="m1",
        doc_id="t:1",
        span=span,
        normalized=lemma_key(surface),
        type=kwargs.pop("type", TechType.ARTIFACT),
        role=kwargs.pop("role", Role.CLAIMED),
        evidence=Evidence(span=ev, sentence_index=0),
        provenance=Provenance(doc_id="t:1", proposers=("pattern", "gazetteer"), content_hash="abc"),
        **kwargs,
    )


def test_evidence_guard_requires_containment():
    d = doc()
    good = make_mention(TEXT, "hub wheel")
    assert EvidenceGuard().check(good, GuardContext(document=d)).verdict is Verdict.PASS

    detached = make_mention(TEXT, "hub wheel", evidence="A wheel bearing device")
    assert EvidenceGuard().check(detached, GuardContext(document=d)).verdict is Verdict.REJECT


def test_temporal_guard_abstains_on_anachronism():
    d = doc(date="2005-01-01")
    mention = make_mention(
        TEXT,
        "hub wheel",
        kb_link=KBLink(kb="cso", entry_id="cso:x", label="x", attested_from=2017),
    )
    assert TemporalGuard().check(mention, GuardContext(document=d)).verdict is Verdict.ABSTAIN

    contemporary = make_mention(
        TEXT,
        "hub wheel",
        kb_link=KBLink(kb="cso", entry_id="cso:x", label="x", attested_from=2001),
    )
    assert TemporalGuard().check(contemporary, GuardContext(document=d)).verdict is Verdict.PASS


def test_guard_stack_short_circuits_on_reject():
    class Boom(NegativeLexiconGuard):
        name = "boom"

        def check(self, item, ctx):  # pragma: no cover - must not run
            raise AssertionError("guard after a REJECT should not run")

    d = doc()
    stack = GuardStack([SpanGroundingGuard(), Boom()])
    bad = Candidate(span=Span(start=2, end=7, surface="wrong"), proposers=("x",))
    assert stack.run(bad, GuardContext(document=d)).verdict is Verdict.REJECT


# --- canonicalisation ------------------------------------------------------


def test_abbreviation_pairs_merge_short_and_long_forms():
    from tekne.recall.abbrev import AbbreviationPair

    text = "a convolutional neural network (CNN) and the CNN again"
    canon = Canonicalizer()
    canon.add_abbreviation_pairs(
        [
            AbbreviationPair(
                short_start=0, short_end=3, short="CNN",
                long_start=0, long_end=0, long="convolutional neural network",
            )
        ]
    )
    long_m = make_mention(text, "convolutional neural network")
    short_m = make_mention(text, "CNN")
    canon.fit([long_m, short_m], {"t:1": 2016})
    assert canon.assign(long_m).canonical_id == canon.assign(short_m).canonical_id


def test_nil_cluster_gets_a_stable_synthetic_id():
    text = "a spiking neural network and another spiking neural network"
    canon = Canonicalizer()
    m = make_mention(text, "spiking neural network")
    canon.fit([m], {"t:1": 2024})
    assigned = canon.assign(m)
    assert assigned.canonical_id.startswith("tekne:")
    assert canon.assign(m).canonical_id == assigned.canonical_id


# --- metrics and calibration ----------------------------------------------


def test_alignment_is_one_to_one():
    text = TEXT
    predicted = [make_mention(text, "hub wheel"), make_mention(text, "bearing device")]
    gold = [
        GoldMention(**_gm(text, "hub wheel")),
        GoldMention(**_gm(text, "bearing device")),
    ]
    pairs, extra_sys, extra_gold = align(predicted, gold, strict=True)
    assert len(pairs) == 2 and not extra_sys and not extra_gold


def _gm(text: str, surface: str) -> dict:
    start = text.index(surface)
    return {
        "start": start,
        "end": start + len(surface),
        "surface": surface,
        "type": TechType.ARTIFACT,
        "role": Role.CLAIMED,
    }


def test_evaluate_counts_a_perfect_system_correctly():
    text = TEXT
    d = doc(text)
    gold = {
        "t:1": GoldDocument(
            doc_id="t:1",
            genre=Genre.PATENT,
            zones=[(0, len(text))],
            mentions=[GoldMention(**_gm(text, "hub wheel"))],
        )
    }
    system = {"t:1": [make_mention(text, "hub wheel")]}
    report = evaluate(system, gold, {"t:1": d})
    assert report.strict.f1 == 1.0
    assert report.hallucination_rate == 0.0


def test_risk_coverage_rewards_ranking_errors_last():
    good = risk_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    bad = risk_coverage([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1])
    assert good.aurc < bad.aurc
    assert good.coverage_at_precision(1.0) == pytest.approx(0.5)


def test_choose_threshold_meets_the_precision_target():
    curve = risk_coverage([0.95, 0.9, 0.6, 0.4, 0.1], [1, 1, 1, 0, 0])
    threshold = choose_threshold(curve, target_precision=0.9)
    kept = [s for s in [0.95, 0.9, 0.6, 0.4, 0.1] if s >= threshold]
    assert len(kept) == 3


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    scores = [0.05] * 20 + [0.95] * 20
    labels = [0] * 19 + [1] + [1] * 19 + [0]
    assert expected_calibration_error(scores, labels) < 0.02


def test_confidence_ordering_is_sensible():
    model = ConfidenceModel()
    strong = ConfidenceInputs(
        n_proposers=3, trusted_proposer=True, kb_hit=True, type_prob=0.9,
        verifier_passed=True, verifier_score=1.0, section_weight=0.9,
    )
    weak = ConfidenceInputs(n_proposers=1, type_prob=0.3, verifier_abstained=True, guard_flags=2)
    assert model.score(strong) > 0.9 > model.score(weak)


# --- configuration ---------------------------------------------------------


def test_config_digest_changes_with_the_configuration():
    base = Config()
    changed = base.with_overrides(**{"guards.consensus": False})
    assert base.digest() != changed.digest()
    assert Config().digest() == base.digest()


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        Config.from_dict({"guards": {"not_a_real_guard": True}})


# --- gold file -------------------------------------------------------------


def test_gold_file_matches_the_corpus():
    """Guards against a re-fetch or a normalisation change silently shifting the
    gold offsets, which would produce a confidently wrong evaluation."""
    from pathlib import Path

    from tekne.ingest.sources import read_jsonl

    gold_path = Path("data/gold/gold.jsonl")
    if not gold_path.is_file():
        pytest.skip("gold not built; run scripts/make_gold.py")

    documents = {}
    for name in ("papers_eval.jsonl", "patents_eval.jsonl"):
        path = Path("data/raw") / name
        if path.is_file():
            documents.update({d.doc_id: d for d in read_jsonl(path)})
    if not documents:
        pytest.skip("corpora not fetched; run scripts/fetch_data.py")

    problems = validate_gold(read_gold(gold_path), documents)
    assert problems == [], problems[:5]


def test_shipped_default_yaml_matches_the_dataclass_defaults():
    """`Config()` and configs/default.yaml must describe the same pipeline.

    They drifted once (the LLM verifier was on in the YAML and off in the code),
    which made `tekne plan` price a configuration nobody would ever run.
    """
    from pathlib import Path

    path = Path("configs/default.yaml")
    if not path.is_file():
        pytest.skip("configs/default.yaml not present")

    code = Config().to_dict()
    shipped = Config.load(path).to_dict()
    differences = {
        f"{section}.{key}": (code[section][key], shipped[section][key])
        for section in code
        if isinstance(code[section], dict)
        for key in code[section]
        # YAML cannot express a tuple; compare sequences by content.
        if list(_seq(code[section][key])) != list(_seq(shipped[section][key]))
    }
    assert differences == {}, differences


def _seq(value):
    return value if isinstance(value, (list, tuple)) else [value]


def test_no_gold_row_names_something_the_negative_lexicon_rejects():
    """The guideline and the lexicon must agree.

    A gold mention that the negative lexicon rejects is unreachable by
    construction, so it silently caps recall and makes the lexicon ablation look
    better than it is. Three such rows existed in an early version.
    """
    import csv
    from pathlib import Path

    from tekne.lexicons import negative_lexicon

    path = Path("data/gold/annotations.tsv")
    if not path.is_file():
        pytest.skip("annotation table not present")

    negative = negative_lexicon()
    offenders = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or row[0].startswith("#") or len(row) < 4:
                continue
            surface = row[1].strip()
            if lemma_key(surface) in negative or surface.lower() in negative:
                offenders.append((row[0].strip(), surface))
    assert offenders == [], offenders
