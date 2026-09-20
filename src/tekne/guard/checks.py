"""The remaining guards: evidence, consensus, temporal consistency, provenance.

Grouped in one module because each is short and they share no state; splitting
them into four files would be filing, not structure.
"""

from __future__ import annotations

from ..schema import GuardVerdict, Verdict
from .base import Guard, GuardContext


class EvidenceGuard(Guard):
    """Every mention must cite a sentence from the document that contains it.

    This is the second structural constraint after grounding, and it is what
    makes a human audit cheap: the reviewer is shown a sentence, not a term in
    isolation, and the sentence is guaranteed to be the one the decision was made
    from rather than a plausible-looking reconstruction.
    """

    name = "evidence"
    stage = "mention"

    def check(self, item, ctx: GuardContext) -> GuardVerdict:
        evidence = getattr(item, "evidence", None)
        span = getattr(item, "span", None)
        if evidence is None or span is None:
            return self._reject("missing evidence or span")
        text = ctx.document.text
        ev = evidence.span
        if ev.end > len(text) or text[ev.start : ev.end] != ev.surface:
            return self._reject("evidence span does not match document")
        if not ev.contains(span):
            return self._reject(
                f"evidence [{ev.start},{ev.end}) does not contain mention "
                f"[{span.start},{span.end})"
            )
        if span.surface not in ev.surface:
            return self._reject("mention surface absent from evidence sentence")
        return self._pass()


class ConsensusGuard(Guard):
    """Require independent agreement, unless a trusted recaller fired alone.

    The abbreviation miner and the gazetteer are deterministic and near-exact, so
    a solo hit from either is accepted.  Everything else needs a second opinion:
    a span proposed only by the chunker is precisely the population where the
    chunker's errors live.
    """

    name = "consensus"
    stage = "candidate"

    TRUSTED_SOLO = frozenset({"abbrev", "gazetteer"})

    def __init__(self, *, min_proposers: int = 2, enabled: bool = True) -> None:
        self.min_proposers = min_proposers
        self.enabled = enabled

    def check(self, item, ctx: GuardContext) -> GuardVerdict:
        if not self.enabled:
            return self._pass("disabled")
        proposers = set(getattr(item, "proposers", ()) or ())
        if proposers & self.TRUSTED_SOLO:
            return self._pass(f"trusted solo: {sorted(proposers & self.TRUSTED_SOLO)}")
        if len(proposers) >= self.min_proposers:
            return self._pass(f"{len(proposers)} proposers")
        return self._abstain(
            f"only {len(proposers)} proposer(s): {sorted(proposers)}", score=float(len(proposers))
        )


class TemporalGuard(Guard):
    """Catch anachronistic knowledge-base links.

    The failure this prevents is specific to technology tracking and easy to
    miss.  Vocabulary is reused across eras: "transformer" in a 2005 power-systems
    patent is a magnetic component, and linking it to the 2017 neural
    architecture back-dates that architecture by twelve years.  A single such
    error is not noise -- it lands at the start of the time series, exactly where
    an emergence detector is most sensitive.

    We compare the document's date against the earliest attestation the KB or the
    corpus itself records, and abstain on a link that predates it.  The mention
    survives; only the link is withheld.
    """

    name = "temporal"
    stage = "mention"

    def __init__(self, *, slack_years: int = 1, enabled: bool = True) -> None:
        self.slack_years = slack_years
        self.enabled = enabled

    def check(self, item, ctx: GuardContext) -> GuardVerdict:
        if not self.enabled:
            return self._pass("disabled")
        link = getattr(item, "kb_link", None)
        if link is None or link.attested_from is None:
            return self._pass("no dated link")
        year = ctx.document.year()
        if year is None:
            return self._pass("document undated")
        if year + self.slack_years < link.attested_from:
            return self._abstain(
                f"document year {year} precedes first attestation "
                f"{link.attested_from} of {link.entry_id}"
            )
        return self._pass()


class ProvenanceGuard(Guard):
    """Refuse to emit a record that cannot be traced back.

    Last in the stack and deliberately paranoid.  If a refactor ever drops the
    provenance plumbing, this turns a silent loss of auditability into a visible
    rejection.
    """

    name = "provenance"
    stage = "mention"

    REQUIRED = ("doc_id", "proposers", "content_hash")

    def check(self, item, ctx: GuardContext) -> GuardVerdict:
        prov = getattr(item, "provenance", None)
        if prov is None:
            return self._reject("no provenance record")
        for field in self.REQUIRED:
            value = getattr(prov, field, None)
            if not value:
                return self._reject(f"provenance missing {field}")
        if prov.doc_id != ctx.document.doc_id:
            return self._reject(f"provenance doc_id {prov.doc_id} != {ctx.document.doc_id}")
        return self._pass()


class VerifierGuard(Guard):
    """Carry the adjudicated verifier decision into the guard trace.

    The verification itself happens in :mod:`tekne.verify`; this guard is the
    point at which its verdict becomes binding, so that every reason a mention
    was withheld appears in one place.
    """

    name = "verifier"
    stage = "mention"

    def check(self, item, ctx: GuardContext) -> GuardVerdict:
        decision = ctx.extras.get("verifier_decisions", {}).get(_key(item))
        if decision is None:
            return self._pass("not verified")
        if decision.verdict is Verdict.PASS:
            return self._pass(decision.reason, score=decision.score)
        if decision.verdict is Verdict.REJECT:
            return self._reject(decision.reason, score=decision.score)
        return self._abstain(decision.reason, score=decision.score)


def _key(item) -> tuple[int, int]:
    span = getattr(item, "span", None)
    return (span.start, span.end) if span else (-1, -1)
