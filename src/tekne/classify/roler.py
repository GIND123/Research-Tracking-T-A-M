"""Stance classification: what the document does with the technology it names.

A mention count is a weak evolution signal.  "BERT" appearing in a 2023 paper
says almost nothing; "BERT" appearing as a *baseline* in a 2023 paper and as the
*proposed contribution* in a 2019 one is the technology's life cycle, visible in
two data points.  Role is therefore not a decoration on the extraction task, it
is most of the reason to do the extraction carefully.

Evidence combined here, in rough order of reliability:

* the structural zone (a term in an independent claim is claimed, full stop);
* explicit cues in a window around the mention, from the curated lexicon;
* the zone's prior over roles, which carries the decision when no cue fires --
  which is the common case in patents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..lexicons import role_cues
from ..schema import Role, SectionKind

_ROLE_BY_NAME = {
    "proposed": Role.PROPOSED,
    "used": Role.USED,
    "compared": Role.COMPARED,
    "background": Role.BACKGROUND,
}


@dataclass
class RolePrediction:
    role: Role
    score: float
    margin: float
    fired_cues: list[str] = field(default_factory=list)
    scores: dict[Role, float] = field(default_factory=dict)


class RoleClassifier:
    version = "2"

    def __init__(self) -> None:
        cues = role_cues()
        self.window = int(cues.get("window", 140))
        self.cues = {
            name: [(phrase.lower(), float(weight)) for phrase, weight in entries]
            for name, entries in cues["cues"].items()
        }
        self.section_prior = cues["section_prior"]

    def predict(
        self,
        text: str,
        start: int,
        end: int,
        section: SectionKind,
        *,
        evidence: tuple[int, int] | None = None,
    ) -> RolePrediction:
        # Patent claims are unambiguous: the claim defines the monopoly, so a
        # technology named inside an independent claim is claimed by definition.
        if section is SectionKind.CLAIM_INDEP:
            return RolePrediction(Role.CLAIMED, 1.0, 1.0, ["section:claim_independent"])

        lo = max(0, start - self.window)
        hi = min(len(text), end + self.window)
        if evidence is not None:
            lo = min(lo, evidence[0])
            hi = max(hi, evidence[1])
        context = text[lo:hi].lower()
        mention = text[start:end].lower()

        prior = self.section_prior.get(section.value, {})
        scores = {name: float(prior.get(name, 0.0)) for name in self.cues}
        fired: list[str] = []

        for name, entries in self.cues.items():
            for phrase, weight in entries:
                if phrase not in context:
                    continue
                # A cue that is part of the mention itself is not evidence about
                # the mention ("the proposed method" as a candidate surface).
                if phrase in mention:
                    continue
                distance = _cue_distance(context, phrase, start - lo, end - lo)
                scores[name] += weight * _decay(distance, self.window)
                fired.append(f"{name}:{phrase}")

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_name, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - runner_up

        # No cue fired and the zone prior is flat: say so rather than guess.
        if not fired and abs(margin) < 0.35:
            return RolePrediction(Role.UNKNOWN, 0.0, margin, fired, _as_roles(scores))

        probs = _softmax([s for _n, s in ranked])
        return RolePrediction(
            role=_ROLE_BY_NAME[best_name],
            score=probs[0],
            margin=margin,
            fired_cues=fired[:8],
            scores=_as_roles(scores),
        )


def _cue_distance(context: str, phrase: str, mention_start: int, mention_end: int) -> int:
    best = 10**6
    pos = context.find(phrase)
    while pos != -1:
        if pos + len(phrase) <= mention_start:
            distance = mention_start - (pos + len(phrase))
        elif pos >= mention_end:
            distance = pos - mention_end
        else:
            distance = 0
        best = min(best, distance)
        pos = context.find(phrase, pos + 1)
    return best


def _decay(distance: int, window: int) -> float:
    """Linear-ish decay; a cue at the window edge counts about a third."""
    if distance >= window:
        return 0.25
    return 1.0 - 0.75 * (distance / max(window, 1))


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _as_roles(scores: dict[str, float]) -> dict[Role, float]:
    return {_ROLE_BY_NAME[name]: value for name, value in scores.items() if name in _ROLE_BY_NAME}
