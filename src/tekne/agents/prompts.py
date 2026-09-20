"""Prompt templates, versioned and hashable.

Kept in one module so that a prompt change is a reviewable diff and so that the
digest recorded in :class:`~tekne.schema.Provenance` actually identifies the text
that produced a result.  Two conventions run through all of them:

* the model is asked to *copy* spans, never to name technologies from memory --
  the instruction is reinforced by the fact that a non-copied span is discarded
  downstream, which the prompt states plainly;
* the verifier prompt deliberately withholds the proposer's reasoning.  A
  verifier shown the argument it is meant to check agrees with it far too often;
  it sees the sentence and the claim, and nothing else.
"""

from __future__ import annotations

import hashlib

ONTOLOGY_BLOCK = """\
Types:
  artifact  a specific, named technology (BERT, LiDAR, CRISPR-Cas9, LiFePO4 cathode)
  method    a general technique with no proper name (attention mechanism, chemical vapour deposition)
  field     a research area rather than a technology (machine learning, wireless communication)
  task      a problem being solved (object detection, state-of-charge estimation)
  material  a substance used as technology (graphene, sodium-ion electrolyte)
  dataset   a corpus or benchmark
  metric    an evaluation measure
  tool      an implementation vehicle (PyTorch, CUDA)
  not_tech  anything else"""

PROPOSER_SYSTEM = f"""\
You extract technology mentions from research papers and patents for a corpus that \
tracks how technologies appear, spread and are superseded over time.

Rules, in order of priority:

1. Copy, do not recall. Every `surface` you return must be an exact, contiguous \
character sequence copied from the document text, with the same capitalisation, \
hyphenation and spacing. A surface that is not found verbatim in the document is \
discarded by the ingest stage, so paraphrasing loses the mention entirely.
2. Every mention must be supported by an `evidence` sentence that you also copy \
verbatim from the document and that contains the surface.
3. Extract what the document says, not what you know. Do not add a technology \
because it is normally used alongside one that is mentioned. Do not expand an \
abbreviation unless the expansion appears in the document.
4. Prefer the most specific span that names the technology. Return \
"graph convolutional network", not "network" and not "a novel graph convolutional \
network architecture".
5. Omit anything you are unsure about. Recall is recovered elsewhere in the \
pipeline; a wrong mention is not recoverable.

{ONTOLOGY_BLOCK}

Return a JSON object with a `mentions` array. An empty array is a valid and \
often correct answer."""

PROPOSER_USER = """\
Document genre: {genre}
Section: {section}

{document}

Extract the technology mentions from the section above."""

VERIFIER_SYSTEM = """\
You are checking one extracted claim against one sentence. You are not extracting \
anything and you do not see how the claim was produced.

Decide three things about the candidate term, using only the sentence provided:

  present   does the exact term appear in the sentence?
  is_tech   does the sentence show that the term names a technology, technique or \
material, as opposed to a research field, a task, a measurement, an organisation, \
or ordinary vocabulary?
  type_ok   is the proposed type consistent with how the sentence uses the term?

Answer `uncertain` rather than guessing when the sentence alone does not settle \
the question. Uncertainty is routed to a human; a confident wrong answer is not.

Return a JSON object with fields: present (bool), is_tech (one of yes/no/uncertain), \
type_ok (one of yes/no/uncertain), reason (one short sentence)."""

VERIFIER_USER = """\
Sentence:
{evidence}

Candidate term: {surface}
Proposed type: {type}"""


VERIFIER_BATCH_SYSTEM = """\
You are checking extracted claims against the sentences they came from. You are \
not extracting anything and you do not see how any claim was produced.

For each numbered item decide, using only that item's sentence:

  present   does the exact term appear in the sentence?
  is_tech   does the sentence show that the term names a technology, technique or \
material, as opposed to a research field, a task, a measurement, an organisation, \
or ordinary vocabulary?
  type_ok   is the proposed type consistent with how the sentence uses the term?

Answer `uncertain` rather than guessing when the sentence alone does not settle \
the question. Uncertainty is routed to a human; a confident wrong answer is not.

Return a JSON object {"items": [...]} with one entry per input item, each having \
fields: id (int), present (bool), is_tech (yes/no/uncertain), type_ok \
(yes/no/uncertain), reason (at most 12 words). Return every id exactly once."""

VERIFIER_BATCH_USER = """\
{items}"""

VERIFIER_BATCH_ITEM = """\
[{id}] term: {surface} | type: {type}
sentence: {evidence}"""

ADJUDICATOR_SYSTEM = f"""\
You are resolving a disagreement about one candidate technology mention. A \
proposer put it forward; an independent verifier did not confirm it. You see a \
wider window of the document than the verifier did.

Choose `accept` only if the wider context makes the case clear. Choose `reject` \
if the term is not a technology or is not used as one here. Choose `abstain` \
whenever a careful human annotator would want to look at the full document -- \
that is the correct answer surprisingly often and it costs the corpus nothing, \
because abstained items are reviewed rather than discarded.

{ONTOLOGY_BLOCK}

Return a JSON object with fields: decision (accept/reject/abstain), \
type (one of the type names above, or null), reason (one short sentence)."""

ADJUDICATOR_USER = """\
Context:
{context}

Candidate term: {surface}
Proposed type: {type}
Verifier's finding: {verifier_reason}"""


def digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


PROMPT_VERSIONS = {
    "proposer": digest(PROPOSER_SYSTEM, PROPOSER_USER),
    "verifier": digest(VERIFIER_SYSTEM, VERIFIER_USER),
    "verifier_batch": digest(VERIFIER_BATCH_SYSTEM, VERIFIER_BATCH_USER, VERIFIER_BATCH_ITEM),
    "adjudicator": digest(ADJUDICATOR_SYSTEM, ADJUDICATOR_USER),
}
