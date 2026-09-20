# Annotation guidelines

These are the rules the gold set in `data/gold/annotations.tsv` was built under.
They are written to be arguable: where a rule makes a call that could reasonably
go the other way, it says so, because a guideline that pretends the boundary is
obvious produces a gold set that silently encodes one annotator's habits.

## What gets annotated

A **technology mention** is a span of text that names a specific means of doing
something — an artefact, a technique, or a material used as one.

The unit of annotation is the **technology per document**, not the occurrence. A
row in the table asserts "this document treats X as a technology"; the expander
in `scripts/make_gold.py` anchors that to every occurrence inside the evaluation
zone. This is a deliberate choice: deciding per-occurrence whether the
fourteenth "grinding tool" is a technology mention is not a judgement anyone
makes consistently, and the disagreement it produces is noise rather than signal.

### Evaluation zone

| Genre  | Zone |
|--------|------|
| Paper  | title + abstract |
| Patent | title + abstract + the first independent claim (capped at 1400 characters) |

Everything outside the zone is neither annotated nor scored. The claim cap exists
because independent claims in chemical cases are Markush structures — thousands
of characters of variable definitions (`RA2 is selected from the group consisting
of ...`) that name no technology. Annotating them exhaustively would spend most
of the budget on text that contributes nothing to a technology trend.

## Types

| Type | Test | Examples |
|------|------|----------|
| `artifact` | A specific, nameable thing. Has a proper name, or a configuration specific enough to be re-identified. | `SwitchRoute`, `LiDAR sensor`, `Talbot-Lau interferometer`, `hub wheel`, `constant velocity universal joint` |
| `method` | A general technique with no proper name. | `chemical vapour deposition`, `on-policy distillation`, `press fitting`, `spread spectrum` |
| `material` | A substance used as a technology. | `graphene`, `carbon nanotube dispersion liquid`, `Ziegler-Natta catalyst` |
| `task` | The problem being solved, not the means. | `object detection`, `support recovery`, `melanoma` |
| `field` | A research area. | `machine learning`, `magnonics`, `signal processing` |
| `dataset` | A corpus or benchmark. | `SWE-Bench Verified`, `ALFWorld`, `Paint-500K` |
| `metric` | An evaluation measure. | `BLEU`, `coulombic efficiency`, `specific surface area` |
| `tool` | An implementation vehicle rather than a contribution. | `PyTorch`, `bash`, `TikTok` |

Only `artifact`, `method` and `material` are scored by default. The others are
annotated anyway, because the type-confusion table is where the interesting
errors are: a system that calls `machine learning` an artefact will report it as
the fastest-growing technology of every year in the corpus.

**The hard boundary is `method` vs `task`.** Test: could you *use* it, or do you
*solve* it? You use chemical vapour deposition; you solve object detection.
Nominalisations are ambiguous by construction (`segmentation` is both), and the
rule is to follow the document's own framing.

**The second hard boundary is `artifact` vs `method`.** Test: does the document
treat it as a particular thing, or as a way of doing things? `graph neural
network` is an artefact (you can point at one); `knowledge distillation` is a
method. Named things are almost always artefacts; this is the type that
orthography predicts well.

## Span boundaries

Annotate the **minimal span that names the technology**.

- Include discriminating modifiers: `graph convolutional network`, not `network`.
- Exclude determiners, quantifiers and evaluative adjectives: `string
  similarity`, not `superficial string similarity`; `coding harness`, not
  `lightweight coding harness`.
- Exclude the head noun when it is a bare category word appended to a real term:
  `chemical vapour deposition`, not `chemical vapour deposition process`.
- For patents, include the component qualifier: `rear cross member`, not
  `member`.

Boundary choice is genuinely ambiguous and we do not pretend otherwise. This is
why the evaluation reports both strict (exact offsets) and partial (overlap)
matching, and why the partial figure is the more meaningful one for comparing
systems. A large strict/partial gap means boundary disagreement, not failure to
find the technology.

## Roles

| Role | Meaning |
|------|---------|
| `proposed` | The document's own contribution. |
| `used` | Employed as a component or tool. |
| `compared` | A baseline or prior system contrasted against. |
| `background` | Cited context, no commitment. |
| `claimed` | Patent-specific: inside an independent claim. |
| `unknown` | No clear stance. |

Because a row covers a whole document, the role recorded is the **strongest
stance** the document takes, ordered `claimed > proposed > compared > used >
background`. In practice this makes patent annotations overwhelmingly `claimed`
(566 of 908 gold mentions), which is correct — a term in an independent claim is
being monopolised regardless of how the abstract also uses it — but it does mean
role accuracy on patents measures something closer to zone detection than to
stance classification. Per-occurrence role annotation would be the right fix and
is left to future work.

## Things that are deliberately not annotated

- Organisation, product-company and author names (`Google`, `Devlin et al.`).
- Units, quantities and reference numerals (`1.4 mm`, `(10)`).
- Patent drafting boilerplate (`the present invention`, `an exemplary
  embodiment`, `one skilled in the art`).
- Bare category nouns (`apparatus`, `system`, `method`, `model`) when unmodified.
- Section furniture (`Figure 5`, `experimental results`, `future work`).

The negative lexicon in `src/tekne/resources/negative_lexicon.txt` is the
machine-readable form of this list.

## Known limitations of this gold set

1. **Single annotator.** There is no inter-annotator agreement figure, so the
   type and boundary decisions carry one person's judgement. The figures in the
   paper should be read as characterising this pipeline against this annotation,
   not as an absolute accuracy.
2. **Recent papers.** The paper sample is drawn from the most recent arXiv
   submissions in each category, which over-represents LLM-adjacent work in the
   CS categories.
3. **Document-level roles.** See above.
4. **Genre imbalance in mention count.** Patents contribute 569 of 908 gold
   mentions from 20 of 44 documents, because claim language repeats component
   names. Per-genre figures are reported separately for this reason.
