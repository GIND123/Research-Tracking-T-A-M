# TEKNE

Extraction of technology mentions from scientific papers and patents, built as
the front end of a technology-evolution tracking corpus.

The problem is not hard to state — find the technologies a document talks about —
and it is easy to build something that mostly works. It is much harder to build
something whose output you would be willing to aggregate into a trend line over a
million documents, because that use has an asymmetry ordinary IE evaluation does
not capture: **a fabricated technology is not averaged away by scale.** A term
that is spuriously extracted at a rate proportional to corpus growth produces a
rising curve indistinguishable from a real emerging technology, and nobody
downstream will ever go back and check. A missed mention, by contrast, mostly
rescales a series without changing its shape.

So the design here is organised around making bad output *unrepresentable*
rather than unlikely:

- An emitted mention is a pair of character offsets into the source. Its surface
  form is defined as the slice at those offsets, so a model that returns a
  technology name we cannot locate has returned nothing.
- Every mention carries the sentence it was read out of, and that sentence is
  verified to contain it.
- Candidates are proposed generously by five independent recallers and filtered
  by a guard stack, an independent verifier and one calibrated abstention
  threshold — not by making any single stage conservative.
- Everything the pipeline declined to emit is kept, with the reason.

A companion write-up is in [`report/main.tex`](report/main.tex), and
[`docs/decisions.md`](docs/decisions.md) records the design calls — including
five that were wrong and what the measurements were that reversed them.

## Quick start

```sh
make setup          # venv, package, spaCy model
make kb             # compile the Computer Science Ontology gazetteer (~2 min)
make data           # fetch 24 arXiv abstracts + 20 USPTO publications
make gold           # expand the annotation table into offset-anchored gold
make eval           # baselines + ablations -> runs/eval/results.{json,tex}
```

No API key is needed. Without one the pipeline runs its deterministic and neural
tiers, reports that it did so, and produces every offline row of every table. To
enable the model-backed stages, put a key in `.env`:

```sh
cp .env.example .env
$EDITOR .env        # ANTHROPIC_API_KEY=sk-ant-...
```

Check what a run would cost before making it:

```sh
.venv/bin/python -m tekne.cli plan data/raw/patents_eval.jsonl
```

## Looking at one document

`inspect` is the most useful entry point — it prints the emitted mentions, the
review queue, the rejections with reasons, and the stage-by-stage trace.

```sh
.venv/bin/python -m tekne.cli inspect arxiv:2609.20800v1 \
    --corpus data/raw/papers_eval.jsonl --show-rejected
```

## How it fits together

```
ingest/      normalisation with an offset map back to the raw text;
             genre-aware segmentation (paper sections, patent claims)
recall/      five recallers: noun-phrase lattice, Schwartz-Hearst abbreviations,
             C-value/NC-value terminology statistics, KB gazetteer, LLM proposer
guard/       grounding, negative lexicon, injection, consensus, evidence,
             temporal consistency, provenance
classify/    type (9-way ontology) and stance (proposed/used/compared/claimed)
verify/      structural -> embedding -> batched LLM judge -> adjudicator
calib/       confidence model, risk-coverage curves, selective prediction
link/        abbreviation-aware canonicalisation; NIL clustering for technologies
             no ontology has heard of
track/       per-year series, emergence ranking, adoption shift
agents/      the blackboard, stage order, escalation policy, budgets, prompts
```

The orchestrator is in [`src/tekne/agents/orchestrator.py`](src/tekne/agents/orchestrator.py);
reading that file top to bottom is the fastest way to understand the system.

### Cost control

Model calls are the dominant cost and most naive designs waste almost all of
them. Three reductions, applied in order before anything is sent:

1. **Deduplicate.** Verification is a function of (surface, type, sentence). A
   term repeated forty times in a specification is one question.
2. **Escalate selectively.** Only the embedding verifier's uncertainty band
   reaches a model; confident accepts and rejects never do.
3. **Batch.** Survivors are packed ~20 per request behind a cached system
   prompt.

Together these take a patent from roughly 120 verifier calls to one or two. The
proposer is capped at three windows and 9k characters per document, chosen by
section priority rather than by reading the whole specification.

## Reproducibility

- Every result row is a `Condition` in [`src/tekne/eval/runner.py`](src/tekne/eval/runner.py) —
  ablations are configuration changes, not code paths.
- Every mention records the digest of the config that produced it, the recallers
  that proposed it, and the prompt versions involved.
- Model calls are cached content-addressed in SQLite, keyed on the full request,
  so a cached run replays byte-for-byte and a changed prompt shows up as a cache
  miss rather than as a silent behaviour change.

## Data

| Source | Use | Access |
|--------|-----|--------|
| arXiv API | paper abstracts, 5 category queries | public, no key (rate limited) |
| Google Patents | patent full text | public, no key |
| HUPD metadata ([patentdataset.org](https://patentdataset.org/)) | patent sampling frame, CPC labels, filing dates | public |
| CSO 3.4 | gazetteer and hierarchy | public |

The gold set is 895 mentions over 44 documents (24 papers, 20 patents),
annotated by one person under
[`docs/annotation-guidelines.md`](docs/annotation-guidelines.md). Its limitations
are listed at the end of that file and should be read before quoting any number
from it.

`make trend-data` builds the larger time-sliced corpus the tracking demo wants,
one year per request. The arXiv endpoint throttles aggressively — it answers 406
rather than 429, in bursts, and a retry storm makes the next query *more* likely
to fail — so expect this to take tens of minutes and to need re-running. Our own
run got one year before being blocked, which is why the tracking results in the
report are reported as a mechanism demonstration rather than as a trend finding.

## Tests

```sh
make test
```

`tests/test_adversarial.py` is the one worth reading. It drives the pipeline with
a scripted backend that fabricates entities, paraphrases spans, returns malformed
JSON, emits a thousand inventions at once, and carries a prompt-injection payload,
and asserts that none of it reaches the output. One test in there records a
*negative* result — grounding alone does not stop an attacker who writes their
phantom technology into their own patent — because that is the limitation that
motivated the sentence-scoped injection guard.

## Licence

MIT.
