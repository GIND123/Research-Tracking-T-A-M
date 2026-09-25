# Reproduce everything from a clean checkout:
#   make setup data kb gold eval
# The pipeline runs without an API key; `make eval` then reports the offline
# tier and says so. Put a key in .env to enable the model-backed stages.

PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
CORPORA  = data/raw/papers_eval.jsonl data/raw/patents_eval.jsonl
RUN     ?= runs/eval

.PHONY: help setup data kb gold eval baselines ablations demo trends test lint paper clean distclean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

setup: ## create the venv and install the package with all extras
	uv venv --python 3.12 .venv 2>/dev/null || python3 -m venv .venv
	$(PIP) install -e '.[neural,llm,plots,dev]'
	$(PY) -m spacy download en_core_web_sm

data: ## fetch a NEW sample (does not reproduce the committed corpora -- see README)
	$(PY) scripts/fetch_data.py eval
	$(PY) scripts/fetch_data.py patents --n 24

trend-data: ## fetch the larger time-sliced corpus for the evolution demo
	$(PY) scripts/fetch_data.py --delay 6 trend --start-year 2015 --end-year 2025 --per-year 50

kb: ## compile the Computer Science Ontology into a gazetteer
	$(PY) scripts/build_kb.py --out data/kb/cso.json

gold: ## re-expand the annotation table against the committed corpora
	$(PY) scripts/make_gold.py

eval: ## run baselines and ablations, write runs/eval/results.{json,tex}
	$(PY) scripts/run_eval.py --suite all --out $(RUN)

baselines: ## baselines only
	$(PY) scripts/run_eval.py --suite baselines --out $(RUN)

ablations: ## ablations only
	$(PY) scripts/run_eval.py --suite ablations --out $(RUN)

demo: ## extract one document and print every decision
	$(PY) -m tekne.cli inspect arxiv:2609.20800v1 --corpus data/raw/papers_eval.jsonl --show-rejected

# Includes the time-sliced corpus when `make trend-data` has produced one; the
# evaluation corpora alone give only two time slices.
TREND_CORPUS = $(wildcard data/raw/papers_trend.jsonl)
TRACK_CORPORA = $(CORPORA) $(TREND_CORPUS)

trends: ## build technology trend series from an extraction run
	$(PY) -m tekne.cli extract $(TRACK_CORPORA) --out runs/extract
	$(PY) -m tekne.cli trends runs/extract/mentions.jsonl \
	    $(foreach c,$(TRACK_CORPORA),--corpus $(c))

plan: ## project API cost without issuing a call
	$(PY) -m tekne.cli plan $(CORPORA)

test: ## run the test suite
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src tests scripts

tables: ## regenerate the report's tables and figures from runs/eval
	$(PY) scripts/make_tables.py
	$(PY) scripts/make_figures.py

paper: tables ## build report/main.pdf
	$(MAKE) -C report

clean:
	rm -rf runs/eval runs/extract report/*.aux report/*.log report/*.out report/*.bbl report/*.blg

distclean: clean
	rm -rf .venv data/raw data/kb runs
