# Two passes, then bibtex, then two more: standard for natbib + acl.
MAIN = main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib $(wildcard tables/*.tex) $(wildcard figures/*)
	pdflatex -interaction=nonstopmode $(MAIN) >/dev/null
	bibtex $(MAIN) >/dev/null || true
	pdflatex -interaction=nonstopmode $(MAIN) >/dev/null
	pdflatex -interaction=nonstopmode $(MAIN) | tail -5

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf
