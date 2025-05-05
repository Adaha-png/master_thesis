PDFLATEX=pdflatex
BIBTEX=biber
TARGET=main

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex $(TARGET).bbl
	$(PDFLATEX) $(TARGET).tex
	$(PDFLATEX) $(TARGET).tex
	

$(TARGET).bbl: $(TARGET).aux ref.bib
	$(BIBTEX) $(TARGET)

$(TARGET).aux: $(TARGET).tex
	$(PDFLATEX) $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).bbl $(TARGET).blg $(TARGET).log $(TARGET).out $(TARGET).toc $(TARGET).lof $(TARGET).lot $(TARGET).bcf $(TARGET).run.xml $(TARGET).fdb_latexmk $(TARGET).auxlock $(TARGET).fls




