all: presentation.pdf

clean:
	rm -f pygpu_ndarray.so core.* *.o *~

cleantmp:
	rm -f core.* *.o *~

presentation.pdf: presentation.tex
	pdflatex presentation
	pdflatex presentation
	bibtexall
	pdflatex presentation
