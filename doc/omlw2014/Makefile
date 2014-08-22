all: presentation.pdf sharing.pdf

clean:
	rm -f pygpu_ndarray.so core.* *.o *~

cleantmp:
	rm -f core.* *.o *~

presentation.pdf: presentation.tex
	pdflatex presentation
	pdflatex presentation

sharing.pdf: sharing.tex
	pdflatex sharing
	pdflatex sharing
