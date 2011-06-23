all: presentation.pdf

presentation.pdf: presentation.tex pics/f_optimized.png pics/logreg_pydotprint_prediction.png
# pics/f_unoptimized.png pics/logreg_pydotprint_predic.png pics/logreg_pydotprint_train.png
	pdflatex presentation.tex

pics/f_optimized.png: simple_example.py
	python simple_example.py

pics/logreg_pydotprint_prediction.png: logreg_example.py
	python logreg_example.py
#pics/f_unoptimized.png: simple_example.py
#	python simple_example.py

