TARGET = overview.pdf \
		 basic_ml.pdf

%.pdf: %.tex
	pdflatex $<
	open $@

main: $(TARGET) 

clean:
	rm *.aux *.log *.nav *.out *.snm *.toc *.pdf
