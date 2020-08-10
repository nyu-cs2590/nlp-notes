.PHONY: clean

all: neural_networks/fflm.svg \
	 neural_networks/rnn.svg

%.pdf: %.tex
	pdflatex --output-directory $(dir $<) $< 

%.svg: %.pdf
	inkscape --export-type svg $<

clean:
	rm */*.log */*.aux
