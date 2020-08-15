.PHONY: clean

all: neural_networks/fflm.pdf \
	 neural_networks/rnn.pdf \
	 neural_networks/birnn.pdf \
	 sequence/viterbi.pdf

%.pdf: %.tex style.tex
	pdflatex --output-directory $(dir $<) $< 

%.svg: %.pdf
	inkscape --export-type svg $<

clean:
	rm */*.log */*.aux
