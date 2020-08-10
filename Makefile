.PHONY: clean plots build deploy

build:
	d2lbook build html

deploy:
	cp -r _build/html/* ~/website/hhexiy.github.io/nlp/
	mv ~/website/hhexiy.github.io/nlp/_static ~/website/hhexiy.github.io/nlp/static
	mv ~/website/hhexiy.github.io/nlp/_sources ~/website/hhexiy.github.io/nlp/sources
	mv ~/website/hhexiy.github.io/nlp/_images ~/website/hhexiy.github.io/nlp/images

clean:
	rm -rf _build/

plots:
	cd plots; make all
