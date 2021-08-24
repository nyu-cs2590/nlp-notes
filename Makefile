.PHONY: clean plots build deploy

build:
	d2lbook build html

deploy:
	cp -r _build/html/slides/lec12 ~/website/hhexiy.github.io/nlp/slides
	cp -r _build/html/schedule.html ~/website/hhexiy.github.io/nlp/
	#mv ~/website/hhexiy.github.io/nlp/_static ~/website/hhexiy.github.io/nlp/static
	#mv ~/website/hhexiy.github.io/nlp/_sources ~/website/hhexiy.github.io/nlp/sources
	#mv ~/website/hhexiy.github.io/nlp/_images ~/website/hhexiy.github.io/nlp/images

deploy_nyu_all:
	scp -r _build/html/* hehe@access.cims.nyu.edu:/usr/httpd/htdocs_cs/courses/fall20/CSCI-GA.2590-001

deploy_nyu_home:
	scp -r _build/html/*.html hehe@access.cims.nyu.edu:/usr/httpd/htdocs_cs/courses/fall20/CSCI-GA.2590-001/

deploy_nyu_slides:
	scp -r _build/html/slides/lec12 hehe@access.cims.nyu.edu:/usr/httpd/htdocs_cs/courses/fall20/CSCI-GA.2590-001/slides/

deploy_nyu_notes:
	scp -r _build/html/notes/*.html hehe@access.cims.nyu.edu:/usr/httpd/htdocs_cs/courses/fall20/CSCI-GA.2590-001/notes/

deploy_nyu_images:
	scp -r _build/html/_images/* hehe@access.cims.nyu.edu:/usr/httpd/htdocs_cs/courses/fall20/CSCI-GA.2590-001/_images

clean:
	rm -rf _build/

plots:
	cd plots; make all
