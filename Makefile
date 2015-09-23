all: guide figs

guide:
	cd code && ./ipnbdoctest.py user_guide.ipynb temp.ipynb
	cd code && ipython nbconvert --to html --template basic temp.ipynb
	mv code/temp.html ../_includes/notebook.html
	rm code/temp.ipynb

figs:
	cd code && ./ipnbdoctest.py sim_output.ipynb temp.ipynb
	cd code && ipython nbconvert --to html --template basic temp.ipynb
	mv code/temp.html ../_includes/vis.html
	rm code/temp.ipynb  
