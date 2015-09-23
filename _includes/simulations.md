# Simulation code

There are three main scripts used to for the simulations in the paper, and all
are included in the code folder:

* `schelling.py` contains the core methods to set up and run a Schelling-type
  agent-based model. See some instructions on how to use it below.
* `sim_enginge_scoop.py` is the script used to run the simulations on a
  super-computer. It relies on the library
  [`scoop`](http://scoop.readthedocs.org/en/0.7/) for the parallel
  computations. The results can be found in the code folder and are named 
  `sim_res.csv`.
* `results.py` contains the methods required to generate the figures in the 
  paper and those included additionally in this page. An example on how to use 
  them can be found in the [visualizations](vis.html) section of this website.
