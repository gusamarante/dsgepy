# pydsge
This is a Python library to calibrate, estimate and analyze linearized DSGE models. The interface is inpired by the dynare 
interface which allows for symbolic declarations of the variables and equations. The implemented bayesian estimation method uses markov chain monte carlo (MCMC) to simulate the posterior distributions of the parameters. This library is an effort to bring the DSGE toolset into the open-source world in a fully python implementation, which allows to fully embrace the advantages of the programming language when working with DSGEs.

# Installation
* INSTAL HERE

# Usage
A full example on how to use this library with a small New Keynesian model is available in [this Jupyter notebook](https://github.com/gusamarante/pydsge/blob/master/Example/example_snkm.ipynb).

# Model Especification
* Model declaration with symbolic variables
* linearized

# Model Solution
Based on the implementation of Christopher A. Sims' `gensys` function.
You can find the author's original matlab code
[here](https://dge.repec.org/codes/sims/linre3a/).
The paper explaining the solution method is
[this one](https://dge.repec.org/codes/sims/linre3a/LINRE3A.pdf).

# Model Estimation
* Bayesian

# Drawbacks
* slow performance due to the symbolic library

# Contributing
If you would like to contribute, plese check the [contributing guidelines](https://github.com/gusamarante/pydsge/blob/master/CONTRIBUTING.md) here. A list of feature suggestions (https://github.com/gusamarante/pydsge/projects) is available on the projects page of this repository.

# More Information and Help


