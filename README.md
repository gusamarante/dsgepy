# pydsge
Python class to calibrate, estimate and analyze linearized DSGE models. The interface is inpired by the dynare 
interface which allows for symbolic declarations of the variables. The bayesian estimation method uses markov chain 
monte carlo (MCMC) to simulates the posterior distributions of the parameters.

This library is an effort to bring the DSGE toolset into the python open-source world 

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

# Colaboration
* Feature suggestion on the project page



