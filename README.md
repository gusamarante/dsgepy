# dsgepy
This is a Python library to calibrate, estimate and analyze linearized DSGE models. The interface is inpired by the 
dynare interface which allows for symbolic declarations of the variables and equations. The implemented bayesian 
estimation method uses markov chain monte carlo (MCMC) to simulate the posterior distributions of the parameters. This 
library is an effort to bring the DSGE toolset into the open-source world in a full python implementation, which allows 
to embrace the advantages of this programming language when working with DSGEs.

---
# Installation
You can install this development version using:
```
pip install dsgepy
```

### Kalman Filter
Computing the likelihood of models involve using the kalman filter. `pykalman` is available for python, but some 
adjustments to the original library are needed to use with this library. So **in order for `dsgepy` to work you need 
the corrected version of `pykalman`, available [here](https://github.com/gusamarante/pykalman). Make sure to clone this 
version and add it to the interpreter befor using `dsgepy`**. The corrections here correct the way `pykalman` handles 
masked numpy arrays and deals with ill-estimated covariance matrices. 

---
# Example
A full example on how to use this library with a small New Keynesian model is available in 
[this Jupyter notebook](https://github.com/gusamarante/pydsge/blob/master/Example/example_snkm.ipynb). The model used 
in the example is descibred briefly by the following equations: 

<img src="http://latex.codecogs.com/gif.latex?\tilde{y}_{t}=E_{t}\left(\tilde{y}_{t+1}\right)-\frac{1}{\sigma}\left[\hat{i}_{t}-E_{t}\left(\pi_{t+1}\right)\right]+\psi_{ya}^{n}\left(\rho_{a}-1\right)a_{t}" />

<img src="http://latex.codecogs.com/gif.latex?\pi_{t}=\beta E_{t}\left(\pi_{t+1}\right)+\kappa\tilde{y}_{t}+\sigma_{\pi}\varepsilon_{t}^{\pi}" />

<img src="http://latex.codecogs.com/gif.latex?\hat{i}_{t}=\phi_{\pi}\pi_{t}+\phi_{y}\tilde{y}_{t}+v_{t}" />

<img src="http://latex.codecogs.com/gif.latex?a_{t}=\rho_{a}a_{t-1}+\sigma_{a}\varepsilon_{t}^{a}" />

<img src="http://latex.codecogs.com/gif.latex?v_{t}=\rho_{v}v_{t-1}+\sigma_{v}\varepsilon_{t}^{v}" />


# Model Especification
For now, the model equations have to be linearized around its steady-state. 
Soon, there will be a functionality that allows for non-linearized declaration 
of the equilibrium conditions.

# Model Solution
The solution method used is based on the implementation of Christopher A. Sims' `gensys` function. You can find the 
author's original matlab code [here](https://dge.repec.org/codes/sims/linre3a/). The paper explaining the solution 
method is [this one](https://dge.repec.org/codes/sims/linre3a/LINRE3A.pdf).

# Model Estimation
The models are estimated using Bayesian methdos, specifically, by simulating the posterior distribution using MCMC 
sampling. Simulations are typically long, so there is a functionality that allows you to stop a simulation and continue 
it later from where it stoped.

# Analysis
There are functionalities for computing Impulse-Response funcions for both state variables and observed variables. 
Historical decomposition is also available, but only when the number of exogenous shocks matches the number of 
observed variables.

---
# Drawbacks
Since there is symbolic declaration of variables and equations, methdos involving them are slow, so the MCMC methods 
typically take a long time to run. Although there is room for improvement for the efficiency of these estimation
algorithms.

---
# Contributing
If you would like to contribute to this repository, plese check the 
[contributing guidelines](https://github.com/gusamarante/pydsge/blob/master/CONTRIBUTING.md) here. A 
[list of feature suggestions](https://github.com/gusamarante/pydsge/projects) is available on the projects page of this
repository.

---
# More Information and Help
If you need more information and help, specially about contributing, you can contact Gustavo Amarante on 
developer@dsgepy.com
