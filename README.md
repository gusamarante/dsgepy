# dsgepy
This is a Python library to specify, calibrate, solve, simulate, estimate and 
analyze linearized DSGE models. The specification interface is inpired by 
dynare, which allows for symbolic declarations of parameters, variables and 
equations. Once a model is calbrated or estimated, it is solved using Sims 
(2002) methodology. 
Simulated trajectories can be generated from a calibrated model. Estimation 
uses bayesian methods, specifically markov chain monte carlo (MCMC), to 
simulate the posterior distributions of the parameters. Analysis tools include 
impulse-response functions, historical decompostion and extraction of latent 
variables.

This library is an effort to bring the DSGE toolset into the open-source world 
in a full python implementation, which allows to embrace the advantages of this 
programming language when working with DSGEs.

---
# Installation
You can install this development version using:
```
pip install dsgepy
```

---
# Example
A full example on how to use this library with a small New Keynesian model is available in 
[this Jupyter notebook](https://github.com/gusamarante/pydsge/blob/master/Example/example_snkm.ipynb). The model used 
in the example is descibred briefly by the following equations:

$$\tilde{y}\_{t}=E\_{t}\left(\tilde{y}\_{t+1}\right)-\frac{1}{\sigma}\left[\hat{i}\_{t}-E\_{t}\left(\pi\_{t+1}\right)\right]+\psi\_{ya}^{n}\left(\rho\_{a}-1\right)a\_{t}$$

$$\pi\_{t}=\beta E\_{t}\left(\pi\_{t+1}\right)+\kappa\tilde{y}\_{t}+\sigma\_{\pi}\varepsilon\_{t}^{\pi}$$

$$\hat{i}\_{t}=\phi\_{\pi}\pi\_{t}+\phi\_{y}\tilde{y}\_{t}+v\_{t}$$

$$a_{t}=\rho\_{a}a\_{t-1}+\sigma\_{a}\varepsilon\_{t}^{a}$$

$$v_{t}=\rho_{v}v_{t-1}+\sigma_{v}\varepsilon_{t}^{v}$$


# Model Especification
For now, the model equations have to be linearized around its steady-state. 
Soon, there will be a functionality that allows for declaration with 
non-linearized equilibrium conditions.

# Model Solution
The solution method used is based on the implementation of Christopher A. Sims' `gensys` function. You can find the 
author's original matlab code [here](https://dge.repec.org/codes/sims/linre3a/). The paper explaining the solution 
method is [this one](https://dge.repec.org/codes/sims/linre3a/LINRE3A.pdf).

# Model Estimation
The models are estimated using Bayesian methdos, specifically, by simulating the posterior distribution using MCMC 
sampling. This process is slow, so there is a functionality that allows you to stop a simulation and continue 
it later from where it stoped.

# Analysis
There are functionalities for computing Impulse-Response funcions for both state variables and observed variables. 
Historical decomposition is also available, but only when the number of exogenous shocks matches the number of 
observed variables.

---
# Drawbacks
Since there is symbolic declaration of variables and equations, methdos 
involving them are slow. Also, MCMC methods for macroeconomic models require 
many iterations to achieve convergence. Clearly, there is room for improvement 
on the efficiency of these estimation algorithms. Contributions are welcome.
Speaking of contributions...

---
# Contributing
If you would like to contribute to this repository, plese check the 
[contributing guidelines](https://github.com/gusamarante/pydsge/blob/master/CONTRIBUTING.md) here. A 
[list of feature suggestions](https://github.com/gusamarante/pydsge/projects) is available on the projects page of this
repository.

---
# More Information and Help
If you need more information and help, specially about contributing, you can 
contact Gustavo Amarante on developer@dsgepy.com
