from sympy import *
from lindsge import DSGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# ===== MODEL ESPECIFICATION =====
# ================================
# endogenous variables at t
y, pi, r, g, z, exy, expi = symbols('y, pi, r, g, z, exy, expi')
endog = Matrix([y, pi, r, g, z, exy, expi])

# endogenous variables at t - 1
yl, pil, rl, gl, zl, exyl, expil = symbols('yl, pil, rl, gl, zl, exyl, expil')
endogl = Matrix([yl, pil, rl, gl, zl, exyl, expil])

# exogenous shocks
epsr, epsg, epsz = symbols('epsr, epsg, epsz')
exog = Matrix([epsr, epsg, epsz])

# expectational shocks
etay, etapi = symbols('etay, etapi')
expec = Matrix([etay, etapi])

# parameters
tau, beta, kappa, psi1, psi2, rhor, rhog, rhoz, sigr, sigg, sigz = \
    symbols('tau, beta, kappa, psi1, psi2, rhor, rhog, rhoz, sigr, sigg, sigz')
param = Matrix([tau, beta, kappa, psi1, psi2, rhor, rhog, rhoz, sigr, sigg, sigz])

# model equations
eq1 = y - exy + (1/tau)*(r-expi-rhoz*z) - (1-rhog)*g
eq2 = pi - beta*expi - kappa*(y - g)
eq3 = r - rhor*rl - (1-rhor)*psi1*pi - (1-rhor)*psi2*(y-g) - sigr*epsr
eq4 = g - rhog*gl - sigg*epsg
eq5 = z - rhoz*zl - sigz*epsz
eq6 = y - exyl - etay
eq7 = pi - expil - etapi

equations = Matrix([eq1, eq2, eq3, eq4, eq5, eq6, eq7])