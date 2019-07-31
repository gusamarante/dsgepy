from sympy import *
from lindsge import DSGE
import numpy as np
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

# =============================================
# ===== MODEL CALIBRATION AND SIMULATION  =====
# =============================================

subs_dict = {tau: 1.2,
             beta: 0.99,
             kappa: 0.3,
             psi1: 1.5,
             psi2: 0.5,
             rhor: 0.8,
             rhog: 0.3,
             rhoz: 0.6,
             sigr: 0.01,
             sigg: 0.02,
             sigz: 0.03}

obs_matrix = np.zeros((3, 7))
obs_matrix[0, 0] = 1
obs_matrix[1, 1] = 1
obs_matrix[2, 2] = 1

dsge_simul = DSGE(endog, endogl, exog,expec, param, equations, subs_dict, obs_matrix)

df_obs, df_states = dsge_simul.simulate(n_obs=50)

df_obs.plot()
plt.show()

# =============================
# ===== MODEL ESTIMATION  =====
# =============================

# priors
# 'gamma'
prior_dict = {tau:   {'dist': 'gamma', 'param a': 1, 'param b': 1},
              beta:  {'dist': 'gamma', 'param a': 1, 'param b': 1},
              kappa: {'dist': 'gamma', 'param a': 1, 'param b': 1},
              psi1:  {'dist': 'gamma', 'param a': 1, 'param b': 1},
              psi2:  {'dist': 'gamma', 'param a': 1, 'param b': 1},
              rhor:  {'dist': 'gamma', 'param a': 1, 'param b': 1},
              rhog:  {'dist': 'gamma', 'param a': 1, 'param b': 1},
              rhoz:  {'dist': 'gamma', 'param a': 1, 'param b': 1},
              sigr:  {'dist': 'gamma', 'param a': 1, 'param b': 1},
              sigg:  {'dist': 'gamma', 'param a': 1, 'param b': 1},
              sigz:  {'dist': 'gamma', 'param a': 1, 'param b': 1}}

dsge = DSGE(endog, endogl, exog,expec, param, equations, prior_info=prior_dict)