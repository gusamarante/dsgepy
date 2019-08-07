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

# =============================================
# ===== MODEL CALIBRATION AND SIMULATION  =====
# =============================================

subs_dict = {tau: 1.2,
             beta: 0.99,
             kappa: 0.3,
             psi1: 1.5,
             psi2: 0.5,
             rhor: 0.90,
             rhog: 0.3,
             rhoz: 0.6,
             sigr: 0.1,
             sigg: 0.02,
             sigz: 0.03}

obs_matrix = Matrix(np.zeros((3, 7)))
obs_matrix[0, 0] = 1
obs_matrix[1, 1] = 1
obs_matrix[2, 2] = 1

obs_offset = Matrix(np.zeros(3))
obs_offset[1] = 3
obs_offset[2] = 4*(1/beta - 1)*100

dsge_simul = DSGE(endog, endogl, exog,expec, param, equations, subs_dict, obs_matrix, obs_offset)

df_obs, df_states = dsge_simul.simulate(n_obs=200)

# df_obs.plot()
# plt.show()

# =============================
# ===== MODEL ESTIMATION  =====
# =============================

# priors
prior_dict = {tau:   {'dist': 'gamma',    'param a': 2,   'param b': 0.50},
              beta:  {'dist': 'beta',     'param a': 10,  'param b': 0.10},
              kappa: {'dist': 'uniform',  'param a': 0,   'param b': 1.00},
              psi1:  {'dist': 'gamma',    'param a': 1.5, 'param b': 1.00},
              psi2:  {'dist': 'gamma',    'param a': 0.8, 'param b': 1.00},
              rhor:  {'dist': 'uniform',  'param a': 0,   'param b': 1.00},
              rhog:  {'dist': 'uniform',  'param a': 0,   'param b': 1.00},
              rhoz:  {'dist': 'uniform',  'param a': 0,   'param b': 1.00},
              sigr:  {'dist': 'invgamma', 'param a': 4,   'param b': 0.12},
              sigg:  {'dist': 'invgamma', 'param a': 4,   'param b': 0.12},
              sigz:  {'dist': 'invgamma', 'param a': 4,   'param b': 0.12}}

dsge = DSGE(endog, endogl, exog, expec, param, equations, prior_dict=prior_dict,
            obs_matrix=obs_matrix, obs_data=df_obs, obs_offset=obs_offset)

df_chains, accepted = dsge.estimate(nsim=200, ck=0.01, head_start='example3.h5')
print(accepted)
df_chains.plot()
plt.show()

df_chains.astype(float).hist()
plt.show()

# store = pd.HDFStore(r'C:\Users\gamarante\Desktop\chains.h5')
# store['df'] = df_chains
# store.close()
