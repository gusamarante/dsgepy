import numpy as np
from sympy import *
from pydsge import DSGE
import matplotlib.pyplot as plt


# ================================
# ===== MODEL ESPECIFICATION =====
# ================================
# endogenous variables at t
y, pi, i, a, v, exp_y, exp_pi = symbols('y, pi, i, a, v, exp_y, exp_pi')

endog = Matrix([y, pi, i, a, v, exp_y, exp_pi])

# endogenous variables at t - 1
yl, pil, il, al, vl, exp_yl, exp_pil = symbols('yl, pil, il, al, vl, exp_yl, exp_pil')

endogl = Matrix([yl, pil, il, al, vl, exp_yl, exp_pil])

# exogenous shocks
eps_a, eps_v = symbols('eps_a, eps_v')

exog = Matrix([eps_a, eps_v])

# expectational shocks
eta_y, eta_pi = symbols('eta_y, eta_pi')

expec = Matrix([eta_y, eta_pi])

# parameters
sigma, psi, beta, kappa, phi_pi, phi_y, rho_a, sigma_a, rho_v, sigma_v = \
    symbols('sigma, psi, beta, kappa, phi_pi, phi_y, rho_a, sigma_a, rho_v, sigma_v')

param = Matrix([sigma, psi, beta, kappa, phi_pi, phi_y, rho_a, sigma_a, rho_v, sigma_v])

rho = (1/beta) - 1

# model equations
eq1 = y - exp_y + (1/sigma)*(i - exp_pi - psi * rho_a * a)
eq2 = pi - beta * exp_pi - kappa * y
eq3 = i - phi_pi * pi - phi_y * y - v
eq4 = a - rho_a * al - sigma_a * eps_a
eq5 = v - rho_v * vl - sigma_v * eps_v
eq6 = y - exp_yl - eta_y
eq7 = pi - exp_pil - eta_pi


equations = Matrix([eq1, eq2, eq3, eq4, eq5, eq6, eq7])

# ======================
# ===== SIMULATION =====
# ======================

calib_dict = {sigma: 1,
              psi: 0.2,
              beta: 0.99,
              kappa: 0.5,
              phi_pi: 1.5,
              phi_y: 0.5/4,
              rho_a: 0.9,
              sigma_a: 0.1,
              rho_v: 0.5,
              sigma_v: 0.0625}

obs_matrix = Matrix(np.zeros((3, 7)))
obs_matrix[0, 0] = 1
obs_matrix[1, 1] = 1
obs_matrix[2, 2] = 1

obs_offset = Matrix(np.zeros(3))
obs_offset[0] = rho

dsge_simul = DSGE(endog, endogl, exog, expec, param, equations, calib_dict, obs_matrix, obs_offset)
print(dsge_simul.eu)

# TODO give random seed
df_obs, df_states = dsge_simul.simulate(n_obs=150, random_seed=1)

df_states = df_states.tail(100)
df_obs = df_obs.tail(100)

df_states.plot()
plt.show()


# =============================
# ===== MODEL ESTIMATION  =====
# =============================

# TODO change prior table to choose mean and std

# priors
prior_dict = {sigma:   {'dist': 'gamma',    'param a': 2,     'param b': 1},
              psi:     {'dist': 'gamma',    'param a': 2,     'param b': 1},
              beta:    {'dist': 'beta',     'param a': 2.625, 'param b': 2.625},
              kappa:   {'dist': 'gamma',    'param a': 2,     'param b': 1},
              phi_pi:  {'dist': 'gamma',    'param a': 2,     'param b': 1},
              phi_y:   {'dist': 'gamma',    'param a': 2,     'param b': 1},
              rho_a:   {'dist': 'beta',     'param a': 2.625, 'param b': 2.625},
              sigma_a: {'dist': 'invgamma', 'param a': 3,     'param b': 1},
              rho_v:   {'dist': 'beta',     'param a': 2.625, 'param b': 2.625},
              sigma_v: {'dist': 'invgamma', 'param a': 3,     'param b': 1}}

dsge = DSGE(endog, endogl, exog, expec, param, equations, prior_dict=prior_dict,
            obs_matrix=obs_matrix, obs_data=df_obs, obs_offset=obs_offset)

df_chains, accepted = dsge.estimate(nsim=100, ck=0.01, file_path='snkm2.h5')
print(accepted)
df_chains.plot()
plt.show()

df_chains.astype(float).hist(bins=50)
plt.show()
