from sympy import *
from pydsge import DSGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# ===== MODEL ESPECIFICATION =====
# ================================
# endogenous variables at t
y, pi, r, eps_y, eps_pi, eps_r, exp_y, exp_pi = symbols('y, pi, r, eps_y, eps_pi, eps_r, exp_y, exp_pi')

endog = Matrix([y, pi, r, eps_y, eps_pi, eps_r, exp_y, exp_pi])

# endogenous variables at t - 1
yl, pil, rl, eps_yl, eps_pil, eps_rl, exp_yl, exp_pil = symbols('yl, pil, rl, eps_yl, eps_pil, eps_rl, exp_yl, exp_pil')

endogl = Matrix([yl, pil, rl, eps_yl, eps_pil, eps_rl, exp_yl, exp_pil])

# exogenous shocks
xi_y, xi_pi, xi_r = symbols('xi_y, xi_pi, xi_r')

exog = Matrix([xi_y, xi_pi, xi_r])

# expectational shocks
eta_y, eta_pi = symbols('eta_y, eta_pi')

expec = Matrix([eta_y, eta_pi])

# parameters
w_f, w_b, w_y, beta_f, beta_b, beta_r, rho, gamma_pi, \
    gamma_y, rho_y, rho_pi, rho_r, sigma_y, sigma_pi, sigma_r = \
    symbols('w_f, w_b, w_y, beta_f, beta_b, beta_r, rho, gamma_pi, '
            'gamma_y, rho_y, rho_pi, rho_r, sigma_y, sigma_pi, sigma_r')

param = Matrix([w_f, w_b, w_y, beta_f, beta_b, beta_r, rho, gamma_pi,
                gamma_y, rho_y, rho_pi, rho_r, sigma_y, sigma_pi, sigma_r])

# model equations
eq1 = y - beta_f * exp_y - beta_b * yl + beta_r * (r - exp_pi) - eps_y
eq2 = pi - w_f * exp_pi - w_b * pil - w_y * y - eps_pi
eq3 = r - rho * rl - (1-rho) * (gamma_pi * pi + gamma_y * y) - eps_r
eq4 = eps_y - rho_y * eps_yl - sigma_y * xi_y
eq5 = eps_pi - rho_pi * eps_pil - sigma_pi * xi_pi
eq6 = eps_r - rho_r * eps_rl - sigma_r * xi_r
eq7 = y - exp_yl - eta_y
eq8 = pi - exp_pil - eta_pi

equations = Matrix([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])

# ======================
# ===== SIMULATION =====
# ======================

calib_dict = {w_f:      0.30,
              w_b:      0.70,
              w_y:      0.13,
              beta_f:   0.30,
              beta_b:   0.70,
              beta_r:   0.09,
              rho:      0.50,
              gamma_pi: 1.50,
              gamma_y:  0.50,
              rho_y:    0.00,
              rho_pi:   0.00,
              rho_r:    0.80,
              sigma_y:  0.33,
              sigma_pi: 0.50,
              sigma_r:  0.43}

obs_matrix = Matrix(np.zeros((3, 8)))
obs_matrix[0, 0] = 1
obs_matrix[1, 1] = 1
obs_matrix[2, 2] = 1

obs_offset = Matrix(np.zeros(3))

dsge_simul = DSGE(endog, endogl, exog, expec, param, equations, calib_dict, obs_matrix, obs_offset)
print(dsge_simul.eu)

# df_obs, df_states = dsge_simul.simulate(n_obs=100)

# df_states.plot()
# plt.show()




