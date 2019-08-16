import numpy as np
from sympy import *
from pydsge import DSGE
import matplotlib.pyplot as plt


# ================================
# ===== MODEL ESPECIFICATION =====
# ================================
# endogenous variables at t
y, pi, i, a, v, exp_y, exp_pi = symbols('y')

endog = Matrix([y])

# endogenous variables at t - 1
yl, pil, il, al, vl, exp_yl, exp_pil = symbols('yl, pil, il, al, vl, exp_yl, exp_pil')

endogl = Matrix([yl, pil, il, al, vl, exp_yl, exp_pil])

# exogenous shocks
eps_a, eps_v, eps_pi = symbols('eps_a, eps_v, eps_pi')

exog = Matrix([eps_a, eps_v, eps_pi])

# expectational shocks
eta_y, eta_pi = symbols('eta_y, eta_pi')

expec = Matrix([eta_y, eta_pi])

# parameters
sigma, varphi, alpha, beta, theta, phi_pi, phi_y, rho_a, sigma_a, rho_v, sigma_v, sigma_pi = \
    symbols('sigma, varphi, alpha, beta, theta, phi_pi, phi_y, rho_a, sigma_a, rho_v, sigma_v, sigma_pi')

estimate_param = Matrix([sigma, theta, phi_pi, phi_y, rho_a, sigma_a, rho_v, sigma_v, sigma_pi])
calib_param = {varphi: 1, alpha:0.4, beta: 0.997805}

# Summary parameters
psi_nya = (1 + varphi) / (sigma*(1-alpha) + varphi + alpha)
kappa = (1 - theta)*(1 - theta * beta)*(sigma*(1-alpha) + varphi + alpha)

# model equations
eq1 = y - exp_y + (1/sigma)*(i - exp_pi) - psi_nya * (rho_a - 1) * a
eq2 = pi - beta * exp_pi - kappa * y - sigma_pi * eps_pi
eq3 = i - phi_pi * pi - phi_y * y - v
eq4 = a - rho_a * al - sigma_a * eps_a
eq5 = v - rho_v * vl - sigma_v * eps_v
eq6 = y - exp_yl - eta_y
eq7 = pi - exp_pil - eta_pi

equations = Matrix([eq1, eq2, eq3, eq4, eq5, eq6, eq7])
