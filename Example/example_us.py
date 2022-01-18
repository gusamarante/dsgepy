import numpy as np
import pandas as pd
from pydsge import DSGE
import matplotlib.pyplot as plt
from sympy import Matrix, symbols
from statsmodels.tsa.filters.hp_filter import hpfilter

from pydsge import FRED

fred = FRED()

# ===== Grab and organize Data ===== #
# Get data from the FRED
series_dict = {'CPIAUCSL': 'CPI',
               'GDP': 'GDP',
               'DFF': 'Fed Funds Rate'}

df = fred.fetch(series_id=series_dict)

# Observed varibles
df_obs = pd.DataFrame()
df_obs['CPI'] = df['CPI'].dropna().resample('Q').last().pct_change(1) * 4
df_obs['FFR'] = df['Fed Funds Rate'].resample('Q').mean()
df_obs['outputgap'], _ = hpfilter(df['GDP'].resample('Q').last().dropna(), 1600)

df_obs = df_obs.dropna()

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
eps_a, eps_v, eps_pi = symbols('eps_a, eps_v, eps_pi')
exog = Matrix([eps_a, eps_v, eps_pi])

# expectational shocks
eta_y, eta_pi = symbols('eta_y, eta_pi')
expec = Matrix([eta_y, eta_pi])

# parameters
sigma, varphi, alpha, beta, theta, phi_pi, phi_y, rho_a, sigma_a, rho_v, sigma_v, sigma_pi = \
    symbols('sigma, varphi, alpha, beta, theta, phi_pi, phi_y, rho_a, sigma_a, rho_v, sigma_v, sigma_pi')

# Summary parameters
psi_nya = (1 + varphi) / (sigma * (1 - alpha) + varphi + alpha)
kappa = (1 - theta) * (1 - theta * beta) * (sigma * (1 - alpha) + varphi + alpha)

# model (state) equations
eq1 = y - exp_y + (1/sigma)*(i - exp_pi) - psi_nya * (rho_a - 1) * a
eq2 = pi - beta * exp_pi - kappa * y - sigma_pi * eps_pi
eq3 = i - phi_pi * pi - phi_y * y - v
eq4 = a - rho_a * al - sigma_a * eps_a
eq5 = v - rho_v * vl - sigma_v * eps_v
eq6 = y - exp_yl - eta_y
eq7 = pi - exp_pil - eta_pi

equations = Matrix([eq1, eq2, eq3, eq4, eq5, eq6, eq7])

# observation equations
obs01 = y
obs02 = pi
obs03 = 1/beta - 1 + i

obs_names = ['Output Gap', 'Inflation', 'Interest Rate']

obs_equations = Matrix([obs01, obs02, obs03])

# =============================
# ===== MODEL ESTIMATION  =====
# =============================
# Not all parameters need to be estimated, these are going to be calibrated
calib_param = {varphi: 1, alpha: 0.4, beta: 0.9950615}
estimate_param = Matrix([sigma, theta, phi_pi, phi_y, rho_a, sigma_a, rho_v, sigma_v, sigma_pi])

# priors
prior_dict = {sigma:    {'dist': 'normal',   'mean':  1.30, 'std': 0.20, 'label': '$\\sigma$'},
              theta:    {'dist': 'beta',     'mean':  0.60, 'std': 0.20, 'label': '$\\theta$'},
              phi_pi:   {'dist': 'normal',   'mean':  1.50, 'std': 0.35, 'label': '$\\phi_{\\pi}$'},
              phi_y:    {'dist': 'gamma',    'mean':  0.25, 'std': 0.10, 'label': '$\\phi_{y}$'},
              rho_a:    {'dist': 'beta',     'mean':  0.50, 'std': 0.25, 'label': '$\\rho_a$'},
              sigma_a:  {'dist': 'invgamma', 'mean':  0.50, 'std': 0.25, 'label': '$\\sigma_a$'},
              rho_v:    {'dist': 'beta',     'mean':  0.50, 'std': 0.25, 'label': '$\\rho_v$'},
              sigma_v:  {'dist': 'invgamma', 'mean':  0.50, 'std': 0.25, 'label': '$\\sigma_v$'},
              sigma_pi: {'dist': 'invgamma', 'mean':  0.50, 'std': 0.25, 'label': '$\\sigma_{\\pi}$'}}


dsge = DSGE(endog, endogl, exog, expec, equations,
            estimate_params=estimate_param,
            calib_dict=calib_param,
            obs_equations=obs_equations,
            prior_dict=prior_dict,
            obs_data=df_obs,
            obs_names=obs_names,
            verbose=True)

dsge.estimate(nsim=1000, ck=0.1, file_path='example_us.h5')
dsge.eval_chains(burnin=0.1, show_charts=True)

# print(dsge.posterior_table)

# IRFs from the estimated Model
# dsge.irf(periods=24, show_charts=True)

# Extraxct state variables
# df_states_hat, df_states_se = dsge.states()

# Historical Decomposition
# df_hd = dsge.hist_decomp(show_charts=True)
