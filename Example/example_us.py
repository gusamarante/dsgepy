import numpy as np
import pandas as pd
from pydsge import DSGE
import matplotlib.pyplot as plt
from sympy import Matrix, symbols

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
# df_obs['CPI'] = df['CPI'].dropna().resample('Q').last().pct_change(1) * 4
# df_obs['FFR'] = df['Fed Funds Rate'].resample('Q').mean()
df_obs['logGDP'] = np.log(df['GDP'].resample('Q').last().dropna())

# ================================
# ===== MODEL ESPECIFICATION =====
# ================================
# endogenous variables at t
y, yp, g, gm1, mu = symbols('y, yp, g, gm1, mu')
endog = Matrix([y, yp, g, gm1, mu])

# endogenous variables at t - 1
yl, ypl, gl, gm1l, mul = symbols('yl, ypl, gl, gm1l, mul')
endogl = Matrix([yl, ypl, gl, gm1l, mul])

# exogenous shocks
eps_p, eps_g, eps_mu = symbols('eps_p, eps_g, eps_mu')
exog = Matrix([eps_p, eps_g, eps_mu])

# parameters
sigma_p, sigma_mu, phi1, phi2, sigma_g = symbols('sigma_p, sigma_mu, phi1, phi2, sigma_g')

# expectational shocks
eta_y, eta_pi = symbols('eta_y, eta_pi')
expec = Matrix([eta_y, eta_pi])

# model (state) equations
eq1 = y - yp - g
eq2 = yp - mu - ypl - sigma_p * eps_p
eq3 = mu - mul - sigma_mu * eps_mu
eq4 = g - phi1 * gl - phi2 * gm1l - sigma_g * eps_g
eq5 = gm1 - gm1l

equations = Matrix([eq1, eq2, eq3, eq4, eq5])

# observation equations
obs01 = y
obs_equations = Matrix([obs01])
obs_names = ['logGDP']

# =============================
# ===== MODEL ESTIMATION  =====
# =============================
estimate_param = Matrix([sigma_p, sigma_mu, phi1, phi2, sigma_g])

# priors
prior_dict = {sigma_p:  {'dist': 'invgamma', 'mean':  0.50, 'std': 0.25, 'label': r'$\sigma_{p}$'},
              sigma_mu: {'dist': 'invgamma', 'mean':  0.50, 'std': 0.25, 'label': r'$\sigma_{\mu}$'},
              sigma_g:  {'dist': 'invgamma', 'mean':  0.50, 'std': 0.25, 'label': r'$\sigma_{\mu}$'},
              phi1:     {'dist': 'normal', 'mean':  0.9, 'std': 0.3, 'label': r'$\phi_{1}$'},
              phi2:     {'dist': 'normal', 'mean':  0.9, 'std': 0.3, 'label': r'$\phi_{1}$'}}

dsge = DSGE(endog=endog,
            endogl=endogl,
            exog=exog,
            expec=expec,
            state_equations=equations,
            estimate_params=estimate_param,
            obs_equations=obs_equations,
            prior_dict=prior_dict,
            obs_data=df_obs,
            obs_names=obs_names,
            verbose=True)

dsge.estimate(nsim=100, ck=0.1, file_path='example_us.h5')
dsge.eval_chains(burnin=0.1, show_charts=False)

# print(dsge.posterior_table)

# Extraxct state variables
df_states_hat, df_states_se = dsge.states()

a = 1

