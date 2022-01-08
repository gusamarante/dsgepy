from pydsge import DSGE
import matplotlib.pyplot as plt
from sympy import symbols, Matrix


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


# ======================
# ===== SIMULATION =====
# ======================

calib_dict = {sigma: 1.3,
              varphi: 1,
              alpha: 0.4,
              beta: 0.997805,
              theta: 0.75,
              phi_pi: 1.5,
              phi_y: 0.2,
              rho_a: 0.9,
              sigma_a: 1.1,
              rho_v: 0.5,
              sigma_v: 0.3,
              sigma_pi: 0.8}

dsge_simul = DSGE(endog, endogl, exog, expec, equations,
                  calib_dict=calib_dict,
                  obs_equations=obs_equations,
                  obs_names=obs_names)

# IRFs from the theoretical Model
# dsge_simul.irf(periods=24, show_charts=True)

# Check existance and uniqueness
# print(dsge_simul.eu)

# Simulate observations
df_obs, df_states = dsge_simul.simulate(n_obs=200, random_seed=1)

df_states = df_states.tail(100).reset_index(drop=True)
df_obs = df_obs.tail(100).reset_index(drop=True)

# df_obs.plot()
# plt.show()


# =============================
# ===== MODEL ESTIMATION  =====
# =============================
# Not all parameters need to be estimated, these are going to be calibrated
calib_param = {varphi: 1, alpha: 0.4, beta: 0.997805}
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

dsge.estimate(nsim=10, ck=0.1, file_path='snkm.h5')

# dsge.eval_chains(burnin=0.1, show_charts=True)
dsge.eval_chains(burnin=0.1, show_charts=False)

# print(dsge.posterior_table)

# IRFs from the estimated Model
# dsge.irf(periods=24, show_charts=True)

# Extraxct state variables  # TODO compare with the originals
# df_states_hat, df_states_se = dsge.states()

# TODO Analyze residuals VS the hypothesis
# dsge._get_residuals()

# Historical Decomposition
df_hd = dsge.hist_decomp()
df_hd.loc['Inflation'].plot(kind='bar', stacked=True, width=1)
df_obs['Inflation'].plot(color='black')
plt.show()
