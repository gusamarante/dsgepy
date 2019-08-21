import numpy as np
from sympy import symbols, Matrix
from pydsge import DSGE
import matplotlib.pyplot as plt

# ================================
# ===== MODEL ESPECIFICATION =====
# ================================
# endogenous variables at t
y, c, i, l, w, r, pi, mc, k, v, rk, tau, x, m, e, cry, q, psi, n, world, T, s, g, pi_focus, exp_c, exp_i, exp_l, exp_pi, exp_rk, exp_e, exp_cry, eps_b, eps_r, eps_m, eps_pi, eps_a, eps_n, eps_x, eps_exp, eps_g, y_lag1, c_lag1, i_lag1, l_lag1, x_lag1, m_lag1, e_lag1, cry_lag1, world_lag1 = \
    symbols('y, c, i, l, w, r, pi, mc, k, v, rk, tau, x, m, e, cry, q, psi, n, world, T, s, g, pi_focus, exp_c, exp_i, exp_l, exp_pi, exp_rk, exp_e, exp_cry, eps_b, eps_r, eps_m, eps_pi, eps_a, eps_n, eps_x, eps_exp, eps_g, y_lag1, c_lag1, i_lag1, l_lag1, x_lag1, m_lag1, e_lag1, cry_lag1, world_lag1')

endog = Matrix([y, c, i, l, w, r, pi, mc, k, v, rk, tau, x, m, e, cry, q, psi, n, world, T, s, g, pi_focus, exp_c, exp_i, exp_l, exp_pi, exp_rk, exp_e, exp_cry, eps_b, eps_r, eps_m, eps_pi, eps_a, eps_n, eps_x, eps_exp, eps_g, y_lag1, c_lag1, i_lag1, l_lag1, x_lag1, m_lag1, e_lag1, cry_lag1, world_lag1])

# endogenous variables at t - 1
yl, cl, il, ll, wl, rl, pil, mcl, kl, vl, rkl, taul, xl, ml, el, cryl, ql, psil, nl, worldl, Tl, sl, gl, pi_focusl, \
exp_cl, exp_il, exp_ll, exp_pil, exp_rkl, exp_el, exp_cryl, \
eps_bl, eps_rl, eps_ml, eps_pil, eps_al, eps_nl, eps_xl, eps_expl, eps_gl, \
y_lag1l, c_lag1l, i_lag1l, l_lag1l, x_lag1l, m_lag1l, e_lag1l, cry_lag1l, world_lag1l = \
    symbols('yl, cl, il, ll, wl, rl, pil, mcl, kl, vl, rkl, taul, xl, ml, el, cryl, ql, psil, nl, worldl, Tl, '
            'sl, gl, pi_focusl, exp_cl, exp_il, exp_ll, exp_pil, exp_rkl, exp_el, exp_cryl, '
            'eps_bl, eps_rl, eps_ml, eps_pil, eps_al, eps_nl, eps_xl, eps_expl, eps_gl, y_lag1l, c_lag1l, i_lag1l, l_lag1l, x_lag1l, m_lag1l, e_lag1l, cry_lag1l, world_lag1l')

endogl = Matrix([yl, cl, il, ll, wl, rl, pil, mcl, kl, vl, rkl, taul, xl, ml, el, cryl, ql, psil, nl, worldl,
                 Tl, sl, gl, pi_focusl, exp_cl, exp_il, exp_ll, exp_pil, exp_rkl, exp_el, exp_cryl,
                 eps_bl, eps_rl, eps_ml, eps_pil, eps_al, eps_nl, eps_xl, eps_expl, eps_gl, y_lag1l, c_lag1l, i_lag1l,
                 l_lag1l, x_lag1l, m_lag1l, e_lag1l, cry_lag1l, world_lag1l])

# exogenous shocks
xi_b, xi_a, xi_g, xi_r, xi_pi, xi_n, xi_x, xi_m, xi_exp, xi_e, xi_cry, xi_world, xi_tau, xi_s = \
    symbols('xi_b, xi_a, xi_g, xi_r, xi_pi, xi_n, xi_x, xi_m, xi_exp, xi_e, xi_cry, xi_world, xi_tau, xi_s')

exog = Matrix([xi_b, xi_a, xi_g, xi_r, xi_pi, xi_n, xi_x, xi_m, xi_exp, xi_e, xi_cry, xi_world, xi_tau, xi_s])

# expectational shocks
eta_c, eta_i, eta_l, eta_pi, eta_rk, eta_e, eta_cry = \
    symbols('eta_c, eta_i, eta_l, eta_pi, eta_rk, eta_e, eta_cry')

expec = Matrix([eta_c, eta_i, eta_l, eta_pi, eta_rk, eta_e, eta_cry])

# parameters
sigma_c, sigma_h, sigma_l, beta, theta_lag, theta_mc, theta_e, theta_cry, alpha, varphi, delta, chi_s, chi_n, \
gamma_r, gamma_pi, gamma_exp_pi, gamma_y, theta_back, theta_forw, kappa_world, kappa_e, kappa_cry, \
mu_c, mu_i, mu_cry, c_ss, w_ss, l_ss, rk_ss, tau_ss, kn_ss, g_ss, i_ss, x_ss, m_ss, \
rho_b, rho_a, rho_g, rho_r, rho_pi, rho_n, rho_x, rho_m, rho_exp, rho_e1, rho_e2, rho_cry, rho_world, \
rho_tau, rho_s, sigma_b, sigma_a, sigma_g, sigma_r, sigma_pi, sigma_n, sigma_x, sigma_m, sigma_exp, sigma_e, \
sigma_cry, sigma_world, sigma_tau, sigma_s, sigma_employ, sigma_durable = \
    symbols('sigma_c, sigma_h, sigma_l, beta, theta_lag, theta_mc, theta_e, theta_cry, alpha, varphi, delta, '
            'chi_s, chi_n, gamma_r, gamma_pi, gamma_exp_pi, gamma_y, theta_back, theta_forw, kappa_world, '
            'kappa_e, kappa_cry, mu_c, mu_i, mu_cry, c_ss, w_ss, l_ss, rk_ss, tau_ss, kn_ss, g_ss, i_ss, x_ss, '
            'm_ss, rho_b, rho_a, rho_g, rho_r, rho_pi, rho_n, rho_x, rho_m, rho_exp, rho_e1, rho_e2, rho_cry, '
            'rho_world, rho_tau, rho_s, sigma_b, sigma_a, sigma_g, sigma_r, sigma_pi, sigma_n, sigma_x, sigma_m, '
            'sigma_exp, sigma_e, sigma_cry, sigma_world, sigma_tau, sigma_s, sigma_employ, sigma_durable')

# ===== model equations =====
# equilibrium conditions
eq01 = - c + (sigma_h / (1 + sigma_h)) * cl + (1 / (1 + sigma_h)) * exp_c \
       + (w_ss * l_ss * (sigma_c - 1)) / (c_ss * sigma_c * (1 + sigma_h)) * (l - exp_l) \
       - (1 - sigma_h) / (sigma_c * (1 + sigma_h)) * (r - exp_pi + eps_b)

eq02 = - m + mu_c * c + mu_i * i - mu_cry * cry + eps_m

eq03 = - pi + (1 / (1 + beta * theta_lag)) * (
        beta * exp_pi + theta_lag * pil + theta_mc * mc + theta_e * (e - beta * exp_e) + theta_cry * (
        cry - beta * exp_cry)) + eps_pi

eq04 = -k - v + l + w

eq05 = - mc + alpha * v + (1 - alpha) * w - eps_a

eq06 = - y + alpha * k + (1 - alpha) * l + eps_a

eq07 = - w + (1 / (1 - tau_ss)) * tau + sigma_l * l + (1 / (1 - sigma_h)) * c - (sigma_h / (1 - sigma_h)) * cl

eq08 = - i + (1 / (1 + beta)) * il + (beta / (1 + beta)) * exp_i + (1 / (varphi * (1 + beta))) * q

eq09 = - k + (1 - delta) * kl + delta * il

eq10 = - rk + ((1 - delta) / (rk_ss + 1 - delta)) * q + (rk_ss / (rk_ss + 1 - delta)) * (
        v - (1 / (1 - tau_ss)) * tau) - ql

eq11 = - psi + exp_rk - (r - exp_pi)

eq12 = - psi - chi_s * (n - q - k)

eq13 = -n + kn_ss * rk - (kn_ss - 1) * (psil + rl - pi) + chi_n * nl - eps_n

eq14 = - x + kappa_world * world + kappa_e * e + kappa_cry * cry + eps_x

eq15 = - r + gamma_r * rl + gamma_pi * pi + gamma_exp_pi * exp_pi + gamma_y * y + eps_r

eq16 = - T + tau - s - g_ss * (g - y + eps_g)

eq17 = -y + c_ss * c + i_ss * i + g_ss * g + x_ss * x - m_ss * m

eq18 = - pi_focus + theta_back * pi + theta_forw * exp_pi + eps_exp

# AR(1)
eq19 = - eps_b + rho_b * eps_bl + sigma_b * xi_b
eq20 = - eps_a + rho_a * eps_al + sigma_a * xi_a
eq21 = - eps_g + rho_g * eps_gl + sigma_g * xi_g
eq22 = - eps_r + rho_r * eps_rl + sigma_r * xi_r
eq23 = - eps_pi + rho_pi * eps_pil + sigma_pi * xi_pi
eq24 = - eps_n + rho_n * eps_nl + sigma_n * xi_n
eq25 = - eps_x + rho_x * eps_xl + sigma_x * xi_x
eq26 = - eps_m + rho_m * eps_ml + sigma_m * xi_m
eq27 = - eps_exp + rho_exp * eps_expl + sigma_exp * xi_exp
eq28 = - e + rho_e1 * el + rho_e2 * xi_cry + xi_e
eq29 = - cry + rho_cry * cryl + xi_cry
eq30 = - world + rho_world * worldl + xi_world
eq31 = -tau + rho_tau * taul + xi_tau
eq32 = -s + rho_s * sl + xi_s

# expectational errors
eq33 = c - exp_cl - eta_c
eq34 = i - exp_il - eta_i
eq35 = l - exp_ll - eta_l
eq36 = pi - exp_pil - eta_pi
eq37 = rk - exp_rkl - eta_rk
eq38 = e - exp_el - eta_e
eq39 = cry - exp_cryl - eta_cry

# Auxiliar for obs equation
eq40 = y_lag1 - yl
eq41 = c_lag1 - cl
eq42 = i_lag1 - il
eq43 = l_lag1 - ll
eq44 = x_lag1 - xl
eq45 = m_lag1 - ml
eq46 = e_lag1 - el
eq47 = cry_lag1 - cryl
eq48 = world_lag1 - worldl

state_equations = Matrix([eval('eq' + str(n).zfill(2)) for n in range(1, 49)])

# Observation Matrix
obs01 = y - y_lag1
obs02 = (1 - sigma_durable) * (c - c_lag1) + sigma_durable * (i - i_lag1)
obs03 = i - i_lag1
obs04 = (1 / sigma_employ) * (l - l_lag1)
obs05 = r
obs06 = pi
obs07 = x - x_lag1
obs08 = m - m_lag1
obs09 = e - e_lag1
obs10 = cry - cry_lag1
obs11 = pi_focus
obs12 = s
obs13 = tau
obs14 = world - world_lag1

obs_equations = Matrix([eval('obs' + str(n).zfill(2)) for n in range(1, 15)])

# =============================
# ===== MODEL ESTIMATION  =====
# =============================

# calibrated parameters
calib_param = {delta: 0.025,
               g_ss: 0.2,
               x_ss: 0.14,
               m_ss: 0.13,
               kn_ss: 1.2,
               chi_n: 0.99,
               sigma_c: 0.92,
               sigma_h: 0.76,
               sigma_l: 1.1,
               beta: 0.975,
               theta_lag: 0.71,
               theta_mc: 0.015,
               theta_e: 0.054,
               theta_cry: 0.054,
               alpha: 0.29,
               varphi: 3.6,
               chi_s: 0.34,
               gamma_r: 0.79,
               gamma_pi: 0.77,  # very weird
               gamma_exp_pi: 3.3,
               gamma_y: 0.15,
               theta_back: 0.14,
               theta_forw: 0.14,
               kappa_world: 1.4,
               kappa_e: 0.10,
               kappa_cry: 0.24,
               mu_c: 0.17,
               mu_i: 0.8,
               mu_cry:0.17,
               c_ss: 0.63,
               w_ss: 7,
               l_ss: 2.5,
               rk_ss: 0.09,
               tau_ss: 0.3,
               i_ss: 0.19,
               rho_b: 0.5,
               rho_a: 0.5,
               rho_g: 0.5,
               rho_r: 0.5,
               rho_pi: 0.5,
               rho_n: 0.5,
               rho_x: 0.5,
               rho_m: 0.5,
               rho_exp: 0.5,
               rho_e1: 0.5,
               rho_e2: 0.5,
               rho_cry: 0.5,
               rho_world: 0.5,
               rho_tau: 0.5,
               rho_s: 0.5,
               sigma_b: 0.1,
               sigma_a: 0.1,
               sigma_g: 0.1,
               sigma_r: 0.1,
               sigma_pi: 0.1,
               sigma_n: 0.1,
               sigma_x: 0.1,
               sigma_m: 0.1,
               sigma_exp: 0.1,
               sigma_e: 0.1,
               sigma_cry: 0.1,
               sigma_world: 0.1,
               sigma_tau: 0.1,
               sigma_s: 0.1,
               sigma_employ: 2.6,
               sigma_durable: 0.05}

dsge = DSGE(endog, endogl, exog, expec, state_equations, obs_equations,
            calib_dict=calib_param, verbose=True)

print(dsge.eu)
