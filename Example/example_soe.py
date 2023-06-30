from dsgepy import DSGE
from sympy import symbols, Matrix


# ================================
# ===== MODEL ESPECIFICATION =====
# ================================
# endogenous variables at t
c, r, r_s, pi, pi_s, pi_h, pi_f, y, y_s, s, q, v_a, v_s, v_q, exp_c, exp_pi, exp_pih, exp_pif, exp_pis, exp_q = \
    symbols('c, r, r_s, pi, pi_s, pi_h, pi_f, y, y_s, s, q, v_a, v_s, v_q, exp_c, exp_pi, exp_pih, exp_pif, exp_pis, exp_q')
endog = Matrix([c, r, r_s, pi, pi_s, pi_h, pi_f, y, y_s, s, q, v_a, v_s, v_q, exp_c, exp_pi, exp_pih, exp_pif, exp_pis, exp_q])

# endogenous variables at t - 1
cl, rl, r_sl, pil, pi_sl, pi_hl, pi_fl, yl, y_sl, sl, ql, v_al, v_sl, v_ql, exp_cl, exp_pil, exp_pihl, exp_pifl, exp_pisl, exp_ql = \
    symbols('cl, rl, r_sl, pil, pi_sl, pi_hl, pi_fl, yl, y_sl, sl, ql, v_al, v_sl, v_ql, exp_cl, exp_pil, exp_pihl, exp_pifl, exp_pisl, exp_ql')
endogl = Matrix([cl, rl, r_sl, pil, pi_sl, pi_hl, pi_fl, yl, y_sl, sl, ql, v_al, v_sl, v_ql, exp_cl, exp_pil, exp_pihl, exp_pifl, exp_pisl, exp_ql])

# exogenous shocks
eps_a, eps_h, eps_f, eps_q, eps_s, eps_r, eps_pis, eps_ys, eps_rs = \
    symbols('eps_a, eps_h, eps_f, eps_q, eps_s, eps_r, eps_pis, eps_ys, eps_rs')
exog = Matrix([eps_a, eps_h, eps_f, eps_q, eps_s, eps_r, eps_pis, eps_ys, eps_rs])

# expectational shocks
eta_c, eta_pi, eta_pih, eta_pif, eta_pis, eta_q = \
    symbols('eta_c, eta_pi, eta_pih, eta_pif, eta_pis, eta_q')
expec = Matrix([eta_c, eta_pi, eta_pih, eta_pif, eta_pis, eta_q])

# parameters
sigma, h, phi, beta, alpha, eta, delta_h, delta_f, theta_h, theta_f, rho_r, mu_pi, mu_q, mu_y, sigma_r, rho_a, sigma_a, rho_s, sigma_s, rho_q, sigma_q, rho_pis, sigma_pis, rho_ys, sigma_ys, rho_rs, sigma_rs, sigma_h, sigma_f = \
    symbols('sigma, h, phi, beta, alpha, eta, delta_h, delta_f, theta_h, theta_f, rho_r, mu_pi, mu_q, mu_y, sigma_r, rho_a, sigma_a, rho_s, sigma_s, rho_q, sigma_q, rho_pis, sigma_pis, rho_ys, sigma_ys, rho_rs, sigma_rs, sigma_h, sigma_f')

# Summary parameters
lambda_h = (1 - theta_h * beta) * (1 - theta_h) / theta_h
lambda_f = (1 - theta_f * beta) * (1 - theta_f) / theta_f

# model (state) equations
eq01 = (1+h)*c - h*cl - exp_c + ((1-h)/sigma)*r - ((1-h)/sigma)*exp_pi
eq02 = (1+delta_h)*pi_h - beta*exp_pih - delta_h*pi_hl - lambda_h*phi*y + lambda_h*(1+phi)*v_a - lambda_h*alpha*s - lambda_h*(sigma/(1-h))*c + lambda_h*(sigma/(1-h))*h*cl - lambda_h*sigma_h*eps_h
eq03 = (1+delta_f)*pi_f - beta*exp_pif - delta_f*pi_fl - lambda_f*q + lambda_f*(1-alpha)*s - lambda_f*sigma_f*eps_f
eq04 = exp_q - q - r + exp_pi + r_s - exp_pis - v_q
eq05 = s - sl - pi_f + pi_h - v_s
eq06 = y - (1-alpha)*c - alpha*eta*q - alpha*eta*s - alpha*y_s
eq07 = pi - (1-alpha)*pi_h - alpha*pi_f
eq08 = v_a - rho_a*v_al - sigma_a*eps_a
eq09 = v_s - rho_s*v_sl - sigma_s*eps_s
eq10 = v_q - rho_q*v_ql - sigma_q*eps_q
eq11 = pi_s - rho_pis*pi_sl - sigma_pis*eps_pis
eq12 = y_s - rho_ys*y_sl - sigma_ys*eps_ys
eq13 = r_s - rho_rs*r_sl - sigma_rs*eps_rs
eq14 = r - rho_r*rl - (1-rho_r)*mu_pi*pi - (1-rho_r)*mu_q*q - (1-rho_r)*mu_y*y - sigma_r*eps_r
eq15 = c - exp_cl - eta_c
eq16 = pi - exp_pil - eta_pi
eq17 = pi_h - exp_pihl - eta_pih
eq18 = pi_f - exp_pifl - eta_pif
eq19 = pi_s - exp_pisl - eta_pis
eq20 = q - exp_ql - eta_q

equations = Matrix([eq01, eq02, eq03, eq04, eq05, eq06, eq07, eq08, eq09, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20])

# =======================
# ===== CALIBRATION =====
# =======================

calib_dict = {
    sigma: 0.81,
    h: 0.91,
    phi: 1.58,
    beta: 0.99,
    alpha: 0.45,
    eta: 0.36,
    delta_h: 0.25,
    delta_f: 0.05,
    theta_h: 0.77,
    theta_f: 0.68,
    rho_r: 0.7,
    mu_pi: 1.2,
    mu_q: 0,
    mu_y: 0,
    sigma_r: 0.36,
    rho_a: 0.81,
    sigma_a: 5.17,
    rho_s: 0.81,
    sigma_s: 5.45,
    rho_q: 0.68,
    sigma_q: 0.75,
    rho_pis: 0.26,
    sigma_pis: 0.42,
    rho_ys: 0.72,
    sigma_ys: 0.54,
    rho_rs: 0.89,
    sigma_rs: 0.22,
    sigma_h: 1.05,
    sigma_f: 4.43,
}

dsge_simul = DSGE(endog=endog,
                  endogl=endogl,
                  exog=exog,
                  expec=expec,
                  state_equations=equations,
                  calib_dict=calib_dict,
                  )

# Check existance and uniqueness
print(dsge_simul.eu)

# IRFs from the theoretical Model
dsge_simul.irf(periods=24, show_charts=True)
