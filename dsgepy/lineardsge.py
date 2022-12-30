"""
Author: Gustavo Amarante
Classes and functions for linearized DSGEs.
"""

# import warnings
import pandas as pd
from tqdm import tqdm
from scipy.linalg import qz
from math import ceil, floor
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sympy import simplify, Matrix
# from tables import PerformanceWarning
from dsgepy.pycsminwel import csminwel
from scipy.optimize import minimize, basinhopping
from numpy.random import multivariate_normal, rand, seed
from scipy.stats import beta, gamma, invgamma, norm, uniform
from numpy.linalg import svd, inv, eig, matrix_power, LinAlgError
from numpy import diagonal, vstack, array, eye, where, diag, sqrt, hstack, zeros, \
    arange, exp, log, inf, nan, isnan, isinf, set_printoptions, matrix, linspace, ndarray

# pd.set_option('display.max_columns', 20)
# set_printoptions(precision=4, suppress=True, linewidth=150)
# warnings.filterwarnings('ignore', category=RuntimeWarning)
# warnings.filterwarnings('ignore', category=PerformanceWarning)
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class DSGE(object):
    """
    This is the main class which holds a DSGE model with all its attributes and methods.
    """

    optim_methods = ['csminwel', 'basinhopping', 'bfgs']

    resid = None
    chains = None
    prior_info = None
    has_solution = False
    posterior_table = None
    acceptance_rate = None

    def __init__(self, endog, endogl, exog, expec=None, state_equations=None, obs_equations=None, estimate_params=None,
                 calib_dict=None, prior_dict=None, obs_data=None, verbose=False, optim_method='csminwel',
                 obs_names=None):
        """
        Model declaration requires passing SymPy symbols as variables and parameters. Some arguments can be left empty
        if you are working with simulations of calibrated models.

        @param endog: SymPy matrix of symbols containing the endogenous variables.
        @param endogl: SymPy matrix of symbols containing the lagged endogenous variables.
        @param exog: SymPy matrix of symbols containing the exogenous shocks.
        @param expec: SymPy matrix of symbols containing the expectational errors.
        @param state_equations: SymPy matrix of symbolic expressions representing the model's equilibrium conditions,
                                with zeros on the right-hand side of the equality.
        @param obs_equations: SymPy matrix of symbolic expressions representing the model's observation equations, with
                              observable variables on the left-hand side of the equation. This is only required if the
                              model is going to be estimated. You do not need to provide observation equations to run
                              simulations on a calibrated model.
        @param estimate_params: SymPy matrix of symbols containing the parameters that are free to be estimated.
        @param calib_dict: dict. Keys are the symbols of parameters that are going to be calibrated, and values are
                           their calibrated value.
        @param prior_dict: dict. Entries must have symbols of parameters that are going to be estimated. Values are
                           dictionaries containing the following entries:
                           - 'dist': prior distribution. 'normal', 'beta', 'gamma' or 'invgamma'.
                           - 'mean': mean of the prior distribution.
                           - 'std': standard deviation of the prior distribution.
                           - 'label': str with name/representation of the estimated parameter. This argument accepts
                                      LaTeX representations.
        @param obs_data: pandas DataFrame with the observable variables. Columns must be in the same order as the
                         'obs_equations' declarations.
        @param verbose: <not implemented yet>
        """
        # TODO document residuals require that n_obs == n_exog
        # TODO document all of the attributes
        # TODO full review of documentation
        # TODO add functionality to save the chart to a given path

        assert optim_method in self.optim_methods, f"optimization method '{optim_method}' not implemented"

        self.optim_method = optim_method
        self.verbose = verbose
        self.endog = endog
        self.endogl = endogl
        self.exog = exog
        self.expec = expec
        self.params = estimate_params
        self.state_equations = state_equations
        self.obs_equations = obs_equations
        self.obs_names = obs_names
        self.prior_dict = prior_dict
        self.data = obs_data
        self.n_state = len(endog)
        self.n_exog = len(exog)
        self.n_obs = len(endog) if obs_equations is None else len(obs_equations)
        self.n_param = None if estimate_params is None else len(estimate_params)

        if (obs_equations is None) and (obs_data is None):
            generate_obs = True
        else:
            generate_obs = False

        self._get_jacobians(generate_obs=generate_obs)

        if estimate_params is None:
            # If no parameters are going to be estimated, calibrate the whole model
            self.Gamma0, self.Gamma1, self.Psi, self.Pi, self.C_in, self.obs_matrix, self.obs_offset = \
                self._eval_matrix(calib_dict, to_array=True)

            self.G1, self.C_out, self.impact, self.fmat, self.fwt, self.ywt, self.gev, self.eu, self.loose = \
                gensys(self.Gamma0, self.Gamma1, self.C_in, self.Psi, self.Pi)

            self.has_solution = True
        else:
            # Otherwise, calibrate only the required parameters, if calibration is requested
            if calib_dict is not None:
                self.Gamma0, self.Gamma1, self.Psi, self.Pi, self.C_in, self.obs_matrix, self.obs_offset = \
                    self._eval_matrix(calib_dict, to_array=False)

            self.prior_info = self._get_prior_info()

    def simulate(self, n_obs=100, random_seed=None):
        """
        Given a calibration or estimated model, simulates values of the endogenous variables based on random samples of
        the exogenous shocks.
        @param n_obs: number of observation in the time dimension.
        @param random_seed: random seed for the simulation.
        @return: pandas DataFrame. 'df_obs' contains the simualtions for the observable variables. 'df_state' contains
                 the simulations for the state/endogenous variables.
        """

        assert self.has_solution, "No solution was generated yet"

        if not (random_seed is None):
            seed(random_seed)

        kf = KalmanFilter(self.G1, self.obs_matrix, self.impact @ self.impact.T, None,
                          self.C_out.reshape(self.n_state), self.obs_offset.reshape(self.n_obs))
        simul_data = kf.sample(n_obs)

        state_names = [str(s) for s in list(self.endog)]
        if self.obs_names is None:
            obs_names = [f'obs {i+1}' for i in range(self.obs_matrix.shape[0])]
        else:
            obs_names = self.obs_names

        df_obs = pd.DataFrame(data=simul_data[1], columns=obs_names)
        df_states = pd.DataFrame(data=simul_data[0], columns=state_names)

        return df_obs, df_states

    def estimate(self, file_path, nsim=1000, ck=0.2, save_interval=100):
        """
        Run the MCMC estimation.
        @param file_path: str. Save path where the MCMC chains are saved. The file format is HDF5 (.h5). This file
                          format gets very heavy but has very fast read/write speed. If the file already exists, the
                          estimation will resume from these previously simulated chains.
        @param nsim: Length of the MCMC chains to be generated. If the chains are already stable, this is the number of
                     draws from the posterior distribution.
        @param ck: float. Scaling factor of the hessian matrix of the mode of the posterior distribution, which is used
                   as the covariance matrix for the MCMC algorithm. Bayesian literature says this value needs to be
                   calibrated in order to achieve your desired acceptance rate from the posterior draws.
        @param save_interval: int. Save the MCMC chains every `save_interval` iterations. This is useful in case
                              something goes wrong and interrupts the program, so you can restart the MCMC sampling
                              where it last saved.
        @return: the 'chains' attribute of this DSGE instance is generated.
        """

        try:
            df_chains = pd.read_hdf(file_path, key='chains')
            sigmak = pd.read_hdf(file_path, key='sigmak')
            start = df_chains.index[-1]

        except FileNotFoundError:
            def obj_func(theta_irr):
                theta_irr = {k: v for k, v in zip(self.params, theta_irr)}
                theta_res = self._irr2res(theta_irr)
                return -1 * self._calc_posterior(theta_res)

            theta_res0 = {k: v for k, v in zip(self.params, self.prior_info['mean'].values)}
            theta_irr0 = self._res2irr(theta_res0)
            theta_irr0 = array(list(theta_irr0.values()))

            # Optimization to find the posterior mode
            if self.optim_method == 'csminwel':
                _, theta_mode_irr, _, h, _, _, retcodeh = csminwel(fcn=obj_func, x0=theta_irr0, crit=1e-14, nit=100,
                                                                   verbose=True, h0=1*eye(len(theta_irr0)))
                theta_mode_irr = {k: v for k, v in zip(self.params, theta_mode_irr)}
                theta_mode_res = self._irr2res(theta_mode_irr)
                sigmak = ck * h

            elif self.optim_method == 'bfgs':
                res = minimize(obj_func, theta_irr0, options={'disp': True}, method='BFGS')
                theta_mode_irr = {k: v for k, v in zip(self.params, res.x)}
                theta_mode_res = self._irr2res(theta_mode_irr)
                sigmak = ck * res.hess_inv

            elif self.optim_method == 'basinhopping':
                res = basinhopping(obj_func, theta_irr0)
                theta_mode_irr = {k: v for k, v in zip(self.params, res.x)}
                theta_mode_res = self._irr2res(theta_mode_irr)
                sigmak = ck * res.hess_inv

            else:
                msg = f"optimization method '{self.optim_method}' not implemented"
                raise NotImplementedError(msg)

            if self.verbose:
                print('===== Posterior Mode =====')
                print(theta_mode_res, '\n')
                print('===== MH jump covariance =====')
                print(sigmak, '\n')
                print('===== Eigenvalues of MH jump convariance =====')
                print(eig(sigmak)[0], '\n')

            df_chains = pd.DataFrame(columns=[str(p) for p in list(self.params)], index=range(nsim))
            df_chains.loc[0] = list(theta_mode_res.values())
            start = 0

        # Metropolis-Hastings
        muk = zeros(self.n_param)
        accepted = 0

        for ii in tqdm(range(start + 1, start+nsim), 'Metropolis-Hastings'):
            theta1 = {k: v for k, v in zip(self.params, df_chains.loc[ii - 1].values)}
            pos1 = self._calc_posterior(theta1)
            omega1 = self._res2irr(theta1)
            omega2 = array(list(omega1.values())) + multivariate_normal(muk, sigmak)
            omega2 = {k: v for k, v in zip(self.params, omega2)}
            theta2 = self._irr2res(omega2)
            pos2 = self._calc_posterior(theta2)

            ratio = exp(pos2 - pos1)

            if ratio > rand(1)[0]:
                accepted += 1
                df_chains.loc[ii] = list(theta2.values())
            else:
                df_chains.loc[ii] = df_chains.loc[ii - 1]

            # Save the partial chains
            if ii % save_interval == 0:
                store = pd.HDFStore(file_path)
                store['chains'] = df_chains
                store['sigmak'] = pd.DataFrame(data=sigmak)
                store.close()

        store = pd.HDFStore(file_path)
        store['chains'] = df_chains
        store['sigmak'] = pd.DataFrame(data=sigmak)
        store.close()

        self.chains = df_chains.astype(float)
        self.acceptance_rate = accepted / nsim

        if self.verbose:
            print('Acceptance rate:', 100 * (accepted / nsim), 'percent')

    def eval_chains(self, burnin=0.3, conf=0.90, load_chain=None, show_charts=False):
        """
        This function evaluates the chains generated in the estimation step and calibrates the model with the posterior
        mode. It also has functionalities to plot the generated chains and posterior distributions.
        @param burnin: int or float. Number of observations on the begging of the chain that are going to be dropped to
                       compute posterior statistics.
        @param conf: Level of credibility for the credibility intervals inferred from the posteriors.
        @param load_chain: str. Save pathe of the HDF5 file with the chains. Only required if the chains were not loaded
                           in the estimation step.
        @param show_charts: bool. If True, prior-posterior chart is shown. Red line are the theoretical prior densities,
                            blues bars are the empirical posterior densities.
        @return: the 'posterior_table' attribute of this DSGE instance is generated.
        """

        if not (load_chain is None):
            try:
                self.chains = pd.read_hdf(load_chain, key='chains').astype(float)
            except FileNotFoundError:
                raise FileNotFoundError('Chain file not found')

        assert not (self.chains is None), 'There are no loaded chains'
        chain_size = self.chains.shape[0]

        if type(burnin) is float and 0 <= burnin < 1:
            df_chains = self.chains.iloc[int(chain_size * burnin):]
        elif type(burnin) is int and burnin < chain_size:
            df_chains = self.chains.iloc[burnin + 1:]
        else:
            raise ValueError("'burnin' must be either an int smaller than the chain size or a float between 0 and 1")

        if show_charts:
            self._plot_chains(chains=df_chains, show_charts=show_charts)
            self._plot_prior_posterior(chains=df_chains, show_charts=show_charts)

        self.posterior_table = self._posterior_table(chains=df_chains, conf=conf)

        # Calibrate model with the posterior mode
        calib_dict = self.posterior_table['posterior mode'].to_dict()

        self.Gamma0, self.Gamma1, self.Psi, self.Pi, self.C_in, self.obs_matrix, self.obs_offset = \
            self._eval_matrix(calib_dict, to_array=True)

        self.G1, self.C_out, self.impact, self.fmat, self.fwt, self.ywt, self.gev, self.eu, self.loose = \
            gensys(self.Gamma0, self.Gamma1, self.C_in, self.Psi, self.Pi)

        self.has_solution = True

        if self.n_obs == self.n_exog:
            self.resid = self._get_residuals()
        else:
            self.resid = None

    def irf(self, periods=12, show_charts=False):
        """
        Generates and plot impulse-response functions (IRFs).
        @param periods: int. Number of periods in the IRFs.
        @param show_charts: bool. If true, plot the IRFs for each shock.
        @return: a MultiIndex pandas.DataFrame with (shock, period) as the index and variables as the columns.
        """
        # TODO IRF standard errors

        assert self.has_solution, 'Model does not have a solution yet. Cannot compute IRFs'

        # number os periods ahead plus the contemporaneous effect
        periods = periods + 1

        # Initialiaze with zeros
        irf_values = zeros((periods, self.n_state, self.n_exog))
        obs_irf_values = zeros((periods, self.n_obs, self.n_exog))

        # Compute the IRFs
        partial = self.impact
        for tt in range(periods):
            irf_values[tt, :, :] = partial
            obs_irf_values[tt, :, :] = self.obs_offset + self.obs_matrix @ partial
            partial = self.G1 @ partial

        # organize state variables in a MultiIndex DataFrame
        col_names = [str(var) for var in self.endog]
        idx_names = [str(var) for var in self.exog]
        mindex = pd.MultiIndex.from_product([idx_names, list(range(periods))], names=['exog', 'periods'])
        df_irf = pd.DataFrame(columns=col_names, index=mindex)

        for count, var in enumerate(idx_names):
            df_irf.loc[var] = irf_values[:, :, count]

        # organize observed variables in a MultiIndex DataFrame
        if self.obs_names is None:
            col_names_obs = ['obs ' + str(var + 1).zfill(2) for var in range(self.n_obs)]
        else:
            col_names_obs = self.obs_names

        df_irf_obs = pd.DataFrame(columns=col_names_obs, index=mindex)

        for count, var in enumerate(idx_names):
            df_irf_obs.loc[var] = obs_irf_values[:, :, count]

        # Charts
        if show_charts:
            shocks = df_irf.index.get_level_values(0).unique()

            n_subplots = df_irf.shape[1] + df_irf_obs.shape[1]
            n_rows = floor(sqrt(n_subplots))
            n_cols = ceil(n_subplots / n_rows)
            subplot_shape = (n_rows, n_cols)

            for ss in shocks:
                plot_irfs = pd.concat([df_irf.loc[ss], df_irf_obs.loc[ss]], axis=1)
                ax = plot_irfs.plot(title=f'Impulse Response Functions: Shock {ss}',
                                    subplots=True,
                                    layout=subplot_shape,
                                    legend=None,
                                    grid=True,
                                    figsize=(12, 7),
                                    sharey=False)

                zero_irf = list(plot_irfs.columns[(plot_irfs.abs() < 1e-10).all()])

                for axis, subptitle in zip(ax.ravel(), list(plot_irfs.columns)):
                    axis.set_title(subptitle)
                    if subptitle in zero_irf:
                        axis.set_ylim((-1, 1))

                plt.tight_layout()
                plt.show()

        # Concatenate state and observed variables
        df_irf = pd.concat([df_irf, df_irf_obs], axis=1)

        return df_irf

    def states(self, smoothed=True):

        kf = KalmanFilter(self.G1, self.obs_matrix, self.impact @ self.impact.T, None,
                          self.C_out.reshape(self.n_state), self.obs_offset.reshape(self.n_obs))

        if smoothed:
            states, states_cov = kf.smooth(self.data)
        else:
            states, states_cov = kf.filter(self.data)

        # Orgnize in a dataframe
        states = pd.DataFrame(data=states,
                              index=self.data.index,
                              columns=[str(var) for var in self.endog])

        states_std = pd.DataFrame(index=self.data.index,
                                  columns=[str(var) for var in self.endog])

        for ii in range(states_cov.shape[0]):
            states_std.iloc[ii] = diagonal(states_cov[ii]) ** 0.5

        return states, states_std

    def hist_decomp(self, smoothed=True, show_charts=False):
        # TODO documentation
        states, _ = self.states(smoothed=smoothed)
        x0 = states.iloc[0].values.reshape((-1, 1))
        a = self.obs_matrix
        p0 = self.C_out
        p1 = self.G1
        b = self.impact
        eps = self.resid

        # organize state variables in a MultiIndex DataFrame
        if self.obs_names is None:
            idx_names = ['obs ' + str(var + 1).zfill(2) for var in range(self.n_obs)]
        else:
            idx_names = self.obs_names

        col_names = [str(var) for var in self.exog]
        mindex = pd.MultiIndex.from_product([idx_names, eps.index], names=['obs', 'date'])
        df_hd = pd.DataFrame(columns=col_names, index=mindex)

        # Compute historical decomposition
        for count, tt in tqdm(enumerate(eps.index), 'Computing historical decomposition', disable=not self.verbose):

            matrix_sum = zeros((self.n_state, self.n_exog))

            for k in range(count + 1):
                matrix_sum += eps.iloc[k].values * (matrix_power(p1, count - k) @ b)

            df_hd[df_hd.index.get_level_values(1) == tt] = a @ matrix_sum

        # compute the contributions of steady states and initial conditions
        extra_contributions = []
        for var in self.data.columns:
            extra = self.data[var] - df_hd.loc[var].sum(axis=1)
            extra = extra.rename('Steady State and Initial Conditions')
            extra.index = pd.MultiIndex.from_tuples([(var, i) for i in extra.index])
            extra_contributions.append(extra)

        # Organize the data
        df_extras = pd.concat(extra_contributions, axis=0)
        df_hd = pd.concat([df_extras, df_hd], axis=1)

        if show_charts:
            self._plot_historical_decomposition(df_hd, show_charts=show_charts)

        return df_hd

    def _get_residuals(self):

        c = self.obs_offset
        a = self.obs_matrix
        p0 = self.C_out
        p1 = self.G1
        b = self.impact
        ss = inv(eye(self.n_state) - p1) @ p0

        df_resid = pd.DataFrame(columns=[str(var) for var in self.exog],
                                index=self.data.index)

        # Compute residuals
        acum_schoks = zeros((self.n_state, 1))
        for tt in self.data.index:
            y = self.data.loc[tt].values.reshape((self.n_obs, -1))
            eps = inv(a @ b) @ ((y - c - a @ ss) - a @ acum_schoks)
            df_resid.loc[tt] = eps.flatten()
            acum_schoks = p1 @ (acum_schoks + b @ eps)

        df_resid = df_resid.astype(float)
        return df_resid

    def _get_jacobians(self, generate_obs):
        # State Equations
        self.Gamma0 = self.state_equations.jacobian(self.endog)
        self.Gamma1 = -self.state_equations.jacobian(self.endogl)
        self.Psi = -self.state_equations.jacobian(self.exog)
        self.Pi = -self.state_equations.jacobian(self.expec)
        self.C_in = simplify(self.state_equations
                             - self.Gamma0 @ self.endog
                             + self.Gamma1 @ self.endogl
                             + self.Psi @ self.exog
                             + self.Pi @ self.expec)

        # Obs Equation
        if generate_obs:
            self.obs_matrix = Matrix(eye(self.n_obs))
            self.obs_offset = Matrix(zeros(self.n_obs))
        else:
            self.obs_matrix = self.obs_equations.jacobian(self.endog)
            self.obs_offset = self.obs_equations - self.obs_matrix @ self.endog

    def _calc_posterior(self, theta):
        P = self._calc_prior(theta)
        L = self._log_likelihood(theta)
        f = P + L
        return f*1000  # x1000 is here to increase precison of the posterior mode-finding algorithm.

    def _calc_prior(self, theta):
        prior_dict = self.prior_dict
        df_prior = self.prior_info.copy()
        df_prior['pdf'] = nan

        for param in prior_dict.keys():
            mu = df_prior.loc[str(param)]['mean']
            sigma = df_prior.loc[str(param)]['std']
            dist = df_prior.loc[str(param)]['distribution'].lower()
            theta_i = theta[param]

            # since we are goig to take logs, the density function only needs the terms that depend on
            # theta_i, this will help speed up the code a little and will not affect optimization output.
            if dist == 'beta':
                a = ((mu ** 2) * (1 - mu)) / (sigma ** 2) - mu
                b = a * mu / (1 - mu)
                pdf_i = (theta_i**(a - 1)) * ((1 - theta_i)**(b - 1))

            elif dist == 'gamma':
                a = (mu/sigma)**2
                b = mu/a
                pdf_i = theta_i**(a - 1) * exp(-theta_i/b)

            elif dist == 'invgamma':
                a = (mu/sigma)**2 + 2
                b = mu * (a - 1)
                pdf_i = (theta_i**(- a - 1)) * exp(-b/theta_i)

            elif dist == 'uniform':
                a = mu - sqrt(3) * sigma
                b = 2 * mu - a
                pdf_i = 1/(b - a)

            else:  # Normal
                pdf_i = exp(-((theta_i - mu)**2)/(2 * (sigma**2)))

            df_prior.loc[str(param), 'pdf'] = pdf_i

        df_prior['log pdf'] = log(df_prior['pdf'].astype(float))

        P = df_prior['log pdf'].sum()

        return P

    def _log_likelihood(self, theta):
        Gamma0, Gamma1, Psi, Pi, C_in, obs_matrix, obs_offset = self._eval_matrix(theta, to_array=True)

        for mat in [Gamma0, Gamma1, Psi, Pi, C_in]:
            if isnan(mat).any() or isinf(mat).any():
                return -inf

        G1, C_out, impact, fmat, fwt, ywt, gev, eu, loose = gensys(Gamma0, Gamma1, C_in, Psi, Pi)

        if eu[0] == 1 and eu[1] == 1:
            kf = KalmanFilter(G1, obs_matrix, impact @ impact.T, None, C_out.reshape(self.n_state),
                              obs_offset.reshape(self.n_obs))
            try:
                L = kf.loglikelihood(self.data)
            except (LinAlgError, ValueError):
                L = - inf
        else:
            L = - inf

        return L

    def _eval_matrix(self, theta, to_array):

        if to_array:
            # state matrices
            Gamma0 = array(self.Gamma0.subs(theta)).astype(float)
            Gamma1 = array(self.Gamma1.subs(theta)).astype(float)
            Psi = array(self.Psi.subs(theta)).astype(float)
            Pi = array(self.Pi.subs(theta)).astype(float)
            C_in = array(self.C_in.subs(theta)).astype(float)

            # observation matrices
            obs_matrix = array(self.obs_matrix.subs(theta)).astype(float)
            obs_offset = array(self.obs_offset.subs(theta)).astype(float)
        else:
            # state matrices
            Gamma0 = self.Gamma0.subs(theta)
            Gamma1 = self.Gamma1.subs(theta)
            Psi = self.Psi.subs(theta)
            Pi = self.Pi.subs(theta)
            C_in = self.C_in.subs(theta)

            # observation matrices
            obs_matrix = self.obs_matrix.subs(theta)
            obs_offset = self.obs_offset.subs(theta)

        return Gamma0, Gamma1, Psi, Pi, C_in, obs_matrix, obs_offset

    def _get_prior_info(self):
        prior_info = self.prior_dict

        param_names = [str(s) for s in list(self.params)]

        df_prior = pd.DataFrame(columns=['distribution', 'mean', 'std', 'param a', 'param b'],
                                index=param_names)

        for param in prior_info.keys():
            mu = prior_info[param]['mean']
            sigma = prior_info[param]['std']
            dist = prior_info[param]['dist'].lower()

            if dist == 'beta':
                a = ((mu ** 2) * (1 - mu)) / (sigma ** 2) - mu
                b = a * mu / (1 - mu)

            elif dist == 'gamma':
                a = (mu / sigma) ** 2
                b = mu / a

            elif dist == 'invgamma':
                a = (mu / sigma) ** 2 + 2
                b = mu * (a - 1)

            elif dist == 'uniform':
                a = mu - sqrt(3) * sigma
                b = 2 * mu - a

            else:  # Normal
                a = mu
                b = sigma

            df_prior.loc[str(param)] = [dist, mu, sigma, a, b]

        return df_prior

    def _res2irr(self, theta_res):
        """
        converts the prior distribution from restricted to irrestricted
        :param theta_res:
        :return:
        """
        prior_info = self.prior_info
        theta_irr = theta_res.copy()

        for param in theta_res.keys():
            a = prior_info.loc[str(param)]['param a']
            b = prior_info.loc[str(param)]['param b']
            dist = prior_info.loc[str(param)]['distribution'].lower()
            theta_i = theta_res[param]

            if dist == 'beta':
                theta_irr[param] = log(theta_i / (1 - theta_i))

            elif dist == 'gamma' or dist == 'invgamma':
                theta_irr[param] = log(theta_i)

            elif dist == 'uniform':
                theta_irr[param] = log((theta_i - a) / (b - theta_i))

            else:  # Normal
                theta_irr[param] = theta_i

        return theta_irr

    def _irr2res(self, theta_irr):
        """
        converts the prior distribution from irrestricted to restricted
        :param theta_irr:
        :return:
        """
        prior_info = self.prior_info
        theta_res = theta_irr.copy()

        for param in theta_irr.keys():
            a = prior_info.loc[str(param)]['param a']
            b = prior_info.loc[str(param)]['param b']
            dist = prior_info.loc[str(param)]['distribution'].lower()
            lambda_i = theta_irr[param]

            if dist == 'beta':
                theta_res[param] = exp(lambda_i) / (1 + exp(lambda_i))

            elif dist == 'gamma':
                theta_res[param] = exp(lambda_i)

            elif dist == 'invgamma':
                theta_res[param] = exp(lambda_i)

            elif dist == 'uniform':
                theta_res[param] = (a + b * exp(lambda_i)) / (1 + exp(lambda_i))

            else:  # Normal
                theta_res[param] = lambda_i

        return theta_res

    def _plot_chains(self, chains, show_charts):
        n_cols = int(self.n_param ** 0.5)
        n_rows = n_cols + 1 if self.n_param > n_cols ** 2 else n_cols
        subplot_shape = (n_rows, n_cols)

        plt.figure(figsize=(7*1.61, 7))

        for count, param in enumerate(list(self.params)):
            ax = plt.subplot2grid(subplot_shape, (count // n_cols, count % n_cols))
            ax.plot(chains[str(param)], linewidth=0.5, color='darkblue')
            ax.set_title(self.prior_dict[param]['label'])

        plt.tight_layout()

        if show_charts:
            plt.show()

    def _plot_prior_posterior(self, chains, show_charts):
        n_bins = int(sqrt(chains.shape[0]))  # TODO use the same logic as the IRF charts
        n_cols = int(self.n_param ** 0.5)
        n_rows = n_cols + 1 if self.n_param > n_cols ** 2 else n_cols
        subplot_shape = (n_rows, n_cols)

        plt.figure(figsize=(7 * 1.61, 7))

        for count, param in enumerate(list(self.params)):
            mu = self.prior_info.loc[str(param)]['mean']
            sigma = self.prior_info.loc[str(param)]['std']
            a = self.prior_info.loc[str(param)]['param a']
            b = self.prior_info.loc[str(param)]['param b']
            dist = self.prior_info.loc[str(param)]['distribution']

            ax = plt.subplot2grid(subplot_shape, (count // n_cols, count % n_cols))
            ax.hist(chains[str(param)], bins=n_bins, density=True,
                    color='royalblue', edgecolor='black')
            ax.set_title(self.prior_dict[param]['label'])
            x_min, x_max = ax.get_xlim()
            x = linspace(x_min, x_max, n_bins)

            if dist == 'beta':
                y = beta.pdf(x, a, b)
                ax.plot(x, y, color='red')

            elif dist == 'gamma':
                y = gamma.pdf(x, a, scale=b)
                ax.plot(x, y, color='red')

            elif dist == 'invgamma':
                y = (b ** a) * invgamma.pdf(x, a) * exp((1 - b) / x)
                ax.plot(x, y, color='red')

            elif dist == 'uniform':
                y = uniform.pdf(x, loc=a, scale=b - a)
                ax.plot(x, y, color='red')

            else:  # Normal
                y = norm.pdf(x, loc=mu, scale=sigma)
                ax.plot(x, y, color='red')

        plt.tight_layout()

        if show_charts:
            plt.show()

    def _posterior_table(self, chains, conf):

        low_conf = (1 - conf) / 2
        high_conf = (1 + conf) / 2

        df = self.prior_info[['distribution', 'mean', 'std']]
        df = df.rename({'distribution': 'prior dist', 'mean': 'prior mean', 'std': 'prior std'}, axis=1)
        df['posterior mode'] = chains.mode().mean()
        df['posterior mean'] = chains.mean()
        df[f'posterior {100 * round(low_conf, 3)}%'] = chains.quantile(low_conf)
        df[f'posterior {100 * round(high_conf, 3)}%'] = chains.quantile(high_conf)

        return df

    def _plot_historical_decomposition(self, df_hd, show_charts=False):

        for var in df_hd.index.get_level_values(0).unique():

            fig = plt.figure(figsize=(7 * 1.61, 7))
            ax = fig.gca()

            ax = df_hd.loc[var].plot(kind='bar', stacked=True, width=1, ax=ax, legend='best',
                                     title=var)
            ax.plot(self.data[var], color='black')

            ax.grid(axis='y')
            y_min, y_max = ax.get_ylim()

            if y_min <= 0 <= y_max:
                ax.axhline(y=0, color='black', linewidth=0.5)

            plt.tight_layout()

            if show_charts:
                plt.show()


def gensys(g0, g1, c, psi, pi, div=None, realsmall=0.000001):
    """
    This code is a translation from matlab to python of Christopher Sim's 'gensys'.
    https://dge.repec.org/codes/sims/linre3a/
    """

    # Assertions
    all_inputs = [g0, g1, c, psi, pi]
    assert_cond = all([isinstance(inpt, ndarray) for inpt in all_inputs])
    assert assert_cond, "Inputs 'g0', 'g1', 'c', 'psi' and 'pi' must all be numpy.ndarray"

    unique = False
    eu = [0, 0]
    nunstab = 0
    zxz = False

    n = g1.shape[0]

    if div is None:
        div = 1.01
        fixdiv = False
    else:
        fixdiv = True

    a, b, q, z = qz(g0, g1, 'complex')

    # the numpy.matrix class may be deprecated in a future version of numpy. Should be changed to the ndarray.
    # Scipy's version of 'qz' is different from MATLAB's, Q needs to be hermitian transposed to get same output
    q = array(matrix(q).H)

    for i in range(n):
        if not fixdiv:
            if abs(a[i, i]) > 0:

                divhat = abs(b[i, i]/a[i, i])

                if (1 + realsmall < divhat) and divhat <= div:
                    div = 0.5 * (1 + divhat)

        nunstab = nunstab + (abs(b[i, i]) > div * abs(a[i, i]))

        if (abs(a[i, i]) < realsmall) and abs(b[i, i] < realsmall):
            zxz = True

    if not zxz:
        a, b, q, z, _ = qzdiv(div, a, b, q, z)

    gev = vstack([diagonal(a), diagonal(b)]).T

    if zxz:
        # print('Coincident zeros. Indeterminancy and/or nonexistence')
        eu = [-2, -2]
        return None, None, None, None, None, None, None, eu, None

    q1 = q[:n - nunstab, :]
    q2 = q[n - nunstab:, :]

    # This was in the author's original code, but seems unused
    # z1 = z[:, :n - nunstab].T
    # z2 = z[:, n - nunstab:]
    # a2 = a[n - nunstab:, n - nunstab:]
    # b2 = a[n - nunstab:, n - nunstab:]

    etawt = q2 @ pi
    neta = pi.shape[1]

    # Case for no stable roots
    if nunstab == 0:
        bigev = 0
        # etawt = zeros((0, neta))
        ueta = zeros((0, 0))
        deta = zeros((0, 0))
        veta = zeros((neta, 0))
    else:
        ueta, deta, veta = svd(etawt)
        deta = diag(deta)
        veta = array(matrix(veta).H)
        md = min(deta.shape)
        bigev = where(diagonal(deta[:md, :md]) > realsmall)[0]
        ueta = ueta[:, bigev]
        veta = veta[:, bigev]
        deta = deta[bigev, bigev]
        if deta.ndim == 1:
            deta = diag(deta)

    try:
        if len(bigev) >= nunstab:
            eu[0] = 1
    except TypeError:
        if bigev >= nunstab:
            eu[0] = 1

    # Case for all stable roots
    if nunstab == n:
        etawt1 = zeros((0, neta))
        ueta1 = zeros((0, 0))
        deta1 = zeros((0, 0))
        veta1 = zeros((neta, 0))
        # bigev = 0
    else:
        etawt1 = q1 @ pi
        # ndeta1 = min(n - nunstab, neta)
        ueta1, deta1, veta1 = svd(etawt1)
        deta1 = diag(deta1)
        veta1 = array(matrix(veta1).H)
        md = min(deta1.shape)
        bigev = where(diagonal(deta1[:md, :md]) > realsmall)[0]
        ueta1 = ueta1[:, bigev]
        veta1 = veta1[:, bigev]
        deta1 = deta1[bigev, bigev]
        if deta1.ndim == 1:
            deta1 = diag(deta1)

    if 0 in veta1.shape:
        unique = True
    else:
        loose = veta1 - veta @ veta.T @ veta1
        ul, dl, vl = svd(loose)
        if dl.ndim == 1:
            dl = diag(dl)
        nloose = sum(abs(diagonal(dl))) > realsmall*n
        if not nloose:
            unique = True

    if unique:
        eu[1] = 1
    else:
        # print(f'Indeterminacy. Loose endogenous variables.')
        pass

    tmat = hstack((eye(n - nunstab), -(ueta @ (inv(deta) @ matrix(veta).H) @ veta1 @ deta1 @ matrix(ueta1).H).H))
    G0 = vstack((tmat @ a, hstack((zeros((nunstab, n - nunstab)), eye(nunstab)))))
    G1 = vstack((tmat @ b, zeros((nunstab, n))))

    G0I = inv(G0)
    G1 = G0I @ G1
    usix = arange(n - nunstab, n)
    C = vstack((tmat @ q @ c, inv(a[usix, :][:, usix] - b[usix, :][:, usix]) @ q2 @ c))
    impact = G0I @ vstack((tmat @ q @ psi, zeros((nunstab, psi.shape[1]))))
    fmat = inv(b[usix, :][:, usix]) @ a[usix, :][:, usix]
    fwt = -inv(b[usix, :][:, usix]) @ q2 @ psi
    ywt = G0I[:, usix]

    loose = G0I @ vstack((etawt1 @ (eye(neta) - veta @ matrix(veta).H), zeros((nunstab, neta))))
    G1 = array((z @ G1 @ matrix(z).H).real)
    C = array((z @ C).real)
    impact = array((z @ impact).real)
    loose = array((z @ loose).real)
    ywt = array(z @ ywt)

    return G1, C, impact, fmat, fwt, ywt, gev, eu, loose


def qzdiv(stake, A, B, Q, Z, v=None):
    """
    This code is a translation from matlab to python of Christopher Sim's 'qzdiv'.
    https://dge.repec.org/codes/sims/linre3a/
    """

    n = A.shape[0]

    root = abs(vstack([diagonal(A), diagonal(B)]).T)

    root[:, 0] = root[:, 0] - (root[:, 0] < 1.e-13) * (root[:, 0] + root[:, 1])
    root[:, 1] = root[:, 1] / root[:, 0]

    for i in reversed(range(n)):
        m = None
        for j in reversed(range(0, i+1)):
            if (root[j, 1] > stake) or (root[j, 1] < -0.1):
                m = j
                break

        if m is None:
            return A, B, Q, Z, v

        for k in range(m, i):
            A, B, Q, Z = qzswitch(k, A, B, Q, Z)
            temp = root[k, 1]
            root[k, 1] = root[k+1, 1]
            root[k + 1, 1] = temp

            if not (v is None):
                temp = v[:, k]
                v[:, k] = v[:, k+1]
                v[:, k + 1] = temp

    return A, B, Q, Z, v


def qzswitch(i, A, B, Q, Z):
    """
    This code is a translation from matlab to python of Christopher Sim's 'qzswitch'.
    https://dge.repec.org/codes/sims/linre3a/
    """

    eps = 2.2204e-16
    realsmall = sqrt(eps)*10

    a, b, c = A[i, i], A[i, i + 1], A[i + 1, i + 1]
    d, e, f = B[i, i], B[i, i + 1], B[i + 1, i + 1]

    if (abs(c) < realsmall) and (abs(f) < realsmall):
        if abs(a) < realsmall:
            # l.r. coincident zeros with u.l. of A=0. Do Nothing
            return A, B, Q, Z
        else:
            # l.r. coincident zeros. put zeros in u.l. of a.
            wz = array([[b], [-a]])
            wz = wz / ((matrix(wz).H @ matrix(wz))[0, 0] ** 0.5)
            wz = array([[wz[0][0],  wz[1][0].conj()], [wz[1][0], -wz[0][0].conj()]])
            xy = eye(2)
    elif (abs(a) < realsmall) and (abs(d) < realsmall):
        if abs(c) < realsmall:
            # u.l. coincident zeros with u.l. of A=0. Do Nothing
            return A, B, Q, Z
        else:
            # u.l. coincident zeros. put zeros in u.l. of A
            wz = eye(2)
            xy = array([c, -b])
            xy = xy / (matrix(xy) @ matrix(xy).H)[0, 0] ** 0.5
            xy = array([[xy[1].conj(),  -xy[0].conj()], [xy[0], xy[1]]])
    else:
        # Usual Case
        wz = array([c*e - f*b, (c*d - f*a).conj()])
        xy = array([(b*d - e*a).conj(), (c*d - f*a).conj()])
        n = (matrix(wz) @ matrix(wz).H)[0, 0] ** 0.5
        m = (matrix(xy) @ matrix(xy).H)[0, 0] ** 0.5

        if m < eps*100:
            # all elements of A and B are proportional
            return A, B, Q, Z

        wz = wz / n
        xy = xy / m
        wz = array([[wz[0], wz[1]], [-wz[1].conj(), wz[0].conj()]])
        xy = array([[xy[0], xy[1]], [-xy[1].conj(), xy[0].conj()]])

    A[i:i + 2, :] = xy @ A[i:i + 2, :]
    B[i:i + 2, :] = xy @ B[i:i + 2, :]
    A[:, i:i + 2] = A[:, i:i + 2] @ wz
    B[:, i:i + 2] = B[:, i:i + 2] @ wz
    Z[:, i:i + 2] = Z[:, i:i + 2] @ wz
    Q[i:i + 2, :] = xy @ Q[i:i + 2, :]

    return A, B, Q, Z
