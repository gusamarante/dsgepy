import warnings
import pandas as pd
from tqdm import tqdm
from sympy import simplify
from scipy.linalg import qz
from pykalman import KalmanFilter
from numpy.linalg import svd, inv
from tables import PerformanceWarning
from scipy.optimize import minimize, basinhopping
from numpy.random import multivariate_normal, rand, seed
from numpy import diagonal, vstack, array, eye, where, diag, sqrt, hstack, zeros, \
    arange, exp, log, inf, nan, isnan, isinf, set_printoptions, matrix
set_printoptions(precision=4, suppress=True, linewidth=150)


warnings.filterwarnings('ignore')  # category=PerformanceWarning


class DSGE(object):

    # TODO Estimation
    # TODO impulse response function
    # TODO Forecast error variance
    # TODO Series Forecast
    # TODO Historical Decomposition
    # TODO Prior Posterior Table in Latex
    # TODO Model Identification (Iskrev's paper)

    prior_info = None

    def __init__(self, endog, endogl, exog, expec, params, equations, calib_dict=None,
                 obs_matrix=None, obs_offset=None, prior_dict=None, obs_data=None, verbose=True):

        # TODO assert prior info (inv gamma a!=2 and a!=1)
        # TODO mudar default verbose para False

        self.endog = endog
        self.endogl = endogl
        self.exog = exog
        self.expec = expec
        self.params = params
        self.equations = equations
        self.obs_matrix = obs_matrix
        self.obs_offset = obs_offset
        self.prior_dict = prior_dict
        self.data = obs_data
        self.n_state = len(endog)
        self.n_param = len(params)
        self.n_obs = obs_matrix.shape[0]
        self._has_solution = False
        self.verbose = verbose
        self._get_jacobians()

        # If subs_dict is passed, generate the solution
        if not (calib_dict is None):
            self.Gamma0, self.Gamma1, self.Psi, self.Pi, self.C_in, self.obs_matrix, self.obs_offset = \
                self._eval_matrix(calib_dict)

            self.G1, self.C_out, self.impact, self.fmat, self.fwt, self.ywt, self.gev, self.eu, self.loose = \
                gensys(self.Gamma0, self.Gamma1, self.C_in, self.Psi, self.Pi)

            self._has_solution = True
        else:
            self.prior_info = self._get_prior_info()

    def simulate(self, n_obs=100, random_seed=None):

        assert self._has_solution, "No solution was generated yet"

        if not (random_seed is None):
            seed(random_seed)

        # TODO add observation covariance to allow for measurment errors
        kf = KalmanFilter(self.G1, self.obs_matrix, self.impact @ self.impact.T, None,
                          self.C_out.reshape(self.n_state), self.obs_offset.reshape(self.n_obs))
        simul_data = kf.sample(n_obs)

        state_names = [str(s) for s in list(self.endog)]
        obs_names = [f'obs {i+1}' for i in range(self.obs_matrix.shape[0])]

        df_obs = pd.DataFrame(data=simul_data[1], columns=obs_names)
        df_states = pd.DataFrame(data=simul_data[0], columns=state_names)

        return df_obs, df_states

    def estimate(self, nsim=1000, ck=0.2, file_path=None):

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

            # Optimization - SciPy minimize
            # res = minimize(obj_func, theta_irr0, options={'disp': False})
            # theta_mode_irr = {k: v for k, v in zip(self.params, res.x)}
            # theta_mode_res = self._irr2res(theta_mode_irr)
            # sigmak = ck * res.hess_inv

            # Optimization - Basinhoping
            # res = basinhopping(obj_func, theta_irr0)
            # theta_mode_irr = {k: v for k, v in zip(self.params, res.x)}
            # theta_mode_res = self._irr2res(theta_mode_irr)
            # sigmak = ck * res.hess_inv

            # Overrides the result of the optimization
            theta_mode_res = self.prior_info['mean']
            sigmak = ck * eye(self.n_param)

            df_chains = pd.DataFrame(columns=[str(p) for p in list(self.params)], index=range(nsim))
            df_chains.loc[0] = list(theta_mode_res.values)
            start = 0

        # Metropolis-Hastings
        muk = zeros(self.n_param)
        accepted = 0

        # TODO optimize with pymc
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

            if ii % 100 == 0:
                store = pd.HDFStore(file_path)
                store['chains'] = df_chains
                store['sigmak'] = pd.DataFrame(data=sigmak)
                store.close()

        store = pd.HDFStore(file_path)
        store['chains'] = df_chains
        store['sigmak'] = pd.DataFrame(data=sigmak)
        store.close()

        return df_chains, accepted / nsim

    def _get_jacobians(self):
        self.Gamma0 = self.equations.jacobian(self.endog)
        self.Gamma1 = -self.equations.jacobian(self.endogl)
        self.Psi = -self.equations.jacobian(self.exog)
        self.Pi = -self.equations.jacobian(self.expec)
        self.C_in = simplify(self.equations
                             - self.Gamma0 @ self.endog
                             + self.Gamma1 @ self.endogl
                             + self.Psi @ self.exog
                             + self.Pi @ self.expec)

    def _calc_posterior(self, theta):
        P = self._calc_prior(theta)
        L = self._log_likelihood(theta)
        f = P + L
        return f*1000

    def _calc_prior(self, theta):
        prior_dict = self.prior_dict
        df_prior = self.prior_info
        df_prior['pdf'] = nan

        for param in prior_dict.keys():
            a = prior_dict[param]['param a']
            b = prior_dict[param]['param b']
            dist = prior_dict[param]['dist'].lower()
            theta_i = theta[param]

            # since we are goig to take logs, the density function only needs the terms that depend on
            # theta_i, this will help speed up the code a little and will not affect optimization output.
            if dist == 'beta':
                pdf_i = (theta_i**(a - 1)) * ((1 - theta_i)**(b - 1))

            elif dist == 'gamma':
                pdf_i = theta_i**(a - 1) * exp(-theta_i/b)

            elif dist == 'invgamma':
                pdf_i = (theta_i**(- a - 1)) * exp(-b/theta_i)

            elif dist == 'uniform':
                pdf_i = 1/(b - a)

            else:  # Normal
                pdf_i = exp(((theta_i - a)**2)/(2 * (b**2)))

            df_prior.loc[str(param), 'pdf'] = pdf_i

        df_prior['log pdf'] = log(df_prior['pdf'].astype(float))

        P = df_prior['log pdf'].sum()

        return P

    def _log_likelihood(self, theta):
        Gamma0, Gamma1, Psi, Pi, C_in, obs_matrix, obs_offset = self._eval_matrix(theta)

        for mat in [Gamma0, Gamma1, Psi, Pi, C_in]:
            if isnan(mat).any() or isinf(mat).any():
                return -inf

        G1, C_out, impact, fmat, fwt, ywt, gev, eu, loose = gensys(Gamma0, Gamma1, C_in, Psi, Pi)

        if eu[0] == 1 and eu[1] == 1:
            # TODO add observation covariance to allow for measurment errors
            kf = KalmanFilter(G1, obs_matrix, impact @ impact.T, None, C_out.reshape(self.n_state),
                              obs_offset.reshape(self.n_obs))
            L = kf.loglikelihood(self.data)
        else:
            L = - inf

        return L

    def _eval_matrix(self, theta):

        # state matrices
        Gamma0 = array(self.Gamma0.subs(theta)).astype(float)
        Gamma1 = array(self.Gamma1.subs(theta)).astype(float)
        Psi = array(self.Psi.subs(theta)).astype(float)
        Pi = array(self.Pi.subs(theta)).astype(float)
        C_in = array(self.C_in.subs(theta)).astype(float)

        # observation matrices
        obs_matrix = array(self.obs_matrix.subs(theta)).astype(float)
        obs_offset = array(self.obs_offset.subs(theta)).astype(float)

        return Gamma0, Gamma1, Psi, Pi, C_in, obs_matrix, obs_offset

    def _get_prior_info(self):
        # TODO add distribution column
        prior_info = self.prior_dict

        param_names = [str(s) for s in list(self.params)]

        df_prior = pd.DataFrame(columns=['mean', 'std'],
                                index=param_names)

        for param in prior_info.keys():
            a = prior_info[param]['param a']
            b = prior_info[param]['param b']
            dist = prior_info[param]['dist'].lower()

            if dist == 'beta':
                mean_i = a / (a + b)
                std_i = ((a * b) / (((a + b) ** 2) * (a + b + 1))) ** 0.5

            elif dist == 'gamma':
                mean_i = a * b
                std_i = (a * (b ** 2)) ** 0.5

            elif dist == 'invgamma':
                mean_i = b / (a - 1)
                std_i = ((b ** 2) / (((a - 1) ** 2) * (a - 2))) ** 0.5

            elif dist == 'uniform':
                mean_i = (a + b) / 2
                std_i = (((b - a) ** 2) / 12) ** 0.5

            else:  # Normal
                mean_i = a
                std_i = b

            df_prior.loc[str(param)] = [mean_i, std_i]

        return df_prior

    def _res2irr(self, theta_res):
        prior_info = self.prior_dict
        theta_irr = theta_res.copy()

        for param in theta_res.keys():
            a = prior_info[param]['param a']
            b = prior_info[param]['param b']
            dist = prior_info[param]['dist'].lower()
            theta_i = theta_res[param]

            if dist == 'beta':
                theta_irr[param] = log(theta_i / (1 - theta_i))

            elif dist == 'gamma':
                theta_irr[param] = log(theta_i)

            elif dist == 'invgamma':
                theta_irr[param] = log(theta_i)

            elif dist == 'uniform':
                theta_irr[param] = log((theta_i - a) / (b - theta_i))

            else:  # Normal
                theta_irr[param] = theta_i

        return theta_irr

    def _irr2res(self, theta_irr):
        prior_info = self.prior_dict
        theta_res = theta_irr.copy()

        for param in theta_irr.keys():
            a = prior_info[param]['param a']
            b = prior_info[param]['param b']
            dist = prior_info[param]['dist'].lower()
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


def gensys(g0, g1, c, psi, pi, div=None, realsmall=0.000001):
    """
    This code is a translation from matlab to python of Christopher Sim's 'gensys'.
    https://dge.repec.org/codes/sims/linre3a/
    """

    # TODO Assert variable types
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
    z1 = z[:, :n - nunstab].T
    z2 = z[:, n - nunstab:]
    a2 = a[n - nunstab:, n - nunstab:]
    b2 = a[n - nunstab:, n - nunstab:]

    etawt = q2 @ pi
    neta = pi.shape[1]

    # Case for no stable roots
    if nunstab == 0:
        etawt = None
        ueta = None
        deta = None
        veta = None
        bigev = 0
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

    if len(bigev) >= nunstab:
        eu[0] = 1

    # Case for all stable roots
    if nunstab == n:
        etawt1 = None
        ueta1 = None
        deta1 = None
        veta1 = None
        bigev = 0
    else:
        etawt1 = q1 @ pi
        ndeta1 = min(n - nunstab, neta)
        ueta1, deta1, veta1 = svd(etawt1)
        deta1 = diag(deta1)
        veta1 = array(matrix(veta1).H)  # TODO check if transpose, instead of hermitian
        md = min(deta1.shape)
        bigev = where(diagonal(deta1[:md, :md]) > realsmall)[0]
        ueta1 = ueta1[:, bigev]
        veta1 = veta1[:, bigev]
        deta1 = deta1[bigev, bigev]
        if deta1.ndim == 1:
            deta1 = diag(deta1)

    if veta1 is None:
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


def evaluate_chains():
    # TODO Class or function?
    # TODO Prior Density VS Posterior histogram
    # TODO Table with mean and std from priors and poteriors
    # TODO Output a model calibrated with posteriors
    pass
