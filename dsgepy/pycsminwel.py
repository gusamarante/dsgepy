"""
All the functions in this file were created by Christopher Sims in 1996.
They are translations to python from the author's original MATLAB files.

The originals can be found in:
http://sims.princeton.edu/yftp/optimize/

I kept the author's original variable names so it is easier to compare
this code with his. Also, this code is not very pythonic, it is very
matlaby. I need a better understanding of the algorithm before making
it more pythonic.
"""

import numpy as np
from warnings import warn


def csminwel(fcn, x0, h0=None, grad=None, crit=1e-14, nit=100, verbose=False):
    """
    This is a locol minimization algorithm. Uses a quasi-Newton method with BFGS
    update of the estimated inverse hessian. It is robust against certain
    pathologies common on likelihood functions. It attempts to be robust against
    "cliffs", i.e. hyperplane discontinuities, though it is not really clear
    whether what it does in such cases succeeds reliably.

    The author of this algorithm is Christopher Sims.

    :param fcn: The function to be minimized. Must have only one input.
    :param x0: Initial guess for the optimization.
    :param h0: Initial guess for the inverse hessian matrix. Must be a positive
               definite matrix.
    :param grad: A function that calculates the gradient of 'fcn'. The function
                 must output a single array with the same dimension as x.
                 If None, the optimization calculates a numerical gradient.
    :param crit: Convergence criterion. Iteration will cease when it proves
                 impossible to improve 'fcn' value by more than 'crit'.
    :param nit: Maximum number of iterations.
    :param verbose: If True, prints the steps of the algorithm.
    :return: fh: minimal value of the function
             xh: optimal input
             gh: gradient at the optimal value
             h: inverse hessian matrix at the optimal value
             itct: number of iterations
             fcount: number of times 'fcn' was evaluated
             retcodeh: return code.
    """

    assert isinstance(x0, np.ndarray) and x0.ndim == 1, "'x0' must be a numpy.ndarray with only 1 dimension"
    assert callable(fcn), "'fcn' must be a callable function"

    nx = x0.shape[0]

    if grad is not None:
        assert callable(grad), "'grad' must be a callable function"
        test_grad = grad(x0)
        assert test_grad.shape[0] == nx, "output of 'grad' does not match the size of 'x'"
        assert test_grad.ndim == 1, "output of 'grad' must have ndim equal to 1"

    itct = 0
    fcount = 0
    numGrad = True if grad is None else False

    if h0 is None:  # This is not a desired behaviour. Passing h0 is recommended.
        h0 = 0.5 * np.eye(nx)
    else:
        assert np.all(np.linalg.eigvals(h0) > 0), "'h0' must be positive definite"

    f0 = fcn(x0)

    if numGrad:
        g, badg = numgrad(fcn, x0)
    else:
        g = grad(x0)
        badg = False

    # To avoid declaration before assignment
    badg1 = None
    badg2 = None
    badg3 = None
    x2 = None
    x3 = None
    xh = None
    fh = None
    g1 = None
    g2 = None
    g3 = None
    gh = None
    retcodeh = None

    # Start the optimization
    x = x0
    f = f0
    h = h0
    done = False

    while not done:
        itct += 1

        if verbose:
            print(f'f at the beginning of iteration {itct} is', f)

        f1, x1, fc, retcode1 = csminit(fcn, x, f, g, badg, h)

        fcount += fc

        if retcode1 != 1:
            if retcode1 == 2 or retcode1 == 4:
                wall1 = True
                badg1 = True
            else:
                if numGrad:
                    g1, badg1 = numgrad(fcn, x1)
                else:
                    g1 = grad(x1)
                    badg1 = False

                wall1 = badg1

            if wall1 and h.shape[0] > 1:
                # Bad gradient or back and forth on step length. Possibly at cliff edge.
                # Try perturbing search direction if problem not 1D
                hCliff = h + np.diag(np.diag(h) * np.random.rand(nx))
                f2, x2, fc, retcode2 = csminit(fcn, x, f, g, badg, hCliff)
                fcount += fc

                if f2 < f:
                    if retcode2 == 2 or retcode2 == 4:
                        wall2 = True
                        badg2 = True
                    else:
                        if numGrad:
                            g2, badg2 = numgrad(fcn, x2)
                        else:
                            g2 = grad(x2)
                            badg2 = False

                        wall2 = badg2

                    if wall2:
                        # cliff again. Try traversing.
                        if np.linalg.norm(x2 - x1) < 1e-13:
                            f3 = f
                            x3 = x
                            badg3 = True
                            retcode3 = 101
                        else:
                            gcliff = ((f2 - f1) / ((np.linalg.norm(x2 - x1)) ** 2)) * (x2 - x1)
                            # "if (size(x0, 2) > 1), gcliff=gcliff', end"   <-- this is matlab code, only
                            #  needed if x is a matrix. Not our case.
                            f3, x3, fc, retcode3 = csminit(fcn, x, f, gcliff, False, np.eye(nx))
                            fcount += fc

                            if retcode3 == 2 or retcode3 == 4:
                                wall3 = True
                                badg3 = True
                            else:
                                if numGrad:
                                    g3, badg3 = numgrad(fcn, x3)
                                else:
                                    g3 = grad(x3)
                                    badg3 = False

                                wall3 = badg3

                    else:
                        f3 = f
                        x3 = x
                        badg3 = True
                        retcode3 = 101

                else:
                    f3 = f
                    x3 = x
                    badg3 = True
                    retcode3 = 101

            else:
                # normal iteration, no walls, or else 1D, or else we're finished here
                f2 = f
                f3 = f
                badg2 = True
                badg3 = True
                retcode2 = 101
                retcode3 = 101

        else:
            f2 = f
            f3 = f
            f1 = f
            retcode2 = retcode1
            retcode3 = retcode1

        if (f3 < f - crit) and (badg3 is False):
            ih = 3
            fh = f3
            xh = x3
            gh = g3
            badgh = badg3
            retcodeh = retcode3

        elif (f2 < f - crit) and (badg2 is False):
            ih = 2
            fh = f2
            xh = x2
            gh = g2
            badgh = badg2
            retcodeh = retcode2

        elif (f1 < f - crit) and (badg1 is False):
            ih = 1
            fh = f1
            xh = x1
            gh = g1
            badgh = badg1
            retcodeh = retcode1
        else:
            fh = min(f1, f2, f3)
            ih = np.argmin([f1, f2, f3])

            if ih == 0:
                xh = x1
            elif ih == 1:
                xh = x2
            elif ih == 2:
                xh = x3

            retcodei = [retcode1, retcode2, retcode3]
            retcodeh = retcodei[ih]

            if 'gh' in locals():
                nogh = gh.size == 0
            else:
                nogh = True

            if nogh:
                if numGrad:
                    gh, badgh = numgrad(fcn, xh)
                else:
                    gh = grad(xh)
                    badgh = False

            badgh = True

        stuck = (np.abs(fh - f) < crit)

        if (not badg) and (not badgh) and (not stuck):
            h = bfgsi(h, gh - g, xh - x, verbose=verbose)

        if verbose:
            print(f'Improvement on iteration {itct} was {f - fh}')

        if itct > nit:
            if verbose:
                print('Maximum number of iterations reached')
            done = True

        elif stuck:
            if verbose:
                print("Convergence achieved. Improvement lower than 'crit'.")
            done = True

        if verbose:
            if retcodeh == 1:
                print('Zero gradient')
            elif retcodeh == 6:
                print('Smallest step still improving too slow, reversed gradient')
            elif retcodeh == 5:
                print('Largest step still improving too fast')
            elif retcodeh == 4 or retcodeh == 2:
                print('Back and forth on step length never finished')
            elif retcodeh == 3:
                print('Smallest step still improving too slow')
            elif retcodeh == 7:
                warn('Possible inaccuracy in the Hessian matrix')

            print('\n')

        f = fh
        x = xh
        g = gh
        badg = badgh

    return fh, xh, gh, h, itct, fcount, retcodeh


def csminit(fcn, x0, f0, g0, badg, h0):

    # Do not ask me where these come from
    angle = 0.005
    theta = 0.3
    fchange = 1000
    minlamb = 1e-9
    mindfac = 0.01

    retcode = None
    fcount = 0
    lambda_ = 1
    xhat = x0
    fhat = f0
    g = g0
    gnorm = np.linalg.norm(g)

    if gnorm < 1e-12 and not badg:
        # gradient convergence
        retcode = 1
    else:
        # with badg true, we don't try to match rate of improvement to
        # directional derivative.  We're satisfied just to get *some*
        # improvement in f.
        dx = - h0 @ g
        dxnorm = np.linalg.norm(dx)

        if dxnorm > 1e12:  # near singular H problem
            dx = dx @ fchange / dxnorm

        dfhat = dx.T @ g0

        if not badg:
            # test for alignment of dx with gradient and fix if necessary
            a = - dfhat / (gnorm * dxnorm)
            if a < angle:
                dx = dx - (angle * dxnorm / gnorm + dfhat/(gnorm ** 2)) * g
                dx = dx * dxnorm / np.linalg.norm(dx)
                dfhat = dx.T @ g

        done = False
        factor = 3
        shrink = True
        lambdaMax = np.inf
        lambdaPeak = 0
        fPeak = f0

        while not done:
            if x0.shape[0] > 1:
                dxtest = x0 + dx.T * lambda_
            else:
                dxtest = x0 + dx * lambda_

            f = fcn(dxtest)

            if f < fhat:
                fhat = f
                xhat = dxtest

            fcount += 1

            shrinkSignal = ((not badg) and (f0 - f < max(- theta * dfhat * lambda_, 0))) or (badg and (f0 - f < 0))
            growSignal = (not badg) and ((lambda_ > 0) and (f0 - f > - (1 - theta) * dfhat * lambda_))

            if shrinkSignal and ((lambda_ > lambdaPeak) or (lambda_ < 0)):
                if lambda_ > 0 and ((not shrink) or (lambda_ / factor <= lambdaPeak)):
                    shrink = True
                    factor = factor ** 0.6

                    while lambda_/factor <= lambdaPeak:
                        factor = factor ** 0.6

                    if np.abs(factor - 1) < mindfac:

                        if np.abs(lambda_) < 4:
                            retcode = 2
                        else:
                            retcode = 7

                        done = True

                if (lambda_ < lambdaMax) and (lambda_ > lambdaPeak):
                    lambdaMax = lambda_

                lambda_ = lambda_ / factor

                if np.abs(lambda_) < minlamb:
                    if (lambda_ > 0) and (f0 <= fhat):
                        # try going against gradient, which may be inaccurate
                        lambda_ = - lambda_ * factor ** 6
                    else:
                        if lambda_ < 0:
                            retcode = 6
                        else:
                            retcode = 3

                        done = True

            elif (growSignal and lambda_ > 0) or (shrinkSignal and ((lambda_ <= lambdaPeak) and (lambda_ > 0))):
                if shrink:
                    shrink = False
                    factor = factor ** 0.6

                    if np.abs(factor - 1) < mindfac:
                        if np.abs(lambda_) < 4:
                            retcode = 4
                        else:
                            retcode = 7

                        done = True

                if (f < fPeak) and (lambda_ > 0):
                    fPeak = f
                    lambdaPeak = lambda_
                    if lambdaMax <= lambdaPeak:
                        lambdaMax = lambdaPeak * (factor ** 2)

                lambda_ = lambda_ * factor

                if np.abs(lambda_) > 1e20:
                    retcode = 5
                    done = True

            else:
                done = True
                if factor < 1.2:
                    retcode = 7
                else:
                    retcode = 0

    return fhat, xhat, fcount, retcode


def numgrad(fcn, x):
    """
    Computes the numerical gradient of a function at point x.
    :param fcn: python function that must depend on a single argument
    :param x: 1-D numpy.array
    :return: 1-D numpy.array
    """

    delta = 1e-6
    bad_gradient = False
    n = x.shape[0]
    tvec = delta * np.eye(n)
    g = np.zeros(n)

    f0 = fcn(x)

    for i in range(n):
        tvecv = tvec[i, :]

        g0 = (fcn(x + tvecv) - f0) / delta

        if np.abs(g0) < 1e15:  # good gradient
            g[i] = g0
        else:  # bad gradient
            g[i] = 0
            bad_gradient = True

    return g, bad_gradient


def bfgsi(h0, dg, dx, verbose=False):
    """
    BFGS update for the inverse hessian, ;
    :param h0: previous value of the inverse hessian
    :param dg: dg is previous change in gradient
    :param dx: dx is previous change in x
    :param verbose: If True, prints updates of the algorithm
    :return: updated inverse hessian matrix
    """

    dx = dx.reshape(-1, 1)
    dg = dg.reshape(-1, 1)

    hdg = h0 @ dg
    dgdx = (dg.T @ dx)[0, 0]

    if np.abs(dgdx) > 1e-12:
        h = h0 + (1 + (dg.T @ hdg) / dgdx) * (dx @ dx.T) / dgdx - (dx @ hdg.T + hdg @ dx.T) / dgdx

    else:
        if verbose:
            print('bfgs update failed')
            # disp(['|dg| = ' num2str(sqrt(dg'*dg)) ' | dx | = ' num2str(sqrt(dx' * dx))]);
            # disp(['dg''*dx = ' num2str(dgdx)])
            # disp(['|H*dg| = ' num2str(Hdg'*Hdg)])

        h = h0

    return h
