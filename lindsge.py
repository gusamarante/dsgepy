"""
This code is a translation from matlab to python of Christopher Sim's 'gensys'.
https://dge.repec.org/codes/sims/linre3a/
"""

from scipy.linalg import qz
from numpy.linalg import svd, inv
from numpy import diagonal, vstack, array, eye, where, diag, sqrt, hstack, \
    zeros, arange


class DSGE(object):

    def __init__(self, gamma0, gamma1, c, psi, pi):
        self.g0 = gamma0
        self.g1 = gamma1
        self.c_in = c
        self.psi = psi
        self.pi = pi


def gensys(g0, g1, c, psi, pi, div=None, realsmall=0.000001):

    # TODO Assert variable types
    unique = False
    eu = [0, 0]
    nunstab = 0
    zxz = False

    n = g1.shape[0]

    if div is None:
        div = 1.01

    a, b, q, z = qz(g0, g1)

    for i in range(n):
        if div is None:
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
        print('Coincident zeros. Indeterminancy and/or nonexistence')
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
        print(f'Indeterminacy. {nloose} endogenous variables.')

    tmat = hstack((eye(n - nunstab), -(ueta @ (inv(deta) @ veta.T) @ veta1.T @ deta1 @ ueta1.T).T))
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

    loose = G0I @ vstack((etawt1 @ (eye(neta) - veta @ veta.T), zeros((nunstab, neta))))
    G1 = (z @ G1 @ z.T).real
    C = (z @ C).real
    impact = (z @ impact).real
    loose = (z @ loose).real
    ywt = z @ ywt

    return G1, C, impact, fmat, fwt, ywt, gev, eu, loose


def qzdiv(stake, A, B, Q, Z, v=None):

    n = A.shape[0]

    root = vstack([diagonal(A), diagonal(B)]).T

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
            wz = wz / ((wz.T @ wz) ** 0.5)
            wz = array([[wz[0][0],  wz[1][0]], [wz[1][0], -wz[0][0]]])
            xy = eye(2)
    elif (abs(a) < realsmall) and (abs(d) < realsmall):
        if abs(c) < realsmall:
            # u.l. coincident zeros with u.l. of A=0. Do Nothing
            return A, B, Q, Z
        else:
            # u.l. coincident zeros. put zeros in u.l. of A
            wz = eye(2)
            xy = array([b, -a])
            xy = xy / ((xy @ xy)**0.5)
            xy = array([[xy[1][0],  -xy[0][0]], [xy[0][0], xy[1][0]]])
    else:
        # Usual Case
        wz = array([c*e - f*b, c*d - f*a])
        xy = array([b*d - e*a, c*d - f*a])
        n = ((wz @ wz) ** 0.5)
        m = ((xy @ xy) ** 0.5)

        if m < eps*100:
            # all elements of A and B are proportional
            return A, B, Q, Z

        wz = wz / n
        xy = xy / m
        wz = array([[wz[0], wz[1]], [-wz[1], wz[0]]])
        xy = array([[xy[0], xy[1]], [-xy[1], xy[0]]])

    A[i:i + 2, :] = xy @ A[i:i + 2, :]
    B[i:i + 2, :] = xy @ B[i:i + 2, :]
    A[:, i:i + 2] = A[:, i:i + 2] @ wz
    B[:, i:i + 2] = B[:, i:i + 2] @ wz
    Z[:, i:i + 2] = Z[:, i:i + 2] @ wz
    Q[i:i + 2, :] = xy @ Q[i:i + 2, :]

    return A, B, Q, Z
