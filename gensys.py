from scipy.linalg import qz
from numpy.linalg import svd
from numpy import diagonal, vstack, array, eye, where


def gensys(g0, g1, c, psi, pi, div=None, realsmall=0.000001):

    # TODO Assert variable types

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

    q1 = q[:n - nunstab + 1, :]
    q2 = q[n - nunstab + 1:, :]
    z1 = z[:, :n - nunstab + 1].T
    z2 = z[:, n - nunstab + 1:]
    a2 = a[n - nunstab + 1:, n - nunstab + 1:]
    b2 = a[n - nunstab + 1:, n - nunstab + 1:]

    etawt = q2 @ pi
    neta = pi.shape[1]

    if nunstab == 0:
        etawt = None
        ueta = None
        deta = None
        veta = None
        bigev = 0
    else:
        ueta, deta, veta = svd(etawt)
        md = min(deta.shape)

        if len(deta.shape) == 1:
            deta = deta.reshape((max(1, deta.shape[0]), max(1, deta.shape[0])))

        bigev = where(diagonal(deta[:md, :md]) > realsmall)
        ueta = ueta[:, bigev]
        veta = veta[:, bigev]
        deta = deta[bigev, bigev]

    if len(bigev) >= nunstab:
        eu[0] = 1

    # PAREI NA LINHA 121 NO GENSYS

    return G1, C, impact, fmat, fwt, ywt, gev, eu, loose


def qzdiv(stake, A, B, Q, Z, v=None):

    n = A.shape[0]

    root = vstack([diagonal(A), diagonal(B)]).T

    root[:, 0] = root[:, 0] - (root[:, 0] < 1.e-13) * (root[:, 0] + root[:, 1])
    root[:, 1] = root[:, 1] / root[:, 0]

    for i in reversed(range(n)):
        m = None
        for j in reversed(range(n)):
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

            if not(v is None):
                temp = v[:, k]
                v[:, k] = v[:, k+1]
                v[:, k + 1] = temp

    return A, B, Q, Z, v


def qzswitch(i, A, B, Q, Z):

    eps = 1.0e-15

    a, b, c = A[i, i], A[i, i + 1], A[i + 1, i + 1]
    d, e, f = B[i, i], B[i, i + 1], B[i + 1, i + 1]

    if (abs(c) < eps) and (abs(f) < eps):
        if abs(a) < eps:
            # l.r. coincident zeros with u.l. of A=0. Do Nothing
            return A, B, Q, Z
        else:
            # l.r. coincident zeros. put zeros in u.l. of a.
            wz = array([[b], [-a]])
            wz = wz / ((wz.T @ wz) ** 0.5)
            wz = array([[wz[0][0],  wz[1][0]], [wz[1][0], -wz[0][0]]])
            xy = eye(2)
    elif (abs(a) < eps) and (abs(d) < eps):
        if abs(c) < eps:
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
        xy = array([[xy[0], xy[0]], [-xy[1], xy[0]]])

    A[i:i + 2, :] = xy @ A[i:i + 2, :]
    B[i:i + 2, :] = xy @ B[i:i + 2, :]
    A[:, i:i + 2] = A[:, i:i + 2] @ wz
    B[:, i:i + 2] = B[:, i:i + 2] @ wz
    Z[:, i:i + 2] = Z[:, i:i + 2] @ wz
    Q[i:i + 2, :] = xy @ Q[i:i + 2, :]

    return A, B, Q, Z
