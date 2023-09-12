
import numpy as np
import scipy as sp
from functools import partial


def interp(z0, z1, z2, I0, I1, I2, flag):
    x0 = 0
    x1 = np.linalg.norm(z1 - z0)
    x2 = np.linalg.norm(z2 - z0)
    wan = np.array([[x0**2, x0, 1], [x1**2, x1, 1], [x2**2, x2, 1]])
    b = [I0, I1, I2]
    alpha = np.linalg.solve(wan, b)
    V = [-alpha[1] / (2 * alpha[0]), alpha[2] - alpha[1]**2 / (4 * alpha[0])]
    zM = z0 + V[0] * (z2 - z0) / np.linalg.norm(z2 - z0)
    if V[0] < x0 or V[1] > x2:
        zvect = np.array([z0, z2])
        Ivect = [I0, I2]
    else:
        zvect = np.array([z0, zM, z2])
        Ivect = [I0, V[1], I2]

    if flag == -1:
        IM = min(Ivect)
        ind = Ivect.index(min(Ivect))
        zM = zvect[ind]
    else:
        IM = max(Ivect)
        ind = Ivect.index(max(Ivect))
        zM = zvect[ind]

    return zM, IM


def discretize(a, b, hx, hy):

    x = np.arange(0, a + hx, hx)
    y = np.arange(0, b + hy, hy)

    nx = len(x)
    ny = len(y)
    nTx = nx - 1
    nTy = ny - 1
    [X, Y] = np.meshgrid(x, y)
    P = np.array([np.reshape(X, nx * nx, order='F'), np.reshape(Y, ny * ny, order='F')])
    k = 0
    T = np.ones((3, 2 * (nTx * nTy)), dtype=int)
    for i in range(1, nTx + 1):
        for j in range(1, nTy + 1):
            C1 = j + (i - 1) * ny
            C2 = (j + 1) + (i - 1) * ny
            C3 = (j) + (i) * ny
            C4 = (j + 1) + (i) * ny
            T[:, k * 2] = [C3, C1, C4]
            T[:, k * 2 + 1] = [C2, C4, C1]
            k = k + 1

    s1 = np.arange(1, ny + 1, 1)
    s2 = np.arange(ny, (nx * ny) + 1, ny)
    s3 = np.arange(((ny * nx) - ny + 1), (nx * ny) + 1, 1)
    s4 = np.arange(1, ((nx * ny) - ny + 2), ny)
    return P, T, s1, s2, s3, s4


def alokA(P, T, S):

    n = len(P[0])
    nt = len(T[0])

    ID = S
    natNumN = np.arange(1, n + 1, 1)
    A = np.zeros([n, n])
    Q = np.array([[-1, 1, 0], [-1, 0, 1]])

    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        ST = abs(np.linalg.det(G)) / 2
        Bv = np.matmul(np.transpose(np.linalg.inv(G)), Q)

        Av = np.matmul(np.transpose(Bv), Bv) * ST
        A[Ti, Ti[0]] = A[Ti, Ti[0]] + Av[0, :]
        A[Ti, Ti[1]] = A[Ti, Ti[1]] + Av[1, :]
        A[Ti, Ti[2]] = A[Ti, Ti[2]] + Av[2, :]
    NonD = np.setdiff1d(natNumN, ID)
    Ad = np.eye(n, n)
    Ad[np.ix_(NonD - 1, NonD - 1)] = A[np.ix_(NonD - 1, NonD - 1)]
    return Ad


# %%%%%%%%%%%%% functionals and norms%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def qnorm(q, P, T, S, c):

    nt = len(T[0])
    Jp = 0
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        cvect = [-c1 + c2, -c1 + c3]
        JpT = ST * (np.linalg.norm(np.matmul(invG, cvect)))**q
        Jp = Jp + JpT

    qnorm = (Jp)**(1 / q)
    return qnorm


def Lpnorma(q, P, T, S, c):

    nt = len(T[0])
    Jp = 0
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        JpT = 1 / 30 * (c1**4 + c1**3 * c2 + c1**3 * c3 + c1**2 * c2**2 + c1**2 * c2 * c3 + c1**2 * c3**2 + c1 * c2 **
                        3 + c1 * c2**2 * c3 + c1 * c2 * c3**2 + c1 * c3**3 + c2**4 + c2**3 * c3 + c2**2 * c3**2 + c2 * c3**3 + c3**4)
        Jp = Jp + abs(np.linalg.det(G)) * JpT

    return Jp


def Laplace(p, P, T, S, c):
    nt = len(T[0])
    Jp = 0
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        cvect = [-c1 + c2, -c1 + c3]
        JpT = ST * (np.linalg.norm(np.matmul(invG, cvect)))**p  # IlIl1l1l1l1l
        Jp = Jp + (1 / p) * JpT
    return Jp


def RHS(p, P, T, S, c):

    nt = len(T[0])
    integ = 0
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        intT = 1 / 30 * c1**4 + 1 / 30 * c1**3 * c2 + 1 / 30 * c1**3 * c3 + 1 / 30 * c1**2 * c2**2 + 1 / 30 * c1**2 * c2 * c3 + 1 / 30 * c1**2 * c3**2 +\
            1 / 30 * c1 * c2**3 + 1 / 30 * c1 * c2**2 * c3 + 1 / 30 * c1 * c2 * c3**2 + 1 / 30 * c1 * c3**3 + 1 / 30 * c2**4 + 1 / 30 * c2**3 * c3 +\
            1 / 30 * c2**2 * c3**2 + 1 / 30 * c2 * c3**3 + 1 / 30 * c3**4
        intT = abs(np.linalg.det(G)) * intT
        integ = integ + intT

    f = integ
    J = 1 / 4 * f
    return J


def alokJ(p, P, T, S, c):
    nt = len(T[0])
    integ = 0
    Jp = 0
    c[S - 1] = 0
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]).T
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        intT = 1 / 30 * c1**4 + 1 / 30 * c1**3 * c2 + 1 / 30 * c1**3 * c3 + 1 / 30 * c1**2 * c2**2 + 1 / 30 * c1**2 * c2 * c3 + 1 / 30 * c1**2 * c3**2 + 1 / 30 * c1 * c2**3 + 1 / \
            30 * c1 * c2**2 * c3 + 1 / 30 * c1 * c2 * c3**2 + 1 / 30 * c1 * c3**3 + 1 / 30 * c2**4 + \
            1 / 30 * c2**3 * c3 + 1 / 30 * c2**2 * c3**2 + 1 / 30 * c2 * c3**3 + 1 / 30 * c3**4
        intT = abs(np.linalg.det(G)) * intT
        integ = integ + intT
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(G.T)
        cvect = [-c1 + c2, -c1 + c3]
        JpT = (1 / p) * ST * (np.linalg.norm(np.matmul(invG, cvect)))**p
        Jp = Jp + JpT
    J = Jp - 1 / 4 * integ
    if (p > 4):
        J = -J
    return J


def alokJforf(p, P, T, S, f, c):
    nt = len(T[0])
    Jp = 0
    c[S - 1] = 0

    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        f1 = f[Ti[0]]
        f2 = f[Ti[1], 0]
        f3 = f[Ti[2]]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        cvect = [-c1 + c2, -c1 + c3]
        JpT = (1 / p) * ST * (np.linalg.norm(np.matmul(invG, cvect)))**p
        Jpf = 2 * ST * ((1 / 24) * f1 * c3 + (1 / 24) * f3 * c1 + (1 / 12) * f3 * c3 + (1 / 24) * f2 * c1 +
                        (1 / 24) * f2 * c3 + (1 / 24) * f1 * c2 + (1 / 24) * f3 * c2 + (1 / 12) * f2 * c2 + (1 / 12) * f1 * c1)
        Jp = Jp + JpT - Jpf
    J = Jp
    if (p > 4):
        J = -J
    return J


def alokJmin(p, P, T, S, c):

    J = qnorm(p, P, T, S, c) / (Lpnorma(p, P, T, S, c)**(1 / 4))
    return J


def alokDJmin_exact_functional(p, P, T, S, u, c):

    nt = len(T[0])
    c[S - 1] = 0
    b1 = 0
    b2 = 0

    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        u1 = u[Ti[0], 0]
        u2 = u[Ti[1], 0]
        u3 = u[Ti[2], 0]

        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        invG_u = np.matmul(invG, [u2 - u1, u3 - u1])

        if (c1 == c2 and c2 == c3):
            b1T = 0
        else:
            b1T = ST * np.linalg.norm(invG_u)**(p - 2) * np.matmul(np.transpose(invG_u),
                                                                   invG_c)
        b2T = 2 * ST / 120 * (4 * c1 * u1**3 + 3 * c1 * u1**2 * u2 + 3 * c1 * u1**2 * u3 + 2 * c1 * u1 * u2**2 + 2 * c1 * u1 * u2 * u3 + 2 * c1 * u1 * u3**2 + c1 * u2**3 + c1 * u2**2 * u3 + c1 * u2 * u3**2 + c1 * u3**3 + c2 * u1**3 + 2 * c2 * u1**2 * u2 + c2 * u1**2 * u3 + 3 * c2 * u1 * u2**2 + 2 * c2 *
                              u1 * u2 * u3 + c2 * u1 * u3**2 + 4 * c2 * u2**3 + 3 * c2 * u2**2 * u3 + 2 * c2 * u2 * u3**2 + c2 * u3**3 + c3 * u1**3 + c3 * u1**2 * u2 + 2 * c3 * u1**2 * u3 + c3 * u1 * u2**2 + 2 * c3 * u1 * u2 * u3 + 3 * c3 * u1 * u3**2 + c3 * u2**3 + 2 * c3 * u2**2 * u3 + 3 * c3 * u2 * u3**2 + 4 * c3 * u3**3)
        b1 = b1 + b1T
        b2 = b2 + b2T

    Cb1 = (qnorm(p, P, T, S, u)**p)**(1 / p - 1) * Lpnorma(4, P, T, S, u)**(-1 / 4)
    Cb2 = qnorm(p, P, T, S, u) * Lpnorma(4, P, T, S, u)**(-5 / 4)
    b = b1 * Cb1 - b2 * Cb2
    J = Laplace(p, P, T, S, c) - b
    if (p > 4):
        J = J
    return J


def alokDJminmax_exact_functional(p, P, T, S, u, c):
    nt = len(T[0])
    c[S - 1] = 0
    Jp = 0
    JpU1 = 0
    JpT = 0
    JpU2 = 0
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        u1 = u[Ti[0], 0]
        u2 = u[Ti[1], 0]
        u3 = u[Ti[2], 0]

        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        invG_u = np.matmul(invG, [u2 - u1, u3 - u1])
        if (c1 == 0 and c2 == 0 and c3 == 0):
            JpT = JpT + 0
        else:
            JpT = JpT + (1 / p) * np.linalg.norm(invG_c)**p * ST

        if (u1 == 0 and u2 == 0 and u3 == 0):
            JpU1 = JpU1 + 0
        else:
            JpU1 = JpU1 + ST * np.linalg.norm(invG_u)**(p - 2) * np.matmul(np.transpose(invG_u), invG_c)

        JpU2T = (2 * u1 * u2 * u3 * (c1 + c2 + c3) + u1**3 * (4 * c1 + c2 + c3) + u2**3 * (c1 + 4 * c2 + c3) + u3**3 * (c1 + c2 + 4 * c3) + u1**2 * ((3 * c1 + 2 * c2 + c3) * u2 +
                 u3 * (3 * c1 + c2 + 2 * c3)) + ((2 * c1 + 3 * c2 + c3) * u1 + u3 * (c1 + 3 * c2 + 2 * c3)) * u2**2 + ((2 * c1 + c2 + 3 * c3) * u1 + u2 * (c1 + 2 * c2 + 3 * c3)) * u3**2)
        JpU2 = JpU2 + JpU2T * 2 * ST / 120

    if (p < 4):
        J = Jp - JpU1 + JpU2 + JpT
    else:
        J = Jp + JpU1 - JpU2 + JpT
    return J

# %%%%%%%%%%%%% gradients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def alokDJ(p, P, T, S, A, c):
    n = len(P[0])
    nt = len(T[0])
    c[S - 1] = 0
    dJ = np.zeros([n, 1])

    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        dJc = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [1, 0]))),
                       np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)
        fCT = np.array([(4 * c1**3 + 3 * c1**2 * c2 + 3 * c1**2 * c3 + 2 * c1 * c2**2 + 2 * c1 * c2 * c3 + 2 * c1 * c3**2 + c2**3 + c2**2 * c3 + c2 * c3**2 + c3**3), (c1**3 + 2 * c1**2 * c2 + c1**2 * c3 + 3 * c1 * c2**2 + 2 * c1 * c2 * c3 + c1 *
                       c3**2 + 4 * c2**3 + 3 * c2**2 * c3 + 2 * c2 * c3**2 + c3**3), (c1**3 + c1**2 * c2 + 2 * c1**2 * c3 + c1 * c2**2 + 2 * c1 * c2 * c3 + 3 * c1 * c3**2 + c2**3 + 2 * c2**2 * c3 + 3 * c2 * c3**2 + 4 * c3**3)]) * 2 * ST / 120
        dJ[Ti, 0] = dJ[Ti, 0] + dJc - fCT

    b = dJ
    b[S - 1] = 0
    dJ = np.linalg.solve(A, b)
    dJ[S - 1] = 0

    if (p > 4):
        dJ = -dJ

    return dJ


def alokDJforf(p, P, T, S, A, f, c):

    n = len(P[0])
    nt = len(T[0])
    c[S - 1] = 0
    dJ = np.zeros([n, 1])
    c[S - 1] = 0
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        dJc = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [1, 0]))),
                       np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)
        dJ[Ti, 0] = dJ[Ti, 0] + dJc
    b1 = dJ
    fC = np.zeros([n, 1])
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        f1 = f[Ti[0], 0]
        f2 = f[Ti[1]]
        f3 = f[Ti[2], 0]
        fCT = np.array([(2 * f1 + f2 + f3), (f1 + 2 * f2 + f3), (f1 + f2 + 2 * f3)]) * 2 * ST * 1 / 24
        fC[Ti, 0] = fC[Ti, 0] + fCT
    b2 = fC
    b = b1 - b2
    b[S - 1] = 0
    dJ = np.linalg.solve(A, b)
    dJ[S - 1] = 0

    if (p > 4):
        dJ = -dJ

    return dJ


def alokDJmin(p, P, T, S, A, c):

    n = len(P[0])
    nt = len(T[0])
    c[S - 1] = 0
    b1 = np.zeros([n, 1])
    b2 = np.zeros([n, 1])
    c[S - 1] = 0
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        if (c1 == c2 and c2 == c3):
            b1T = [0, 0, 0]
        else:
            b1T = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [1, 0]))),
                           np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)

        b1[Ti, 0] = b1[Ti, 0] + b1T

        b2T = np.array([(4 * c1**3 + 3 * c1**2 * c2 + 3 * c1**2 * c3 + 2 * c1 * c2**2 + 2 * c1 * c2 * c3 + 2 * c1 * c3**2 + c2**3 + c2**2 * c3 + c2 * c3**2 + c3**3), (c1**3 + 2 * c1**2 * c2 + c1**2 * c3 + 3 * c1 * c2**2 + 2 * c1 * c2 * c3 + c1 *
                       c3**2 + 4 * c2**3 + 3 * c2**2 * c3 + 2 * c2 * c3**2 + c3**3), (c1**3 + c1**2 * c2 + 2 * c1**2 * c3 + c1 * c2**2 + 2 * c1 * c2 * c3 + 3 * c1 * c3**2 + c2**3 + 2 * c2**2 * c3 + 3 * c2 * c3**2 + 4 * c3**3)]) * 2 * ST / 120

        b2[Ti, 0] = b2[Ti, 0] + b2T
    Cb1 = (qnorm(p, P, T, S, c)**p)**(1 / p - 1) * Lpnorma(4, P, T, S, c)**(-1 / 4)
    Cb2 = qnorm(p, P, T, S, c) * Lpnorma(4, P, T, S, c)**(-5 / 4)
    b = b1 * Cb1 - b2 * Cb2
    b[S - 1] = 0
    dJ = np.linalg.solve(A, b)
    dJ[S - 1] = 0
    if (p > 4):
        dJ = -dJ
    return dJ


def alokDJmin_exact_descent(p, P, T, S, A, u, c):
    n = len(P[0])
    nt = len(T[0])
    c[S - 1] = 0
    dJ = np.zeros([n, 1])
    c[S - 1] = 0
    b1 = np.zeros([n, 1])
    b2 = np.zeros([n, 1])
    bw = np.zeros([n, 1])
    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        u1 = u[Ti[0], 0]
        u2 = u[Ti[1], 0]
        u3 = u[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        invG_u = np.matmul(invG, [u2 - u1, u3 - u1])
        dJc = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [1, 0]))),
                       np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)
        dJ[Ti, 0] = dJ[Ti, 0] + dJc
        if (u1 == u2 and u2 == u3):
            b1T = [0, 0, 0]
        else:
            b1T = np.array([np.matmul(np.transpose(invG_u), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_u), (np.matmul(invG, [1, 0]))),
                           np.matmul(np.transpose(invG_u), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_u)**(p - 2)
        b1[Ti, 0] = b1[Ti, 0] + b1T
        if (c1 == c2 and c2 == c3):
            bwT = [0, 0, 0]
        else:
            bwT = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [1, 0]))),
                           np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)

        bw[Ti, 0] = bw[Ti, 0] + bwT
        b2T = np.array([(4 * u1**3 + 3 * u1**2 * u2 + 3 * u1**2 * u3 + 2 * u1 * u2**2 + 2 * u1 * u2 * u3 + 2 * u1 * u3**2 + u2**3 + u2**2 * u3 + u2 * u3**2 + u3**3), (u1**3 + 2 * u1**2 * u2 + u1**2 * u3 + 3 * u1 * u2**2 + 2 * u1 * u2 * u3 + u1 *
                       u3**2 + 4 * u2**3 + 3 * u2**2 * u3 + 2 * u2 * u3**2 + u3**3), (u1**3 + u1**2 * u2 + 2 * u1**2 * u3 + u1 * u2**2 + 2 * u1 * u2 * u3 + 3 * u1 * u3**2 + u2**3 + 2 * u2**2 * u3 + 3 * u2 * u3**2 + 4 * u3**3)]) * 2 * ST / 120
        b2[Ti, 0] = b2[Ti, 0] + b2T
    Cb1 = (qnorm(p, P, T, S, u)**p)**(1 / p - 1) * Lpnorma(4, P, T, S, u)**(-1 / 4)
    Cb2 = qnorm(p, P, T, S, u) * Lpnorma(4, P, T, S, u)**(-5 / 4)
    # if(p<4):
    #    b=bw-b1*Cb1+b2*Cb2;
    # else:
    b = bw + b1 * Cb1 - b2 * Cb2
    b[S - 1] = 0
    dJ = np.linalg.solve(A, b)
    dJ[S - 1] = 0
    return dJ


def alokDJminmax_exact_descent(p, P, T, S, A, u, c):

    n = len(P[0])
    nt = len(T[0])
    c[S - 1] = 0
    dJ = np.zeros([n, 1])
    c[S - 1] = 0

    b = np.zeros([n, 1])
    dJ = np.zeros([n, 1])

    for i in range(0, nt):
        Ti = (T[0, i] - 1, T[1, i] - 1, T[2, i] - 1)
        Pi = P[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = c[Ti[0], 0]
        c2 = c[Ti[1], 0]
        c3 = c[Ti[2], 0]
        u1 = u[Ti[0], 0]
        u2 = u[Ti[1], 0]
        u3 = u[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        invG_u = np.matmul(invG, [u2 - u1, u3 - u1])
        if (c1 == c2 and c2 == c3):
            dJc = [0, 0, 0]
        else:
            dJc = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [1, 0]))),
                           np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)
        if (u1 == u2 and u2 == u3):
            dJu = [0, 0, 0]
        else:
            dJu = np.array([np.matmul(np.transpose(invG_u), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_u), (np.matmul(invG, [1, 0]))),
                           np.matmul(np.transpose(invG_u), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_u)**(p - 2)

        uCT = np.array([(4 * u1**3 + 3 * u1**2 * u2 + 3 * u1**2 * u3 + 2 * u1 * u2**2 + 2 * u1 * u2 * u3 + 2 * u1 * u3**2 + u2**3 + u2**2 * u3 + u2 * u3**2 + u3**3), (u1**3 + 2 * u1**2 * u2 + u1**2 * u3 + 3 * u1 * u2**2 + 2 * u1 * u2 * u3 + u1 *
                       u3**2 + 4 * u2**3 + 3 * u2**2 * u3 + 2 * u2 * u3**2 + u3**3), (u1**3 + u1**2 * u2 + 2 * u1**2 * u3 + u1 * u2**2 + 2 * u1 * u2 * u3 + 3 * u1 * u3**2 + u2**3 + 2 * u2**2 * u3 + 3 * u2 * u3**2 + 4 * u3**3)]) * ST * 2 / 120

        if (p < 4):
            b[Ti, 0] = b[Ti, 0] + dJc - dJu + uCT
        else:
            b[Ti, 0] = b[Ti, 0] + dJc + dJu - uCT

    b[S - 1] = 0
    dJ = np.linalg.solve(A, b)
    dJ[S - 1] = 0
    return dJ


# %%%%%%%%%%%%% optimization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def maxRay(p, P, T, S, c):

    a = Laplace(p, P, T, S, c)
    b = RHS(p, P, T, S, c)
    t = (4 * b / (p * a))**(1 / (p - 4))

    J = alokJ(p, P, T, S, t * c)
    return t, J


def maxRayVal(p, P, T, S, c):
    a = Laplace(p, P, T, S, c)
    b = RHS(p, P, T, S, c)
    t = (4 * b / (p * a))**(1 / (p - 4))

    J = alokJ(p, P, T, S, t * c)
    return J


def zlatyrez(f, a, b, x, ddd, tol):

    gamma = 1 / 2 + np.sqrt(5) / 2
    a0 = a
    b0 = b
    d0 = (b0 - a0) / gamma + a0
    c0 = a0 + b0 - d0
    it = 0
    an = a0
    bn = b0
    cn = c0
    dn = d0

    fcn = f(x + cn * ddd)
    fdn = f(x + dn * ddd)

    while bn - an > tol:
        a = an
        b = bn
        c = cn
        d = dn
        fc = fcn
        fd = fdn

        if fc < fd:
            an = a
            bn = d
            dn = c
            cn = an + bn - dn
            fcn = f(x + cn * ddd)
            fdn = fc
        else:
            an = c
            bn = b
            cn = d
            dn = an + bn - cn
            fcn = fd
            fdn = f(x + dn * ddd)
        it = it + 1

    t = (an + bn) / 2
    val = f(x + t * ddd)
    return t, val


def gradientfunGoldenSection(p, P, T, S, init, f, df, tol):

    epsilon = 1e-8
    x = (init)
    it = 0
    q = p / (p - 1)
    g = df(x)
    gnorm = qnorm(p, P, T, S, g)
    if (qnorm(p, P, T, S, g) <= epsilon):
        gnorm = epsilon

    d = -(g) / gnorm
    t, valJ = zlatyrez(f, 0, min(qnorm(p, P, T, S, x) / 2, 1), x, d, 1e-6)
    xn = x + t * d

    while ((qnorm(q, P, T, S, g)) > tol):
        it = it + 1
        gnorm = qnorm(p, P, T, S, g)
        if (gnorm <= epsilon):
            gnorm = epsilon
        d = -(g) / gnorm
        t, valJ = zlatyrez(f, 0, min(qnorm(p, P, T, S, x) / 2, 1), x, d, 1e-6)
        xn = x + t * d

        x = xn
        g = df(x)
        qnorm(q, P, T, S, g)
    x = xn
    J = f(x)
    return x, J, it


def gradientfunBisection(p, P, T, S, init, f, df, tol):

    epsilon = 1e-8
    x = (init)
    it = 0
    q = p / (p - 1)
    g = df(x)
    gnorm = qnorm(p, P, T, S, g)
    if (qnorm(p, P, T, S, g) <= epsilon):
        gnorm = epsilon

    d = -(g) / gnorm

    Fx = f(x)
    xd = x + d
    Fdx = f(xd)
    while (Fx < Fdx):
        d = 1 / 2 * d
        xd = x + d
        Fdx = f(xd)
    xn = xd

    while ((qnorm(q, P, T, S, g)) > tol):
        it = it + 1
        gnorm = qnorm(p, P, T, S, g)
        if (gnorm <= epsilon):
            gnorm = epsilon
        d = -(g) / gnorm
        Fx = f(x)
        xd = x + d
        Fdx = f(xd)
        while (Fx < Fdx):
            d = 1 / 2 * d
            xd = x + d
            Fdx = f(xd)
        x = xd
        g = df(x)
        qnorm(q, P, T, S, g)
    x = xn
    J = f(x)
    return x, J, it


def DCCh(p, P, T, S, init, f, df, tol):

    it = 0
    q = p / (p - 1)
    w1 = init
    w1[S - 1] = 0
    t, Jw1 = maxRay(p, P, T, S, w1)
    w1 = t * w1
    dJ = df(w1)

    while (qnorm(q, P, T, S, dJ) > tol):
        dJ = dJ / qnorm(p, P, T, S, dJ)
        w = w1 - dJ
        t, Jw = maxRay(p, P, T, S, w)
# %%%%%%%%%%%%% puleni
#  while (Jw1<Jw):
#  dJ=1/2*dJ;
#  w= w1-dJ;
#  t,Jw=maxRay(w);
# %%%%%%%%%%%%%
        funGS = partial(maxRayVal, p, P, T, S)
        Te, val = zlatyrez(funGS, 0, 1, w1, -dJ, 1e-6)
        w = w1 - Te * dJ
# %%%%%%%%%%%%%
        t = maxRay(p, P, T, S, w)
        w1 = t * w
        Jw1 = Jw
        dJ = df(w1)
        it = it + 1
        Jw1 = f(w1)
    return w1, Jw1, it
