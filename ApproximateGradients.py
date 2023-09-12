"""
approximate gradients
"""

import numpy as np
import scipy as sp
from Tools import qnorm, Lpnorm
from numba import jit, prange


@jit(nopython=True, parallel=True, fastmath=True)
def __energyGradientMinmax(p, Nodes, Triangles, Sides, u):
    # reseni Laplace b = J'(u)
    n = len(Nodes[0])
    nt = len(Triangles[0])
    u[Sides - 1] = 0
    dJ = np.zeros((n, 1))

    for i in prange(0, nt):
        #  Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        #  Pi = Nodes[:, Ti]
        Ti1 = Triangles[0, i] - 1
        Ti2 = Triangles[1, i] - 1
        Ti3 = Triangles[2, i] - 1
        t1 = Nodes[:, Ti1]
        t2 = Nodes[:, Ti2]
        t3 = Nodes[:, Ti3]
        Pi = np.empty((3, len(Nodes)))
        Pi[0, :] = t1
        Pi[1, :] = t2
        Pi[2, :] = t3
        Pi = Pi.T
        G = np.empty((len(Pi), 2))
        G[:, 0] = Pi[:, 1] - Pi[:, 0]
        G[:, 1] = Pi[:, 2] - Pi[:, 0]

        c1 = u[Ti1, 0]
        c2 = u[Ti2, 0]
        c3 = u[Ti3, 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = invG @ np.array([c2 - c1, c3 - c1])
        # dJc = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(
        # invG, [1, 0]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [0,
        # 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)
        dJc = np.array([
            invG_c.T @ (invG @ np.array([-1.0, -1.0])),
            invG_c.T @ (invG @ np.array([1.0, 0.0])),
            invG_c.T @ (invG @ np.array([0.0, 1.0]))
        ]) * ST * np.linalg.norm(invG_c)**(p - 2)

        fCT = np.array([(4 * c1**3 + 3 * c1**2 * c2 + 3 * c1**2 * c3 + 2 * c1 * c2**2 + 2 * c1 * c2 * c3 + 2 * c1 * c3**2 + c2**3 + c2**2 * c3 + c2 * c3**2 + c3**3),
                        (c1**3 + 2 * c1**2 * c2 + c1**2 * c3 + 3 * c1 * c2**2 + 2 * c1 * c2 *
                         c3 + c1 * c3**2 + 4 * c2**3 + 3 * c2**2 * c3 + 2 * c2 * c3**2 + c3**3),
                        (c1**3 + c1**2 * c2 + 2 * c1**2 * c3 + c1 * c2**2 + 2 * c1 * c2 * c3 + 3 * c1 * c3**2 + c2**3 + 2 * c2**2 * c3 + 3 * c2 * c3**2 + 4 * c3**3)]) * 2 * ST / 120

        #  dJ[Ti, 0] = dJ[Ti, 0] + dJc - fCT
        dJ[Ti1, 0] += dJc[0] - fCT[0]
        dJ[Ti2, 0] += dJc[1] - fCT[1]
        dJ[Ti3, 0] += dJc[2] - fCT[2]

    b = dJ
    b[Sides - 1] = 0
    return b


def energyGradientMinmax(p, Nodes, Triangles, Sides, A, u):
    b = __energyGradientMinmax(p, Nodes, Triangles, Sides, u)
    # dJ = np.linalg.solve(A, b)
    dJ = sp.sparse.linalg.spsolve(A, b)[:, None]
    dJ[Sides - 1] = 0

    if (p > 4):
        dJ = -dJ

    return dJ


def proportionGradientMin(p, Nodes, Triangles, Sides, A, u):
    # reseni Laplace b = I'(u)
    n = len(Nodes[0])
    nt = len(Triangles[0])
    u[Sides - 1] = 0
    b1 = np.zeros([n, 1])
    b2 = np.zeros([n, 1])
    u[Sides - 1] = 0
    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = u[Ti[0], 0]
        c2 = u[Ti[1], 0]
        c3 = u[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        if (c1 == c2 and c2 == c3):
            b1T = [0, 0, 0]
        else:
            b1T = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(
                invG, [1, 0]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)

        b1[Ti, 0] = b1[Ti, 0] + b1T

        b2T = np.array([(4 * c1**3 + 3 * c1**2 * c2 + 3 * c1**2 * c3 + 2 * c1 * c2**2 + 2 * c1 * c2 * c3 + 2 * c1 * c3**2 + c2**3 + c2**2 * c3 + c2 * c3**2 + c3**3),
                        (c1**3 + 2 * c1**2 * c2 + c1**2 * c3 + 3 * c1 * c2**2 + 2 * c1 * c2 *
                         c3 + c1 * c3**2 + 4 * c2**3 + 3 * c2**2 * c3 + 2 * c2 * c3**2 + c3**3),
                        (c1**3 + c1**2 * c2 + 2 * c1**2 * c3 + c1 * c2**2 + 2 * c1 * c2 * c3 + 3 * c1 * c3**2 + c2**3 + 2 * c2**2 * c3 + 3 * c2 * c3**2 + 4 * c3**3)]) * 2 * ST / 120

        b2[Ti, 0] = b2[Ti, 0] + b2T
    Cb1 = (qnorm(p, Nodes, Triangles, Sides, u)**p)**(1 / p - 1) * Lpnorm(4, Nodes, Triangles, Sides, u)**(-1 / 4)
    Cb2 = qnorm(p, Nodes, Triangles, Sides, u) * Lpnorm(4, Nodes, Triangles, Sides, u)**(-5 / 4)
    b = b1 * Cb1 - b2 * Cb2
    b[Sides - 1] = 0
    dJ = sp.sparse.linalg.spsolve(A, b)[:, None]
    dJ[Sides - 1] = 0
    if (p > 4):
        dJ = -dJ
    return dJ

# for num. error


def energyGradientPoisson(p, Nodes, Triangles, Sides, A, f, u):
    # reseni Laplace u = f
    n = len(Nodes[0])
    nt = len(Triangles[0])
    u[Sides - 1] = 0
    dJ = np.zeros([n, 1])
    u[Sides - 1] = 0
    ST = 0
    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = u[Ti[0], 0]
        c2 = u[Ti[1], 0]
        c3 = u[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        dJc = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(
            invG, [1, 0]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)
        dJ[Ti, 0] = dJ[Ti, 0] + dJc
    b1 = dJ
    fC = np.zeros([n, 1])
    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = u[Ti[0], 0]
        c2 = u[Ti[1], 0]
        c3 = u[Ti[2], 0]
        f1 = f[Ti[0], 0]
        f2 = f[Ti[1]]
        f3 = f[Ti[2], 0]
        fCT = np.array([(2 * f1 + f2 + f3), (f1 + 2 * f2 + f3), (f1 + f2 + 2 * f3)]) * 2 * ST * 1 / 24
        fC[Ti, 0] = fC[Ti, 0] + fCT
    b2 = fC
    b = b1 - b2
    b[Sides - 1] = 0
    dJ = sp.sparse.linalg.spsolve(A, b)[:, None]
    dJ[Sides - 1] = 0

    if (p > 4):
        dJ = -dJ

    return dJ
