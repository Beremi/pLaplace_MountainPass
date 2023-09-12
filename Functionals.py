"""
energy functionals
for mountain pass/alternative mountain pass - outer loop
"""

import numpy as np
from Tools import qnorm, Lpnorm
from numba import jit, prange


@jit(nopython=True, parallel=True, fastmath=True)
def energyFunctionalMinimax(p, Nodes, Triangles, Sides, u):
    # 1/p*(int |grad(u)|^p) - 1/4*(int u^4)
    nt = len(Triangles[0])
    integ = 0
    Jp = 0
    u[Sides - 1] = 0
    for i in prange(0, nt):
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
        intT = (1 / 30 * c1**4 + 1 / 30 * c1**3 * c2 + 1 / 30 * c1**3 * c3 + 1 / 30 * c1**2 * c2**2 + 1 / 30 * c1**2 * c2 * c3 +
                1 / 30 * c1**2 * c3**2 + 1 / 30 * c1 * c2**3 + 1 / 30 * c1 * c2**2 * c3 + 1 / 30 * c1 * c2 * c3**2 +
                1 / 30 * c1 * c3**3 + 1 / 30 * c2**4 + 1 / 30 * c2**3 * c3 + 1 / 30 * c2**2 * c3**2 +
                1 / 30 * c2 * c3**3 + 1 / 30 * c3**4)
        intT = abs(np.linalg.det(G)) * intT
        integ = integ + intT
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        cvect = np.array([-c1 + c2, -c1 + c3]).T
        JpT = (1 / p) * ST * (np.linalg.norm(invG @ cvect))**p

        Jp = Jp + JpT
    J = Jp - 1 / 4 * integ
    if (p > 4):
        J = -J
    return J


def proportionFunctionalMin(p, Nodes, Triangles, Sides, u):
    # ||u||1,p,0 / ||u||L4

    J = qnorm(p, Nodes, Triangles, Sides, u) / (Lpnorm(p, Nodes, Triangles, Sides, u)**(1 / 4))
    return J

# for evaluating num. error


def energyFunctionalPoisson(p, Nodes, Triangles, Sides, f, u):
    # 1/p*(int |grad(u)|^p) - (int fu)
    nt = len(Triangles[0])
    Jp = 0
    u[Sides - 1] = 0

    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = u[Ti[0], 0]
        c2 = u[Ti[1], 0]
        c3 = u[Ti[2], 0]
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
