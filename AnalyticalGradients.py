"""
for finding analytical "gradient" using Poisson problem - lower loop
functionals
"""

import numpy as np
import scipy as sp
from functools import partial
from ApproximateGradients import energyGradientMinmax, proportionGradientMin
from OptimizationAlgorithms import gradientfunGSgrad
from Tools import Lpnorm, qnorm, Laplace


def analyticalGradEnergyMinmax_functional(p, Nodes, Triangles, Sides, uInitial, v):
    # pro reseni pLaplace v = J'(uInitial)
    nt = len(Triangles[0])
    v[Sides - 1] = 0
    Jp = 0
    JpU1 = 0
    JpT = 0
    JpU2 = 0
    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = v[Ti[0], 0]
        c2 = v[Ti[1], 0]
        c3 = v[Ti[2], 0]
        u1 = uInitial[Ti[0], 0]
        u2 = uInitial[Ti[1], 0]
        u3 = uInitial[Ti[2], 0]

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


def analyticalGradProportionMin_functional(p, Nodes, Triangles, Sides, uInitial, v):
    # pro reseni pLaplace v = I'(uInitial)
    nt = len(Triangles[0])
    v[Sides - 1] = 0
    b1 = 0
    b2 = 0

    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = v[Ti[0], 0]
        c2 = v[Ti[1], 0]
        c3 = v[Ti[2], 0]
        u1 = uInitial[Ti[0], 0]
        u2 = uInitial[Ti[1], 0]
        u3 = uInitial[Ti[2], 0]

        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        invG_u = np.matmul(invG, [u2 - u1, u3 - u1])

        if (c1 == c2 and c2 == c3):
            b1T = 0
        else:
            b1T = ST * np.linalg.norm(invG_u)**(p - 2) * np.matmul(np.transpose(invG_u), invG_c)
        b2T = 2 * ST / 120 * (4 * c1 * u1**3 + 3 * c1 * u1**2 * u2 + 3 * c1 * u1**2 * u3 + 2 * c1 * u1 * u2**2 + 2 * c1 * u1 * u2 * u3 +
                              2 * c1 * u1 * u3**2 + c1 * u2**3 + c1 * u2**2 * u3 + c1 * u2 * u3**2 + c1 * u3**3 + c2 * u1**3 +
                              2 * c2 * u1**2 * u2 + c2 * u1**2 * u3 + 3 * c2 * u1 * u2**2 + 2 * c2 * u1 * u2 * u3 + c2 * u1 * u3**2 +
                              4 * c2 * u2**3 + 3 * c2 * u2**2 * u3 + 2 * c2 * u2 * u3**2 + c2 * u3**3 + c3 * u1**3 + c3 * u1**2 * u2 +
                              2 * c3 * u1**2 * u3 + c3 * u1 * u2**2 + 2 * c3 * u1 * u2 * u3 + 3 * c3 * u1 * u3**2 + c3 * u2**3 +
                              2 * c3 * u2**2 * u3 + 3 * c3 * u2 * u3**2 + 4 * c3 * u3**3)
        b1 = b1 + b1T
        b2 = b2 + b2T

    Cb1 = (qnorm(p, Nodes, Triangles, Sides, uInitial)**p)**(1 / p - 1) * \
        Lpnorm(4, Nodes, Triangles, Sides, uInitial)**(-1 / 4)
    Cb2 = qnorm(p, Nodes, Triangles, Sides, uInitial) * Lpnorm(4, Nodes, Triangles, Sides, uInitial)**(-5 / 4)
    b = b1 * Cb1 - b2 * Cb2
    J = Laplace(p, Nodes, Triangles, Sides, v) - b
    if (p > 4):
        J = J
    return J

# gradients for Poisson problem


def analyticalGradProportionMin_gradient(p, Nodes, Triangles, Sides, A, uInitial, v):
    n = len(Nodes[0])
    nt = len(Triangles[0])
    v[Sides - 1] = 0
    dJ = np.zeros([n, 1])
    v[Sides - 1] = 0
    b1 = np.zeros([n, 1])
    b2 = np.zeros([n, 1])
    bw = np.zeros([n, 1])
    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = v[Ti[0], 0]
        c2 = v[Ti[1], 0]
        c3 = v[Ti[2], 0]
        u1 = uInitial[Ti[0], 0]
        u2 = uInitial[Ti[1], 0]
        u3 = uInitial[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        invG_u = np.matmul(invG, [u2 - u1, u3 - u1])
        dJc = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(
            invG, [1, 0]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)
        dJ[Ti, 0] = dJ[Ti, 0] + dJc
        if (u1 == u2 and u2 == u3):
            b1T = [0, 0, 0]
        else:
            b1T = np.array([np.matmul(np.transpose(invG_u), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_u), (np.matmul(
                invG, [1, 0]))), np.matmul(np.transpose(invG_u), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_u)**(p - 2)
        b1[Ti, 0] = b1[Ti, 0] + b1T
        if (c1 == c2 and c2 == c3):
            bwT = [0, 0, 0]
        else:
            bwT = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(
                invG, [1, 0]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)

        bw[Ti, 0] = bw[Ti, 0] + bwT
        b2T = np.array([(4 * u1**3 + 3 * u1**2 * u2 + 3 * u1**2 * u3 + 2 * u1 * u2**2 + 2 * u1 * u2 * u3 + 2 * u1 * u3**2 + u2**3 + u2**2 * u3 + u2 * u3**2 + u3**3),
                        (u1**3 + 2 * u1**2 * u2 + u1**2 * u3 + 3 * u1 * u2**2 + 2 * u1 * u2 *
                         u3 + u1 * u3**2 + 4 * u2**3 + 3 * u2**2 * u3 + 2 * u2 * u3**2 + u3**3),
                        (u1**3 + u1**2 * u2 + 2 * u1**2 * u3 + u1 * u2**2 + 2 * u1 * u2 * u3 + 3 * u1 * u3**2 + u2**3 + 2 * u2**2 * u3 + 3 * u2 * u3**2 + 4 * u3**3)]) * 2 * ST / 120
        b2[Ti, 0] = b2[Ti, 0] + b2T
    Cb1 = (qnorm(p, Nodes, Triangles, Sides, uInitial)**p)**(1 / p - 1) * \
        Lpnorm(4, Nodes, Triangles, Sides, uInitial)**(-1 / 4)
    Cb2 = qnorm(p, Nodes, Triangles, Sides, uInitial) * Lpnorm(4, Nodes, Triangles, Sides, uInitial)**(-5 / 4)
    if (p < 4):
        b = bw - b1 * Cb1 + b2 * Cb2
    else:
        b = bw + b1 * Cb1 - b2 * Cb2
    b[Sides - 1] = 0
    dJ = sp.sparse.linalg.spsolve(A, b)[:, None]
    dJ[Sides - 1] = 0
    return dJ


def analyticalGradEnergyMinmax_gradient(p, Nodes, Triangles, Sides, A, uInitial, v):
    n = len(Nodes[0])
    nt = len(Triangles[0])
    v[Sides - 1] = 0
    dJ = np.zeros([n, 1])
    v[Sides - 1] = 0

    b = np.zeros([n, 1])
    dJ = np.zeros([n, 1])

    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = v[Ti[0], 0]
        c2 = v[Ti[1], 0]
        c3 = v[Ti[2], 0]
        u1 = uInitial[Ti[0], 0]
        u2 = uInitial[Ti[1], 0]
        u3 = uInitial[Ti[2], 0]
        ST = abs(np.linalg.det(G)) / 2
        invG = np.linalg.inv(np.transpose(G))
        invG_c = np.matmul(invG, [c2 - c1, c3 - c1])
        invG_u = np.matmul(invG, [u2 - u1, u3 - u1])
        if (c1 == c2 and c2 == c3):
            dJc = [0, 0, 0]
        else:
            dJc = np.array([np.matmul(np.transpose(invG_c), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_c), (np.matmul(
                invG, [1, 0]))), np.matmul(np.transpose(invG_c), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_c)**(p - 2)
        if (u1 == u2 and u2 == u3):
            dJu = [0, 0, 0]
        else:
            dJu = np.array([np.matmul(np.transpose(invG_u), (np.matmul(invG, [-1, -1]))), np.matmul(np.transpose(invG_u), (np.matmul(
                invG, [1, 0]))), np.matmul(np.transpose(invG_u), (np.matmul(invG, [0, 1])))]) * ST * np.linalg.norm(invG_u)**(p - 2)

        uCT = np.array([(4 * u1**3 + 3 * u1**2 * u2 + 3 * u1**2 * u3 + 2 * u1 * u2**2 + 2 * u1 * u2 * u3 + 2 * u1 * u3**2 + u2**3 + u2**2 * u3 + u2 * u3**2 + u3**3),
                        (u1**3 + 2 * u1**2 * u2 + u1**2 * u3 + 3 * u1 * u2**2 + 2 * u1 * u2 *
                         u3 + u1 * u3**2 + 4 * u2**3 + 3 * u2**2 * u3 + 2 * u2 * u3**2 + u3**3),
                        (u1**3 + u1**2 * u2 + 2 * u1**2 * u3 + u1 * u2**2 + 2 * u1 * u2 * u3 + 3 * u1 * u3**2 + u2**3 + 2 * u2**2 * u3 + 3 * u2 * u3**2 + 4 * u3**3)]) * ST * 2 / 120

        if (p < 4):
            b[Ti, 0] = b[Ti, 0] + dJc - dJu + uCT
        else:
            b[Ti, 0] = b[Ti, 0] + dJc + dJu - uCT

    b[Sides - 1] = 0
    dJ = sp.sparse.linalg.spsolve(A, b)[:, None]
    dJ[Sides - 1] = 0
    return dJ

# analytical gradients


def energyGradientMinmax_analytical(p, Nodes, Triangles, Sides, A, u):
    init = energyGradientMinmax(p, Nodes, Triangles, Sides, A, u)
    f = partial(analyticalGradEnergyMinmax_functional, p, Nodes, Triangles, Sides, u)
    df = partial(analyticalGradEnergyMinmax_gradient, p, Nodes, Triangles, Sides, A, u)
    dJ, val, it = gradientfunGSgrad(p, Nodes, Triangles, Sides, init, f, df, 1e-3)
    return dJ


def proportionGradientMin_analytical(p, Nodes, Triangles, Sides, A, u):
    init = proportionGradientMin(p, Nodes, Triangles, Sides, A, u)
    f = partial(analyticalGradProportionMin_functional, p, Nodes, Triangles, Sides, u)
    df = partial(analyticalGradProportionMin_gradient, p, Nodes, Triangles, Sides, A, u)
    dJ, val, it = gradientfunGSgrad(p, Nodes, Triangles, Sides, init, f, df, 1e-3)
    return dJ
