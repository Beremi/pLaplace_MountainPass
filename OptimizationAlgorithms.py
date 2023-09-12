"""
Optimization tools.
"""

import numpy as np
from functools import partial
from Tools import qnorm, Laplace, RHS
from Functionals import energyFunctionalMinimax


def maxRay(p, Nodes, Triangles, Sides, u):

    a = Laplace(p, Nodes, Triangles, Sides, u)
    b = RHS(p, Nodes, Triangles, Sides, u)
    t = (4 * b / (p * a))**(1 / (p - 4))

    J = energyFunctionalMinimax(p, Nodes, Triangles, Sides, t * u)
    return t, J


def maxRayVal(p, Nodes, Triangles, Sides, u):
    a = Laplace(p, Nodes, Triangles, Sides, u)
    b = RHS(p, Nodes, Triangles, Sides, u)
    t = (4 * b / (p * a))**(1 / (p - 4))

    J = energyFunctionalMinimax(p, Nodes, Triangles, Sides, t * u)
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


def gradientfunGSgrad(p, Nodes, Triangles, Sides, init, f, df, tol):
    epsilon = 1e-8
    x = init
    it = 0
    q = p / (p - 1)
    g = df(x)
    gnorm = qnorm(p, Nodes, Triangles, Sides, g)
    if (gnorm <= epsilon):
        gnorm = epsilon

    d = -(g) / gnorm
    t, valJ = zlatyrez(f, 0, min(qnorm(p, Nodes, Triangles, Sides, x) / 2, 1), x, d, 1e-3)
    xn = x + t * d
    x = xn
    g = df(x)

    while ((qnorm(q, Nodes, Triangles, Sides, g)) > tol):
        it = it + 1
        gnorm = qnorm(p, Nodes, Triangles, Sides, g)
        if (gnorm <= epsilon):
            gnorm = epsilon
        d = -(g) / gnorm
        t, valJ = zlatyrez(f, 0, min(qnorm(p, Nodes, Triangles, Sides, x) / 2, 1), x, d, 1e-3)
        xn = x + t * d
        x = xn
        g = df(x)
    x = xn
    J = f(x)
    return x, J, it


def gradientfunGoldenSection(p, Nodes, Triangles, Sides, init, f, df, tol):
    epsilon = 1e-8
    x = init
    it = 0
    q = p / (p - 1)
    g = df(x)
    gnorm = qnorm(p, Nodes, Triangles, Sides, g)
    if (gnorm <= epsilon):
        gnorm = epsilon

    d = -(g) / gnorm
    t, valJ = zlatyrez(f, 0, min(qnorm(p, Nodes, Triangles, Sides, x) / 2, 1), x, d, 1e-3)
    xn = x + t * d
    x = xn
    g = df(x)

    while ((qnorm(q, Nodes, Triangles, Sides, g)) > tol):
        it = it + 1
        gnorm = qnorm(p, Nodes, Triangles, Sides, g)
        if (gnorm <= epsilon):
            gnorm = epsilon
        d = -(g) / gnorm
        t, valJ = zlatyrez(f, 0, min(qnorm(p, Nodes, Triangles, Sides, x) / 2, 1), x, d, 1e-3)
        xn = x + t * d
        x = xn
        g = df(x)
        print('out', gnorm, it)
    x = xn
    J = f(x)
    return x, J, it


def gradientfunBisection(p, Nodes, Triangles, Sides, init, f, df, tol):
    epsilon = 1e-8
    x = init
    it = 0
    q = p / (p - 1)
    g = df(x)
    gnorm = qnorm(p, Nodes, Triangles, Sides, g)
    if (qnorm(p, Nodes, Triangles, Sides, g) <= epsilon):
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

    while ((qnorm(q, Nodes, Triangles, Sides, g)) > tol):
        it = it + 1
        gnorm = qnorm(p, Nodes, Triangles, Sides, g)
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
    x = xn
    J = f(x)
    return x, J, it


def DCChMinimax(p, Nodes, Triangles, Sides, init, f, df, tol):
    it = 0
    q = p / (p - 1)
    u = init
    u[Sides - 1] = 0
    t, Ju = maxRay(p, Nodes, Triangles, Sides, u)
    u = t * u
    dJ = df(u)

    while (qnorm(q, Nodes, Triangles, Sides, dJ) > tol):
        dJ = dJ / qnorm(p, Nodes, Triangles, Sides, dJ)
        w = u - dJ
        t, Jw = maxRay(p, Nodes, Triangles, Sides, w)
        funGS = partial(maxRayVal, p, Nodes, Triangles, Sides)
        T, val = zlatyrez(funGS, 0, 1, u, -dJ, 1e-6)
        w = u - T * dJ
        t, J = maxRay(p, Nodes, Triangles, Sides, w)
        u = t * w
        Ju = Jw
        dJ = df(u)
        it = it + 1
        Ju = f(u)
        print(f"Iteration {it}, J = {Ju}.")
    return u, Ju, it
