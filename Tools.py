import numpy as np
from numba import jit, prange


def discretize(a, b, hx, hy):
    # diskretizace obdelniku o stranach a, b s kroky hx a hy, vraci uzly, indexy
    # uzlu v trojuhelniku a indexy uzlu na stranach obdelniku
    x = np.arange(0, a + hx, hx)
    y = np.arange(0, b + hy, hy)

    nx = len(x)
    ny = len(y)
    nTx = nx - 1
    nTy = ny - 1
    [X, Y] = np.meshgrid(x, y)
    Nodes = np.array([np.reshape(X, nx * nx, order='F'), np.reshape(Y, ny * ny, order='F')])
    k = 0
    Triangles = np.ones((3, 2 * (nTx * nTy)), dtype=int)
    for i in range(1, nTx + 1):
        for j in range(1, nTy + 1):
            C1 = j + (i - 1) * ny
            C2 = (j + 1) + (i - 1) * ny
            C3 = (j) + (i) * ny
            C4 = (j + 1) + (i) * ny
            Triangles[:, k * 2] = [C3, C1, C4]
            Triangles[:, k * 2 + 1] = [C2, C4, C1]
            k = k + 1

    side1 = np.arange(1, ny + 1, 1)
    side2 = np.arange(ny, (nx * ny) + 1, ny)
    side3 = np.arange(((ny * nx) - ny + 1), (nx * ny) + 1, 1)
    side4 = np.arange(1, ((nx * ny) - ny + 2), ny)
    return Nodes, Triangles, side1, side2, side3, side4


def alokA(Nodes, Triangles, Sides_Dirichlet):

    n = len(Nodes[0])
    nt = len(Triangles[0])

    ID = Sides_Dirichlet
    natNumN = np.arange(1, n + 1, 1)
    A = np.zeros([n, n])
    Q = np.array([[-1, 1, 0], [-1, 0, 1]])

    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
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


@jit(nopython=True, parallel=True, fastmath=True)
def qnorm(q, Nodes, Triangles, Sides_Dirichlet, u):
    # (int |grad(u)|^p)^(1/p)
    u = u.astype(np.float64)
    nt = len(Triangles[0])
    Jp = 0
    for i in prange(0, nt):
        # Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        # Pi = Nodes[:, Ti]
        # G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        # c1 = u[Ti[0], 0]
        # c2 = u[Ti[1], 0]
        # c3 = u[Ti[2], 0]
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
        cvect = np.array([-c1 + c2, -c1 + c3]).T
        JpT = (1 / q) * ST * (np.linalg.norm(invG @ cvect))**q
        Jp = Jp + JpT

    qnorm = (Jp)**(1 / q)
    return qnorm


def Lpnorm(q, Nodes, Triangles, S, u):
    # (int |(u)|^p)^(1/p)
    nt = len(Triangles[0])
    Jp = 0
    for i in range(0, nt):
        Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        Pi = Nodes[:, Ti]
        G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        c1 = u[Ti[0]]
        c2 = u[Ti[1]]
        c3 = u[Ti[2]]
        JpT = 1 / 30 * (c1**4 + c1**3 * c2 + c1**3 * c3 + c1**2 * c2**2 + c1**2 * c2 * c3 + c1**2 * c3**2 + c1 * c2 **
                        3 + c1 * c2**2 * c3 + c1 * c2 * c3**2 + c1 * c3**3 + c2**4 + c2**3 * c3 + c2**2 * c3**2 + c2 * c3**3 + c3**4)
        Jp = Jp + abs(np.linalg.det(G)) * JpT

    return Jp

# %%%%%%%%%%%%% misc - parts of the energy functional %%%%%%%%%%%%%%


@jit(nopython=True, parallel=True, fastmath=True)
def Laplace(p, Nodes, Triangles, Sides, u):
    # (int |grad(u)|^p)
    nt = len(Triangles[0])
    Jp = 0
    for i in prange(0, nt):
        # Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        # Pi = Nodes[:, Ti]
        # G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        # c1 = u[Ti[0], 0]
        # c2 = u[Ti[1], 0]
        # c3 = u[Ti[2], 0]
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
        cvect = np.array([-c1 + c2, -c1 + c3]).T
        JpT = ST * (np.linalg.norm(invG @ cvect))**p
        Jp = Jp + (1 / p) * JpT
    return Jp


@jit(nopython=True)
def RHS(p, Nodes, Triangles, Sides, u):
    # int u^4
    nt = len(Triangles[0])
    integ = 0
    for i in range(0, nt):
        # Ti = (Triangles[0, i] - 1, Triangles[1, i] - 1, Triangles[2, i] - 1)
        # Pi = Nodes[:, Ti]
        # G = np.transpose(np.array([Pi[:, 1] - Pi[:, 0], Pi[:, 2] - Pi[:, 0]]))
        # c1 = u[Ti[0], 0]
        # c2 = u[Ti[1], 0]
        # c3 = u[Ti[2], 0]
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
        intT = 1 / 30 * c1**4 + 1 / 30 * c1**3 * c2 + 1 / 30 * c1**3 * c3 + 1 / 30 * c1**2 * c2**2 + 1 / 30 * c1**2 * c2 * c3 + 1 / 30 * c1**2 * c3**2 +\
            1 / 30 * c1 * c2**3 + 1 / 30 * c1 * c2**2 * c3 + 1 / 30 * c1 * c2 * c3**2 + 1 / 30 * c1 * c3**3 + 1 / 30 * c2**4 + 1 / 30 * c2**3 * c3 +\
            1 / 30 * c2**2 * c3**2 + 1 / 30 * c2 * c3**3 + 1 / 30 * c3**4
        intT = abs(np.linalg.det(G)) * intT
        integ = integ + intT

    f = integ
    J = 1 / 4 * f
    return J
