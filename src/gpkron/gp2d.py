import numpy as np


def RBF(obst, pret, tau):
    """RBF kernel
    Args:
        obst: input vector
        pret: prediction vector
        tau: scale

    Returns:
        kernel
    """
    Dt = obst - np.array([pret]).T
    return np.exp(-(Dt)**2/2/(tau**2))


def Matern32(obst, pret, tau):
    """Matern 3/2 kernel
    Args:
        obst: input vector
        pret: prediction vector
        tau: scale

    Returns:
        kernel
    """

    Dt = obst - np.array([pret]).T
    fac = np.sqrt(3.0)*np.abs(Dt)/tau
    return (1.0+fac)*np.exp(-fac)


def GP2D(Dmat, gpkernel, sigma, scale, pshape=None):
    """GP 2D for different size between input and prediction.

    Args:
        Dmat: input 2D matrix
        gpkernel: GP kernel
        sigma: observational Gaussian noise std
        xscale: GP correlated length (hyperparameter) for (X,Y)
        kernel: GP kernel, rbf or matern32
        pshape: prediction matrix shape. If None, use the same shape as Dmat

    Returns:
        prediction 2D matrix
    """
    if pshape == None:
        pshape = np.shape(Dmat)
    rat = np.array(pshape)/np.array(np.shape(Dmat))
    Nx, Ny = np.shape(Dmat)
    x = (np.array(list(range(0, Nx))))*rat[0]
    y = (np.array(list(range(0, Ny))))*rat[1]
    Nxp, Nyp = pshape
    xp = np.array(list(range(0, Nxp)))
    yp = np.array(list(range(0, Nyp)))
    Kx = gpkernel(x, x, scale[0])
    Ky = gpkernel(y, y, scale[1])
    kapx, Ux = np.linalg.eigh(Kx)
    kapy, Uy = np.linalg.eigh(Ky)
    invL = 1.0/(np.outer(kapx, kapy)+sigma**2)
    P = invL*(np.dot(Ux.T, np.dot(Dmat, Uy)))
    Kxp = gpkernel(x, xp, scale[0])
    Kyp = gpkernel(y, yp, scale[1])
    Dest = (Kxp@Ux@P@Uy.T@Kyp.T)
    return Dest
