import numpy as np
from numpy import linalg as LA


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


def GP2D(Dmat, gpkernel, sigma, xscale, yscale, pshape=None):
    """GP 2D for different size between input and prediction.

    Args:
        Dmat: input 2D matrix
        gpkernel: GP kernel
        sigma: observational Gaussian noise std
        xscale: GP correlated length (hyperparameter) for X
        yscale: GP correlated length (hyperparameter) for Y
        kernel: GP kernel, rbf or matern32
        pshape: prediction matrix shape. If None, use the same shape as Dmat

    Returns:
        prediction 2D matrix
    """
    if pshape == None:
        pshape = np.shape(Dmat)

    rat = np.array(pshape)/np.array(np.shape(Dmat))
    Nx, Ny = np.shape(Dmat)

    x = (np.array(list(range(0, Nx)))+0.5)*rat[0]
    y = (np.array(list(range(0, Ny)))+0.5)*rat[1]

    Nxp, Nyp = pshape
    xp = np.array(list(range(0, Nxp)))+0.5
    yp = np.array(list(range(0, Nyp)))+0.5

    Kx = gpkernel(x, x, xscale)
    Ky = gpkernel(y, y, yscale)
    kapx, Ux = LA.eigh(Kx)
    kapy, Uy = LA.eigh(Ky)
    invL = 1.0/(np.outer(kapx, kapy)+sigma**2)
    P = invL*(np.dot(Ux.T, np.dot(Dmat, Uy)))
    Kxp = gpkernel(x, xp, xscale)
    Kyp = gpkernel(y, yp, yscale)
    Dest = (Kxp@Ux@P@Uy.T@Kyp.T)
    return Dest


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(seed=1)

    Nx = 128; Ny = 256
    xgrid = np.linspace(0, Nx, Nx)
    ygrid = np.linspace(0, Ny, Ny)
    sigma = 0.2
    Dmat = np.sin(xgrid[:, np.newaxis]/20) * np.sin(ygrid[np.newaxis, :]/20) + \
        np.random.randn(Nx, Ny)*sigma

    Dprer = GP2D(Dmat, RBF, sigma, 20., 20., pshape=None)
    Dprem = GP2D(Dmat, Matern32, sigma, 40., 40., pshape=None)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    plt.imshow(Dmat)
    ax.set_title('input')

    ax = fig.add_subplot(223)
    plt.imshow(Dprer)
    ax.set_title('prediction (RBF)')

    ax = fig.add_subplot(224)
    plt.imshow(Dprem)
    ax.set_title('prediction (Matern3/2)')
    plt.savefig("sample.png")
    plt.show()
