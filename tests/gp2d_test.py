if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from gpkron.gp2d import GP2D, RBF, Matern32
    np.random.seed(seed=1)

#    Nx = 128; Ny = 256
#    pshape=None
    Nx = 16; Ny = 32
    pshape=(64,128)

    xgrid = np.linspace(0, Nx, Nx)
    ygrid = np.linspace(0, Ny, Ny)
    sigma = 0.2
#    Dmat = np.sin(xgrid[:, np.newaxis]/20) * np.sin(ygrid[np.newaxis, :]/20) + \
#        np.random.randn(Nx, Ny)*sigma
    Dmat = np.sin(xgrid[:, np.newaxis]/4) * np.sin(ygrid[np.newaxis, :]/4) + \
        np.random.randn(Nx, Ny)*sigma

    Dprer = GP2D(Dmat, RBF, sigma, (20., 20.), pshape=pshape)
    Dprem = GP2D(Dmat, Matern32, sigma, (40., 40.), pshape=pshape)

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
