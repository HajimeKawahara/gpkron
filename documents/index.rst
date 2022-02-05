.. exojax documentation master file, created by
   sphinx-quickstart on Mon Jan 11 14:38:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gpkron
==================================

Example 1
----------------

.. code:: python
              
    from gpkron.gp2d import GP2D, RBF, Matern32
    import numpy as np
    Nx = 128; Ny = 256
    xgrid = np.linspace(0, Nx, Nx)
    ygrid = np.linspace(0, Ny, Ny)
    sigma = 0.2
    Dmat = np.sin(xgrid[:, np.newaxis]/20) * np.sin(ygrid[np.newaxis, :]/20) + \
        np.random.randn(Nx, Ny)*sigma
    Dprer = GP2D(Dmat, RBF, sigma, (20., 20.))
    Dprem = GP2D(Dmat, Matern32, sigma, (40., 40.))

Example 2
----------------
    
.. code:: python
                  
    from gpkron.gp2d import GP2D, RBF, Matern32
    import numpy as np
    Nx = 16; Ny = 32
    pshape=(64,128)

    xgrid = np.linspace(0, Nx, Nx)
    ygrid = np.linspace(0, Ny, Ny)
    sigma = 0.2
    Dmat = np.sin(xgrid[:, np.newaxis]/4) * np.sin(ygrid[np.newaxis, :]/4) + \
        np.random.randn(Nx, Ny)*sigma
    Dprer = GP2D(Dmat, RBF, sigma, (20., 20.), pshape=pshape)
    Dprem = GP2D(Dmat, Matern32, sigma, (40., 40.), pshape=pshape)



Contents
==================================

   
.. toctree::
   :maxdepth: 1
   :caption: API:

   gpkron/gpkron.rst
