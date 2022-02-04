# gpkron

A simple and fast 2D gaussian process fitting using Kronecker product.

```python
    from gpkron.gp2d import GP2D, RBF, Matern32
    import numpy as np
    Nx = 128; Ny = 256
    xgrid = np.linspace(0, Nx, Nx)
    ygrid = np.linspace(0, Ny, Ny)
    sigma = 0.2
    Dmat = np.sin(xgrid[:, np.newaxis]/20) * np.sin(ygrid[np.newaxis, :]/20) + \
        np.random.randn(Nx, Ny)*sigma
    Dprer = GP2D(Dmat, RBF, sigma, 20., 20., pshape=None)
    Dprem = GP2D(Dmat, Matern32, sigma, 40., 40., pshape=None)
```

![sample](https://user-images.githubusercontent.com/15956904/152613707-75c7843b-605d-4e62-bf04-32ce5bfa3551.png)