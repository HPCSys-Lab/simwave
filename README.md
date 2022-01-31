[![PyPI version](https://badge.fury.io/py/simwave.svg)](https://badge.fury.io/py/simwave)
[![DOI](https://zenodo.org/badge/285108327.svg)](https://zenodo.org/badge/latestdoi/285108327)

# Simwave

`Simwave` is a Python package to simulate the propagation of the constant or variable density acoustic wave in an isotropic 2D/3D medium using the finite difference method. Finite difference kernels of aribtrary spatial order (up to 20th order) are written in C for performance and compiled at run time. These kernels are called via a user-friendly Python interface for easy integration with several scientific and engineering libraries to, for example perform full-waveform inversion.

For further information on the `simwave` design and implementation, please see the paper https://arxiv.org/abs/2201.05278

## Installation:

For installation, `simwave` needs only scipy, numpy, and segyio. See `requirements.txt`. If you wish to plot, then `matplotlib` is additionally required. `simwave` compiles finite difference stencils at run time in C for performance and thus requires a working C compiler.

`git clone https://github.com/HPCSys-Lab/simwave.git`

`cd simwave`

`pip3 install -e .`


## Contributing

All contributions are welcome.

To contribute to the software:

1. [Fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) the repository.
2. Clone the forked repository, add your contributions and push the changes to your fork.
3. Create a [Pull request](https://github.com/HPCSys-Lab/simwave/pulls)

Before creating the pull request, make sure that the tests pass by running
```
pytest
```
Some things that will increase the chance that your pull request is accepted:
-  Write tests.
- Add Python docstrings that follow the [Sphinx](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html).
- Write good commit and pull request messages.


[style]: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html

Problems?
==========

If something isn't working as it should or you'd like to recommend a new addition/feature to the software, please let us know by starting an issue through the [issues](https://github.com/HPCSys-Lab/pywave/issues) tab. I'll try to get to it as soon as possible.

Examples
========

Simulation with `simwave` is simple and can be accomplished in a dozen or so lines of Python! Jupyter notebooks with tutorials can be found here [here](https://github.com/HPCSys-Lab/simwave/tree/master/tutorial).

Here we show how to simulate the constant density acoustic wave equation on a simple two layer velocity model.
```python
from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler,
    Receiver, Source, plot_wavefield, plot_shotrecord, plot_velocity_model
)
import numpy as np


# set compiler options
# available language options: c (sequential) or  cpu_openmp (parallel CPU)
compiler = Compiler(
    cc='gcc',
    language='cpu_openmp',
    cflags='-O3 -fPIC -Wall -std=c99 -shared'
)

# Velocity model
vel = np.zeros(shape=(512, 512), dtype=np.float32)
vel[:] = 1500.0
vel[100:] = 2000.0

# create the space model
space_model = SpaceModel(
    bounding_box=(0, 5120, 0, 5120),
    grid_spacing=(10, 10),
    velocity_model=vel,
    space_order=4,
    dtype=np.float32
)

# config boundary conditions
# (none,  null_dirichlet or null_neumann)
space_model.config_boundary(
    damping_length=0,
    boundary_condition=(
        "null_neumann", "null_dirichlet",
        "none", "null_dirichlet"
    ),
    damping_polynomial_degree=3,
    damping_alpha=0.001
)

# create the time model
time_model = TimeModel(
    space_model=space_model,
    tf=1.0,
    saving_stride=0
)

# create the set of sources
source = Source(
    space_model,
    coordinates=[(2560, 2560)],
    window_radius=1
)

# crete the set of receivers
receiver = Receiver(
    space_model=space_model,
    coordinates=[(2560, i) for i in range(0, 5112, 10)],
    window_radius=1
)

# create a ricker wavelet with 10hz of peak frequency
ricker = RickerWavelet(10.0, time_model)

# create the solver
solver = Solver(
    space_model=space_model,
    time_model=time_model,
    sources=source,
    receivers=receiver,
    wavelet=ricker,    
    compiler=compiler
)

# run the forward
u_full, recv = solver.forward()

print("u_full shape:", u_full.shape)
plot_velocity_model(space_model.velocity_model)
plot_wavefield(u_full[-1])
plot_shotrecord(recv)
```


