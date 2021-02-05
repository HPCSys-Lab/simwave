# pywave

`pywave` is a Python package to simulate the propagation of the constant or variable density acoustic wave in a isotropic 2D/3D medium using the finite difference method. Finite difference kernels of aribtrary spatial order (up to 16th order) are written in C for performance and compiled at run time. These kernels are called via a user-friendly Python interface for easy integration with several scientific and engineering libraries to, for example perform full-waveform inversion. 

## Installation:

For installation, `pywave` needs only scipy, numpy, and segyio. See `requirements.txt`. If you wish to plot, then `matplotlib` is additionally required. `pywave` compiles finite difference stencils at run time in C for performance and thus requires a working C compiler.

`git clone https://github.com/HPCSys-Lab/pywave.git`

`cd pywave`

`pip3 install -e .`


## Contributing

All contributions are welcome!

To contribute to the software:

1. [Fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) the repository.
2. Clone the forked repository, add your contributions and push the changes to your fork.
3. Create a [Pull request](https://github.com/HPCSys-Lab/pywave/pulls)

Before creating the pull request, make sure that the tests pass by running 
```
tox
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

Simulation with `pywave` is simple! Jupyter notebooks with tutorials can be found here [here](https://github.com/HPCSys-Lab/pywave/tree/master/tutorial).

Here we show how to simulate the constant density acoustic wave equation on a simple two layer velocity model. 
```python
from pywave import *
import numpy as np

# shape of the grid
shape = (512, 512)

# spacing
spacing = (15.0, 15.0)

# propagation time
time = 2000

# Velocity model
vel = np.zeros(shape, dtype=np.float32)
vel[:] = 1500.0
velModel = Model(ndarray=vel)

# Compiler
compiler = Compiler(program_version='sequential')

# domain extension (damping + spatial order halo)
extension = DomainExtension(nbl=50, degree=3, alpha=0.0001)

# Wavelet
wavelet = Wavelet(frequency=5.0)

# Source
source = Source(kws_half_width=1, wavelet=wavelet)
source.add(position=(30,0))

# receivers
receivers = Receiver(kws_half_width=1)

for i in range(512):
    receivers.add(position=(15,i))

setup = Setup(
    velocity_model=velModel,
    sources=source,
    receivers=receivers,
    domain_extension=extension,
    spacing=spacing,
    propagation_time=time,
    jumps=1,
    compiler=compiler
)

solver = AcousticSolver(setup=setup)

wavefields, rec, exec_time = solver.forward()

'''
count=0
for wavefield in wavefields:
    plot(wavefield, file_name="arq-"+str(count))
    count += 1
'''

print("Forward execution time: %f seconds" % exec_time)

plot_wavefield(wavefields)
plot_shotrecord(rec)
```

## Performance 

- TO DO 
