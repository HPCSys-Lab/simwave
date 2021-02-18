from pywave import *
import numpy as np

# shape of the grid
shape = (100, 100, 100)

# spacing
spacing = (15.0, 15.0, 15.0)

# propagation time
time = 600

# Velocity model
vel = np.zeros(shape, dtype=np.float32)
vel[:] = 1500.0
velModel = Model(ndarray=vel)

# Compiler
compiler = Compiler()

# domain extension (damping + spatial order halo)
extension = BoundaryProcedures(
    nbl=0,
    boundary_condition=(("NN", "NN"), ("NN", "NN"), ("NN", "NN")),
    damping_polynomial_degree=3,
    alpha=0.0001,
)

# Wavelet
wavelet = Wavelet(frequency=15.0)

# Source
source = Source(kws_half_width=1, wavelet=wavelet)
source.add(position=(50, 50, 50))

# receivers
receivers = Receiver(kws_half_width=1)

for i in range(100):
    receivers.add(position=(50, i, i))

setup = Setup(
    velocity_model=velModel,
    sources=source,
    receivers=receivers,
    domain_pad=extension,
    spacing=spacing,
    propagation_time=time,
    jumps=0,
    compiler=compiler,
    space_order=2,
)

solver = AcousticSolver(setup=setup)

wavefields, rec, exec_time = solver.forward()

print(wavefields.shape)

if len(wavefields.shape) > 3:
    count = 0
    for wavefield in wavefields:
        plot_wavefield(wavefield[50, :, :], file_name="arq-" + str(count))
        count += 1
else:
    plot_wavefield(wavefields[50, :, :])


print("Forward execution time: %f seconds" % exec_time)

plot_shotrecord(rec)
