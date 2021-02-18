from pywave import *
import numpy as np

# shape of the grid
shape = (512, 512)

# spacing
spacing = (15.0, 15.0)

# propagation time
time = 1500

# Velocity model
vel = np.zeros(shape, dtype=np.float32)
vel[:] = 1500.0
vel[200:] = 2000.0
velModel = Model(ndarray=vel)

# Compiler
compiler = Compiler(cc="gcc", cflags="-O3 -shared")

# domain extension (damping + spatial order halo)
extension = BoundaryProcedures(
    nbl=50,
    boundary_condition=(("NN", "NN"), ("NN", "NN")),
    damping_polynomial_degree=3,
    alpha=0,
)

# Wavelet
wavelet = Wavelet(frequency=5.0)

# Source
source = Source(kws_half_width=1, wavelet=wavelet)
source.add(position=(255.5, 255.5))

# receivers
receivers = Receiver(kws_half_width=1)

for i in range(512):
    receivers.add(position=(255.5, i))

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

if len(wavefields.shape) > 2:
    count = 0
    for wavefield in wavefields:
        plot_wavefield(wavefield, file_name="arq-" + str(count))
        count += 1
else:
    plot_wavefield(wavefields)

print("Forward execution time: %f seconds" % exec_time)

# plot(wavefield[1:512,damp+1:512+damp])

plot_shotrecord(rec)
