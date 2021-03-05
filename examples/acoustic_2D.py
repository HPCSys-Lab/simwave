from pywave import *
import numpy as np

# Velocity model
vel = np.zeros(shape=(512,512), dtype=np.float32)
vel[:] = 1500.0
vel[100:] = 2000.0

# create the space model
space_model = SpaceModel(
    bbox=(0, 5120, 0, 5120),
    grid_spacing=(10., 10.),
    velocity_model=vel,
    space_order=2
)

# config boundary conditions
space_model.config_boundary(
    damping_length=0.0,
    boundary_condition=("ND", "NN", "N", "NN"),
    damping_polynomial_degree=1,
    damping_alpha=0.001
)

# create the time model
time_model = TimeModel(
    space_model=space_model,
    t0=0.0,
    tf=1.0
)

# create the set of sources
source = Source(space_model, coordinates=[], window_radius=4)
source.add((10,2560))
source.add((2560,2560))

# crete the set of receivers
receiver = Receiver(
    space_model=space_model,
    coordinates=[(2560,i) for i in range(0,5112,10)],
    window_radius=4
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
    saving_jump=0,
    compiler=None
)

# run the forward
u_full, recv = solver.forward()

print("u_full shape:", u_full.shape)
plot_wavefield(u_full[-1])
plot_shotrecord(recv)
