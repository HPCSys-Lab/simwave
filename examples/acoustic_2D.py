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
    cflags='-O3 -fPIC -ffast-math -Wall -std=c99 -shared'
    # cflags='-O3 -fPIC -ffast-math -fopenmp \
    #       -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75'
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
    saving_stride=12
)

# create the set of sources
source = Source(
    space_model,
    coordinates=[(2560, 2560)],
    window_radius=4
)

# crete the set of receivers
receiver = Receiver(
    space_model=space_model,
    coordinates=[(2560, i) for i in range(0, 5120, 10)],
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
    compiler=compiler
)

print("Timesteps:", time_model.timesteps)

# run the forward
u_full, recv = solver.forward()


print("u_full shape:", u_full.shape)
plot_velocity_model(space_model.velocity_model)
plot_wavefield(u_full[-1])
plot_shotrecord(recv)
