from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler,
    Receiver, Source, read_2D_segy,
    plot_wavefield, plot_shotrecord, plot_velocity_model
)
import numpy as np

'''
Use the Marmousi 2 model available in
https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz
http://mcee.ou.edu/aaspi/publications/2006/martin_etal_TLE2006.pdf
'''

# Velocity model
# 3.5 km depth and 17 km width
marmousi_model = read_2D_segy('MODEL_P-WAVE_VELOCITY_1.25m.segy')

compiler = Compiler(
    cc='gcc',
    language='cpu_openmp',
    cflags='-O3 -fPIC -ffast-math -Wall -std=c99 -shared'
)

# create the space model
space_model = SpaceModel(
    bounding_box=(0, 3500, 0, 17000),
    grid_spacing=(10.0, 10.0),
    velocity_model=marmousi_model,
    space_order=4,
    dtype=np.float64
)

# config boundary conditions
# (none,  null_dirichlet or null_neumann)
space_model.config_boundary(
    damping_length=(0, 700, 700, 700),
    boundary_condition=(
        "null_neumann", "null_dirichlet",
        "null_dirichlet", "null_dirichlet"
    ),
    damping_polynomial_degree=3,
    damping_alpha=0.001
)

# create the time model
time_model = TimeModel(
    space_model=space_model,
    tf=2
)

# create the set of sources
source = Source(
    space_model,
    coordinates=[(20, 8500)],
    window_radius=1
)

# crete the set of receivers
receiver = Receiver(
    space_model=space_model,
    coordinates=[(20, i) for i in range(0, 17000, 10)],
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
    saving_stride=0,
    compiler=compiler
)

# run the forward
u_full, recv = solver.forward()

# remove damping extension from u_full
u_full = space_model.remove_nbl(u_full)

extent = [0, 17000, 3500, 0]

print("u_full shape:", u_full.shape)
print("timesteps:", time_model.timesteps)
plot_velocity_model(space_model.velocity_model, extent=extent)
plot_wavefield(u_full[-1], extent=extent)
plot_shotrecord(recv)
