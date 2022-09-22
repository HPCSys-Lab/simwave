from simwave import SpaceModel, TimeModel, RickerWavelet, Solver
from simwave import Receiver, Source, Compiler
import numpy as np


def generate_one(dimension, space_order):

    if dimension == 2:
        shape = (500,)*dimension
        bbox = (0, 5000, 0, 5000)
        spacing = (10, 10)
        damping_length = 100
        boundary_condition = (
            "null_neumann", "null_dirichlet",
            "none", "null_dirichlet"
        )
        source_position = [(2500, 2500)]
        tf = 1.0
        f0 = 10.0
    else:
        shape = (100,)*dimension
        bbox = (0, 1000, 0, 1000, 0, 1000)
        spacing = (10, 10, 10)
        damping_length = 50
        boundary_condition = (
            "null_neumann", "null_dirichlet",
            "none", "null_dirichlet",
            "null_neumann", "null_dirichlet",
        )
        source_position = [(500, 495, 505)]
        tf = 0.4
        f0 = 15.0

    result_file = "wavefield-{}d-so-{}".format(dimension, space_order)

    # Velocity model
    vel = np.zeros(shape=shape, dtype=np.float32)
    vel[:] = 1500.0
    vel[shape[0]//2:] = 2000.0

    # create the space model
    space_model = SpaceModel(
        bounding_box=bbox,
        grid_spacing=spacing,
        velocity_model=vel,
        space_order=space_order,
        dtype=np.float32
    )

    # config boundary conditions
    space_model.config_boundary(
        damping_length=damping_length,
        boundary_condition=boundary_condition,
        damping_polynomial_degree=3,
        damping_alpha=0.001
    )

    # create the time model
    time_model = TimeModel(
        space_model=space_model,
        tf=tf,
        saving_stride=0
    )

    # create the set of sources
    source = Source(
        space_model=space_model,
        coordinates=source_position,
        window_radius=1
    )

    # crete the set of receivers
    receiver = Receiver(
        space_model=space_model,
        coordinates=source_position,
        window_radius=1
    )

    # create a ricker wavelet with 10hz of peak frequency
    ricker = RickerWavelet(f0, time_model)

    # create a compiler
    compiler = Compiler(language='c')

    # create the solver
    solver = Solver(
        compiler=compiler,
        space_model=space_model,
        time_model=time_model,
        sources=source,
        receivers=receiver,
        wavelet=ricker
    )

    # run the forward
    u, recv = solver.forward()

    np.save(result_file, u)


def generate_all():

    dimension = [2, 3]
    space_order = [2, 8]

    for dim in dimension:
        for so in space_order:
            generate_one(dim, so)


if __name__ == "__main__":
    generate_all()
