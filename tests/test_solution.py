from pywave import SpaceModel, TimeModel, RickerWavelet, Solver
from pywave import Receiver, Source
import numpy as np
import pytest
import os


class TestSolution:

    @pytest.mark.parametrize(
        'dimension, space_order', [(2, 2), (2, 8), (3, 2), (3, 8)]
    )
    def test_solution(self, dimension, space_order):

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

        ref_file = os.path.join(
            os.path.dirname(__file__),
            "reference_solution/wavefield-{}d-so-{}.npy".format(
                dimension, space_order
            )
        )

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
            tf=tf
        )

        # create the set of sources
        source = Source(
            space_model=space_model,
            coordinates=source_position
        )

        # crete the set of receivers
        receiver = Receiver(
            space_model=space_model,
            coordinates=source_position
        )

        # create a ricker wavelet with 10hz of peak frequency
        ricker = RickerWavelet(f0, time_model)

        # create the solver
        solver = Solver(
            space_model=space_model,
            time_model=time_model,
            sources=source,
            receivers=receiver,
            wavelet=ricker,
            saving_stride=0
        )

        # run the forward
        u, recv = solver.forward()

        # load the reference result
        u_ref = np.load(ref_file)

        assert np.allclose(u, u_ref, rtol=1e-05, atol=1e-05)
