from simwave import SpaceModel, TimeModel, RickerWavelet, Solver
from simwave import Receiver, Source, Compiler
import numpy as np
import pytest


def run_forward_2d(density, language, saving_stride):
    compiler = Compiler(
        cc='gcc',
        language=language,
        cflags='-O3 -fPIC -ffast-math -Wall -std=c99 -shared'
    )

    vel = np.zeros(shape=(512, 512), dtype=np.float32)
    vel[:] = 1500.0
    vel[250:] = 3000.0

    if density:
        density = np.zeros(shape=(512, 512), dtype=np.float32)
        density[:] = 5
    else:
        density = None

    # create the space model
    space_model = SpaceModel(
        bounding_box=(0, 5120, 0, 5120),
        grid_spacing=(10, 10),
        velocity_model=vel,
        density_model=density,
        space_order=4,
        dtype=np.float32
    )

    space_model.config_boundary(
        damping_length=0,
        boundary_condition=(
            "null_neumann", "null_dirichlet",
            "none", "null_dirichlet"
        ),
        damping_polynomial_degree=3,
        damping_alpha=0.001
    )

    time_model = TimeModel(
        space_model=space_model,
        tf=1.0,
        saving_stride=saving_stride
    )

    source = Source(
        space_model,
        coordinates=[(2560, 2560)],
        window_radius=4
    )

    receiver = Receiver(
        space_model=space_model,
        coordinates=[(2560, i) for i in range(0, 5120, 10)],
        window_radius=4
    )

    ricker = RickerWavelet(10.0, time_model)

    solver = Solver(
        space_model=space_model,
        time_model=time_model,
        sources=source,
        receivers=receiver,
        wavelet=ricker,
        compiler=compiler
    )

    u_full, _ = solver.forward()

    return u_full[-1]


def run_forward_3d(density, language, saving_stride):
    compiler = Compiler(
        cc='gcc',
        language=language,
        cflags='-O3 -fPIC -ffast-math -Wall -std=c99 -shared'
    )

    vel = np.zeros(shape=(100, 100, 100), dtype=np.float32)
    vel[:] = 1500.0

    if density:
        density = np.zeros(shape=(100, 100, 100), dtype=np.float32)
        density[:] = 5
    else:
        density = None

    # create the space model
    space_model = SpaceModel(
        bounding_box=(0, 1000, 0, 1000, 0, 1000),
        grid_spacing=(10, 10, 10),
        velocity_model=vel,
        density_model=density,
        space_order=4,
        dtype=np.float32
    )

    space_model.config_boundary(
        damping_length=0,
        boundary_condition=(
            "null_neumann", "null_dirichlet",
            "null_dirichlet", "null_dirichlet",
            "null_dirichlet", "null_dirichlet"
        ),
        damping_polynomial_degree=3,
        damping_alpha=0.001
    )

    time_model = TimeModel(
        space_model=space_model,
        tf=0.4,
        saving_stride=saving_stride
    )

    source = Source(
        space_model,
        coordinates=[(500, 500, 500)],
        window_radius=4
    )

    receiver = Receiver(
        space_model=space_model,
        coordinates=[(500, 500, i) for i in range(0, 1000, 10)],
        window_radius=4
    )

    ricker = RickerWavelet(15.0, time_model)

    solver = Solver(
        space_model=space_model,
        time_model=time_model,
        sources=source,
        receivers=receiver,
        wavelet=ricker,
        compiler=compiler
    )

    u_full, _ = solver.forward()

    return u_full[-1]


class TestUSaving:

    @pytest.mark.parametrize(
        'saving_stride, language, density', [
            (0, 'c', False),
            (1, 'c', False),
            (2, 'c', False),
            (5, 'c', False),
            (0, 'c', True),
            (1, 'c', True),
            (2, 'c', True),
            (5, 'c', True)
        ]

    )
    def test_u_saving_2d(self, saving_stride, language, density):

        # baseline result
        u_base = run_forward_2d(
            density=density,
            language='c',
            saving_stride=0
        )

        u_last = run_forward_2d(
            density=density,
            language=language,
            saving_stride=saving_stride
        )

        assert np.array_equal(u_base, u_last)

    @pytest.mark.parametrize(
        'saving_stride, language, density', [
            (0, 'c', False),
            (1, 'c', False),
            (2, 'c', False),
            (3, 'c', False),
            (4, 'c', False),
            (5, 'c', False),
            (0, 'c', True),
            (1, 'c', True),
            (2, 'c', True),
            (3, 'c', True),
            (4, 'c', True),
            (5, 'c', True)
        ]

    )
    def test_u_saving_3d(self, saving_stride, language, density):

        # baseline result
        u_base = run_forward_3d(
            density=density,
            language='c',
            saving_stride=0
        )

        u_last = run_forward_3d(
            density=density,
            language=language,
            saving_stride=saving_stride
        )

        assert np.array_equal(u_base, u_last)
