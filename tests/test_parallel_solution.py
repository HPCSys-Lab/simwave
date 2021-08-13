from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler,
    Receiver, Source
)
import numpy as np
import pytest


def run_forward(dimension, density, language, dtype):

    if dimension == 2:
        shape = (128, 128)
        bounding_box = (0, 1280, 0, 1280)
        grid_spacing = (10., 10.)
        boundary_condition = (
            "null_neumann", "null_dirichlet",
            "null_neumann", "null_dirichlet"
        )
        source_coordinates = [(10, i) for i in range(128, 1280, 128)]
        receiver_coordinates = [(10, i) for i in range(0, 1280, 10)]
        damping_length = 128
    else:
        shape = (128, 128, 128)
        bounding_box = (0, 1280, 0, 1280, 0, 1280)
        grid_spacing = (10., 10., 10.)
        boundary_condition = (
            "null_neumann", "null_dirichlet",
            "null_neumann", "null_dirichlet",
            "null_neumann", "null_dirichlet"
        )
        source_coordinates = [(10, 640, i) for i in range(128, 1280, 128)]
        receiver_coordinates = [(10, 650, i) for i in range(0, 1280, 10)]
        damping_length = 128

    if density:
        density = np.zeros(shape=shape, dtype=dtype)
        density[:] = 5
    else:
        density = None

    vel = np.zeros(shape=shape, dtype=dtype)
    vel[:] = 1500.0

    if language == 'gpu_openmp':
        cflags = '-O3 -fPIC -ffast-math -fopenmp \
        -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_75'

        cc = 'clang'
    else:
        cflags = '-O3'
        cc = 'gcc'

    compiler = Compiler(
        cc=cc,
        language=language,
        cflags=cflags
    )

    space_model = SpaceModel(
        bounding_box=bounding_box,
        grid_spacing=grid_spacing,
        velocity_model=vel,
        density_model=density,
        space_order=4,
        dtype=dtype
    )

    space_model.config_boundary(
        damping_length=damping_length,
        boundary_condition=boundary_condition
    )

    time_model = TimeModel(
        space_model=space_model,
        tf=0.4,
        saving_stride=0
    )

    source = Source(
        space_model,
        coordinates=source_coordinates,
        window_radius=8
    )

    receiver = Receiver(
        space_model=space_model,
        coordinates=receiver_coordinates,
        window_radius=8
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

    u_full, recv = solver.forward()

    return u_full[-1], recv


class TestParallelSolutionCPU:

    @pytest.mark.parametrize(
        'dimension, density, language, dtype', [
            (2, False, 'cpu_openmp', np.float64),
            (3, False, 'cpu_openmp', np.float64),
            (2, True, 'cpu_openmp', np.float64),
            (3, True, 'cpu_openmp', np.float64)
        ]

    )
    def test_parallel_cpu(self, dimension, density, language, dtype):

        # baseline result
        u_base, rec_base = run_forward(
            dimension=dimension,
            density=density,
            language='c',
            dtype=dtype
        )

        u, rec = run_forward(
            dimension=dimension,
            density=density,
            language=language,
            dtype=dtype
        )

        # assert np.array_equal(u_base, u)
        # assert np.array_equal(rec_base, rec)

        assert np.allclose(u_base, u, atol=1e-08)
        assert np.allclose(rec_base, rec, atol=1e-08)


@pytest.mark.gpu
class TestParallelSolutionGPU:

    @pytest.mark.parametrize(
        'dimension, density, language, dtype', [
            (2, False, 'gpu_openmp', np.float64),
            (3, False, 'gpu_openmp', np.float64),
            (2, True, 'gpu_openmp', np.float64),
            (3, True, 'gpu_openmp', np.float64)
        ]

    )
    def test_parallel_gpu(self, dimension, density, language, dtype):

        # baseline result
        u_base, rec_base = run_forward(
            dimension=dimension,
            density=density,
            language='c',
            dtype=dtype
        )

        u, rec = run_forward(
            dimension=dimension,
            density=density,
            language=language,
            dtype=dtype
        )

        assert np.allclose(u_base, u, atol=1e-08)
        assert np.allclose(rec_base, rec, atol=1e-08)
