from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver,
    Compiler, Receiver, Source
)
import numpy as np
import h5py
import sys
import argparse


def get_args(args=sys.argv[1:]):

    parser = argparse.ArgumentParser(description='How to use this program')
    parser.add_argument("--model_file", type=str,
                        help="Path to the velocity model file")
    parser.add_argument("--language", type=str, default="c",
                        help="Language: c, cpu_openmp, gpu_openmp")
    parser.add_argument("--space_order", type=int, default=2,
                        help="Space order")
    parser.add_argument("--stride", type=int, default=0,
                        help="Saving stride")
    parsed_args = parser.parse_args(args)

    return parsed_args


def read_model(filename):
    with h5py.File(filename, "r") as f:

        # Get the data
        data = list(f['m'])

        data = np.array(data)

        # convert to m/s
        data = (1 / (data ** (1 / 2))) * 1000.0

        return data


if __name__ == '__main__':

    args = get_args()

    data = read_model(args.model_file)

    print("Overthrust original shape:", data.shape)

    language = args.language

    if language == 'c' or language == 'cpu_openmp':
        cc = 'gcc'
        cflags = '-O3 -fPIC -ffast-math -std=c99'
    elif language == 'gpu_openmp':
        cc = 'clang'
        cflags = '-O3 -fPIC -ffast-math -fopenmp \
               -fopenmp-targets=nvptx64-nvidia-cuda \
               -Xopenmp-target -march=sm_75'
    else:
        raise ValueError('Language not available')

    compiler = Compiler(
        cc=cc,
        language=language,
        cflags=cflags
    )

    space_model = SpaceModel(
        bounding_box=(0, 4120, 0, 16000, 0, 16000),
        grid_spacing=(20., 20., 20.),
        velocity_model=data,
        space_order=args.space_order,
        dtype=np.float64
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
        tf=4,
        saving_stride=args.stride
    )

    source = Source(
        space_model,
        coordinates=[(20, 8000, 8000)],
        window_radius=1
    )

    receiver = Receiver(
        space_model=space_model,
        coordinates=[(20, 8000, i) for i in range(0, 16000, 20)],
        window_radius=1
    )

    ricker = RickerWavelet(8.0, time_model)

    solver = Solver(
        space_model=space_model,
        time_model=time_model,
        sources=source,
        receivers=receiver,
        wavelet=ricker,
        compiler=compiler
    )

    # run the forward
    u_full, recv = solver.forward()

    print("Timesteps:", time_model.timesteps)
    print("u_full shape:", u_full.shape)
    print("Receivers:", receiver.count)
