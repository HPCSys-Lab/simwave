from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler,
    Receiver, Source, read_2D_segy,
    plot_wavefield, plot_shotrecord, plot_velocity_model
)
import numpy as np
import sys
import argparse


'''
Use the Marmousi 2 model available in
https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz
http://mcee.ou.edu/aaspi/publications/2006/martin_etal_TLE2006.pdf
'''


def get_args(args=sys.argv[1:]):

    parser = argparse.ArgumentParser(description='How to use this program')
    parser.add_argument("--model_file", type=str,
                        help="Path to the velocity model file")
    parser.add_argument("--language", type=str, default="c",
                        help="Language: c, cpu_openmp, gpu_openmp, cuda")
    parser.add_argument("--space_order", type=int, default=2,
                        help="Space order")
    parser.add_argument("--stride", type=int, default=0,
                        help="Saving stride")
    parsed_args = parser.parse_args(args)

    return parsed_args


if __name__ == '__main__':

    args = get_args()

    # Velocity model
    # 3.5 km depth and 17 km width
    marmousi_model = read_2D_segy(args.model_file)

    language = args.language

    if language == 'c' or language == 'cpu_openmp':
        cc = 'gcc'
        cflags = '-O3 -fPIC -ffast-math -std=c99'
    elif language == 'gpu_openmp':
        cc = 'clang'
        cflags = '-O3 -fPIC -ffast-math -fopenmp \
               -fopenmp-targets=nvptx64-nvidia-cuda \
               -Xopenmp-target -march=sm_75'
    elif language == 'cuda':
        cc = 'nvcc'
        cflags = '-O3 -gencode arch=compute_75,code=sm_75 \
                  --compiler-options -fPIC,-Wall \
                  --use_fast_math -std=c++17 -shared \
                  -DDEBUG -DTX=32 -DTY=4 -DTZ=2'
    else:
        raise ValueError('Language not available')

    compiler = Compiler(
        cc=cc,
        language=language,
        cflags=cflags
    )

    # create the space model
    space_model = SpaceModel(
        bounding_box=(0, 3500, 0, 17000),
        grid_spacing=(10.0, 10.0),
        velocity_model=marmousi_model,
        space_order=args.space_order,
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
        tf=2,
        saving_stride=0
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
        compiler=compiler
    )

    # run the forward
    u_full, recv = solver.forward()

    print("u_full shape:", u_full.shape)
    print("timesteps:", time_model.timesteps)

    # remove damping extension from u_full
    u_full = space_model.remove_nbl(u_full)

    extent = [0, 17000, 3500, 0]

    plot_velocity_model(space_model.velocity_model, extent=extent)
    plot_wavefield(u_full[-1], extent=extent)
    plot_shotrecord(recv)
