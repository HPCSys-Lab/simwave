"""Converge test for constant density acoustic wave equation

VERBOSE OPTIONS

--mode
        'single': runs a single time and plots comparison at a point
        'time': test order-accuracy in time
        'full': displays absolute and relative difference between wavefield and interpolated receiver values on the whole domain
--data
        'default': same as in devito's accuracy jupyter-notebook, but with scaling usually adopted in the simwave examples
        'devito_scale': same as in devito's accuracy jupyter-notebook, including the scaling (kHz, domain in meters but velocity in km/s, etc)
        'test_params': same as in test_convergence.py pytest file, which currently passes (for a point source)
--signal
        'cosine': source time dependency as cos(w*t)
        'squared_sine': source time dependency as sin(w*t) ** 2
        'keith_cosine': source time dependency as cos(w*t) * cos(w*(t+dt))

--------------------------------------------------------------------------

Consider the acoustic wave equation with constant density

1/c**2 * d^2u/dt^2 - div(grad(u)) = f,                               (1)

Assuming a solution of the type

u(z, x, t) = sin(kz * z) * sin(kx * x) * cos(w * t),                 (2)

where kz = pi / Lz, kx = pi / Lx, and w = 2*pi*f. Lx is the width of the
domain, Lz is its depth, and 'f' is some arbitrary frequency. This choice
implies that u(z, k, t) = 0 at the boundaries. It also implies an initial
condition of u(z, x, 0) = sin(kz * z) * sin(kx * x), and initial 'velocity'
du/dt(z, x, 0) = 0.

In particular, the source whose solution to (1) results in (2) is:

f(z, x, t) = (kz**2 + kx**2 - (w/c)**2) * u(z, x, t),                (3)

"""


import sys

import matplotlib.pyplot as plt
import numpy as np

from simwave import (Compiler, Receiver, RickerWavelet, Solver, Source,
                     SpaceModel, TimeModel, Wavelet, plot_shotrecord,
                     plot_velocity_model, plot_wavefield)

# from sympy import cos, diff, simplify, sin, symbols

#######################################################################
######################  CLASSES AND FUNCTIONS  ########################
#######################################################################


class MySource(Source):
    @property
    def interpolated_points_and_values(self):
        """Calculate the amplitude value at all points"""

        points = np.array([], dtype=np.uint)
        values = np.array([], dtype=self.space_model.dtype)
        offset = np.array([0], dtype=np.uint)

        nbl_pad_width = self.space_model.nbl_pad_width
        halo_pad_width = self.space_model.halo_pad_width
        dimension = self.space_model.dimension
        bbox = self.space_model.bounding_box

        # senoid window
        for dim, dim_length in enumerate(self.space_model.shape):
        # for dim, dim_length in enumerate(self.space_model.extended_shape):

            # points
            lpad = halo_pad_width[dim][0] + nbl_pad_width[dim][0]
            # lpad = 0
            rpad = halo_pad_width[dim][1] + nbl_pad_width[dim][1] + dim_length - 1
            # rpad = dim_length
            points = np.append(points, np.array([lpad, rpad], dtype=np.uint))
            # values
            coor_min, coor_max = bbox[dim * dimension : 2 + dim * dimension]
            coordinates = np.linspace(coor_min, coor_max, dim_length)
            amplitudes = np.sin(np.pi / (coor_max - coor_min) * coordinates)
            values = np.append(values, amplitudes.astype(values.dtype))
        # offset
        offset = np.append(offset, np.array([values.size], dtype=np.uint))

        return points, values, offset


def cosenoid(time_model, amplitude, omega=2 * np.pi):
    """Function that generates the signal satisfying s(0) = ds/dt(0) = 0"""
    return amplitude * np.cos(omega * time_model.time_values)


def squared_senoid(time_model, kx=None, kz=None, vp=None, omega=None):
    """Function that generates the signal satisfying s(0) = ds/dt(0) = 0"""
    return (kx ** 2 + kz ** 2) * np.sin(omega * time_model.time_values) ** 2 + 2 * (
        omega / vp
    ) ** 2 * np.cos(2 * omega * time_model.time_values)


def keith_cosenoid(time_model, kx=None, kz=None, vp=None, omega=None):
    """Function that generates the signal satisfying s(0) = 0 and s(-dt) = 0"""
    dt = time_model.dt
    t_ = time_model.time_values
    return (
        vp ** 2 * (kx ** 2 + kz ** 2) * np.sin(omega * t_) * np.sin(omega * (dt + t_))
        + 2 * omega ** 2 * np.cos(omega * (dt + 2 * t_))
    ) / vp ** 2


def initial_shape(space_model, kx, kz):
    nbl_pad_width = space_model.nbl_pad_width
    halo_pad_width = space_model.halo_pad_width
    bbox = space_model.bounding_box
    # values
    xmin, xmax = bbox[2:4]
    zmin, zmax = bbox[0:2]
    x_coords = np.linspace(xmin, xmax, space_model.shape[1])
    z_coords = np.linspace(zmin, zmax, space_model.shape[0])
    Z, X = np.meshgrid(z_coords, x_coords)

    return np.sin(kx * X) * np.sin(kz * Z)


def ansatz(Lx, Lz, h, omega, tp, dt=None, signal=None):
    # Params
    kz = np.pi / Lx
    kx = np.pi / Lz
    nz = int((Lz) / h + 1)
    nx = int((Lx) / h + 1)
    zcoords = np.linspace(0, Lz, nz)
    xcoords = np.linspace(0, Lx, nx)
    meshgrid = np.meshgrid(zcoords, xcoords)
    # Solution
    space_solution = np.sin(kx * meshgrid[0]) * np.sin(kz * meshgrid[1])
    if signal in ["cosine"]:
        time_solution = np.cos(omega * tp)
    elif signal in ["squared_sine"]:
        time_solution = np.sin(omega * tp) ** 2
    elif signal in ["keith_cosine"]:
        time_solution = np.sin(omega * tp) * np.sin(omega * (tp + dt))
    return (space_solution[:, :, None] * time_solution).T


def create_models(Lz, Lx, spacing, order, vel, dens, tf, dt, stride):

    # create the space model
    space_model = SpaceModel(
        bounding_box=(0, Lz, 0, Lx),
        grid_spacing=(spacing, spacing),
        velocity_model=vel,
        density_model=dens,
        space_order=order,
        dtype=np.float64,
    )

    # config boundary conditions
    # (none,  null_dirichlet or null_neumann)
    space_model.config_boundary(
        boundary_condition=(
            "null_dirichlet",
            "null_dirichlet",
            "null_dirichlet",
            "null_dirichlet",
        ),
    )

    # create the time model
    time_model = TimeModel(space_model=space_model, tf=tf, dt=dt, saving_stride=stride)
    print(f"dt size = {time_model.dt}")

    return space_model, time_model


def acquisition_geometry(
    space_model, time_model, src, rec, vp, kz, kx, omega, signal=None
):
    # check for rec
    if rec is None:
        rec = [
            (260, i * space_model.grid_spacing[1]) for i in range(space_model.shape[1])
        ]

    # create the set of sources
    my_source = MySource(space_model, coordinates=[])

    # crete the set of receivers
    receiver = Receiver(space_model=space_model, coordinates=rec, window_radius=4)

    # create a cosine wavelet with 90hz of linear frequency
    if signal in ["cosine"]:
        amp_no_density = 1 * ((kx ** 2 + kz ** 2) - omega ** 2 / vp ** 2)
        wavelet = Wavelet(
            cosenoid, time_model=time_model, amplitude=amp_no_density, omega=omega
        )
    elif signal in ["squared_sine"]:
        wavelet = Wavelet(
            squared_senoid, time_model=time_model, kx=kx, kz=kz, vp=vp, omega=omega
        )
    elif signal in ["keith_cosine"]:
        wavelet = Wavelet(
            keith_cosenoid, time_model=time_model, kx=kx, kz=kz, vp=vp, omega=omega
        )

    return my_source, receiver, wavelet


def run(space_model, time_model, source, receiver, wavelet, omega, init_grid=False):

    # create the solver
    solver = Solver(
        space_model=space_model,
        time_model=time_model,
        sources=source,
        receivers=receiver,
        wavelet=wavelet,
        compiler=compiler,
    )

    print("Timesteps:", time_model.timesteps)

    # initial grid
    if init_grid:
        if signal in ["cosine"]:
            init_shape = initial_shape(space_model, kx, kz)
            initial_grid = [
                init_shape * np.cos(-1 * time_model.dt * omega),
                init_shape * np.cos(-0 * time_model.dt * omega),
            ]
        elif signal in ["squared_sine"]:
            init_shape = initial_shape(space_model, kx, kz)
            initial_grid = [
                init_shape * np.sin(-1 * time_model.dt * omega) ** 2,
                init_shape * np.sin(-0 * time_model.dt * omega) ** 2,
            ]
        elif signal in ["keith_cosine"]:
            initial_grid = None

    else:
        initial_grid = None

    # run the forward
    u_full, recv = solver.forward(initial_grid)

    return u_full, recv


def plot_at_point(u_num, u_ana, time_values, label):
    plt.plot(time_values, u_ana, time_values, u_num, label=label)
    plt.legend(["analytical", "numerical"])
    plt.title("Comparison between analytical and numerical result (simwave)")
    plt.xlabel("time")


def convergence_time(
    Lz, Lx, base_h, base_p, vel, tf, dts, stride, src, rec, vp, kz, kx, omega, signal
):
    error = []
    for i, dt_ in enumerate(dts):
        space_model, time_model = create_models(
            Lz, Lx, base_h, base_p, vel, None, tf, dt_, stride=stride
        )
        source, receiver, wavelet = acquisition_geometry(
            space_model, time_model, src, rec, vp, kz, kx, omega, signal
        )
        u_full, recv = run(
            space_model, time_model, source, receiver, wavelet, omega, init_grid=True
        )
        if signal in ["cosine"]:
            analyticv = (
                np.sin(kz * rec[0][0])
                * np.sin(kx * rec[0][1])
                * np.cos(omega * time_model.time_values)
            )
        elif signal in ["squared_sine"]:
            analyticv = (
                np.sin(kz * rec[0][0])
                * np.sin(kx * rec[0][1])
                * np.sin(omega * time_model.time_values) ** 2
            )
        elif signal in ["keith_cosine"]:
            print("in here")
            analyticv = (
                np.sin(kz * rec[0][0])
                * np.sin(kx * rec[0][1])
                * np.sin(omega * time_model.time_values[::stride])
                * np.sin(omega * (time_model.time_values[::stride] - dt_))
            )

        idx_z = int(receiver.grid_positions[0][0])
        idx_x = int(receiver.grid_positions[0][1])
        u_recv = u_full[:, idx_z, idx_x]

        print(u_recv.shape, analyticv.shape)

        u_recv = np.asarray(u_recv.flatten())
        tmp = np.mean(u_recv - analyticv)
        print(tmp)
        # u_recv -= tmp
        plot_at_point(
            u_recv - analyticv,
            u_recv - analyticv,
            np.arange(time_model.time_values[::stride].size),
            label="error",
        )
        plt.legend()
        plt.show()

        error.append(
            np.linalg.norm(u_recv - analyticv, 2)
            / np.sqrt(time_model.time_values[::stride].size)
        )
        # error.append(np.linalg.norm(u_recv.flatten() - analyticv, 2) / np.sqrt(time_model.time_values.size))
        if i > 0:
            ratio_dt = dts[i - 1] / dts[i]
            ratio_e = error[-2] / error[-1]
            print(error)
            print(
                f"for dt = {dt_}, error={error[-1]}, squared dt ratio = {ratio_dt**2}, error ratio = {ratio_e}"
            )


def run_and_compare_all(
    Lz, Lx, h, p, vel, dens, tf, dt, stride, src, rec, vp, kz, kx, omega, signal
):
    space_model, time_model = create_models(
        Lz, Lx, h, p, vel, dens, tf, dt, stride=stride
    )
    source, receiver, wavelet = acquisition_geometry(
        space_model, time_model, src, rec, vp, kz, kx, omega, signal
    )
    u_full, recv = run(
        space_model, time_model, source, receiver, wavelet, omega, init_grid=True
    )
    u_reshaped = u_full.reshape((time_model.timesteps, receiver.count), order="C")
    u_interp_error = u_reshaped - recv
    delta_error = np.linalg.norm(u_interp_error) / np.linalg.norm(recv)

    print(
        f"Norm of the difference between 'u_full' and 'recv' is {np.linalg.norm(u_interp_error)}"
    )
    print(f"Relative difference of {delta_error * 100} %")

    return u_interp_error


if __name__ == "__main__":

    compiler = Compiler(
        cc="gcc",
        language="c",
        cflags="-O2 -fPIC -Wall -std=c99 -shared",
    )

    dts = [0.0005, 0.0005 / 2]

    # Velocity model
    vp = 1500.0
    vel = vp * np.ones(shape=(513, 513), dtype=np.float64)

    tf = 1 * 0.10
    base_dt = 0.0005
    freq = 10
    stride = 2

    Lz, Lx = 440, 440
    base_h = 2
    base_p = 16

    src = [(200, 200)]
    rec = [(260, 260)]

    kz, kx = np.pi / Lz, np.pi / Lx
    omega = 2 * np.pi * freq

    signal = "keith_cosine"

    # Lz, Lx, base_h, base_p, vel, tf, dts, stride, src, rec, vp, kz, kx, omega, signal
    convergence_time(
        Lz,
        Lx,
        base_h,
        base_p,
        vel,
        tf,
        dts,
        stride,
        src,
        rec,
        vp,
        kz,
        kx,
        omega,
        signal,
    )
