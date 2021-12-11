import numpy as np
from matplotlib import pyplot as plt

from simwave import (Compiler, Receiver, Solver, Source, SpaceModel, TimeModel,
                     Wavelet, MultiWavelet, plot_wavefield)
import sympy


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

        dim_length_z, dim_length_x = self.space_model.shape

        lpad_z = halo_pad_width[0][0] + nbl_pad_width[0][0]
        rpad_z = halo_pad_width[0][1] + nbl_pad_width[0][1] + dim_length_z - 1
        lpad_x = halo_pad_width[1][0] + nbl_pad_width[1][0]
        rpad_x = halo_pad_width[1][1] + nbl_pad_width[1][1] + dim_length_x - 1

        zcoor_min, zcoor_max = bbox[0:2]
        xcoor_min, xcoor_max = bbox[2:4]
        coord_z = np.linspace(zcoor_min, zcoor_max, dim_length_z)
        coord_x = np.linspace(xcoor_min, xcoor_max, dim_length_x)

        kz = np.pi / (zcoor_max - zcoor_min)
        kx = np.pi / (xcoor_max - xcoor_min)

        Ax, Az, Bx, Bz, Cx, Cz = analytical_solution(kx, kz)

        # import IPython; IPython.embed(); exit()

        # source A
        points = np.append(points, np.array([lpad_z, rpad_z, lpad_x, rpad_x], dtype=np.uint))

        values_z = Az(coord_z)
        values_x = Ax(coord_x)
        amplitudes = np.array([values_z, values_x])
        values = np.append(values, amplitudes.astype(values.dtype))

        offset = np.append(offset, np.array([values.size], dtype=np.uint))

        # source B
        points = np.append(points, np.array([lpad_z, rpad_z, lpad_x, rpad_x], dtype=np.uint))

        values_z = Bz(coord_z)
        values_x = Bx(coord_x)
        amplitudes = np.array([values_z, values_x])
        values = np.append(values, amplitudes.astype(values.dtype))

        offset = np.append(offset, np.array([values.size], dtype=np.uint))

        # source C
        points = np.append(points, np.array([lpad_z, rpad_z, lpad_x, rpad_x], dtype=np.uint))

        values_z = Cz(coord_z)
        values_x = Cx(coord_x)
        amplitudes = np.array([values_z, values_x])
        values = np.append(values, amplitudes.astype(values.dtype))

        offset = np.append(offset, np.array([values.size], dtype=np.uint))

        return points, values, offset


def cosenoid(time_model, kx=None, kz=None, vp=None, omega=None):
    dt = time_model.dt
    t_ = time_model.time_values
    return (
        vp ** 2 * (kx ** 2 + kz ** 2) * np.sin(omega * t_) * np.sin(omega * (dt + t_))
        + 2 * omega ** 2 * np.cos(omega * (dt + 2 * t_))
    ) / vp ** 2

def psi_tt(time_model, vp=None, omega=None):
    dt = time_model.dt
    t_ = time_model.time_values
    return (
        1
        / vp ** 2
        * 2
        * omega ** 2
        * (
            -np.sin(omega * t_) * np.sin(omega * (dt + t_))
            + np.cos(omega * t_) * np.cos(omega * (dt + t_))
        )
    )


def psi(time_model, vp=None, omega=None):
    dt = time_model.dt
    t_ = time_model.time_values
    return - np.sin(omega * t_) * np.sin(omega * (dt + t_))

def analytical_solution(num_kx, num_kz):
    x, z, kx, kz, Lx, Lz = sympy.symbols("x z kx kz Lx Lz")
    c, dt, t, omega = sympy.symbols("c dt t omega")
    Lx, Lz = sympy.pi / kx, sympy.pi / kz

    window_x = ((sympy.tanh(0.5*(x - Lx/10))) / 2 + (-sympy.tanh(0.5*(x - 9*Lx/10))) / 2)
    window_z = ((sympy.tanh(0.5*(z - Lz/10))) / 2 + (-sympy.tanh(0.5*(z - 9*Lz/10))) / 2)

    phi = sympy.sin(kx*x) * window_x
    theta = sympy.sin(kz*z) * window_z
    psi = sympy.sin(omega*t) * sympy.sin(omega*(t + dt))
    u = phi * theta * psi

    phi_xx = sympy.diff(phi, x, 2)
    theta_zz = sympy.diff(theta, z, 2)
    psi_tt = sympy.diff(psi, t, 2)

    num_phi = sympy.lambdify(x, phi.subs(kx, num_kx), "numpy")
    num_phi_xx = sympy.lambdify(x, phi_xx.subs(kx, num_kx), "numpy")
    num_theta = sympy.lambdify(z, theta.subs(kz, num_kz), "numpy")
    num_theta_zz = sympy.lambdify(z, theta_zz.subs(kz, num_kz), "numpy")

    A_x = num_phi
    A_z = num_theta
    B_x = num_phi
    B_z = num_theta_zz
    C_x = num_phi_xx
    C_z = num_theta

    return A_x, A_z, B_x, B_z, C_x, C_z

def create_models(Lz, Lx, spacing, order, vel, dens, tf, dt, stride):

    space_model = SpaceModel(
        bounding_box=(0, Lz, 0, Lx),
        grid_spacing=(spacing, spacing),
        velocity_model=vel,
        density_model=dens,
        space_order=order,
        dtype=np.float64,
    )

    space_model.config_boundary(
        boundary_condition=(
            "null_dirichlet",
            "null_dirichlet",
            "null_dirichlet",
            "null_dirichlet",
        ),
    )

    time_model = TimeModel(space_model=space_model, tf=tf, dt=dt, saving_stride=stride)

    return space_model, time_model


def acquisition_geometry(space_model, time_model, src, rec, vp, kz, kx, omega):
    if rec is None:
        rec = [
            (260, i * space_model.grid_spacing[1]) for i in range(space_model.shape[1])
        ]

    my_source = MySource(space_model, coordinates=[])

    # import IPython; IPython.embed(); exit()

    receiver = Receiver(space_model=space_model, coordinates=rec, window_radius=4)

    wavelet_psi = Wavelet(psi, time_model=time_model, vp=vp, omega=omega)
    wavelet_psi_tt = Wavelet(psi_tt, time_model=time_model, vp=vp, omega=omega)
    wavelet = MultiWavelet(
        np.vstack((wavelet_psi_tt.values, wavelet_psi.values, wavelet_psi.values)).T,
        time_model,
    )
    my_source = MySource(space_model, coordinates=[(200, 200) for _ in range(3)])

    return my_source, receiver, wavelet


def run(space_model, time_model, source, receiver, wavelet, omega):

    solver = Solver(
        space_model=space_model,
        time_model=time_model,
        sources=source,
        receivers=receiver,
        wavelet=wavelet,
        compiler=compiler,
    )

    print("Timesteps:", time_model.timesteps)

    u_full, recv = solver.forward()

    return u_full, recv


def convergence_space(
    lz,
    lx,
    hs,
    order,
    vel,
    tf,
    dt,
    stride,
    src,
    rec,
    vp,
    kz,
    kx,
    omega,
):
    base_h = hs[0]
    space_model, time_model = create_models(
        Lz, Lx, base_h, order, vel, None, tf, dt, stride=stride
    )
    source, receiver, wavelet = acquisition_geometry(
        space_model, time_model, src, rec, vp, kz, kx, omega
    )
    u_full, u_recv = run(space_model, time_model, source, receiver, wavelet, omega)
    # for i, u in enumerate(u_full):
    #    plot_wavefield(u, f"snap_{i}.png", clim=[-1, 1])

    analyticv = (
        np.sin(kz * rec[0][0])
        * np.sin(kx * rec[0][1])
        * np.sin(omega * time_model.time_values[::stride])
        * np.sin(omega * (time_model.time_values[::stride] + dt))
    )

    u_recv = np.array(u_recv.flatten(), np.float64)

    fig, ax = plt.subplots()
    _t = time_model.time_values
    ax.plot(_t, u_recv, "b--", label=f"gridspacing={base_h} m")
    ax.plot(_t, analyticv, "k-", label="analytical")

    error = []
    error.append(
        np.linalg.norm(u_recv - analyticv, 2)
        / np.sqrt(time_model.time_values[::stride].size)
    )
    count = 1
    for h in hs[1:]:
        space_model, time_model = create_models(
            Lz, Lx, h, order, vel, None, tf, dt, stride=stride
        )
        source, receiver, wavelet = acquisition_geometry(
            space_model, time_model, src, rec, vp, kz, kx, omega
        )
        u_full, u_recv = run(space_model, time_model, source, receiver, wavelet, omega)
        u_recv = np.array(u_recv.flatten(), dtype=np.float64)

        ax.plot(_t, u_recv, label=f"gridspacing={h} m")

        analyticv = (
            np.sin(kz * rec[0][0])
            * np.sin(kx * rec[0][1])
            * np.sin(omega * time_model.time_values[::stride])
            * np.sin(omega * (time_model.time_values[::stride] + dt))
        )

        error.append(
            np.linalg.norm(u_recv - analyticv, 2)
            / np.sqrt(time_model.time_values[::stride].size)
        )
        if count > 0:
            ratio_gridspace = hs[count - 1] / hs[count]
            ratio_e = error[-2] / error[-1]
            print(error)
            print(
                f"for h = {h}, error={error[-1]}, gridspace ratio = {ratio_gridspace**order}, error ratio = {ratio_e}"
            )
    plt.legend()
    plt.show()


if __name__ == "__main__":

    compiler = Compiler(
        cc="gcc",
        language="c",
        cflags="-O2 -fPIC -Wall -std=c99 -shared",
    )

    # Velocity model m/s
    vp = 3000.0
    vel = vp * np.ones(shape=(513, 513), dtype=np.float64)

    tf = 1.0
    freq = 10.0
    stride = 1

    Lz, Lx = 440, 440
    """
    dt = 0.00001  # seconds
    hs = [10, 5, 2.5]
    order = 4  # order to test
    """
    dt = 0.0001  # seconds
    hs = [20, 10, 5, 2.5]
    order = 2  # order to test

    src = [(200, 200)]
    rec = [(280, 280)]

    kz, kx = np.pi / Lz, np.pi / Lx
    omega = 2 * np.pi * freq

    convergence_space(
        Lz,
        Lx,
        hs,
        order,
        vel,
        tf,
        dt,
        stride,
        src,
        rec,
        vp,
        kz,
        kx,
        omega,
    )

