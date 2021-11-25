from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from simwave import (
    SpaceModel, TimeModel, RickerWavelet, MultiWavelet, Wavelet, Solver, Compiler,
    Receiver, Source, plot_wavefield, plot_shotrecord, plot_velocity_model
)


class MySource(Source):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 1

    @property
    def count(self):
        """Return the number of sources/receives."""
        return self._count
    
    @count.setter
    def count(self, count):
        """Return the number of sources/receives."""
        self._count = count
   
    @property
    def interpolated_points_and_values(self):
        """ Calculate the amplitude value at all points"""

        points = np.array([], dtype=np.uint)
        values = np.array([], dtype=self.space_model.dtype)
        offset = np.array([0], dtype=np.uint)

        nbl_pad_width = self.space_model.nbl_pad_width
        halo_pad_width = self.space_model.halo_pad_width
        dimension = self.space_model.dimension
        bbox = self.space_model.bounding_box

        # senoid window
        for dim, dim_length in enumerate(self.space_model.shape):

            # points
            lpad = halo_pad_width[dim][0] + nbl_pad_width[dim][0]
            rpad = halo_pad_width[dim][1] + nbl_pad_width[dim][1] + dim_length
            points = np.append(points, np.array([lpad, rpad], dtype=np.uint))
            # values
            coor_min, coor_max = bbox[dim * dimension:2 + dim * dimension]
            coordinates = np.linspace(coor_min, coor_max, dim_length)
            amplitudes = np.sin(np.pi / (coor_max - coor_min) * coordinates)
            values = np.append(values, amplitudes.astype(values.dtype))
        # offset
        offset = np.append(offset, np.array([values.size], dtype=np.uint))

        # add window for density term

        # points
        lpad_x = halo_pad_width[0][0] + nbl_pad_width[0][0]
        rpad_x = halo_pad_width[0][1] + nbl_pad_width[0][1] + self.space_model.shape[0]
        points = np.append(points, np.array([lpad_x, rpad_x], dtype=np.uint))

        lpad_z = halo_pad_width[1][0] + nbl_pad_width[1][0]
        rpad_z = halo_pad_width[1][1] + nbl_pad_width[1][1] + self.space_model.shape[1]
        points = np.append(points, np.array([lpad_z, rpad_z], dtype=np.uint))

        xmin, xmax = bbox[0: 2]
        zmin, zmax = bbox[2: 4]
        kx = np.pi / (xmax - xmin)
        kz = np.pi / (zmax - zmin)
        x_coordinates = np.linspace(xmin, xmax, self.space_model.shape[0])
        z_coordinates = np.linspace(zmin, zmax, self.space_model.shape[1])
        X, Z = np.meshgrid(z_coordinates, z_coordinates)

        # (1 / rho) * drho_dz * dphi_dz
        x_amplitudes = np.sin(kx * x_coordinates)

        dens = self.space_model.density_model
        mid_z = dens.shape[-1] // 2
        drho_dz = np.gradient(dens[:, mid_z], z_coordinates).mean()
        z_amplitudes = kz * drho_dz * np.cos(kz * z_coordinates) / dens[:,mid_z]

        values = np.append(values, x_amplitudes.astype(values.dtype))
        values = np.append(values, z_amplitudes.astype(values.dtype))

        offset = np.append(offset, np.array([values.size], dtype=np.uint))

        return points, values, offset

def grid(Lx, Lz, h):
    nz = int((Lz) / h + 1)
    nx = int((Lx) / h + 1)
    zcoords = np.linspace(0, Lz, nz)
    xcoords = np.linspace(0, Lx, nx)
    return np.meshgrid(zcoords, xcoords)

def ansatz(Lx, Lz, h, freq, tp):
    # Params
    kz = np.pi / Lx
    kx = np.pi / Lz
    meshgrid = grid(Lx, Lz, h)
    omega = 2*np.pi*freq
    # Solution
    space_solution = np.sin(kx*meshgrid[0]) * np.sin(kz*meshgrid[1])
    time_solution = np.cos(omega*tp)
    return (space_solution[:,:,None] * time_solution).T


def initial_shape(space_model, kx, kz):
    nbl_pad_width = space_model.nbl_pad_width
    halo_pad_width = space_model.halo_pad_width
    bbox = space_model.bounding_box
    # values
    xmin, xmax = bbox[0: 2]
    zmin, zmax = bbox[2: 4]
    x_coords = np.linspace(xmin, xmax, space_model.shape[0])
    z_coords = np.linspace(zmin, zmax, space_model.shape[1])
    X, Z = np.meshgrid(x_coords, z_coords)

    return np.sin(kx * X) * np.sin(kz * Z)
   
def cosenoid(time_model, amplitude, omega=2*np.pi*1):
    """Function that generates the signal satisfying s(0) = ds/dt(0) = 0"""
    return amplitude * np.cos(omega * time_model.time_values)

def create_models(Lx, Lz, vel, dens, tf, h, p, saving_stride=1):
    # create the space model
    space_model = SpaceModel(
        bounding_box=(0, Lx, 0, Lz),
        grid_spacing=(h, h),
        velocity_model=vel,
        density_model=dens,
        space_order=p,
        dtype=np.float32
    )

    # config boundary conditions
    # (none,  null_dirichlet or null_neumann)
    space_model.config_boundary(
        damping_length=0,
        boundary_condition=(
            "null_dirichlet", "null_dirichlet",
            "null_dirichlet", "null_dirichlet"
        ),
        damping_polynomial_degree=3,
        damping_alpha=0.001
    ) 

    # create the time model
    time_model = TimeModel(
        space_model=space_model,
        tf=tf,
        saving_stride=saving_stride
    )

    return space_model, time_model

def geometry_acquisition(space_model, receivers=None):
    if receivers is None:
        receivers=[(2560, i) for i in range(0, 5120, 10)]

    # personalized source
    my_source = MySource(space_model, coordinates=[])

    # crete the set of receivers
    receiver = Receiver(
        space_model=space_model,
        coordinates=receivers,
        window_radius=4
    )

    return my_source, receiver

def build_solver(space_model, time_model, vp, kx, kz, omega, density):

    # geometry acquisition
    source, receiver = geometry_acquisition(space_model)

    # create a cosenoid wavelet
    amp_no_density = 1 * ((kx**2+kz**2) - omega**2/vp**2)
    amp_with_density = 1 if density else 0
    wavelet1 = Wavelet(cosenoid, time_model=time_model, amplitude=amp_no_density, omega=omega)
    wavelet2 = Wavelet(cosenoid, time_model=time_model, amplitude=amp_with_density, omega=omega)
    wavelets = MultiWavelet(np.vstack((wavelet1.values, wavelet2.values)).T, time_model)

    source.count = wavelets.values.shape[-1]

    # set compiler options
    compiler = Compiler( cc='gcc', language='cpu_openmp', cflags='-O3 -fPIC -ffast-math -Wall -std=c99 -shared')

    # create the solver
    solver = Solver(
        space_model=space_model,
        time_model=time_model,
        sources=source,
        receivers=receiver,
        wavelet=wavelets,
        compiler=compiler
    )

    return solver

def accuracy(Lx, Lz, tf, freq, vp, rho, rho_z, h, p, dt=None, density=True, saving_stride=1):
    """Compare numerical and analytical results"""
    # harmonic parameters
    kx = np.pi / Lx
    kz = np.pi / Lz
    omega = 2*np.pi*freq

    n = Lx // h + 1
    # velocity model
    vel = vp * np.ones(shape=(n, n), dtype=np.float32)
    # density model
    dens = rho * np.ones(shape=(n, n), dtype=np.float32)
    for depth, dens_values in enumerate(dens):
        dens_values[:] = rho + rho_z * depth / dens.shape[0] 

    # discretization
    space_model, time_model = create_models(Lx, Lz, vel, dens, tf, h, p, saving_stride=saving_stride)
    print(f"CFL dt = {time_model.dt}")
    if dt is not None:
        time_model.dt = dt
    
    # initial grid
    initial_grid = initial_shape(space_model, kx, kz)
    # get solver
    solver = build_solver(space_model, time_model, vp, kx, kz, omega, density)

    # run the forward
    u_full, recv = solver.forward(
        [initial_grid * np.cos(-1 * time_model.dt * omega),
         initial_grid * np.cos(-0 * time_model.dt * omega)
        ]
    )
    time_values = solver.time_model.time_values

    return u_full, time_values


if __name__ == "__main__":

    # space order params
    orders = [2, 4, 6, 8]
    orders = [2]
    spacings = [6, 5, 4, 3]
    # spacings = [6, 5, 4]

    # problem params
    Lx_, Lz_ = 5120, 5120
    tf_ = 1  * 1.5
    dt = 0.001
    stride = 1
    freq_ = 5
    vp_ = 1500
    rho_ = 1
    # rho_z_ = 1e4
    rho_z_ = 1
    rho_x_ = 0

    # discretization params
    h_ = 10
    p_ = 2

    # check accuracy
    # num = accuracy(Lx_, Lz_, tf_, freq_, vp_, rho_, rho_z_, h_, p_, density=False)

    accs = {}
    conv_rates = []
    for order in orders:
        accs[order] = {}
        for spacing in spacings:
            print(f"space order = {order}, spacing = {spacing}")
            u_num, t_vals = accuracy(Lx_, Lz_, tf_, freq_, vp_, rho_, rho_z_, spacing, order, dt=dt, density=False, saving_stride=stride)
            init_time = datetime.now()
            np.savez(f"num_sol_p_{order}_h_{spacing}_rho_false", u_num=u_num, t_vals=t_vals)
            total_time = datetime.now() - init_time
            print(f"total time for saving arrays: {total_time.seconds} seconds")

    # np.save("full_accs", accs)


