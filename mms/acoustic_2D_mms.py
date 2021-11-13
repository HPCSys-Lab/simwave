import numpy as np
import matplotlib.pyplot as plt

from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Wavelet, Solver, Compiler,
    Receiver, Source, plot_wavefield, plot_shotrecord, plot_velocity_model
)


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
   

class MySource(Source):
    
    @property
    def interpolated_points_and_values(self):
        """ Calculate the amplitude value at all points"""

        points = np.array([], dtype=np.uint)
        values = np.array([], dtype=self.space_model.dtype)

        nbl_pad_width = self.space_model.nbl_pad_width
        halo_pad_width = self.space_model.halo_pad_width
        dimension = self.space_model.dimension
        bbox = self.space_model.bounding_box

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
            offset = np.array([0, values.size], dtype=np.uint)
    
        return points, values, offset

def cosenoid(time_model, amplitude, omega=2*np.pi*1):
    """Function that generates the signal satisfying s(0) = ds/dt(0) = 0"""
    return amplitude * np.cos(omega * time_model.time_values)

def create_models(Lx, Lz, vel, tf, h, p):
    # create the space model
    space_model = SpaceModel(
        bounding_box=(0, Lx, 0, Lz),
        grid_spacing=(h, h),
        velocity_model=vel,
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
        saving_stride=1
    )

    return space_model, time_model

def geometry_acquisition(space_model, sources=None, receivers=None):
    if sources is None:
        sources=[(2560, 2560)]
    if receivers is None:
        receivers=[(2560, i) for i in range(0, 5120, 10)]

    # create the set of sources
    # source = Source(
    #     space_model,
    #     coordinates=sources,
    #     window_radius=4
    # )

    # personalized source
    my_source = MySource(space_model, coordinates=[])

    # crete the set of receivers
    receiver = Receiver(
        space_model=space_model,
        coordinates=receivers,
        window_radius=4
    )

    return my_source, receiver

def build_solver(space_model, time_model, freq, vp, kx, kz, omega):

    # geometry acquisition
    source, receiver = geometry_acquisition(space_model)

    # create a ricker wavelet with 10hz of peak frequency
    # ricker = RickerWavelet(freq, time_model)

    # create a cosenoid wavelet
    amplitude = (kx**2+kz**2) - omega**2/vp**2
    # amplitude = 0
    wavelet = Wavelet(cosenoid, time_model=time_model, amplitude=amplitude, omega=omega)

    # set compiler options
    compiler = Compiler( cc='gcc', language='cpu_openmp', cflags='-O3 -fPIC -ffast-math -Wall -std=c99 -shared')

    # create the solver
    solver = Solver(
        space_model=space_model,
        time_model=time_model,
        sources=source,
        receivers=receiver,
        # wavelet=ricker,
        wavelet=wavelet,
        compiler=compiler
    )

    return solver


if __name__ == "__main__":
    
    # problem params
    Lx, Lz = 5120, 5120
    tf = 1.5
    freq = 5
    vp = 1500

    # discretization params
    n = 513    
    h = 10
    p = 4

    # harmonic parameters
    kx = np.pi / Lx
    kz = np.pi / Lz
    omega = 2*np.pi*freq

    # velocity model
    vel = vp * np.ones(shape=(n, n), dtype=np.float32)

    # discretization
    space_model, time_model = create_models(Lx, Lz, vel, tf, h, p)

    # initial grid
    # initial_grid = np.zeros(shape=space_model.shape)
    initial_grid = initial_shape(space_model, kx, kz)

    # get solver
    solver = build_solver(space_model, time_model, freq, vp, kx, kz, omega)

    # run the forward
    u_full, recv = solver.forward(
        [initial_grid * np.cos(-1 * time_model.dt * omega),
         initial_grid * np.cos(-0 * time_model.dt * omega)
        ]
    )

    # plot stuff
    print("u_full shape:", u_full.shape)
    plot_wavefield(u_full[-1])
    plot_shotrecord(recv)

    # save numpy array
    # print("saving numpy array")
    # np.save("numerical_harmonic", u_full)    

    # load analytical array
    u_analytic = np.load("analytical_harmonic.npy")
    # plot comparison
    numerical_values = u_full[:, 256, 256]
    analytic_values = u_analytic[:, 256, 256]
    time_values = time_model.time_values
    plt.plot(time_values, analytic_values, time_values, numerical_values)
    plt.legend(["analytical", "numerical"])
    plt.title("Comparison between analytical and numerical result (simwave)")
    plt.xlabel("time")
    plt.show()


