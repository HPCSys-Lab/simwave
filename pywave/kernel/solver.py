import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import time
from pywave.kernel import cfl

class Solver():

    """
    Base class to implement the solver

    Parameters
    ----------
    grid : object
        Object that represents the grid
    velocity_model: object
        Object that represents the velocity model
    density: object
        Object that represents the density model
    compiler: object
        Object that represents the compiler
    spacing: tuple
        Spacing along each axis
    progatation_time: int
        Propagation time in miliseconds
    print_steps: int
        Print intermediate wavefields according
    """
    def __init__(self, grid=None, velocity_model=None, density=None,
                       compiler=None, spacing=None, progatation_time=1000,
                       print_steps=0):

        self.grid = grid
        self.velocity_model = velocity_model
        self.density  = density
        self.compiler = compiler
        self.spacing = spacing
        self.print_steps = print_steps
        self.progatation_time = progatation_time

        self.dimension = len(self.grid.shape())
        self.space_order = 2

        # validate dimensions
        self.__validate_dimensions()

        # load the lib
        self.__load_lib()

    def __validate_dimensions(self):

        if self.density is None:
            if self.grid.shape() != self.velocity_model.shape():
                raise Exception("Grid and Velocity Model must have the same dimensions")
        else:
            if (self.grid.shape() != self.velocity_model.shape()) or (self.grid.shape() != self.density.shape()):
               raise Exception("Grid, Velocity Model and Density Model must have the same dimensions")

        if self.dimension != len(self.spacing):
            raise Exception("Spacing must have {} values".format(self.dimension))

    def __load_lib(self):
        # compile the code
        shared_object = self.compiler.compile()

        # load the library
        self.library = ctypes.cdll.LoadLibrary(shared_object)

    def _print_params(self):
        print("Dimension: %dD" % self.dimension)
        print("Shape:", self.grid.shape())
        print("Spacing:", self.spacing)
        print("Density:", ("constant" if self.density is None else "variable") )
        print("Space Order:", self.space_order)
        print("Propagation time: %d miliseconds " % self.progatation_time)
        print("DT: %f seconds" % self.dt)
        print("Timesteps:", self.timesteps)

class AcousticSolver(Solver):

    # apply CFL conditions
    def __apply_cfl_conditions(self):

        self.dt = cfl.calc_dt(
            dimension = self.dimension,
            space_order = self.space_order,
            spacing = self.spacing,
            vel_model = self.velocity_model.model
        )

        self.timesteps = cfl.calc_num_timesteps(self.progatation_time, self.dt)

    def __print_params(self):
        print("Model: Acoustic")
        super(AcousticSolver, self)._print_params()

    def __forward_2D_constant_density(self):

        self.forward = self.library.forward_2D_constant_density

        self.forward.restype = ctypes.c_double

        self.forward.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]

        nz, nx = self.grid.shape()
        dz, dx = self.spacing

        self.elapsed_time = self.forward(
            self.grid.wavefield,
            self.velocity_model.model,
            self.source,
            nz,
            nx,
            self.timesteps,
            dz,
            dx,
            self.dt,
            self.print_steps
        )

    def __forward_2D_variable_density(self):

        self.forward = self.library.forward_2D_variable_density

        self.forward.restype = ctypes.c_double

        self.forward.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]

        nz, nx = self.grid.shape()
        dz, dx = self.spacing

        self.elapsed_time = self.forward(
            self.grid.wavefield,
            self.velocity_model.model,
            self.density.model,
            nz,
            nx,
            self.timesteps,
            dz,
            dx,
            self.dt,
            self.print_steps
        )

    def __forward_3D_constant_density(self):

        self.forward = self.library.forward_3D_constant_density

        self.forward.restype = ctypes.c_double

        self.forward.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]

        nz, nx, ny = self.grid.shape()
        dz, dx, dy = self.spacing

        self.elapsed_time = self.forward(
            self.grid.wavefield,
            self.velocity_model.model,
            nz,
            nx,
            ny,
            self.timesteps,
            dz,
            dx,
            dy,
            self.dt,
            self.print_steps
        )

    def __forward_3D_variable_density(self):

        self.forward = self.library.forward_3D_variable_density

        self.forward.restype = ctypes.c_double

        self.forward.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]

        nz, nx, ny = self.grid.shape()
        dz, dx, dy = self.spacing

        self.elapsed_time = self.forward(
            self.grid.wavefield,
            self.velocity_model.model,
            self.density.model,
            nz,
            nx,
            ny,
            self.timesteps,
            dz,
            dx,
            dy,
            self.dt,
            self.print_steps
        )

    def forward(self):

        self.__apply_cfl_conditions()

        self.__print_params()

        # **************************
        sigma = 0.011
        t0 = self.dt
        t = t0

        self.source = np.zeros(self.timesteps, dtype=np.float32)

        for i in range(self.timesteps):
            f = (1.0 - ((t-t0)**2)/(sigma**2)) * np.exp(-((t-t0)**2)/(2*(sigma**2))) / (np.sqrt(2 * np.pi) * (sigma**3))
            t += self.dt
            self.source[i] = f
        # **************************

        print("Computing forward...")

        if self.dimension == 2:
            # constant density
            if self.density is None:
                self.__forward_2D_constant_density()
            else:
                self.__forward_2D_variable_density()
        elif self.dimension == 3:
            # constant density
            if self.density is None:
                self.__forward_3D_constant_density()
            else:
                self.__forward_3D_variable_density()
        else:
            raise Exception("Grid dimension {} not supported".format(self.dimension))

        return self.grid.wavefield, self.elapsed_time
