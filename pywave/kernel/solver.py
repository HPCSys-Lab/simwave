import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import time

class Solver():

    """
    Base class to implement the solver

    Parameters
    ----------
    model: object
        Define a model with execution parameters
    compiler: object
        Object that represents the compiler
    """
    def __init__(self, model=None, compiler=None):

        self.model = model
        self.compiler = compiler

        # load the lib
        self.__load_lib()

    def __load_lib(self):
        # compile the code
        shared_object = self.compiler.compile()

        # load the library
        self.library = ctypes.cdll.LoadLibrary(shared_object)

    def _print_params(self):
        print("Dimension: %dD" % self.model.dimension)
        print("Shape:", self.model.grid.shape())
        print("Spacing:", self.model.spacing)
        print("Density:", ("constant" if self.model.density is None else "variable") )
        print("Space Order:", self.model.space_order)
        print("Propagation time: %d miliseconds " % self.model.progatation_time)
        print("DT: %f seconds" % self.model.dt)
        print("Frequency: %0.1f Hz" % self.model.frequency)
        print("Timesteps:", self.model.timesteps)

class AcousticSolver(Solver):

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
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int
        ]

        nz, nx = self.model.grid.shape()
        dz, dx = self.model.spacing
        origin_z, origin_x = self.model.origin

        self.elapsed_time = self.forward(
            self.model.grid.wavefield,
            self.model.velocity.model,
            self.model.wavelet,
            origin_z,
            origin_x,
            nz,
            nx,
            self.model.timesteps,
            dz,
            dx,
            self.model.dt,
            0
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
            ctypes.c_float,
            ctypes.c_float,
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_int
        ]

        nz, nx = self.model.grid.shape()
        dz, dx = self.model.spacing
        origin_z, origin_x = self.model.origin

        self.elapsed_time = self.forward(
            self.model.grid.wavefield,
            self.model.velocity.model,
            self.model.density.model,
            nz,
            nx,
            dz,
            dx,
            self.model.wavelet,
            origin_z,
            origin_x,
            self.model.timesteps,
            self.model.dt,
            self.model.coeff,
            self.model.space_order,
            0
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

        nz, nx, ny = self.model.grid.shape()
        dz, dx, dy = self.model.spacing

        self.elapsed_time = self.forward(
            self.model.grid.wavefield,
            self.model.velocity.model,
            nz,
            nx,
            ny,
            self.model.timesteps,
            dz,
            dx,
            dy,
            self.model.dt,
            0
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

        nz, nx, ny = self.model.grid.shape()
        dz, dx, dy = self.model.spacing

        self.elapsed_time = self.forward(
            self.model.grid.wavefield,
            self.model.velocity.model,
            self.model.density.model,
            nz,
            nx,
            ny,
            self.model.timesteps,
            dz,
            dx,
            dy,
            self.model.dt,
            0
        )

    def forward(self):

        self.__print_params()

        print("Computing forward...")

        if self.model.dimension == 2:
            # constant density
            if self.model.density is None:
                self.__forward_2D_constant_density()
            else:
                self.__forward_2D_variable_density()
        elif self.model.dimension == 3:
            # constant density
            if self.model.density is None:
                self.__forward_3D_constant_density()
            else:
                self.__forward_3D_variable_density()
        else:
            raise Exception("Grid dimension {} not supported".format(self.model.dimension))

        return self.model.grid.wavefield, self.elapsed_time
