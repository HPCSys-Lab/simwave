import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import time

class Solver():
    """
    Implement the operators for the solver.

    Parameters
    ----------
    setup : object
        Define the configuration for the execution.
    """
    def __init__(self, setup=None):

        self.setup = setup

        # load the lib
        self.__load_lib()

    def __load_lib(self):

        # dimension
        dimension='{}d'.format(self.setup.dimension)

        # constant or variable density
        if self.setup.density_model and self.setup.space_order == 2:
            density = 'variable_density'
        else:
            density = 'constant_density'

        # space order mode
        space_order_mode = 'multiple_space_order'

        # compile the code
        shared_object = self.setup.compiler.compile(
                            dimension = dimension,
                            density = density,
                            space_order_mode = space_order_mode,
                        )

        # load the library
        self.library = ctypes.cdll.LoadLibrary(shared_object)

    def _print_params(self):
        print("Dimension: %dD" % self.setup.dimension)
        print("Shape:", self.setup.grid.shape())
        print("Spacing:", self.setup.spacing)
        print("Density:", ("constant" if self.setup.density_model is None else "variable") )
        print("Space Order:", self.setup.space_order)
        print("Propagation time: %d miliseconds " % self.setup.propagation_time)
        print("DT: %f seconds" % self.setup.dt)
        print("Frequency: %0.1f Hz" % self.setup.sources.wavelet.frequency)
        print("Timesteps:", self.setup.timesteps)

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
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t
        ]

        nz, nx = self.setup.grid.shape()
        dz, dx = self.setup.spacing
        bc = self.setup.domain_pad.get_boundary_conditions(self.setup.dimension)

        #self.wavefields = np.zeros((self.setup.timesteps, nz, nx), dtype=np.float32)

        self.elapsed_time = self.forward(
            #self.wavefields,
            self.setup.grid.data,
            self.setup.velocity_model.data,
            self.setup.damp,
            self.setup.wavelet,
            self.setup.coeff,
            bc,
            self.setup.src_points_interval,
            self.setup.src_points_values,
            self.setup.rec_points_interval,
            self.setup.rec_points_values,
            self.setup.shot_record,
            self.setup.sources.count(),
            self.setup.receivers.count(),
            nz,
            nx,
            dz,
            dx,
            self.setup.jumps,
            self.setup.dt,
            0,
            self.setup.timesteps,
            self.setup.space_order
        )

    def __forward_3D_constant_density(self):

        self.forward = self.library.forward_3D_constant_density

        self.forward.restype = ctypes.c_double

        self.forward.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
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
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t
        ]

        nz, nx, ny = self.setup.grid.shape()
        dz, dx, dy  = self.setup.spacing
        bc = self.setup.domain_pad.get_boundary_conditions(self.setup.dimension)

        #self.wavefields = np.zeros((self.setup.timesteps, nz, nx), dtype=np.float32)

        self.elapsed_time = self.forward(
            #self.wavefields,
            self.setup.grid.data,
            self.setup.velocity_model.data,
            self.setup.damp,
            self.setup.wavelet,
            self.setup.coeff,
            bc,
            self.setup.src_points_interval,
            self.setup.src_points_values,
            self.setup.rec_points_interval,
            self.setup.rec_points_values,
            self.setup.shot_record,
            self.setup.sources.count(),
            self.setup.receivers.count(),
            nz,
            nx,
            ny,
            dz,
            dx,
            dy,
            self.setup.jumps,
            self.setup.dt,
            0,
            self.setup.timesteps,
            self.setup.space_order
        )


    """
    def __forward_2D_variable_density(self):

        self.forward = self.library.forward_2D_variable_density

        self.forward.restype = ctypes.c_double

        self.forward.argtypes = [
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
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
            ctypes.c_size_t
        ]

        nz, nx = self.setup.grid.shape()
        dz, dx = self.setup.spacing
        origin_z, origin_x = self.setup.origin

        self.elapsed_time = self.forward(
            self.setup.grid.wavefield,
            self.setup.velocity.model,
            self.setup.density.model,
            self.setup.damp,
            nz,
            nx,
            dz,
            dx,
            self.setup.wavelet,
            origin_z,
            origin_x,
            self.setup.timesteps,
            self.setup.dt,
            self.setup.coeff,
            self.setup.space_order
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
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t
        ]

        nz, nx, ny = self.setup.grid.shape()
        dz, dx, dy = self.setup.spacing
        origin_z, origin_x, origin_y = self.setup.origin

        self.elapsed_time = self.forward(
            self.setup.grid.wavefield,
            self.setup.velocity.model,
            nz,
            nx,
            ny,
            dz,
            dx,
            dy,
            self.setup.wavelet,
            origin_z,
            origin_x,
            origin_y,
            self.setup.timesteps,
            self.setup.dt,
            self.setup.coeff,
            self.setup.space_order
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
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_size_t
        ]

        nz, nx, ny = self.setup.grid.shape()
        dz, dx, dy = self.setup.spacing
        origin_z, origin_x, origin_y = self.setup.origin

        self.elapsed_time = self.forward(
            self.setup.grid.wavefield,
            self.setup.velocity.model,
            self.setup.density.model,
            nz,
            nx,
            ny,
            dz,
            dx,
            dy,
            self.setup.wavelet,
            origin_z,
            origin_x,
            origin_y,
            self.setup.timesteps,
            self.setup.dt,
            self.setup.coeff,
            self.setup.space_order
        )
    """
    def forward(self):

        self.__print_params()

        print("Computing forward...")

        if self.setup.dimension == 2:
            # constant density
            if self.setup.density_model is None:
                self.__forward_2D_constant_density()
            else:
                self.__forward_2D_variable_density()
        elif self.setup.dimension == 3:
            # constant density
            if self.setup.density_model is None:
                self.__forward_3D_constant_density()
            else:
                self.__forward_3D_variable_density()
        else:
            raise Exception("Grid dimension {} not supported".format(self.setup.dimension))

        return self.setup.grid.data, self.setup.shot_record, self.elapsed_time
        #return self.wavefields, self.elapsed_time
