import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import time

class Operator():

    """
    Base class to implement the operator

    Parameters
    ----------
    params : dictionary
        Dictionary with propagation params
    """
    def __init__(self, params):
        self.params = params

        self.dimension = params['dimension']
        self.grid = params['grid']
        self.compiler = params['compiler']
        self.velmodel = params['vel_model']
        self.density  = params['density']
        self.timesteps = params['timesteps']
        self.dz = params['dz']
        self.dx = params['dx']
        self.dt = params['dt']
        self.print_steps = params['print_steps']

        # load the lib
        self.load_lib()

    def load_lib(self):

        # compile the code
        shared_object = self.compiler.compile()

        lib = ctypes.cdll.LoadLibrary(shared_object)

        self.acoustic_forward = lib.acoustic_forward

        self.acoustic_forward.restype = ctypes.c_int

        self.acoustic_forward.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                          ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                          ctypes.c_size_t,
                                          ctypes.c_size_t,
                                          ctypes.c_size_t,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_float,
                                          ctypes.c_int]

    def forward(self):

        nz, nx = self.grid.shape()

        print("Computing forward...")

        start_time = time.time()

        status = self.acoustic_forward(self.grid.wavefield,
                                       self.velmodel.model,
                                       nz,
                                       nx,
                                       self.timesteps,
                                       self.dz,
                                       self.dx,
                                       self.dt,
                                       self.print_steps)

        elapsed_time = (time.time() - start_time)

        if status != 0:
            raise Exception("Operator forward lib failed")

        return self.grid.wavefield, elapsed_time
