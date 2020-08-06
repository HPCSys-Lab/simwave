from io import *
import numpy as np
import segyio

class DensityModel():
    """
    Base class to implement the density model

    Parameters
    ----------
    density : float
     Velocity of P wave (m/s)
    shape : (int, ...)
     Size of the base along each axis
    file : str
     Path to the density model file
    """
    def __init__(self, density=1.0, shape=None, file=None):

        # create a density model
        if shape is not None:
            self.model = np.zeros(shape, dtype=np.float32)
            self.model[:] = density

        # read the density model from a file .segy
        if file is not None:
            self.model = io.read_2D_segy(file)

    def shape(self):
        return self.model.shape
