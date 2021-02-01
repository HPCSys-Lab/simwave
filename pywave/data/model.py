from pywave.data import *
import numpy as np
import segyio

class Model():
    """
    Base class to implement the velocity model in m/s or density model.

    Parameters
    ----------
    array : ndarray, optional
        Numpy n-dimensional array that represents the Velocity of P wave (m/s) or density.
    file : str, optional
        Path to the velocity/density model file.
    """
    def __init__(self, ndarray=None, file=None):

        # create a velocity model from array
        if ndarray is not None:
            self.data = np.float32(ndarray)

        # read the velocity model from a file .segy
        if file is not None:
            self.data = io.read_2D_segy(file)

    def shape(self):
        return self.data.shape
