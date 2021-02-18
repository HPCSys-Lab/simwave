import numpy as np


class Grid:
    """
    Base class to implement the grid

    Parameters
    ----------
    shape : tuple(int, ...)
        Size of the grid along each axis (Z, X [, Y])
    """

    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=np.float32)

    def shape(self):
        return self.data.shape

    def replicate_for_timesteps(self, num_timesteps):
        new_shape = (num_timesteps,) + self.shape()

        self.data = np.zeros(new_shape, dtype=np.float32)
