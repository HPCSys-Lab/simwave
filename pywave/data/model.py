from pywave.data import *

import numpy as np
import segyio
from scipy.interpolate import RegularGridInterpolator


class Model:
    """
    Base class to implement the velocity model in m/s or density model.

    Parameters
    ----------
    array : ndarray, optional
        Numpy n-dimensional array that represents the Velocity of P wave (m/s) or density.
    file : str, optional
        Path to the P/S velocity or density model file.
    bbox : tuple, optional
        Min and maximum coordinates in meters of domain corners
        e.g., (zmin,zmax,xmin,max)
    grid_spacing : tuple, optional
        Grid spacing in meters each axis
    """

    def __init__(self, ndarray=None, file=None, bbox=None, grid_spacing=None):

        if ndarray is not None:
            self._read_from_ndarray(ndarray)
        elif file is not None:
            self._read_from_segy(file, bbox, grid_spacing)
        else:
            raise ValueError("No model specified")

    def shape(self):
        return self.data.shape

    def _read_from_ndarray(self, ndarray):
        # create a velocity model from array
        if ndarray is not None:
            self.data = np.float32(ndarray)

    def _read_from_segy(self, file, bbox, grid_spacing):
        # read the velocity model from a file .segy
        tmp = io.read_2D_segy(file)
        # interpolate this data to a structured grid
        # that covers bbox and has grid_spacing
        minz, maxz, minx, maxx = bbox
        nrow, ncol = tmp.shape
        z = np.linspace(minz, maxz, nrow)
        x = np.linspace(minx, maxx, ncol)
        interpolant = RegularGridInterpolator((z, x), tmp)
        # number of grid points (z,x)
        nz = int((maxz - minz)/grid_spacing[0])
        nx = int((maxx - minx)/grid_spacing[1])
        shape = (nz, nx)
        z = np.linspace(minz, maxz, shape[0])
        x = np.linspace(minx, maxx, shape[1])
        X, Z = np.meshgrid(x, z)
        self.data = interpolant((Z, X))


