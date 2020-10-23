import numpy as np
from pywave.kernel import fd, RickerSource

class Model():
    """
    Implement a model of the execution

    Parameters
    ----------
    grid : object
        Object that represents the grid
    velocity: object
        Object that represents the velocity model
    density: object
        Object that represents the density model
    spacing: tuple
        Spacing along each axis
    space_order: int
        Define spatial order
    origin: tuple
        Source coordinates
    progatation_time: int
        Propagation time in miliseconds
    frequency: float
        Peak frequency for Ricker wavelet in Hz
    nbl: int
        Number of boundary layers
    """
    def __init__(self, grid=None, velocity=None, density=None,
                 spacing=None, space_order=2, origin=None,
                 progatation_time=1000, frequency=10, nbl=10):

        self.grid = grid
        self.velocity = velocity
        self.density  = density
        self.spacing = spacing
        self.space_order = space_order
        self.origin = origin
        self.progatation_time = progatation_time
        self.frequency = frequency
        self.nbl = nbl
        self.dimension = len(self.grid.shape())

        # validate dimensions
        self.__validate_dimensions()

        # apply CFL conditions
        self.__apply_cfl_conditions()

        # calc ricker source
        self.__calc_source()

        # get coefficients for FD
        self.coeff = fd.get_right_side_coefficients(self.space_order)

        # calculate padding
        self.__data_padding()

    def __validate_dimensions(self):

        if self.density is None:
            if self.grid.shape() != self.velocity.shape():
                raise Exception("Grid and Velocity Model must have the same dimensions")
        else:
            if (self.grid.shape() != self.velocity.shape()) or (self.grid.shape() != self.density.shape()):
               raise Exception("Grid, Velocity Model and Density Model must have the same dimensions")

        if self.dimension != len(self.spacing):
            raise Exception("Spacing must have {} values".format(self.dimension))

        if self.dimension != len(self.origin):
            raise Exception("Origin must have {} values".format(self.dimension))

    # calculate the list of time values
    def __calc_time_values(self):
        self.time_values = np.zeros(self.timesteps, dtype=np.float32)
        t = 0
        for i in range(self.timesteps):
            self.time_values[i] = t
            t += self.dt

    # apply CFL conditions
    def __apply_cfl_conditions(self):

        self.dt = fd.calc_dt(
            dimension = self.dimension,
            space_order = self.space_order,
            spacing = self.spacing,
            vel_model = self.velocity.model
        )

        self.timesteps = fd.calc_num_timesteps(self.progatation_time, self.dt)

        # calculate the time values
        self.__calc_time_values()

    # calcute a ricker source
    def __calc_source(self):
        src = RickerSource(frequency=self.frequency, time_values=self.time_values)
        self.wavelet = src.wavelet()

    def __data_padding(self):

        stencil_radius = self.space_order // 2

        num_layers = self.nbl + stencil_radius

        if len(self.grid.shape()) == 2:
            padding_size = ((stencil_radius, num_layers), (num_layers, num_layers))
        else:
            padding_size = ((stencil_radius, num_layers), (num_layers, num_layers), (num_layers, num_layers))

        # create damp grid
        damp_grid = np.zeros(self.grid.shape(), dtype=np.float32)
        self.damp = np.pad(damp_grid, padding_size, mode='linear_ramp', end_values=0.5)

        # pad grid
        self.grid.wavefield = np.pad(self.grid.wavefield, padding_size)

        # pad velocity model
        self.velocity.model = np.pad(self.velocity.model, padding_size, mode='edge')

        # pad density model
        if self.density:
            self.density.model = np.pad(self.density.model, padding_size, mode='edge')
