import numpy as np
from pywave.data import Grid
from pywave.kernel import fd, Compiler


class Setup:
    """
    Define the parameters and configuration for the solver.

    Parameters
    ----------
    velocity_model : object
        Velocity model object.
    sources : object
        Object that represents the set of sources.
    receivers : object
        Object that represents the set of receivers.
    boundary_config : object
        Object that holds the domain extension configuration.
    spacing : tuple(int,...)
        Spacing along each axis.
    progatation_time : int
        Propagation time in miliseconds.
    space_order : int, optional
        Define spatial order. Defaut is 2.
    jumps : int, optional
        Skipping factor when saving the wavefields. If jumps is 0, only the last wavefield is saved. Default is 0.
    compiler : object, optional
        Object that represents the compiler.
    density: object, optional
        Density model object.
    """

    def __init__(
        self,
        velocity_model,
        sources,
        receivers,
        boundary_config,
        spacing,
        propagation_time,
        space_order=2,
        jumps=0,
        compiler=Compiler(),
        density_model=None,
    ):

        self.velocity_model = velocity_model
        self.density_model = density_model
        self.sources = sources
        self.receivers = receivers
        self.compiler = compiler
        self.boundary_config = boundary_config
        self.spacing = spacing
        self.space_order = space_order
        self.propagation_time = propagation_time
        self.jumps = jumps

        # create a finite differences grid
        self.grid = Grid(shape=self.velocity_model.shape())

        # get the velocity_model dimension
        self.dimension = len(self.velocity_model.shape())

        # validate dimensions
        self.__validate_dimensions()

        # apply CFL conditions
        self.dt, self.timesteps = self.__apply_cfl_conditions()

        # calculate the time values
        self.time_values = self.__calc_time_values()

        # calculate the wavelet
        self.wavelet = self.sources.wavelet.ricker(time_values=self.time_values)

        # extend the domain
        self.__extend_domain()

        # calcute the sources and receveivers interpolation
        self.__source_receiver_interpolation()

        # shot record
        self.shot_record = np.zeros(
            shape=(self.timesteps, self.receivers.count()), dtype=np.float32
        )

        # replicate the extended grid for all timesteps
        self.__replicate_grid_for_timesteps()

        # get coefficients for FD
        self.coeff = fd.get_right_side_coefficients(self.space_order)

    def get_skipped_timesteps_indexes(self):
        """
        Return a list for timesteps indexes according to the number of jumps.

        Returns
        ----------
        list
            List of timesteps indexes.
        """
        timesteps_indexes = list(range(0, self.timesteps, self.jumps))

        # always add the last timestep
        if timesteps_indexes[-1] != self.timesteps - 1:
            timesteps_indexes.append(self.timesteps - 1)

        return timesteps_indexes

    def __replicate_grid_for_timesteps(self):
        """
        Replicate the the grid to save all (or less according to jump) wavefields.
        """

        if self.jumps < 0 or self.jumps > self.timesteps:
            raise Exception(
                "jumps can not be less than zero or greater than the number of timesteps."
            )

        if self.jumps > 0:

            timesteps_indexes = self.get_skipped_timesteps_indexes()

            num_wavefields = len(timesteps_indexes)

            # replicate grid
            self.grid.replicate_for_timesteps(num_wavefields)

    def __validate_dimensions(self):
        """
        Verify if the velocity model, density model, and spacing have the same dimension.
        Raise an exception otherwise.
        """
        if self.density_model is not None:
            if self.velocity_model.shape() != self.density_model.shape():
                raise Exception(
                    "Velocity Model and Density Model must have the same dimensions"
                )

        if self.dimension != len(self.spacing):
            raise Exception("Spacing must have {} values".format(self.dimension))

    def __calc_time_values(self):
        """
        Calculate the list of time values (current time in each timestep).

        Returns
        ----------
        ndarray
            Numpy array of time values.
        """
        time_values = np.zeros(self.timesteps, dtype=np.float32)
        t = 0
        for i in range(self.timesteps):
            time_values[i] = t
            t += self.dt

        return time_values

    def __apply_cfl_conditions(self):
        """
        Apply CFL conditions to calculte dt and timesteps.

        Returns
        ----------
        float
            Dt (timestep variation).
        int
            Number of timesteps.
        """
        dt = fd.calc_dt(
            dimension=self.dimension,
            space_order=self.space_order,
            spacing=self.spacing,
            vel_model=self.velocity_model.data,
        )

        timesteps = fd.calc_num_timesteps(self.propagation_time, dt)

        return dt, timesteps

    def __extend_domain(self):
        """
        Extend the domain (grid, velocity model, density model) and generate the damping mask.
        """
        self.damp = self.boundary_config.get_damping_mask(
            grid_shape=self.grid.shape(), space_order=self.space_order
        )
        self.grid = self.boundary_config.extend_grid(
            grid=self.grid, space_order=self.space_order
        )
        self.velocity_model = self.boundary_config.extend_model(
            model=self.velocity_model, space_order=self.space_order
        )

        if self.density_model is not None:
            self.density_model = self.boundary_config.extend_model(
                model=self.density_model, space_order=self.space_order
            )

    def __source_receiver_interpolation(self):
        """
        Apply source/receiver interpolation and get list of points and values of it.
        """

        # sources
        points, values = self.sources.get_interpolated_points_and_values(
            grid_shape=self.grid.shape(),
            extension=self.boundary_config,
            space_order=self.space_order,
        )
        self.src_points_interval = points
        self.src_points_values = values

        # receivers
        points, values = self.receivers.get_interpolated_points_and_values(
            grid_shape=self.grid.shape(),
            extension=self.boundary_config,
            space_order=self.space_order,
        )
        self.rec_points_interval = points
        self.rec_points_values = values
