from scipy.interpolate import RegularGridInterpolator
import numpy as np
from simwave.kernel.frontend import fd


class SpaceModel:
    """
    Encapsulates the spatial model of the simulation.

    Parameters
    ----------
    bbox : tuple of float
        Minimum and maximum coordinates in meters of domain corners.
        e.g., (z_min, z_max, x_min, x_max [, y_min, y_max]).
    grid_spacing : tuple of float
        Grid spacing in meters in each axis (z, x [, y]).
    velocity_model : ndarray
        Numpy n-dimensional array with P wave velocity (m/s) profile.
    density_model : ndarray, optional
        Numpy n-dimensional array with the density profile.
    space_order : int, optional
        Spatial order of the stencil. Accepts even orders.
        Default is 2.
    dtype : data-type, optional
        Numpy array float data-type (numpy.float32 or numpy.float64).
        Default is numpy.float32.
    """
    def __init__(self, bounding_box, grid_spacing, velocity_model,
                 density_model=None, space_order=2, dtype=np.float32):

        self._dtype = dtype

        # make sure each bounding is float
        self._bounding_box = tuple([self.dtype(i) for i in bounding_box])

        # make sure each spacing is float
        self._grid_spacing = tuple([self.dtype(i) for i in grid_spacing])

        self._space_order = space_order

        # space_order are limited to even number
        if space_order % 2 != 0:
            raise ValueError(
                "Odd space order {} not supported".format(space_order)
            )

        # space_order are limited from 2 to 20
        if not 2 <= space_order <= 20:
            raise ValueError(
                "Space order limited from 2 to 20."
            )

        # get the space dimension according to the velocity model
        self._dimension = len(velocity_model.shape)

        # intepolate the velocity model
        self._velocity_model = self.interpolate(velocity_model)

        # if provided, intepolate the density model as well
        if density_model is not None:
            self._density_model = self.interpolate(density_model)
        else:
            self._density_model = density_model

    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def grid_spacing(self):
        return self._grid_spacing

    @property
    def velocity_model(self):
        return self._velocity_model

    @property
    def density_model(self):
        return self._density_model

    @property
    def space_order(self):
        return self._space_order

    @property
    def dimension(self):
        return self._dimension

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        """Shape of the grid and interpolated velocity/density model."""

        # number of grid points (z,x [,y])
        if self.dimension == 2:
            # 2 dimension
            z_min, z_max, x_min, x_max = self.bounding_box
            z_spacing, x_spacing = self.grid_spacing
            nz = int((z_max - z_min) / z_spacing) + 1
            nx = int((x_max - x_min) / x_spacing) + 1
            return (nz, nx)
        else:
            # 3 dimension
            z_min, z_max, x_min, x_max, y_min, y_max = self.bounding_box
            z_spacing, x_spacing, y_spacing = self.grid_spacing
            nz = int((z_max - z_min) / z_spacing) + 1
            nx = int((x_max - x_min) / x_spacing) + 1
            ny = int((y_max - y_min) / y_spacing) + 1
            return (nz, nx, ny)

    @property
    def extended_shape(self):
        """Shape of the grid after damping and halo padding."""

        nbl_extension = [
            self.nbl[::2][i] + self.nbl[1::2][i]
            for i in range(len(self.nbl[::2]))
        ]

        halo_extension = [
            self.halo_size[::2][i] + self.halo_size[1::2][i]
            for i in range(len(self.halo_size[::2]))
        ]

        extended_shape = [
            self.shape[i] + nbl_extension[i] + halo_extension[i]
            for i in range(len(self.shape))
        ]

        return tuple(extended_shape)

    @property
    def grid(self):
        """Finite differences grid."""
        # create the grid only once
        return np.zeros(self.shape, dtype=self.dtype)

    @property
    def nbl(self):
        """
        Number of boundary layers (in grid points) on the edges of each axis.
        """
        # if boundary is not configured, nbl is 0 in all edges of all axis
        try:
            return self._nbl
        except AttributeError:
            return (0,) * self.dimension * 2

    @property
    def damping_length(self):
        """Length (in meters) of padded extension to the edges of each axis."""
        # if boundary is not configured, damping lenght is 0.0
        try:
            return self._damping_length
        except AttributeError:
            return (0.0,) * self.dimension * 2

    @property
    def boundary_condition(self):
        """
        Boundary condition implementation on the edges of each axis.
        (none, null_dirichlet, null_neumann)
        """
        # if boundary is not configured, boundary conditions
        # is 0 (none) in all edges of all axis
        try:
            return self._boundary_condition
        except AttributeError:
            return ('none',) * self.dimension * 2

    def fd_coefficients(self, derivative_order):
        """
        Central and right side finite differences coefficients.

        Parameters
        ----------
        derivative_order : int
            Derivative order.

        Returns
        ----------
        ndarray
            Central and right side FD coefficients.
        """
        return self.dtype(
            fd.half_coefficients(derivative_order, self.space_order)
        )

    def interpolate(self, data):
        """
        Interpolate data (ndarray of velocity/density model) to a structured
        grid that covers bbox and has grid_spacing.

        Parameters
        ----------
        data : ndarray
            Numpy array with data to be interpolated.

        Returns
        ----------
        ndarray
            Interpolated data.
        """

        if self.dimension == 2:
            z_min, z_max, x_min, x_max = self.bounding_box

            # original shape
            z = np.linspace(z_min, z_max, data.shape[0])
            x = np.linspace(x_min, x_max, data.shape[1])

            interpolant = RegularGridInterpolator((z, x), data)

            # new shape
            nz, nx = self.shape
            z = np.linspace(z_min, z_max, nz)
            x = np.linspace(x_min, x_max, nx)

            X, Z = np.meshgrid(x, z)

            return self.dtype(interpolant((Z, X)))
        else:
            z_min, z_max, x_min, x_max, y_min, y_max = self.bounding_box

            # original shape
            z = np.linspace(z_min, z_max, data.shape[0])
            x = np.linspace(x_min, x_max, data.shape[1])
            y = np.linspace(y_min, y_max, data.shape[2])

            interpolant = RegularGridInterpolator((z, x, y), data)

            # new shape
            nz, nx, ny = self.shape
            z = np.linspace(z_min, z_max, nz)
            x = np.linspace(x_min, x_max, nx)
            y = np.linspace(y_min, y_max, ny)

            X, Z, Y = np.meshgrid(x, z, y)

            return self.dtype(interpolant((Z, X, Y)))

    @property
    def halo_size(self):
        """
        Size (in grid points) of the halo region on the edges of each axis.
        """
        space_order_radius = self.space_order // 2
        return (space_order_radius, ) * self.dimension * 2

    def config_boundary(self, damping_length=0.0, boundary_condition="none"):
        """
        Applies the domain extension (for absorbing layers with damping)
        and boundary conditions.

        Parameters
        ----------
        damping_length : float or tuple of float, optional
            Length (in meters) of padded extension to the edges of each axis.
            e.g., (z_before, z_after, x_before, x_after [, y_before, y_after]).
            Float is a shortcut for before = after for all axes.
            Default is 0.
        boundary_condition : str of tuple of str
            Boundary condition implementation on the edges of each axis.
            e.g., (z_before, z_after, x_before, x_after [, y_before, y_after]).
            Str is a shortcut for before = after width for all axes.
            Options: none, null_dirichlet, null_neumann.
            Default is N (no boundaray condition).
        """

        # if it is not, convert damping_length to tuple
        if isinstance(damping_length, (float, int)):
            self._damping_length = (self.dtype(damping_length),) \
                                    * self.dimension * 2
        else:
            # make sure damping length is float
            self._damping_length = tuple(
                [self.dtype(i) for i in damping_length]
            )

        # if it is not, convert boundary_condition to tuple
        if isinstance(boundary_condition, str):
            self._boundary_condition = (boundary_condition,) * \
                                        self.dimension * 2
        else:
            self._boundary_condition = boundary_condition

        # validate boundary condition
        for bc in self.boundary_condition:
            if bc not in ('none', 'null_dirichlet', 'null_neumann'):
                raise ValueError(
                    'Boundary condition {} not available.'.format(
                        self.boundary_condition
                    )
                )

        # convert damping length (in meters) to nbl (in grid points)
        if self.dimension == 2:
            lenz1, lenz2, lenx1, lenx2 = self._damping_length
            z_spacing, x_spacing = self.grid_spacing

            z1 = int(lenz1 / z_spacing)
            z2 = int(lenz2 / z_spacing)
            x1 = int(lenx1 / x_spacing)
            x2 = int(lenx2 / x_spacing)

            self._nbl = (z1, z2, x1, x2)
        else:
            lenz1, lenz2, lenx1, lenx2, leny1, leny2 = self._damping_length
            z_spacing, x_spacing, y_spacing = self.grid_spacing

            z1 = int(lenz1 / z_spacing)
            z2 = int(lenz2 / z_spacing)
            x1 = int(lenx1 / x_spacing)
            x2 = int(lenx2 / x_spacing)
            y1 = int(leny1 / y_spacing)
            y2 = int(leny2 / y_spacing)

            self._nbl = (z1, z2, x1, x2, y1, y2)

    @property
    def nbl_pad_width(self):
        """
        Number of nbl values padded to the edges of each axis.
        It is in numpy.pad format, ready to be used in the domain padding.
        """
        # convert the nbl tuple to numpy pad format
        nbl_pad = tuple(
            [(i, j) for i, j in zip(self.nbl[::2], self.nbl[1::2])]
        )

        return nbl_pad

    @property
    def halo_pad_width(self):
        """
        Number of halo values padded to the edges of each axis.
        It is in numpy.pad format, ready to be used in the domain padding.
        """
        # convert the halo size tuple to numpy pad format
        halo_pad = tuple(
            [(i, j) for i, j in zip(self.halo_size[::2], self.halo_size[1::2])]
        )

        return halo_pad

    @property
    def damping_mask(self):
        """
        Return the damping mask (ndarray) of a grid.
        Damping value is zero inside the original domain,
        while in the extended region it grows according to a function:
        damp(z,x,[y]) = alpha * d(z,x,[y])**degree
        """
        # damp mask in the original domain
        damp_mask = np.zeros(self.shape, dtype=self.dtype)
        
        # damp mask in the damping extention
        # use the perpendicular distance from the point to
        # the boundary between original and extended domain
        damp_mask = np.pad(
            array=damp_mask,
            pad_width=self.nbl_pad_width,
            mode="linear_ramp",
            end_values=self.nbl_pad_width
        )

        delta = max(self.damping_length)
        temp_spacing = np.max(self.grid_spacing)
        damp_mask = ( 3 * np.max(self.velocity_model) / (2 * delta ) ) * np.log10(1/10 ** -3) * (( damp_mask * temp_spacing / delta ) ** 2)
        
        damp_mask = np.pad(array=damp_mask, mode="edge", pad_width=self.halo_pad_width)
        return damp_mask

    @property
    def extended_grid(self):
        """Return the extended grid (original + nbl + halo region)."""

        # extension to the damping region
        grid = np.pad(array=self.grid, pad_width=self.nbl_pad_width)

        # extension to the spatial order halo
        grid = np.pad(array=grid, pad_width=self.halo_pad_width)

        return grid

    @property
    def extended_velocity_model(self):
        """
        Return the extended velocity model
        (interpolated original + nbl + halo region).
        """

        # extension to the damping region
        vel_model = np.pad(
            array=self.velocity_model,
            pad_width=self.nbl_pad_width,
            mode="edge"
        )

        # extension to the spatial order halo
        vel_model = np.pad(
            array=vel_model,
            pad_width=self.halo_pad_width,
            mode="edge"
        )

        return vel_model

    @property
    def extended_density_model(self):
        """
        Return the extended density model
        (interpolated original + nbl + halo region).
        """
        if self.density_model is None:
            return None

        # extension to the damping region
        den_model = np.pad(
            array=self.density_model,
            pad_width=self.nbl_pad_width,
            mode="edge"
        )

        # extension to the spatial order halo
        den_model = np.pad(
            array=den_model,
            pad_width=self.halo_pad_width,
            mode="edge"
        )

        return den_model

    def remove_halo_region(self, u):
        """
        Remove the halo region outside some wavefield u.

        Parameters
        ----------
        u : ndarray
            Full wavefield (with snapshots)

        Returns
        ----------
        ndarray
            Full wavefield without halo zone.
        """

        # number of halo grid points
        # It is the same in all edges and axis
        halo = self.halo_size[0]

        if self.dimension == 2:
            return u[:, halo:-halo, halo:-halo]
        elif self.dimension == 3:
            return u[:, halo:-halo, halo:-halo, halo:-halo]
        else:
            raise Exception("Wavefield dimension not supported.")

    def remove_nbl(self, u):
        """
        Remove the damping region (nbl) outside some wavefield u.

        Parameters
        ----------
        u : ndarray
            Full wavefield (with snapshots)

        Returns
        ----------
        ndarray
            Full wavefield without nbl extension.
        """
        nbl = self.nbl

        if self.dimension == 2:
            return u[:, nbl[0]:-nbl[1], nbl[2]:-nbl[3]]
        elif self.dimension == 3:
            return u[:, nbl[0]:-nbl[1], nbl[2]:-nbl[3], nbl[4]:-nbl[5]]
        else:
            raise Exception("Wavefield dimension not supported.")


class TimeModel:
    """
    Encapsulates the time model (or time axis) of the simulation.

    Parameters
    ----------
    space_model : object
        Space model object.
    tf : float
        End time in seconds.
    t0 : float, optional
        Start time in seconds. Default is 0.0.
    dt : float. optional
        Timestep variation in seconds.
    saving_stride : int
        Skipping factor when saving the wavefields.
        If saving_jump is 0, only the last wavefield is saved. Default is 0.
    """
    def __init__(self, space_model, tf, dt=None, t0=0.0, saving_stride=0):
        self._space_model = space_model
        self._tf = self.space_model.dtype(tf)
        self._t0 = self.space_model.dtype(t0)
        self._saving_stride = saving_stride

        # calculate dt according to cfl
        self._dt = fd.calculate_dt(
            dimension=self.space_model.dimension,
            space_order=self.space_model.space_order,
            grid_spacing=self.space_model.grid_spacing,
            velocity_model=self.space_model.velocity_model
        )

        # if setted, update dt to a custom value
        if dt is not None:
            self.dt = dt

        # validate the saving stride
        if not (0 <= self.saving_stride <= self.timesteps):
            raise Exception(
                "Saving jumps can not be less than zero or "
                "greater than the number of timesteps."
            )

    @property
    def space_model(self):
        """Corresponding space model."""
        return self._space_model

    @property
    def tf(self):
        """End time value in seconds."""
        return self._tf

    @property
    def t0(self):
        """Initial time value in seconds."""
        return self._t0

    @property
    def saving_stride(self):
        """Skipping factor when saving the wavefields."""
        return self._saving_stride

    @property
    def dt(self):
        return self.dtype(self._dt)

    @property
    def dtype(self):
        return self.space_model.dtype

    @dt.setter
    def dt(self, value):
        """Set time step in seconds"""
        if value < 0:
            raise ValueError("Time step cannot be negative.")
        elif value > self.dt:
            raise ValueError("Time step value violates CFL condition.")
        else:
            self._dt = value

    @property
    def timesteps(self):
        """Number of timesteps of the propagation."""
        num_timesteps = int(np.ceil((self.tf - self.t0 + self.dt) / self.dt))

        # adjust the number of timesteps according to saving_stride
        if 1 < self.saving_stride <= num_timesteps:
            while num_timesteps % self.saving_stride != 1:
                num_timesteps += 1

        return num_timesteps

    @property
    def time_indexes(self):
        """Time indexes of the time values."""
        return np.linspace(0, self.timesteps-1, self.timesteps, dtype=np.uint)

    @property
    def time_values(self):
        """Time values of the propagation timesteps."""
        # time values starting from t0 to tf with dt interval
        return np.linspace(self.t0, self.tf, self.timesteps,
                           dtype=self.dtype)

    def remove_time_halo_region(self, u):
        """
        Remove the time halo region from the snapshots.

        Parameters
        ----------
        u : ndarray
            Full wavefield with time halo region.

        Returns
        ----------
        ndarray
            Full wavefield without time halo region.
        """

        # it needs the last timestep only
        if self.saving_stride == 0:
            last_snapshot = self.timesteps % 3
            return u[last_snapshot:last_snapshot+1]
        else:
            return u[1:-1]
