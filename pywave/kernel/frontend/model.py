from scipy.interpolate import RegularGridInterpolator
from cached_property import cached_property
import numpy as np
from pywave.kernel.frontend import fd


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
    density_model : ndarray
        Numpy n-dimensional array with the density profile.
    space_order: int, optional
        Spatial order of the stencil. Accepts even orders from 2 to 16.
        Default is 2.
    """
    def __init__(self, bbox, grid_spacing, velocity_model,
                 density_model=None, space_order=2):
        self._bbox = bbox
        self._grid_spacing = grid_spacing
        self._space_order = space_order

        # space_order are limited to [2,4,6,8,10,12,14,16]
        if self.space_order not in list([2, 4, 6, 8, 10, 12, 14, 16]):
            raise Exception(
                "Space order {} not supported".format(self.space_order)
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

        # evaluate timestep size
        self._dt = fd.calculate_dt(
            dimension=self.dimension,
            space_order=self.space_order,
            grid_spacing=self.grid_spacing,
            velocity_model=self.velocity_model
        )

    @property
    def bbox(self):
        return self._bbox

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
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        """Set time step in seconds"""
        if value < 0:
            print("Time step cannot be negative")
        elif value > self._dt:
            print("Time step given violates CFL condition")
        else:
            self._dt = value

    @property
    def shape(self):
        """Shape of the grid and interpolated velocity/density model."""

        # number of grid points (z,x [,y])
        if self.dimension == 2:
            # 2 dimension
            z_min, z_max, x_min, x_max = self.bbox
            z_spacing, x_spacing = self.grid_spacing
            nz = int((z_max - z_min) / z_spacing)
            nx = int((x_max - x_min) / x_spacing)
            return (nz, nx)
        else:
            # 3 dimension
            z_min, z_max, x_min, x_max, y_min, y_max = self.bbox
            z_spacing, x_spacing, y_spacing = self.grid_spacing
            nz = int((z_max - z_min) / z_spacing)
            nx = int((x_max - x_min) / x_spacing)
            ny = int((y_max - y_min) / y_spacing)
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

    @cached_property
    def grid(self):
        """Finite differences grid."""
        # create the grid only once
        return np.zeros(self.shape, dtype=np.float32)

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
        'NN' (null neumann), 'ND' (null dirichlet) and 'N' (none).
        """
        # if boundary is not configured, boundary conditions
        # is 0 (Nothing) in all edges of all axis
        try:
            return self._boundary_condition
        except AttributeError:
            return ('N',) * self.dimension * 2

    @property
    def damping_polynomial_degree(self):
        """Degree of the polynomial in the extension function."""
        # if boundary is not configured, damping_polynomial_degree is 1
        try:
            return self._damping_polynomial_degree
        except AttributeError:
            return 3

    @property
    def damping_alpha(self):
        """Constant parameter of the extension function."""
        # if boundary is not configured, damping_alpha is 0.0001
        try:
            return self._damping_alpha
        except AttributeError:
            return 0.001

    @property
    def fd_coefficients(self):
        """Central and right side finite differences coefficients."""
        return fd.get_right_side_coefficients(self.space_order)

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
            z_min, z_max, x_min, x_max = self.bbox

            # original shape
            z = np.linspace(z_min, z_max, data.shape[0])
            x = np.linspace(x_min, x_max, data.shape[1])

            interpolant = RegularGridInterpolator((z, x), data)

            # new shape
            nz, nx = self.shape
            z = np.linspace(z_min, z_max, nz)
            x = np.linspace(x_min, x_max, nx)

            X, Z = np.meshgrid(x, z)

            return np.float32(interpolant((Z, X)))
        else:
            z_min, z_max, x_min, x_max, y_min, y_max = self.bbox

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

            return np.float32(interpolant((Z, X, Y)))

    @property
    def halo_size(self):
        """
        Size (in grid points) of the halo region on the edges of each axis.
        """
        space_order_radius = self.space_order // 2
        return (space_order_radius, ) * self.dimension * 2

    def config_boundary(self, damping_length=0.0, boundary_condition="N",
                        damping_polynomial_degree=3, damping_alpha=0.001):
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
            Options: 'NN' (null neumann), 'ND' (null dirichlet) and 'N' (none).
            Default is N (no boundaray condition).
        damping_polynomial_degree : int, optional
            Degree of the polynomial in the extension function.
            Default is 1 (linear).
        alpha : float, optional
            Constant parameter of the extension function.
            Default is 0.0001.
        """

        self._damping_polynomial_degree = damping_polynomial_degree
        self._damping_alpha = damping_alpha

        # if it is not, convert damping_length to tuple
        if isinstance(damping_length, (float, int)):
            self._damping_length = (damping_length,) * self.dimension * 2
        else:
            self._damping_length = damping_length

        # if it is not, convert boundary_condition to tuple
        if isinstance(boundary_condition, str):
            self._boundary_condition = (boundary_condition,) * \
                                        self.dimension * 2
        else:
            self._boundary_condition = boundary_condition

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
        damp_mask = np.zeros(self.shape, dtype=np.float32)

        # damp mask in the damping extention
        # use the perpendicular distance from the point to
        # the boundary between original and extended domain
        damp_mask = np.pad(
            array=damp_mask,
            pad_width=self.nbl_pad_width,
            mode="linear_ramp",
            end_values=self.nbl_pad_width
        )

        # change the damping values (coefficients) according to a function
        degree = self.damping_polynomial_degree
        damp_mask = (damp_mask ** degree) * self.damping_alpha

        # damp mask in the halo region
        # The values in this extended region is zero
        damp_mask = np.pad(array=damp_mask, pad_width=self.halo_pad_width)

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
    """
    def __init__(self, space_model, tf, t0=0.0):
        self._space_model = space_model
        self._tf = tf
        self._t0 = t0

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
    def dt(self):
        """Time variation in seconds."""
        return self.space_model.dt

    @property
    def timesteps(self):
        """Number of timesteps of the propagation."""
        return int(np.ceil((self.tf - self.t0 + self.dt) / self.dt))

    @cached_property
    def time_indexes(self):
        """Time indexes of the time values."""
        return np.linspace(0, self.timesteps-1, self.timesteps, dtype=np.uint)

    @cached_property
    def time_values(self):
        """Time values of the propagation timesteps."""
        # time values starting from t0 to tf with dt interval
        return np.linspace(self.t0, self.tf, self.timesteps, dtype=np.float32)
