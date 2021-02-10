import numpy as np

class DomainPad():
    """
    Implement a domain extension configuration according to the absorbing layers (damping) and spatial order.

    Parameters
    ----------
    nbl : {sequence, int}, optional
        Number of grid points (boundary layers) padded to the edges of each axis (Z,X,[Y]).
        ((before_1, after_1), … (before_N, after_N)) unique pad widths for each side of each axis.
        Int is a shortcut for before = after = pad width for all axes.
        Default is 0 points.
    space_order : int, optional
        Finite differences spatial order.
        Default is order 2.
    damping_polynomial_degree : int, optional
        Degree of the polynomial in the extension function.
        Default is 1 (linear)
    alpha : float, optional
        Constant parameter of the extension function.
        Default is 0.0001
    """
    def __init__(self, nbl=0, space_order=2, damping_polynomial_degree=1, alpha=0.0001):
        self.nbl = nbl
        self.space_order = space_order
        self.damping_polynomial_degree = damping_polynomial_degree
        self.alpha = alpha

    def get_damping_padding(self, dimension):
        """
        Calcute the number of damping points to extend the domain on each side.

        Parameters
        ----------
        dimension : int
            Dimension of the domain (2d or 3d).

        Returns
        ----------
        tuple(tuple(int,...),...)
            Number of points to pad along each side of the numpy array.
        """
        if dimension == 2:
            if isinstance(self.nbl, int):
                padding = ((self.nbl, self.nbl), (self.nbl, self.nbl))

            elif len(self.nbl) == 2 and len(self.nbl[0]) == 2 and len(self.nbl[1]) == 2:
                padding = ((self.nbl[0][0], self.nbl[0][1]), (self.nbl[1][0], self.nbl[1][1]))

            else:
                raise Exception("nbl should have the ((int,int),(int,int)) format.")
        else:
            if isinstance(self.nbl, int):
                padding = ((self.nbl, self.nbl), (self.nbl, self.nbl), (self.nbl, self.nbl))

            elif len(self.nbl) == 3 and len(self.nbl[0]) == 2 and len(self.nbl[1]) == 2 \
                 and len(self.nbl[2]) == 2:
                padding = (
                            (self.nbl[0][0], self.nbl[0][1]),
                            (self.nbl[1][0], self.nbl[1][1]),
                            (self.nbl[2][0], self.nbl[2][1])
                          )

            else:
                raise Exception("nbl should have the ((int,int),(int,int),(int,int)) format.")

        return padding

    def get_spatial_order_padding(self, dimension):
        """
        Calcute the number of space order radius points (halo) to extend the domain on each side.

        Parameters
        ----------
        dimension : int
            Dimension of the domain (2d or 3d).

        Returns
        ----------
        tuple(tuple(int,...),...)
            Number of points to pad along each side of the numpy array.
        """
        # stencil radius
        radius = self.space_order // 2

        if dimension == 2:
            padding = ((radius, radius), (radius, radius))
        else:
            padding = ((radius, radius), (radius, radius), (radius, radius))

        return padding

    def get_damping_mask(self, grid_shape):
        """
        Calcute the damping mask (numpy array) of a grid.
        Damping value is zero inside the original domain, while in the extended region it grows according to a function.

        Parameters
        ----------
        grid_shape : tuple(int,..)
            Shape of the grid.

        Returns
        ----------
        ndarray
            Numpy array with the damping values along the extended grid.
        """

        dimension = len(grid_shape)

        # damp mask in the original domain
        damp_mask = np.zeros(grid_shape, dtype=np.float32)

        # damp mask in the damping extention
        # use the perpendicular distance from the point to the boundary between original and extended domain
        damp_mask = np.pad(damp_mask, self.get_damping_padding(dimension), mode='linear_ramp', end_values=self.nbl)

        # change the damping values (coefficients) according to a function
        damp_mask = (damp_mask ** self.damping_polynomial_degree) * self.alpha

        # damp mask in the halo region
        # The values in this extended region is zero
        damp_mask = np.pad(damp_mask, self.get_spatial_order_padding(dimension))

        return damp_mask

    def extend_grid(self, grid):
        """
        Extend the grid.

        Parameters
        ----------
        grid : object
            The finite differences grid object.

        Returns
        ----------
        object
            Extended grid.
        """

        dimension = len(grid.shape())

        # extension to the damping region
        grid.data = np.pad(grid.data, self.get_damping_padding(dimension))

        # extension to the spatial order halo
        grid.data = np.pad(grid.data, self.get_spatial_order_padding(dimension))

        return grid

    def extend_model(self, model):
        """
        Extend the velocity/density model.

        Parameters
        ----------
        model : object
            Velocity or density model object.

        Returns
        ----------
        ndarray
            Extended model.
        """

        dimension = len(model.shape())

        # extension to the damping region
        model.data = np.pad(model.data, self.get_damping_padding(dimension), mode='edge')

        # extension to the spatial order halo
        model.data = np.pad(model.data, self.get_spatial_order_padding(dimension), mode='edge')

        return model

    def adjust_source_position(self, position):
        """
        Adjust the position of a source/receiver in the extend domain.

        Parameters
        ----------
        position : tuple(float,...)
            Source/receiver position (in grid points) along each axis.

        Returns
        ----------
        tuple(float,...)
            Adjusted source/receiver position
        """

        dimension = len(position)

        padding = self.get_damping_padding(dimension)

        # stencil_radius
        stencil_radius = self.space_order // 2

        if dimension == 2:
            z_pad = padding[0][0]
            x_pad = padding[1][0]

            source_position = (
                position[0] + stencil_radius + z_pad,
                position[1] + stencil_radius + x_pad
            )
        else:
            z_pad = padding[0][0]
            x_pad = padding[1][0]
            y_pad = padding[2][0]

            source_position = (
                position[0] + stencil_radius + z_pad,
                position[1] + stencil_radius + x_pad,
                position[2] + stencil_radius + y_pad
            )

        return source_position
