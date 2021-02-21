import numpy as np
from pywave.data import Model


class BoundaryProcedures:
    """
    Implement a domain extension configuration according to the absorbing layers (damping) and spatial order.
    Also config the boundary conditions.

    Parameters
    ----------
    nbl : {sequence, int}, optional
        Number of grid points (boundary layers) padded to the edges of each axis (Z,X,[Y]).
        ((before_1, after_1), … (before_N, after_N)) unique pad widths for each side of each axis.
        Int is a shortcut for before = after for all axes.
        Default is 0 points.
    length : {sequence, float}, optional
        Length of padded extension to the edges in the directions of each axis (Z,X,[Y]).
        ((before_1, after_1), … (before_N, after_N)) unique pad widths (in meters) for each side of each axis.
        Float is a shortcut for before = after for all axes.
        If length is given grid_spacing is mandatory.
    grid_spacing : tuple, optional
        Grid spacing in meters each axis. Mandatory if length is given.
    boundary_condition : {sequence, str}
        Boundary condition implementation on the edges of each axis (Z,X,[Y]).
        ((before_1, after_1), … (before_N, after_N)) unique boundary condition for each side of each axis.
        Str is a shortcut for before = after width for all axes.
        Available options include: 'NN' (null neumann), 'ND' (null dirichlet) and 'N' (none).
        Default is N (no boundaray condition).
    damping_polynomial_degree : int, optional
        Degree of the polynomial in the extension function.
        Default is 1 (linear)
    alpha : float, optional
        Constant parameter of the extension function.
        Default is 0.0001
    """

    def __init__(
        self, nbl=0, boundary_condition="N", damping_polynomial_degree=1, alpha=0.0001, length = None, grid_spacing = None
    ):
        if not (length or grid_spacing):
            self.nbl = nbl
        else:
            self.nbl = self._nbl_from_geometry(length, grid_spacing)
        self.boundary_condition = boundary_condition
        self.damping_polynomial_degree = damping_polynomial_degree
        self.alpha = alpha
        self.spacing = grid_spacing

    def _nbl_from_geometry(self, length, spacing):
        """
        Return number of grid points from bbox specification

        Parameters
        ----------
        length : {sequence, float}, optional
            Length of padded extension to the edges in the directions of each axis (Z,X,[Y]).
        spacing : tuple, optional
            Grid spacing in meters each axis. Mandatory if length is given.

        Returns
        -------
        nbl : {sequence, int}, optional
            Number of grid points (boundary layers) padded to the edges of each axis (Z,X,[Y]).
        """
        try:
            if isinstance(length, float) or isinstance(length, int):
                nbl = [tuple(int(length // spc) for x in range(2)) for spc in spacing]
            elif len(length) == 2 and len(length[0]) == 2 and len(length[1]) == 2:
                nbl = [tuple(int(ext // spc) for ext in ax) for ax, spc in zip(length, spacing)]
        except TypeError as err:
            print("If length is given, spacing is mandatory and vice-versa", err)
        except:
            print("Couldn't assign value to nbl")
            raise

        return nbl


    def get_boundary_conditions(self, dimension):
        """
        Return the boundary conditions on each edge of each axis.
        0 is 'N' (no boundary condition).
        1 is 'ND' (null dirichlet)
        2 is 'NN' (null neumann)

        Parameters
        ----------
        dimension : int
            Dimension of the domain (2d or 3d).

        Returns
        ----------
        ndarray
            List of boundary conditions respectively [z_before, z_after, x_before, x_after, [y_before, y_after]]
        """

        bc = {"N": 0, "ND": 1, "NN": 2}

        if dimension == 2:
            if isinstance(self.boundary_condition, str):
                all_bc = [bc[self.boundary_condition]] * 4

            elif (
                len(self.boundary_condition) == 2
                and len(self.boundary_condition[0]) == 2
                and len(self.boundary_condition[1]) == 2
            ):

                all_bc = [
                    bc[self.boundary_condition[0][0]],
                    bc[self.boundary_condition[0][1]],
                    bc[self.boundary_condition[1][0]],
                    bc[self.boundary_condition[1][1]],
                ]
            else:
                raise Exception(
                    "boundary_conditon should have the ((str,str),(str,str)) format."
                )

        else:
            if isinstance(self.boundary_condition, str):
                all_bc = [bc[self.boundary_condition]] * 6

            elif (
                len(self.boundary_condition) == 3
                and len(self.boundary_condition[0]) == 2
                and len(self.boundary_condition[1]) == 2
                and len(self.boundary_condition[2]) == 2
            ):

                all_bc = [
                    bc[self.boundary_condition[0][0]],
                    bc[self.boundary_condition[0][1]],
                    bc[self.boundary_condition[1][0]],
                    bc[self.boundary_condition[1][1]],
                    bc[self.boundary_condition[2][0]],
                    bc[self.boundary_condition[2][1]],
                ]

            else:
                raise Exception(
                    "boundary_conditon should have the ((str,str),(str,str),(str,str)) format."
                )

        return np.array(all_bc, dtype=np.uint)

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
                padding = (
                    (self.nbl[0][0], self.nbl[0][1]),
                    (self.nbl[1][0], self.nbl[1][1]),
                )

            else:
                raise Exception("nbl should have the ((int,int),(int,int)) format.")
        else:
            if isinstance(self.nbl, int):
                padding = (
                    (self.nbl, self.nbl),
                    (self.nbl, self.nbl),
                    (self.nbl, self.nbl),
                )

            elif (
                len(self.nbl) == 3
                and len(self.nbl[0]) == 2
                and len(self.nbl[1]) == 2
                and len(self.nbl[2]) == 2
            ):
                padding = (
                    (self.nbl[0][0], self.nbl[0][1]),
                    (self.nbl[1][0], self.nbl[1][1]),
                    (self.nbl[2][0], self.nbl[2][1]),
                )

            else:
                raise Exception(
                    "nbl should have the ((int,int),(int,int),(int,int)) format."
                )

        return padding

    def get_spatial_order_padding(self, dimension, space_order):
        """
        Calcute the number of space order radius points (halo) to extend the domain on each side.

        Parameters
        ----------
        dimension : int
            Dimension of the domain (2d or 3d).
        space_order : int
            Spatial order.

        Returns
        ----------
        tuple(tuple(int,...),...)
            Number of points to pad along each side of the numpy array.
        """
        # stencil radius
        radius = space_order // 2

        if dimension == 2:
            padding = ((radius, radius), (radius, radius))
        else:
            padding = ((radius, radius), (radius, radius), (radius, radius))

        return padding

    def get_damping_mask(self, grid_shape, space_order):
        """
        Calcute the damping mask (numpy array) of a grid.
        Damping value is zero inside the original domain, while in the extended region it grows according to a function.

        Parameters
        ----------
        grid_shape : tuple(int,..)
            Shape of the grid.
        space_order : int
            Spatial order.

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
        damp_mask = np.pad(
            damp_mask,
            self.get_damping_padding(dimension),
            mode="linear_ramp",
            end_values=self.nbl,
        )

        # change the damping values (coefficients) according to a function
        damp_mask = (damp_mask ** self.damping_polynomial_degree) * self.alpha

        # damp mask in the halo region
        # The values in this extended region is zero
        damp_mask = np.pad(
            damp_mask, self.get_spatial_order_padding(dimension, space_order)
        )

        return damp_mask

    def extend_grid(self, grid, space_order):
        """
        Extend the grid.

        Parameters
        ----------
        grid : object
            The finite differences grid object.
        space_order : int
            Spatial order.

        Returns
        ----------
        object
            Extended grid.
        """

        dimension = len(grid.shape())

        # extension to the damping region
        grid.data = np.pad(grid.data, self.get_damping_padding(dimension))

        # extension to the spatial order halo
        grid.data = np.pad(
            grid.data, self.get_spatial_order_padding(dimension, space_order)
        )

        return grid

    def extend_model(self, model, space_order):
        """
        Extend the velocity/density model. The original model remains unchanged.

        Parameters
        ----------
        model : object
            Velocity or density model object.
        space_order : int
            Spatial order.

        Returns
        ----------
        ndarray
            Extended model.
        """

        dimension = len(model.shape())

        # extension to the damping region
        data = np.pad(model.data, self.get_damping_padding(dimension), mode="edge")

        # extension to the spatial order halo
        data = np.pad(
            data, self.get_spatial_order_padding(dimension, space_order), mode="edge"
        )

        return Model(ndarray=data)

    def adjust_source_position(self, position, space_order):
        """
        Adjust the position of a source/receiver in the extend domain.

        Parameters
        ----------
        position : tuple(float,...)
            Source/receiver position (in grid points) along each axis.
        space_order : int
            Spatial order.

        Returns
        ----------
        tuple(float,...)
            Adjusted source/receiver position
        """

        dimension = len(position)

        padding = self.get_damping_padding(dimension)

        # stencil_radius
        stencil_radius = space_order // 2

        if dimension == 2:
            z_pad = padding[0][0]
            x_pad = padding[1][0]

            source_position = (
                position[0] + stencil_radius + z_pad,
                position[1] + stencil_radius + x_pad,
            )
        else:
            z_pad = padding[0][0]
            x_pad = padding[1][0]
            y_pad = padding[2][0]

            source_position = (
                position[0] + stencil_radius + z_pad,
                position[1] + stencil_radius + x_pad,
                position[2] + stencil_radius + y_pad,
            )

        return source_position
