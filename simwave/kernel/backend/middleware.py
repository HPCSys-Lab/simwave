import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from simwave.kernel.backend.compiler import Compiler


class Middleware:
    """
    Communication interface between frontend and backend.

    Parameters
    ----------
    compiler : Compiler
        Compiler object.
    """
    def __init__(self, compiler):
        if compiler is None:
            self._compiler = Compiler()
        else:
            self._compiler = compiler

    @property
    def compiler(self):
        return self._compiler

    def library(self, dimension, density, dtype):
        """Load and return the C library."""

        # convert dtype to C compiling value
        # that define float precision
        type = {
            'float32': '-DFLOAT',
            'float64': '-DDOUBLE'
        }

        # constant or variable density
        if density is not None:
            density = "variable_density"
        else:
            density = "constant_density"

        # compile the code
        shared_object = self.compiler.compile(
            dimension=dimension,
            density=density,
            float_precision=type[str(dtype)],
            operator="forward"
        )

        # load the library
        return ctypes.cdll.LoadLibrary(shared_object)

    def exec(self, operator, **kwargs):
        """
        Run an operator.

        Parameters
        ----------
        operator : str
            operator to be executed.
        kwargs : dict
            List of keyword arguments.

        Returns
        ----------
        tuple
            The operation results
        """

        # convert the boundary condition to specification to int
        bc = self._convert_boundary_condition(
            kwargs.get('boundary_condition')
        )
        kwargs.update({'boundary_condition': bc})

        # if density model is None, remove it
        # and remove first_order_fd_coefficients
        if kwargs.get('density_model') is None:
            kwargs.pop('density_model')
            kwargs.pop('first_order_fd_coefficients')

        # get the grid shape
        grid_shape = kwargs.get('velocity_model').shape

        try:
            nz, nx = grid_shape
            kwargs.update({'nz': nz, 'nx': nx})
        except ValueError:
            nz, nx, ny = grid_shape
            kwargs.update({'nz': nz, 'nx': nx, 'ny': ny})

        # unpack the grid spacing
        grid_spacing = kwargs.get('grid_spacing')
        kwargs.pop('grid_spacing')

        try:
            dz, dx = grid_spacing
            kwargs.update({'dz': dz, 'dx': dx})
        except ValueError:
            dz, dx, dy = grid_spacing
            kwargs.update({'dz': dz, 'dx': dx, 'dy': dy})

        # run the forward operator
        if operator == 'forward':
            return self._exec_forward(**kwargs)

    def _exec_forward(self, **kwargs):
        """
        Run the forward operator.

        Parameters
        ----------
        kwargs : dict
            Dictonary of keyword arguments.

        Returns
        ----------
        ndarray
            Full wavefield after timestepping.
        ndarray
            Shot record after timestepping.
        """

        # load the C library
        lib = self.library(
            dimension=len(kwargs.get('velocity_model').shape),
            density=kwargs.get('density_model'),
            dtype=kwargs.get('velocity_model').dtype
        )

        # get the argtype for each arg key
        types = self._argtypes(**kwargs)

        # get the all possible keys in order
        ordered_keys = self._keys_in_order

        # list of argtypes
        argtypes = []

        # list of args to pass to C function
        args = []

        # compose the list of args and arg types
        for key in ordered_keys:
            if kwargs.get(key) is not None:
                argtypes.append(types.get(key))
                args.append(kwargs.get(key))

        forward = lib.forward
        forward.restype = ctypes.c_double
        forward.argtypes = argtypes

        # run the C forward function
        exec_time = forward(*args)

        print('Run forward in %f seconds.' % exec_time)

        return kwargs.get('u_full'), kwargs.get('shot_record')

    @property
    def _keys_in_order(self):
        """
        Return all possible arg keys in the expected order in the C function.
        """

        key_order = [
            'u_full',
            'velocity_model',
            'density_model',
            'damping_mask',
            'wavelet',
            'wavelet_size',
            'wavelet_count',
            'second_order_fd_coefficients',
            'first_order_fd_coefficients',
            'boundary_condition',
            'src_points_interval',
            'src_points_interval_size',
            'src_points_values',
            'src_points_values_size',
            'src_points_values_offset',
            'rec_points_interval',
            'rec_points_interval_size',
            'rec_points_values',
            'rec_points_values_size',
            'rec_points_values_offset',
            'shot_record',
            'num_sources',
            'num_receivers',
            'nz',
            'nx',
            'ny',
            'dz',
            'dx',
            'dy',
            'saving_stride',
            'dt',
            'begin_timestep',
            'end_timestep',
            'space_order',
            'num_snapshots'
        ]

        return key_order

    def _argtypes(self, **kwargs):
        """
        Get the ctypes argtypes of the keyword arguments.

        Parameters
        ----------
        kwargs : dict
            Dictonary of keyword arguments.

        Returns
        ----------
        dict
            Dictonary of argtypes with the same keys.
        """
        types = {}

        for key, value in kwargs.items():

            if isinstance(value, np.ndarray):
                types[key] = self._convert_type_to_ctypes(
                    'np({})'.format(str(value.dtype))
                )
            else:
                types[key] = self._convert_type_to_ctypes(
                    type(value).__name__
                )

        return types

    def _convert_boundary_condition(self, boundary_condition):
        """
        Convert a boundary condition str to int.
        (none : 0, null_dirichlet : 1, null_neumann : 2).

        Parameters
        ----------
        boundary_condition : tuple of str
            Boundary conditions on the edges of each axis.

        Returns
        ----------
        ndarray
            Boundary conditions sequence as a numpy array.
        """

        bc = {
            'none': 0,
            'null_dirichlet': 1,
            'null_neumann': 2
        }

        conv_bc = [bc[i] for i in boundary_condition]

        return np.uint(conv_bc)

    def _convert_type_to_ctypes(self, type):
        """
        Convert a given type in python to a ctypes.argtypes format.

        Parameters
        ----------
        type : str
            Native type in python or numpy dtype.

        Returns
        ----------
        object
            Argtype in ctypes format.
        """
        argtype = {
            'int': ctypes.c_size_t,
            'float': ctypes.c_float,
            'float32': ctypes.c_float,
            'float64': ctypes.c_double,
            'np(uint64)': ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
            'np(float32)': ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            'np(float64)': ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
        }

        return argtype[type]
