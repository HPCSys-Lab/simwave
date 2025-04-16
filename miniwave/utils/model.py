import numpy as np
from typing import Tuple, Optional
import findiff


class Model:
    """
    Encapsulates the both spatial and time model of the simulation.

    Parameters
    ----------
    velocity_model : ndarray
        Numpy 3-dimensional array with P wave velocity (m/s) profile.
    grid_spacing : tuple of float
        Grid spacing in meters in each axis (z, x, y).
    dt : float.
        Time step in seconds.
    num_timesteps : int
        Number of timesteps to run the simulation.
    space_order : int, optional
        Spatial order of the stencil. Accepts even orders. Default is 2.
    dtype : str, optional
        Float precision. Default is float64.
    """
    def __init__(
        self,
        velocity_model: np.ndarray,
        grid_spacing: Tuple[float, float, float],
        dt: float,
        num_timesteps: int,
        space_order: Optional[int] = 2,
        dtype: Optional[str] = 'float64'
    ):

        if dtype == 'float64':
            self.dtype = np.float64
        elif dtype == 'float32':
            self.dtype = np.float32
        else:
            raise ValueError("Unknown dtype. It must be float32 or float64")

        self.velocity_model = velocity_model
        self.grid_spacing = grid_spacing
        self.num_timesteps = num_timesteps
        self.space_order = space_order
        self.dt = dt

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, value: np.dtype) -> None:
        self._dtype = value

    @property
    def velocity_model(self) -> np.ndarray:
        return self._velocity_model

    @velocity_model.setter
    def velocity_model(self, value: np.ndarray):

        if value is None or value.ndim != 3:
            raise ValueError("Velocity model must nd-array and 3-dimensional")

        self._velocity_model = self.dtype(value)

    @property
    def grid_spacing(self) -> Tuple[float, float, float]:
        return self._grid_spacing

    @grid_spacing.setter
    def grid_spacing(self, value: Tuple[float, float, float]):
        if len(value) != 3:
            raise ValueError("Grid spacing must be 3-dimensional")

        self._grid_spacing = (
            self.dtype(value[0]),
            self.dtype(value[1]),
            self.dtype(value[2])
        )

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, value: float):
        if value < 0:
            raise ValueError("Time step cannot be negative.")
        elif value > self.critical_dt:
            raise ValueError("Time step value violates CFL condition.")
        else:
            self._dt = self.dtype(value)

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, value: int):
        self._num_timesteps = value

    @property
    def space_order(self) -> int:
        return self._space_order

    @space_order.setter
    def space_order(self, value: int):
        self._space_order = value

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return self.velocity_model.shape

    @property
    def critical_dt(self) -> float:
        """
        Calculate dt with CFL conditions
        Based on https://library.seg.org/doi/pdf/10.1190/1.1444605
        for the acoustic case.

        Returns
        ----------
        float
            Critical dt in seconds.
        """

        # 2nd order in time
        a1 = 4

        # fixed 2nd time derivative
        fd_coeffs = findiff.coefficients(
            deriv=2,
            acc=self.space_order
        )['center']['coefficients']

        a2 = self.velocity_model.ndim * np.sum(np.abs(fd_coeffs))
        coeff = np.sqrt(a1 / a2)
        dt = coeff * np.min(self.grid_spacing) / np.max(self.velocity_model)

        return self.dtype(dt)

    @property
    def stencil_coefficients(self) -> np.ndarray:
        # fixed second derivative
        coeffs = findiff.coefficients(
            deriv=2,
            acc=self.space_order
        )['center']['coefficients']

        # calculate the center point index
        middle = len(coeffs) // 2

        # coefficients starting from the center
        coeffs = coeffs[middle:]

        return self.dtype(coeffs)

    @property
    def grid(self) -> np.ndarray:

        base_grid = np.zeros(shape=self.grid_shape, dtype=self.dtype)

        return self._add_initial_source(base_grid)

    @property
    def u_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        # return prev_u and next_u arrays

        prev_u = self.grid
        next_u = self.grid.copy()

        return prev_u, next_u

    def _add_initial_source(self, grid: np.ndarray) -> np.ndarray:

        n1, n2, n3 = grid.shape

        val = 50.0

        for s in range(4, -1, -1):
            grid[n1//2-s:n1//2+s, n2//2-s:n2//2+s, n3//2-s:n3//2+s] = val
            val *= 0.9

        return grid
