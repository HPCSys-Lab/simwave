from scipy.special import i0
import numpy as np
import warnings


def get_kaiser_half_width(half_width):
    """
    Get the optimal b parameter of the kaiser windowing function
    according to the window half-width.

    Parameters
    ----------
    half_width: int
        Window half-width of the kaiser windowing function.

    Returns
    ----------
    float
        Value of b parameter for the window half-width.
    """

    # half-width options are limited from 1 to 10
    if half_width not in list(range(1, 11)):
        raise Exception(
            "Kaiser windowing half-width {} not supported".format(half_width)
        )

    kaiser_b = {
        1: 1.24,
        2: 2.94,
        3: 4.53,
        4: 6.31,
        5: 7.91,
        6: 9.42,
        7: 10.95,
        8: 12.53,
        9: 14.09,
        10: 14.18,
    }

    return kaiser_b[half_width]


def kaiser_windowing_sinc(num_points, source_point, half_width):
    """
    Calculate the Kaiser windowed sinc function of a
    source/receiver grid point position.

    Based on Hicks, Graham J. (2002) Arbitrary source and receiver positioning
    in finite-difference schemes using Kaiser windowed sinc functions.

    Parameters
    ----------
    num_points: int
        Number of grid points along the axis.

    source_point: float
        Source position (in grid points) in the axis.

    half_width: int
        Window half-width of the kaiser windowing function.

    Returns
    ----------
    ndarray
        1D numpy array with the source kaiser windowed sinc value
        on each point. In the points outside the window, the value is NaN.
    """

    x = np.linspace(start=0, stop=num_points-1,
                    num=num_points, dtype=np.float32)

    # calculate the distance (in grid points) of each point of the axis
    # to the point of the source/receiver
    x = x - source_point

    # calculate the square root term of the kaiser windowing function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sqrt = np.sqrt(1 - (x / half_width) ** 2)

    # get the b term value
    b = get_kaiser_half_width(half_width)

    # calculate the kaiser window
    kaiser = i0(b * sqrt) / i0(b)

    # calculate the sinc function of x
    sinc = np.sinc(x)

    # calculate the kaiser windowed sinc function
    kws = kaiser * sinc

    return kws


def get_kws_valid_points(kaiser_windowed_array):
    """
    Given a kaiser windowed array, returns the valid (non-NaN)
    values and points of the source/receiver location.

    Parameters
    ----------
    kaiser_windwed_array: ndarray
        1D numpy array with the source kaiser windowed sinc values
        on all points of the axis.

    Returns
    ----------
    int
        index of first valid point.
    int
        index of last valid point.
    ndarray
        Numpy array with valid (non-NaN) values.
    """

    indexes = []
    values = []

    for index, value in enumerate(kaiser_windowed_array):
        if not np.isnan(value):
            indexes.append(index)
            values.append(value)

    if len(indexes) == 0:
        raise Exception(
            "There is no valid point in the source/receiver location"
        )

    begin_index = indexes[0]
    end_index = indexes[-1]
    values = np.array(values, dtype=np.float32)

    return begin_index, end_index, values


def get_source_points(grid_shape, source_location, half_width):
    """
    Return the point interval of a source/receiver and ther values.

    Parameters
    ----------
    grid_shape: tuple of int
        Number of grid points in each grid axis.
    source_location: tuple of float or list of float
        Source/receiver location (in grid points) in each axis.
    half_width: int
        Window half-width of the kaiser windowing function.

    Returns
    ----------
    ndarray
        1D Numpy array with [begin_point_axis1, end_point_axis1, ..,
        begin_point_axisN, end_point_axisN].
    ndarray
        1D Numpy array with [source_values_axis1, .., source_values_axisN].
    """

    # check with grid and source location have the same dimension
    if len(grid_shape) != len(source_location):
        raise Exception(
            "Grid and source/receiver location must have the same dimension."
        )

    source_points = []
    source_values = np.array([], dtype=np.float32)

    # for each axis
    for axis in range(len(grid_shape)):
        num_points = grid_shape[axis]
        source_point = source_location[axis]

        kws = kaiser_windowing_sinc(num_points, source_point, half_width)
        begin_index, end_index, values = get_kws_valid_points(kws)

        source_points.append(begin_index)
        source_points.append(end_index)
        source_values = np.append(source_values, values)

    source_points = np.array(source_points, dtype=np.uint)

    return source_points, source_values
