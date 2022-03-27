import numpy as np
import findiff


def half_coefficients(derivative_order, space_order):
    """
    Return a list of right side finite differences coefficients
    according to the space order and derivative order.

    Parameters
    ----------
    space_order : int
        Spatial order.

    Returns
    ----------
    ndarray
        List of FD coefficients.
    """
    # all coefficients
    coeffs = coefficients(derivative_order, space_order)

    middle = len(coeffs) // 2

    # coefficients starting from the center
    coeffs = coeffs[middle:]

    return coeffs


def coefficients(derivative_order, space_order):
    """
    Return a list of finite differences coefficients according
    to the space order and derivative order.

    Parameters
    ----------
    derivative_order : int
        Derivative order
    space_order : int
        Spatial order.

    Returns
    ----------
    ndarray
        List of FD coefficients.
    """

    # fixed second derivative
    coeffs = findiff.coefficients(deriv=derivative_order, acc=space_order)

    return coeffs['center']['coefficients']


def calculate_dt(dimension, space_order, grid_spacing, velocity_model):
    """
    Calculate dt with CFL conditions
    Based on https://library.seg.org/doi/pdf/10.1190/1.1444605
    for the acoustic case.

    Parameters
    ----------
    dimension : int
        Domain dimension. 2 (2D) or 3 (3D).
    space_order : int
        Spatial order
    grid_spacing : tuple(float, float, flsoat)
        Spacing between grid points.
    velocity_model : ndarray
        Velocity model.

    Returns
    ----------
    float
        dt in seconds.
    """

    # 2nd order in time
    a1 = 4

    # FD coeffs to the specific space order
    fd_coeffs = coefficients(
        derivative_order=2,
        space_order=space_order
    )

    a2 = dimension * np.sum(np.abs(fd_coeffs))

    coeff = np.sqrt(a1 / a2)

    # The CFL condtion is then given by
    # dt <= coeff * h / max(velocity)
    dt = coeff * np.min(grid_spacing) / np.max(velocity_model)

    return dt
