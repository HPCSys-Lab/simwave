import numpy as np


def get_right_side_coefficients(space_order):
    """
    Return a list of right side finite differences coefficients
    according to the space order.

    Parameters
    ----------
    space_order : int
        Spatial order.

    Returns
    ----------
    list
        List of FD coefficients.
    """

    # space_order are limited in [2,4,6,8,10,12,14,16]
    if space_order not in list([2, 4, 6, 8, 10, 12, 14, 16]):
        raise Exception("Space order {} not supported".format(space_order))

    coeffs = {
        2: [-2 / 1, 1 / 1],
        4: [-30 / 12, 16 / 12, -1 / 12],
        6: [-490 / 180, 270 / 180, -27 / 180, 2 / 180],
        8: [-14350 / 5040, 8064 / 5040, -1008 / 5040, 128 / 5040, -9 / 5040],
        10: [
            -73766 / 25200,
            42000 / 25200,
            -6000 / 25200,
            1000 / 25200,
            -125 / 25200,
            8 / 25200,
        ],
        12: [
            -2480478 / 831600,
            1425600 / 831600,
            -222750 / 831600,
            44000 / 831600,
            -7425 / 831600,
            864 / 831600,
            -50 / 831600,
        ],
        14: [
            -228812298 / 75675600,
            132432300 / 75675600,
            -22072050 / 75675600,
            4904900 / 75675600,
            -1003275 / 75675600,
            160524 / 75675600,
            -17150 / 75675600,
            900 / 75675600,
        ],
        16: [
            -924708642 / 302702400,
            538137600 / 302702400,
            -94174080 / 302702400,
            22830080 / 302702400,
            -5350800 / 302702400,
            1053696 / 302702400,
            -156800 / 302702400,
            15360 / 302702400,
            -735 / 302702400,
        ],
    }

    return np.float32(coeffs[space_order])


def get_all_coefficients(space_order):
    """
    Return a list of finite differences coefficients according
    to the space order.

    Parameters
    ----------
    space_order : int
        Spatial order.

    Returns
    ----------
    list
        List of FD coefficients.
    """

    # get the right side coefficients of a given spatial order
    right_side_coeff = get_right_side_coefficients(space_order)

    # get all coeffients, right side
    # and left side (inverse of right side coefficients)
    all_coeff = np.append(
                    np.flip(np.delete(right_side_coeff, 0)), right_side_coeff
                )

    return all_coeff


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
    fd_coeffs = get_all_coefficients(space_order)

    a2 = dimension * np.sum(np.abs(fd_coeffs))

    coeff = np.sqrt(a1 / a2)

    # The CFL condtion is then given by
    # dt <= coeff * h / max(velocity)
    dt = coeff * np.min(grid_spacing) / np.max(velocity_model)

    return np.float32(dt)
