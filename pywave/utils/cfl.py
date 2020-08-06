import numpy as np

"""
Return a list of finite differences coefficients according to the space order

Parameters
----------
space_order : int
    Spatial order

Returns
----------
list
    List of FD coefficients
"""
def get_coefficients(space_order):

    coeffs = {
        2 : [1, -2, 1],
        4 : [-1/12, 16/12, -30/12, 16/12, -1/12]
    }

    return coeffs[space_order]

"""
Calculate dt with CFL conditions
Based on https://library.seg.org/doi/pdf/10.1190/1.1444605 for the acoustic case

Parameters
----------
dimension : int
    Domain dimension. 2 (2D) or 3 (3D)

space_order : int
    Spatial order

spacing : tuple(float, float, float)
    Spacing between grid points

vel_model : grid
    Velocity model

Returns
----------
float
    dt in miliseconds
"""
def calc_dt(dimension, space_order, spacing, vel_model):

    # 2nd order in time
    a1 = 4

    # FD coeffs to the specific space order
    fd_coeffs = get_coefficients(space_order)

    a2 = dimension * np.sum( np.abs(fd_coeffs) )

    coeff = np.sqrt(a1/a2)

    # The CFL condtion is then given by
    # dt <= coeff * h / max(velocity)
    dt = coeff * np.min(spacing) / np.max(vel_model)

    return dt

"""
Calc the number of timesteps

Parameters
----------
time : int
    Propagation simulation time in miliseconds

dt : float
    Timestep variation

Returns
----------
int
    Number of timesteps
"""
def calc_num_timesteps(time, dt):

    return int(np.floor(time / dt))
