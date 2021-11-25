"""Generates an harmonic solution to the wave equation

div(grad(u)) - 1/c^2 * d^2u/dt^2 = f

where u = sin(kx*x) * sin(kz*z) * cos(w*t)
"""


# from simwave import plot_shotrecord
import numpy as np
import matplotlib.pyplot as plt


def grid(Lx, Lz, h):
    nz = int((Lz) / h + 1)
    nx = int((Lx) / h + 1)
    zcoords = np.linspace(0, Lz, nz)
    xcoords = np.linspace(0, Lx, nx)
    return np.meshgrid(zcoords, xcoords)

def ansatz(Lx, Lz, h, freq, tp):
    # Params
    kz = np.pi / Lx
    kx = np.pi / Lz
    meshgrid = grid(Lx, Lz, h)
    omega = 2*np.pi*freq
    # Solution
    space_solution = np.sin(kx*meshgrid[0]) * np.sin(kz*meshgrid[1])
    time_solution = np.cos(omega*tp)
    return (space_solution[:,:,None] * time_solution).T

if __name__ == "__main__":
    # Domain info
    Lx, Lz = 5120, 5120
    h = 10
    # Params
    freq= 5
    kz = np.pi / Lx
    kx = np.pi / Lz
    # Time info
    t0 = 0
    tf = 1.5
    nt = 369
    tp = np.linspace(t0, tf, nt)
    # Solution
    solution = ansatz(Lx, Lz, h, freq, tp)

    # Save it
    print("saving analytical solution to .npy file")
    np.save("analytical_harmonic", solution)

