"""Generates an harmonic solution to the wave equation

div(grad(u)) - 1/c^2 * d^2u/dt^2 = f

where u = sin(kx*x) * sin(kz*z) * cos(w*t)
"""


# from simwave import plot_shotrecord
import numpy as np
import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_zlim(-1.01, 1.01)

# Domain info
dz, dx = (10, 10)
zmin, zmax, xmin, xmax = (0, 5120, 0, 5120)
nz = int((zmax - zmin) / dz + 1)
nx = int((xmax - xmin) / dx + 1)
zcoords = np.linspace(zmin, zmax, nz)
xcoords = np.linspace(xmin, xmax, nx)
Z, X = np.meshgrid(zcoords, xcoords)

# Params
c = 1500
freq= 5
w = 2*np.pi*freq
kz = np.pi / (zmax - zmin)
kx = np.pi / (xmax - xmin)

# Time info
t0 = 0
tf = 1.5
nt = 369
tp = np.linspace(t0, tf, nt)

# Solution
space_solution = np.sin(kx*X) * np.sin(kz*Z)
time_solution = np.cos(w*tp)
solution = (space_solution[:,:,None] * time_solution).T

# Save it
print("saving analytical solution to .npy file")
np.save("analytical_harmonic", solution)

