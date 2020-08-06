from pywave import *
from utils import *

# shape of the grid
shape = (512, 512)

# spacing
spacing = (20.0, 20.0)

# propagation time
time = 1000

# get the velocity model
vel = VelocityModel(shape=shape)

# get the density model
density = DensityModel(shape=shape)

# get the dimension
dimension = len(shape)

if dimension == 2:
    nz, nx = tuple(shape)
    dz, dx = tuple(spacing)
else:
    nz, nx, ny = tuple(shape)
    dz, dx, dy = tuple(spacing)

compiler = Compiler(program_version='sequential')

grid = Grid(shape=shape)
grid.add_source()

# apply CFL conditions
dt = cfl.calc_dt(dimension=dimension, space_order=2, spacing=spacing, vel_model=vel.model)
timesteps = cfl.calc_num_timesteps(time, dt)
timesteps = 500

params = {
    'compiler' : compiler,
    'grid' : grid,
    'vel_model' : vel,
    'density' : density,
    'timesteps': timesteps,
    'dimension' : dimension,
    'dz' : dz,
    'dx' : dx,
    #'dy' : dy,
    'dt' : dt,
    'print_steps' : 0
}

op = Operator(params)

wavefield, exec_time = op.forward()

print("dt: %f miliseconds" % dt)
print("Number of timesteps:", timesteps)
print("Forward execution time: %f seconds" % exec_time)

plot(wavefield)
