from pywave import *

# shape of the grid
shape = (256, 256, 256)

# spacing
spacing = (20.0, 20.0, 20.0)

# propagation time
time = 1000

# get the velocity model
vel = VelocityModel(shape=shape)

# get the density model
density = DensityModel(shape=vel.shape())

# get the compiler
compiler = Compiler(program_version='sequential')

# create a grid
grid = Grid(shape=vel.shape())
grid.add_source()

solver = AcousticSolver(
    grid = grid,
    velocity_model = vel,
#    density = density,
    compiler = compiler,
    spacing = spacing,
    progatation_time = time
)

wavefield, exec_time = solver.forward()

print("Forward execution time: %f seconds" % exec_time)

plot(wavefield[128,:,:])
