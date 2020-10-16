from pywave import *

# shape of the grid
shape = (512, 512)

# spacing
spacing = (10.0, 10.0)

# propagation time
time = 1500

# get the velocity model
vel = Data(shape=shape)

# get the density model
density = Data(shape=vel.shape(), constant=1)

# get the compiler
compiler = Compiler(program_version='sequential')

# create a grid
grid = Grid(shape=vel.shape())

model = Model(
    grid = grid,
    velocity = vel,
    #density = density,
    origin = (256, 256),
    spacing = spacing,
    progatation_time = time,
    frequency = 11.0
)

solver = AcousticSolver(
    model = model,
    compiler = compiler
)

wavefield, exec_time = solver.forward()

print("Forward execution time: %f seconds" % exec_time)

plot(wavefield)
