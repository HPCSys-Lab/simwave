from pywave import *

# shape of the grid
shape = (100, 100)

# spacing
spacing = (5.0, 5.0)

# propagation time
time = 200

# get the velocity model
vel = Data(shape=shape)

# get the density model
density = Data(shape=vel.shape(), constant=1)

# get the compiler
compiler = Compiler(program_version='sequential')

# create a grid
grid = Grid(shape=vel.shape())

setup = Setup(
    grid = grid,
    velocity = vel,
    #density = density,
    origin = (50, 50),
    spacing = spacing,
    progatation_time = time,
    frequency = 15.0,
    nbl = 10,
    compiler = compiler
)

solver = AcousticSolver(setup = setup)

wavefield, exec_time = solver.forward()

print("Forward execution time: %f seconds" % exec_time)

#plot(wavefield[1:512,damp+1:512+damp])

plot(wavefield)
