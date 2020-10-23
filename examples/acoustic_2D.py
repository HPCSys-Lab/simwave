from pywave import *

# shape of the grid
shape = (512, 512)

# spacing
spacing = (10.0, 10.0)

# propagation time
time = 2000

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
    origin = (256+1, 256+101),
    spacing = spacing,
    progatation_time = time,
    frequency = 10.0,
    nbl = 100
)

solver = AcousticSolver(
    model = model,
    compiler = compiler
)

wavefield, exec_time = solver.forward()

print("Forward execution time: %f seconds" % exec_time)

plot(wavefield[1:512,101:612])

#plot(wavefield)
