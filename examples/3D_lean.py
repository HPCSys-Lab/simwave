from pywave import *
import numpy as np

# shape of the grid
shape = (128, 128, 128)

# spacing
spacing = (15.0, 15.0, 15.0)

# propagation time
time = 800

# Velocity model
vel = np.zeros(shape, dtype=np.float32)
vel[:] = 1500.0
velModel = Model(ndarray=vel)

# Compiler
compiler = Compiler(program_version='sequential', c_code='lean/sequential.c')

setup = Setup(
    velocity_model=velModel,
    spacing=spacing,
    propagation_time=time,
    compiler=compiler
)

solver = AcousticSolver(setup=setup)

wavefield, exec_time = solver.forward()

print("Forward execution time: %f seconds" % exec_time)

plot_wavefield(wavefield[5,:,:])
