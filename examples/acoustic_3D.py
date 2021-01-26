from pywave import *
import numpy as np

# shape of the grid
shape = (256, 256, 256)

# spacing
spacing = (15.0, 15.0, 15.0)

# propagation time
time = 1000

# get the velocity model
vel = Data(shape=shape)

# get the density model
density = Data(shape=vel.shape(), constant=1)

# get the compiler
compiler = Compiler(program_version='sequential')

# create a grid
grid = Grid(shape=vel.shape())

#-----------------------------------------------
src_points, src_values = get_source_points(grid_shape=shape,source_location=(30,128,128),half_width=4)

rec_points = np.array([], dtype=np.uint)

rec_values = np.array([], dtype=np.float32)

for i in range(256):
    points, values = get_source_points(grid_shape=shape,source_location=(30,i,i),half_width=4)

    rec_points = np.append(rec_points, points)
    rec_values = np.append(rec_values, values)
#-----------------------------------------------

setup = Setup(
    grid = grid,
    velocity = vel,
    #density = density,
    origin = (30,128,128),
    spacing = spacing,
    progatation_time = time,
    frequency = 5.0,
    nbl = 0,
    compiler = compiler,
    src_points_interval = src_points,
    src_points_values = src_values,
    rec_points_interval = rec_points,
    rec_points_values = rec_values,
    num_receivers=255
)

solver = AcousticSolver(setup = setup)

wavefields, rec, exec_time = solver.forward()

'''
count=0
for wavefield in wavefields:
    plot(wavefield, file_name="arq-"+str(count))
    count += 1
'''

print("Forward execution time: %f seconds" % exec_time)

#plot(wavefield[1:512,damp+1:512+damp])

plot_wavefield(wavefields[5,:,:])
#plot_wavefield(wavefields[:,:,128])
plot_shotrecord(rec)
