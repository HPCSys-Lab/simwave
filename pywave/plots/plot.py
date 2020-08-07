import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def plot(wavefield, file_name = 'wavefield'):

    # create the destination dir
    os.makedirs('plots', exist_ok=True)

    # process data e generate the plot
    plt.imshow(wavefield)
    plt.savefig('plots/{}.png'.format(file_name), format='png')

    print("Plot saved in plots/{}.png".format(file_name))

def plot3D(wavefield, file_name = 'wavefield'):

    # create the destination dir
    os.makedirs('plots', exist_ok=True)

    z=wavefield[:,0]
    x=wavefield[:,1]
    y=wavefield[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dem3d=ax.plot_surface(z,x,y)
    plt.show()
