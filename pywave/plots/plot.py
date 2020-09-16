import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def plot(wavefield, file_name = 'wavefield'):

    # create the destination dir
    os.makedirs('plots', exist_ok=True)

    # process data and generate the plot
    plt.imshow(wavefield)
    #plt.colorbar()
    plt.savefig('plots/{}.png'.format(file_name), format='png')

    print("Plot saved in plots/{}.png".format(file_name))
