import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot(wavefield, file_name = 'wavefield'):

    # create the destination dir
    os.makedirs('plots', exist_ok=True)

    # process data e generate the plot
    plt.imshow(wavefield)
    plt.savefig('plots/{}.png'.format(file_name), format='png')

    print("Plot saved in plots/{}.png".format(file_name))

 
def show(wavefield):
    plt.imshow(wavefield)
    plt.show()
