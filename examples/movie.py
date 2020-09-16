import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2

parser = argparse.ArgumentParser(description='How to use this program')
parser.add_argument("--wavefields", type=str, required=True, help="Path to the dir with the wavefiels")
parser.add_argument("--iterations", type=int, required=True, help="Number of iterations")
parser.add_argument("--n1", type=int, required=True, help="Rowns of the grid")
parser.add_argument("--n2", type=int, required=True, help="Columns of the grid")
parser.add_argument("--hop", type=int, required=True, help="Number of hops in the iterations to save the file")
parser.add_argument("--name", type=str, required=False, default="wavefield", help="Name of the video to be generated")
args = parser.parse_args()

img_array = []

for i in range(args.iterations):

    if( i % args.hop == 0 ):

        if args.wavefields[-1] != '/':
            args.wavefields += '/'

        file_name = '%swavefield-iter-%d-grid-%d-%d.txt' % (args.wavefields, i, args.n1, args.n2);

        # read the input (wavefield N x M) file
        input = np.loadtxt(file_name)
        # process data e generate the plot
        plt.imshow(input)
        plt.savefig('plots/plot-{}.png'.format(str(i)), format='png')

        # generate a movie
        img = cv2.imread('plots/plot-{}.png'.format(str(i)))
        (height, width, layers) = img.shape
        size = (width,height)
        img_array.append(img)

out = cv2.VideoWriter('{}.avi'.format(args.name), cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()

print("Video saved in {}.avi".format(args.name))
