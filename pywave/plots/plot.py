import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_wavefield(
    wavefield,
    file_name="wavefield",
    colorbar=True,
    cmap="gray",
    show=False,
    clim=[-5, 5],
):

    # create the destination dir
    os.makedirs("plots", exist_ok=True)

    # process data and generate the plot
    plot = plt.imshow(wavefield, cmap=cmap)

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
        plt.clim(clim)

    plt.savefig("plots/{}.png".format(file_name), format="png")

    if show:
        plt.show()

    plt.close()

    print("Wavefield saved in plots/{}.png".format(file_name))


def plot_shotrecord(rec, file_name="shotrecord", colorbar=True, show=False):
    """
    Plot a shot record (receiver values over time).

    Parameters
    ----------
    rec :
        Receiver data with shape (time, points).
    """
    scale = np.max(rec) / 10.0
    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cm.gray)
    plt.xlabel("X position")
    plt.ylabel("Time")

    # Create colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)

    # create the destination dir
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/{}.png".format(file_name), format="png")

    if show:
        plt.show()

    plt.close()

    print("Shot record saved in plots/{}.png".format(file_name))


def plot_velocity_model(
    model, file_name="velocity_model", colorbar=True, cmap="jet", show=False
):

    # create the destination dir
    os.makedirs("plots", exist_ok=True)

    # process data and generate the plot
    plot = plt.imshow(model.data, cmap=cmap)

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)

    plt.savefig("plots/{}.png".format(file_name), format="png")

    if show:
        plt.show()

    plt.close()

    print("Velocity model saved in plots/{}.png".format(file_name))
