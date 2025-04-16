import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot(wavefield, file_name="wavefield", colorbar=True, cmap="gray", extent=None, show=False, clim=[-5, 5]):
    """
    Plot the wavefield.

    Parameters
    ----------
    wavefield : ndarray
        Wavefield data.
    file_name : str, optional
        Name of the image to be saved.
        Default is wavefield.
    colorbar : bool, optional
        If True, show a colorbar. Default is True.
    cmap : str, optional
        The Colormap instance or registered colormap name
        used to map scalar data to colors.
        Default is gray.
    extent : floats(left, right, bottom, top), optional
        The bounding box in data coordinates that the image will fill.
    show : bool, optional
        If True, show the image on a pop up window.
        Default is False.
    clim : list
        Set the color limits of the current image.
        Default is (vmin=-5, vmax=5).
    """

    # create the destination dir
    os.makedirs("plots", exist_ok=True)

    # process data and generate the plot
    plot = plt.imshow(wavefield, cmap=cmap, extent=extent)

    m_unit = 'm' if extent is not None else 'points'

    # labels
    plt.xlabel('Width ({})'.format(m_unit))
    plt.ylabel('Depth ({})'.format(m_unit))

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