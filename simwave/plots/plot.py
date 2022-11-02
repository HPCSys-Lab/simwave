import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_wavefield(wavefield, file_name="wavefield", colorbar=True,
                   cmap="gray", solver=None, show=False, clim=[-5, 5]):
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
    solver : Solver, optional
        Solver object. If provided, the plot sets
        the extent values.
    show : bool, optional
        If True, show the image on a pop up window.
        Default is False.
    clim : list
        Set the color limits of the current image.
        Default is (vmin=-5, vmax=5).
    """

    # create the destination dir
    os.makedirs("plots", exist_ok=True)

    if solver is not None:
        # left, right, bottom, top
        left = solver.space_model.bounding_box[2]
        right = solver.space_model.bounding_box[3]
        bottom = solver.space_model.bounding_box[1]
        top = solver.space_model.bounding_box[0]

        extent = [left, right, bottom, top]
    else:
        extent = None

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


def plot_shotrecord(rec, file_name="shotrecord", colorbar=True,
                    show=False, solver=None):
    """
    Plot a shot record (receiver values over time).

    Parameters
    ----------
    rec : ndarray
        Receiver data with shape (time, points).
    file_name : str, optional
        Name of the image to be saved.
        Default is wavelet.
    colorbar : bool, optional
        If True, show a colorbar. Default is True.
    show : bool, optional
        If True, show the image on a pop up window.
        Default is False.
    solver : Solver, optional
        Solver object. If provided, the plot sets
        the extent values.
    """
    scale = np.max(rec) / 10.0

    if solver is not None:
        # left, right, bottom, top
        left = solver.space_model.bounding_box[2]
        right = solver.space_model.bounding_box[3]
        bottom = solver.time_model.tf * 1000
        top = solver.time_model.t0 * 1000

        extent = [left, right, bottom, top]
        x_label = "Width (m)"
    else:
        extent = None
        x_label = "Receivers"

    plot = plt.imshow(rec, vmin=-scale, vmax=scale,
                      cmap=cm.gray, extent=extent)
    plt.xlabel(x_label)
    plt.ylabel("Time (ms)")

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


def plot_velocity_model(model, sources=None, receivers=None,
                        file_name="velocity_model", colorbar=True,
                        solver=None, cmap="jet", show=False):
    """
    Plot the velocity model.

    Parameters
    ----------
    model : ndarray
        Velocity model data.
    sources : ndarray
        Coordinates of the sources.
    receivers : ndarray
        Coordinates of the receivers.
    file_name : str, optional
        Name of the image to be saved.
        Default is velocity_model.
    colorbar : bool, optional
        If True, show a colorbar. Default is True.
    cmap : str, optional
        The Colormap instance or registered colormap name
        used to map scalar data to colors.
        Default is jet.
    solver : Solver, optional
        Solver object. If provided, the plot sets
        the extent values.
    show : bool, optional
        If True, show the image on a pop up window.
        Default is False.
    """

    # create the destination dir
    os.makedirs("plots", exist_ok=True)

    if solver is not None:
        # left, right, bottom, top
        left = solver.space_model.bounding_box[2]
        right = solver.space_model.bounding_box[3]
        bottom = solver.space_model.bounding_box[1]
        top = solver.space_model.bounding_box[0]

        extent = [left, right, bottom, top]
    else:
        extent = None

    # process data and generate the plot
    plot = plt.imshow(model, cmap=cmap, extent=extent)

    m_unit = 'm' if extent is not None else 'points'

    # labels
    plt.xlabel('Width ({})'.format(m_unit))
    plt.ylabel('Depth ({})'.format(m_unit))

    # plot receiver points, if provided
    if receivers is not None:
        # in simwave, it's assumed
        # Z (vertical) and X (horizontal) order
        plt.scatter(receivers[:, 1], receivers[:, 0],
                    s=20, c='green', marker='D')

    # plot sources points, if provided
    if sources is not None:
        plt.scatter(sources[:, 1], sources[:, 0],
                    s=20, c='red', marker='o')

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


def plot_wavelet(time_values, wavelet_values,
                 file_name="wavelet", show=False):
    """
    Show the wavelet in a graph.

    Parameters
    ----------
    time_values : list
        Discretized values of time in seconds.
    wavelet_values : list
        Pulse of the wavelet.
    file_name : str, optional
        Name of the image to be saved.
        Default is wavelet.
    show : bool, optional
        If True, show the image on a pop up window.
        Default is False.
    """
    plt.plot(time_values, wavelet_values)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tick_params()

    plt.savefig("plots/{}.png".format(file_name), format="png")

    if show:
        plt.show()

    plt.close()

    print("Wavelet saved in plots/{}.png".format(file_name))
