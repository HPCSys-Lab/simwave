from pywave.kernel import kws
from pywave.kernel.wavelet import Wavelet
import numpy as np

class SourceReceiver():
    """
    Implement a parent class for both sources and receivers

    Parameters
    ----------
    kws_half_width : int
        Window half-width of the kaiser windowing function.
    """
    def __init__(self, kws_half_width):
        self.kws_half_width = kws_half_width

        # create an empty list of source/receivers
        self.locations = []

    def add(self, position):
        """
        Add a new source/receiver.

        Parameters
        ----------
        position : tuple(float,...)
            Source/receiver position (in grid_points) along each axis.
        """
        self.locations.append(position)

    def remove_all(self):
        """
        Remove all sources/receiver.
        """
        self.locations = []

    def get_all(self):
        """
        Get all source/receive positions.

        Returns
        ----------
        list
            List of all source/receiver positions.
        """
        return self.locations

    def count(self):
        """
        Get the number of sources/receives.

        Returns
        ----------
        int
            Number of sources/receivers.
        """
        return len(self.locations)

    def get_adjusted_positions(self, extension, space_order):
        """
        Get all source/receive positions adjusted according to the domain extension.

        Parameters
        ----------
        extension : object
            Domain extension object.
        space_order: int
            Spatial order.

        Returns
        ----------
        list
            List of all source/receiver adjusted positions.
        """
        adjusted_list = [extension.adjust_source_position(i, space_order) for i in self.locations]

        return adjusted_list

    def get_interpolated_points_and_values(self, grid_shape, extension, space_order):
        """
        Return the point interval of a source/receiver and ther values.

        Parameters
        ----------
        grid_shape : tuple(int,...)
            Number of grid points in each grid axis of the extended domain.
        extension : object
            Domain extension object.
        space_order : int
            Spatial order.

        Returns
        ----------
        ndarray
            1D Numpy array with [begin_point_axis1, end_point_axis1, .., begin_point_axisN, end_point_axisN].
        ndarray
            1D Numpy array with [source_values_axis1, .., source_values_axisN].
        """

        # adjust the positions
        adjusted_list = self.get_adjusted_positions(extension, space_order)

        points = np.array([], dtype=np.uint)
        values = np.array([], dtype=np.float32)

        for position in adjusted_list:
            # apply kasier window to interpolate the source/receiver in a region of grid points
            p, v = kws.get_source_points(grid_shape=grid_shape,source_location=position,half_width=self.kws_half_width)
            points = np.append(points, p)
            values = np.append(values, v)

        return points, values

class Source(SourceReceiver):
    """
    Implement the set of sources.

    Parameters
    ----------
    kws_half_width : int, optional
        Window half-width of the kaiser windowing function.
    wavelet : object, optional
        Wavelet for the source
    """
    def __init__(self, kws_half_width=4, wavelet=None):
        super().__init__(kws_half_width)

        # if none, create a default wavelet
        if wavelet is None:
            wavelet = Wavelet()

        self.wavelet = wavelet

class Receiver(SourceReceiver):
    """
    Implement the set of receivers.

    Parameters
    ----------
    kws_half_width : int, optional
        Window half-width of the kaiser windowing function.
    """
    def __init__(self, kws_half_width=4):
        super().__init__(kws_half_width)
