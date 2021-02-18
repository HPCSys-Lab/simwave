import numpy as np
import matplotlib.pyplot as plt


class Wavelet:
    """
    Implement a wavelet for the source.

    Parameters
    ----------
    frequency : float, optional
        Peak frequency for the wavelet in Hz. Default is 5.0
    amplitude : float, optional
        Amplitude of the wavelet. Default is 1.0
    """

    def __init__(self, frequency=5.0, amplitude=1.0):
        self.frequency = frequency
        self.amplitude = amplitude

    def ricker(self, time_values):
        """
        Return a ricker wavelet.

        Parameters
        ----------
        time_values : list
            Discretized values of time in seconds
        """
        t0 = 1 / self.frequency
        r = np.pi * self.frequency * (time_values - t0)
        return self.amplitude * (1 - 2.0 * r ** 2) * np.exp(-(r ** 2))

    def show(self, pulse, time_values):
        """
        Show the ricker wavelet in a graph.

        Parameters
        ----------
        pulse : list
            Pulse of the wavelet.
        time_values : list
            Discretized values of time in seconds.
        """
        plt.plot(time_values, pulse)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tick_params()
        plt.show()
