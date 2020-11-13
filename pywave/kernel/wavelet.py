import numpy as np
import matplotlib.pyplot as plt

class Wavelet():
    """
    Implement a wavelet for the source

    Parameters
    ----------
    frequency: float
        Peak frequency for the wavelet in Hz
    amplitude: float
        Amplitude of the wavelet
    time_values: list
        Discretized values of time in seconds
    """
    def __init__(self, frequency=None, amplitude=1, time_values=None):
        self.frequency = frequency
        self.amplitude = amplitude
        self.time_values = time_values

    def ricker(self):
        t0 = 1 / self.frequency
        r = (np.pi * self.frequency * (self.time_values - t0))
        return self.amplitude * (1-2.*r**2)*np.exp(-r**2)

    def show(self, pulse):
        plt.plot(self.time_values, pulse)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tick_params()
        plt.show()
