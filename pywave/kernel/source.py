import numpy as np
import matplotlib.pyplot as plt

class RickerSource():
    """
    Implement a ricker source

    Parameters
    ----------
    frequency: float
        Peak frequency for Ricker wavelet in Hz
    amplitude: float
        Amplitude of the wavelet
    time_values: list
        Discretized values of time in seconds
    """
    def __init__(self, frequency=None, amplitude=1, time_values=None):
        self.frequency = frequency
        self.amplitude = amplitude
        self.time_values = time_values

    def wavelet(self):
        t0 = 1 / self.frequency        
        r = (np.pi * self.frequency * (self.time_values - t0))
        return self.amplitude * (1-2.*r**2)*np.exp(-r**2)

    def show(self):
        plt.plot(self.time_values, self.wavelet())
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tick_params()
        plt.show()
