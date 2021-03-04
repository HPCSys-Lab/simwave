import numpy as np
import segyio


def read_2D_segy(file):
    """
    Build a 2D velocity model from a SEG-Y format.
    It uses the 'segyio' from https://github.com/equinor/segyio

    Parameters
    ----------
    file : str
        Path to the velocity/density model file.

    Returns
    ----------
    ndarray
        2D velocity model.
    """
    with segyio.open(file, ignore_geometry=True) as f:
        n_samples = len(f.samples)
        n_traces = len(f.trace)
        data = np.zeros(shape=(n_samples, n_traces), dtype=np.float32)
        index = 0
        for trace in f.trace:
            data[:, index] = trace
            index += 1

        return data
