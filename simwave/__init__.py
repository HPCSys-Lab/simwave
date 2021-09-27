from simwave.io import read_2D_segy

from simwave.kernel import (
    Compiler,
    SpaceModel,
    TimeModel,
    Source,
    Receiver,
    Wavelet,
    RickerWavelet,
    MultiWavelet,
    Solver
)

from simwave.plots import (
    plot_wavefield,
    plot_shotrecord,
    plot_velocity_model,
    plot_wavelet
)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__all__ = [
    "Compiler",
    "SpaceModel",
    "TimeModel",
    "Source",
    "Receiver",
    "Wavelet",
    "RickerWavelet",
    "MultiWavelet",
    "Solver",
    "plot_wavefield",
    "plot_shotrecord",
    "plot_velocity_model",
    "plot_wavelet",
    "read_2D_segy"
]
