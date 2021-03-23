# test quality of numerical solution
from scipy.special import hankel2
import numpy.linalg as la
import numpy as np
import pywave
import pytest


def _create_space_model(bbox, spacing, vel, order, dt):
    space_model = pywave.SpaceModel(
        bounding_box=bbox,
        grid_spacing=spacing,
        velocity_model=vel * np.ones((100, 100), dtype=np.float64),
        space_order=order
    )
    space_model.dt = dt
    return space_model


def _acquisition(space_model, time_model, source, receiver, freq):
    source = pywave.Source(space_model, coordinates=source)
    receiver = pywave.Receiver(space_model, coordinates=receiver)
    ricker = pywave.RickerWavelet(freq, time_model)

    return source, receiver, ricker


def analytical_solution(space_model, freq, src, recs):
    time_model = pywave.TimeModel(space_model=space_model, t0=0, tf=3000)
    ricker = pywave.RickerWavelet(freq, time_model)
    # Ricker's FFT
    nf = int(time_model.timesteps / 2 + 1)
    df = 1 / time_model.tf
    frequencies = df * np.arange(nf)
    q = np.fft.fft(ricker.values)[:nf]
    # Coordinates
    xg = np.array([rec[0] for rec in recs])
    zg = np.array([rec[1] for rec in recs])
    r = np.sqrt((xg - src[0]) ** 2 + (zg - src[1]) ** 2)
    # Analytical solution
    k = 2 * np.pi * frequencies / np.unique(space_model.velocity_model)
    u = np.zeros((nf), dtype=complex)
    u[1:-1] = hankel2(0, k[1:-1][None, :] * r[:, None])
    ui = np.fft.ifft(- 1j * np.pi * u * q, time_model.timesteps)
    # Scale correctly
    # ui = 1 / (2 * np.pi) * np.real(ui) * space_model.grid_spacing[0] ** 2
    ui = 1 / (2 * np.pi) * np.real(ui)

    return ui


def accuracy(spacing, bbox, order, dt, t0, tf, c, f0, src, rec):
    space_model = _create_space_model(bbox, spacing, c, order, dt)

    time_model = pywave.TimeModel(space_model=space_model, t0=t0, tf=tf)

    source, receiver, wavelet = _acquisition(
        space_model, time_model, src, rec, f0
    )

    solver = pywave.Solver(space_model, time_model, source, receiver, wavelet)

    # numerical solution
    # u_num = solver.forward()[-1].flatten()
    u_num = solver.forward()[-1].flatten() / spacing[0] ** 2

    # analytical solution
    u_exact = analytical_solution(
                 space_model, f0, src[0], rec
              ).flatten()[:time_model.timesteps]

    # compare solutions
    return la.norm(u_num - u_exact) / np.sqrt(u_num.size)


@pytest.mark.parametrize(
    "spacing, bbox, order, dt, t0, tf, c, f0, src, rec, tol",
    [
        (
            (0.5, 0.5), (-40, 440, -40, 440), 8, 0.1, 0, 150.075,
            1.5, 0.09, [(200, 200)], [(260, 260)], 1e-4
        )
    ]
)
def test_accuracy(spacing, bbox, order, dt, t0, tf, c, f0, src, rec, tol):
    assert accuracy(spacing, bbox, order, dt, t0, tf, c, f0, src, rec) < tol


@pytest.mark.parametrize(
    "spacing, bbox, order, timesteps, t0, tf, c, \
     f0, src, rec, time_convergence",
    [
        (
            (0.5, 0.5), (-40, 440, -40, 440), 8, [0.1, 0.075, 0.04, 0.025],
            0, 150.075, 1.5, 0.09, [(200, 200)], [(260, 260)], 1.7
        )
    ]
)
def test_convergence_in_time(spacing, bbox, order, timesteps, t0, tf, c, f0,
                             src, rec, time_convergence):
    accs = []
    for dt in timesteps:
        accs.append(
            accuracy(spacing, bbox, order, dt, t0, tf, c, f0, src, rec)
        )

    assert np.poly1d(np.polyfit(np.log(timesteps),
                     np.log(accs), 1))[1] > time_convergence


@pytest.mark.parametrize(
    "spacings, bbox, space_orders, dt, t0, tf, c, f0, src, rec, space_rates",
    [
        (
            [2.0, 2.5, 4.0], (-40, 440, -40, 440), [2, 4, 6, 8, 10], 0.025,
            0, 150.075, 1.5, 0.09, [(200, 200)], [(260, 260)],
            [1.7, 4, 6, 7.7, 8.7]
        )
    ]
)
def test_convergence_in_space(spacings, bbox, space_orders, dt, t0, tf, c,
                              f0, src, rec, space_rates):
    accs = {}
    conv_rates = []
    for order in space_orders:
        accs[order] = {}
        for spacing in spacings:
            accs[order][spacing] = accuracy(
                tuple([spacing, spacing]), bbox,
                order, dt, t0, tf, c, f0, src, rec
            )
        step = list(accs[order].keys())
        error = list(accs[order].values())
        conv_rates.append(
            np.poly1d(np.polyfit(np.log(step), np.log(error), 1))[1]
        )
    for rate, min_rate in zip(conv_rates, space_rates):
        assert rate > min_rate
