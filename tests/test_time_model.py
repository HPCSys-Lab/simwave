from simwave import SpaceModel, TimeModel
import numpy as np
import pytest


class TestTimeModel:

    def test_properties(self):

        vel = 1500 * np.ones(shape=(10, 10))

        space_model = SpaceModel(
            bounding_box=(0, 100, 0, 100),
            grid_spacing=(10, 10),
            velocity_model=vel,
        )

        time_model = TimeModel(
            space_model=space_model,
            tf=1.0,
            t0=0.0,
            saving_stride=1
        )

        time_model.dt = 0.001

        assert time_model.space_model == space_model
        assert time_model.tf == 1.0
        assert time_model.t0 == 0.0
        assert time_model.saving_stride == 1
        assert time_model.dt == space_model.dtype(0.001)

    @pytest.mark.parametrize(
        'dt, tf, t0, saving_stride, timesteps', [
            (0.001, 1.0, 0.0, 0, 1001),
            (0.001, 1.0, 0.5, 0, 501),
            (0.002, 2.0, 0.0, 0, 1001),
            (0.001, 1.0, 0.0, 1, 1001),
            (0.001, 1.0, 0.5, 1, 501),
            (0.002, 2.0, 0.0, 1, 1001),
            (0.001, 1.0, 0.0, 2, 1001),
            (0.001, 1.0, 0.5, 6, 505),
            (0.002, 2.0, 0.0, 3, 1003),
        ]
    )
    def test_timesteps(self, dt, tf, t0, saving_stride, timesteps):

        vel = 1500 * np.ones(shape=(10, 10))

        space_model = SpaceModel(
            bounding_box=(0, 100, 0, 100),
            grid_spacing=(10, 10),
            velocity_model=vel,
        )

        time_model = TimeModel(
            space_model=space_model,
            tf=tf,
            t0=t0,
            saving_stride=saving_stride
        )

        time_model.dt = dt

        assert time_model.timesteps == timesteps
