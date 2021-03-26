from pywave import SpaceModel, Source
import numpy as np
import pytest


class TestSource:

    @pytest.mark.parametrize(
        'dimension, coords', [
            (2, [(0, 25)]),
            (2, [(0, 25), [250, 250]]),
            (3, [(0.5, 50.8, 500)])
        ]
    )
    def test_properties(self, dimension, coords):
        shape = (50,) * dimension
        bbox = (0, 500) * dimension
        spacing = (10, ) * dimension

        # Velocity model
        vel = 1500 * np.ones(shape=shape)

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel
        )

        source = Source(
            space_model,
            coordinates=coords,
            window_radius=8
        )

        assert source.space_model == space_model
        assert source.window_radius == 8
        assert source.count == len(coords)
        assert np.array_equal(
            source.coordinates,
            space_model.dtype(coords)
        )

    @pytest.mark.parametrize(
        'dimension, bbox, spacing, coords, grid_positions', [
            # 2D tests
            (2, (0, 5120, 0, 5120), (10, 10), (0, 512), (0, 51.2)),
            (2, (0, 5120, 0, 5120), (10, 5), (120, 500), (12, 100)),
            # 3D tests
            (3, (0, 500, 20, 500, 0, 200), (20, 20, 20),
                (10, 20, 100), (0.5, 0.0, 5.0))
        ]
    )
    def test_conversion_coords_to_points(self, dimension, bbox,
                                         spacing, coords, grid_positions):

        shape = (50,)*dimension

        # Velocity model
        vel = np.zeros(shape=shape, dtype=np.float32)
        vel[:] = 1500.0

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel
        )

        source = Source(space_model, coordinates=coords, window_radius=4)

        assert np.array_equal(
                    source.grid_positions,
                    np.asarray([grid_positions], dtype=space_model.dtype)
               )

    @pytest.mark.parametrize(
        'dimension, damping_length, space_order, coords, expected_position', [
            (2, 500, 2, (0, 0), [(51, 51)]),
            (2, 0, 2, (0, 0), [(1, 1)]),
            (2, 0, 4, (250, 100), [(27, 12)]),
            (2, 50, 16, (255, 255), [(38.5, 38.5)]),
            (3, (50, 0, 50, 0, 0, 0), 16, (255, 255, 0), [(38.5, 38.5, 8)]),
            (3, 2, 4, (250, 100, 100), [(27, 12, 12)]),
        ]
    )
    def test_adjusted_grid_positions(self, dimension, damping_length,
                                     space_order, coords, expected_position):

        shape = (50,) * dimension
        bbox = (0, 500) * dimension
        spacing = (10, ) * dimension

        # Velocity model
        vel = 1500 * np.ones(shape=shape)

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel,
            space_order=space_order
        )

        space_model.config_boundary(damping_length=damping_length)

        source = Source(space_model, coordinates=coords)

        assert np.array_equal(
            source.adjusted_grid_positions,
            space_model.dtype(expected_position)
        )

    @pytest.mark.parametrize(
        'dimension, window_radius, damping_length, \
        space_order, coords, points, values', [
            (
                2, 1, 50, 2, (250, 250), [30, 32, 30, 32],
                [-1.9556131e-08,  1.0000001e+00, -1.9556131e-08,
                 -1.9556131e-08, 1.0000001e+00, -1.9556131e-08]
            ),
            (
                3, 4, 0, 4, (255, 250, 100), [24, 31, 23, 31,  8, 16],
                [
                    -5.1983586e-03, 3.6357623e-02, -1.3933791e-01,
                    6.0838646e-01, 6.0838646e-01, -1.3933791e-01,
                    3.6357623e-02, -5.1983586e-03, 3.1170111e-10,
                    -3.7239498e-10, 1.2889303e-08, -2.3163784e-08,
                    9.9999994e-01, -2.3163784e-08, 1.2889303e-08,
                    -3.7239498e-10, 3.1170111e-10, 3.1170111e-10,
                    -3.7239498e-10, 1.2889303e-08, -2.3163784e-08,
                    9.9999994e-01, -2.3163784e-08, 1.2889303e-08,
                    -3.7239498e-10, 3.1170111e-10
                ]
            )
        ]
    )
    def test_interpolated_points_and_values(self, dimension, window_radius,
                                            damping_length, space_order,
                                            coords, points, values):

        shape = (50,) * dimension
        bbox = (0, 500) * dimension
        spacing = (10, ) * dimension

        # Velocity model
        vel = 1500 * np.ones(shape=shape)

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel,
            space_order=space_order
        )

        space_model.config_boundary(damping_length=damping_length)

        source = Source(
            space_model,
            coordinates=coords,
            window_radius=window_radius
        )

        p, v = source.interpolated_points_and_values

        assert np.array_equal(p, np.asarray(points))
        assert np.array_equal(v, space_model.dtype(values))
