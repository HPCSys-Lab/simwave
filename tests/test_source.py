from pywave import SpaceModel, TimeModel, Source
import numpy as np
import pytest


class TestSource:

    @pytest.mark.parametrize('dimension, bbox, spacing, coords, grid_positions', [
        # 2D tests
        (2, (0,5120,0,5120), (10,10), (0, 512), (0, 51.2)),
        (2, (0,5120,0,5120), (10,5), (120, 500), (12, 100)),
        # 3D tests
        (3, (0,500,20,500,0,200), (20,20,20), (10,20,100), (0.5, 0.0, 5.0))
    ])
    def test_conversion_coords_to_points(self, dimension, bbox,
                                         spacing, coords, grid_positions):

        shape = (50,)*dimension

        # Velocity model
        vel = np.zeros(shape=shape, dtype=np.float32)
        vel[:] = 1500.0

        space_model = SpaceModel(
            bbox=bbox,
            grid_spacing=spacing,
            velocity_model=vel
        )

        source = Source(space_model, coordinates=[], window_radius=4)
        source.add(coords)

        assert source.grid_positions == [grid_positions]
