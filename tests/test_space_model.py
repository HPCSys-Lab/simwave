from simwave import SpaceModel
import numpy as np
import pytest


class TestSpaceModel:

    @pytest.mark.parametrize('dimension', [(2), (3)])
    def test_properties(self, dimension):

        vel = 1500 * np.ones(shape=(101,)*dimension)
        den = 15 * np.ones(shape=(101,)*dimension)
        bbox = (0, 1000) * dimension
        spacing = (10, ) * dimension

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel,
            density_model=den,
            space_order=4,
            dtype=np.float32
        )

        space_model.dt = 0.001

        assert space_model.bounding_box == bbox
        assert space_model.grid_spacing == spacing
        assert np.array_equal(space_model.velocity_model, vel)
        assert np.array_equal(space_model.density_model, den)
        assert space_model.space_order == 4
        assert space_model.dimension == dimension
        assert space_model.dt == space_model.dtype(0.001)
        assert space_model.dtype == np.float32
        assert space_model.damping_length == (0.0,) * dimension * 2
        assert space_model.boundary_condition == ('none',) * dimension * 2
        assert space_model.damping_polynomial_degree == 3
        assert space_model.damping_alpha == 0.001

    @pytest.mark.parametrize(
        'dimension, bbox, spacing, shape', [
            (2, (0, 1000, 0, 1000), (10, 10), (101, 101)),
            (2, (100., 1000, 0, 1250.5), (10, 10), (91, 126)),
            (2, (-100, 1000, 500, 1000), (20, 10), (56, 51)),
            (3, (0, 1000, 0, 1000, 0, 1000.0), (5, 10, 20), (201, 101, 51)),
            (3, (10, 100, -10, 100, 0.0, 100.0), (5, 5, 5), (19, 23, 21)),
        ]
    )
    def test_shape(self, dimension, bbox, spacing, shape):

        vel = 1500 * np.ones(shape=(50,)*dimension)

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel
        )

        assert space_model.shape == shape

    @pytest.mark.parametrize(
        'dimension, damping_length, space_order, extended_shape', [
            (2, 100, 2, (123, 123)),
            (2, 100, 4, (125, 125)),
            (2, 150, 16, (147, 147)),
            (2, (150, 100, 150, 100), 8, (134, 134)),
            (2, (150, 100, 0, 100), 8, (134, 119)),
            (3, 100, 2, (123, 123, 123)),
            (3, 100, 4, (125, 125, 125)),
            (3, 150, 16, (147, 147, 147)),
            (3, (150, 100, 150, 100, 150, 100), 8, (134, 134, 134)),
            (3, (150, 100, 0, 100, 0, 150), 8, (134, 119, 124))
        ]
    )
    def test_extended_shape(self, dimension, damping_length,
                            space_order, extended_shape):

        vel = 1500 * np.ones(shape=(50,)*dimension)
        bbox = (0, 1000) * dimension
        spacing = (10, ) * dimension

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel,
            space_order=space_order
        )

        space_model.config_boundary(damping_length=damping_length)

        assert space_model.extended_shape == extended_shape

    @pytest.mark.parametrize('dimension', [(2), (3)])
    def test_grid(self, dimension):

        vel = 1500 * np.ones(shape=(50,)*dimension)
        bbox = (0, 1000) * dimension
        spacing = (10, ) * dimension

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel
        )

        assert space_model.grid.shape == space_model.shape
        assert space_model.grid.dtype == space_model.dtype

    @pytest.mark.parametrize(
        'dimension, damping_length, nbl', [
            (2, None, (0, 0, 0, 0)),
            (3, None, (0, 0, 0, 0, 0, 0)),
            (2, 120, (12, 12, 12, 12)),
            (3, 90, (9, 9, 9, 9, 9, 9)),
            (2, (50, 60, 75, 80), (5, 6, 7, 8)),
            (3, (0, 10, 8, 50, 20, 30), (0, 1, 0, 5, 2, 3))
        ]
    )
    def test_nbl(self, dimension, damping_length, nbl):

        vel = 1500 * np.ones(shape=(50,)*dimension)
        bbox = (0, 1000) * dimension
        spacing = (10,) * dimension

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel
        )

        if damping_length is not None:
            space_model.config_boundary(damping_length=damping_length)

        assert space_model.nbl == nbl

    @pytest.mark.parametrize(
        'time_order, space_order, coeffs', [
            (2, 2, [-2., 1.]),
            (2, 4, [-2.5, 1.3333334, -0.08333334]),
            (2, 8, [-2.8472223e+00, 1.6e+00, -2.0e-01,
                    2.5396826e-02, -1.7857143e-03])
        ]
    )
    def test_fd_coefficients(self, time_order, space_order, coeffs):

        vel = 1500 * np.ones(shape=(50, 50))

        space_model = SpaceModel(
            bounding_box=(0, 500, 0, 500),
            grid_spacing=(10, 10),
            velocity_model=vel,
            space_order=space_order
        )

        assert np.allclose(
                   space_model.fd_coefficients,
                   np.asarray(coeffs, dtype=space_model.dtype)
               )

    @pytest.mark.parametrize(
        'dimension, space_order, halo_size', [
            (2, 2, (1, 1, 1, 1)),
            (2, 4, (2, 2, 2, 2)),
            (2, 20, (10, 10, 10, 10)),
            (3, 2, (1, 1, 1, 1, 1, 1)),
            (3, 8, (4, 4, 4, 4, 4, 4)),
            (3, 10, (5, 5, 5, 5, 5, 5))
        ]
    )
    def test_halo_size(self, dimension, space_order, halo_size):

        vel = 1500 * np.ones(shape=(50,)*dimension)
        bbox = (0, 500) * dimension
        spacing = (10,) * dimension

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel,
            space_order=space_order
        )

        assert space_model.halo_size == halo_size

    @pytest.mark.parametrize(
        'dimension, damping_length, boundary_condition, \
        damping_polynomial_degree, damping_alpha',
        [
            (2, 500, "none", 2, 0.1),
            (2, (50, 50, 40, 40), "null_neumann", 4, 0.001),
            (3, 100, "null_dirichlet", 4, 0.001),
            (2, 75, ("none", "null_neumann", "none", "null_dirichlet"),
             4, 0.001),
            (3, (5, 5, 5, 5, 6, 6), "null_dirichlet", 1, 0.001)
        ]
    )
    def test_config_boundary(self, dimension, damping_length,
                             boundary_condition, damping_polynomial_degree,
                             damping_alpha):

        vel = 1500 * np.ones(shape=(50,)*dimension)
        bbox = (0, 500) * dimension
        spacing = (10,) * dimension

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel
        )

        space_model.config_boundary(
            damping_length=damping_length,
            boundary_condition=boundary_condition,
            damping_polynomial_degree=damping_polynomial_degree,
            damping_alpha=damping_alpha
        )

        if isinstance(damping_length, (float, int)):
            assert space_model.damping_length == (damping_length,) \
                                                  * dimension * 2
        else:
            assert space_model.damping_length == damping_length

        if isinstance(boundary_condition, str):
            assert space_model.boundary_condition == (boundary_condition,) \
                                                      * dimension * 2
        else:
            assert space_model.boundary_condition == boundary_condition

        assert space_model.damping_alpha == damping_alpha

        assert space_model.damping_polynomial_degree == \
            damping_polynomial_degree

    @pytest.mark.parametrize(
        'dimension, damping_length, space_order', [
            (2, 100, 2),
            (2, 100, 4),
            (2, 150, 16),
            (2, (150, 100, 150, 100), 8),
            (2, (150, 100, 0, 100), 8),
            (3, 100, 2),
            (3, 100, 4),
            (3, 150, 16),
            (3, (150, 100, 150, 100, 150, 100), 8),
            (3, (150, 100, 0, 100, 0, 150), 8)
        ]
    )
    def test_padding(self, dimension, damping_length, space_order):

        vel = 1500 * np.ones(shape=(50,)*dimension)
        den = 15 * np.ones(shape=(50,)*dimension)
        bbox = (0, 1000) * dimension
        spacing = (10, ) * dimension

        space_model = SpaceModel(
            bounding_box=bbox,
            grid_spacing=spacing,
            velocity_model=vel,
            density_model=den,
            space_order=space_order
        )

        space_model.config_boundary(damping_length=damping_length)

        assert space_model.damping_mask.shape == space_model.extended_shape
        assert space_model.extended_grid.shape == space_model.extended_shape
        assert space_model.extended_velocity_model.shape == \
            space_model.extended_shape
        assert space_model.extended_density_model.shape == \
            space_model.extended_shape
