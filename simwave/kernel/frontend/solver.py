import numpy as np
from simwave.kernel.backend.middleware import Middleware


class Solver:
    """
    Acoustic solver for the simulation.

    Parameters
    ----------
    space_model : SpaceModel
        Space model object.
    time_model: TimeModel
        Time model object.
    sources : Source
        Source object.
    receivers : Receiver
        Receiver object.
    wavelet : Wavelet
        Wavelet object.
    compiler : Compiler
        Backend compiler object.
    """
    def __init__(self, space_model, time_model, sources,
                 receivers, wavelet, compiler=None):

        self._space_model = space_model
        self._time_model = time_model
        self._sources = sources
        self._receivers = receivers
        self._wavelet = wavelet
        self._compiler = compiler

        # create a middleware to communicate with backend
        self._middleware = Middleware(compiler=self.compiler)

    @property
    def space_model(self):
        """Space model object."""
        return self._space_model

    @property
    def time_model(self):
        """Time model object."""
        return self._time_model

    @property
    def sources(self):
        """Source object."""
        return self._sources

    @property
    def receivers(self):
        """Receiver object."""
        return self._receivers

    @property
    def wavelet(self):
        """Wavelet object."""
        return self._wavelet

    @property
    def compiler(self):
        """Compiler object."""
        return self._compiler

    @property
    def snapshot_indexes(self):
        """List of snapshot indexes (wavefields to be saved)."""

        # if saving_stride is 0, only saves the last timestep
        if self.time_model.saving_stride == 0:
            return [self.time_model.time_indexes[-1]]

        snap_indexes = list(
            range(
                self.time_model.time_indexes[0],
                self.time_model.timesteps,
                self.time_model.saving_stride
            )
        )

        return snap_indexes

    @property
    def num_snapshots(self):
        """Number of snapshots (wavefields to be saved)."""
        return len(self.snapshot_indexes)

    @property
    def shot_record(self):
        """Return the shot record array."""
        u_recv = np.zeros(
            shape=(self.time_model.timesteps, self.receivers.count),
            dtype=self.space_model.dtype
        )

        return u_recv

    @property
    def u_full(self):
        """Return the complete grid (snapshots, nz. nz [, ny])."""

        # add 2 halo snapshots (second order in time)
        snapshots = self.num_snapshots + 2

        # define the final shape (snapshots + domain)
        shape = (snapshots,) + self.space_model.extended_shape

        return np.zeros(shape, dtype=self.space_model.dtype)

    def forward(self):
        """
        Run the forward propagator.

        Returns
        ----------
        ndarray
            Full wavefield with snapshots.
        ndarray
            Shot record.
        """

        src_points, src_values, src_offsets = \
            self.sources.interpolated_points_and_values
        rec_points, rec_values, rec_offsets = \
            self.receivers.interpolated_points_and_values

        u_full, recv = self._middleware.exec(
            operator='forward',
            u_full=self.u_full,
            velocity_model=self.space_model.extended_velocity_model,
            density_model=self.space_model.extended_density_model,
            damping_mask=self.space_model.damping_mask,
            wavelet=self.wavelet.values,
            wavelet_size=self.wavelet.timesteps,
            wavelet_count=self.wavelet.num_sources,
            second_order_fd_coefficients=self.space_model.fd_coefficients(2),
            first_order_fd_coefficients=self.space_model.fd_coefficients(1),
            boundary_condition=self.space_model.boundary_condition,
            src_points_interval=src_points,
            src_points_interval_size=len(src_points),
            src_points_values=src_values,
            src_points_values_offset=src_offsets,
            src_points_values_size=len(src_values),
            rec_points_interval=rec_points,
            rec_points_interval_size=len(rec_points),
            rec_points_values=rec_values,
            rec_points_values_offset=rec_offsets,
            rec_points_values_size=len(rec_values),
            shot_record=self.shot_record,
            num_sources=self.sources.count,
            num_receivers=self.receivers.count,
            grid_spacing=self.space_model.grid_spacing,
            saving_stride=self.time_model.saving_stride,
            dt=self.time_model.dt,
            begin_timestep=1,
            end_timestep=self.time_model.timesteps,
            space_order=self.space_model.space_order,
            num_snapshots=self.u_full.shape[0]
        )

        # remove time halo region
        u_full = self.time_model.remove_time_halo_region(u_full)

        # remove spatial halo region
        u_full = self.space_model.remove_halo_region(u_full)

        return u_full, recv
