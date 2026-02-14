# visualization/range_doppler_plot.py

import numpy as np
import matplotlib.pyplot as plt


class LiveRangeDopplerPlot:
    """
    Displays a Range–Doppler magnitude map with correctly scaled physical axes.

    Assumes magnitude_map shape is:
        (num_pulses, num_samples)
    where axis-0 is Doppler bins (fftshifted output) and axis-1 is Range bins.

    - x axis: Range (m)
    - y axis: Velocity (m/s)
    """

    def __init__(
        self,
        dB_scale: bool = True,
        dynamic_range_dB: float = 100,
        wavelength: float = 0.03,
        prf: float = 1000,
        num_pulses: int = 32,
        num_samples: int = 64,
        range_resolution: float = 1.0,  # meters per range bin
    ):
        self.dB_scale = bool(dB_scale)
        self.dynamic_range_dB = float(dynamic_range_dB)

        self.wavelength = float(wavelength)
        self.prf = float(prf)
        self.num_pulses = int(num_pulses)
        self.num_samples = int(num_samples)
        self.range_resolution = float(range_resolution)

        # Build physical axes
        # Range bins (0..N-1) -> meters
        self.range_axis_m = np.arange(self.num_samples, dtype=float) * self.range_resolution

        # Doppler bins (fftshifted) -> velocity (m/s)
        doppler_freqs = np.fft.fftshift(np.fft.fftfreq(self.num_pulses, d=1.0 / self.prf))
        self.velocity_axis_mps = doppler_freqs * self.wavelength / 2.0

        # Edges for correct extent (bin edges, not centers)
        # Range edges in meters
        x0 = 0.0
        x1 = self.num_samples * self.range_resolution
        # Velocity edges in m/s (compute edges from centers)
        # If centers are v[k], approximate constant spacing:
        if self.num_pulses > 1:
            dv = float(self.velocity_axis_mps[1] - self.velocity_axis_mps[0])
        else:
            dv = 1.0
        y0 = float(self.velocity_axis_mps[0] - dv / 2.0)
        y1 = float(self.velocity_axis_mps[-1] + dv / 2.0)

        # extent maps array indices to physical coordinates:
        # extent = [xmin, xmax, ymin, ymax]
        self.extent = [x0, x1, y0, y1]

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.im = None
        self.scatter = None

        plt.ion()

    def _to_display_db(self, magnitude_map: np.ndarray) -> np.ndarray:
        eps = 1e-12
        display_map = 10.0 * np.log10(magnitude_map + eps)

        # Normalize relative to peak
        peak = float(np.max(display_map))
        display_map = display_map - peak

        # Clip to dynamic range
        display_map = np.clip(display_map, -self.dynamic_range_dB, 0.0)
        return display_map

    def bins_to_physical(self, range_bins: np.ndarray, doppler_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert bin indices (can be float) to physical coordinates at bin centers.

        range_bins: 0..num_samples-1
        doppler_bins: 0..num_pulses-1  (fftshifted bin index)

        Returns:
            range_m, vel_mps
        """
        range_bins = np.asarray(range_bins, dtype=float)
        doppler_bins = np.asarray(doppler_bins, dtype=float)

        # Range: bin center
        range_m = (range_bins + 0.5) * self.range_resolution

        # Velocity: interpolate for fractional doppler bins
        doppler_bins = np.clip(doppler_bins, 0.0, self.num_pulses - 1.0)
        vel_mps = np.interp(doppler_bins, np.arange(self.num_pulses, dtype=float), self.velocity_axis_mps)

        return range_m, vel_mps

    def update(self, magnitude_map: np.ndarray, detections: np.ndarray | None = None, title: str = "Range-Doppler Map"):
        """
        magnitude_map is expected shape (num_pulses, num_samples).
        detections is boolean mask same shape.
        """
        if self.dB_scale:
            display_map = self._to_display_db(magnitude_map)
        else:
            display_map = magnitude_map

        if self.im is None:
            self.im = self.ax.imshow(
                display_map,
                aspect="auto",
                origin="lower",
                extent=self.extent,            # ✅ physical axes
                cmap="jet",
                vmin=-self.dynamic_range_dB if self.dB_scale else None,
                vmax=0.0 if self.dB_scale else None,
                interpolation="bicubic",
            )

            self.fig.colorbar(self.im, ax=self.ax, label="Power (dB)" if self.dB_scale else "Magnitude")

            self.scatter = self.ax.scatter(
                [], [],
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
            )
        else:
            self.im.set_data(display_map)

        # Update detections (white circles) in physical coordinates
        if detections is not None:
            det_indices = np.argwhere(detections)  # rows: [doppler_idx, range_idx]
            if det_indices.size > 0:
                doppler_idx, range_idx = det_indices.T

                x_m, y_v = self.bins_to_physical(range_idx, doppler_idx)
                offsets = np.column_stack((x_m, y_v))
                self.scatter.set_offsets(offsets)
            else:
                self.scatter.set_offsets(np.empty((0, 2)))

        self.ax.set_title(title)
        self.ax.set_xlabel("Range (m)")
        self.ax.set_ylabel("Velocity (m/s)")

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.show()
