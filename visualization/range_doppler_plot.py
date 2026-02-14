import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class LiveRangeDopplerPlot:
    def __init__(self,
                 dB_scale=True,
                 dynamic_range_dB=100,
                 wavelength=0.03,     # 10 GHz radar default
                 prf=1000,
                 num_pulses=32,
                 num_samples=64,
                 range_resolution=1.0  # meters per bin
                 ):

        self.dB_scale = dB_scale
        self.dynamic_range_dB = dynamic_range_dB

        self.wavelength = wavelength
        self.prf = prf
        self.num_pulses = num_pulses
        self.num_samples = num_samples
        self.range_resolution = range_resolution

        # --- Custom Dark Red -> Dark Blue Colormap ---
        self.cmap = LinearSegmentedColormap.from_list(
            "radar_cmap",
            ["#00008B", "#0000FF", "#800000"],  # dark blue -> blue -> dark red
            N=256
        )

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.im = None
        self.scatter = None

        plt.ion()

    def update(self, magnitude_map, detections=None, title="Range-Doppler Map"):
        eps = 1e-12

        # Convert to dB
        display_map = 10 * np.log10(magnitude_map + eps)

        # Normalize relative to peak
        peak = np.max(display_map)
        display_map = display_map - peak

        # Clip to dynamic range
        display_map = np.clip(display_map, -self.dynamic_range_dB, 0)

        if self.im is None:
            self.im = self.ax.imshow(
                display_map,
                aspect="auto",
                origin="lower",
                cmap="jet",                 # 👈 KEY CHANGE
                vmin=-self.dynamic_range_dB,
                vmax=0,
                interpolation="bicubic",    # 👈 smoother / less boxy
            )

            self.fig.colorbar(self.im, ax=self.ax, label="Power (dB)")

            self.scatter = self.ax.scatter(
                [], [],
                marker="o",
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
            )

        else:
            self.im.set_data(display_map)

        # Update detections
        if detections is not None:
            det_indices = np.argwhere(detections)
            if len(det_indices) > 0:
                doppler_idx, range_idx = det_indices.T
                offsets = np.column_stack((range_idx, doppler_idx))
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
