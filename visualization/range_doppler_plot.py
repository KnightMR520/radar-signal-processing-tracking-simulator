import numpy as np
import matplotlib.pyplot as plt


class LiveRangeDopplerPlot:
    def __init__(self, dB_scale=True):
        self.dB_scale = dB_scale

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.im = None
        self.scatter = None

        plt.ion()

    def update(self, magnitude_map, detections=None, title="Range-Doppler Map"):
        eps = 1e-12

        if self.dB_scale:
            display_map = 10 * np.log10(magnitude_map + eps)
        else:
            display_map = magnitude_map

        if self.im is None:
            # First frame setup
            self.im = self.ax.imshow(
                display_map,
                aspect="auto",
                origin="lower",
            )
            self.fig.colorbar(self.im, ax=self.ax, label="Power (dB)")

            if detections is not None:
                self.scatter = self.ax.scatter(
                    [], [],
                    marker="o",
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.5,
                )

        else:
            # Update image data
            self.im.set_data(display_map)

        # Update detections
        if detections is not None and self.scatter is not None:
            det_indices = np.argwhere(detections)
            if len(det_indices) > 0:
                doppler_idx, range_idx = det_indices.T
                offsets = np.column_stack((range_idx, doppler_idx))
                self.scatter.set_offsets(offsets)
            else:
                self.scatter.set_offsets(np.empty((0, 2)))

        self.ax.set_title(title)
        self.ax.set_xlabel("Range Bin")
        self.ax.set_ylabel("Doppler Bin")

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.show()
