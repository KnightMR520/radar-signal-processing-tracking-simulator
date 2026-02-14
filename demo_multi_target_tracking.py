import numpy as np
import matplotlib.pyplot as plt

from processing.pipeline import process_iq_data
from processing.tracker import MultiTargetTracker
from visualization.range_doppler_plot import LiveRangeDopplerPlot


# -----------------------------
# Radar Parameters
# -----------------------------
num_pulses = 32
num_samples = 64
prf = 1000

t_slow = np.arange(num_pulses) / prf
fast_time = np.arange(num_samples)


# -----------------------------
# Plot Setup
# -----------------------------
plotter = LiveRangeDopplerPlot(
    prf=prf,
    num_pulses=num_pulses,
    num_samples=num_samples,
    wavelength=0.03,
    range_resolution=1.0,
    dynamic_range_dB=80
)

tracker = MultiTargetTracker()


# -----------------------------
# Target Definitions
# -----------------------------
targets = [
    {"range_bin": 15, "doppler_bin": -5},
    {"range_bin": 40, "doppler_bin": 6},
    {"range_bin": 25, "doppler_bin": 2},
]


# -----------------------------
# Main Loop
# -----------------------------
for frame in range(200):

    iq_data = np.zeros((num_pulses, num_samples), dtype=np.complex128)

    for tgt in targets:

        range_bin = tgt["range_bin"]
        doppler_bin = tgt["doppler_bin"]

        range_freq = range_bin / num_samples
        doppler_freq = doppler_bin * prf / num_pulses

        for p in range(num_pulses):
            iq_data[p, :] += (
                np.exp(1j * 2 * np.pi * range_freq * fast_time)
                * np.exp(1j * 2 * np.pi * doppler_freq * t_slow[p])
            )

        # Move targets slowly
        tgt["range_bin"] += np.random.choice([-1, 0, 1])
        tgt["doppler_bin"] += np.random.choice([-1, 0, 1])

        tgt["range_bin"] = np.clip(tgt["range_bin"], 5, num_samples - 5)
        tgt["doppler_bin"] = np.clip(tgt["doppler_bin"], -10, 10)

    # Process pipeline
    rd_map, magnitude_map, detection_mask = process_iq_data(iq_data)

    # Convert detections to measurement list
    detection_indices = np.argwhere(detection_mask)
    detections = []

    for d in detection_indices:
        doppler_idx, range_idx = d
        detections.append([range_idx, doppler_idx])

    # Tracker step
    tracks = tracker.step(detections)

    # Update plot
    plotter.update(
        magnitude_map,
        detections=detection_mask,
        title=f"Frame {frame}"
    )

    # Remove old text labels
    for txt in plotter.ax.texts:
        txt.remove()

    # Draw confirmed tracks only
    for track in tracks:
        if not track.confirmed:
            continue

        r = track.x[0]
        d = track.x[2]

        plotter.ax.text(
            r,
            d,
            f"ID {track.id}",
            color="white",
            fontsize=9,
            weight="bold"
        )

plt.show()
