import numpy as np
import matplotlib.pyplot as plt

from processing.pipeline import process_iq_data


# -------------------------
# Radar Parameters
# -------------------------
num_pulses = 32
num_samples = 64
prf = 1000  # Pulse Repetition Frequency (Hz)

t_slow = np.arange(num_pulses) / prf
fast_time = np.arange(num_samples)


# -------------------------
# Setup Live Plot
# -------------------------
plt.ion()
fig, ax = plt.subplots()

# Empty initial map
dummy_map = np.zeros((num_pulses, num_samples))
im = ax.imshow(
    dummy_map,
    aspect="auto",
    origin="lower",
    cmap="viridis"
)

scatter = ax.scatter([], [], facecolors="none", edgecolors="r")

ax.set_xlabel("Range Bin")
ax.set_ylabel("Doppler Bin")
title = ax.set_title("Live Range-Doppler")

plt.colorbar(im, ax=ax)


# -------------------------
# Live Loop
# -------------------------
for doppler_bin in range(-8, 9):

    # Reset IQ data every frame
    iq_data = np.zeros((num_pulses, num_samples), dtype=np.complex128)

    # ----- Synthetic Target -----
    range_bin = 25
    range_freq = range_bin / num_samples
    doppler_freq = doppler_bin * prf / num_pulses

    for p in range(num_pulses):
        iq_data[p, :] = (
            np.exp(1j * 2 * np.pi * range_freq * fast_time)
            * np.exp(1j * 2 * np.pi * doppler_freq * t_slow[p])
        )

    # ----- Process -----
    rd_map, magnitude_map, detections = process_iq_data(iq_data)

    # ----- Update Image -----
    im.set_data(magnitude_map)
    im.set_clim(vmin=np.min(magnitude_map), vmax=np.max(magnitude_map))

    # ----- Update Detections -----
    if len(detections) > 0:
        det = np.array(detections)
        scatter.set_offsets(np.c_[det[:, 1], det[:, 0]])
    else:
        scatter.set_offsets([])

    title.set_text(f"Doppler Bin {doppler_bin}")

    plt.pause(0.3)

plt.ioff()
plt.show()
