# demo/demo_multi_target_tracking.py

import numpy as np
import matplotlib.pyplot as plt

from processing.pipeline import process_iq_data
from tracking.tracker_manager import TrackerManager
from visualization.range_doppler_plot import LiveRangeDopplerPlot


# -----------------------------
# Radar Parameters
# -----------------------------
num_pulses = 32
num_samples = 64
prf = 1000

wavelength = 0.03          # meters (10 GHz radar)
range_resolution = 1.0     # meters per range bin (USED FOR SYNTH ONLY here)
dt = 1 / prf

t_slow = np.arange(num_pulses) / prf
fast_time = np.arange(num_samples)

# -----------------------------
# CFAR window sizes (must match pipeline defaults or pass explicitly)
# -----------------------------
guard_cells = (1, 1)
training_cells = (4, 4)
g_r, g_d = guard_cells
t_r, t_d = training_cells

# CFAR valid region margin in bins (anything closer than this to an edge can't be CUT tested)
margin_r = t_r + g_r
margin_d = t_d + g_d


# -----------------------------
# Plot Setup
# NOTE: Your LiveRangeDopplerPlot currently plots in BIN coordinates:
#   x-axis = range_bin index, y-axis = doppler_bin index
# (no extent -> labels "Range(m)" / "Velocity(m/s)" are just labels right now)
# -----------------------------
plotter = LiveRangeDopplerPlot(
    prf=prf,
    num_pulses=num_pulses,
    num_samples=num_samples,
    wavelength=wavelength,
    range_resolution=range_resolution,
    dynamic_range_dB=80
)

# Bin-space tracker
tracker = TrackerManager(
    dt_default=1.0,
    gate_mahalanobis=16.0,
    max_misses=6,                   # survive edge blink / occasional misses
    confirm_hits=2,
    measurement_R=np.diag([4.0, 4.0])
)

# -----------------------------
# Target Definitions (synthetic truth) in BIN space
# -----------------------------
targets = [
    {"range_bin": 15, "doppler_bin": 8,  "amp": 1.0},
    {"range_bin": 40, "doppler_bin": 26, "amp": 1.0},
    {"range_bin": 25, "doppler_bin": 20, "amp": 1.0},
]


# -----------------------------
# Main Loop
# -----------------------------
for frame in range(200):

    iq_data = np.zeros((num_pulses, num_samples), dtype=np.complex128)

    # --- Generate targets ---
    for tgt in targets:
        range_bin = int(tgt["range_bin"])
        doppler_bin = int(tgt["doppler_bin"])
        amp = float(tgt["amp"])

        # Range synthesis uses "fast time" normalized frequency
        range_freq = range_bin / num_samples

        # Doppler synthesis: doppler_bin is an FFTSHIFTED bin index [0..N-1]
        doppler_freq = (doppler_bin - num_pulses // 2) * prf / num_pulses

        for p in range(num_pulses):
            iq_data[p, :] += amp * (
                np.exp(1j * 2 * np.pi * range_freq * fast_time)
                * np.exp(1j * 2 * np.pi * doppler_freq * t_slow[p])
            )

        # Random walk in bin-space
        tgt["range_bin"] += np.random.choice([-1, 0, 1])
        tgt["doppler_bin"] += np.random.choice([-1, 0, 1])

        # KEEP TARGETS IN CFAR-VALID REGION (prevents edge disappear)
        tgt["range_bin"] = int(np.clip(tgt["range_bin"], margin_r, num_samples - 1 - margin_r))
        tgt["doppler_bin"] = int(np.clip(tgt["doppler_bin"], margin_d, num_pulses - 1 - margin_d))

    # Optional noise
    noise_power = 0.01
    iq_data += np.sqrt(noise_power / 2) * (
        np.random.randn(*iq_data.shape) + 1j * np.random.randn(*iq_data.shape)
    )

    # --- Process pipeline ---
    # IMPORTANT: your pipeline already suppresses to local max per cluster
    rd_map, magnitude_map, detection_mask = process_iq_data(
        iq_data,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=1e-3,
        use_power=True
    )

    # --- Build measurements from detections (bin space) ---
    det_idxs = np.argwhere(detection_mask)  # [doppler_idx, range_idx]

    measurements = []
    if det_idxs.size > 0:
        scores = magnitude_map[detection_mask]
        order = np.argsort(scores)[::-1]

        # For the demo we know we have 3 true targets
        K = 3
        det_idxs = det_idxs[order[:K]]

        for doppler_idx, range_idx in det_idxs:
            measurements.append(np.array([float(range_idx), float(doppler_idx)], dtype=float))

    # --- Tracking step ---
    tracks = tracker.step(measurements, dt=1.0)

    # --- Plot update (bin-space image + bin-space circles) ---
    plotter.update(
        magnitude_map,
        detections=detection_mask,
        title=f"Frame {frame}"
    )

    # Clear old ID labels
    for txt in list(plotter.ax.texts):
        txt.remove()

    # Draw confirmed tracks only
    for trk in tracks:
        if not trk.confirmed:
            continue

        # Track state in BIN space
        range_bin_est = float(trk.kf.x[0])
        doppler_bin_est = float(trk.kf.x[2])

        # ✅ Convert bin estimates -> physical coordinates (Range meters, Velocity m/s)
        x_m, y_v = plotter.bins_to_physical(range_bin_est, doppler_bin_est)

        plotter.ax.text(
            x_m,
            y_v,
            f"ID {trk.track_id}",
            color="white",
            fontsize=9,
            weight="bold",
            ha="center",
            va="center"
        )


    plt.pause(0.01)

plt.show()
