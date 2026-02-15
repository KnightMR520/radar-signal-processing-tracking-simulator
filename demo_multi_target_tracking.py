# demo/demo_multi_target_tracking.py

import numpy as np
import matplotlib.pyplot as plt

from processing.pipeline import process_iq_data
from tracking.tracker_manager import TrackerManager
from visualization.range_doppler_plot import LiveRangeDopplerPlot

from diagnostics.metrics import (
    MetricsRegistry,
    TRACKS_ACTIVE,
    TRACKS_CONFIRMED,
    PIPELINE_LATENCY,
    CFAR_DETECTIONS,
    FAULTS_INJECTED,
)
from diagnostics.fault_injection import FaultInjector, FaultConfig


# -----------------------------
# Radar Parameters
# -----------------------------
num_pulses = 32
num_samples = 64
prf = 1000

wavelength = 0.03          # meters (10 GHz radar)
range_resolution = 1.0     # meters per range bin (USED FOR SYNTH ONLY here)

t_slow = np.arange(num_pulses) / prf
fast_time = np.arange(num_samples)


# -----------------------------
# CFAR window sizes
# -----------------------------
guard_cells = (1, 1)
training_cells = (4, 4)
g_r, g_d = guard_cells
t_r, t_d = training_cells

margin_r = t_r + g_r
margin_d = t_d + g_d


# -----------------------------
# Metrics + Fault Injection
# -----------------------------
metrics = MetricsRegistry(window_size=200)

fault_injector = FaultInjector(FaultConfig(
    enabled=True,
    rng_seed=123,

    # IQ-level faults
    noise_spike_prob=0.02,      # 2% of frames get burst noise
    noise_spike_scale=8.0,
    drop_iq_frames=0.01,        # 1% of frames disappear entirely

    # Measurement-level faults
    drop_measurements=0.05,     # 5% of detections dropped before tracking

    # Param faults (optional)
    # cfar_pfa_override=1e-2,    # uncomment to force worse CFAR
))


# -----------------------------
# Plot Setup (physical axes)
# -----------------------------
plotter = LiveRangeDopplerPlot(
    prf=prf,
    num_pulses=num_pulses,
    num_samples=num_samples,
    wavelength=wavelength,
    range_resolution=range_resolution,
    dynamic_range_dB=80
)


# -----------------------------
# Tracker (bin-space)
# -----------------------------
tracker = TrackerManager(
    dt_default=1.0,
    gate_mahalanobis=16.0,
    max_misses=6,                   # survive occasional misses / blinks
    confirm_hits=2,
    measurement_R=np.diag([4.0, 4.0])
)


# -----------------------------
# Target Definitions (synthetic truth) in BIN space
# doppler_bin here is FFTSHIFTED index [0..N-1]
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

        range_freq = range_bin / num_samples
        doppler_freq = (doppler_bin - num_pulses // 2) * prf / num_pulses

        for p in range(num_pulses):
            iq_data[p, :] += amp * (
                np.exp(1j * 2 * np.pi * range_freq * fast_time)
                * np.exp(1j * 2 * np.pi * doppler_freq * t_slow[p])
            )

        # Random walk motion
        tgt["range_bin"] += np.random.choice([-1, 0, 1])
        tgt["doppler_bin"] += np.random.choice([-1, 0, 1])

        # Keep targets inside CFAR-valid region
        tgt["range_bin"] = int(np.clip(tgt["range_bin"], margin_r, num_samples - 1 - margin_r))
        tgt["doppler_bin"] = int(np.clip(tgt["doppler_bin"], margin_d, num_pulses - 1 - margin_d))

    # Optional baseline noise (normal realism)
    noise_power = 0.01
    iq_data += np.sqrt(noise_power / 2) * (
        np.random.randn(*iq_data.shape) + 1j * np.random.randn(*iq_data.shape)
    )

    # --- Process pipeline (metrics + IQ faults happen inside pipeline now) ---
    rd_map, magnitude_map, detection_mask = process_iq_data(
        iq_data,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=1e-3,
        use_power=True,
        metrics=metrics,
        fault_injector=fault_injector,
    )

    # --- Build measurements from detections (bin space) ---
    det_idxs = np.argwhere(detection_mask)  # [doppler_idx, range_idx]

    measurements = []
    if det_idxs.size > 0:
        scores = magnitude_map[detection_mask]
        order = np.argsort(scores)[::-1]

        # we know we have 3 truth targets; keep top-3 strongest peaks
        K = 3
        det_idxs = det_idxs[order[:K]]

        for doppler_idx, range_idx in det_idxs:
            measurements.append(np.array([float(range_idx), float(doppler_idx)], dtype=float))

    # --- Fault injection: drop measurements BEFORE tracking ---
    before = len(measurements)
    measurements = fault_injector.maybe_drop_measurements(measurements)
    dropped = before - len(measurements)
    if dropped > 0:
        metrics.inc(FAULTS_INJECTED, dropped)

    # --- Tracking step ---
    tracks = tracker.step(measurements, dt=1.0)

    # --- Metrics on tracking ---
    metrics.set_gauge(TRACKS_ACTIVE, len(tracks))
    metrics.set_gauge(TRACKS_CONFIRMED, sum(1 for t in tracks if t.confirmed))

    # Periodic observability printout
    if frame % 25 == 0:
        snap = metrics.snapshot()
        mean_lat_ms = 1000.0 * snap["timers"].get(PIPELINE_LATENCY, {}).get("mean_s", 0.0)
        det_count = snap["counters"].get(CFAR_DETECTIONS, 0)
        faults = snap["counters"].get(FAULTS_INJECTED, 0)
        print(
            f"[frame={frame:03d}] "
            f"lat_mean={mean_lat_ms:.2f}ms "
            f"cfar_dets_total={det_count} "
            f"tracks={snap['gauges'].get(TRACKS_ACTIVE, 0):.0f} "
            f"confirmed={snap['gauges'].get(TRACKS_CONFIRMED, 0):.0f} "
            f"faults_total={faults}"
        )

    # --- Plot update ---
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

        range_bin_est = float(trk.kf.x[0])
        doppler_bin_est = float(trk.kf.x[2])

        # Convert bin estimates -> physical coords
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
