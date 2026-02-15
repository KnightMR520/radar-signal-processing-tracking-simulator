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
    CFAR_RAW_THIS_FRAME,
    FAULTS_INJECTED,
)
from diagnostics.fault_injection import FaultInjector, FaultConfig
from diagnostics.health_monitor import HealthMonitor, HealthConfig


# -----------------------------
# Radar Parameters
# -----------------------------
num_pulses = 32
num_samples = 64
prf = 1000

wavelength = 0.03
range_resolution = 1.0

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
    jammer_line_prob=1,     
    jammer_line_scale=0.5,   # <- strong
    jammer_mode="range",      # "range" tends to create vertical ridge
    # jammer_mode="doppler",  # or try this for horizontal ridge
))


# -----------------------------
# Health Monitor
# Tune RAW CFAR enter/exit so it triggers on your spikes
# -----------------------------
health = HealthMonitor(HealthConfig(
    raw_cfar_enter=32,   # enter degraded if >=200 raw CFAR hits/frame
    raw_cfar_exit=28,
    enter_count_required=2,
    exit_count_required=4,
))



# -----------------------------
# Plot
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
# Tracker
# -----------------------------
tracker = TrackerManager(
    dt_default=1.0,
    gate_mahalanobis=16.0,
    max_misses=6,
    confirm_hits=2,
    measurement_R=np.diag([4.0, 4.0])
)


# -----------------------------
# Targets (bin space)
# -----------------------------
targets = [
    {"range_bin": 15, "doppler_bin": 8,  "amp": 1.0},
    {"range_bin": 40, "doppler_bin": 26, "amp": 1.0},
    {"range_bin": 25, "doppler_bin": 20, "amp": 1.0},
]


# -----------------------------
# Degraded-mode controls (PERSIST across frames)
# -----------------------------
pfa = 1e-3
K = 3
tracker.set_birth_enabled(True)


# -----------------------------
# Main Loop
# -----------------------------
for frame in range(200):

    iq_data = np.zeros((num_pulses, num_samples), dtype=np.complex128)

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

        tgt["range_bin"] += np.random.choice([-1, 0, 1])
        tgt["doppler_bin"] += np.random.choice([-1, 0, 1])

        tgt["range_bin"] = int(np.clip(tgt["range_bin"], margin_r, num_samples - 1 - margin_r))
        tgt["doppler_bin"] = int(np.clip(tgt["doppler_bin"], margin_d, num_pulses - 1 - margin_d))

    # Baseline noise
    noise_power = 0.01
    iq_data += np.sqrt(noise_power / 2) * (
        np.random.randn(*iq_data.shape) + 1j * np.random.randn(*iq_data.shape)
    )

    # -----------------------------
    # Process pipeline (uses CURRENT pfa)
    # -----------------------------
    rd_map, magnitude_map, detection_mask = process_iq_data(
        iq_data,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=pfa,
        use_power=True,
        metrics=metrics,
        fault_injector=fault_injector,
    )

    # -----------------------------
    # Health evaluation
    # -----------------------------
    snap = metrics.snapshot()
    state, reason = health.update(snap)

    # -----------------------------
    # Apply degraded-mode behaviors (PERSIST into next frame)
    # -----------------------------
    if state == "DEGRADED":
        # tighten CFAR threshold (lower PFA -> fewer false alarms)
        pfa = 1e-5
        # reduce measurement bandwidth into tracker
        K = 2
        # freeze births (stop new IDs from noise peaks)
        tracker.set_birth_enabled(False)
    else:
        pfa = 1e-3
        K = 3
        tracker.set_birth_enabled(True)

    # -----------------------------
    # Build measurements (bin space)
    # -----------------------------
    det_idxs = np.argwhere(detection_mask)  # [doppler_idx, range_idx]
    measurements = []

    if det_idxs.size > 0:
        scores = magnitude_map[detection_mask]
        order = np.argsort(scores)[::-1]
        det_idxs = det_idxs[order[:K]]

        for doppler_idx, range_idx in det_idxs:
            measurements.append(np.array([float(range_idx), float(doppler_idx)], dtype=float))

    # Drop measurements BEFORE tracking
    before = len(measurements)
    measurements = fault_injector.maybe_drop_measurements(measurements)
    dropped = before - len(measurements)
    if dropped > 0:
        metrics.inc(FAULTS_INJECTED, dropped)

    # Tracking
    tracks = tracker.step(measurements, dt=1.0)

    metrics.set_gauge(TRACKS_ACTIVE, len(tracks))
    metrics.set_gauge(TRACKS_CONFIRMED, sum(1 for t in tracks if t.confirmed))

    # Periodic print
    if frame % 25 == 0:
        mean_lat_ms = 1000.0 * snap["timers"].get(PIPELINE_LATENCY, {}).get("mean_s", 0.0)
        det_total = snap["counters"].get(CFAR_DETECTIONS, 0)
        raw_cfar = int(snap["gauges"].get(CFAR_RAW_THIS_FRAME, 0))
        faults_total = snap["counters"].get(FAULTS_INJECTED, 0)
        print(
            f"[frame={frame:03d}] state={state:<8} "
            f"lat_mean={mean_lat_ms:.2f}ms raw_cfar={raw_cfar:4d} det_total={det_total:4d} "
            f"K={K} pfa={pfa:.0e} births={'ON' if tracker.birth_enabled else 'OFF'} "
            f"tracks={len(tracks)} confirmed={sum(1 for t in tracks if t.confirmed)} "
            f"faults_total={faults_total} "
            f"{('reason=' + reason) if reason else ''}"
        )

    # Plot
    plotter.update(
        magnitude_map,
        detections=detection_mask,
        title=f"Frame {frame} | {state}"
    )

    for txt in list(plotter.ax.texts):
        txt.remove()

    for trk in tracks:
        if not trk.confirmed:
            continue

        range_bin_est = float(trk.kf.x[0])
        doppler_bin_est = float(trk.kf.x[2])
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
