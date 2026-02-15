# processing/pipeline.py

from __future__ import annotations

import numpy as np

from processing.fft_processing import range_doppler_map
from processing.detection import cfar_2d, suppress_to_local_max

from diagnostics.metrics import (
    MetricsRegistry,
    Timer,
    PIPELINE_LATENCY,
    CFAR_DETECTIONS,
    FAULTS_INJECTED,
)
from diagnostics.fault_injection import FaultInjector


def process_iq_data(
    iq_data: np.ndarray,
    n_range_fft: int | None = None,
    n_doppler_fft: int | None = None,
    guard_cells=(1, 1),
    training_cells=(4, 4),
    pfa: float = 1e-3,
    use_power: bool = True,
    *,
    metrics: MetricsRegistry | None = None,
    fault_injector: FaultInjector | None = None,
):
    """
    Full radar signal processing pipeline.

    Steps:
        1. Range FFT
        2. Doppler FFT
        3. Magnitude / Power conversion
        4. CFAR detection
        5. Local-max suppression

    Optional:
        - metrics: records latency + detection counts
        - fault_injector: can drop IQ frame, inject noise spikes, override CFAR pfa

    Returns:
        rd_map: complex range-Doppler map
        magnitude_map: magnitude or power map
        detections: boolean detection mask
    """

    # -----------------------
    # Fault injection (IQ)
    # -----------------------
    if fault_injector is not None:
        maybe_iq = fault_injector.maybe_drop_iq(iq_data)
        if maybe_iq is None:
            # Dropped frame: return "empty" outputs with correct shapes
            if metrics is not None:
                metrics.inc(FAULTS_INJECTED, 1)

            # best-effort output sizing (keeps downstream stable)
            num_pulses, num_samples = iq_data.shape
            n_r = num_samples if n_range_fft is None else int(n_range_fft)
            n_d = num_pulses if n_doppler_fft is None else int(n_doppler_fft)

            rd_map = np.zeros((n_d, n_r), dtype=np.complex128)
            magnitude_map = np.zeros((n_d, n_r), dtype=float)
            detections = np.zeros((n_d, n_r), dtype=bool)
            if metrics is not None:
                metrics.inc(CFAR_DETECTIONS, 0)
            return rd_map, magnitude_map, detections

        iq_data = fault_injector.maybe_noise_spike(maybe_iq)
        if metrics is not None and not np.shares_memory(iq_data, maybe_iq):
            metrics.inc(FAULTS_INJECTED, 1)

        # Parameter fault: override pfa if configured
        pfa = fault_injector.maybe_override_cfar(pfa)

    # -----------------------
    # Metrics timing wrapper
    # -----------------------
    if metrics is None:
        # fast path (no timing context manager)
        rd_map, magnitude_map, detections = _process_iq_data_core(
            iq_data=iq_data,
            n_range_fft=n_range_fft,
            n_doppler_fft=n_doppler_fft,
            guard_cells=guard_cells,
            training_cells=training_cells,
            pfa=pfa,
            use_power=use_power,
        )
    else:
        with Timer(metrics, PIPELINE_LATENCY):
            rd_map, magnitude_map, detections = _process_iq_data_core(
                iq_data=iq_data,
                n_range_fft=n_range_fft,
                n_doppler_fft=n_doppler_fft,
                guard_cells=guard_cells,
                training_cells=training_cells,
                pfa=pfa,
                use_power=use_power,
            )

    # Count detections for observability
    if metrics is not None:
        metrics.inc(CFAR_DETECTIONS, int(np.count_nonzero(detections)))

    return rd_map, magnitude_map, detections


def _process_iq_data_core(
    iq_data: np.ndarray,
    n_range_fft: int | None,
    n_doppler_fft: int | None,
    guard_cells,
    training_cells,
    pfa: float,
    use_power: bool,
):
    # Step 1–2: FFT Processing
    rd_map = range_doppler_map(
        iq_data,
        n_range_fft=n_range_fft,
        n_doppler_fft=n_doppler_fft,
    )

    # Step 3: Convert to magnitude or power
    if use_power:
        magnitude_map = np.abs(rd_map) ** 2
    else:
        magnitude_map = np.abs(rd_map)

    # Step 4: CFAR Detection
    raw_detections = cfar_2d(
        magnitude_map,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=pfa,
    )

    # Step 5: Local-max suppression (1 detection per cluster)
    detections = suppress_to_local_max(raw_detections, magnitude_map)

    return rd_map, magnitude_map, detections
