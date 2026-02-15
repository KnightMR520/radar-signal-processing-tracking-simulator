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
    CFAR_RAW_THIS_FRAME,
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
    # -----------------------
    # Fault injection (IQ)
    # -----------------------
    if fault_injector is not None:
        maybe_iq = fault_injector.maybe_drop_iq(iq_data)
        if maybe_iq is None:
            if metrics is not None:
                metrics.inc(FAULTS_INJECTED, 1)

            num_pulses, num_samples = iq_data.shape
            n_r = num_samples if n_range_fft is None else int(n_range_fft)
            n_d = num_pulses if n_doppler_fft is None else int(n_doppler_fft)

            rd_map = np.zeros((n_d, n_r), dtype=np.complex128)
            magnitude_map = np.zeros((n_d, n_r), dtype=float)
            detections = np.zeros((n_d, n_r), dtype=bool)

            if metrics is not None:
                metrics.set_gauge(CFAR_RAW_THIS_FRAME, 0.0)
                metrics.inc(CFAR_DETECTIONS, 0)
            return rd_map, magnitude_map, detections

        # 1) Noise spike
        iq_data2, injected_noise = fault_injector.maybe_noise_spike(maybe_iq)
        iq_data = iq_data2
        if metrics is not None and injected_noise:
            metrics.inc(FAULTS_INJECTED, 1)

        # 2) Jammer line (structured interference)
        iq_data2, injected_noise = fault_injector.maybe_noise_spike(maybe_iq)
        iq_data3, injected_jammer = fault_injector.maybe_jammer_line(iq_data2)
        iq_data = iq_data3

        if metrics is not None:
            if injected_noise:
                metrics.inc(FAULTS_INJECTED, 1)
            if injected_jammer:
                metrics.inc(FAULTS_INJECTED, 1)

        # 3) Param faults
        pfa = fault_injector.maybe_override_cfar(pfa)


    # -----------------------
    # Run core + time it
    # -----------------------
    if metrics is None:
        rd_map, magnitude_map, detections, raw_cfar_count = _process_iq_data_core(
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
            rd_map, magnitude_map, detections, raw_cfar_count = _process_iq_data_core(
                iq_data=iq_data,
                n_range_fft=n_range_fft,
                n_doppler_fft=n_doppler_fft,
                guard_cells=guard_cells,
                training_cells=training_cells,
                pfa=pfa,
                use_power=use_power,
            )

    if metrics is not None:
        metrics.set_gauge(CFAR_RAW_THIS_FRAME, float(raw_cfar_count))
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
    rd_map = range_doppler_map(
        iq_data,
        n_range_fft=n_range_fft,
        n_doppler_fft=n_doppler_fft,
    )

    magnitude_map = (np.abs(rd_map) ** 2) if use_power else np.abs(rd_map)

    raw_detections = cfar_2d(
        magnitude_map,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=pfa,
    )
    raw_cfar_count = int(np.count_nonzero(raw_detections))

    detections = suppress_to_local_max(raw_detections, magnitude_map)
    return rd_map, magnitude_map, detections, raw_cfar_count
