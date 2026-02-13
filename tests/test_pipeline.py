import numpy as np
from processing.pipeline import process_iq_data


def test_pipeline_detects_known_target():
    num_pulses = 32
    num_samples = 64

    doppler_bin = 4
    prf = 1000
    t_slow = np.arange(num_pulses) / prf
    doppler_freq = doppler_bin * prf / num_pulses

    iq_data = np.zeros((num_pulses, num_samples), dtype=complex)
    iq_data[:, 20] = np.exp(1j * 2 * np.pi * doppler_freq * t_slow)

    rd_map, magnitude_map, detections = process_iq_data(
        iq_data,
        guard_cells=(1, 1),
        training_cells=(4, 4),
        pfa=1e-3,
    )

    assert detections.shape == magnitude_map.shape
    assert np.sum(detections) >= 1
