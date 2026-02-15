import numpy as np
from processing.pipeline import process_iq_data
from visualization.range_doppler_plot import plot_range_doppler


# Simulate simple moving target
num_pulses = 32
num_samples = 64

doppler_bin = 5
prf = 1000
t_slow = np.arange(num_pulses) / prf
doppler_freq = doppler_bin * prf / num_pulses

iq_data = np.zeros((num_pulses, num_samples), dtype=complex)
iq_data[:, 25] = np.exp(1j * 2 * np.pi * doppler_freq * t_slow)

rd_map, magnitude_map, detections = process_iq_data(iq_data)

plot_range_doppler(
    magnitude_map,
    detections=detections,
    title="Simulated Target"
)
