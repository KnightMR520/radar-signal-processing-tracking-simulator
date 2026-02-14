import numpy as np
from processing.pipeline import process_iq_data
from visualization.range_doppler_plot import LiveRangeDopplerPlot


# -------------------------
# Radar Parameters
# -------------------------
num_pulses = 32
num_samples = 64
prf = 1000

t_slow = np.arange(num_pulses) / prf
fast_time = np.arange(num_samples)

plotter = LiveRangeDopplerPlot(
    prf=prf,
    num_pulses=num_pulses,
    num_samples=num_samples,
    wavelength=0.03,
    range_resolution=1.0,
    dynamic_range_dB=100
)

# -------------------------
# Live Loop
# -------------------------
for doppler_bin in range(-8, 9):

    iq_data = np.zeros((num_pulses, num_samples), dtype=np.complex128)

    range_bin = 25
    range_freq = range_bin / num_samples
    doppler_freq = doppler_bin * prf / num_pulses

    for p in range(num_pulses):
        iq_data[p, :] = (
            np.exp(1j * 2 * np.pi * range_freq * fast_time)
            * np.exp(1j * 2 * np.pi * doppler_freq * t_slow[p])
        )

    rd_map, magnitude_map, detections = process_iq_data(iq_data)

    plotter.update(
        magnitude_map,
        detections=detections,
        title=f"Doppler Bin {doppler_bin}"
    )

plotter.close()
