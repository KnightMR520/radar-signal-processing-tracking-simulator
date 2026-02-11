import numpy as np
from processing.fft_processing import range_fft, doppler_fft, range_doppler_map


def test_range_fft_shape():
    num_pulses = 16
    num_samples = 128
    iq_data = np.random.randn(num_pulses, num_samples) + 1j * np.random.randn(num_pulses, num_samples)

    range_profiles = range_fft(iq_data)

    assert range_profiles.shape == (num_pulses, num_samples)
    assert np.iscomplexobj(range_profiles)


def test_doppler_fft_shape():
    num_pulses = 32
    num_range_bins = 64
    range_profiles = np.random.randn(num_pulses, num_range_bins) + 1j * np.random.randn(num_pulses, num_range_bins)

    doppler_map = doppler_fft(range_profiles)

    assert doppler_map.shape == (num_pulses, num_range_bins)
    assert np.iscomplexobj(doppler_map)


def test_range_doppler_map_shape():
    num_pulses = 16
    num_samples = 128
    iq_data = np.random.randn(num_pulses, num_samples) + 1j * np.random.randn(num_pulses, num_samples)

    rd_map = range_doppler_map(iq_data)

    assert rd_map.shape == (num_pulses, num_samples)
    assert np.iscomplexobj(rd_map)


def test_doppler_peak_location():
    """
    Inject a known Doppler frequency and verify FFT peak occurs
    near the expected bin.
    """
    num_pulses = 32
    num_samples = 64
    doppler_bin = 5  # expected Doppler index after fftshift
    prf = 1000  # Hz

    t_slow = np.arange(num_pulses) / prf
    doppler_freq = doppler_bin * prf / num_pulses

    # Simulate a single range bin with Doppler tone
    iq_data = np.zeros((num_pulses, num_samples), dtype=complex)
    iq_data[:, 10] = np.exp(1j * 2 * np.pi * doppler_freq * t_slow)

    rd_map = range_doppler_map(iq_data)

    magnitude = np.abs(rd_map)
    peak_doppler_bin = np.argmax(np.sum(magnitude, axis=1))

    expected_bin = doppler_bin + num_pulses // 2  # because of fftshift
    assert abs(peak_doppler_bin - expected_bin) <= 1
