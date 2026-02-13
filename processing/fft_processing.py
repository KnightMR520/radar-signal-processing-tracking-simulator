import numpy as np
from typing import Tuple


def apply_window(signal: np.ndarray, window_type: str = "hann") -> np.ndarray:
    """
    Apply a window to a 1D or 2D signal.
    """
    if window_type == "hann":
        window = np.hanning(signal.shape[-1])
    elif window_type == "hamming":
        window = np.hamming(signal.shape[-1])
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    return signal * window


def range_fft(iq_data: np.ndarray, n_fft: int = None) -> np.ndarray:
    """
    Perform FFT across fast-time samples to generate range profiles.

    Parameters:
        iq_data: shape (num_pulses, num_samples_per_pulse)
        n_fft: optional FFT size

    Returns:
        range_profiles: complex ndarray, shape (num_pulses, n_fft)
    """
    if n_fft is None:
        n_fft = iq_data.shape[1]

    windowed = apply_window(iq_data, window_type="hann")
    return np.fft.fft(windowed, n=n_fft, axis=1)

    #return np.fft.fft(iq_data, n=n_fft, axis=1)


def doppler_fft(range_profiles: np.ndarray, n_fft: int = None) -> np.ndarray:
    """
    Perform FFT across slow-time (pulses) to generate Doppler spectrum.

    Parameters:
        range_profiles: shape (num_pulses, num_range_bins)
        n_fft: optional FFT size

    Returns:
        doppler_map: complex ndarray, shape (n_fft, num_range_bins)
    """
    if n_fft is None:
        n_fft = range_profiles.shape[0]

    windowed = apply_window(range_profiles.T, window_type="hann").T
    doppler = np.fft.fft(windowed, n=n_fft, axis=0)
    #doppler = np.fft.fft(range_profiles, n=n_fft, axis=0)
    return np.fft.fftshift(doppler, axes=0)


def range_doppler_map(iq_data: np.ndarray,
                      n_range_fft: int = None,
                      n_doppler_fft: int = None) -> np.ndarray:
    """
    Compute a full range-Doppler map from raw IQ data.

    Parameters:
        iq_data: shape (num_pulses, num_samples_per_pulse)

    Returns:
        rd_map: complex ndarray, shape (n_doppler_fft, n_range_fft)
    """
    range_profiles = range_fft(iq_data, n_fft=n_range_fft)
    rd_map = doppler_fft(range_profiles, n_fft=n_doppler_fft)
    return rd_map