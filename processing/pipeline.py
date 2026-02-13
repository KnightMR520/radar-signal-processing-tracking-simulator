import numpy as np
from processing.fft_processing import range_doppler_map
from processing.detection import cfar_2d


def process_iq_data(
    iq_data: np.ndarray,
    n_range_fft: int = None,
    n_doppler_fft: int = None,
    guard_cells=(1, 1),
    training_cells=(4, 4),
    pfa=1e-3,
    use_power=True,
):
    """
    Full radar signal processing pipeline.

    Steps:
        1. Range FFT
        2. Doppler FFT
        3. Magnitude / Power conversion
        4. CFAR detection

    Returns:
        rd_map: complex range-Doppler map
        magnitude_map: magnitude or power map
        detections: boolean detection mask
    """

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
    detections = cfar_2d(
        magnitude_map,
        guard_cells=guard_cells,
        training_cells=training_cells,
        pfa=pfa,
    )

    return rd_map, magnitude_map, detections
