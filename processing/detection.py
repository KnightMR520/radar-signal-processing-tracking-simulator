import numpy as np
from scipy.ndimage import label

def cfar_2d(
    rd_map: np.ndarray,
    guard_cells=(1, 1),
    training_cells=(4, 4),
    pfa=1e-3
) -> np.ndarray:
    """
    2D Cell-Averaging CFAR detector.

    Parameters:
        rd_map: 2D magnitude map (range x Doppler)
        guard_cells: (g_r, g_d)
        training_cells: (t_r, t_d)
        pfa: Probability of false alarm

    Returns:
        detections: Boolean array same shape as rd_map
    """
    num_rows, num_cols = rd_map.shape
    g_r, g_d = guard_cells
    t_r, t_d = training_cells

    detections = np.zeros_like(rd_map, dtype=bool)

    # Total training cells
    num_train = (2 * (t_r + g_r) + 1) * (2 * (t_d + g_d) + 1) - (2 * g_r + 1) * (2 * g_d + 1)

    # CFAR threshold scaling factor (CA-CFAR)
    alpha = num_train * (pfa ** (-1 / num_train) - 1)

    for r in range(t_r + g_r, num_rows - (t_r + g_r)):
        for d in range(t_d + g_d, num_cols - (t_d + g_d)):
            r_start = r - (t_r + g_r)
            r_end = r + (t_r + g_r) + 1
            d_start = d - (t_d + g_d)
            d_end = d + (t_d + g_d) + 1

            window = rd_map[r_start:r_end, d_start:d_end]

            # Zero out guard cells + CUT
            cut_r_start = t_r
            cut_r_end = t_r + 2 * g_r + 1
            cut_d_start = t_d
            cut_d_end = t_d + 2 * g_d + 1
            window_copy = window.copy()
            window_copy[cut_r_start:cut_r_end, cut_d_start:cut_d_end] = 0

            noise_level = np.sum(window_copy) / num_train
            threshold = alpha * noise_level

            if rd_map[r, d] > threshold:
                detections[r, d] = True

    return detections

def suppress_to_local_max(detections: np.ndarray,
                          magnitude_map: np.ndarray) -> np.ndarray:
    """
    Reduce clustered detections to a single local maximum per cluster.

    Parameters:
        detections: boolean array from CFAR (same shape as magnitude_map)
        magnitude_map: power map used for detection

    Returns:
        pruned_detections: boolean array with one True per cluster
    """

    # Label connected detection regions
    labeled_array, num_features = label(detections)

    pruned = np.zeros_like(detections, dtype=bool)

    for region_id in range(1, num_features + 1):
        region_mask = labeled_array == region_id

        if not np.any(region_mask):
            continue

        # Find index of maximum magnitude within this region
        region_values = magnitude_map * region_mask
        max_index = np.unravel_index(
            np.argmax(region_values),
            magnitude_map.shape
        )

        pruned[max_index] = True

    return pruned