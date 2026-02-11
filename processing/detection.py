import numpy as np


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
