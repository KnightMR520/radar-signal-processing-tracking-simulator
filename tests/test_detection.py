import numpy as np
from processing.detection import cfar_2d


def test_cfar_output_shape():
    rd_map = np.random.rand(32, 64)
    detections = cfar_2d(rd_map, guard_cells=(1, 1), training_cells=(4, 4), pfa=1e-3)

    assert detections.shape == rd_map.shape
    assert detections.dtype == bool


def test_cfar_detects_strong_target():
    rd_map = np.random.rand(32, 64) * 0.1
    rd_map[15, 30] = 10.0  # strong target

    detections = cfar_2d(rd_map, guard_cells=(1, 1), training_cells=(4, 4), pfa=1e-3)

    assert detections[15, 30]


def test_cfar_no_false_alarms_on_noise_only():
    np.random.seed(42)
    rd_map = np.random.rand(32, 64) * 0.1

    detections = cfar_2d(rd_map, guard_cells=(1, 1), training_cells=(4, 4), pfa=1e-6)

    # Very low PFA → expect few or zero detections
    assert np.sum(detections) <= 1


def test_cfar_edges_are_false():
    rd_map = np.random.rand(32, 64)

    detections = cfar_2d(rd_map, guard_cells=(2, 2), training_cells=(5, 5), pfa=1e-3)

    # Edges cannot be evaluated safely
    assert not np.any(detections[:5, :])
    assert not np.any(detections[-5:, :])
    assert not np.any(detections[:, :5])
    assert not np.any(detections[:, -5:])
