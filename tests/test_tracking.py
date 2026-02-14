import numpy as np
from tracking.kalman_filter import KalmanFilterCV1D
from tracking.tracker_manager import TrackerManager


def test_kf_converges_on_constant_velocity():
    dt = 0.1
    true_r0 = 100.0
    true_v = 10.0

    kf = KalmanFilterCV1D(x0=np.array([0.0, 0.0]), P0=np.eye(2) * 100.0)

    for k in range(50):
        r = true_r0 + true_v * (k * dt)
        z = np.array([r, true_v]) + np.random.randn(2) * np.array([0.5, 0.2])
        kf.predict(dt)
        kf.update(z)

    assert abs(kf.x[1] - true_v) < 1.0


def test_tracker_keeps_id_with_stable_measurements():
    tm = TrackerManager(dt_default=0.1, confirm_hits=2, max_misses=3)
    meas = [np.array([100.0, 5.0])]

    trks = tm.step(meas)
    tid = trks[0].track_id

    for _ in range(5):
        trks = tm.step([np.array([100.0, 5.0])])
        assert any(t.track_id == tid for t in trks)


def test_tracker_deletes_after_misses():
    tm = TrackerManager(dt_default=0.1, confirm_hits=1, max_misses=3)

    trks = tm.step([np.array([100.0, 5.0])])
    assert len(trks) == 1

    tm.step([])  # miss 1
    tm.step([])  # miss 2
    trks = tm.step([])  # miss 3 -> should be deleted

    assert len(trks) == 0


def test_tracker_survives_single_missed_detection():
    tm = TrackerManager(dt_default=0.1, confirm_hits=1, max_misses=3)

    trks = tm.step([np.array([100.0, 5.0])])
    tid = trks[0].track_id

    tm.step([])  # one miss

    trks = tm.step([np.array([100.5, 5.0])])
    assert any(t.track_id == tid for t in trks)
