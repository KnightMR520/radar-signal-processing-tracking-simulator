# tests/test_tracking.py

import numpy as np
from tracking.kalman_filter import KalmanFilterCV2D
from tracking.tracker_manager import TrackerManager


def test_kf_converges_on_constant_velocity_in_range_bin():
    """
    KF tracks in BIN space:
      state = [range_bin, range_rate, doppler_bin, doppler_rate]
      measurement = [range_bin, doppler_bin]

    This test checks that range_rate converges to the true constant rate.
    """
    rng = np.random.default_rng(0)

    dt = 0.1
    true_r0 = 100.0
    true_rdot = 10.0  # bins per second (for test)
    true_doppler_bin = 16.0

    # x0 must be length 4 for CV2D
    kf = KalmanFilterCV2D(
        x0=np.array([0.0, 0.0, true_doppler_bin, 0.0], dtype=float),
        P0=np.eye(4) * 100.0,
        R=np.diag([0.5**2, 0.2**2]),
        q_pos=0.05,
        q_rate=0.01,
    )

    for k in range(80):
        r = true_r0 + true_rdot * (k * dt)
        # measurement = [range_bin, doppler_bin]
        z = np.array([r, true_doppler_bin], dtype=float) + rng.normal(0.0, [0.5, 0.2])
        kf.predict(dt)
        kf.update(z)

    # range_rate is state[1]
    assert abs(kf.x[1] - true_rdot) < 1.5


def _get_confirmed_ids(tracks):
    return [t.track_id for t in tracks if t.confirmed]


def test_tracker_keeps_id_with_stable_measurements():
    """
    TrackerManager uses candidate-based birth:
      - no tracks may exist after 1st measurement
      - track is spawned after `birth_hits` frames (default 2)
    """
    tm = TrackerManager(dt_default=0.1, confirm_hits=2, max_misses=3)

    z = np.array([100.0, 5.0], dtype=float)

    # Frame 1: candidate only, likely no track yet
    trks = tm.step([z])
    assert len(trks) in (0, 1)  # depending on your birth settings

    # Frame 2: should spawn a real track now
    trks = tm.step([z])
    assert len(trks) >= 1

    tid = trks[0].track_id

    # Continue feeding stable measurements; ID should persist
    for _ in range(5):
        trks = tm.step([z])
        assert any(t.track_id == tid for t in trks)


def test_tracker_deletes_after_misses():
    tm = TrackerManager(dt_default=0.1, confirm_hits=1, max_misses=3)

    z = np.array([100.0, 5.0], dtype=float)

    # Need two frames to spawn due to candidate-based birth
    tm.step([z])
    trks = tm.step([z])
    assert len(trks) >= 1

    # Now miss until deletion
    tm.step([])  # miss 1
    tm.step([])  # miss 2
    trks = tm.step([])  # miss 3 -> should be deleted (misses >= max_misses)
    assert len(trks) == 0


def test_tracker_survives_single_missed_detection():
    tm = TrackerManager(dt_default=0.1, confirm_hits=1, max_misses=3)

    z1 = np.array([100.0, 5.0], dtype=float)

    # Spawn track (2 frames)
    tm.step([z1])
    trks = tm.step([z1])
    assert len(trks) >= 1
    tid = trks[0].track_id

    # One miss
    tm.step([])

    # Measurement returns near prior
    z2 = np.array([100.5, 5.0], dtype=float)
    trks = tm.step([z2])

    assert any(t.track_id == tid for t in trks)
