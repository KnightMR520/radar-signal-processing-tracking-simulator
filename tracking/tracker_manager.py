# tracking/tracker_manager.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.kalman_filter import KalmanFilterCV2D


@dataclass
class Track:
    track_id: int
    kf: KalmanFilterCV2D
    age: int = 1
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


class TrackerManager:
    """
    Multi-target tracker manager using:
      - KalmanFilterCV2D (bin-space constant velocity)
      - Mahalanobis gating
      - Hungarian global assignment
      - Track initiation/confirmation/deletion

    Key upgrade:
      - Candidate-based track birth: require persistence before spawning a real track
        to prevent ghost IDs from one-off CFAR peaks.
    """

    def __init__(
        self,
        dt_default: float,
        gate_mahalanobis: float = 9.0,   # ~3-sigma for 2D measurement (depends on R)
        max_misses: int = 5,
        confirm_hits: int = 3,
        cost_unassigned: float = 1e6,    # big cost for forbidden assignments
        measurement_R: np.ndarray | None = None,
    ):
        self.dt_default = float(dt_default)
        self.gate_mahalanobis = float(gate_mahalanobis)
        self.max_misses = int(max_misses)
        self.confirm_hits = int(confirm_hits)
        self.cost_unassigned = float(cost_unassigned)

        # Measurement noise in bin units: [range_bin, doppler_bin]
        self.measurement_R = np.eye(2) if measurement_R is None else np.array(measurement_R, dtype=float)

        self._tracks: list[Track] = []
        self._next_id = 1

        # ---------------------------
        # Candidate-based track birth
        # ---------------------------
        # Require measurements to persist near the same bin location before spawning a KF.
        self.birth_hits = 2                         # frames needed to spawn a real track
        self.birth_tol = np.array([2.0, 2.0])       # bins: [range_tol, doppler_tol]
        self.cand_max_misses = 1                    # candidate can disappear briefly
        self._candidates: list[dict] = []           # {"z": np.array([r,d]), "hits": int, "misses": int}

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def _spawn_track(self, z: np.ndarray) -> None:
        """
        z is measurement: [range_bin, doppler_bin]
        """
        z = np.array(z, dtype=float).reshape(2,)
        x0 = np.array([z[0], 0.0, z[1], 0.0], dtype=float)

        kf = KalmanFilterCV2D(
            x0=x0,
            P0=np.eye(4) * 25.0,
            q_pos=0.05,
            q_rate=0.01,
            R=self.measurement_R,
        )
        self._tracks.append(Track(track_id=self._next_id, kf=kf))
        self._next_id += 1

    def _is_near_any_track(self, z: np.ndarray) -> bool:
        """
        Prevent spawning candidates/tracks right on top of an existing gated prediction.
        """
        z = np.array(z, dtype=float).reshape(2,)
        for trk in self._tracks:
            d2 = trk.kf.mahalanobis(z)
            if d2 <= self.gate_mahalanobis:
                return True
        return False

    def _update_candidates_and_spawn(self, unassigned_Z: list[np.ndarray]) -> None:
        """
        Update candidate pool with unassigned measurements.
        Spawn a real track only after `birth_hits` consistent observations.
        """
        # Age candidates
        for c in self._candidates:
            c["misses"] += 1

        for z in unassigned_Z:
            z = np.array(z, dtype=float).reshape(2,)

            # If it's near an existing track prediction, don't create a new birth candidate
            if self._is_near_any_track(z):
                continue

            matched = False
            for c in self._candidates:
                if np.all(np.abs(z - c["z"]) <= self.birth_tol):
                    # Update candidate position (EMA reduces jitter)
                    c["z"] = 0.7 * c["z"] + 0.3 * z
                    c["hits"] += 1
                    c["misses"] = 0
                    matched = True

                    if c["hits"] >= self.birth_hits:
                        self._spawn_track(c["z"])
                        c["misses"] = 999  # mark for removal
                    break

            if not matched:
                self._candidates.append({"z": z.copy(), "hits": 1, "misses": 0})

        # Prune stale/spawned candidates
        self._candidates = [c for c in self._candidates if c["misses"] <= self.cand_max_misses]

    def step(self, measurements: list[np.ndarray], dt: float | None = None) -> list[Track]:
        """
        measurements: list of [range_bin, doppler_bin]
        """
        dt = self.dt_default if dt is None else float(dt)
        Z = [np.array(m, dtype=float).reshape(2,) for m in measurements]

        # 1) Predict all tracks
        for trk in self._tracks:
            trk.kf.predict(dt)
            trk.age += 1

        # No tracks yet: DON'T instantly spawn everything as tracks; use candidates
        if len(self._tracks) == 0:
            self._update_candidates_and_spawn(Z)
            return list(self._tracks)

        # No measurements: everybody misses
        if len(Z) == 0:
            for trk in self._tracks:
                trk.misses += 1
            self._tracks = [t for t in self._tracks if t.misses < self.max_misses]
            return list(self._tracks)

        # 2) Build global cost matrix using Mahalanobis distance
        T = len(self._tracks)
        M = len(Z)
        cost = np.full((T, M), self.cost_unassigned, dtype=float)

        for i, trk in enumerate(self._tracks):
            for j, z in enumerate(Z):
                d2 = trk.kf.mahalanobis(z)
                if d2 <= self.gate_mahalanobis:
                    cost[i, j] = d2  # lower is better

        # 3) Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_meas = set()

        # 4) Apply valid assignments
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= self.cost_unassigned:
                continue

            trk = self._tracks[r]
            z = Z[c]

            trk.kf.update(z)
            trk.hits += 1
            trk.misses = 0
            if trk.hits >= self.confirm_hits:
                trk.confirmed = True

            assigned_tracks.add(r)
            assigned_meas.add(c)

        # 5) Unassigned tracks miss++
        for i, trk in enumerate(self._tracks):
            if i not in assigned_tracks:
                trk.misses += 1

        # 6) Delete stale tracks
        self._tracks = [t for t in self._tracks if t.misses < self.max_misses]

        # 7) Candidate-based spawning for unassigned measurements
        unassigned_Z = [Z[j] for j in range(M) if j not in assigned_meas]
        self._update_candidates_and_spawn(unassigned_Z)

        return list(self._tracks)
