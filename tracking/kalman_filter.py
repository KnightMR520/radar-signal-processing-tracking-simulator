# tracking/kalman_filter.py
from __future__ import annotations
import numpy as np


class KalmanFilterCV2D:
    """
    Constant-velocity KF in "bin space".

    State:
      x = [range_bin, range_rate, doppler_bin, doppler_rate]^T
    Measurement:
      z = [range_bin, doppler_bin]^T
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray | None = None,
        q_pos: float = 0.05,
        q_rate: float = 0.01,
        R: np.ndarray | None = None,
    ):
        self.x = np.array(x0, dtype=float).reshape(4,)
        self.P = np.eye(4) * 10.0 if P0 is None else np.array(P0, dtype=float)

        self.q_pos = float(q_pos)
        self.q_rate = float(q_rate)

        # Measure [range_bin, doppler_bin]
        self.H = np.array(
            [[1, 0, 0, 0],
             [0, 0, 1, 0]],
            dtype=float
        )
        self.R = np.eye(2) * 1.0 if R is None else np.array(R, dtype=float)

    def _F(self, dt: float) -> np.ndarray:
        return np.array(
            [[1, dt, 0,  0],
             [0,  1, 0,  0],
             [0,  0, 1, dt],
             [0,  0, 0,  1]],
            dtype=float
        )

    def _Q(self, dt: float) -> np.ndarray:
        # Light process noise; tune as needed
        return np.diag([
            self.q_pos * dt * dt,
            self.q_rate * dt,
            self.q_pos * dt * dt,
            self.q_rate * dt,
        ]).astype(float)

    def predict(self, dt: float) -> None:
        F = self._F(dt)
        Q = self._Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.array(z, dtype=float).reshape(2,)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis(self, z: np.ndarray) -> float:
        y, S = self.innovation(z)
        return float(y.T @ np.linalg.inv(S) @ y)

    def update(self, z: np.ndarray, R: np.ndarray | None = None) -> None:
        z = np.array(z, dtype=float).reshape(2,)
        if R is not None:
            self.R = np.array(R, dtype=float)

        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
