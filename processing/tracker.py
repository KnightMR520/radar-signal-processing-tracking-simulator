import numpy as np


class Track:
    def __init__(self, track_id, initial_measurement):
        """
        Simple constant-velocity Kalman filter in 2D:
        State: [range, velocity_range, doppler, velocity_doppler]
        """
        self.id = track_id

        # State vector [r, vr, d, vd]
        self.x = np.array([
            initial_measurement[0], 0.0,
            initial_measurement[1], 0.0
        ], dtype=float)

        # Covariance
        self.P = np.eye(4) * 10.0

        # Motion model
        self.F = np.array([
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ])

        self.Q = np.eye(4) * 0.01

        # Measurement model
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ])

        self.R = np.eye(2) * 5.0

        # Lifecycle
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.confirmed = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        z = np.array(measurement)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class MultiTargetTracker:
    def __init__(self, max_misses=5, confirmation_hits=3, gating_threshold=15):
        self.tracks = []
        self.next_id = 1

        self.max_misses = max_misses
        self.confirmation_hits = confirmation_hits
        self.gating_threshold = gating_threshold

    def _associate(self, detections):
        """
        Nearest neighbor association with gating.
        """
        unmatched_detections = detections.copy()
        matches = []

        for track in self.tracks:
            if len(unmatched_detections) == 0:
                break

            predicted = track.x[[0, 2]]  # [range, doppler]

            distances = [
                np.linalg.norm(predicted - det)
                for det in unmatched_detections
            ]

            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist < self.gating_threshold:
                matches.append((track, unmatched_detections[min_idx]))
                unmatched_detections.pop(min_idx)

        return matches, unmatched_detections

    def step(self, detections):
        """
        detections: list of [range, doppler]
        """
        detections = [np.array(d) for d in detections]

        for track in self.tracks:
            track.predict()
            track.age += 1

        matches, unmatched = self._associate(detections)

        matched_tracks = set()

        for track, measurement in matches:
            track.update(measurement)
            track.hits += 1
            track.misses = 0

            if track.hits >= self.confirmation_hits:
                track.confirmed = True

            matched_tracks.add(track)

        for track in self.tracks:
            if track not in matched_tracks:
                track.misses += 1

        self.tracks = [
            t for t in self.tracks
            if t.misses < self.max_misses
        ]

        for det in unmatched:
            new_track = Track(self.next_id, det)
            self.tracks.append(new_track)
            self.next_id += 1

        return self.tracks
