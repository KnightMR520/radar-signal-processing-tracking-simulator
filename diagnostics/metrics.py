# diagnostics/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import time
from typing import Dict, Any


# -----------------------------
# Metric keys
# -----------------------------
TRACKS_ACTIVE = "tracks_active"
TRACKS_CONFIRMED = "tracks_confirmed"
PIPELINE_LATENCY = "pipeline_latency_s"
CFAR_DETECTIONS = "cfar_detections_total"
FAULTS_INJECTED = "faults_injected_total"

# NEW: raw CFAR count per frame (pre-suppression)
CFAR_RAW_THIS_FRAME = "cfar_raw_this_frame"


@dataclass
class _TimerStats:
    samples: deque  # of float seconds

    def add(self, x: float, maxlen: int):
        self.samples.append(x)
        while len(self.samples) > maxlen:
            self.samples.popleft()

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {"count": 0, "mean_s": 0.0, "p95_s": 0.0, "max_s": 0.0}
        arr = sorted(self.samples)
        n = len(arr)
        mean_s = sum(arr) / n
        p95_s = arr[int(0.95 * (n - 1))]
        max_s = arr[-1]
        return {"count": n, "mean_s": mean_s, "p95_s": p95_s, "max_s": max_s}


class MetricsRegistry:
    """
    Very small metrics registry:
      - counters: monotonically increasing
      - gauges: last value (overwrite)
      - timers: sliding-window timings (seconds)
    """

    def __init__(self, window_size: int = 200):
        self.window_size = int(window_size)
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, _TimerStats] = {}

    def inc(self, key: str, amount: int = 1) -> None:
        self._counters[key] = int(self._counters.get(key, 0)) + int(amount)

    def set_gauge(self, key: str, value: float) -> None:
        self._gauges[key] = float(value)

    def observe(self, key: str, value_s: float) -> None:
        if key not in self._timers:
            self._timers[key] = _TimerStats(samples=deque())
        self._timers[key].add(float(value_s), self.window_size)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "timers": {k: v.summary() for k, v in self._timers.items()},
        }


class Timer:
    """Context manager for timing blocks into MetricsRegistry timers (seconds)."""

    def __init__(self, metrics: MetricsRegistry, key: str):
        self.metrics = metrics
        self.key = key
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self._t0
        self.metrics.observe(self.key, dt)
        return False
