# diagnostics/metrics.py
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, Optional
import time


@dataclass
class RollingWindow:
    """
    Fixed-size rolling window for numeric values.
    Used for computing rolling mean/max/min without heavy dependencies.
    """
    maxlen: int
    values: Deque[float] = field(default_factory=deque)

    def add(self, x: float) -> None:
        if self.maxlen <= 0:
            return
        if len(self.values) >= self.maxlen:
            self.values.popleft()
        self.values.append(float(x))

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def max(self) -> float:
        if not self.values:
            return 0.0
        return max(self.values)

    def min(self) -> float:
        if not self.values:
            return 0.0
        return min(self.values)

    def count(self) -> int:
        return len(self.values)


class MetricsRegistry:
    """
    Simple in-process metrics registry.

    Types:
      - counters: monotonically increasing
      - gauges: instantaneous numeric values
      - timers: durations (records into rolling window)
    """

    def __init__(self, window_size: int = 200):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, RollingWindow] = {}
        self._window_size = int(window_size)

    # -------- Counters --------
    def inc(self, name: str, amount: int = 1) -> None:
        self._counters[name] = int(self._counters.get(name, 0) + amount)

    def get_counter(self, name: str) -> int:
        return int(self._counters.get(name, 0))

    # -------- Gauges --------
    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = float(value)

    def get_gauge(self, name: str) -> float:
        return float(self._gauges.get(name, 0.0))

    # -------- Timers --------
    def observe(self, name: str, seconds: float) -> None:
        if name not in self._timers:
            self._timers[name] = RollingWindow(self._window_size)
        self._timers[name].add(seconds)

    def timer_mean(self, name: str) -> float:
        return self._timers[name].mean() if name in self._timers else 0.0

    def timer_max(self, name: str) -> float:
        return self._timers[name].max() if name in self._timers else 0.0

    def snapshot(self) -> dict:
        """
        Stable snapshot for logging / tests.
        """
        timers = {
            k: {
                "count": w.count(),
                "mean_s": w.mean(),
                "max_s": w.max(),
                "min_s": w.min(),
            }
            for k, w in self._timers.items()
        }
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "timers": timers,
        }


class Timer:
    """
    Context manager for timing blocks.
    """
    def __init__(self, registry: MetricsRegistry, name: str):
        self.registry = registry
        self.name = name
        self.t0: Optional[float] = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.t0 is None:
            return
        dt = time.perf_counter() - self.t0
        self.registry.observe(self.name, dt)


# Convenience constants for names (helps consistency)
PIPELINE_LATENCY = "pipeline.latency_s"
CFAR_DETECTIONS = "cfar.detections"
TRACKS_ACTIVE = "tracks.active"
TRACKS_CONFIRMED = "tracks.confirmed"
FAULTS_INJECTED = "faults.injected"
