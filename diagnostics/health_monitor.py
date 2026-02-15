# diagnostics/health_monitor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from diagnostics.metrics import PIPELINE_LATENCY, CFAR_RAW_THIS_FRAME


@dataclass
class HealthConfig:
    # Enter/exit thresholds
    latency_mean_ms_enter: float = 25.0
    latency_mean_ms_exit: float = 15.0

    raw_cfar_enter: float = 80.0     # per-frame raw CFAR hits
    raw_cfar_exit: float = 40.0

    enter_count_required: int = 3
    exit_count_required: int = 5


class HealthMonitor:
    """
    Simple hysteresis state machine:
      NORMAL -> DEGRADED when conditions persist for N frames
      DEGRADED -> NORMAL when conditions relax for M frames

    It evaluates:
      - latency mean (ms) from timer PIPELINE_LATENCY
      - raw CFAR hits this frame from gauge CFAR_RAW_THIS_FRAME
    """

    def __init__(self, cfg: HealthConfig):
        self.cfg = cfg
        self.state = "NORMAL"
        self._enter_streak = 0
        self._exit_streak = 0

    def update(self, snap: Dict[str, Any]) -> Tuple[str, str]:
        timers = snap.get("timers", {})
        gauges = snap.get("gauges", {})

        mean_s = float(timers.get(PIPELINE_LATENCY, {}).get("mean_s", 0.0))
        lat_ms = 1000.0 * mean_s

        raw_cfar = float(gauges.get(CFAR_RAW_THIS_FRAME, 0.0))

        # Decide if we should be degraded based on current frame
        should_degrade = (lat_ms >= self.cfg.latency_mean_ms_enter) or (raw_cfar >= self.cfg.raw_cfar_enter)
        should_recover = (lat_ms <= self.cfg.latency_mean_ms_exit) and (raw_cfar <= self.cfg.raw_cfar_exit)

        reason = ""
        if lat_ms >= self.cfg.latency_mean_ms_enter:
            reason += f"lat_ms={lat_ms:.1f} "
        if raw_cfar >= self.cfg.raw_cfar_enter:
            reason += f"raw_cfar={raw_cfar:.0f} "

        if self.state == "NORMAL":
            if should_degrade:
                self._enter_streak += 1
                if self._enter_streak >= self.cfg.enter_count_required:
                    self.state = "DEGRADED"
                    self._exit_streak = 0
            else:
                self._enter_streak = 0
        else:
            if should_recover:
                self._exit_streak += 1
                if self._exit_streak >= self.cfg.exit_count_required:
                    self.state = "NORMAL"
                    self._enter_streak = 0
            else:
                self._exit_streak = 0

        return self.state, reason.strip()
