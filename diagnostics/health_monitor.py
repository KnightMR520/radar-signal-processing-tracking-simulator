# diagnostics/health_monitor.py
from __future__ import annotations
from dataclasses import dataclass

from diagnostics.metrics import CFAR_RAW_THIS_FRAME  # ✅ use the same key


@dataclass
class HealthConfig:
    raw_cfar_enter: int = 10_000
    raw_cfar_exit: int = 8_000
    raw_cfar_low_enter: int = -1
    raw_cfar_low_exit: int = -1
    enter_count_required: int = 3
    exit_count_required: int = 5


class HealthMonitor:
    def __init__(self, cfg: HealthConfig):
        self.cfg = cfg
        self.state = "NORMAL"
        self._enter_streak = 0
        self._exit_streak = 0
        self._last_reason = ""
        self._cause: str | None = None  # ✅ remember what caused DEGRADED ("low" or "high")

    def update(self, snap: dict) -> tuple[str, str]:
        # ✅ use same key constant as pipeline/demo
        raw_cfar = int(snap.get("gauges", {}).get(CFAR_RAW_THIS_FRAME, 0))

        too_high = raw_cfar >= self.cfg.raw_cfar_enter
        high_recovered = raw_cfar <= self.cfg.raw_cfar_exit

        too_low = (self.cfg.raw_cfar_low_enter >= 0) and (raw_cfar <= self.cfg.raw_cfar_low_enter)
        low_recovered = (self.cfg.raw_cfar_low_exit >= 0) and (raw_cfar >= self.cfg.raw_cfar_low_exit)

        if self.state == "NORMAL":
            if too_high:
                self._enter_streak += 1
                self._last_reason = f"raw_cfar_high={raw_cfar}"
                self._cause = "high"
            elif too_low:
                self._enter_streak += 1
                self._last_reason = f"raw_cfar_low={raw_cfar}"
                self._cause = "low"
            else:
                self._enter_streak = 0
                self._last_reason = ""
                self._cause = None

            if self._enter_streak >= self.cfg.enter_count_required:
                self.state = "DEGRADED"
                self._exit_streak = 0
                return self.state, self._last_reason

            return self.state, ""

        # DEGRADED: only recover based on the *cause* that put us here
        if self.state == "DEGRADED":
            if self._cause == "high":
                recovered = high_recovered and not too_high
            elif self._cause == "low":
                recovered = low_recovered and not too_low
            else:
                recovered = (high_recovered and not too_high) or (low_recovered and not too_low)

            if recovered:
                self._exit_streak += 1
            else:
                self._exit_streak = 0

            if self._exit_streak >= self.cfg.exit_count_required:
                self.state = "NORMAL"
                self._enter_streak = 0
                self._last_reason = ""
                self._cause = None
                return self.state, "recovered"

            return self.state, self._last_reason

        return self.state, ""
