# diagnostics/fault_injection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class FaultConfig:
    enabled: bool = False

    # fault toggles
    drop_iq_frames: float = 0.0              # probability [0..1]
    drop_measurements: float = 0.0           # probability [0..1]
    noise_spike_prob: float = 0.0            # probability [0..1]
    noise_spike_scale: float = 10.0          # multiplier
    cfar_pfa_override: Optional[float] = None
    freeze_tracker_updates: float = 0.0      # probability [0..1]

    rng_seed: Optional[int] = None


class FaultInjector:
    """
    Applies configured faults at well-defined injection points.

    Injection points:
      - IQ data: corrupt/add noise/drop frames
      - Detections/measurements: drop or perturb
      - Parameters: override CFAR settings
      - Tracker behavior: freeze updates
    """

    def __init__(self, config: FaultConfig | Dict[str, Any] | None = None):
        if config is None:
            config = FaultConfig()
        if isinstance(config, dict):
            config = FaultConfig(**config)
        self.cfg: FaultConfig = config
        self.rng = np.random.default_rng(self.cfg.rng_seed)

    def _p(self, prob: float) -> bool:
        if not self.cfg.enabled:
            return False
        return self.rng.random() < float(prob)

    # -------- IQ faults --------
    def maybe_drop_iq(self, iq_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Return None to simulate a dropped frame.
        """
        if self._p(self.cfg.drop_iq_frames):
            return None
        return iq_data

    def maybe_noise_spike(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Inject a noise burst into IQ.
        """
        if self._p(self.cfg.noise_spike_prob):
            spike = self.cfg.noise_spike_scale
            noise = (self.rng.standard_normal(iq_data.shape) + 1j * self.rng.standard_normal(iq_data.shape))
            iq_data = iq_data + spike * noise
        return iq_data

    # -------- Parameter faults --------
    def maybe_override_cfar(self, pfa: float) -> float:
        if self.cfg.enabled and self.cfg.cfar_pfa_override is not None:
            return float(self.cfg.cfar_pfa_override)
        return float(pfa)

    # -------- Measurement faults --------
    def maybe_drop_measurements(self, measurements: list[np.ndarray]) -> list[np.ndarray]:
        if not self.cfg.enabled:
            return measurements
        if self.cfg.drop_measurements <= 0:
            return measurements
        kept = []
        for m in measurements:
            if self.rng.random() >= float(self.cfg.drop_measurements):
                kept.append(m)
        return kept

    # -------- Tracker faults --------
    def freeze_tracker(self) -> bool:
        """
        If True, skip tracker update this frame (predict-only).
        """
        return self._p(self.cfg.freeze_tracker_updates)
