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

    # OLD "uniform noise spike" (kept, but not great for CFAR blow-up)
    noise_spike_prob: float = 0.0            # probability [0..1]
    noise_spike_scale: float = 10.0          # multiplier

    # NEW: structured interference that *will* create many detections
    jammer_line_prob: float = 0.0            # probability [0..1]
    jammer_line_scale: float = 50.0          # amplitude of jammer
    jammer_mode: str = "range"               # "range" or "doppler"
    jammer_bin: Optional[int] = None         # if None, random each time

    cfar_pfa_override: Optional[float] = None
    freeze_tracker_updates: float = 0.0      # probability [0..1]

    rng_seed: Optional[int] = None


class FaultInjector:
    """
    Applies configured faults at well-defined injection points.

    Key note:
      - Uniformly increasing noise often does NOT increase CFAR detections
        because CA-CFAR adapts its threshold to the noise estimate.
      - Structured interference (range line / Doppler line) DOES create many detections.
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
        """Return None to simulate a dropped frame."""
        if self._p(self.cfg.drop_iq_frames):
            return None
        return iq_data

    def maybe_noise_spike(self, iq_data: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Inject a noise burst into IQ.

        Returns:
            (iq_out, injected)
        """
        if self._p(self.cfg.noise_spike_prob):
            spike = float(self.cfg.noise_spike_scale)
            noise = (
                self.rng.standard_normal(iq_data.shape)
                + 1j * self.rng.standard_normal(iq_data.shape)
            )
            # Make a NEW array (do not mutate input in-place)
            return (iq_data + spike * noise), True

        return iq_data, False
    
    
    def maybe_jammer_line(self, iq_data: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Inject structured interference (range or doppler line).

        Returns:
            (iq_out, injected)
        """
        if self._p(self.cfg.jammer_line_prob):
            return self._inject_jammer_line(iq_data), True
        return iq_data, False



    def _inject_jammer_line(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Create a coherent tone that appears as:
          - a vertical ridge at a given RANGE bin, or
          - a horizontal ridge at a given DOPPLER bin (after FFTshift)

        That kind of structured interference tends to produce lots of CFAR hits.
        """
        num_pulses, num_samples = iq_data.shape
        scale = float(self.cfg.jammer_line_scale)
        mode = str(self.cfg.jammer_mode).lower().strip()

        # choose a bin
        if self.cfg.jammer_bin is None:
            if mode == "range":
                k = int(self.rng.integers(0, num_samples))
            elif mode == "doppler":
                k = int(self.rng.integers(0, num_pulses))
            else:
                k = int(self.rng.integers(0, max(num_pulses, num_samples)))
        else:
            k = int(self.cfg.jammer_bin)

        # random phase helps avoid always identical frames
        phi0 = float(self.rng.random() * 2.0 * np.pi)

        if mode == "range":
            # Fast-time sinusoid at normalized frequency k/N -> shows up as RANGE bin k
            n = np.arange(num_samples, dtype=float)
            tone_fast = np.exp(1j * (2.0 * np.pi * (k / max(num_samples, 1)) * n + phi0))
            jammer = np.tile(tone_fast, (num_pulses, 1))

        elif mode == "doppler":
            # Slow-time sinusoid across pulses -> shows up at a Doppler bin
            p = np.arange(num_pulses, dtype=float)
            tone_slow = np.exp(1j * (2.0 * np.pi * (k / max(num_pulses, 1)) * p + phi0))
            jammer = tone_slow[:, None] * np.ones((num_pulses, num_samples), dtype=complex)

        else:
            # fallback: impulsive burst in a random rectangular patch
            p0 = int(self.rng.integers(0, num_pulses))
            s0 = int(self.rng.integers(0, num_samples))
            jammer = np.zeros_like(iq_data)
            jammer[p0, s0] = np.exp(1j * phi0)

        return iq_data + scale * jammer

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
        """If True, skip tracker update this frame (predict-only)."""
        return self._p(self.cfg.freeze_tracker_updates)
