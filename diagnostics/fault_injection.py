# diagnostics/fault_injection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class FaultConfig:
    enabled: bool = False

    drop_iq_frames: float = 0.0
    drop_measurements: float = 0.0

    # Noise spike (NEW: mode + patch sizing)
    noise_spike_prob: float = 0.0
    noise_spike_scale: float = 10.0
    noise_spike_mode: str = "uniform"   # "uniform" | "patch" | "impulse"
    noise_patch_pulses: int = 8         # used when mode="patch"
    noise_patch_samples: int = 16       # used when mode="patch"
    impulse_density: float = 0.02       # used when mode="impulse" (fraction of cells)

    # NEW: structured interference that will create many detections
    jammer_line_prob: float = 0.0            # probability [0..1]
    jammer_line_scale: float = 50.0          # amplitude of jammer
    jammer_mode: str = "range"               # "range" or "doppler"
    jammer_bin: Optional[int] = None         # if None, random each time

    cfar_pfa_override: Optional[float] = None
    freeze_tracker_updates: float = 0.0

    rng_seed: Optional[int] = None


class FaultInjector:
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

    def maybe_drop_iq(self, iq_data: np.ndarray) -> Optional[np.ndarray]:
        if self._p(self.cfg.drop_iq_frames):
            return None
        return iq_data

    def maybe_noise_spike(self, iq_data: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Returns (iq_out, injected).
        """
        if not self._p(self.cfg.noise_spike_prob):
            return iq_data, False

        scale = float(self.cfg.noise_spike_scale)
        mode = str(self.cfg.noise_spike_mode).lower().strip()

        P, N = iq_data.shape
        noise = (self.rng.standard_normal((P, N)) + 1j * self.rng.standard_normal((P, N)))

        if mode == "uniform":
            # Old behavior: CFAR often adapts, so detections may NOT explode.
            return (iq_data + scale * noise), True

        if mode == "patch":
            # Burst in a random rectangular region (creates many outliers vs neighbors)
            hp = int(np.clip(self.cfg.noise_patch_pulses, 1, P))
            hn = int(np.clip(self.cfg.noise_patch_samples, 1, N))
            p0 = int(self.rng.integers(0, P - hp + 1))
            n0 = int(self.rng.integers(0, N - hn + 1))

            out = iq_data.copy()
            out[p0:p0 + hp, n0:n0 + hn] += scale * noise[p0:p0 + hp, n0:n0 + hn]
            return out, True

        if mode == "impulse":
            # Sparse impulsive cells (very CFAR-unfriendly)
            density = float(self.cfg.impulse_density)
            density = float(np.clip(density, 0.0, 1.0))
            mask = self.rng.random((P, N)) < density

            out = iq_data.copy()
            out[mask] += scale * noise[mask]
            return out, True

        # Fallback: behave like uniform
        return (iq_data + scale * noise), True
    
    def maybe_jammer_line(self, iq_data: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Inject a structured jammer ridge (range or doppler line).
        Returns (iq_out, injected)
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

    def maybe_override_cfar(self, pfa: float) -> float:
        if self.cfg.enabled and self.cfg.cfar_pfa_override is not None:
            return float(self.cfg.cfar_pfa_override)
        return float(pfa)

    def maybe_drop_measurements(self, measurements: list[np.ndarray]) -> list[np.ndarray]:
        if not self.cfg.enabled or self.cfg.drop_measurements <= 0:
            return measurements
        kept = []
        for m in measurements:
            if self.rng.random() >= float(self.cfg.drop_measurements):
                kept.append(m)
        return kept

    def freeze_tracker(self) -> bool:
        return self._p(self.cfg.freeze_tracker_updates)
