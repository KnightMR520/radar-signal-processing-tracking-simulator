# tests/test_fault_injection.py
import numpy as np

from diagnostics.fault_injection import FaultInjector, FaultConfig


def test_drop_iq_frames_returns_none_when_enabled():
    cfg = FaultConfig(enabled=True, drop_iq_frames=1.0, rng_seed=123)
    fi = FaultInjector(cfg)

    iq = np.zeros((4, 8), dtype=np.complex128)
    assert fi.maybe_drop_iq(iq) is None


def test_noise_spike_changes_iq_when_prob_1():
    cfg = FaultConfig(enabled=True, noise_spike_prob=1.0, noise_spike_scale=5.0, rng_seed=123)
    fi = FaultInjector(cfg)

    iq = np.zeros((4, 8), dtype=np.complex128)
    out = fi.maybe_noise_spike(iq)
    assert np.any(np.abs(out) > 0)


def test_drop_measurements_reduces_list():
    cfg = FaultConfig(enabled=True, drop_measurements=1.0, rng_seed=123)
    fi = FaultInjector(cfg)

    meas = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    out = fi.maybe_drop_measurements(meas)
    assert out == []
