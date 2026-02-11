# radar/signal_generator.py

from dataclasses import dataclass
import numpy as np
from typing import List, Dict


@dataclass
class RadarParams:
    carrier_frequency: float  # Hz
    bandwidth: float          # Hz
    prf: float                # Hz
    sampling_rate: float      # Hz
    tx_power: float           # Watts
    antenna_gain: float       # dB


@dataclass
class Target:
    position: np.ndarray      # [x, y, z] meters
    velocity: np.ndarray      # [vx, vy, vz] m/s
    rcs: float                # Radar Cross Section (m^2)


@dataclass
class Environment:
    noise_figure: float       # dB
    clutter_density: float    # unitless (0–1)
    interference_level: float # unitless (0–1)


@dataclass
class Scenario:
    radar_params: RadarParams
    radar_position: np.ndarray
    radar_velocity: np.ndarray
    targets: List[Target]
    environment: Environment
    duration: float           # seconds
    time_step: float          # seconds


def generate_scenario(config: Dict) -> Scenario:
    """
    Generates a simulation scenario from a configuration dictionary.

    This function acts as the system boundary between configuration input
    and simulation execution.
    """
    radar_cfg = config["radar"]
    env_cfg = config["environment"]
    tgt_cfgs = config["targets"]

    radar_params = RadarParams(
        carrier_frequency=radar_cfg["carrier_frequency"],
        bandwidth=radar_cfg["bandwidth"],
        prf=radar_cfg["prf"],
        sampling_rate=radar_cfg["sampling_rate"],
        tx_power=radar_cfg["tx_power"],
        antenna_gain=radar_cfg["antenna_gain"]
    )

    environment = Environment(
        noise_figure=env_cfg["noise_figure"],
        clutter_density=env_cfg["clutter_density"],
        interference_level=env_cfg["interference_level"]
    )

    targets = []
    for tgt in tgt_cfgs:
        targets.append(Target(
            position=np.array(tgt["position"], dtype=float),
            velocity=np.array(tgt["velocity"], dtype=float),
            rcs=float(tgt["rcs"])
        ))

    scenario = Scenario(
        radar_params=radar_params,
        radar_position=np.array(config["radar_position"], dtype=float),
        radar_velocity=np.array(config["radar_velocity"], dtype=float),
        targets=targets,
        environment=environment,
        duration=float(config["duration"]),
        time_step=float(config["time_step"])
    )

    return scenario

class ScenarioGenerator:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.time_vector = np.arange(0, scenario.duration, scenario.time_step)

    def generate_target_trajectories(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: shape (num_targets, num_time_steps, 3)
        """
        t = np.arange(0, self.scenario.duration, self.scenario.time_step)
        num_steps = len(t)
        num_targets = len(self.scenario.targets)
        
        trajectories = np.zeros((num_targets, num_steps, 3))
        
        for i, tgt in enumerate(self.scenario.targets):
            start_pos = tgt.position
            vel = tgt.velocity
            # linear trajectory: pos = pos0 + vel * t
            trajectories[i] = start_pos + np.outer(t, vel)
        
        return trajectories

def generate_iq_returns(scenario: Scenario, trajectories: np.ndarray) -> np.ndarray:
    """
    Generates complex IQ radar returns for all targets over time,
    including noise, clutter, and interference.
    """
    c = 3e8  # speed of light (m/s)
    t = np.arange(0, scenario.duration, scenario.time_step)
    num_samples = len(t)
    iq_signal = np.zeros(num_samples, dtype=complex)

    num_targets = trajectories.shape[0]
    radar_pos = np.array(scenario.radar_position)

    for target_idx in range(num_targets):
        positions = trajectories[target_idx]
        rcs = scenario.targets[target_idx].rcs

        for i, pos in enumerate(positions):
            range_to_target = np.linalg.norm(pos - radar_pos)
            delay = 2 * range_to_target / c
            phase = 2 * np.pi * scenario.radar_params.carrier_frequency * delay

            amplitude = rcs / (range_to_target ** 4 + 1e-6)
            iq_signal[i] += amplitude * np.exp(1j * phase)

    # Add noise, clutter, and interference
    env = scenario.environment

    noise = generate_thermal_noise(
        num_samples,
        noise_figure_db=env.noise_figure,
        bandwidth=scenario.radar_params.bandwidth,
    )

    clutter = generate_clutter(
        num_samples,
        clutter_density=env.clutter_density,
    )

    interference = generate_interference(
        num_samples,
        interference_level=env.interference_level,
    )

    iq_signal += noise + clutter + interference

    return iq_signal

def generate_thermal_noise(num_samples, noise_figure_db, bandwidth):
    """
    Generate complex thermal noise based on kTB and noise figure.
    """
    k = 1.38e-23  # Boltzmann constant
    T = 290       # Standard noise temperature (K)
    noise_power = k * T * bandwidth
    noise_figure_linear = 10 ** (noise_figure_db / 10)
    total_noise_power = noise_power * noise_figure_linear

    sigma = np.sqrt(total_noise_power / 2)
    noise = sigma * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    return noise


def generate_clutter(num_samples, clutter_density):
    """
    Generate clutter as low-frequency correlated complex noise.
    """
    white = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)

    # Simple low-pass filter for correlation (moving average)
    kernel_size = max(1, int(clutter_density * 20))
    kernel = np.ones(kernel_size) / kernel_size
    clutter = np.convolve(white, kernel, mode="same")

    return clutter


def generate_interference(num_samples, interference_level):
    """
    Generate structured interference (e.g., narrowband jammer).
    """
    t = np.arange(num_samples)
    freq = np.random.uniform(0.01, 0.1)  # normalized frequency
    phase = np.random.uniform(0, 2 * np.pi)
    tone = np.exp(1j * (2 * np.pi * freq * t + phase))

    amplitude = interference_level * np.random.uniform(0.5, 1.5)
    interference = amplitude * tone

    return interference