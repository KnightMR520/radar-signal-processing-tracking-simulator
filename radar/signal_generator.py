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

    Parameters:
        scenario: Scenario object with radar and environment parameters
        trajectories: np.ndarray of shape (num_targets, num_time_steps, 3) containing
                      target positions at each time step

    Returns:
        iq_signal: complex ndarray of IQ samples
    """
    c = 3e8  # speed of light (m/s)
    t = np.arange(0, scenario.duration, scenario.time_step)
    iq_signal = np.zeros(len(t), dtype=complex)

    num_targets = trajectories.shape[0]

    for i in range(num_targets):
        pos = trajectories[i]  # shape: (num_time_steps, 3)
        # Compute instantaneous range from radar
        ranges = np.linalg.norm(pos - scenario.radar_position, axis=1)

        # Round-trip delay (for future pulse compression)
        range_delays = 2 * ranges / c

        # Doppler approximation: use velocity along line-of-sight
        # Compute velocity as finite difference along trajectory
        vel = np.gradient(pos, scenario.time_step, axis=0)
        rel_vel = vel - scenario.radar_velocity
        los_unit = (pos - scenario.radar_position) / (ranges[:, np.newaxis] + 1e-6)
        doppler_freq = 2 * np.sum(rel_vel * los_unit, axis=1) / (c / scenario.radar_params.carrier_frequency)

        # Amplitude model
        amplitude = np.sqrt(scenario.targets[i].rcs) / (ranges ** 2 + 1e-6)
        phase = 2 * np.pi * doppler_freq * t

        iq_signal += amplitude * np.exp(1j * phase)

    # --- Noise model ---
    noise_power = 10 ** (scenario.environment.noise_figure / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    iq_signal += noise

    # --- Clutter model ---
    clutter_power = scenario.environment.clutter_density
    clutter = np.sqrt(clutter_power / 2) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    iq_signal += clutter

    # --- Interference model ---
    interference_power = scenario.environment.interference_level
    interference = np.sqrt(interference_power / 2) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    iq_signal += interference

    return iq_signal
