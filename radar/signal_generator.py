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
