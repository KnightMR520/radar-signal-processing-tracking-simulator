import numpy as np
from radar.signal_generator import generate_scenario, ScenarioGenerator, generate_iq_returns


def test_generate_scenario_basic():
    config = {
        "radar": {
            "carrier_frequency": 10e9,
            "bandwidth": 20e6,
            "prf": 1000,
            "sampling_rate": 40e6,
            "tx_power": 1000,
            "antenna_gain": 30
        },
        "environment": {
            "noise_figure": 5,
            "clutter_density": 0.1,
            "interference_level": 0.05
        },
        "targets": [
            {
                "position": [1000, 0, 0],
                "velocity": [-50, 0, 0],
                "rcs": 1.0
            }
        ],
        "radar_position": [0, 0, 0],
        "radar_velocity": [0, 0, 0],
        "duration": 1.0,
        "time_step": 0.001
    }

    scenario = generate_scenario(config)

    assert scenario.radar_params.carrier_frequency == 10e9
    assert len(scenario.targets) == 1
    assert np.allclose(scenario.targets[0].position, [1000, 0, 0])
    assert scenario.duration == 1.0

def test_generate_target_trajectories():
    config = {
        "radar": {
            "carrier_frequency": 10e9,
            "bandwidth": 20e6,
            "prf": 1000,
            "sampling_rate": 40e6,
            "tx_power": 1000,
            "antenna_gain": 30
        },
        "environment": {
            "noise_figure": 5,
            "clutter_density": 0.1,
            "interference_level": 0.05
        },
        "targets": [
            {
                "position": [0, 0, 0],
                "velocity": [10, 0, 0],
                "rcs": 1.0
            }
        ],
        "radar_position": [0, 0, 0],
        "radar_velocity": [0, 0, 0],
        "duration": 1.0,
        "time_step": 0.1
    }

    scenario = generate_scenario(config)
    generator = ScenarioGenerator(scenario)
    trajectories = generator.generate_target_trajectories()

    positions = trajectories[0]

    expected = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0],
        [5, 0, 0],
        [6, 0, 0],
        [7, 0, 0],
        [8, 0, 0],
        [9, 0, 0],
    ])

    np.testing.assert_allclose(positions, expected, atol=1e-6)


def test_generate_iq_returns_shape_and_type():
    config = {
        "radar": {
            "carrier_frequency": 10e9,
            "bandwidth": 20e6,
            "prf": 1000,
            "sampling_rate": 40e6,
            "tx_power": 1000,
            "antenna_gain": 30
        },
        "environment": {
            "noise_figure": 5,
            "clutter_density": 0.1,
            "interference_level": 0.05
        },
        "targets": [
            {
                "position": [1000, 0, 0],
                "velocity": [-50, 0, 0],
                "rcs": 1.0
            }
        ],
        "radar_position": [0, 0, 0],
        "radar_velocity": [0, 0, 0],
        "duration": 1.0,
        "time_step": 0.01
    }

    scenario = generate_scenario(config)
    generator = ScenarioGenerator(scenario)
    trajectories = generator.generate_target_trajectories()

    iq_returns = generate_iq_returns(scenario, trajectories)

    assert isinstance(iq_returns, np.ndarray)
    assert iq_returns.dtype == np.complex128 or iq_returns.dtype == np.complex64
    assert iq_returns.shape[0] == int(scenario.duration / scenario.time_step)

def test_noise_and_interference_added():
    config = {
        "radar": {
            "carrier_frequency": 10e9,
            "bandwidth": 20e6,
            "prf": 1000,
            "sampling_rate": 40e6,
            "tx_power": 1000,
            "antenna_gain": 30
        },
        "environment": {
            "noise_figure": 10,
            "clutter_density": 0.2,
            "interference_level": 0.3
        },
        "targets": [
            {
                "position": [1000, 0, 0],
                "velocity": [-50, 0, 0],
                "rcs": 1.0
            }
        ],
        "radar_position": [0, 0, 0],
        "radar_velocity": [0, 0, 0],
        "duration": 1.0,
        "time_step": 0.001
    }

    scenario = generate_scenario(config)
    generator = ScenarioGenerator(scenario)
    trajectories = generator.generate_target_trajectories()
    iq = generate_iq_returns(scenario, trajectories)

    # Ensure non-zero power
    power = np.mean(np.abs(iq) ** 2)
    assert power > 0, "IQ signal should contain noise/clutter/interference power"
