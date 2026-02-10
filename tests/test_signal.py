import numpy as np
from radar.signal_generator import generate_scenario


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
