# Radar Signal Processing & Multi-Target Tracking Simulator

## Overview

This project is a modular radar simulation environment designed to demonstrate:

- Range–Doppler signal processing
- CFAR-based target detection
- Multi-target tracking using Kalman filters
- Fault injection (noise, drops, jammer interference)
- System health monitoring with degraded-mode behavior
- Metrics and live visualization

It is intended for technical demonstration, experimentation, and portfolio presentation.

---

## Key Features

- Synthetic radar IQ signal generation
- Range FFT + Doppler FFT pipeline
- 2D CFAR detection
- Multi-target tracking with persistent IDs
- Fault injection:
  - Noise spikes
  - Measurement drops
  - Jammer lines

- Health monitor:
  - Detects abnormal conditions
  - Enters/exits degraded mode

- Metrics collection:
  - Latency
  - Detection counts
  - Track statistics

- Interactive visual demos

---

## Project Structure

```
processing/        Signal processing pipeline (FFT + CFAR)
tracking/          Kalman filter tracker + manager
diagnostics/       Metrics, health monitoring, fault injection
visualization/     Range–Doppler plotting
demos/             End-to-end simulation scenarios
docs/              RTM, V&V, and documentation
tests/             Unit and integration tests
```

---

## Quick Start

### Install Dependencies

```
pip install -r requirements.txt
```

### Run a Demo

```
python -m demos.demo_multi_target_tracking
```

### Noise Injection Demo

```
python -m demos.demo_multi_target_tracking_noise_injection
```

### Jammer Simulation

```
python -m demos.demo_multi_target_tracking_jammer_injection
```

---

## Testing

Run all tests:

```
pytest -q
```

---

## Docker (Optional)

Build:

```
docker build -t radar-sim .
```

Run tests:

```
docker run radar-sim
```

Run a demo headless:

```
docker run -e HEADLESS=1 radar-sim python -m demos.demo_multi_target_tracking
```

---

## Verification & Validation

See:

- `docs/test_plan.md`
- `docs/vv_summary.md`
- `docs/rtm.md`

---

## Future Enhancements

- Clutter modeling
- Adaptive CFAR
- Multi-sensor fusion
- Real radar dataset integration
- Web-based visualization dashboard

---

## License

MIT License
