# Requirements Traceability Matrix (RTM)

This matrix links system requirements to their implementation and verification evidence.

| Req ID | Requirement                           | Implementation Location        | Verification Method | Evidence               | Status |
| ------ | ------------------------------------- | ------------------------------ | ------------------- | ---------------------- | ------ |
| R1     | Generate synthetic radar IQ data      | demo scripts                   | Demo                | multi-target demos     | ✅     |
| R2     | Produce range–doppler maps via FFT    | processing/pipeline.py         | Test + Demo         | RD plots               | ✅     |
| R3     | Detect targets using CFAR             | processing/detection.py        | Test + Demo         | CFAR mask outputs      | ✅     |
| R4     | Track multiple moving targets         | tracking/tracker_manager.py    | Demo                | persistent IDs         | ✅     |
| R5     | Maintain track identity across frames | tracker + KF                   | Demo                | stable labels          | ✅     |
| R6     | Handle missed detections gracefully   | tracker logic                  | Test                | max_misses behavior    | ✅     |
| R7     | Inject noise spikes into IQ data      | diagnostics/fault_injection.py | Test + Demo         | noise injection demo   | ✅     |
| R8     | Simulate jammer interference          | fault injector                 | Demo                | jammer demo            | ✅     |
| R9     | Drop measurements randomly            | fault injector                 | Test                | measurement drop tests | ✅     |
| R10    | Detect abnormal detection counts      | diagnostics/health_monitor.py  | Test                | raw CFAR triggers      | ✅     |
| R11    | Enter degraded mode on anomalies      | health monitor                 | Demo                | degraded state logs    | ✅     |
| R12    | Exit degraded mode after recovery     | health monitor                 | Test                | hysteresis behavior    | ✅     |
| R13    | Collect runtime metrics               | diagnostics/metrics.py         | Demo                | latency + counts       | ✅     |
| R14    | Visualize detections and tracks       | visualization module           | Demo                | live RD plots          | ✅     |
| R15    | Provide reproducible results          | RNG seed control               | Test                | deterministic runs     | ✅     |
| R16    | Support long-running simulation       | demos                          | Demo                | 200-frame runs         | ✅     |

## Notes

- Demos serve as primary system-level validation.
- Unit tests verify correctness of individual components.
- Metrics provide runtime observability and debugging insight.
