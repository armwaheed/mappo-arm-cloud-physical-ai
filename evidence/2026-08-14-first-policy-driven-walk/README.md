<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-14 — the first time the policy moved a leg

Eleven live runs on the Go2, tethered, with the D1 arm fitted, on office carpet. The
policy drove the robot to a chair **2.78 m away with 0.6° of yaw drift**, and it took
five failed runs and a wrong diagnosis to get there. Both are in here, because the
failures are the more useful half.

## The run that worked

`mappo_drive.py --live --calibration … --goal-class chair --goal-height 1.067
--confidence 0.45 --robot-radius 0.25 --no-latch-arm --arrive 0.8 --max-seconds 45`

| | |
| --- | --- |
| outcome | **arrived, 0.77 m from the goal** |
| ticks driven by the policy | **53 / 53 — 0 vetoed, 0 stopped** |
| travel | 2.78 m path, 2.70 m straight line (ratio 1.03) |
| goal range | 3.41 m → 0.77 m, monotonic |
| yaw drift | **+0.6° over the whole walk** |
| measured speed | mean 0.240 m/s, peak 0.415 |

`{'policy': 53}` is the whole claim: the incumbent planner never took a tick, so this is
the checkpoint driving the legs end to end rather than a supervised blend.

## Files

| file | what it is |
| --- | --- |
| `hero-third-person.gif` | the arriving run, filmed from behind |
| `hero-onboard.gif` | the same run from the robot's own camera, with the planner overlay — every frame reads `cmd policy` |
| `hero-run-telemetry.jsonl` | `go2.visual_nav.telemetry/1` for the arriving run |
| `hero-run-leg-encoders.jsonl` | joint encoders + odom for the same run, downsampled from ~500 Hz to 10 Hz |
| `stalled-run-leg-encoders.jsonl` | the same, for an identical run at 0.21 m/s that stalled |
| `gait-floor.png` | the mechanism — knee swing and forward speed, 0.21 vs 0.35 m/s |
| `stall-distance.png` | distance travelled before stopping, seven runs |
| `make_charts.py` | regenerates both figures from the two encoder logs |

## What the encoder logs are for

Everything else on this robot — odometry, `measured` velocity, the stall gate — comes out
of the same state estimator, so when it says the robot is not moving, those three agree
with each other and prove nothing. The joint encoders are upstream of it: `motor_state[i].q`
is a shaft angle. They are the only instrument here that can distinguish *held* from
*not trying*, and they are what settled the diagnosis after four wrong ones.

Regenerate the figures with:

```bash
python3 make_charts.py        # expects the two encoder logs beside it
```
