<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-27 — 90% of a run was commanded at speeds the robot cannot walk at

`run-20260827T012702Z-00652ea.jsonl` is the telemetry of one live Go2 run, pulled off
the robot before it went offline. It is the measurement behind issue #26's gait-floor
guard. Reproduce every number below with no robot:

```bash
(cd ../../robot-stack/unitree/go2/visual_nav && python3 test_avoidance.py)
```

Two tests in that file read this file directly:
`test_the_guard_fires_on_the_recorded_run_before_the_stall_gate_did` and
`test_the_arriving_runs_on_record_do_not_trip_the_guard`.

## What the run did

Envelope `vx<=0.35 vy<=0.20 wz<=0.70`, `--robot-radius 0.25`, goal 3.00 m ahead, one
staged bin, one person in frame. It ended after 13.5 s with

> `stalled: commanded 0.11 m/s for 4.1s and moved 0.05 m of an expected 0.46 m. Something
> is holding the robot — check the tether, and whether it has walked into something this
> pipeline cannot see.`

**The tether was fine.** The robot was 0.72 m from the bin and had been asked to walk at
speeds it does not have.

| reason | ticks | mean commanded vx | below the 0.35 m/s gait floor |
| --- | ---: | ---: | ---: |
| `policy` | 7 | 0.350 | 0 of 7 |
| **`veto-avoid`** | **37** | **0.181** | **31 of 37** |
| `hold` / `veto-hold` | 16 | 0.000 | — |

90% of all commanded ticks were at or below the floor. Nothing faulted.

## The mechanism, which is not the one the issue assumed

Issue #26 attributes the crawl to the stopping-distance cap ratcheting the window shut
as the robot closes on an obstacle. **On this run the cap was never the constraint.**
Recomputed per tick from the recorded poses and obstacle list, it stood at 0.38-0.39 m/s
through the whole freeze — above the floor. The binding constraint was the dynamic
window's own acceleration limit, anchored at zero:

* Perception ran at ~3 Hz (a 313 ms detect cycle) against a 10 Hz control loop, so
  `perception_timeout_s` fired on **26 of 86 ticks** and each one commanded a stop.
* Every stop re-anchored `avoidance.DynamicWindowPlanner._window` at `vx0 = 0`, where one
  control period of `accel_x = 0.50` reaches 0.05 m/s.
* So the last 3.4 s of the run is a two-tick ramp, restarted eighteen times:

  ```
  0.052 0.103 | 0.050 0.101 | 0.051 | 0.052 0.103 | 0.052 0.104 | 0.050 0.101 | ...
  ```

  It never once got past 0.104 m/s. The robot moved 0.012 m in 3.4 s.

`visual_nav._control_dt`'s own docstring describes this failure — *"never reached the
0.35 m/s gait floor before the next stale-perception hold zeroed it"* — for the case
where the LOOP is slow. This run had a 10 Hz loop and 3 Hz perception, which reaches the
same place by a road that fix does not cover.

## What the guard does with it

`avoidance.DynamicWindowPlanner._gait_floor_stop`, replayed over these bytes in order:

| | |
| --- | --- |
| fires at | **t = 11.44 s**, refusing 0.103 m/s |
| the run's own outcome line | t = 13.48 s |
| margin | **2.04 s**, so the cause is on the screen above the outcome |
| ticks that become a deliberate stop instead of a crawl | 12 of 60 |

## The control, which is the half that makes the number mean something

Every recorded run in `evidence/`, replayed through the same guard, sorted by how much
it fires:

| run | live | ticks | guard fires | outcome |
| --- | :-: | ---: | ---: | --- |
| `2026-08-18-threading-two-bins/run11-SUCCESS-threaded-the-gap` | yes | 66 | **0** | arrived (0.77 m from goal) |
| `2026-08-18-swerve-width-and-veto-precedence/runA-scale2.0-veto-on-servo-on` | yes | 48 | **0** | arrived (0.75 m from goal) |
| `2026-08-18-swerve-width-and-veto-precedence/runB-scale2.0-veto-off-servo-on` | yes | 66 | **0** | arrived (0.75 m from goal) |
| `2026-08-17-corridor-and-room-runs/run0-planner-baseline-corridor` | yes | 77 | **0** | arrived (0.80 m from goal) |
| `2026-08-14-first-policy-driven-walk/hero-run-telemetry` | yes | 53 | **0** | arrived (0.77 m from goal) |
| `2026-08-25-peer-runs/contrast-run-telemetry` | yes | 89 | **0** | arrived (0.30 m from goal) |
| `2026-08-25-peer-runs/hero-run-telemetry` | yes | 57 | **0** | arrived (0.26 m from goal) |
| `evidence/live_run_telemetry` | yes | 116 | **0** | stalled: commanded 0.36 m/s for 4.0s and moved 0.28 m of a |
| `evidence/sample_telemetry` | yes | 116 | **0** | stalled: commanded 0.36 m/s for 4.0s and moved 0.28 m of a |
| `2026-08-18-threading-two-bins/run9-full-speed-reverse-hazard` | yes | 29 | **0** | stalled: commanded 0.35 m/s for 4.1s and moved 0.26 m of a |
| `2026-08-18-swerve-width-and-veto-precedence/runC-scale2.0-veto-on-servo-off` | yes | 67 | **0** | timeout after 25s |
| `2026-08-17-corridor-and-room-runs/dryrun-room-scene-check` | no | 127 | **0** | timeout after 15s |
| `2026-08-17-corridor-and-room-runs/run1-mappo-corridor-wall-contact` | yes | 47 | **0** | stalled: commanded 0.35 m/s for 4.2s and moved 0.23 m of a |
| `2026-08-17-corridor-and-room-runs/run3-control-dt-fix-corridor` | yes | 45 | **0** | stalled: commanded 0.35 m/s for 4.2s and moved 0.29 m of a |
| `2026-08-17-corridor-and-room-runs/run4-room-cabinet-contact` | yes | 29 | **0** | stalled: commanded 0.35 m/s for 4.0s and moved 0.19 m of a |
| `2026-08-17-corridor-and-room-runs/run5-room-policy-driven-success` | yes | 58 | **0** | timeout after 20s |
| `2026-08-17-corridor-and-room-runs/run2-maxvy010-gait-floor-stall` | yes | 19 | **4** | stalled: commanded 0.36 m/s for 4.0s and moved 0.27 m of a |
| `2026-08-18-threading-two-bins/run7-veto-shortened-policy-stall` | yes | 34 | **6** | stalled: commanded 0.10 m/s for 4.2s and moved 0.08 m of a |
| `2026-08-18-threading-two-bins/run10-forward-clamp-pure-strafe` | yes | 32 | **9** | stalled: commanded 0.08 m/s for 4.2s and moved 0.05 m of a |
| `2026-08-18-threading-two-bins/run14-ray0-blocked-policy-retreats` | yes | 32 | **9** | stalled: commanded 0.06 m/s for 4.1s and moved 0.01 m of a |
| `2026-08-18-threading-two-bins/run15-aimed-8deg-off-corridor` | yes | 28 | **10** | stalled: commanded 0.05 m/s for 4.1s and moved 0.01 m of a |
| `2026-08-27-gait-floor-freeze/run-20260827T012702Z-00652ea` | yes | 60 | **12** | stalled: commanded 0.11 m/s for 4.1s and moved 0.05 m of a |
| `2026-08-18-threading-two-bins/run1-gait-floor-stall-veto-crawl` | yes | 45 | **15** | stalled: commanded 0.13 m/s for 4.1s and moved 0.04 m of a |
| `2026-08-18-threading-two-bins/run13-person-track-covers-goal` | yes | 165 | **129** | timeout after 40s |
| `2026-08-17-corridor-and-room-runs/dryrun-corridor-scene-check` | no | 172 | **153** | timeout after 20s |

Read it as three groups.

**Zero on all seven runs that arrived.** A guard that fires on the successes as well as
the stall is not a guard, and this repository has shipped both a check that could never
fail and a threshold under its own sensor's noise floor.

**Zero on every run that stalled while being commanded at or above the floor.** Those are
the ones where something really was holding the robot — a wall, a cabinet, a taut tether
— and the existing stall message is right about them. The guard correctly declines to
claim them.

**It fires on every run that stalled while being commanded below the floor**, which is
six of them plus this one. Two are worth reading twice:

* `run2-maxvy010-gait-floor-stall` fires 4 times although its outcome line quotes
  0.36 m/s. The outcome quotes the LAST command; the ticks before it were not that.
* `run13-person-track-covers-goal` ran for 40 seconds, live, standing under the 3.15 kg
  arm, commanded sub-floor throughout — and reported `timeout`. Nothing in the stack had
  a word for what happened to it.

The two `--live`-less scene checks are the false-positive class: a dry run's legs never
move, so "commanded to move and not moving" is true of every tick by construction. That
is why `visual_nav._blocked_reason` clears the guard under exactly the condition it
already declines to judge itself, and why `test_a_dry_run_clears_the_planners_gait_floor_
guard_as_well` asserts it against a real planner rather than a fake.

## What is still open

`MIN_GAIT_COMMAND_M_S = 0.35` is still a guess — "the lowest speed observed to work" on
2026-08-14 — and run C of 2026-08-18 walked 3 m at a sustained 0.295 with a minimum of
0.189. The guard is deliberately built not to depend on the value; issue #26 proposal 2
is the sweep that would replace it, and it still has not been on a robot.
