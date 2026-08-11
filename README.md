<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multi-Agent Proximal Policy Optimization (MAPPO) using Arm Cloud AI and Arm Physical AI

A MAPPO policy, trained in simulation, driving real quadrupeds to goals in a shared
room. This repository is the meeting point: the **robot-side control stack** that
perceives and moves, the **telemetry contract** between it and the policy, and the
**adapter** that turns one into the other.

Two people, two halves:

| | |
| --- | --- |
| **Waheed Brown** ([@armwaheed](https://github.com/armwaheed)) | robot stack — perception, planning, safety, telemetry |
| **Sagar Surendran** ([@spsagar13](https://github.com/spsagar13)) | MAPPO policy, training environment, checkpoints |

## What actually works today

One Go2 walks to a goal on **RGB alone** — no LiDAR, no depth, no motion capture — and
gives way to people crossing its path. Hardware-verified:

![Go2 walking to a goal and giving way to a person](evidence/go2_nav_run.mp4)

`evidence/go2_nav_run.mp4` + `.log` — 2.0 m to a dead-reckoned waypoint, `arrived
(0.96 m from goal)`, 145 perception cycles, 0 errors, motors 31 → 32 °C. This is the run
Sagar reviewed.

Since then, the stack also **maps a static obstacle and goes around it** to a **detected**
goal (`evidence/dry13.mp4`). That part is dry-run verified but has **not walked** — see
*Status* below.

## The interface: `--telemetry`, not the console log

**Do not parse the console log.** It is prose, and it does not carry what it appears to.
Counted over the 107 control ticks of the run above:

| what a policy needs | in the console log |
| --- | --- |
| motion commands | ✅ every tick |
| goal | ⚠️ a scalar *distance* — no position |
| odometry / pose | ❌ **once**, in a start-up banner |
| camera data | ❌ none (`lat=235ms` is a frame's *age*) |

It is also edited to stay readable — `people=0` became `obst=[binx1,personx1]` inside a
week, because a bare count could no longer distinguish a mapped bin from a coasting
ghost. That is right for prose and fatal for a parser.

So the stack writes JSONL instead, one object per control tick, versioned:

```bash
python3 visual_nav.py ... --record run.mp4 --telemetry run.jsonl
```

```json
{"type": "tick", "t": 6.46, "pose": {"x": -2.464, "y": 2.112, "yaw": -1.613},
 "goal": {"x": -2.332, "y": -0.800, "distance_m": 2.915},
 "obstacles": [{"label": "bin", "x": -2.276, "y": -0.081, "vx": 0.0, "vy": 0.0,
                "radius_m": 0.23}],
 "command": {"vx": 0.35, "vy": -0.135, "wz": 0.09, "reason": "goal", "gap_m": 0.752},
 "perception": {"seq": 83, "frame_age_s": 0.564, "video_frame": null},
 "posture": "standing", "live": false}
```

Every tick is written — holds, stale-perception skips and the goal search included, since
"it stood still for 1.4 s" is a signal rather than a gap. `perception.video_frame` is the
index of the matching frame in `--record`, which is the join back to the footage.
`evidence/sample_telemetry.jsonl` is a real 12 s run: **91 ticks, 91 with pose.**

## The adapter: object list → LiDAR-like range vector

`integration/` turns a tick into the observation the policy expects. Sagar's plan —
*"convert the detected object's position into the LiDAR-like vector that goes into the
MAPPO agent"* — is implemented as exact ray-versus-disc geometry.

```python
from telemetry_reader import read_run
from observation import observation_from_tick

run = read_run("run.jsonl")
for tick in run.ticks:
    observation = observation_from_tick(tick)      # None while searching for the goal
    if observation:
        action = policy(observation.as_vector())   # [goal_r, goal_bearing, *16 ranges, vx, vy, wz]
```

### ⚠️ A 16-ray 360° fan is blind to the staged bin

This is measured, and it is the first thing to check against a checkpoint. Rays *sample*;
they do not integrate. An object is only guaranteed to be hit while it subtends at least
half the ray spacing:

| fan | ray spacing | bin (r = 0.23 m) reliably seen to |
| --- | --- | --- |
| 16 rays over 360° | 22.5° | **1.18 m** |
| 16 rays over 85.27° *(default here)* | 5.3° | **4.95 m** |
| 64 rays over 360° | 5.6° | 4.69 m |

The bin at 2 m subtends 13.2°, so on a 16-ray 360° fan it falls **entirely between two
rays** and the policy is handed open floor where the only obstacle in the scene is.
`observation.reliable_range_m()` computes this for any radius and fan, and the default
FOV is the camera's real 85.27° for exactly this reason. If the checkpoint needs 360°,
raise the ray count.

## What this cannot give the policy

Stated here because a range vector *looks* like a LiDAR scan and is not one:

- **Free space means "nothing recognised", not "nothing there".** The stack sees tracked
  people and one named coloured prop. Walls, table legs and doorframes are invisible.
- **No rear view.** ~85° of camera, and the robot never reverses. Everything else reads
  clear — the optimistic direction. A policy that learned to back out of a dead end will
  believe the space behind it is empty.
- **Peers are invisible.** Another quadruped is not a detector class and not a colour
  profile. For a *multi*-agent demo this is the gap that matters most, and closing it is
  the obvious next piece of work — an ArUco marker or a colour panel on each robot would
  make peers detectable through machinery that already exists.
- **Perception is a few hundred ms behind reality** (median 309 ms, p90 436 ms). The
  stack extrapolates tracks and inflates their radii to cover it; the policy sees the
  result, not the raw sensor.

## Status

| | |
| --- | --- |
| ✅ Walks to a goal, gives way to people | hardware-verified ([PR #10](https://github.com/arm/arm-mhs-unitree-go2/pull/10)) |
| ✅ Runs from a clean clone | [PR #11](https://github.com/arm/arm-mhs-unitree-go2/pull/11) |
| ✅ Maps a static obstacle, goes around it, detected goal | dry-run only ([PR #14](https://github.com/arm/arm-mhs-unitree-go2/pull/14)) |
| ✅ Telemetry contract + observation adapter | 226 offline tests |
| ⛔ **Live run with the bin** | blocked: the D1 arm's servo bus does not energise ([#12](https://github.com/arm/arm-mhs-unitree-go2/issues/12)), so the arm cannot be latched and 3.15 kg back-drives off the dorsal centreline |
| ⛔ Multiple quadrupeds | one robot; peers not detectable (above) |

## Layout

```
robot-stack/     the Go2 control stack, vendored — see PROVENANCE.md
integration/     telemetry reader + observation adapter  (python3 test_observation.py)
evidence/        the approved run, the static-obstacle dry run, a sample telemetry file
```

## Running the tests

```bash
cd integration && python3 test_observation.py                 # 23, pure stdlib
cd robot-stack/unitree/go2/visual_nav && for t in test_*.py; do python3 $t; done   # 203
```

The robot-stack suite needs `numpy` and `opencv-python`; `integration/` needs neither.

## Safety

`robot-stack/SAFETY.md` governs anything that moves a leg, and it is not optional. In
short: `--live` is the only flag that moves the robot, an operator stays on the remote,
the lane is kept clear of everything except the props, and the robot is tethered by
Ethernet — check the slack before anything that turns.
