<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multi-Agent Proximal Policy Optimization (MAPPO) using Arm Cloud AI and Arm Physical AI

A MAPPO policy, trained in simulation, driving real quadrupeds to goals in a shared
room. This repository is the meeting point: the **robot-side control stack** that
perceives and moves, the **telemetry contract** between it and the policy, and the
**adapter** that turns one into the other.

Built on **[Arm Device Connect](https://deviceconnect.dev/)** — the open standard for
describing and driving physical hardware from software
([github.com/arm/device-connect](https://github.com/arm/device-connect)).

| who | what |
| --- | --- |
| **Waheed Brown** ([@armwaheed](https://github.com/armwaheed)) | robot stack — perception, planning, safety, telemetry |
| **Sagar Surendran** ([@spsagar13](https://github.com/spsagar13)) | MAPPO policy, training environment, checkpoints |
| **Deep Robotics** | porting the stack to the Lite3 Venture — see *Porting* below |

## What actually works today

One Go2, **RGB alone** — no LiDAR, no depth, no motion capture. Both of these are live
runs on the real robot.

### Going around a static obstacle, to a detected goal

![Go2 mapping a recycling bin and swerving around it toward a chair](go2-static-obstacle-run.gif)

Real time. The robot has no idea what a recycling bin is — no detector is trained on
one — so it finds it by colour, checks its shape, and **maps it in odom** so it persists
once the swerve takes it out of frame. The goal is the **chair**, found by running the
detector on a centre crop. Boxes carry the monocular range and the prior that produced
it; the inset is the planner's own belief, with the mapped bin and the arc it chose.

`cmd avoid v=(+0.30,−0.20,+0.12)` is the robot deciding to pass on the right.

It walks 1.89 m, draws level with the bin, and stops there — the office lane runs out.
At that moment the planner was satisfied (0.70 m of separation where it needed 0.60);
there was simply no floor left. A corridor problem, not a planning one.
`evidence/live_run.{mp4,log,jsonl}`.

### Giving way to a person

![Go2 walking to a goal and giving way to a person](robot-stack/unitree/go2/visual_nav/images/go2-visual-nav-run.gif)

2.0 m to a dead-reckoned waypoint, `arrived (0.96 m from goal)`, giving way to a person
who repeatedly crossed its path. 145 perception cycles, 0 errors, motors 31 → 32 °C. Of
107 control ticks: 63 `goal`, 24 `avoid`, 20 `hold` — and **every one of the 12 ticks
where the gap went negative commanded a full stop.** This is the run Sagar reviewed;
`evidence/go2_nav_run.{mp4,log}`.

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
`evidence/sample_telemetry.jsonl` is the live run above. Each tick also carries the
**measured** velocity beside the commanded one — without that, "commanded 0.12 m/s and
moved nothing" is indistinguishable from walking, which is a failure that cost three
runs to see.

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
| ✅ Walks to a goal, gives way to people | hardware-verified (Go2 stack PR #10) |
| ✅ Runs from a clean clone | Go2 stack PR #11 |
| ✅ Maps a static obstacle, goes around it, detected goal | live; walked 1.89 m, stopped for lane width |
| ✅ Telemetry contract + observation adapter | 246 offline tests |
| ⚠️ Arriving at the chair past the bin | needs ~0.3 m more lane than this corridor has |
| ⚠️ D1 arm latch | its servo bus does not energise (Go2 stack issue #12); runs use `--no-latch-arm`, and the arm creeps a few degrees off the dorsal line each run |
| ⛔ Multiple quadrupeds | one robot; peers not detectable (above) |
| ⏳ Lite3 Venture port | three vendor seams + a recalibration — see *Porting* |

## Porting to the DeepRobotics Lite3 Venture

The Lite3 Venture has **an RGB camera and nothing else** — no LiDAR, no depth camera, no
back-mounted arm. That is the configuration this stack was built for, so most of it moves
across unchanged, and the parts that do not are small and known.

**Moves as-is.** Everything that turns pixels into a plan is robot-agnostic numpy and
OpenCV: `camera_model` (fisheye pixel ↔ bearing, angular-size ranging), `person_detector`,
`colour_detector`, `tracker`, `static_map`, `avoidance`, `goal`, `overlay`, `telemetry`,
`replay`. None of them imports a robot. They are the majority of the module and the whole
of the integration surface.

**Needs a vendor implementation.** Three seams, all narrow:

| seam | what the Go2 does | what the Lite3 needs |
| --- | --- | --- |
| `camera.py` | grabs JPEGs off the Go2's `VideoClient` RPC | any source of BGR frames with a capture timestamp and a pose stamp |
| `locomotion` | `LocomotionController` — `set_velocity(vx, vy, wz)`, `pose()`, `stand`/`lie` | the same four calls against the Lite3 SDK |
| `safety.py` | motor temperature, battery, and D1 arm stow checks | temperature and battery; **the arm half does not apply** |

**Recalibrate before trusting a single range.** `go2_front_camera.json` is *this unit's*
lens — focal 1290.2 px, HFOV 85.27°. Every distance in the system is proportional to it.
`calibrate_camera.py --spin` measures it with no tape measure and no fiducial: the robot
turns on the spot and uses its own yaw odometry as the angular ruler. Run it on the Lite3
and everything downstream is correct; skip it and every range is wrong by the ratio of the
two lenses.

**No arm is a simplification, not a gap.** The D1 costs this stack a great deal — it is
why the robot rests prone between moves, why the envelope is derated to 0.35 m/s, and why
a run can be refused outright (the arm creeps a few degrees off the dorsal line each run
and the gate is absolute). A Lite3 Venture runs with `--no-require-arm`, skips all of it,
and can use a faster envelope.

## Layout

```
robot-stack/     the Go2 control stack, vendored — see PROVENANCE.md
integration/     telemetry reader + observation adapter  (python3 test_observation.py)
evidence/        the approved run, the static-obstacle dry run, a sample telemetry file
```

## Running the tests

```bash
cd integration && python3 test_observation.py                 # 23, pure stdlib
cd robot-stack/unitree/go2/visual_nav && for t in test_*.py; do python3 $t; done   # 223
```

The robot-stack suite needs `numpy` and `opencv-python`; `integration/` needs neither.

## Safety

`robot-stack/SAFETY.md` governs anything that moves a leg, and it is not optional. In
short: `--live` is the only flag that moves the robot, an operator stays on the remote,
the lane is kept clear of everything except the props, and the robot is tethered by
Ethernet — check the slack before anything that turns.
