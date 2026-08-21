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
| **Deep Robotics** | Lite3 Venture platform support — see *Porting* below |

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
 "obstacles": [{"label": "bin", "kind": "static", "id": "landmark-1",
                "x": -2.276, "y": -0.081, "vx": 0.0, "vy": 0.0, "radius_m": 0.23}],
 "command": {"vx": 0.35, "vy": -0.135, "wz": 0.09, "reason": "goal", "gap_m": 0.752},
 "measured": {"vx": 0.331, "vy": -0.128, "wz": 0.084},
 "perception": {"seq": 83, "frame_age_s": 0.564, "video_frame": null, "stale": false},
 "posture": "standing", "live": false}
```

The header line declares **which frame every vector is in** — `pose`, `goal` and
`obstacles` are odom; `command` and `measured` are body. That is the one thing a consumer
cannot recover from the data, because the two frames agree *exactly* while the robot
faces its start heading and diverge only as it turns. An integration built on the wrong
assumption passes every bench test and fails in the first corner.

`kind` is `"static"` or `"tracked"`, and `label` is **not** a substitute for it: `label`
is a class name, and it separated the two only while the scene had exactly one mapped
prop and one detector class. A person who has *stopped* has a bin's velocity and a
person's claim on the lane. `id` is the stable identity, so a consumer can follow one
object across ticks instead of re-associating by position and merging near neighbours.

Every tick is written — holds, stale-perception skips and the goal search included, since
"it stood still for 1.4 s" is a signal rather than a gap. `perception.video_frame` is the
index of the matching frame in `--record`, which is the join back to the footage.
`evidence/sample_telemetry.jsonl` is the live run above. Each tick also carries the
**measured** velocity beside the commanded one — without that, "commanded 0.12 m/s and
moved nothing" is indistinguishable from walking, which is a failure that cost three
runs to see.

## Driving the MAPPO policy from a tick

**The policy package and its checkpoint are in the tree**, at [`policy/`](policy/) — 262
KiB of weights, so the demo runs from a clean clone and every number quoted in an issue is
one anyone can reproduce. [`policy/PROVENANCE.md`](policy/PROVENANCE.md) lists the five
corrections applied to it on the way in; all five were silent failures, and the delivered
smoke test passed with every one of them in place.

The package does its own ray casting, so the integration is a *mapping*, not an adapter:
`integration/mappo_bridge.py` turns one telemetry tick into one `RobotInput`. Three of the
mappings are not the obvious ones, and each is pinned by a test that says why.

```python
from mappo_bridge import robot_input
from physical_ai_mappo import MappoController, RobotInput, StationaryObject

for tick in read_run("run.jsonl").ticks:
    mapped = robot_input(tick, reset_run=first)     # None while searching for the goal
    if mapped is None:
        continue
    mapped["stationary_objects"] = [StationaryObject(**o)
                                    for o in mapped["stationary_objects"]]
    out = controller.step(RobotInput(**mapped))
```

| mapping | the obvious answer | why it is wrong |
| --- | --- | --- |
| `velocity_frame` | `"odom"` | `measured` is the estimator's **body**-frame velocity |
| `external_hold` | `reason == "hold"` | the planner also holds for the *bin* — forwarding that zeroes the policy in the one scene it exists for |
| `timestamp_s` | `wall_time` | it is compared against `time.monotonic()`; an epoch makes the age ≈ −1.8e9 s, so the staleness gate can never fire |

### Replay is the test that the mapping is right

A field-by-field table cannot catch a frame, a unit, or a field that is present and means
something else. Replaying a recorded run through the real checkpoint can:

```bash
cd integration && python3 replay_mappo.py ../evidence/sample_telemetry.jsonl
```

Every run is paired with its own control — the same ticks, through a second controller,
with the obstacles removed. Without that, "the policy steered 36° off the goal bearing"
is not evidence of anything: this checkpoint carries a 6–16° heading bias with no
obstacle anywhere near it.

### Closing the loop, before the policy drives anything

`replay_mappo.py` is open-loop: the shipped planner drove the path, so the policy never
met the states its own actions produce. `integration/closed_loop_sim.py` closes it —
action → actuator → pose → what the camera can now see → the next observation, through
the same bridge — and runs the policy against **the shipped planner on identical
scenarios**, because "the policy arrived 18 times in 30" is not a result without knowing
what the incumbent does on the same runs.

```bash
cd integration && python3 closed_loop_sim.py --seeds 30 --scale 1.5 2.5 \
    --command-scale 0.3 0.6 1.0
```

Its verdict, and the configuration [`deploy/README.md`](deploy/README.md) recommends, is
that the policy is safe to drive **only under the planner's veto**: raw, it collided in
every configuration tested — 21 times in 30 at the scale the package shipped with.

### ⚠️ For this checkpoint the *horizon* binds, not the ray fan

The general warning still holds — rays sample, they do not integrate, so an object is
only guaranteed to be hit while it subtends half the ray spacing, and
`observation.reliable_range_m()` computes that for any radius and fan. But the delivered
checkpoint's 12-ray 360° fan is **not** what limits it:

| limit on seeing the bin (r = 0.42 m as mapped live) | range |
| --- | --- |
| 12-ray 360° fan geometry | 1.62 m |
| policy sensing horizon — 0.35 VMAS × 2.5 m/unit | **0.875 m** |

The horizon binds first. `meters_per_vmas_unit` is what sets it and it is a **calibration
parameter**, confirmed as such by @spsagar13: the delivered 1.5 matched the *room* to the
trained spawn region, and 2.5 matches the *robot* to the trained agent (the live runs'
0.25 m planner radius ÷ the trained 0.10 VMAS agent radius). Sweeping it with
`replay_mappo.py --scale` shows what it buys and what it does not:

| m/unit | horizon | obstacle seen on | mean steering response inside |
| --- | --- | --- | --- |
| 1.5 | 0.525 m | 59/121 ticks | 96.6° |
| **2.5** | **0.875 m** | **77/121** | **103.4°** |
| 4.0 | 1.400 m | 97/121 | 96.6° |

**The response is a cliff, not a ramp, and no scale fixes that** — it is saturated at
around 100° everywhere, against 0.1° outside the horizon. Raising the scale buys *warning*
and never proportionality. Softening the cliff needs a retrain with a larger
`lidar_range`; issue #4.

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
| ✅ Offline regression suite | 619 tests: policy 33, integration 144, Go2 visual navigation 265, Lite3 72, dashboard 105 |
| ✅ MAPPO policy driven from a recorded run | replayed all 122 ticks; mapping clean apart from object ids, which the log now carries |
| ✅ Policy package + checkpoint in the tree | `policy/`, 262 KiB; six silent defects corrected, each pinned by a test |
| ✅ Closed-loop simulation | 30 seeded scenarios × 3 controllers × 2 scales × 3 command scales, each paired with an ablated control |
| ⚠️ Policy sensing horizon | 0.875 m to the obstacle surface at the recalibrated scale — it sees the bin on 77 of 121 ticks, and the response is a cliff at that range rather than a ramp, at **every** scale |
| ⛔ Policy driving **between** two obstacles | the horizon is shorter than the aperture is wide: both bins were in range on **0** of 137 ticks across three failing runs, and 33 of 79 on the one that worked. Needs a retrain — issue #29, evidence dated 2026-08-19 |
| ⚠️ Policy driving the legs, **supervised** | at the walkable 1.0 command scale: 21/30 arrivals and **1 collision** in sim; planner veto required for obstacles |
| ⛔ Policy driving the legs, **unsupervised** | collided in every simulated configuration — 21/30 at the scale the package shipped with. Not a candidate. |
| ✅ Policy on Go2 hardware, empty lane | arrived 0.77 m from the chair after 2.78 m; policy drove 53/53 ticks, 0 vetoed, 0 stopped; obstacle run remains open |
| ⚠️ Arriving at the chair past the bin | needs ~0.3 m more lane than this corridor has |
| ⚠️ D1 arm latch | its servo bus does not energise (tracked in the upstream Go2 stack); runs use `--no-latch-arm`, and the arm creeps a few degrees off the dorsal line each run |
| ⛔ Multiple quadrupeds | one robot; peers not detectable (above) |
| ✅ Lite3 Venture offline port | high-level ROS locomotion, RGB camera, fail-closed health gate, calibration and MAPPO entry points; 30 platform tests |
| ⏳ Lite3 hardware commissioning | [#13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13): neither event robot has been run; gait floor, actuator gain, loaded radius, camera model/source and health publisher remain measured inputs |
| ✅ Dashboard drives a fleet | every robot listed at once with its own stop, plus STOP ALL; cross-robot stop 4.23 s → 0.06 s, same-robot stop 4.17 s → 0.07 s and it now interrupts the walk |
| ✅ Device Connect dashboard, off-robot | [#43](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/43): events, motion, checkpoint swap and Cloud AI load/unload, end to end over a real D2D mesh against a bench double — see `evidence/2026-08-21-device-connect-dashboard/` |
| ⏳ Device Connect dashboard, on hardware | not yet run on a robot. The bench double delivers 1.00 of what it is commanded, which is exactly the number a real robot does not produce; nothing there tests gait, DDS, the ROS bridge or the SDK import |

## Porting to the Deep Robotics Lite3 Venture

The two event Lite3 Ventures have **an RGB camera and no LiDAR**. The offline port is in
[`robot-stack/deep_robotics/lite3/`](robot-stack/deep_robotics/lite3/README.md). It has
not moved either robot; its runbook names every measurement and vendor feed still needed
at commissioning rather than filling them with Go2 values.

**Moves as-is.** Everything that turns pixels into a plan is robot-agnostic numpy and
OpenCV: `camera_model` (fisheye pixel ↔ bearing, angular-size ranging), `person_detector`,
`colour_detector`, `tracker`, `static_map`, `avoidance`, `goal`, `overlay`, `telemetry`,
`replay`. None of them imports a robot. They are the majority of the module and the whole
of the integration surface.

**The vendor seams are implemented.** Three narrow bindings surround the common loop:

| seam | Go2 | Lite3 Venture |
| --- | --- | --- |
| RGB camera | JPEGs from `VideoClient` | explicit V4L2, RTSP, or GStreamer BGR source with local arrival time and pose stamp |
| locomotion | `SportClient` over CycloneDDS | high-level `Lite3_ROS` `/cmd_vel` + `/leg_odom2`; the low-level MotionSDK is deliberately not used as a gait controller |
| safety | motor temperature, battery, D1 arm stow | standard battery and motor-temperature ROS feeds; missing/stale data refuses a live run; **no arm flags exist** |

**Recalibrate before trusting a single range.** `go2_front_camera.json` is *this unit's*
lens — focal 1290.2 px, HFOV 85.27°. Every distance in the system is proportional to it.
The Lite3 wrapper uses the same spin fit against pose yaw and tags the resulting JSON with
the platform name. A live Lite3 run refuses a Go2, missing, or malformed calibration file.

**No arm is a simplification, not a bypass.** The D1 costs the Go2 stack a great deal — it is
why the robot rests prone between moves, why the envelope is derated to 0.35 m/s, and why
a run can be refused outright (the arm creeps a few degrees off the dorsal line each run
and the gate is absolute). The Lite3 parser has no `--no-require-arm` or
`--no-latch-arm`: the arm subsystem is absent from that platform binding.

**The unresolved item is hardware evidence.** A live run requires a measured gait floor,
actuator gain, loaded radius and Lite3 camera JSON. It also requires battery and motor
temperature topics that the public high-level vendor bridge does not publish. The
binding fails closed until a supported companion feed supplies them. Peer detection is
still the known two-robot gap.

## Watching and driving it from a browser

[`dashboard/`](dashboard/README.md) puts the robot on the Device Connect mesh as a device and
serves a page that discovers it — a live event stream, bounded motion, checkpoint swap, and
loading a checkpoint from an S3 bucket or a server on the LAN. There is no broker, no etcd and
no Docker: D2D mode finds the robot by multicast on the LAN the demo already runs on.

Try it with no robot at all:

```bash
pip install device-connect-edge device-connect-agent-tools aiohttp    # Python >= 3.11

cd dashboard
python3 robot_driver.py --platform sim --package ../policy --allow-motion   # terminal 1
python3 server.py --port 8080                                              # terminal 2
```

Then open <http://127.0.0.1:8080>. `--platform sim` is a bench double that integrates the
commanded velocity into a pose; it exercises the mesh, the schemas, the event stream and every
refusal without a robot in the room.

On the real thing, start **without** `--allow-motion` — the device is then status-and-
checkpoints only — and pass `--bridge-python` pointing at the interpreter that can import the
robot's SDK:

```bash
python3 robot_driver.py --platform go2 --package ../policy \
        --bridge-python /home/unitree/robotics-connect-go2/bin/python
```

Device Connect requires Python ≥ 3.11 and the Go2's Jetson runs its SDK on 3.8, so the driver
reaches the robot by running `drive_bridge.py` as a subprocess in that second environment.
That split is also why a driver that hangs or is killed cannot leave a velocity latched.

**`robot-stack/SAFETY.md` governs the motion buttons exactly as it governs `--live`.**
`--allow-motion` is this directory's `--live`: it needs a clear area, an operator on the
controller abort, and adequate battery. The page has **no login**, so `--host 0.0.0.0` means
anyone who can reach the port can drive any motion-enabled robot on the mesh.

Two things it deliberately does not smooth over. Each motion press is duration-bounded and
open-loop, and reports what the robot *measured* — including the fraction of the commanded
speed actually delivered, which is ~0.45 on this Go2 and is the number that explains a robot
that looks like it is not moving. And the platforms are not interchangeable: `lie_down` on a
Lite3 only *stops* it, because posture there is operator-controlled through the vendor app,
and a Go2 strafe carries a warning because that robot's lateral gait floor has never been
measured ([#42](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/42)). The page
learns both from `get_capabilities()` rather than hard-coding either.

## Layout

```
robot-stack/     Go2 control stack plus Lite3 platform bindings — see PROVENANCE.md
policy/          the MAPPO adapter and checkpoint, vendored — see policy/PROVENANCE.md
integration/     the bridge, the replay, the closed-loop sim, and the two live runners
deploy/          install.sh, uninstall.sh, and the runbook for a day at the robot
dashboard/       the robot as a Device Connect device, and a browser page that drives it
evidence/        the approved run, the static-obstacle dry run, a sample telemetry file
```

| in `integration/` | |
| --- | --- |
| `mappo_bridge.py` | one telemetry tick → one `RobotInput`. The three non-obvious mappings. |
| `mappo_policy.py` | the shared loop: bridge → policy → command, plus the heading servo |
| `replay_mappo.py` | a recorded run through the checkpoint, against an ablated control |
| `render_observation.py` | the camera frame, the ray fan and the observation vector, drawn side by side per tick — what the policy saw and why |
| `closed_loop_sim.py` | the policy's own actions moving a simulated robot — issue #5's gate |
| `mappo_shadow.py` | a **live** run, policy logged beside the planner. Cannot move a leg. |
| `mappo_drive.py` | a live run, the policy driving under the planner's veto, through a supported upstream seam |

## Running the tests

```bash
cd policy      && python3 test_physical_ai_mappo.py                                #  33
cd integration && for t in test_*.py; do python3 $t; done                          # 142
cd robot-stack/unitree/go2/visual_nav && for t in test_*.py; do python3 $t; done   # 265
cd robot-stack/deep_robotics/lite3/locomotion && for t in test_*.py; do python3 $t; done #  17
cd robot-stack/deep_robotics/lite3/visual_nav && for t in test_*.py; do python3 $t; done #  39
cd robot-stack/deep_robotics/lite3/commissioning && python3 test_lite3_state_probe.py #  16
cd dashboard   && for t in test_*.py; do python3 $t; done                          # 105
```

`policy/` and the parts of `integration/` that touch the policy need `numpy`; the
robot-stack suites also need `opencv-python`; `dashboard/` needs `device-connect-edge`,
`device-connect-agent-tools` and `aiohttp` on Python ≥ 3.11. `deploy/install.sh` runs the first two
suites as part of installing, because a truncated checkout should fail there rather than
in the arena. Every listed code directory carries a `ruff.toml`; `ruff check .` is clean
in each.

## Safety

`robot-stack/SAFETY.md` governs anything that moves a leg, and it is not optional. In
short: `--live` is the only flag that moves the robot, an operator stays on the remote,
the lane is kept clear of everything except the props, and the robot is tethered by
Ethernet — check the slack before anything that turns.

For a policy-driven run, [`deploy/README.md`](deploy/README.md) adds the ladder — simulate,
then shadow, then drive — and the measured numbers that decide whether the run does what
it looks like it is doing. Actuator gain is rate-dependent: this Go2 measured about 0.45
when derated, 0.70 at full command, and zero below its gait floor.
