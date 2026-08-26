<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lite3 commissioning: measure this robot, or refuse to run

This directory turns a Lite3 Venture commissioning session into a sequence of single
commands. Every tool here produces **a number, an artefact, and a paragraph to paste into
[issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)** — or it
refuses and says what to go and measure. Nothing here carries a plausible default.

Operators in Shanghai: **[`RUNBOOK.md`](RUNBOOK.md) is written for you**, bilingual
English/中文, with what to set up, what a good result looks like, what a bad one looks
like, and what to do when a gate refuses. Read that first; this file is the map.

| tool | the number it produces | moves the robot |
| --- | --- | --- |
| [`lite3_state_probe.py`](lite3_state_probe.py) | frame rates, battery, mode transitions, the angular-velocity unit | no |
| [`motor_temperature_probe.py`](motor_temperature_probe.py) | 12 Celsius motor temperatures, **or** evidence that the channel does not exist | no |
| [`loaded_radius_probe.py`](loaded_radius_probe.py) | `--robot-radius` and `--policy-scale`, from four tape measurements | no |
| [`camera_calibration.py`](camera_calibration.py) | `focal_px`, HFOV, and the measured lens height | only with `--spin` |
| [`gait_floor_probe.py`](gait_floor_probe.py) | `--gait-floor`, plus two separate lateral numbers | **yes** — velocity transport only |
| [`actuator_gain_probe.py`](actuator_gain_probe.py) | `--actuator-gain` as a fitted ratio with its residual | **yes** — velocity transport only |
| [`axis_primitive_probe.py`](axis_primitive_probe.py) | `measured_m_s` — the speed each evidenced axis primitive delivers | **yes** — axis transport only |
| [`commission.py`](commission.py) | all of the above, in a safe order, in one artefact | only with `--live` |

Start with `lite3_state_probe.py`. It decodes the state stream the Lite3 motion host
already transmits, it cannot move the robot, and the numbers it recovers from an operator
driving on the vendor remote are the inputs the two walking probes need.

## ⚠️ Which transport you are on decides which measurements exist

These probes now take `--locomotion-transport`, matching the navigator's own flag, and the
choice is not a preference. The two interfaces answer different questions because one of
them throws the question away.

| | `udp` — legacy complex velocity | `axis` — profile-gated simple axes |
| --- | --- | --- |
| commanded magnitude | reaches the wire | **discarded**; sign only |
| has a Venture walked on it? | **no** | yes — 0.401 m on 2026-08-24, four `--live` runs on 2026-08-26 |
| gait floor / actuator gain | defined | **undefined** — there is one command per direction |
| per-primitive speed | undefined | defined, and it is what the envelope gate reads |

So the two ladder probes **refuse `--locomotion-transport axis` by name**, and
`axis_primitive_probe.py` refuses `udp`. That refusal is the load-bearing part. A
descending ladder pointed at the axis transport does not crash: every rung above the
profile's linear deadband emits the *same* full-scale primitive, so every rung walks at
the same speed, the anchors pass, the drift controls pass, and the probe reports the
lowest rung it happened to try as this robot's gait floor. A plausible, safe-looking
number that describes the ladder, going straight into `--gait-floor` on a live run.

And on the transport where a ladder *would* mean something, no Venture has been seen to
walk. On 2026-08-24 a `vx=0.10` m/s pulse was captured arriving correctly at the motion
host and the robot did not move — zero world-pose delta, `error_state` 0. The vendor guide
gives the reason: the complex velocity commands require autonomous mode, which a robot in
AI state cannot enter. So a walking probe on `udp` reports 0.000 m/s on every segment,
which reads exactly like a floor above the whole ladder, and the refusal it eventually
raises blames the stand sequence. `require_walked_transport` refuses that at the **dry
run**, before a socket exists, and `--accept-unwalked-transport` is how you say that
finding out is the point of the run.

## Nothing marked `provisional` may be used for live movement

`commission.py` writes one artefact per robot with `"provenance": "provisional"`. A human
reads the numbers and signs for them:

```bash
python3 commission.py --record lite3-commissioning-LITE3-A.json --review 'Your Name'
```

Only then will `--emit-flags` produce the `--gait-floor` / `--actuator-gain` /
`--robot-radius` / `--policy-scale` / `--calibration` line a live run needs. That is the
same shape as `Lite3Bindings.validate_camera_calibration`, which stops a run rather than
warning when a calibration file is not what it claims to be — a number that has been
*measured* and a number that has been *believed* are different things, and only a person
turns one into the other.

## There are no Go2 numbers in here, and there is no way to acquire one

The Go2's measured gait floors are 0.35 m/s forward and 0.20 m/s lateral. They are
properties of a different robot with different legs, a different mass and a different
vendor gait controller. `test_gait_floor_probe.py` walks the AST of every module in this
directory that could hold a velocity default and fails the suite if either value is ever
executable here. Naming them in prose to explain why they are not defaults is fine;
evaluating them is how the next robot silently inherits them.

## What the first bring-up attempt was aiming at

The 2026-08-19 session inspected `192.168.1.120` and reported: no `ros2`, no `ros-*`
packages, no `Lite3_ROS` checkout, no outbound DNS, 20 GB free. It concluded that the
robot needed a firmware-matched ROS 2 Foxy bundle from the vendor before anything could
proceed.

**`192.168.1.120` is the motion host, and none of that is supposed to be there.** Deep
Robotics documents this directly in the `Lite3_MotionSDK` README, §4.1: over Ethernet the
motion host is `192.168.1.120`, and §4.2 gives the default motion-host account as the one
the session logged in with. The vendor's ROS package is named `jetson2motion` because it
runs on a *perception* host and sends UDP **to** `192.168.1.120` — its `target_ip`
parameter defaults to exactly that address. Installing ROS 2 on the motion host would
point the bridge at itself.

So every negative finding was the vendor's normal configuration, and the directory that
actually matters was never opened: the motion host runs `~/jy_exe`, whose
`conf/network.toml` is the whole read-only interface.

## The one setting that decides whether anything is visible

`~/jy_exe/conf/network.toml` on the motion host reads:

```toml
ip = '192.168.1.102'   # the motion host sends its state to THIS address, and only this one
target_port = 43897
local_port = 43893
```

The robot streams to a single configured destination. If no host holds that address,
the telemetry is going nowhere and every tool downstream looks broken for the wrong
reason.

Two ways to become that destination. Prefer the first — it changes nothing on the robot:

1. **Give the laptop the address the robot already sends to.** Set the Ethernet interface
   to a static `192.168.1.102`, netmask `255.255.255.0`, and leave the Router/gateway
   field **empty**. A gateway here installs a default route through the robot and
   black-holes the laptop's normal internet.
2. **Point the robot at the laptop.** SSH to the motion host (credentials are in the
   vendor README §4.2 — do not put them in this repository or in an issue), edit `ip` to
   the laptop's address, then `cd ~/jy_exe/scripts && ./stop.sh && ./run.sh`. This
   restarts the vendor locomotion service, so do it with the robot prone and the remote
   in hand.

Confirm reachability with `ping 192.168.1.120` before going further.

## Run the probe

Needs Python 3.8+ and nothing else — no ROS, no `numpy`, no compiler, no internet. Run it
on the laptop, not on the robot.

```bash
cd robot-stack/deep_robotics/lite3/commissioning
python3 lite3_state_probe.py --seconds 30 --robot-id LITE3-A --record lite3-a-capture.jsonl
```

If it reports zero frames, the destination address is wrong; the report says what to check.

## Mirror existing telemetry without changing the robot

[`lite3_state_relay.py`](lite3_state_relay.py) is a diagnostic-only sender for a computer that
already receives the motion host's UDP state stream. It sends only validated `RobotState`
datagrams to a non-command UDP port; it counts other valid frame types but does not send them.
It rejects port 43893, has no motion/control import, and does not edit `network.toml` or restart
`jy_exe`.

Its `sent` count proves only that the local UDP stack accepted the datagram; UDP has no delivery
acknowledgement. On this Lite3, Mac-to-motion-host UDP frames reached host Ethernet capture but
did not reach any host user-space UDP socket, so this relay is **not** a usable host state path.
Do not use it as the MAPPO state source.

The planned standalone-demo solution is a separately authorized host-local `RobotState`
configuration: backup the current `network.toml`, point the vendor telemetry destination to a
host-local address/port, restart `jy_exe` once, prove valid state at the expected rate, and prove
rollback. That maintenance operation is not performed by this relay and does not authorize robot
motion.

### Capture 1 — the robot is prone and untouched

Establishes the link, the frame rates, `battery_level`, and the resting values of the four
mode fields. Nothing moves.

### Capture 2 — the operator drives on the vendor remote

Keep the probe running. Have the operator stand the robot, put it into high-level
navigation mode, walk it slowly forward, then turn it in place. **This software transmits
nothing during that capture; the vendor's own controller is the only thing commanding the
legs.** It settles four of the open items in issue #13:

| issue #13 item | how this capture settles it |
| --- | --- |
| approved AUTO/manual transition | the four mode fields change at the moment the operator acts; the report timestamps each transition, so you learn this firmware's state machine by watching it |
| `--gait-floor` | `HandleState.goal_vel_forward` is the velocity the firmware derived from the remote's stick, paired against measured `vel_body`; the lowest commanded bin that produces a real walk is the floor |
| `--actuator-gain` | measured ÷ commanded from the same pairs, at the envelope you intend to demo |
| angular-velocity units | Prefer `robot_state.rpy` / `rpy_vel`; when that frame is absent, use IMU `angle_deg` / `angular_velocity`. The report divides observed yaw change by the reported rate: ≈1 means the field is degrees/s, ≈57.3 means radians/s |

Record the robot ID against every number and repeat on the second Venture. None of these
transfer between units, and none of the Go2 values apply.

## What this does not settle

- **Motor temperatures.** Absent from this stream entirely. `battery_level` is present, so
  the battery half of the health gate needs no vendor bridge; the twelve temperatures
  still do.
- **The heartbeat, and the clean stop / return-to-manual path.** A passive capture shows
  what the firmware *does*, never what it *requires*. These stay vendor questions.
- **The camera.** Nothing in this stream is an image. Confirm what forward RGB endpoint
  each Venture actually exposes, and on which computer.

## Why not just launch the vendor bridge to look around

`jetson2motion` is a single executable that constructs a receiver **and** a sender aimed
at the motion host's command port, unconditionally, before it has been told to do
anything. It is not a read-only instrument. This module has no send path at all, and
`test_lite3_state_probe.py` asserts that by parsing its own source, so adding one fails
the suite rather than passing review.

## The two Go2 lessons this directory is built around

**A lateral floor is not a floor on a diagonal.** The Go2's 0.20 m/s lateral floor was
measured as a pure strafe from standstill, and every design decision after it treated 0.20
as a hard floor — which produced a rule that a command had to be nearly 30 degrees off the
nose before any sideways travel happened. Then a robot already walking forward delivered
lateral travel proportionally from 0.05 m/s upward and the whole argument dissolved.
`gait_floor_probe.py` therefore measures **both** cases, in separate phases, and reports
them as two numbers that must never be substituted for one another.

**A probe that fails to stand reports 0.000 m/s on every axis, which reads exactly like a
floor that is real and total.** The Go2's own lateral probe once called `stand()` where
getting up needs two calls; the robot stayed prone, every command was ignored, and the run
produced a table of zeros. Here the forward ladder carries **anchor** segments commanded at
a speed the operator has already watched this robot walk at, and if an anchor does not
travel the run is refused outright. The diagonal phase holds forward velocity on every
segment and so takes the Go2's own refusal unweakened.

## Offline checks

```bash
for test in test_*.py; do python3 "$test"; done   # 233
ruff check .
```

The guards are the point, so they are mutation-tested: breaking any one of them — the
anchor refusal, the drifting-control refusal, the provisional gate, the twelve-channel
refusal, the inference-config assertion, the operator-ready gate — turns a named test red.

Read [`../../../SAFETY.md`](../../../SAFETY.md) before the operator-driven capture, and
before any run of the three probes that walk. Only `--live` together with `--operator-ready`
authorises a leg; every other step here is receive-only.
