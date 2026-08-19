<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lite3 commissioning: read the robot before installing anything on it

This directory holds one tool, `lite3_state_probe.py`. It decodes the state stream the
Lite3 motion host already transmits, and it cannot move the robot. Run it before anything
else in [issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13).

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
| angular-velocity units | `rpy` is degrees; `rpy_vel` is copied into a ROS rad/s field without conversion. The report divides observed yaw change by the reported rate: ≈1 means the field is degrees/s, ≈57.3 means radians/s |

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

## Offline checks

```bash
python3 test_lite3_state_probe.py   # 16
ruff check .
```

Read [`../../../SAFETY.md`](../../../SAFETY.md) before the operator-driven capture. No
step in this directory authorises this repository to command a leg.
