<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The Device Connect dashboard

Watch a quadruped, drive it, and manage the MAPPO checkpoint it is carrying — from a browser,
over [Arm Device Connect](https://github.com/arm/device-connect).

⛔ **[`../robot-stack/SAFETY.md`](../robot-stack/SAFETY.md) governs anything here that moves
a leg, and it is not optional.** Motion is off by default; `--allow-motion` is this
directory's `--live`.

```
   browser
      │  HTTP + Server-Sent Events
   server.py                              ← a workstation. No robot code, no SDK.
      │  device-connect-agent-tools
 ═════╪══════ Device Connect mesh ═══════   D2D by default: no broker, no etcd, no Docker
      │  device-connect-edge
 robot_driver.py                          ← ON the robot, Python ≥ 3.11
      │  subprocess, one command, JSON, exit
 drive_bridge.py                          ← ON the robot, the SDK env, Python 3.8
      │
 Go2Locomotion / Lite3Locomotion
```

## What it does

| | |
| --- | --- |
| **View robot events** | Every command, every refusal, every checkpoint change, plus pose and mode on a 5 s timer. Live, filterable, and replayed from a ring buffer to a page that opens late. |
| **Basic motion** | Walk forward / back, strafe left / right, turn left / right, lie down. Bounded in time, and every press reports what the robot *measured*. |
| **Swap checkpoints** | Arm any `.npz` already on the robot for the next run. |
| **Load / unload from Cloud AI** | Pull a checkpoint from an S3 bucket or a direct server address; delete one off the robot. |

## Run it

Install Device Connect into a Python ≥ 3.11 environment on the robot:

```bash
pip install device-connect-edge boto3          # boto3 only if you want S3 sources
```

On the robot — **status and checkpoints only, which is where to start**:

```bash
python3 robot_driver.py --platform go2 --package ../policy \
        --bridge-python /home/unitree/robotics-connect-go2/bin/python
```

`--bridge-python` is the interpreter that can import `unitree_sdk2py`. It is **not** the one
running the driver, and getting it wrong makes every command fail with an import error —
`get_capabilities` reports the path it will use, so check it before you need it.

Then, with a clear area and an operator on the controller abort, add `--allow-motion`.
On a Lite3, stand the robot and enable high-level navigation mode on the vendor interface
first, then add `--operator-ready`.

On a workstation:

```bash
pip install device-connect-agent-tools aiohttp
python3 server.py --port 8080                  # then open http://127.0.0.1:8080
```

The default bind is loopback. Pass `--host 0.0.0.0` to reach it from the demo LAN, and note
what that means: **this dashboard has no login.** Anyone who can reach the port can drive
any robot on the mesh that was started with motion enabled.

No robot? Everything above works against a bench double:

```bash
python3 robot_driver.py --platform sim --package ../policy --allow-motion
```

## The four things worth knowing before you use it

**A swap takes effect on the next run, not on the one in progress.** `MappoController` loads
its weights once, at construction, so a live `mappo_drive` cannot have the network pulled out
from under it — not because anything checks, but because no code path exists. What the store
*does* check is that the checkpoint you are arming can actually run under the current config:
`lidar_range_vmas` must match the checkpoint's `training_lidar_range_vmas`, or the next run
dies at load. That refusal happens at the click instead.

**Every button is a measurement.** Commands hold a velocity for a fixed time and then stop,
and the result reports pose before, pose after, distance travelled, and the fraction of the
commanded speed that was delivered. Closed-loop distance control would have hidden exactly
the number this repository has spent the most time on — this Go2 delivers about **0.45** of
what it is commanded. A button that says "commanded 0.35 for 1.5 s, travelled 0.012 m"
diagnoses itself.

**The two platforms are not interchangeable, and the page says so per robot.** `lie_down` on
a Go2 issues `StandDown`; on a Lite3 it only *stops*, because posture there is
operator-controlled through the vendor app. Speeds below a measured gait floor are refused
(0.35 m/s forward on both; 0.20 m/s lateral on the Lite3). **The Go2's lateral floor has
never been measured** — issue #42 is about exactly that conflation — so a Go2 strafe is
allowed and carries a warning on every press that it may produce no gait at all, which would
not be a fault.

**Reverse is open-loop into unobserved space.** Neither platform has rear sensing and the
planner never samples that direction; issue #40 caught the policy commanding it. The button
exists because it was asked for, it is capped at 2 s rather than 5, and it says so.

## Files

| | |
| --- | --- |
| `robot_driver.py` | The Device Connect device. 16 RPCs, 7 events. Runs on the robot, Python ≥ 3.11. |
| `drive_bridge.py` | The SDK-env worker: one command, one JSON line, exit. Python 3.8, stdlib only. |
| `model_store.py` | Checkpoints on disk: what is here, what is armed, what may replace it. |
| `cloud_models.py` | S3 and http(s) fetch, with the refusals that make a URL field on a web page safe. |
| `server.py` | The dashboard: discovery, an invoke allow-list, and the SSE event fan-out. |
| `templates/`, `static/` | The page. It renders from `get_capabilities()`, so it never hard-codes what a robot can do. |

## Tests

```bash
for t in test_*.py; do python3 $t; done       # 75
ruff check .                                  # must be clean
```

Needs `device-connect-edge`, `device-connect-agent-tools`, `aiohttp` and `numpy`; `boto3`
only for the S3 path. `test_drive_bridge.py` and `test_model_store.py` run without the
Device Connect packages.

Five guards are mutation-tested rather than assumed — see
[`../evidence/2026-08-21-device-connect-dashboard/`](../evidence/2026-08-21-device-connect-dashboard/),
which also records the two defects the bring-up run found and what the run does **not**
prove. No robot has moved under this yet.

## Two traps, written down because both cost real time

**Do not name an `@rpc` after anything on `DeviceDriver`.** `capabilities`, `status`,
`identity`, `invoke`, `events`, `functions`, `connect`, `registry`, `router`, `transport`
and about twenty more are live mines. `capabilities` is a property the runtime reads to build
the presence announcement; override it and the device **never appears on the mesh**, with no
exception and no log line, looking exactly like a network fault.
`test_no_rpc_shadows_a_base_class_member` now catches this.

**Events arrive grouped by type, not by time.** A `Subscription` keeps one inbox per subject
and each event name is a subject, so a drained batch must be re-sorted before anyone reads
it — otherwise a completion appears above the start that caused it. `server._in_time_order`
does that, by the device's own `ts`, which has one-second resolution and therefore cannot
order two events inside the same second.

## Why not the Device Connect portal

`device-connect-server` ships a full multi-tenant portal with a devices page, a generic
invoke form and an event stream, and for fleet administration it is the right tool. It is
generic by design — a JSON form per function — which is correct for forty device types and
wrong for a motion pad, where the useful property is that "strafe left" is one key press
away from "stop". This page is demo-specific UI and belongs in the demo repository.

They are not exclusive: `connect()` speaks D2D or a router transparently, so a robot
registered with a full server appears here as readily as one announcing itself by multicast.
