---
name: unitree-go2-mhs
description: >-
  The Unitree Go2 (+ D1 arm) DRIVER for the Model Hardware Standard (MHS). Registers the quadruped as a
  DEVICE on the fabric so an orchestrator (or a human agent) can drive it: WALK to a goal (walk_forward /
  walk_to, closed-loop), STAND / SIT / STOP, MOVE the D1 arm (arm_move / arm_reset), and read STATUS
  (motion mode + measured pose) — with arrived / blocked / arm_moved events. Reuses the verified control
  in unitree/go2 through the two-env bridge; motion is gated behind allow_motion so a default device is
  status-only. Env bootstrap is bootstrap-mhs-env; the robot-agnostic control lives in arm-mhs-robotkit + unitree/go2.
metadata:
  tags: [unitree-go2, mhs, deviceconnect, driver, quadruped, fabric, teleoperation]
---

# Unitree Go2 — MHS driver

Register the physical Go2 (+ D1 arm) as an MHS device and drive it over the fabric. The RPC/event
surface, the two-env bridge, and the safety gating are in this file and
[`go2_agent.py`](go2_agent.py) / [`go2_drive.py`](go2_drive.py); the shared MHS shim is
[`arm_mhs_robotkit.mhs_sidecar`](https://github.com/arm/arm-mhs-robotkit/blob/main/lib/mhs_sidecar.py).

## When to use

- You want the Go2 to appear as a **device on the MHS fabric** an orchestrator can call (walk, stand,
  move the arm, read status) — the robot side of a fleet.
- You're wiring the Go2 into a **multi-device task** (e.g. alongside a Human Agent, like the G1 demo).

## RPC / event surface

| RPC | Effect | Motion? |
|---|---|---|
| `get_status()` | motion mode + measured pose | no |
| `stop()` | zero velocity, hold stance | no |
| `stand()` / `stand_down()` | balanced stand / lie down | yes |
| `walk_forward(distance)` / `walk_to(x, y)` | closed-loop walk, then stop | yes |
| `arm_reset()` / `arm_move(joint_angles, gripper)` | D1 arm to zero / to joint targets | yes |

Events: `arrived(x, y)`, `blocked(reason)`, `arm_moved(sent_angles)`.

## How to run (on the robot)

```bash
# status-only device (safe default) — appears on the fabric, no motion RPCs will move it
python driver/go2/go2_agent.py --creds ~/go2.creds.json

# enable motion (clear area + controller abort + e-stop required)
python driver/go2/go2_agent.py --creds ~/go2.creds.json --allow-motion

# status-only smoke test, no fabric
python driver/go2/go2_agent.py --self-test
```

## Architecture

`go2_agent.py` runs in the MHS env (Python ≥3.10) and drives the robot by invoking `go2_drive.py` in the
**SDK env** (`unitree_sdk2py` + CycloneDDS, Python 3.8 on the Go2 Jetson) — the two-env bridge (see
[`bootstrap-mhs-env`](https://github.com/arm/arm-mhs-robotkit/blob/main/skills/bootstrap-mhs-env/SKILL.md)). `go2_drive.py` reuses the verified
[`unitree/go2`](../../unitree/go2/) control (`Go2Locomotion`, `D1Arm`); the driver adds no new robot
logic.

## Safety

Motion RPCs move the robot and are gated behind `--allow-motion` (the worker also requires it), so a
default device is **status-only**. Enable motion only with a clear area, an operator on the
[controller abort](../../unitree/go2/controller/SKILL.md) + physical e-stop, and adequate battery. See
[`SAFETY.md`](../../SAFETY.md).
