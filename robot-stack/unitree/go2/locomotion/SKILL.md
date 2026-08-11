---
name: unitree-go2-locomotion
description: >-
  Walk the Unitree Go2 to a goal under closed-loop control — command a body-frame velocity through
  Unitree's high-level SportClient (the manufacturer gait/balance controller, no RL on the legs) and
  steer on the robot's MEASURED odometry from rt/sportmodestate. Provides walk_to(xy), turn_to(yaw),
  walk_forward(distance), stand/recover/stand_down, and the Go2-specific motion-mode + lease handling
  (ensure_sport_mode / release_control) that the G1 doesn't need. Use when a quadruped needs to
  APPROACH a target or traverse a space. Robot-agnostic core in lib/locomotion.py; the Go2 is one
  binding, and lib/ros2_twist_locomotion.py is the vendor-neutral binding for any other ROS 2 quadruped.
metadata:
  tags: [unitree-go2, quadruped, locomotion, walking, navigation, odometry, sport-mode, mobility]
---

# Unitree Go2 — Locomotion (walk + navigate)

Command a velocity through Unitree's balance controller and close the loop on measured odometry. The
API, the mode/lease rules, the optional planner, and the safety rules are in
**[`README.md`](README.md)** — this skill is the agent entry point. Modules:
[`go2_locomotion.py`](go2_locomotion.py), [`lib/locomotion.py`](https://github.com/arm/arm-mhs-robotkit/blob/main/lib/locomotion.py),
[`lib/navigation.py`](https://github.com/arm/arm-mhs-robotkit/blob/main/lib/navigation.py).

## When to use

- The robot must **approach** something before acting on it (walk to a spot, a person, a table).
- You need a quadruped to **walk to a goal pose** holding a heading, or **turn in place**.
- You need to **route around obstacles** from the onboard LiDAR (feed `rt/utlidar/cloud_deskewed` — via
  [`../lidar_sight`](../lidar_sight/SKILL.md) — to the `Navigator`).
- You're filling a robot **descriptor**'s locomotion notes (velocity API + odometry source + frame).

## How to use

```python
import sys; sys.path.insert(0, "unitree/go2/locomotion")
from go2_locomotion import Go2Locomotion

loco = Go2Locomotion(iface="eth0"); loco.connect()
loco.ensure_sport_mode("normal")   # Go2-specific: make sure Move is accepted
loco.stand()                       # balanced stand, ready to walk
loco.walk_forward(2.0)             # closed-loop on rt/sportmodestate
loco.stop()
```

`connect()` prints the live motion mode. If it isn't `normal`/`ai`, `Move` will be ignored until you
`ensure_sport_mode()` — this is the #1 "why won't my Go2 move" cause.

Off-robot, swap `Go2Locomotion` for `SimLocomotion` (in `lib/locomotion.py`) — the behaviour code and
the closed-loop helpers are identical, so the whole flow is testable in loopback. For a **different
manufacturer's** quadruped (e.g. DEEP Robotics), use `Ros2TwistLocomotion` from
`lib/ros2_twist_locomotion.py` instead of `Go2Locomotion`; everything downstream is unchanged.

## Safety

`set_velocity` / `stand` / `recover` / `walk_*` **move the legs**. `SportClient.Move` commands a
velocity that **persists until the next command** — there is no dead-man — so always drive motion
through the closed-loop helpers (they `stop()` on every exit) or wrap your own loop in
`try/finally: loco.stop()`, and keep the [controller abort](../controller/SKILL.md) and physical
e-stop in reach. Confirm a clear area and adequate battery before commanding motion. The `--forward`
CLI requires a typed confirmation; the default run only streams the measured pose.
