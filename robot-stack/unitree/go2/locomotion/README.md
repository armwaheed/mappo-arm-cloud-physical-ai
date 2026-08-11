# Unitree Go2 — Locomotion (walk + navigate)

Closed-loop velocity walking for the Go2 through Unitree's high-level **`SportClient`** balance/gait
controller, steered on the robot's **measured** odometry (`rt/sportmodestate`). The robot-agnostic
control lives in [`lib/locomotion.py`](https://github.com/arm/arm-mhs-robotkit/blob/main/lib/locomotion.py); this module is the Go2 binding
([`go2_locomotion.py`](go2_locomotion.py) → `Go2Locomotion`).

> ⚠️ Read [`SAFETY.md`](../../../SAFETY.md) first. `SportClient.Move` commands a velocity that
> **persists with no dead-man** — drive motion only through the closed-loop helpers (which `stop()` on
> every exit) or a `try/finally` that stops the robot, and keep the [controller abort](../controller/)
> and physical e-stop in reach.

## The interface

`Go2Locomotion` implements the three `LocomotionController` primitives — `set_velocity(vx, vy, vyaw)`,
`pose()`, `stop()` — so it inherits the closed-loop helpers unchanged:

| Call | What it does |
|---|---|
| `walk_to((x, y))` | Walk to an odom-frame point, holding heading, closed-loop on measured pose. |
| `walk_forward(d)` | Walk `d` m along the current heading. |
| `turn_to(yaw)` | Rotate in place to a heading. |
| `stand()` / `recover()` / `stand_down()` | Balanced stand (ready to walk) / recover from lying/fallen / lie down. |
| `speed_level(l)` | Gait speed level (`-1`…`1`). |
| `.client` | The raw `SportClient` for the full gait menu (StaticWalk, TrotRun, ClassicWalk, obstacle modes). |

Frames are REP-103 (`+x` fwd, `+y` left, `+yaw` CCW), metres/radians. Pose is `position[0:2]` +
`imu_state.rpy[2]` from `rt/sportmodestate` (`SportModeState_`, the same message the G1 reads off
`rt/odommodestate`).

## Motion mode + lease — the Go2 gotcha the G1 doesn't have

A Go2 only accepts `SportClient` commands when its motion service is in a **sport mode**
(`normal`/`ai`) and no other client holds the **lease**. The most common "my Go2 ignores velocity
commands" cause is the wrong mode. So:

- `connect()` reads and prints the live mode (via `MotionSwitcherClient.CheckMode`) and warns if it
  isn't a sport mode.
- `ensure_sport_mode("normal")` selects the sport mode if needed. It reconfigures the controller and
  can make the robot shift/stand — call it only with a clear area + e-stop ready.
- `release_control()` releases the mode/lease — required before **low-level** `rt/lowcmd` control
  (see [`../deploy`](../deploy/)); high-level `Move` stops working until a sport mode is re-selected.

## Localization + planning

`pose()` uses the Go2's built-in state estimator (`rt/sportmodestate`). For obstacle-aware
navigation, feed the onboard LiDAR cloud (`rt/utlidar/cloud_deskewed`, via
[`../lidar_sight`](../lidar_sight/)) to the shared `Navigator`:

```python
from arm_mhs_robotkit.navigation import Navigator
Navigator().navigate(loco, goal_xy=(3.0, 0.0), cloud_source=get_go2_lidar_cloud)
```

Unlike the G1, the Go2 also publishes its **own** LiDAR-SLAM odometry + height/grid maps
(`rt/uslam/*`, `rt/utlidar/*`) and a graph-nav service (`rt/api/slam_operate`). For production
mapping/relocalization, prefer the robot's onboard SLAM over the demo `Navigator` A* — the A* here is
the dependency-free fallback for a quick clear-space traverse.

## Cross-quadruped

`Go2Locomotion` is the native, high-fidelity binding. Any **other** ROS 2 quadruped is driven by the
vendor-neutral [`lib/ros2_twist_locomotion.py`](https://github.com/arm/arm-mhs-robotkit/blob/main/lib/ros2_twist_locomotion.py)
(`Ros2TwistLocomotion` over `cmd_vel`/`odom`) — same `LocomotionController` interface, so every
helper and the `Navigator` work unchanged.

## Try it

```bash
# read-only: stream measured pose + velocity + mode, no motion
python3 unitree/go2/locomotion/go2_locomotion.py --iface eth0 --seconds 8

# DANGER: walk 1 m forward (typed confirmation required; legs move)
python3 unitree/go2/locomotion/go2_locomotion.py --iface eth0 --forward 1.0
```
