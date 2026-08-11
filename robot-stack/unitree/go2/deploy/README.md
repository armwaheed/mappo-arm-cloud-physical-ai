# Unitree Go2 — Deploy (low-level policy on the real motors)

The Go2 binding of the [`lib/policy_deploy`](https://github.com/arm/arm-mhs-robotkit/blob/main/lib/policy_deploy.py) **de-risk ladder**: a
`RobotIO` that reads `rt/lowstate` and commands `rt/lowcmd` so a trained RL policy can drive the legs.
Module: [`go2_robot_io.py`](go2_robot_io.py) → `Go2RobotIO`.

> ⚠️ **This is the most dangerous module in the stack.** `rt/lowcmd` holds the legs rigidly; a bad
> target — or a killed control process — makes the robot run away. Read [`SAFETY.md`](../../../SAFETY.md)
> **in full** before using it. High-level walking does NOT come through here — use
> [`../locomotion`](../locomotion/) (the vendor balancer). This path is only for a low-level whole-body
> policy, climbed through `PolicyDeploy`'s offline → partial → whole rungs, on a gantry first.

## What's Go2-specific

**Joint order.** The SDK `LowState`/`LowCmd` motor array is **per-leg** (`GO2_JOINT_ORDER`: FR
hip/thigh/calf, then FL, RR, RL), while IsaacLab orders the articulation **per-level**
(`GO2_ISAAC_JOINT_ORDER`: all hips, all thighs, all calves). A policy trained in Isaac emits actions in
per-level order; deploying it needs a remap. `lib/policy_deploy` maps by joint **name** (a proper
`DeployContract` handles this), and `reorder(vec, from_order, to_order)` is the pure helper for any path
carrying bare index vectors. Getting this wrong commands the wrong motors — hence the direct unit tests
(`test_go2_robot_io.py`, incl. an Isaac→SDK round-trip).

**Mode release.** Low-level control only works once the sport service has released the legs
(`MotionSwitcher.ReleaseMode`, via `Go2Locomotion.release_control()` or by hand). Until then `rt/lowcmd`
fights the vendor balancer.

## Safety guards built in

- `publish_targets` is **disabled unless `Go2RobotIO(enable_lowcmd=True)`** — importing or instantiating
  this module cannot move motors by accident. The default `main()` is read-only by construction.
- `verify_whole_body_ready()` and `confirm_abort_live()` **default to False** (inherited from `RobotIO`) —
  `run_whole` refuses unless a binding proves the robot is off the balancer and the kill-switch latches.
  Wire a [`Go2Remote`](../controller/) in as `abort_source` for the any-button abort.
- `damp_once()` drops to `kp=0, kd=2` (compliant hold) — the fast, un-blended leg release the ladder
  requires (the Go2 has no compliant overlay like the G1's `rt/arm_sdk`).

## Status

**Scaffold — NOT verified on hardware.** The state read + joint remap are exercised offline; the
`rt/lowcmd` publish path (frame header `head`/`level_flag`, motor mode, CRC, gains) mirrors the
`unitree_sdk2py` Go2 low-level example but has not been run on this robot. Bring it up on a gantry, prove the abort, and climb the de-risk ladder
one rung at a time. The descriptor's `walk_policy` block documents the obs/action layout to deploy
against.

## Try it (read-only)

```bash
python3 unitree/go2/deploy/go2_robot_io.py --iface eth0 --seconds 5   # streams joint state, no motion
```
