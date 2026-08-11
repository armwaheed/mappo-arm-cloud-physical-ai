# arm-mhs-unitree-go2

The **Model Hardware Standard (MHS) driver and on-robot control stack for the
Unitree Go2 EDU quadruped and its Unitree D1 servo arm.** It makes a physical
12-DOF Go2 (carrying the detachable 6-DOF + jaw D1 arm) a first-class MHS device —
reachable on the fabric next to a human partner or a G1 — and ships the
locomotion, arm, perception, and motor-deploy modules that run on the robot itself.

## What's here

- **`driver/go2/`** — the MHS **driver** (`Go2AgentDriver`, `device_type =
  "unitree_go2"`). Registers the robot on the fabric and exposes `get_status()`,
  `stop()`, `stand()` / `stand_down()`, `walk_forward(distance)` / `walk_to(x, y)`
  (closed-loop), and `arm_reset()` / `arm_move(joint_angles, gripper)`, with
  `arrived` / `blocked` / `arm_moved` events. Motion RPCs are gated behind
  `--allow-motion`, so a default device is **status-only**. It uses the two-env
  bridge — `go2_agent.py` (MHS env, Python ≥3.10) drives `go2_drive.py` in the
  Unitree SDK env (`unitree_sdk2py` + CycloneDDS). See
  [`driver/go2/SKILL.md`](driver/go2/SKILL.md).
- **`unitree/go2/`** — the on-robot control + perception stack: host/DDS connect,
  the installer, the `SportClient` velocity-walk locomotion binding (with the Go2
  motion-mode + lease handling), the D1 back-arm driver (DDS `ArmString`), the L1
  dome-LiDAR `lidar_sight`, the handheld any-button abort controller, the
  low-level `rt/lowcmd` policy-deploy `RobotIO` (with the SDK↔Isaac joint-order
  remap), and the depth-camera module. Start at
  [`unitree/go2/README.md`](unitree/go2/README.md).

Read [`SAFETY.md`](SAFETY.md) before running any on-hardware motor control: the
Go2's high-level `SportClient.Move` commands a **persistent** velocity with no
dead-man, and the low-level `rt/lowcmd` path holds the legs rigidly — both run the
robot away if mishandled.

## The arm-mhs-`<vendor>`-`<robot>` family

This repo is one driver in a family that all build on a shared robot-agnostic
core:

- [`arm-mhs-robotkit`](https://github.com/arm/arm-mhs-robotkit) — the shared core
  (locomotion, policy-deploy, safe-stop, the MHS sidecar, discovery + vendor-neutral
  skills). **This repo depends on it.**
- [`arm-mhs-unitree-g1`](https://github.com/arm/arm-mhs-unitree-g1) — the Unitree G1 humanoid driver.
- [`arm-mhs-unitree-go2`](https://github.com/arm/arm-mhs-unitree-go2) — this repo (Go2 quadruped + D1 arm).
- [`arm-mhs-nvidia-isaaclab`](https://github.com/arm/arm-mhs-nvidia-isaaclab) — Isaac Lab / DGX Spark sim + policy skills.
- [`arm-mhs-human-partner`](https://github.com/arm/arm-mhs-human-partner) — the human-agent side of the help loop.

The Go2's locomotion binding subclasses the core's `LocomotionController` / `Pose`
and its deploy `RobotIO` subclasses `RobotIO` / `RobotState`; the driver imports
the MHS sidecar as `arm_mhs_robotkit.mhs_sidecar` and the safe-stop wrapper as
`arm_mhs_robotkit.safe_stop`. Quadruped control and the D1 payload are driven
vendor-neutrally through the core's
[`quadruped-locomotion`](https://github.com/arm/arm-mhs-robotkit/blob/main/skills/quadruped-locomotion/SKILL.md)
and
[`operate-arm-payload`](https://github.com/arm/arm-mhs-robotkit/blob/main/skills/operate-arm-payload/SKILL.md)
skills.

Registry entry: **`arm-mhs-unitree-go2`** in the MHS registry.

## Install & run (pre-launch)

The `mhs` package is **not yet on PyPI** (pre-launch), so install the SDK editable
from a clone of
[`modelhardwarestandard/python-sdk`](https://github.com/modelhardwarestandard/python-sdk),
then the shared core and this package editable:

```bash
# 1) MHS Python SDK, editable from a local clone (pick your wire transport)
git clone https://github.com/modelhardwarestandard/python-sdk
pip install -e "./python-sdk[wire,nats]"

# 2) the shared core, then this driver (editable)
pip install -e ../arm-mhs-robotkit
pip install -e .            # arm-mhs-unitree-go2
```

Run the driver as a script **on the robot** (in the Unitree SDK env, DDS on `eth0`):

```bash
# status-only device (safe default) — appears on the fabric, no motion RPCs will move it
python driver/go2/go2_agent.py --creds /path/to/<robot>.creds.json

# enable motion (clear area + controller abort + e-stop required)
python driver/go2/go2_agent.py --creds /path/to/<robot>.creds.json --allow-motion

# status-only smoke test, no fabric
python driver/go2/go2_agent.py --self-test
```

The on-robot control stack installs to the robot via
[`unitree/go2/install/`](unitree/go2/install/SKILL.md).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
