---
name: unitree-go2-install
description: >-
  Bring up the unitree_sdk2py environment the Go2 control stack needs, on the robot's onboard
  Jetson or any host on the Go2 net. Creates a venv, installs CycloneDDS (linked against the robot's
  existing ~/cyclonedds_ws build to skip the aarch64 source rebuild) + unitree_sdk2py, and with --verify
  runs a read-only probe that PASSES only if rt/lowstate + rt/sportmodestate are actually visible. This
  is the step-zero the discover-robot skill delegates to before characterizing a Go2.
metadata:
  tags: [unitree-go2, install, setup, unitree-sdk2py, cyclonedds, bootstrap, verify]
---

# Unitree Go2 — Install (SDK env bring-up + verify)

Stand up the `unitree_sdk2py` environment, then prove the robot's DDS is visible. Scripts:
[`install.sh`](install.sh), [`verify.sh`](verify.sh), [`requirements.txt`](requirements.txt).

## When to use

- First time on a **fresh host/robot** before running any Go2 skill.
- As the **step-zero** of `discover-robot` (arm-dc-robotkit `skills/discover-robot/SKILL.md`) for a Go2.
- To **verify** the DDS link (`--verify` / `verify.sh`) after a network change.

## How to use

```bash
# on the robot's onboard Jetson (ssh unitree@192.168.123.18) or a host on the Go2 net
cd robotics-connect
./unitree/go2/install/install.sh --verify          # install, then PASS/FAIL probe (no motion)

source ~/robotics-connect-go2/bin/activate         # the env it created
export CYCLONEDDS_URI="file://$PWD/unitree/go2/cyclonedds.xml"
```

`--verify` PASSES only if `rt/lowstate` and `rt/sportmodestate` deliver a frame within 5 s — a real
check that the robot is on and the interface/DDS config is right, not just that the packages imported.

## What it does

1. Creates a Python venv (`~/robotics-connect-go2` by default).
2. Installs `cyclonedds==0.10.2` — **linking against the robot's existing `~/cyclonedds_ws` build**
   (`CYCLONEDDS_HOME`) so it skips the from-source rebuild that bites on aarch64. Falls back to a source
   build if that workspace isn't present.
3. Clones + installs `unitree_sdk2py` editable (it isn't on PyPI).
4. (`--verify`) runs [`verify.sh`](verify.sh): read-only DDS probe of `rt/lowstate` + `rt/sportmodestate`,
   and a resolvability check for the D1 arm's `ArmString` IDL.

The D1 arm messages ship with the Unitree D1 SDK; if it isn't installed,
[`../d1_arm`](../d1_arm/) vendors a self-contained `ArmString` so arm commands still work.
