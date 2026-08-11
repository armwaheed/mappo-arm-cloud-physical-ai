---
name: unitree-go2-controller
description: >-
  Read the Unitree Go2 handheld remote and turn ANY button press into a clean software abort of an
  autonomous routine. Parses LowState_.wireless_remote (the 40-byte xRockerBtnDataStruct: uint16 button
  bitmask + four float sticks), arms once buttons release, and latches abort on the first press. Wire
  Go2Remote.aborted into LocomotionController.set_abort_source so a walk/manipulation routine stops the
  instant an operator touches the remote. Shares the parser + latch with the G1 controller (verified
  button-by-button); only the LowState IDL differs. NOT an e-stop — it holds stance, the physical
  power/e-stop is the real stop.
metadata:
  tags: [unitree-go2, controller, remote, abort, safety, wireless-remote]
---

# Unitree Go2 — Controller abort

Turn a handheld-remote button press into a clean software abort. The struct layout, the arming logic,
and the safety boundary are in **[`README.md`](README.md)** — this skill is the agent entry point.
Module: [`go2_remote.py`](go2_remote.py).

## When to use

- You're about to run an **autonomous** routine (walk to a goal, an arm move) and want a one-touch
  operator abort that stops the software cleanly.
- You need to read the remote's **buttons/sticks** live.

## How to use

```python
import sys; sys.path.insert(0, "unitree/go2/controller"); sys.path.insert(0, "unitree/go2/locomotion")
from go2_remote import Go2Remote
from go2_locomotion import Go2Locomotion

remote = Go2Remote(iface="eth0"); remote.connect(); remote.wait_until_armed()
loco = Go2Locomotion(init_dds=False); loco.connect()   # remote already inited DDS
loco.set_abort_source(remote.aborted)                  # any button press → walk_* returns "aborted"
loco.walk_forward(2.0)
```

Read-only dry test (no motion): `python3 unitree/go2/controller/go2_remote.py --iface eth0` — press
buttons and confirm `ABORTED=True` latches.

## Safety

This is a **clean software abort**, NOT an emergency stop. It stops the routine and holds the stance;
it does not damp the robot to the floor. The controller's firmware damping and the physical power/e-stop
work even if this process is hung — keep them in reach. See [`SAFETY.md`](../../../SAFETY.md).
