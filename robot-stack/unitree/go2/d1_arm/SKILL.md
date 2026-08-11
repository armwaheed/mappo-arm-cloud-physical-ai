---
name: unitree-go2-d1-arm
description: >-
  Drive the Unitree D1 "Small Servo Arm" (the 6-DOF + jaw servo arm on the Go2's back) over DDS.
  Publishes ArmString JSON commands to rt/arm_Command (set one joint, set all 7 servos, per-/all-joint
  damping, reset) and reads status from rt/arm_Feedback + per-servo angles from current_servo_angle. The
  arm reaches the Go2 DDS bus through the expansion dock (piggyback board) — no direct wire. Angles are
  radians, clamped to the D1 spec envelope. Use when the Go2 must MANIPULATE (pick/place, press, hold) as
  opposed to walk. The generalized cross-quadruped version is skills/operate-arm-payload.
metadata:
  tags: [unitree-go2, unitree-d1, servo-arm, manipulation, payload, dds, arm-string, expansion-dock]
---

# Unitree D1 — Go2 back arm (joint-space control)

Command the D1 servo arm through its `ArmString` JSON protocol on `rt/arm_Command`. The protocol, the
funcode table, the joint envelope, and the safety rules are in **[`README.md`](README.md)** — this skill
is the agent entry point. Module: [`d1_arm.py`](d1_arm.py). The arm is characterized as a `payloads[]`
entry in the [robot descriptor](../../../skills/discover-robot/descriptors/unitree_go2_edu.json), reached
via the `expansion-dock` compute node.

## When to use

- The Go2 must **manipulate** something (pick/place a light object ≤0.5 kg, press, hold) — this is the
  arm, not the legs (see [`../locomotion`](../locomotion/SKILL.md) for driving the base).
- You're filling a descriptor's `payloads[]` block (arm interface + joint limits + control topics).
- You need to **reset**, **hold** (damp), or **relax** the arm.

## How to use

```python
import sys; sys.path.insert(0, "unitree/go2/d1_arm")
from d1_arm import D1Arm
import math

arm = D1Arm(iface="eth0"); arm.connect()
arm.reset()                                  # zero posture
arm.move_joint(2, math.radians(-30))         # J2 to -30° (clamped to ±90°)
arm.move_all([0, -0.5, 0.5, 0, 0, 0], gripper_rad=0.3)  # all joints + jaw, one command
print(arm.feedback())                        # {'kind': 'exec_ack', ...}
arm.hold()                                   # engage holding damping
```

Angles are **radians** and every command is **clamped** to the D1 spec envelope
(J1 ±135°, J2 ±90°, J3 ±90°, J4 ±135°, J5 ±90°, J6 ±135°). Offline, `ArmProtocol` encodes every command
as a JSON string with no DDS, so the wire format is fully testable (`python3 test_d1_arm.py`).

## Safety

`reset` / `move_*` / `relax` **move the arm under power**. `relax()` releases the servos and the arm
**drops under gravity** — support it first. Keep clear of the arm's swept volume and the e-stop in reach.
The `--reset` CLI requires a typed confirmation; the default run only streams feedback. This binding's
funcode map was read from the D1 protocol but is **not yet verified on this arm** — confirm joint signs
and the jaw range with one supervised move before trusting it.
