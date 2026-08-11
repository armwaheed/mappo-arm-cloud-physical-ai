# Unitree D1 — the Go2's back arm (`d1_arm`)

The **Unitree D1** ("Small Servo Arm") is a 6-DOF serial servo arm + single-jaw gripper (0.5 kg payload,
~0.55 m reach) that bolts onto the Go2's back. This module drives it from any host on the robot's
DDS net. Module: [`d1_arm.py`](d1_arm.py) → `D1Arm` (transport) + `ArmProtocol` (pure encoder).

> ⚠️ The arm moves under power. `relax()` drops it under gravity — support it. Read
> [`SAFETY.md`](../../../SAFETY.md), keep clear of the swept volume, and keep the e-stop in reach.

## How it reaches the bus — the expansion dock

The D1 speaks its own protocol over an **RJ45** link (24 V / 60 W). It is **not** wired to your host: the
Go2 **expansion dock** (the piggyback board, `192.168.123.100` — see the descriptor's `compute[]`) bridges
that link onto the robot's CycloneDDS bus. So from any host on `192.168.123.0/24`, the arm is three DDS
topics — the same `ChannelPublisher`/`ChannelSubscriber` plumbing as the rest of `unitree_sdk2py`:

| Topic | Dir | Type | Contents |
|---|---|---|---|
| `rt/arm_Command` | → arm | `unitree_arm/ArmString` | JSON command (see funcode table) |
| `rt/arm_Feedback` | ← arm | `unitree_arm/ArmString` | status / acks (address+funcode tagged) |
| `current_servo_angle` | ← arm | `unitree_arm/PubServoInfo` | per-servo angles (7: J1..J6 + jaw) |

## The `ArmString` protocol

Every command is a JSON string in `ArmString.data_`:

```json
{"seq": N, "address": 1, "funcode": F, "data": { ... }}
```

| funcode | method | `data` |
|---|---|---|
| 1 | `move_joint(id, angle)` | `{"id": 1..6, "angle": <rad>, "delay_ms": <ms>}` |
| 2 | `move_all(joints, gripper)` | `{"mode": M, "angle0": J1 … "angle5": J6, "angle6": jaw}` (rad) |
| 4 | (per-joint damping) | `{"id": 1..6, "mode": M}` (0 release · large hold) |
| 5 | `hold()` / `relax()` | `{"mode": M}` |
| 7 | `reset()` | *(no data)* — return to zero posture |

Feedback (`rt/arm_Feedback`) is tagged: `(address 2, funcode 3)` arm status, `(2,4)` motor-online,
`(3,1)` command received, `(3,2)` command executed. `parse_feedback()` labels these.

`ArmProtocol` is a **pure encoder** (no DDS) so the wire format is unit-tested (`test_d1_arm.py`, 12 cases).

## Joint envelope

Angles are **radians**, clamped to the D1 spec (the arm's kinematics model declares `angle="radian"`):

| Joint | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| range | ±135° | ±90° | ±90° | ±135° | ±90° | ±135° |

The jaw (servo 7) range is a **placeholder** pending an on-hardware check. `clamp=True` (default) holds
every command inside these limits.

## The IDL

`unitree_arm/ArmString` + `PubServoInfo` are **not** in stock `unitree_sdk2py` — they ship with the
Unitree D1 SDK. `D1Arm` resolves the `ArmString` type from the SDK, then D1Py, then a **self-contained
vendored CycloneDDS definition** with the on-wire type name (`unitree_arm.msg.dds_.ArmString_`), so this
module works even without the D1 SDK installed. Per-servo readback (`PubServoInfo`) is optional; commands
and `rt/arm_Feedback` status work without it. Pass `armstring_type=`/`servoinfo_type=` to inject the
canonical SDK classes if you have them.

## Kinematics (next step)

This module ships **joint-space** control + spec-limit clamping. Task-space FK/IK needs the D1 URDF/DH
table (the Unitree D1 SDK ships it; the community `Capstone-D1-Arm/d1_kinematics.xml` is a MuJoCo model).
Drop a solver in here and the `operate-arm-payload` skill can pose the arm in Cartesian space; until then,
command joints directly.

## Status

The interface was discovered **live on the bus**; the binding is **not yet exercised on the physical
arm** (`payloads[].verified_on_hardware: false`). Confirm the funcode map, joint signs, and the jaw range
with one supervised move, then promote the descriptor.

## Try it

```bash
# read-only: stream arm feedback + per-servo angles, no motion
python3 unitree/go2/d1_arm/d1_arm.py --iface eth0 --seconds 8

# DANGER: send the arm to zero (typed confirmation required; the arm moves)
python3 unitree/go2/d1_arm/d1_arm.py --iface eth0 --reset
```
