# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unitree D1 (Go2 "Small Servo Arm") — joint-space control over DDS.

The D1 is a 6-DOF serial servo arm + single-jaw gripper that bolts onto the Go2's
back. It speaks its own protocol over an RJ45 link; the Go2 **expansion dock**
(the piggyback board) bridges that link onto the robot's CycloneDDS bus, so from
any host on the ``192.168.123.0/24`` net the arm is three DDS topics:

  * ``rt/arm_Command``   — commands  (``unitree_arm/ArmString``: a JSON string in ``.data_``)
  * ``rt/arm_Feedback``  — status    (same ``ArmString`` type; address/funcode-tagged JSON)
  * ``current_servo_angle`` — per-servo angles (``unitree_arm/PubServoInfo``)

The command JSON is ``{"seq":N,"address":1,"funcode":F,"data":{...}}``. The funcode
table (see :class:`ArmProtocol`) is the arm's whole control surface: set one joint,
set all seven servos (6 joints + jaw), per-joint / all-joint damping, and reset.

This module splits into a **pure encoder** (:class:`ArmProtocol`, fully unit-tested
with no DDS) and a thin **DDS transport** (:class:`D1Arm`). Angles are **DEGREES** on the
wire, in both directions, and joint ids are **0-based** — confirmed by Unitree's service
doc, by the SDK's ``joint_angle_control.cpp`` (``{"id":5,"angle":60}``), and by live
feedback. :data:`JOINT_LIMITS_DEG` is the envelope the FIRMWARE enforces on commands; the
joints back-drive well past it, so a measured angle outside it is normal.

For task-space work use :mod:`d1_fk`; for smooth motion use :mod:`trajectory`, which
streams eased waypoints through funcode 2 ``mode: 1`` instead of stepping joints one at
a time.

STATUS: exercised on hardware (unit 192.168.123.18, 2026-07-09) — a marker was grasped
and lifted using this protocol. SAFETY: the arm moves under power, and **it has no
unconditional software stop.** funcode 6 never cuts power; funcode 5's discharge does not
latch. The only certain stop is the physical power switch. Keep
``controller/arm_panic.py`` in a second shell.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import threading
import time

# ── Command topics ────────────────────────────────────────────────────────────
COMMAND_TOPIC = "rt/arm_Command"
FEEDBACK_TOPIC = "rt/arm_Feedback"
SERVO_STATE_TOPIC = "current_servo_angle"

# ── Joint envelope (Unitree D1 spec) ──────────────────────────────────────────
# Joints are 0-BASED on the wire: `angle0`..`angle5` are J0..J5, `angle6` is the jaw,
# and funcode 1's `id` field is 0..6. Unitree's doc says so ("starting from the base is
# 0, to the clamp is 6") and the SDK's joint_angle_control.cpp sends {"id":5,"angle":60}.
NUM_ARM_JOINTS = 6          # J0..J5
GRIPPER_INDEX = 6           # servo/angle index 6 is the jaw
NUM_SERVOS = 7              # 6 joints + gripper

#: Unitree's documented envelope, in DEGREES, keyed by the 0-based joint id.
#: The FIRMWARE ENFORCES THIS: commanding J1 to -95 parks it at -90.3 (measured
#: 2026-07-09). It is NOT the mechanical travel — the joints back-drive well past it
#: (a hand-taught grasp sat at J1 = +96.7; gravity rests J1 at -91.2), so a *measured*
#: angle outside these bounds is normal and must not be clamped on read-back.
#: See d1_fk.COMMANDABLE_LIMITS_DEG, and plan IK against it.
JOINT_LIMITS_DEG = {
    0: (-135.0, 135.0),
    1: (-90.0, 90.0),
    2: (-90.0, 90.0),
    3: (-135.0, 135.0),
    4: (-90.0, 90.0),
    5: (-135.0, 135.0),
}

#: Jaw travel in degrees. Measured on unit 192.168.123.18: +49.9 is fully open, and the
#: jaw stalls at -1.35 on a 17 mm EXPO barrel. DECREASING angle6 CLOSES. The 65 mm stroke
#: spans roughly +49.9 (open) to -20 (shut).
GRIPPER_LIMIT_DEG = (-20.0, 50.0)

#: funcode 4/5 `mode` is BINARY (Unitree: "mode is 0 to release force, and mode is 1 to
#: enable"). It is NOT a 0..80000 stiffness. The old `DAMP_HOLD = 1000` was outside the
#: protocol entirely.
DAMP_RELEASE = 0            # 卸力 — discharge force. Zero torque. The abort.
DAMP_ENABLE = 1             # 使能 — enable; latches each joint at its current angle.

#: funcode 6 `power`. ⚠ MEASURED: this only ever powers the arm ON. Sending {"power":0}
#: to a cold arm takes power_status 0 -> 1. Unitree documents it as an emergency stop;
#: it is not one. See unitree/go2/controller/arm_panic.py.
POWER_ON = 1
POWER_OFF = 0               # a no-op on this firmware — do not use it as a stop

_dds_ready = False


def _ensure_dds(iface: str) -> None:
    """Initialise the Cyclone DDS channel factory once per process (matches the
    guard used across the Go2 bindings)."""
    global _dds_ready
    if _dds_ready:
        return
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, iface)
    _dds_ready = True


def clamp_joint(joint_id: int, angle_deg: float) -> float:
    """Clamp a joint angle (DEGREES) to the firmware envelope. ``joint_id`` is 0-based."""
    lo, hi = JOINT_LIMITS_DEG[joint_id]
    return max(lo, min(hi, float(angle_deg)))


def clamp_gripper(angle_deg: float) -> float:
    """Clamp the jaw angle (DEGREES). Decreasing closes."""
    lo, hi = GRIPPER_LIMIT_DEG
    return max(lo, min(hi, float(angle_deg)))


class ArmProtocol:
    """Pure encoder for the D1 ``ArmString`` command protocol — no DDS, no I/O.

    Every method returns the JSON string that goes in ``ArmString.data_``. Kept
    separate so the wire format is unit-testable and the transport stays thin.
    """

    ADDRESS = 1

    @classmethod
    def _frame(cls, funcode: int, data: dict | None = None, seq: int = 0) -> str:
        obj: dict = {"seq": seq, "address": cls.ADDRESS, "funcode": funcode}
        if data is not None:
            obj["data"] = data
        return json.dumps(obj, separators=(",", ":"))

    @classmethod
    def reset(cls, seq: int = 0) -> str:
        """funcode 7 — return all joints to the zero posture."""
        return cls._frame(7, seq=seq)

    @classmethod
    def set_joint(cls, joint_id: int, angle_deg: float, delay_ms: float = 0.0,
                  seq: int = 0) -> str:
        """funcode 1 — drive one joint (0-based id 0..6) to ``angle_deg`` over ``delay_ms``."""
        return cls._frame(1, {"id": int(joint_id), "angle": float(angle_deg),
                              "delay_ms": float(delay_ms)}, seq)

    @classmethod
    def set_all_joints(cls, angles7: list[float], mode: int = 0, seq: int = 0) -> str:
        """funcode 2 — set all 7 servos at once. ``angles7`` = [J0..J5, jaw] in DEGREES.

        ``mode`` 0 is point-to-point smoothing of 10 Hz data; ``mode`` 1 is the trajectory
        blender. For smooth motion stream eased waypoints with ``mode=1`` — see
        :mod:`trajectory`. A single mode-1 command to a distant target truncates; that is
        the blender asking for a trajectory, not a bug.
        """
        if len(angles7) != NUM_SERVOS:
            raise ValueError(f"expected {NUM_SERVOS} angles (6 joints + jaw), "
                             f"got {len(angles7)}")
        data: dict = {"mode": int(mode)}
        for i, a in enumerate(angles7):
            data[f"angle{i}"] = float(a)
        return cls._frame(2, data, seq)

    @classmethod
    def enable_joint(cls, joint_id: int, mode: int, seq: int = 0) -> str:
        """funcode 4 — one joint: ``mode`` 0 discharges force, 1 enables. 0-based id."""
        return cls._frame(4, {"id": int(joint_id), "mode": int(mode)}, seq)

    @classmethod
    def enable_all(cls, mode: int, seq: int = 0) -> str:
        """funcode 5 — all joints: ``mode`` 0 discharges force, 1 enables.

        Enabling latches every joint at its current angle (this is what makes Unitree's
        drag-teach, 拖动记忆示教, work). Discharging is the arm's best software abort — but
        it does NOT latch: a later funcode-1/2 angle command silently re-enables the joint.
        """
        return cls._frame(5, {"mode": int(mode)}, seq)

    @classmethod
    def set_power(cls, power: int, seq: int = 0) -> str:
        """funcode 6 — motor power switch.

        ⚠ Measured on hardware: this only ever powers the arm **ON**. ``{"power": 0}``
        sent to a cold arm takes ``power_status`` 0 → 1, and every send acks
        ``exec_status: 1``. Unitree documents it as usable for emergency stop. It is not.
        """
        return cls._frame(6, {"power": int(power)}, seq)


def parse_feedback(data_str: str) -> dict:
    """Decode an ``rt/arm_Feedback`` ``ArmString.data_`` JSON into a labelled dict.

    The arm tags feedback by (address, funcode):
      (2,1)=joint angles, DEGREES, 10 Hz — the frame you actually get most of the time
      (2,3)=arm status · (2,4)=motor online (1 ok / 0 faulty) · (3,1)=command received
      (3,2)=command executed. Unknown tags pass through under ``kind='unknown'``.

    ⚠ ``arm_status.enable_status`` is not reliable: it read 0 immediately after enabling
    all joints, then flipped to 1 once motion started. Do not gate on it.
    """
    data = json.loads(data_str)
    address, funcode = data.get("address"), data.get("funcode")
    kind = {
        (2, 1): "joint_angles",
        (2, 3): "arm_status",
        (2, 4): "motor_online",
        (3, 1): "recv_ack",
        (3, 2): "exec_ack",
    }.get((address, funcode), "unknown")
    return {"kind": kind, "address": address, "funcode": funcode, "data": data.get("data"), "raw": data}


class D1Arm:
    """DDS transport for the D1 arm: publish commands, read feedback.

    ``armstring_type`` / ``servoinfo_type`` let you inject the exact IDL classes
    from the Unitree D1 SDK if it is installed; by default they are resolved from
    the SDK, then D1Py, then a self-contained vendored ``ArmString`` definition
    (see :func:`_resolve_armstring`).
    """

    def __init__(self, iface: str = "eth0", init_dds: bool = True,
                 armstring_type=None, servoinfo_type=None) -> None:
        self._iface = iface
        self._init_dds = init_dds
        self._armstring_type = armstring_type
        self._servoinfo_type = servoinfo_type
        self._pub = None
        self._sub_fb = None
        self._sub_servo = None
        self._lock = threading.Lock()
        self._feedback: dict | None = None
        self._servo = None
        self._seq = 0

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def connect(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber

        ArmString = self._armstring_type or _resolve_armstring()
        self._armstring_type = ArmString

        if self._init_dds:
            _ensure_dds(self._iface)

        self._pub = ChannelPublisher(COMMAND_TOPIC, ArmString)
        self._pub.Init()

        self._sub_fb = ChannelSubscriber(FEEDBACK_TOPIC, ArmString)
        self._sub_fb.Init(self._on_feedback, 10)

        # Per-servo angles need the D1 PubServoInfo IDL; it is optional — the arm
        # status/acks on rt/arm_Feedback (ArmString) cover control without it.
        servo_type = self._servoinfo_type or _resolve_servoinfo()
        if servo_type is not None:
            self._servoinfo_type = servo_type
            self._sub_servo = ChannelSubscriber(SERVO_STATE_TOPIC, servo_type)
            self._sub_servo.Init(self._on_servo, 10)
        else:
            print("[D1Arm] PubServoInfo IDL not found — per-servo angle readback "
                  "disabled (commands + rt/arm_Feedback still work).")

    def _on_feedback(self, msg) -> None:
        try:
            fb = parse_feedback(msg.data_)
        except Exception:
            return
        with self._lock:
            self._feedback = fb

    def _on_servo(self, msg) -> None:
        with self._lock:
            self._servo = msg

    def shutdown(self) -> None:
        for chan in (self._sub_fb, self._sub_servo, self._pub):
            if chan is not None:
                try:
                    chan.Close()
                except Exception:
                    pass
        self._sub_fb = self._sub_servo = self._pub = None

    # ── Commands ────────────────────────────────────────────────────────────
    def _publish(self, data_str: str) -> None:
        if self._pub is None:
            raise RuntimeError("connect() first")
        self._pub.Write(self._armstring_type(data_=data_str))

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) % 65536
        return self._seq

    def reset(self) -> None:
        """Return the arm to its zero posture."""
        self._publish(ArmProtocol.reset(seq=self._next_seq()))

    def move_joint(self, joint_id: int, angle_deg: float, delay_ms: float = 0.0,
                   clamp: bool = True) -> float:
        """Drive one joint (0-based J0..J5) to ``angle_deg``; returns the sent angle."""
        if joint_id not in JOINT_LIMITS_DEG:
            raise ValueError(f"joint_id must be 0..{NUM_ARM_JOINTS - 1}, got {joint_id}")
        angle = clamp_joint(joint_id, angle_deg) if clamp else float(angle_deg)
        self._publish(ArmProtocol.set_joint(joint_id, angle, delay_ms, self._next_seq()))
        return angle

    def move_all(self, joint_angles_deg, gripper_deg: float | None = None,
                 mode: int = 0, clamp: bool = True) -> list[float]:
        """Set all 6 joints (+ optional jaw) at once; returns the 7 sent angles, DEGREES.

        ``joint_angles_deg`` is J0..J5. ``gripper_deg`` sets the jaw; omit it to hold the
        jaw open. With ``clamp`` (default) every value is held inside the firmware envelope
        — which the firmware would do anyway, silently. See :meth:`move_jaw` for the trap.
        """
        joints = list(joint_angles_deg)
        if len(joints) != NUM_ARM_JOINTS:
            raise ValueError(f"expected {NUM_ARM_JOINTS} joint angles, got {len(joints)}")
        jaw = GRIPPER_LIMIT_DEG[1] if gripper_deg is None else float(gripper_deg)
        if clamp:
            joints = [clamp_joint(i, a) for i, a in enumerate(joints)]
            jaw = clamp_gripper(jaw)
        angles7 = joints + [jaw]
        self._publish(ArmProtocol.set_all_joints(angles7, mode, self._next_seq()))
        return angles7

    def move_jaw(self, angle_deg: float, delay_ms: float = 0.0, clamp: bool = True) -> float:
        """Move ONLY the jaw. Decreasing ``angle_deg`` closes.

        Uses funcode 1 on servo 6, so it cannot disturb the arm. The previous
        ``set_gripper`` read the arm's joint angles and fed them back through
        ``move_all``; with the units bug that clamped every joint to its bound and
        commanded the whole arm to its extremes while "only moving the jaw".
        """
        angle = clamp_gripper(angle_deg) if clamp else float(angle_deg)
        self._publish(ArmProtocol.set_joint(GRIPPER_INDEX, angle, delay_ms, self._next_seq()))
        return angle

    def enable(self) -> None:
        """Enable all joints. Each latches at its current angle (drag-teach semantics)."""
        self._publish(ArmProtocol.enable_all(DAMP_ENABLE, self._next_seq()))

    def discharge(self) -> None:
        """Zero-torque all joints — the best software abort this arm has.

        NOT a latch: a later angle command re-enables the joint. NOT a brake either: with
        nothing propping it, a discharged arm falls to its nearest mechanical rest.
        The only certain stop is the physical power switch. See ``controller/arm_panic.py``.
        """
        self._publish(ArmProtocol.enable_all(DAMP_RELEASE, self._next_seq()))

    def power_on(self) -> None:
        """Energise the motor bus (funcode 6). The arm boots with ``power_status: 0``."""
        self._publish(ArmProtocol.set_power(POWER_ON, self._next_seq()))

    # ── Feedback ────────────────────────────────────────────────────────────
    def feedback(self) -> dict | None:
        """The latest decoded ``rt/arm_Feedback`` frame (status / acks), or None."""
        with self._lock:
            return dict(self._feedback) if self._feedback else None

    def servo_angles(self) -> list[float] | None:
        """The latest 7 servo angles [J0..J5, jaw] in DEGREES, or None if unavailable."""
        with self._lock:
            servo = self._servo
        if servo is None:
            return None
        return [float(getattr(servo, f"servo{i}_data_")) for i in range(NUM_SERVOS)]


# ── IDL resolution ────────────────────────────────────────────────────────────
# The D1 arm messages are NOT part of stock unitree_sdk2py; they ship with the
# Unitree D1 SDK. We resolve the ArmString type from the SDK if present, else fall
# back to a self-contained CycloneDDS definition with the on-wire type name.
def _resolve_armstring():
    for modpath, name in (("unitree_arm.msg.dds_", "ArmString_"),
                          ("d1py_sdk.msgs.ArmString", "ArmString_")):
        try:
            return getattr(importlib.import_module(modpath), name)
        except Exception:
            pass
    return _vendored_armstring()


def _resolve_servoinfo():
    for modpath, name in (("unitree_arm.msg.dds_", "PubServoInfo_"),
                          ("d1py_sdk.msgs.PubServoInfo", "PubServoInfo_")):
        try:
            return getattr(importlib.import_module(modpath), name)
        except Exception:
            pass
    return None  # optional — see connect()


_VENDORED_ARMSTRING = None


def _vendored_armstring():
    """A minimal CycloneDDS ``ArmString`` with the D1's on-wire type name.

    Loaded from the sibling ``_arm_idl`` module (by path, under a unique name) rather
    than defined here on purpose: this module uses ``from __future__ import
    annotations``, which stringifies dataclass annotations and makes cyclonedds 0.10.2
    fail to populate the topic type (``Type str ... cannot be resolved``). ``_arm_idl``
    is kept on eager annotations so the IDL resolves. Loading it lazily keeps this
    module importable with no cyclonedds present (the protocol tests don't need it).
    """
    global _VENDORED_ARMSTRING
    if _VENDORED_ARMSTRING is None:
        _spec = importlib.util.spec_from_file_location(
            "d1_arm_vendored_idl", os.path.join(os.path.dirname(__file__), "_arm_idl.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _VENDORED_ARMSTRING = _mod.ArmString_
    return _VENDORED_ARMSTRING


def main() -> None:
    ap = argparse.ArgumentParser(description="Unitree D1 arm diagnostics.")
    ap.add_argument("--iface", default="eth0", help="DDS network interface")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="stream arm feedback (read-only) for this long")
    ap.add_argument("--reset", action="store_true",
                    help="DANGER: send the arm to its zero posture (it will MOVE)")
    args = ap.parse_args()

    arm = D1Arm(iface=args.iface)
    arm.connect()
    try:
        if args.reset:
            reply = input("About to RESET the arm to zero — it WILL move. "
                          "Clear of the arm + e-stop ready? [type 'reset']: ")
            if reply.strip().lower() != "reset":
                print("aborted.")
                return
            arm.reset()
            print("reset sent.")
            time.sleep(2.0)

        print(f"streaming arm feedback for {args.seconds:.0f}s (read-only)…")
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            fb = arm.feedback()
            angles = arm.servo_angles()          # already degrees — do not convert
            deg = [round(a, 1) for a in angles] if angles else None
            print(f"  feedback={fb['kind'] if fb else '—'}  servo_deg={deg}")
            time.sleep(0.5)
    finally:
        arm.shutdown()


if __name__ == "__main__":
    main()
