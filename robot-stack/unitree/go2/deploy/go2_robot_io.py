# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 binding of the ``arm_mhs_robotkit.policy_deploy`` de-risk ladder (LOW-LEVEL).

This is the Go2 ``RobotIO`` for putting a trained RL policy on the **real motors**
via ``rt/lowcmd`` — the most safety-sensitive path in the stack. High-level walking
does NOT come through here (use ``unitree/go2/locomotion``, the vendor balancer);
this is only for a low-level whole-body policy deployed through
``arm_mhs_robotkit.policy_deploy.PolicyDeploy``'s offline → partial → whole de-risk rungs.

Two Go2-specific facts this module carries:

  * **Joint order.** The SDK ``LowState``/``LowCmd`` motor array is **per-leg**
    (FR hip/thigh/calf, then FL, RR, RL — :data:`GO2_JOINT_ORDER`), while IsaacLab
    orders the articulation **per-level** (all hips, all thighs, all calves —
    :data:`GO2_ISAAC_JOINT_ORDER`). A policy trained in Isaac emits actions in the
    per-level order; deploying it requires a remap. ``arm_mhs_robotkit.policy_deploy`` maps by
    joint NAME (so a proper contract handles this), and :func:`reorder` is the pure
    helper for any path that carries bare index vectors.
  * **Mode release.** Low-level control only works once the sport service has
    released the legs (``MotionSwitcher.ReleaseMode``, done by
    ``Go2Locomotion.release_control()`` / by hand). Until then ``rt/lowcmd`` fights
    the vendor balancer. The whole-body gates below fail safe by default.

SAFETY — read ``SAFETY.md``. ``rt/lowcmd`` holds the legs rigidly; a bad target or a
killed process runs the robot away. ``publish_targets`` is HARD-GUARDED behind
``enable_lowcmd=True`` so importing/instantiating this cannot move motors by
accident, and the whole-body readiness + abort gates default to False (refuse). Bring
the robot up on a gantry, prove the abort, and climb the de-risk ladder — never jump
to ``run_whole`` on the floor.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from arm_mhs_robotkit.policy_deploy import RobotIO, RobotState

# ── Go2 joint order ───────────────────────────────────────────────────────────
# SDK LowState/LowCmd motor index order (per-leg): the on-wire order for rt/lowstate
# and rt/lowcmd. LegID FR=0, FL=1, RR=2, RL=3; each leg is hip, thigh, calf.
GO2_JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
# IsaacLab articulation order (per-level): all hips, then all thighs, then all calves.
GO2_ISAAC_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
NUM_MOTORS = 12
_NAME_TO_SDK_INDEX = {name: i for i, name in enumerate(GO2_JOINT_ORDER)}

# Go2 low-level motor mode: 0x01 = servo (PMSM) position/torque control.
MOTOR_MODE_SERVO = 0x01

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


def reorder(vec, from_order: list, to_order: list) -> np.ndarray:
    """Permute a per-joint vector from one joint ordering to another, by name.

    ``reorder(action, GO2_ISAAC_JOINT_ORDER, GO2_JOINT_ORDER)`` turns an Isaac
    (per-level) action vector into SDK (per-leg) motor order. Raises if the orders
    aren't the same joint set.
    """
    vec = np.asarray(vec, dtype=float)
    if set(from_order) != set(to_order):
        raise ValueError("reorder: from_order and to_order must be the same joint set")
    if vec.shape[0] != len(from_order):
        raise ValueError(f"reorder: vec has {vec.shape[0]} entries, order has {len(from_order)}")
    index = {name: i for i, name in enumerate(from_order)}
    return np.array([vec[index[name]] for name in to_order])


class Go2RobotIO(RobotIO):
    """Low-level ``rt/lowstate`` → ``rt/lowcmd`` RobotIO for the Go2.

    ``publish_targets`` is disabled unless ``enable_lowcmd=True`` — a deliberate
    guard so this cannot command motors by accident. Pass a ``Go2Remote`` as
    ``abort_source`` to wire the handheld any-button abort into the de-risk ladder.
    """

    def __init__(self, iface: str = "eth0", init_dds: bool = True,
                 enable_lowcmd: bool = False, abort_source=None) -> None:
        self._iface = iface
        self._init_dds = init_dds
        self._enable_lowcmd = enable_lowcmd
        self._abort_source = abort_source
        self._pub = None
        self._sub = None
        self._crc = None
        self._LowCmd = None
        self._lock = threading.Lock()
        self._state = None

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def connect(self) -> None:
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        if self._init_dds:
            _ensure_dds(self._iface)
        self._LowCmd = LowCmd_
        self._crc = CRC()

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        if self._enable_lowcmd:
            self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self._pub.Init()

        deadline = time.monotonic() + 3.0
        while self._state is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self._state is None:
            self.shutdown()
            raise RuntimeError("no rt/lowstate in 3 s — is the robot on?")

    def _on_state(self, msg) -> None:
        with self._lock:
            self._state = msg

    def shutdown(self) -> None:
        for chan in (self._sub, self._pub):
            if chan is not None:
                try:
                    chan.Close()
                except Exception:
                    pass
        self._sub = self._pub = None

    def _new_lowcmd(self):
        """A ``LowCmd_`` with the Go2 low-level frame header set.

        The firmware ignores an ``rt/lowcmd`` frame whose header/level_flag aren't
        the low-level markers — set them on every command (see the unitree_sdk2 go2
        low-level example). Unused motor slots (12..19) stay at their zeroed default
        (mode 0 = disabled), so only the 12 leg motors are driven.
        """
        cmd = self._LowCmd()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        return cmd

    # ── RobotIO surface ─────────────────────────────────────────────────────
    def read_state(self) -> RobotState:
        with self._lock:
            s = self._state
        if s is None:
            raise RuntimeError("connect() first")
        q = {name: float(s.motor_state[i].q) for name, i in _NAME_TO_SDK_INDEX.items()}
        dq = {name: float(s.motor_state[i].dq) for name, i in _NAME_TO_SDK_INDEX.items()}
        imu = s.imu_state
        quat = tuple(float(v) for v in imu.quaternion)   # (w, x, y, z)
        gyro = tuple(float(v) for v in imu.gyroscope)
        # rt/lowstate carries no base linear velocity; the estimator's is on
        # rt/sportmodestate. Report zeros — a velocity-walk obs uses gyro + gravity.
        return RobotState(q=q, dq=dq, quat_wxyz=quat, gyro=gyro, base_lin_vel=(0.0, 0.0, 0.0))

    def publish_targets(self, targets: dict, gains: dict, weight=None) -> None:
        """Command position targets (name→q) with per-joint (kp, kd) over rt/lowcmd.

        ``weight`` is ignored — the Go2 has no compliant overlay like the G1's
        ``rt/arm_sdk``; low-level here is full rigid authority over the legs.
        """
        if not self._enable_lowcmd or self._pub is None:
            raise RuntimeError(
                "rt/lowcmd is disabled — construct Go2RobotIO(enable_lowcmd=True) to command "
                "motors, and only after releasing sport mode + climbing the de-risk ladder "
                "(see SAFETY.md and arm_mhs_robotkit.policy_deploy).")
        cmd = self._new_lowcmd()
        for name, i in _NAME_TO_SDK_INDEX.items():
            m = cmd.motor_cmd[i]
            m.mode = MOTOR_MODE_SERVO
            m.q = float(targets[name])
            m.dq = 0.0
            m.tau = 0.0
            kp, kd = gains[name]
            m.kp = float(kp)
            m.kd = float(kd)
        cmd.crc = self._crc.Crc(cmd)
        self._pub.Publish(cmd)

    def damp_once(self) -> None:
        """One compliant command — kp=0, small kd on every joint (the safe stop).

        Rigid whole-body authority must damp FAST (never blend a leg release); this
        holds the current position with zero stiffness and light damping.
        """
        if not self._enable_lowcmd or self._pub is None:
            return  # nothing to damp if we never held the legs
        state = self.read_state()
        cmd = self._new_lowcmd()
        for name, i in _NAME_TO_SDK_INDEX.items():
            m = cmd.motor_cmd[i]
            m.mode = MOTOR_MODE_SERVO
            m.q = float(state.q[name])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 2.0
        cmd.crc = self._crc.Crc(cmd)
        self._pub.Publish(cmd)

    def abort_tripped(self) -> bool:
        return bool(self._abort_source and self._abort_source())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Go2 low-level state reader (READ-ONLY).")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    io = Go2RobotIO(iface=args.iface, enable_lowcmd=False)  # read-only by construction
    io.connect()
    try:
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            st = io.read_state()
            fr = st.q["FR_hip_joint"]
            print(f"  FR_hip={fr:+.3f} rad   quat_w={st.quat_wxyz[0]:+.3f}   "
                  f"gyro_z={st.gyro[2]:+.3f}")
            time.sleep(0.5)
    finally:
        io.shutdown()
