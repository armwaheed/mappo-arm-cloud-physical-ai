#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""d1_fk — pure-numpy forward kinematics, Jacobian and IK for the Unitree D1 arm.

Mirrors ``unitree/g1/arm_fk/arm_fk.py``: no ROS, no pinocchio, no mesh loaders — just
numpy and ``xml.etree``. The bundled ``urdf/d1_description.urdf`` carries joint origins
and axes only (see its header for why the meshes are absent).

Why this module exists
----------------------
Before it, the D1 was driven by hand-guessing joint angles. That is not possible on a
6-DOF serial chain: `arm/robotics-connect#30` spent two sessions convinced the joint→world
frame mapping was unknown, when in fact the commanded pose ``[0,-40,71,0,60,0]`` simply
curls the arm to 0.168 m from its own base — it was never going to reach anything. One FK
call says so in a millisecond. :func:`_selftest` pins that as a regression.

⚠ COMMANDABLE is not MECHANICAL
-------------------------------
The URDF's ``<limit>`` values are mechanical travel. The D1 **firmware clamps every
commanded angle** to Unitree's documented envelope (:data:`COMMANDABLE_LIMITS_DEG`).
Measured on hardware 2026-07-09: commanding the shoulder to -95° parks it at -90.3°.
An arm can be *pushed by hand* to 96.7° and hold there, and its encoder will report it —
but no command will ever drive it back. **Always plan IK against COMMANDABLE_LIMITS_DEG.**

Conventions
-----------
* ``q`` is 6 joint angles in **DEGREES**, wire order ``angle0..angle5`` (J0 base-yaw …
  J5 wrist-roll). ``angle6`` is the jaw and is not part of the kinematic chain.
* ``base_link`` is the mounting face on the Go2's trunk: **+x toward the head, +z up**.
  The mount is a pure translation (verified: a hand-taught grasp at base-yaw ≈ 2°
  reaches straight over the robot's nose).
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(_HERE, "urdf", "d1_description.urdf")

NUM_JOINTS = 6

#: Unitree's documented per-joint envelope, in degrees. The FIRMWARE ENFORCES THIS on
#: every command (support.unitree.com/home/en/developer/D1Arm_services). Plan against it.
COMMANDABLE_LIMITS_DEG = np.array([135.0, 90.0, 90.0, 135.0, 90.0, 135.0])

#: Hardware-measured constants, unit 192.168.123.18, 2026-07-09. Used by the selftest.
TAUGHT_GRASP_A = [1.87, 96.75, -43.56, -3.40, -36.40, -6.47]
TAUGHT_GRASP_B = [2.36, 96.74, -36.22, -2.90, -47.90, -3.40]
ISSUE30_FAIL_POSE = [0.0, -40.0, 71.0, 0.0, 60.0, 0.0]


def _vec(s: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.array([float(v) for v in (s or default).split()])


def _rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp,     cp * sr,                cp * cr]])


def _axis_angle_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class D1Kinematics:
    """FK / Jacobian / IK for the D1, from the bundled URDF."""

    def __init__(self, urdf_path: str = URDF_PATH) -> None:
        root = ET.parse(urdf_path).getroot()
        self._joints = []
        for j in root.findall("joint"):
            o, a = j.find("origin"), j.find("axis")
            self._joints.append((
                j.get("type"),
                _vec(o.get("xyz") if o is not None else None),
                _vec(o.get("rpy") if o is not None else None),
                _vec(a.get("xyz") if a is not None else None, "0 0 1"),
            ))
        fingers = [j for j in self._joints if j[0] == "prismatic"]
        # Jaw reference = midpoint of the two finger joint origins, in the last link's frame.
        self._jaw_local = (fingers[0][1] + fingers[1][1]) / 2.0

    # ── forward kinematics ────────────────────────────────────────────────────
    def _chain(self, q_deg, upto: int = NUM_JOINTS) -> np.ndarray:
        T = np.eye(4)
        for i in range(upto):
            _typ, xyz, rpy, axis = self._joints[i]
            R = _rpy_to_matrix(*rpy) @ _axis_angle_matrix(axis, math.radians(q_deg[i]))
            A = np.eye(4)
            A[:3, :3], A[:3, 3] = R, xyz
            T = T @ A
        return T

    def jaw_pose(self, q_deg) -> np.ndarray:
        """4x4 pose of the jaw reference point in ``base_link``."""
        J = np.eye(4)
        J[:3, 3] = self._jaw_local
        return self._chain(q_deg) @ J

    def jaw_xyz(self, q_deg) -> np.ndarray:
        return self.jaw_pose(q_deg)[:3, 3]

    def link_origins(self, q_deg) -> dict:
        """Origins of intermediate links — collision proxies for path checking."""
        # NB: no dict `|` union — the Go2's Jetson runs Python 3.8.
        out = {name: self._chain(q_deg, n)[:3, 3]
               for name, n in (("upper", 3), ("elbow", 4), ("wrist", 6))}
        out["jaw"] = self.jaw_xyz(q_deg)
        return out

    def shoulder_origin(self) -> np.ndarray:
        """Shoulder-pitch axis in ``base_link``. This is the datum Unitree's published
        "arm length / working radius" figures are measured from — NOT ``base_link``,
        which sits 0.11 m lower on the mounting face."""
        return self._chain([0.0] * NUM_JOINTS, 2)[:3, 3]

    def max_reach(self, step_deg: float = 2.0, tip: str = "jaw",
                  from_shoulder: bool = False) -> float:
        """Furthest ``tip`` gets from ``base_link`` (or the shoulder axis), sweeping
        shoulder × elbow. ``tip`` is ``"jaw"`` or ``"wrist"``.

        Reference values (this model): jaw 0.733 m and wrist 0.662 m from ``base_link``;
        jaw 0.624 m and **wrist 0.553 m** from the shoulder axis. That last figure is the
        one to compare against Unitree's published 550 mm arm length / working radius.
        """
        origin = self.shoulder_origin() if from_shoulder else np.zeros(3)
        fn = self.jaw_xyz if tip == "jaw" else (lambda q: self._chain(q, 6)[:3, 3])
        best = 0.0
        rng = np.arange(-90.0, 90.0 + step_deg, step_deg)
        for s in rng:
            for e in rng:
                best = max(best, float(np.linalg.norm(fn([0, s, e, 0, 0, 0]) - origin)))
        return best

    # ── Jacobian + IK ─────────────────────────────────────────────────────────
    def position_jacobian(self, q_deg, h_deg: float = 1e-3) -> np.ndarray:
        """3x6 d(jaw_xyz)/d(q), metres per RADIAN."""
        J = np.zeros((3, NUM_JOINTS))
        p0 = self.jaw_xyz(q_deg)
        for i in range(NUM_JOINTS):
            qp = list(q_deg)
            qp[i] += h_deg
            J[:, i] = (self.jaw_xyz(qp) - p0) / math.radians(h_deg)
        return J

    def ik_position(self, target_xyz, q0, iters: int = 300, lam: float = 0.05,
                    gain: float = 0.4, limits=None):
        """Damped-least-squares IK on jaw POSITION, clamped to ``limits``.

        Returns ``(q_deg, position_error_m)``. Defaults to the COMMANDABLE envelope —
        solving against the mechanical limits will hand you poses the firmware refuses.
        """
        limits = COMMANDABLE_LIMITS_DEG if limits is None else np.asarray(limits, float)
        q = np.array(q0, dtype=float)
        target = np.asarray(target_xyz, dtype=float)
        for _ in range(iters):
            e = target - self.jaw_xyz(q)
            if np.linalg.norm(e) < 1e-4:
                break
            J = self.position_jacobian(q)
            dq = np.linalg.solve(J.T @ J + lam ** 2 * np.eye(NUM_JOINTS), J.T @ e)
            q = np.clip(q + np.degrees(dq) * gain, -limits, limits)
        return q, float(np.linalg.norm(target - self.jaw_xyz(q)))


def _selftest() -> int:
    """Assert the physical invariants this model was verified against — not self-consistency.

    The G1's arm_fk selftest is the template: check facts about the world, so a units or
    sign regression cannot pass.
    """
    k = D1Kinematics()
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            fails.append(name)

    # Unitree quotes reach from the SHOULDER AXIS, not the mounting face. Measured that
    # way the model lands on their published 550 mm to within 3 mm — the strongest single
    # check that these joint origins describe the real arm.
    r_wrist_sh = k.max_reach(tip="wrist", from_shoulder=True)
    check("wrist reach from the shoulder axis == Unitree's published 550 mm arm length",
          abs(r_wrist_sh - 0.550) < 0.01, f"got {r_wrist_sh:.3f} m")

    r_jaw = k.max_reach(tip="jaw")
    check("jaw reach from base_link is 0.733 m",
          abs(r_jaw - 0.733) < 0.01, f"got {r_jaw:.3f} m")

    r_fail = float(np.linalg.norm(k.jaw_xyz(ISSUE30_FAIL_POSE)))
    check("issue #30's fail pose curls the arm to <0.25 m (it was never mis-aimed)",
          r_fail < 0.25, f"jaw {r_fail:.3f} m from base")

    r_curl = float(np.linalg.norm(k.jaw_xyz([0, 0, 71, 0, 0, 0])))
    r_ext = float(np.linalg.norm(k.jaw_xyz([0, 0, -71, 0, 0, 0])))
    check("elbow +71 curls, -71 extends (the inverted sign that caused #30)",
          r_curl < 0.30 < r_ext, f"curl {r_curl:.3f} m, extend {r_ext:.3f} m")

    a, b = k.jaw_xyz(TAUGHT_GRASP_A), k.jaw_xyz(TAUGHT_GRASP_B)
    d = float(np.linalg.norm(a - b)) * 1000
    check("two independent hand-taught grasps agree to <25 mm through this model",
          d < 25.0, f"{d:.1f} mm apart")

    q, err = k.ik_position(b, [2.36, 80, -30, -3, -50, -3])
    saturated = abs(abs(q[1]) - 90.0) < 0.5
    check("the taught grasp point saturates the COMMANDABLE shoulder limit",
          saturated, f"shoulder {q[1]:+.2f} deg, residual {err*1000:.1f} mm")

    near = b.copy()
    near[0] -= 0.06                       # bin 60 mm closer
    q2, err2 = k.ik_position(near, [2.36, 80, -30, -3, -50, -3])
    check("moving the target 60 mm closer yields a commandable solution with margin",
          err2 < 1e-3 and abs(q2[1]) < 89.0, f"shoulder {q2[1]:+.2f} deg, err {err2*1000:.2f} mm")

    print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
