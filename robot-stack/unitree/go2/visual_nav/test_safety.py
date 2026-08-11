#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the walk/no-walk guards — no robot, no DDS, no arm.

``safety.py`` is the module that decides whether an arm-loaded Go2 may walk, and it had
no tests at all. Everything it gates on is arithmetic over joint angles and a lowstate
sample, so all of it is testable off-robot; only the DDS subscription and the funcode-5
enable are not, and those are stubbed out here.

The sway gate is the one that matters most. It is ABSOLUTE — measured from the joint
zero, not from wherever the operator posed the arm — so creep accumulates across runs
and the check has to fail an arm that passed an hour ago.

Run: ``python3 test_safety.py``
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety import (
    BATTERY_SOC_ABORT_PCT,
    MOTOR_TEMP_ABORT_C,
    MOTOR_TEMP_WARN_C,
    STOWED_LATERAL_M,
    STOWED_REACH_M,
    STOWED_YAW_DEG,
    ArmStowMonitor,
    Health,
    HealthMonitor,
    LatchResult,
)

#: The measured dorsal stow on this unit, as recorded in safety.py.
STOWED = [1.4, -90.5, 88.0, 1.3, 20.0, -0.5]


class _FakeKinematics:
    """Jaw position from base yaw alone, matching the real arm's geometry closely enough.

    The live FK puts the jaw 77.6 mm forward and 119.2 mm up at the stow pose, with a
    -1.93 mm lateral offset at J0 = 0 that comes from the wrist. Reproducing that offset
    matters: it is exactly why the sway gate reads J0 rather than jaw y.
    """

    LEVER_M = 0.0776
    WRIST_OFFSET_M = -0.00193

    def jaw_xyz(self, angles):
        yaw = math.radians(angles[0])
        return (self.LEVER_M * math.cos(yaw),
                self.LEVER_M * math.sin(yaw) + self.WRIST_OFFSET_M,
                0.1192)


def _monitor(angles) -> ArmStowMonitor:
    """An ArmStowMonitor wired to fake kinematics and a fixed pose, without DDS."""
    monitor = ArmStowMonitor()
    monitor._kinematics = _FakeKinematics()
    monitor._angles = None if angles is None else list(angles)
    return monitor


def _health_monitor(temp_c: float, soc_pct: float = 80.0) -> HealthMonitor:
    monitor = HealthMonitor()
    monitor._health = Health(max_motor_temp_c=temp_c, hottest_motor=6,
                             battery_soc_pct=soc_pct, sample_time=_now())
    return monitor


def _now() -> float:
    import time
    return time.monotonic()


# ── Sway ────────────────────────────────────────────────────────────────────
def test_sway_is_the_base_yaw_magnitude():
    assert _monitor(STOWED).sway_deg() == 1.4
    assert _monitor([-4.0, *STOWED[1:]]).sway_deg() == 4.0, "sway is unsigned"
    assert _monitor(None).sway_deg() is None


def test_the_reference_stow_passes_the_sway_gate():
    """The pose safety.py documents as correct must not be refused by its own check."""
    assert _monitor(STOWED).blocking_reason() is None


def test_sway_beyond_the_limit_blocks_the_walk():
    reason = _monitor([STOWED_YAW_DEG + 0.5, *STOWED[1:]]).blocking_reason()
    assert reason is not None and "swayed" in reason, reason


def test_the_sway_gate_is_absolute_not_a_creep_budget():
    """Creep accumulates across runs, so a passing arm can fail later without moving far.

    This is the operational consequence of an absolute limit and the reason the
    pre-flight prints the sway against its budget: measured on this robot, the arm sat
    at 1.5 deg before two walking runs and 2.5 deg after them.
    """
    assert _monitor([2.5, *STOWED[1:]]).blocking_reason() is None, "2.5 deg still passes"
    assert _monitor([3.5, *STOWED[1:]]).blocking_reason() is not None, "3.5 deg must not"


def test_a_turning_creep_would_be_refused():
    """5.3 deg is the recorded latched creep across one TURNING test — outside 3 deg.

    Pins the fact that the gate is tight enough to catch turning creep, which is the
    behaviour the limit was tightened for.
    """
    assert _monitor([5.3, *STOWED[1:]]).blocking_reason() is not None


def test_sway_catches_what_the_lateral_check_cannot():
    """The reason the gate moved from jaw-y to base yaw.

    A folded arm is compact, so its jaw barely moves when the base rotates: at 40 deg of
    base yaw — a 3.15 kg mass swung right out over the flank — the jaw is still inside
    STOWED_LATERAL_M. The lateral check alone would pass it.
    """
    monitor = _monitor([40.0, *STOWED[1:]])
    jaw = monitor.jaw_xyz()
    assert abs(jaw[1]) < STOWED_LATERAL_M, (
        f"40 deg of base yaw stays inside the lateral limit: {jaw[1]:.4f} m")
    assert monitor.reach_m() < STOWED_REACH_M, "and inside the reach limit too"
    reason = monitor.blocking_reason()
    assert reason is not None and "swayed" in reason, (
        f"only the sway gate can see this: {reason!r}")


def test_y_is_zero_at_a_nonzero_base_yaw():
    """Documents the trap the sway gate exists to avoid.

    Jaw y is not zero when the base yaw is zero — the wrist sits off-axis — so 'y = 0'
    and 'J0 = 0' describe different poses. A gate on y would silently mean a gate on a
    blend of three joints.
    """
    at_zero_yaw = _monitor([0.0, *STOWED[1:]]).jaw_xyz()
    assert abs(at_zero_yaw[1]) > 1e-4, (
        f"y should be OFF axis at J0=0, got {at_zero_yaw[1]}")


# ── Reach and arm presence ──────────────────────────────────────────────────
def test_a_silent_arm_blocks_unless_it_is_declared_absent():
    monitor = _monitor(None)
    assert monitor.blocking_reason(required=True) is not None
    assert monitor.blocking_reason(required=False) is None


# ── Health ──────────────────────────────────────────────────────────────────
def test_the_warning_threshold_is_actually_evaluated():
    """MOTOR_TEMP_WARN_C was defined and never read, leaving the pre-flight's
    'check motors < 55 C' to the operator's memory of a number the code already had."""
    assert _health_monitor(MOTOR_TEMP_WARN_C - 5.0).warning_reason() is None
    warning = _health_monitor(MOTOR_TEMP_WARN_C + 1.0).warning_reason()
    assert warning is not None and "warning mark" in warning, warning


def test_warning_fires_below_abort():
    """The warning must be reachable while the run is still allowed to continue."""
    monitor = _health_monitor(MOTOR_TEMP_WARN_C + 1.0)
    assert monitor.warning_reason() is not None
    assert monitor.abort_reason() is None, "warning must not itself abort"
    assert MOTOR_TEMP_WARN_C < MOTOR_TEMP_ABORT_C


def test_abort_on_temperature_and_battery():
    assert _health_monitor(MOTOR_TEMP_ABORT_C + 1.0).abort_reason() is not None
    assert _health_monitor(30.0, BATTERY_SOC_ABORT_PCT - 1.0).abort_reason() is not None
    assert _health_monitor(30.0, 80.0).abort_reason() is None
    assert HealthMonitor().abort_reason() == "no rt/lowstate"


# ── The D1 dependency contract ──────────────────────────────────────────────
# safety.py imports three things out of ../d1_arm. None of it was covered, and all of it
# was broken on main: d1_fk.py and _arm_idl.py were not in the repo AT ALL (they existed
# only on the robot's deployed tree, which is not a git checkout), and d1_arm.py had no
# `enable()` because a hardware correction renaming it was never synced back. So a fresh
# clone could not run ArmStowMonitor.start(), and the latch — a hard safety requirement —
# would have died with AttributeError at the moment it was asked to hold 3.15 kg.
#
# These tests are cheap and would have caught all of it.
def test_the_forward_kinematics_module_exists_and_is_importable():
    from d1_arm.d1_fk import D1Kinematics

    kinematics = D1Kinematics()
    jaw = kinematics.jaw_xyz(STOWED)
    assert len(jaw) >= 3


def test_the_kinematics_reproduce_the_measured_stow_geometry():
    """Guards the bundled URDF as well as the code — FK is only as good as its model.

    Measured on this unit at the dorsal stow: jaw 77.6 mm forward, 119.2 mm up. The
    lateral term is the interesting one: at J0 = 0 the jaw sits 1.93 mm OFF axis, which
    is exactly why STOWED_YAW_DEG gates base yaw instead of jaw y.
    """
    from d1_arm.d1_fk import D1Kinematics

    kinematics = D1Kinematics()
    x, y, z = (float(v) for v in kinematics.jaw_xyz([0.0, -90.5, 89.7, -1.0, 14.9,
                                                     -0.5])[:3])
    assert abs(x - 0.07758) < 5e-4, x
    assert abs(y - -0.00193) < 5e-4, y
    assert abs(z - 0.11922) < 5e-4, z


def test_the_arm_client_still_has_every_method_safety_calls():
    """The exact break: latch_arm() calls connect() and enable(), and a rename to
    hold()/relax() left main calling a method that no longer existed."""
    from d1_arm.d1_arm import D1Arm, parse_feedback

    for name in ("connect", "enable", "shutdown"):
        assert hasattr(D1Arm, name), f"safety.latch_arm calls D1Arm.{name}()"
    assert callable(parse_feedback)


def test_the_vendored_idl_module_is_present():
    """_arm_idl.py needs CycloneDDS to import, so only its presence is asserted here —
    but its absence from the repo is precisely what broke the fresh checkout."""
    from pathlib import Path

    idl = Path(__file__).resolve().parents[1] / "d1_arm" / "_arm_idl.py"
    assert idl.is_file(), f"{idl} is missing; ArmStowMonitor.start() imports it"


# ── Latch reporting ─────────────────────────────────────────────────────────
def test_a_silent_feed_is_not_a_successful_latch():
    """No angles is no drift evidence, so it must not read as a successful latch."""
    result = LatchResult(held=False, drift_deg=float("nan"), angles_deg=tuple(STOWED),
                         jaw_xyz_m=(0.074, 0.002, 0.116), reach_m=0.138,
                         powered=True, reason="the joint feed went silent")
    assert result.held is False
    assert "NOT HELD" in str(result)


def test_the_latch_result_carries_the_power_flag():
    """Zero drift is not evidence of a latch, so the operator line must show power too.

    Measured on this robot: an unpowered D1 reported 'drift 0.00 deg — HELD' on every
    run, including both live walking runs, because an arm that cannot move is perfectly
    still. The rendered line now states the power flag alongside the drift so the two
    cannot be confused again.
    """
    unpowered = LatchResult(held=False, drift_deg=0.0, angles_deg=tuple(STOWED),
                            jaw_xyz_m=(0.074, 0.002, 0.116), reach_m=0.138,
                            powered=False,
                            reason="its servo bus is NOT powered (power=0 ...)")
    assert unpowered.held is False, "zero drift must NOT imply held when unpowered"
    rendered = str(unpowered)
    assert "power=False" in rendered and "NOT HELD" in rendered, rendered

    powered = LatchResult(held=True, drift_deg=0.1, angles_deg=tuple(STOWED),
                          jaw_xyz_m=(0.074, 0.002, 0.116), reach_m=0.138, powered=True)
    assert "HELD" in str(powered) and "NOT HELD" not in str(powered)


def test_powered_reads_the_arms_own_status_word():
    """power_status arrives on the SAME topic as the angles, tagged (2, 3).

    It used to be discarded by the feedback filter, which is exactly how an unpowered
    arm passed the latch: the evidence was being published all along and thrown away.
    """
    monitor = _monitor(STOWED)
    assert monitor.powered() is None, "nothing said yet -> undetermined, not False"

    monitor._power_status, monitor._enable_status, monitor._error_status = 0, 0, 0
    assert monitor.powered() is False
    assert monitor.energised() is False
    assert "power=0" in monitor.status_word()

    monitor._power_status, monitor._enable_status = 1, 1
    assert monitor.powered() is True
    assert monitor.energised() is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"safety: {len(tests)}/{len(tests)} passed")
