#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the arrival spin and the surrender shake.

Both gestures are state machines over measured heading, so all of this runs with no robot.
What earns a test is what hardware would otherwise have taught: that a full revolution
crosses the +/-pi discontinuity, that the two gestures are distinguishable rather than one
being a shorter version of the other, and that neither one travels.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from flourish import (
    LEG_TOLERANCE_RAD,
    SHAKE_COUNT,
    SHAKE_SWEEP_RAD,
    STALL_WINDOW_S,
    Flourish,
    Refusal,
    describe,
)
from reverse_along_path import wrap_pi

YAW_RAD_S = 0.8563


def _turn(gesture, *, hz=10.0, delivered_rad_s=YAW_RAD_S, start_yaw=0.0, max_ticks=4000):
    """Run the gesture against a robot that delivers exactly what it is commanded.

    Integrates the command back into a heading, which is the only honest way to test a
    controller that closes its own loop: feeding it the heading it wants to see would pass
    a controller that ignored heading entirely.
    """
    yaw, now, dt = start_yaw, 0.0, 1.0 / hz
    signs, swept = [], 0.0
    while max_ticks > 0:
        max_ticks -= 1
        wz = gesture.step(yaw, now)
        if wz is None:
            return yaw, now, signs, swept
        signs.append(1 if wz > 0 else -1)
        step = math.copysign(delivered_rad_s, wz) * dt
        swept += abs(step)
        yaw = wrap_pi(yaw + step)
        now += dt
    raise AssertionError("the gesture never finished")


# ── the spin ────────────────────────────────────────────────────────────────
def test_the_spin_turns_a_full_circle():
    gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    _, _, signs, swept = _turn(gesture)
    assert abs(swept - 2 * math.pi) < LEG_TOLERANCE_RAD + 0.1, swept
    assert set(signs) == {1}, "a spin is one direction throughout"


def test_the_spin_survives_the_pi_discontinuity():
    """A full revolution passes through the wrap TWICE from some start headings. Comparing
    against the start reads as no rotation there, and the robot spins to its timeout."""
    for start in (0.0, math.pi / 2, math.pi - 0.05, -math.pi + 0.05, 2.5, -2.5):
        gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
        _, _, _, swept = _turn(gesture, start_yaw=start)
        assert abs(swept - 2 * math.pi) < LEG_TOLERANCE_RAD + 0.1, (start, swept)


def test_the_spin_ends_where_it_started():
    """It is fired at the end of a run nobody is steering any more, so it must not leave
    the robot on a new heading for whoever picks it up next."""
    gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    yaw, _, _, _ = _turn(gesture, start_yaw=0.7)
    assert abs(wrap_pi(yaw - 0.7)) < 0.25, yaw


def test_either_direction_spins_the_same_amount():
    """Which way it turns is a CLEARANCE choice, not a distance one."""
    swept = []
    for sign in (+1, -1):
        gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S, turn_sign=sign)
        _, _, signs, s = _turn(gesture)
        swept.append(s)
        assert set(signs) == {sign}
    assert abs(swept[0] - swept[1]) < 0.05, swept


# ── the shake ───────────────────────────────────────────────────────────────
def test_the_shake_alternates_and_is_not_just_a_short_spin():
    """The two gestures have to be told apart at a glance, and from behind. A shorter spin
    would not be: it is the same motion, and an operator cannot judge 'less than a full
    turn' on a robot they are looking at from one side."""
    gesture = Flourish(Flourish.SHAKE, yaw_speed_rad_s=YAW_RAD_S)
    _, _, signs, _ = _turn(gesture)
    assert set(signs) == {1, -1}, "a shake must go both ways"
    changes = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    assert changes == SHAKE_COUNT * 2 - 1, changes


def test_the_shake_ends_on_the_heading_it_began_on():
    """The first and last half-sweeps are half width for exactly this reason; without them
    the gesture drifts a full sweep to one side and leaves the robot facing wrong."""
    gesture = Flourish(Flourish.SHAKE, yaw_speed_rad_s=YAW_RAD_S)
    yaw, _, _, _ = _turn(gesture, start_yaw=-1.2)
    assert abs(wrap_pi(yaw + 1.2)) < SHAKE_SWEEP_RAD, yaw


def test_the_shake_is_much_shorter_than_the_spin():
    """It fires on surrender, when an operator wants the robot to stop being busy."""
    spin = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    shake = Flourish(Flourish.SHAKE, yaw_speed_rad_s=YAW_RAD_S)
    _, spin_s, _, _ = _turn(spin)
    _, shake_s, _, _ = _turn(shake)
    assert shake_s < spin_s, (shake_s, spin_s)


# ── neither gesture travels ─────────────────────────────────────────────────
def test_neither_gesture_ever_commands_a_linear_velocity():
    """`step` returns a YAW command and nothing else, which is the property that makes
    these safe to fire automatically at the end of a run nobody is steering."""
    for kind in (Flourish.SPIN, Flourish.SHAKE):
        gesture = Flourish(kind, yaw_speed_rad_s=YAW_RAD_S)
        yaw, now = 0.0, 0.0
        while True:
            wz = gesture.step(yaw, now)
            if wz is None:
                break
            assert isinstance(wz, float), wz   # one scalar: yaw. No vx, no vy, ever.
            yaw = wrap_pi(yaw + math.copysign(YAW_RAD_S, wz) * 0.1)
            now += 0.1


# ── the abandonments ────────────────────────────────────────────────────────
def test_a_robot_that_is_not_turning_is_abandoned_not_commanded():
    gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    now = 0.0
    try:
        while now < STALL_WINDOW_S * 3:
            gesture.step(0.0, now)      # heading never changes
            now += 0.1
    except Refusal as refusal:
        assert "stopped moving" in str(refusal), refusal
    else:
        raise AssertionError("a stalled gesture must be abandoned")


def test_a_gesture_that_outruns_its_measured_budget_is_abandoned():
    gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    try:
        _turn(gesture, delivered_rad_s=YAW_RAD_S * 0.1, max_ticks=8000)
    except Refusal as refusal:
        assert "outran its budget" in str(refusal), refusal
    else:
        raise AssertionError("a gesture that cannot finish must be abandoned")


def test_it_refuses_to_time_itself_against_an_unmeasured_yaw_speed():
    """An unmeasured speed is an absence, not a zero -- the gait floor's own rule."""
    try:
        Flourish(Flourish.SPIN, yaw_speed_rad_s=0.0)
    except Refusal as refusal:
        assert "axis_primitive_probe" in str(refusal), refusal
    else:
        raise AssertionError("an unmeasured yaw speed must be refused")


def test_an_unknown_gesture_is_refused_rather_than_guessed():
    try:
        Flourish("dance", yaw_speed_rad_s=YAW_RAD_S)
    except Refusal as refusal:
        assert "unknown gesture" in str(refusal), refusal
    else:
        raise AssertionError("an unknown gesture must be refused")


def test_the_plan_says_the_robot_does_not_travel():
    """The property that licenses firing this automatically. If the sentence goes, the
    reason it was safe went with it."""
    for kind in (Flourish.SPIN, Flourish.SHAKE):
        text = describe(kind, YAW_RAD_S)
        assert "travel" in text and "does not move" in text, text
        assert "0.8563" in text, "the plan must quote the MEASURED yaw speed"


def test_the_pose_fields_these_runners_read_actually_exist():
    """⚠️ THE BUG THIS EXISTS FOR. `flourish.perform` and `reverse_along_path.walk` read
    `loco.pose()`, and both were written against the SIBLING UPSTREAM repository, whose
    `Lite3Pose` names the heading `yaw_rad`. This repository names it `yaw`. Nothing
    offline touched a real pose, so it passed every test here and raised
    `AttributeError: 'Lite3Pose' object has no attribute 'yaw_rad'` on a robot that had
    already completed its run.

    The state machines are pure and cannot catch this; only the seam can. So the seam is
    asserted directly: whatever `Lite3Pose` carries, these two files must read that name.
    """
    import ast
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from deep_robotics.lite3.locomotion.lite3_udp_locomotion import Lite3Pose

    fields = set(getattr(Lite3Pose, "__dataclass_fields__", {}))
    assert fields, "Lite3Pose is expected to be a dataclass"

    here = _Path(__file__).resolve().parent
    for name in ("flourish.py", "reverse_along_path.py"):
        tree = ast.parse((here / name).read_text(), filename=name)
        read = {node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id == "pose"}
        assert read, f"{name} no longer reads a pose; delete this test or fix the scan"
        unknown = read - fields
        assert not unknown, (
            f"{name} reads {sorted(unknown)} off a pose, and Lite3Pose has "
            f"{sorted(fields)}. This is the upstream/downstream naming split.")


# ── look: the one gesture that does NOT come home ───────────────────────────
def test_a_look_turns_the_asked_for_angle_and_stays_there():
    """The whole difference from `sweep`. This runs between attempts of a run that ended
    "goal never sighted", so returning to the start heading would leave the next attempt
    facing exactly the direction that saw nothing."""
    for degrees in (90.0, -90.0, 180.0):
        gesture = Flourish(Flourish.LOOK, yaw_speed_rad_s=YAW_RAD_S,
                           look_rad=math.radians(degrees))
        yaw, _, signs, swept = _turn(gesture)
        assert abs(swept - abs(math.radians(degrees))) < LEG_TOLERANCE_RAD + 0.1, swept
        assert len(set(signs)) == 1, "a look is one continuous turn"
        assert abs(wrap_pi(yaw - wrap_pi(math.radians(degrees)))) < 0.25, (degrees, yaw)


def test_the_sign_of_the_angle_picks_the_direction():
    left = Flourish(Flourish.LOOK, yaw_speed_rad_s=YAW_RAD_S, look_rad=math.radians(90))
    right = Flourish(Flourish.LOOK, yaw_speed_rad_s=YAW_RAD_S, look_rad=math.radians(-90))
    assert left.step(0.0, 0.0) > 0
    assert right.step(0.0, 0.0) < 0


def test_a_look_of_zero_is_refused_rather_than_being_a_silent_no_op():
    """It would turn nowhere, see nothing new, and cost an attempt to discover that."""
    try:
        Flourish(Flourish.LOOK, yaw_speed_rad_s=YAW_RAD_S, look_rad=0.0)
    except Refusal as refusal:
        assert "sees nothing new" in str(refusal), refusal
    else:
        raise AssertionError("a zero-degree look must be refused")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"flourish: {len(tests)}/{len(tests)} passed")
