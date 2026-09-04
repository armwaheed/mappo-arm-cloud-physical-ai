#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the turn-instead-of-reverse retreat.

The whole manoeuvre is a state machine over measured pose, so all of it runs here with no
robot, no sockets and no legs. What earns a test is the part that would be discovered on
hardware otherwise: that a half turn lands on the +/-pi discontinuity, that the distance is
closed on ODOMETRY and not on a stopwatch, and that a robot which is not moving is
abandoned rather than commanded.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from reverse_along_path import (
    COMMAND_VX,
    DEFAULT_RETREAT_M,
    DRIVE_TOLERANCE_M,
    HALF_TURN_RAD,
    STALL_WINDOW_S,
    PathHistory,
    Refusal,
    ReverseAlongPath,
    describe,
    wrap_pi,
)

#: The two speeds this robot actually measured, so the tests are timed against the same
#: numbers the manoeuvre is.
FORWARD_M_S = 0.5362
YAW_RAD_S = 0.8563


def _manoeuvre(target_m=DEFAULT_RETREAT_M, **kwargs):
    return ReverseAlongPath(target_m, forward_speed_m_s=FORWARD_M_S,
                            yaw_speed_rad_s=YAW_RAD_S, **kwargs)


def _walk(plan, *, hz=10.0, delivered_m_s=FORWARD_M_S, delivered_rad_s=YAW_RAD_S,
          max_ticks=2000, start=(0.0, 0.0, 0.0)):
    """Run the manoeuvre against a robot that delivers exactly what it is commanded.

    Integrates the commands the plan returns back into a pose, which is the only honest
    way to test a controller that closes its own loop: a test that fed it the pose it
    wanted to see would pass on a controller that ignored the pose entirely.
    """
    x, y, yaw = start
    now = 0.0
    dt = 1.0 / hz
    phases = []
    while max_ticks > 0:
        max_ticks -= 1
        command = plan.step(x, y, yaw, now)
        if command is None:
            return x, y, yaw, now, phases
        if not phases or phases[-1] != command.phase:
            phases.append(command.phase)
        if command.vx:
            x += math.copysign(delivered_m_s, command.vx) * dt * math.cos(yaw)
            y += math.copysign(delivered_m_s, command.vx) * dt * math.sin(yaw)
        if command.wz:
            yaw = wrap_pi(yaw + math.copysign(delivered_rad_s, command.wz) * dt)
        now += dt
    raise AssertionError("the manoeuvre never finished")


# ── the path that licenses the retreat ──────────────────────────────────────
def test_the_history_measures_floor_walked_not_displacement():
    """A metre out and half a metre back is 1.5 m of floor, all of it looked at.

    Displacement would call it 0.5 m and under-serve a legitimate retreat. It is the floor
    the camera has already passed over that licenses backing across it.
    """
    history = PathHistory()
    for step in range(21):
        history.append(step * 0.05, 0.0)
    for step in range(10):
        history.append(1.0 - step * 0.05, 0.0)
    assert abs(history.traversed_m - 1.45) < 1e-6, history.traversed_m
    assert abs(math.hypot(*history.points[-1]) - 0.55) < 1e-6


def test_the_retreat_is_clipped_to_the_floor_already_walked():
    """0.66 m of retreat is not available to a robot that has walked 0.2 m."""
    history = PathHistory()
    for step in range(5):
        history.append(step * 0.05, 0.0)
    assert abs(history.available_m(DEFAULT_RETREAT_M) - 0.2) < 1e-6
    # ...and a robot that has walked far enough gets exactly what it asked for, not more.
    for step in range(5, 40):
        history.append(step * 0.05, 0.0)
    assert history.available_m(DEFAULT_RETREAT_M) == DEFAULT_RETREAT_M


def test_a_robot_that_has_not_moved_has_nothing_to_retreat_over():
    history = PathHistory()
    history.append(0.0, 0.0)
    try:
        _manoeuvre(history.available_m(DEFAULT_RETREAT_M))
    except Refusal as refusal:
        assert "has none to retreat over" in str(refusal), refusal
    else:
        raise AssertionError("a zero-length retreat must be refused")


# ── the manoeuvre ───────────────────────────────────────────────────────────
def test_it_turns_drives_and_turns_back():
    plan = _manoeuvre()
    x, y, yaw, _elapsed, phases = _walk(plan)
    assert phases == [ReverseAlongPath.TURN_AWAY, ReverseAlongPath.DRIVE,
                      ReverseAlongPath.TURN_BACK], phases
    # It ends up BEHIND where it started, along the axis it was facing.
    assert x < 0, f"the robot should have retreated to negative x, got {x:.3f}"
    assert abs(abs(x) - DEFAULT_RETREAT_M) < 0.12, x
    assert abs(y) < 0.12, y
    # ...and facing the way it started, having turned through 360 in total.
    assert abs(wrap_pi(yaw)) < 0.2, yaw


def test_the_half_turn_survives_the_pi_discontinuity():
    """The bug this shape invites, and the reason heading is ACCUMULATED.

    A half turn lands exactly on the +/-pi wrap. Comparing current yaw against the start
    flips sign there and reads as no rotation at all, so the robot spins until its timeout.
    Starting at a yaw that puts the discontinuity mid-turn is the case that catches it.
    """
    for start_yaw in (0.0, math.pi / 2, -math.pi + 0.05, math.pi - 0.05, 3.0):
        plan = _manoeuvre()
        _, _, _, _, phases = _walk(plan, start=(0.0, 0.0, start_yaw))
        assert phases == [ReverseAlongPath.TURN_AWAY, ReverseAlongPath.DRIVE,
                          ReverseAlongPath.TURN_BACK], (start_yaw, phases)


def test_the_distance_is_closed_on_odometry_not_on_a_stopwatch():
    """The sign-only transport has no speed, so a timed drive cannot be right.

    A robot delivering 30% more than the profile claims must still stop at 0.66 m -- it
    just gets there in fewer ticks. A controller that multiplied a speed by a duration
    would overshoot by exactly that 30%.
    """
    travelled = {}
    for delivered in (FORWARD_M_S, FORWARD_M_S * 1.3, FORWARD_M_S * 0.75):
        plan = _manoeuvre()
        _walk(plan, delivered_m_s=delivered)
        travelled[delivered] = plan.driven_m
    for delivered, driven in travelled.items():
        assert abs(driven - DEFAULT_RETREAT_M) <= DRIVE_TOLERANCE_M + 0.06, \
            f"delivered {delivered:.3f} m/s drove {driven:.3f} m"
    spread = max(travelled.values()) - min(travelled.values())
    assert spread < 0.1, f"distance varied {spread:.3f} m with the delivered speed"


def test_it_can_be_left_facing_the_way_it_retreated():
    plan = _manoeuvre(turn_back=False)
    _, _, yaw, _, phases = _walk(plan)
    assert phases == [ReverseAlongPath.TURN_AWAY, ReverseAlongPath.DRIVE], phases
    assert abs(abs(wrap_pi(yaw)) - HALF_TURN_RAD) < 0.2, yaw


def test_either_turn_direction_reaches_the_same_place():
    """Which way it turns is a CLEARANCE choice. Both yaw primitives are measured within
    0.03% of each other, so it must not also be a distance choice."""
    ends = []
    for sign in (+1, -1):
        plan = _manoeuvre(turn_sign=sign)
        x, y, _, _, _ = _walk(plan)
        ends.append((x, y))
    assert abs(ends[0][0] - ends[1][0]) < 0.1, ends
    assert abs(ends[0][1] - ends[1][1]) < 0.15, ends


# ── the abandonments ────────────────────────────────────────────────────────
def test_a_robot_that_is_not_moving_is_abandoned_not_commanded():
    """Flat odometry is a stuck robot, a closed gait gate, or a robot being dragged. None
    of those get better by continuing to command them."""
    plan = _manoeuvre()
    now = 0.0
    try:
        while now < STALL_WINDOW_S * 3:
            plan.step(0.0, 0.0, 0.0, now)      # pose never changes
            now += 0.1
    except Refusal as refusal:
        assert "not moving" in str(refusal), refusal
    else:
        raise AssertionError("a stalled phase must be abandoned")


def test_a_phase_that_outruns_its_measured_budget_is_abandoned():
    """Delivering a tenth of what the profile claims is not a slow robot, it is a robot
    the profile is wrong about."""
    plan = _manoeuvre()
    try:
        _walk(plan, delivered_rad_s=YAW_RAD_S * 0.1, max_ticks=5000)
    except Refusal as refusal:
        assert "outran its budget" in str(refusal), refusal
    else:
        raise AssertionError("a phase that cannot finish must be abandoned")


def test_it_refuses_to_time_itself_against_an_unmeasured_speed():
    """An unmeasured speed is an absence, not a zero -- the same rule the gait floor has."""
    for forward, yaw in ((0.0, YAW_RAD_S), (FORWARD_M_S, 0.0)):
        try:
            ReverseAlongPath(DEFAULT_RETREAT_M, forward_speed_m_s=forward,
                             yaw_speed_rad_s=yaw)
        except Refusal as refusal:
            assert "axis_primitive_probe" in str(refusal), refusal
        else:
            raise AssertionError(f"forward={forward} yaw={yaw} should be refused")


def test_the_commands_are_signs_and_never_speeds():
    """On a sign-only transport a magnitude is a direction. These must sit clear of the
    profile's deadband and must never be mistaken for a measured speed."""
    plan = _manoeuvre()
    command = plan.step(0.0, 0.0, 0.0, 0.0)
    assert command.wz != 0 and command.vx == 0, command
    assert abs(command.wz) > 0.1, "must clear the 0.1 rad/s yaw deadband"
    assert COMMAND_VX > 0.05, "must clear the 0.05 m/s linear deadband"
    assert COMMAND_VX != FORWARD_M_S, "a command magnitude must not read as a real speed"


def test_the_plan_says_when_it_clipped_the_retreat():
    """An operator who asked for 0.66 m and is getting 0.20 m must be told, in the plan,
    before anything moves."""
    text = describe(0.2, DEFAULT_RETREAT_M, 0.2, FORWARD_M_S, YAW_RAD_S, True)
    assert "CLIPPED" in text, text
    assert "0.660" in text and "0.200" in text, text
    untouched = describe(DEFAULT_RETREAT_M, DEFAULT_RETREAT_M, 5.0,
                         FORWARD_M_S, YAW_RAD_S, True)
    assert "CLIPPED" not in untouched, untouched


def test_the_plan_states_that_nothing_in_it_is_a_guess():
    """The whole argument for this shape. If the sentence goes, the reason went with it."""
    text = describe(DEFAULT_RETREAT_M, DEFAULT_RETREAT_M, 5.0, FORWARD_M_S, YAW_RAD_S, True)
    assert "evidence string" in text and "guessed raw value" in text, text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"reverse_along_path: {len(tests)}/{len(tests)} passed")
