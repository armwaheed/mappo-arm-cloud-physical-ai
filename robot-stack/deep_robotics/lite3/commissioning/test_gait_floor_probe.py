#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 gait-floor probe.

Two of these tests are the reason the file exists, and both are carried from the Go2:

* a robot that never stood reports 0.000 m/s at every setting, which reads exactly like a
  floor that is real and total. The probe must refuse that run rather than tabulate it --
  and it must still report a genuine floor, where only the low rungs are silent, as a
  finding. Getting that pair right is the whole difficulty.
* the pure-strafe number and the diagonal number are different measurements of different
  things. The probe measures both and must never let one stand in for the other.

The rest are the guards: room sizing, ladder monotonicity, and the refusal to invent a
default for a value that has to be measured on this robot.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE))

from deep_robotics.lite3.commissioning.gait_floor_probe import (
    build_parser,
    descending_rungs,
    diagonal_plan,
    execute,
    forward_floor_from,
    forward_plan,
    lateral_rows,
    lateral_verdict,
    main,
    planned_forward_metres,
    planned_lateral_excursion_m,
    strafe_plan,
)
from deep_robotics.lite3.commissioning.measurement import Refusal, run_main

from test_measurement import FakeLocomotion

_CONTEXT = ["--robot-id", "LITE3-A", "--firmware", "V1.0.8", "--payload", "none"]

#: The modules in this directory that could hold a velocity default, and so the ones a
#: Go2 gait floor could be pasted into and be used.
VELOCITY_MODULES = ("gait_floor_probe.py", "actuator_gain_probe.py", "measurement.py",
                    "robot_link.py", "commission.py")


def _args(*extra):
    return build_parser().parse_args(_CONTEXT + list(extra))


def _quiet(callable_):
    """Run something chatty and keep its operator briefing out of the test output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return callable_()


def _run(loco, plan, segment=1.0):
    return execute(loco, plan, segment_s=segment, tick_s=0.1,
                   printer=lambda _line: None, clock=loco.clock, sleep=loco.sleep)


# ── the ladder ──────────────────────────────────────────────────────────────────────────
def test_rungs_descend_from_the_top_to_one_step():
    rungs = descending_rungs(0.6, 6)
    assert abs(rungs[0] - 0.6) < 1e-9
    assert abs(rungs[-1] - 0.1) < 1e-9
    assert rungs == sorted(rungs, reverse=True)


def test_a_ladder_too_short_to_show_monotonicity_is_refused():
    for count in (0, 1, 2):
        try:
            descending_rungs(0.5, count)
        except Refusal:
            continue
        raise AssertionError(f"{count} rungs cannot show a trend and must refuse")


def test_every_treatment_has_a_control_immediately_before_it():
    plan = forward_plan(0.5, 4)
    for index, (role, _vx, _vy) in enumerate(plan):
        if role == "treatment":
            assert plan[index - 1][0] == "control", index


def test_the_forward_plan_anchors_at_the_start_the_middle_and_the_end():
    plan = forward_plan(0.5, 4)
    anchors = [index for index, (role, _vx, _vy) in enumerate(plan) if role == "anchor"]
    assert len(anchors) == 3
    assert anchors[0] == 0 and anchors[-1] == len(plan) - 1
    assert 0 < anchors[1] < len(plan) - 1


def test_the_strafe_plan_alternates_sign_so_the_robot_returns_to_the_middle():
    plan = strafe_plan(0.3, 4, anchor_vx=0.5)
    lateral = [vy for role, _vx, vy in plan if role == "treatment"]
    assert [value > 0 for value in lateral] == [True, False, True, False]


def test_the_strafe_plan_still_carries_forward_anchors():
    """A lateral-only phase has nothing else that could catch a robot that never stood."""
    plan = strafe_plan(0.3, 4, anchor_vx=0.5)
    anchors = [(vx, vy) for role, vx, vy in plan if role == "anchor"]
    assert anchors and all(vx > 0 and vy == 0 for vx, vy in anchors)


def test_the_diagonal_plan_holds_forward_velocity_on_every_single_segment():
    """Dropping vx between treatments confounds 'lateral does not execute' with 'stopped'."""
    plan = diagonal_plan(0.28, 0.3, 4)
    assert all(abs(vx - 0.28) < 1e-9 for _role, vx, _vy in plan)


def test_the_diagonal_plan_uses_the_measured_floor_and_not_a_constant():
    assert diagonal_plan(0.11, 0.3, 3)[0][1] == 0.11
    assert diagonal_plan(0.44, 0.3, 3)[0][1] == 0.44


# ── room sizing ─────────────────────────────────────────────────────────────────────────
def test_planned_distance_is_an_over_estimate_of_the_lane_needed():
    plan = [("treatment", 0.5, 0.0), ("control", 0.0, 0.0)]
    assert abs(planned_forward_metres(plan, 2.0) - 1.0) < 1e-9


def test_lateral_excursion_is_the_worst_point_not_the_end_point():
    plan = [("treatment", 0.0, 0.4), ("treatment", 0.0, -0.4)]
    assert abs(planned_lateral_excursion_m(plan, 1.0) - 0.4) < 1e-9


def test_a_lane_that_is_too_short_refuses_rather_than_shortening_the_run():
    code = _quiet(lambda: run_main(lambda: main([*_CONTEXT,
        "--ladder-top", "0.5", "--lateral-top", "0.3", "--phase", "forward",
        "--lane-metres", "0.5", "--lane-width-metres", "2.0"]),
        "gait", printer=lambda _line: None))
    assert code == 2


def test_a_lane_that_is_too_narrow_refuses_because_there_is_no_lateral_sensing():
    code = _quiet(lambda: run_main(lambda: main([*_CONTEXT,
        "--ladder-top", "0.5", "--lateral-top", "0.5", "--phase", "strafe",
        "--lane-metres", "20.0", "--lane-width-metres", "0.1"]),
        "gait", printer=lambda _line: None))
    assert code == 2


# ── reading a floor off the ladder ──────────────────────────────────────────────────────
def test_a_bracketed_ladder_reports_the_floor_and_a_conservative_value_one_step_above():
    loco = FakeLocomotion(gain=0.8, floor=0.25)
    segments = _run(loco, forward_plan(0.6, 6))
    result = forward_floor_from(segments, 0.1)
    assert result["bracketed"] and result["monotonic"]
    assert abs(result["lowest_walking_m_s"] - 0.3) < 1e-9
    assert abs(result["conservative_m_s"] - 0.4) < 1e-9


def test_a_ladder_where_everything_walked_says_the_floor_was_not_found():
    loco = FakeLocomotion(gain=0.8, floor=0.0)
    segments = _run(loco, forward_plan(0.6, 6))
    result = forward_floor_from(segments, 0.1)
    assert result["bracketed"] is False
    assert "has not found it" in result["note"]
    assert result["conservative_m_s"] == result["lowest_walking_m_s"]


def test_a_non_monotonic_ladder_refuses_to_name_a_floor():
    """A robot cannot walk at 0.2 and 0.4 but not 0.3. That is a confound, not a floor."""
    loco = FakeLocomotion(gain=0.8, floor=0.0)
    segments = _run(loco, forward_plan(0.6, 6))
    hole = next(index for index, segment in enumerate(segments)
                if segment.role == "treatment" and abs(segment.commanded_vx - 0.4) < 1e-6)
    segments[hole] = dataclasses.replace(segments[hole], forward_m=0.0)
    result = forward_floor_from(segments, 0.1)
    assert result["monotonic"] is False
    assert result["conservative_m_s"] is None
    assert "confound" in result["note"]


def test_a_ladder_whose_anchors_walked_but_whose_rungs_did_not_is_refused():
    loco = FakeLocomotion(gain=0.8, floor=0.0)
    segments = _run(loco, forward_plan(0.6, 6))
    stalled = [segment if segment.role != "treatment"
               else dataclasses.replace(segment, forward_m=0.0)
               for segment in segments]
    try:
        forward_floor_from(stalled, 0.1)
    except Refusal as refusal:
        assert "contradiction" in str(refusal)
    else:
        raise AssertionError("anchors walking while no rung walked must refuse")


# ── the two lateral numbers ─────────────────────────────────────────────────────────────
def test_a_proportional_lateral_axis_reads_as_a_line_not_a_step():
    """The Go2's finding: delivery from the smallest rung up means there is no floor."""
    loco = FakeLocomotion(gain=1.0, lateral_gain=0.4, lateral_floor=0.0)
    segments = _run(loco, diagonal_plan(0.3, 0.4, 4))
    rows = lateral_rows(segments)
    assert all(abs(row["fraction"] - 0.4) < 0.05 for row in rows), rows
    assert "LINE" in lateral_verdict(rows)


def test_a_real_lateral_floor_reads_as_a_step():
    loco = FakeLocomotion(gain=1.0, lateral_gain=1.0, lateral_floor=0.35)
    segments = _run(loco, diagonal_plan(0.3, 0.4, 4))
    rows = lateral_rows(segments)
    assert rows[0]["delivered_m_s"] < 0.01
    assert "STEP" in lateral_verdict(rows)


def test_the_strafe_and_diagonal_phases_can_disagree_and_both_are_reported():
    """A robot with a standing-start floor and proportional in-gait delivery.

    This is the Go2's actual result, and the shape the probe must be able to express: the
    strafe number does not describe the diagonal case, so neither may stand in for the
    other.
    """
    class _StandingStartFloor(FakeLocomotion):
        def _delivered(self):
            vx, vy, _ = self.command
            forward = self.gain * vx
            # Already walking: lateral is delivered proportionally, no floor at all.
            # From a standstill: a real floor at 0.35. Those are the Go2's two results.
            lateral = (0.4 * vy if forward > 0.0
                       else (vy if abs(vy) >= 0.35 else 0.0))
            return (forward, lateral, 0.0)

    loco = _StandingStartFloor(gain=1.0)
    strafe = lateral_rows(_run(loco, strafe_plan(0.4, 4, anchor_vx=0.5)))
    diagonal = lateral_rows(_run(loco, diagonal_plan(0.3, 0.4, 4)))
    assert "STEP" in lateral_verdict(strafe)
    assert "LINE" in lateral_verdict(diagonal)


def test_lateral_delivery_is_signed_against_the_command():
    """A robot crabbing the wrong way must not report a plausible positive fraction."""
    class _Backwards(FakeLocomotion):
        def _delivered(self):
            vx, vy, _ = self.command
            return (self.gain * vx, -0.4 * vy, 0.0)

    loco = _Backwards(gain=1.0)
    rows = lateral_rows(_run(loco, diagonal_plan(0.3, 0.4, 4)))
    assert all(row["fraction"] < 0 for row in rows), rows


# ── refusing to invent ──────────────────────────────────────────────────────────────────
def test_no_ladder_top_means_refuse_rather_than_borrow_the_go2s_number():
    code = _quiet(lambda: run_main(
        lambda: main([*_CONTEXT, "--lane-metres", "6", "--lane-width-metres", "2"]),
        "gait", printer=lambda _line: None))
    assert code == 2


def test_no_go2_gait_floor_constant_is_executable_anywhere_in_this_directory():
    """Structural, and it covers the whole directory rather than one file.

    The Go2's measured floors are 0.35 m/s forward and 0.20 m/s lateral. Naming them in
    prose is how this code explains why they are not defaults here; evaluating them is
    how the next robot silently inherits them. So the check is over the AST, not the
    text: a *literal* 0.35 or 0.20 in any non-test module in this directory fails the
    suite, and the docstrings that argue against them do not.

    Scoped to the modules where a velocity default could live. 0.2 elsewhere in this
    directory is a socket timeout in seconds, and widening the check to catch those would
    make it noise that somebody eventually deletes.

    Verified by mutation: adding ``FLOOR = 0.35`` to gait_floor_probe.py turns this red.
    """
    banned = (0.35, 0.20)
    offenders = []
    for name in VELOCITY_MODULES:
        path = _HERE / name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, float) \
                    and any(abs(node.value - value) < 1e-12 for value in banned):
                offenders.append(f"{path.name}:{node.lineno} -> {node.value}")
    assert offenders == [], (
        "a Go2 gait-floor constant is executable in the Lite3 commissioning tree: "
        + ", ".join(offenders))


def test_a_dry_run_opens_no_socket_and_returns_zero():
    with tempfile.TemporaryDirectory() as directory:
        code = _quiet(lambda: main([*_CONTEXT,
            "--ladder-top", "0.5", "--lateral-top", "0.3",
            "--lane-metres", "20", "--lane-width-metres", "5",
            "--artefact", str(Path(directory) / "a.json")]))
    assert code == 0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"gait_floor_probe: {len(tests)}/{len(tests)} passed")
