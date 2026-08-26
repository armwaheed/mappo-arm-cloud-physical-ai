#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 actuator-gain probe.

The test this file exists for is
:func:`test_a_real_gain_is_not_readable_as_estimator_noise`. On the G1 a real 0.45
actuator gain was read as 0.17 m/s of velocity-estimate noise, and the robot then
random-walked toward its goal because the controller believed the estimate. The probe has
to fit the pose derivative, report the estimator separately, and say out loud when the two
disagree -- and a test that only checked the pose fit would pass whether or not the
comparison existed at all.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE))

from deep_robotics.lite3.commissioning.actuator_gain_probe import (
    ESTIMATOR_DISAGREEMENT,
    analyse,
    compare_estimators,
    estimator_pairs,
    execute,
    main,
    measured_pairs,
    plan_for,
    planned_forward_metres,
    sample_speeds,
)
from deep_robotics.lite3.commissioning.measurement import Refusal, run_main

from test_measurement import FakeLocomotion

_CONTEXT = ["--robot-id", "LITE3-A", "--firmware", "V1.0.8", "--payload", "none"]


def _quiet(callable_):
    with contextlib.redirect_stdout(io.StringIO()):
        return callable_()


def _run(loco, plan, segment=2.0):
    return execute(loco, plan, segment_s=segment, tick_s=0.1,
                   printer=lambda _line: None, clock=loco.clock, sleep=loco.sleep)


# ── the sample set ──────────────────────────────────────────────────────────────────────
def test_samples_are_anchored_at_the_floor_and_at_the_demo_envelope():
    speeds = sample_speeds(0.20, 0.40, 3)
    assert abs(speeds[0] - 0.20) < 1e-9
    assert abs(speeds[-1] - 0.40) < 1e-9
    assert abs(speeds[1] - 0.30) < 1e-9


def test_no_sample_is_below_the_measured_gait_floor():
    """Below the floor the robot does not walk, so those points are not on the line."""
    speeds = sample_speeds(0.25, 0.45, 5)
    assert all(speed >= 0.25 - 1e-12 for speed in speeds)


def test_an_envelope_below_the_floor_is_refused_rather_than_extrapolated():
    try:
        sample_speeds(0.30, 0.20, 3)
    except Refusal as refusal:
        assert "below the measured gait floor" in str(refusal)
    else:
        raise AssertionError("a demo envelope under the floor must refuse")


def test_a_single_point_is_a_division_not_a_fit_and_is_refused():
    try:
        sample_speeds(0.2, 0.4, 1)
    except Refusal as refusal:
        assert "fit rather than a division" in str(refusal)
    else:
        raise AssertionError("one point cannot be a fit")


def test_repeats_alternate_direction_so_a_warming_robot_is_not_read_as_a_gain():
    plan = plan_for([0.2, 0.3, 0.4], 2)
    treatments = [vx for role, vx in plan if role == "treatment"]
    assert treatments == [0.2, 0.3, 0.4, 0.4, 0.3, 0.2]


def test_every_treatment_has_a_contemporaneous_control_at_the_same_speed():
    plan = plan_for([0.2, 0.4], 1)
    assert plan == [("control", 0.2), ("treatment", 0.2),
                    ("control", 0.4), ("treatment", 0.4)]


def test_planned_distance_counts_the_controls_because_they_walk_too():
    assert abs(planned_forward_metres([("control", 0.3), ("treatment", 0.3)], 2.0)
               - 1.2) < 1e-9


# ── the fit ─────────────────────────────────────────────────────────────────────────────
def test_a_known_gain_is_recovered_from_the_pose_derivative():
    loco = FakeLocomotion(gain=0.62)
    segments = _run(loco, plan_for(sample_speeds(0.2, 0.4, 3), 2))
    result = _quiet(lambda: analyse(segments, printer=lambda _line: None))
    assert abs(result["pose_fit"]["gain"] - 0.62) < 0.02, result["pose_fit"]
    assert result["pose_fit"]["residual_rms_m_s"] < 0.01


def test_a_real_gain_is_not_readable_as_estimator_noise():
    """The G1 failure, reproduced and caught.

    The robot delivers 0.45 of what it is asked for while its own velocity estimate
    insists it is delivering everything. The pose fit must find 0.45, the estimator fit
    must find 1.0, and the report must say they are not describing the same event.
    """
    loco = FakeLocomotion(gain=0.45, estimator_gain=1.0)
    segments = _run(loco, plan_for(sample_speeds(0.2, 0.4, 3), 2))
    result = _quiet(lambda: analyse(segments, printer=lambda _line: None))
    assert abs(result["pose_fit"]["gain"] - 0.45) < 0.02
    assert abs(result["estimator_fit"]["gain"] - 1.0) < 0.05
    assert "not describing the same event" in result["estimator_note"]


def test_agreeing_instruments_are_reported_as_agreeing_and_the_pose_number_still_ships():
    loco = FakeLocomotion(gain=0.6, estimator_gain=0.6)
    segments = _run(loco, plan_for(sample_speeds(0.2, 0.4, 3), 1))
    result = _quiet(lambda: analyse(segments, printer=lambda _line: None))
    assert "agree" in result["estimator_note"]
    assert "pose number still ships" in result["estimator_note"]


def test_the_disagreement_threshold_is_the_thing_that_decides():
    tight = compare_estimators({"gain": 0.60},
                               {"gain": 0.60 + ESTIMATOR_DISAGREEMENT / 2})
    wide = compare_estimators({"gain": 0.60},
                              {"gain": 0.60 + ESTIMATOR_DISAGREEMENT * 2})
    assert "agree" in tight and "not describing the same event" in wide


def test_a_missing_estimator_is_reported_as_missing_not_as_agreement():
    assert "nothing to cross-check" in compare_estimators({"gain": 0.6}, None)


def test_the_two_instruments_are_never_merged_into_one_sample_set():
    loco = FakeLocomotion(gain=0.45, estimator_gain=1.0)
    segments = _run(loco, plan_for([0.3], 1))
    pose = measured_pairs(segments)
    estimate = estimator_pairs(segments)
    assert len(pose) == len(estimate) == 1
    assert pose[0][1] != estimate[0][1]


def test_a_run_where_the_robot_never_walked_is_refused_not_reported_as_gain_zero():
    """Every segment here is at or above the floor, so a zero can only mean dead legs."""
    loco = FakeLocomotion(stood=False)
    segments = _run(loco, plan_for(sample_speeds(0.2, 0.4, 3), 1))
    try:
        analyse(segments, printer=lambda _line: None)
    except Refusal as refusal:
        assert "real and total" in str(refusal)
    else:
        raise AssertionError("a gain of zero from a prone robot must be refused")


def test_a_partially_stalled_run_is_refused_too():
    """A robot that stops walking part way through would otherwise halve the gain."""
    import dataclasses

    loco = FakeLocomotion(gain=0.6)
    segments = _run(loco, plan_for(sample_speeds(0.2, 0.4, 3), 1))
    segments[-1] = dataclasses.replace(segments[-1], forward_m=0.0)
    try:
        analyse(segments, printer=lambda _line: None)
    except Refusal:
        pass
    else:
        raise AssertionError("one dead segment must refuse the whole fit")


# ── refusing to invent ──────────────────────────────────────────────────────────────────
def test_no_gait_floor_means_refuse_rather_than_assume_one():
    code = _quiet(lambda: run_main(
        lambda: main([*_CONTEXT, "--envelope-vx", "0.35", "--lane-metres", "6"]),
        "gain", printer=lambda _line: None))
    assert code == 2


def test_a_lane_too_short_for_the_sweep_refuses():
    code = _quiet(lambda: run_main(
        lambda: main([*_CONTEXT, "--gait-floor", "0.2", "--envelope-vx", "0.4",
                                 "--lane-metres", "0.5"]),
        "gain", printer=lambda _line: None))
    assert code == 2


def test_a_dry_run_returns_zero_and_writes_nothing():
    with tempfile.TemporaryDirectory() as directory:
        code = _quiet(lambda: main([*_CONTEXT,
            "--gait-floor", "0.2", "--envelope-vx", "0.4", "--lane-metres", "40",
            "--artefact", str(Path(directory) / "a.json")]))
        assert code == 0
        assert not (Path(directory) / "a.json").exists()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"actuator_gain_probe: {len(tests)}/{len(tests)} passed")
