#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the shared commissioning measurement plumbing.

These are guard tests before they are arithmetic tests. Every refusal in this directory
exists because some version of this measurement has already been got wrong somewhere in
this corpus, and a refusal nobody can make fire is not a refusal. So each test below was
written by breaking the thing it guards and confirming the test went red first.

:class:`FakeLocomotion` is the only robot any test in this directory talks to. It
integrates whatever it is commanded, with a settable gait floor and actuator gain, so a
probe can be driven through a plausible robot, a robot with a real floor, and -- the case
that matters most -- a robot that never stood up and silently ignores everything.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning.measurement import (
    PROVISIONAL,
    REVIEWED,
    WALKED_MARGIN_M,
    Record,
    Refusal,
    body_delta,
    brief,
    check_anchors_walked,
    check_controls_are_still,
    check_every_segment_walked,
    control_baseline,
    fit_ratio,
    merge_measurement,
    new_record,
    paste_block,
    read_record,
    refuse_unmeasured,
    require_positive_finite,
    require_reviewed,
    run_main,
    run_segment,
    write_record,
)


@dataclass(frozen=True)
class _Pose:
    x: float
    y: float
    yaw: float


class FakeLocomotion:
    """A robot that walks exactly as told, above a settable floor.

    ``gain`` scales delivered speed. ``floor`` is the commanded speed below which nothing
    happens at all. ``estimator_gain`` is what the platform's own velocity estimate would
    report, so a test can make the two instruments disagree the way the G1's did.
    ``stood`` False models the failure this whole harness is shaped around: every command
    is accepted, acknowledged, and ignored.
    """

    def __init__(self, *, gain: float = 1.0, floor: float = 0.0,
                 lateral_gain: float = 1.0, lateral_floor: float = 0.0,
                 estimator_gain=None, stood: bool = True, yaw_drift_rad_s: float = 0.0):
        self.gain = gain
        self.floor = floor
        self.lateral_gain = lateral_gain
        self.lateral_floor = lateral_floor
        self.estimator_gain = gain if estimator_gain is None else estimator_gain
        self.stood = stood
        self.yaw_drift = yaw_drift_rad_s
        self.time = 0.0
        self.x = self.y = self.yaw = 0.0
        self.command = (0.0, 0.0, 0.0)
        self.commands = []
        self.stops = 0

    # -- the interface the probes use -------------------------------------------------
    def set_velocity(self, vx, vy, vyaw):
        self.command = (vx, vy, vyaw)
        self.commands.append((vx, vy, vyaw))

    def velocity(self):
        vx, vy, _ = self._delivered()
        return (vx / max(self.gain, 1e-9) * self.estimator_gain, vy, 0.0)

    def pose(self):
        return _Pose(self.x, self.y, self.yaw)

    def stop(self):
        self.stops += 1
        self.command = (0.0, 0.0, 0.0)

    # -- the fake's own clock ----------------------------------------------------------
    def clock(self):
        return self.time

    def sleep(self, seconds):
        vx, vy, _ = self._delivered()
        # Body-frame delivery integrated in the world frame, so a yaw drift shows up as
        # travel on the wrong axis -- which is exactly what body_delta has to undo.
        self.x += (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * seconds
        self.y += (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * seconds
        self.yaw += self.yaw_drift * seconds
        self.time += seconds

    def _delivered(self):
        vx, vy, _ = self.command
        if not self.stood:
            return (0.0, 0.0, 0.0)
        forward = self.gain * vx if abs(vx) >= self.floor else 0.0
        lateral = self.lateral_gain * vy if abs(vy) >= self.lateral_floor else 0.0
        return (forward, lateral, 0.0)


def _segment(loco, *, role="treatment", vx=0.0, vy=0.0, duration=1.0, tick=0.1):
    return run_segment(loco, role=role, vx=vx, vy=vy, duration_s=duration,
                       tick_s=tick, clock=loco.clock, sleep=loco.sleep)


# ── geometry ────────────────────────────────────────────────────────────────────────────
def test_body_delta_returns_forward_and_lateral_in_the_segments_own_frame():
    forward, lateral = body_delta((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert abs(forward - 1.0) < 1e-9 and abs(lateral) < 1e-9
    forward, lateral = body_delta((0.0, 0.0, math.pi / 2), (0.0, 1.0, math.pi / 2))
    assert abs(forward - 1.0) < 1e-9 and abs(lateral) < 1e-9


def test_body_delta_uses_the_mean_yaw_so_a_turning_segment_is_not_mis_split():
    """Either endpoint attributes part of the arc to the wrong axis; the mean does not."""
    start, end = (0.0, 0.0, -0.4), (1.0, 0.0, 0.4)
    mean_forward, mean_lateral = body_delta(start, end)
    end_forward = 1.0 * math.cos(0.4)
    assert mean_forward > end_forward
    assert abs(mean_lateral) < 1e-9


def test_a_yaw_drift_does_not_become_lateral_travel():
    loco = FakeLocomotion(gain=1.0, yaw_drift_rad_s=0.2)
    segment = _segment(loco, vx=0.4, duration=1.0)
    assert abs(segment.lateral_m) < 0.02, segment.lateral_m
    assert segment.forward_m > 0.35


# ── segments ────────────────────────────────────────────────────────────────────────────
def test_run_segment_measures_pose_not_the_platform_estimate():
    """The G1 lesson: a real gain must not be readable as estimator noise.

    The fake delivers 0.45 of what it is asked for while its own velocity estimate claims
    the full command. The segment must report the pose truth and carry the estimate
    separately, never blended.
    """
    loco = FakeLocomotion(gain=0.45, estimator_gain=1.0)
    segment = _segment(loco, vx=0.40, duration=1.0)
    assert abs(segment.forward_mps - 0.18) < 0.02, segment.forward_mps
    assert abs(segment.estimator_forward_mps - 0.40) < 0.02
    assert segment.estimator_samples > 0


def test_run_segment_resends_the_command_every_tick():
    """The vendor interface applies what it last received; one send is one datagram."""
    loco = FakeLocomotion()
    _segment(loco, vx=0.3, duration=1.0, tick=0.1)
    assert len(loco.commands) >= 9, len(loco.commands)


def test_run_segment_counts_estimator_failures_instead_of_hiding_them():
    class _NoEstimate(FakeLocomotion):
        def velocity(self):
            raise RuntimeError("no state frame")

    loco = _NoEstimate()
    segment = _segment(loco, vx=0.3, duration=0.5)
    assert segment.estimator_samples == 0
    assert segment.estimator_failures >= 4
    assert math.isnan(segment.estimator_forward_mps)


def test_a_clock_that_does_not_advance_is_refused_rather_than_divided_by():
    loco = FakeLocomotion()
    try:
        run_segment(loco, role="treatment", vx=0.3, vy=0.0, duration_s=0.0,
                    tick_s=0.1, clock=lambda: 5.0, sleep=lambda _s: None)
    except Refusal as refusal:
        assert "clock" in str(refusal)
    else:
        raise AssertionError("a zero-length segment must refuse, not divide by zero")


# ── the refusals ────────────────────────────────────────────────────────────────────────
def test_a_robot_that_never_stood_is_refused_rather_than_reported_as_a_total_floor():
    """The Go2's own failure, reproduced: every axis reads 0.000 and that is not a floor."""
    loco = FakeLocomotion(stood=False)
    segments = [_segment(loco, role="anchor", vx=0.5),
                _segment(loco, role="control", vx=0.0),
                _segment(loco, role="treatment", vx=0.3)]
    assert all(segment.forward_mps == 0.0 for segment in segments)
    try:
        check_anchors_walked(segments)
    except Refusal as refusal:
        assert "not walking" in str(refusal)
    else:
        raise AssertionError("an all-zero run must be refused")


def test_a_run_with_no_anchor_at_all_is_refused():
    loco = FakeLocomotion()
    segments = [_segment(loco, role="treatment", vx=0.3)]
    try:
        check_anchors_walked(segments)
    except Refusal as refusal:
        assert "no anchor" in str(refusal)
    else:
        raise AssertionError("a run with nothing proving the legs work must be refused")


def test_a_treatment_that_did_not_travel_is_a_finding_not_a_refusal():
    """A floor IS a treatment that does not move. Refusing it would refuse every result."""
    loco = FakeLocomotion(floor=0.25)
    segments = [_segment(loco, role="anchor", vx=0.5),
                _segment(loco, role="treatment", vx=0.1)]
    check_anchors_walked(segments)     # must not raise
    assert not segments[1].travelled


def test_a_drifting_control_refuses_the_run_so_the_margin_cannot_be_an_article_of_faith():
    class _Drifting(FakeLocomotion):
        def _delivered(self):
            return (WALKED_MARGIN_M * 2.0, 0.0, 0.0)   # drifts even when commanded zero

    loco = _Drifting()
    segments = [_segment(loco, role="control", vx=0.0, duration=1.0)]
    try:
        check_controls_are_still(segments, "forward")
    except Refusal as refusal:
        assert "drift" in str(refusal)
    else:
        raise AssertionError("a control that drifts as far as a walk must refuse the run")


def test_the_control_check_looks_at_the_axis_under_test_not_always_forward():
    """A diagonal control walks forward on purpose; checking forward would refuse it."""
    loco = FakeLocomotion()
    segments = [_segment(loco, role="control", vx=0.35, vy=0.0, duration=1.0)]
    check_controls_are_still(segments, "lateral")     # must not raise
    try:
        check_controls_are_still(segments, "forward")
    except Refusal:
        pass
    else:
        raise AssertionError("the forward axis check should have objected to this control")


def test_every_segment_walked_is_the_go2_refusal_unweakened():
    loco = FakeLocomotion(floor=0.25)
    segments = [_segment(loco, role="treatment", vx=0.35),
                _segment(loco, role="treatment", vx=0.10)]
    try:
        check_every_segment_walked(segments)
    except Refusal as refusal:
        assert "real and total" in str(refusal)
    else:
        raise AssertionError("a phase where every segment should walk must refuse a zero")


def test_control_baseline_is_the_mean_of_the_controls_on_the_named_axis():
    loco = FakeLocomotion()
    segments = [_segment(loco, role="control", vy=0.2, duration=1.0),
                _segment(loco, role="control", vy=0.4, duration=1.0),
                _segment(loco, role="treatment", vy=1.0, duration=1.0)]
    assert abs(control_baseline(segments, "lateral") - 0.3) < 1e-6


# ── fits ────────────────────────────────────────────────────────────────────────────────
def test_fit_ratio_recovers_a_known_gain_through_the_origin():
    result = fit_ratio([(0.2, 0.12), (0.3, 0.18), (0.4, 0.24)])
    assert abs(result["gain"] - 0.6) < 1e-9
    assert result["residual_rms_m_s"] < 1e-12
    assert result["samples"] == 3


def test_fit_ratio_has_no_intercept_so_a_creeping_robot_cannot_buy_accuracy():
    """With an intercept the fit would report gain 1.0 and call the offset a constant."""
    result = fit_ratio([(0.2, 0.3), (0.3, 0.4), (0.4, 0.5)])
    assert result["gain"] > 1.0
    assert result["residual_rms_m_s"] > 0.01


def test_fit_ratio_reports_the_spread_of_the_point_ratios():
    result = fit_ratio([(0.2, 0.10), (0.4, 0.32)])
    assert abs(result["ratio_min"] - 0.5) < 1e-9
    assert abs(result["ratio_max"] - 0.8) < 1e-9


def test_fit_ratio_refuses_an_all_zero_command_set():
    try:
        fit_ratio([(0.0, 0.1), (0.0, 0.2)])
    except Refusal as refusal:
        assert "undefined" in str(refusal)
    else:
        raise AssertionError("a gain from zero commands must refuse")


def test_fit_ratio_refuses_an_empty_sample_set():
    try:
        fit_ratio([])
    except Refusal:
        pass
    else:
        raise AssertionError("an empty fit must refuse")


# ── unmeasured sentinels ────────────────────────────────────────────────────────────────
def test_refuse_unmeasured_names_the_flag_not_the_variable():
    try:
        refuse_unmeasured(**{"--gait-floor": None, "--actuator-gain": 0.6})
    except Refusal as refusal:
        assert "--gait-floor" in str(refusal) and "--actuator-gain" not in str(refusal)
    else:
        raise AssertionError("an unmeasured value must refuse")


def test_require_positive_finite_rejects_zero_nan_and_negatives():
    for value in (0.0, -1.0, float("nan"), float("inf")):
        try:
            require_positive_finite(**{"--x": value})
        except Refusal:
            continue
        raise AssertionError(f"{value} should not have been accepted")
    require_positive_finite(**{"--x": 0.25})


# ── the record ──────────────────────────────────────────────────────────────────────────
def test_a_new_record_is_always_provisional_and_there_is_no_way_to_make_a_reviewed_one():
    record = new_record("LITE3-A", firmware="V1.0.8", payload="none")
    assert record.provenance == PROVISIONAL
    assert record.reviewed_by is None


def test_a_record_without_a_robot_id_is_refused():
    for value in ("", "   "):
        try:
            new_record(value)
        except Refusal as refusal:
            assert "--robot-id" in str(refusal)
        else:
            raise AssertionError("a number with no robot on it is not a measurement")


def test_record_round_trips_through_json():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.json"
        record = merge_measurement(new_record("LITE3-B", firmware="V1"), "gait",
                                   {"conservative_m_s": 0.3})
        write_record(path, record)
        back = read_record(path)
        assert back.robot_id == "LITE3-B"
        assert back.measurements["gait"]["conservative_m_s"] == 0.3
        assert back.provenance == PROVISIONAL
        assert "measured_utc" in back.measurements["gait"]


def test_a_record_from_a_future_schema_is_refused_rather_than_half_read():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.json"
        path.write_text(json.dumps({"schema": "lite3-commissioning/v99",
                                    "robot_id": "LITE3-A"}))
        try:
            read_record(path)
        except Refusal as refusal:
            assert "misread" in str(refusal)
        else:
            raise AssertionError("an unknown schema must refuse")


def test_require_reviewed_refuses_a_provisional_record_and_says_how_to_fix_it():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.json"
        write_record(path, new_record("LITE3-A"))
        try:
            require_reviewed(path)
        except Refusal as refusal:
            assert PROVISIONAL in str(refusal)
            assert "--review" in str(refusal)
        else:
            raise AssertionError("a provisional record must not reach live movement")


def test_a_provisional_record_is_refused_even_when_a_reviewer_name_is_present():
    """Isolates the provenance check from the signature check.

    Both gates exist, and on an ordinary provisional record either one alone would refuse
    it -- which makes a mutation that removes just one invisible. This record carries a
    reviewer name and is still marked provisional, so only the provenance check can
    refuse it. Verified by mutation: neutering that check turns this red and leaves every
    other test green.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.json"
        record = new_record("LITE3-A")
        record.reviewed_by = "Somebody Who Did Not Sign It"
        write_record(path, record)
        assert record.provenance == PROVISIONAL
        try:
            require_reviewed(path)
        except Refusal as refusal:
            assert PROVISIONAL in str(refusal)
        else:
            raise AssertionError("a provisional record must be refused on its provenance")


def test_require_reviewed_refuses_a_review_nobody_signed():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.json"
        record = new_record("LITE3-A")
        record.provenance = REVIEWED
        write_record(path, record)
        try:
            require_reviewed(path)
        except Refusal as refusal:
            assert "names no reviewer" in str(refusal)
        else:
            raise AssertionError("an unsigned review is not a review")


def test_require_reviewed_accepts_a_signed_review():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.json"
        record = new_record("LITE3-A")
        record.provenance = REVIEWED
        record.reviewed_by = "A Reviewer"
        write_record(path, record)
        assert require_reviewed(path).reviewed_by == "A Reviewer"


def test_read_record_ignores_keys_it_does_not_know_rather_than_crashing():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.json"
        payload = Record(robot_id="LITE3-A").as_dict()
        payload["some_future_field"] = 1
        path.write_text(json.dumps(payload))
        assert read_record(path).robot_id == "LITE3-A"


# ── presentation ────────────────────────────────────────────────────────────────────────
def test_brief_prints_whether_the_robot_moves_before_anything_else():
    lines = []
    brief("T", does="d", needs=["n"], means="m", moves=True, printer=lines.append)
    assert any("MOVES THE ROBOT: YES" in line for line in lines)
    lines = []
    brief("T", does="d", needs=["n"], means="m", moves=False, printer=lines.append)
    assert any("MOVES THE ROBOT: no" in line for line in lines)


def test_paste_block_is_self_contained_markdown_with_no_attachment_links():
    text = paste_block("T", [("a", "1")], ["- note"])
    assert "| a | 1 |" in text
    assert "user-attachments" not in text and "![" not in text


def test_run_main_turns_a_refusal_into_a_banner_and_exit_two():
    lines = []

    def entry():
        raise Refusal("because")

    assert run_main(entry, "x", printer=lines.append) == 2
    assert any("REFUSING" in line for line in lines)
    assert any("because" in line for line in lines)


def test_run_main_passes_a_success_through():
    assert run_main(lambda: 0, "x", printer=lambda _line: None) == 0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"measurement: {len(tests)}/{len(tests)} passed")
