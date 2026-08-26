#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the per-primitive speed measurement.

The robot these tests use is **not** ``FakeLocomotion``. That one is proportional --
delivered = gain x commanded -- which is the legacy velocity transport's behaviour and
the opposite of the thing under test here. :class:`SignOnlyLocomotion` below runs the
**real** :meth:`AxisProfile.map_velocity` and then delivers one fixed speed per emitted
primitive, so these tests exercise the actual deadband, the actual octant snap, and the
actual lateral inversion rather than a second implementation of them that could agree
with the probe while both were wrong.

The two tests worth reading first:

* ``test_the_reported_speed_does_not_depend_on_the_commanded_magnitude`` is the one that
  says this probe is not a ladder in disguise.
* ``test_the_primitive_table_matches_what_the_real_mapping_actually_emits`` is the one
  that would fail if the lateral pair's inversion were copied out of the field name
  instead of out of the transport.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE))

from deep_robotics.lite3.commissioning.axis_primitive_probe import (
    COMMAND_MARGIN,
    PRIMITIVES,
    _profile_sha256,
    analyse,
    build_parser,
    compare_with_profile,
    delivered_speeds,
    execute,
    main,
    measured_block,
    plan_for,
    planned_excursion_m,
    primitives_in,
    record_context,
)
from deep_robotics.lite3.commissioning.measurement import Refusal, run_main
from deep_robotics.lite3.locomotion.lite3_axis_locomotion import AxisProfile

_CONTEXT = ["--robot-id", "LITE3-A", "--firmware", "V1.0.8", "--payload", "none"]
_AXIS = ["--locomotion-transport", "axis"]

#: Raw axis values that clear the documented per-axis dead zones (forward 6553, lateral
#: 12553) and carry the sign their field name requires.
_RAW = {
    "forward_positive": 32767,
    "forward_negative": -32767,
    "lateral_positive": 32767,
    "lateral_negative": -32767,
}

_BY_NAME = {primitive.name: primitive for primitive in PRIMITIVES}


def _profile_dict(*, deadband: float = 0.05, primitives=None, measured=None) -> dict:
    raw = _RAW if primitives is None else primitives
    return {
        "schema": "lite3-axis-profile/v1",
        "input_deadband": {"linear_m_s": deadband, "yaw_rad_s": 0.1},
        "allowed_gait_states": [0],
        "evidence": {name: "vendor V1.0.8 reference control script"
                     for name, value in raw.items() if value is not None},
        "measured_m_s": measured or {},
        "measured_rad_s": {},
        "primitives": {**dict.fromkeys(_RAW), **raw},
    }


def _profile(**kwargs) -> AxisProfile:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.json"
        path.write_text(json.dumps(_profile_dict(**kwargs)), encoding="utf-8")
        return AxisProfile.load(path)


class _Pose:
    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw


class SignOnlyLocomotion:
    """A robot reached through the real sign-only mapping.

    ``speeds`` is the magnitude each primitive delivers, and the commanded magnitude is
    discarded exactly as the transport discards it. The vendor's frame is modelled where
    it physically belongs -- in the robot -- so a **positive** raw lateral axis value
    moves this robot to its RIGHT, which is negative body-frame y. If the probe's own
    table disagrees with that, its direction check fires, which is the point.

    ``lateral_convention`` flips that, modelling a firmware whose lateral axis does not
    follow the documented convention. Nothing in this repository would notice such a unit
    except the direction check.
    """

    def __init__(self, profile, speeds, *, stood: bool = True,
                 lateral_convention: int = +1, drift_m_s: float = 0.0):
        self._profile = profile
        self._speeds = speeds
        self._stood = stood
        self._lateral_convention = lateral_convention
        self._drift = drift_m_s
        self.time = 0.0
        self.x = self.y = self.yaw = 0.0
        self._body = (0.0, 0.0)
        self.commands = []
        self.stops = 0

    def set_velocity(self, vx, vy, vyaw):
        self.commands.append((vx, vy, vyaw))
        axes = self._profile.map_velocity(vx, vy, vyaw)
        self._body = (0.0, 0.0) if not self._stood else self._body_for(axes)

    def _body_for(self, axes):
        forward = lateral = 0.0
        if axes.forward and axes.forward == self._profile.forward_positive:
            forward = +self._speeds.get("forward_positive", 0.0)
        elif axes.forward and axes.forward == self._profile.forward_negative:
            forward = -self._speeds.get("forward_negative", 0.0)
        # A vendor-POSITIVE raw lateral value drives the robot to its right, which is
        # negative body y. The navigator's +y is left; the transport is what inverts.
        if axes.lateral and axes.lateral == self._profile.lateral_positive:
            lateral = -self._speeds.get("lateral_positive", 0.0)
        elif axes.lateral and axes.lateral == self._profile.lateral_negative:
            lateral = +self._speeds.get("lateral_negative", 0.0)
        return (forward, lateral * self._lateral_convention)

    def velocity(self):
        return (self._body[0], self._body[1], 0.0)

    def pose(self):
        return _Pose(self.x, self.y, self.yaw)

    def stop(self):
        self.stops += 1
        self._body = (0.0, 0.0)

    def clock(self):
        return self.time

    def sleep(self, seconds):
        forward, lateral = self._body
        self.x += (forward * math.cos(self.yaw) - lateral * math.sin(self.yaw)) * seconds
        self.y += (forward * math.sin(self.yaw) + lateral * math.cos(self.yaw)) * seconds
        self.x += self._drift * seconds
        self.time += seconds


def _run(loco, plan, segment=2.0):
    return execute(loco, plan, segment_s=segment, tick_s=0.1,
                   printer=lambda _line: None, clock=loco.clock, sleep=loco.sleep)


def _measure(primitive_name, speeds, *, deadband=0.05, repeats=3, **kwargs):
    profile = _profile(deadband=deadband)
    primitive = _BY_NAME[primitive_name]
    loco = SignOnlyLocomotion(profile, speeds, **kwargs)
    segments = _run(loco, plan_for(primitive, deadband, repeats))
    return analyse(segments, primitive, printer=lambda _line: None)


def _refuses(callable_):
    try:
        callable_()
    except Refusal as refusal:
        return str(refusal)
    raise AssertionError("expected a Refusal and did not get one")


# ── the property that says this is not a ladder ─────────────────────────────────────────
def test_the_reported_speed_does_not_depend_on_the_commanded_magnitude():
    """The whole justification for this probe replacing the gait ladder.

    Two profiles with different deadbands make the probe command two different speeds --
    0.075 m/s and 0.60 m/s, an eightfold difference. On this transport both emit the same
    datagram, so the reported delivery has to be identical. If it ever tracks the command,
    this file is measuring the probe.
    """
    slow = _measure("forward_positive", {"forward_positive": 0.42}, deadband=0.05)
    fast = _measure("forward_positive", {"forward_positive": 0.42}, deadband=0.40)
    assert abs(slow["declare_m_s"] - 0.42) < 1e-6
    assert abs(slow["declare_m_s"] - fast["declare_m_s"]) < 1e-9


def test_the_primitive_table_matches_what_the_real_mapping_actually_emits():
    """Each row must fire the primitive it names, through the transport's own mapping.

    The lateral pair is why this exists: ``lateral_positive`` is a vendor-positive value
    and is reached by a NEGATIVE commanded vy. Reading that off the field name gets it
    backwards, and the resulting probe would confidently record each lateral primitive's
    speed under the other one's name.
    """
    profile = _profile()
    command = 0.05 * COMMAND_MARGIN
    for primitive in PRIMITIVES:
        axes = profile.map_velocity(primitive.vx_sign * command,
                                    primitive.vy_sign * command, 0.0)
        emitted = axes.forward if primitive.axis == "forward" else axes.lateral
        assert emitted == getattr(profile, primitive.name), primitive.name


def test_every_primitive_travels_the_direction_its_row_claims():
    for primitive in PRIMITIVES:
        result = _measure(primitive.name, {primitive.name: 0.30})
        assert result["primitive"] == primitive.name
        assert abs(result["declare_m_s"] - 0.30) < 1e-6


# ── the refusals ────────────────────────────────────────────────────────────────────────
def test_a_primitive_that_does_not_move_the_robot_is_refused():
    """A raw value under the firmware dead zone, or a robot that never stood."""
    message = _refuses(lambda: _measure("forward_positive", {"forward_positive": 0.0}))
    assert "there is no low rung in this probe" in message
    assert "it is an absent one" in message


def test_a_prone_robot_is_refused_rather_than_reported_as_a_slow_primitive():
    message = _refuses(
        lambda: _measure("forward_positive", {"forward_positive": 0.4}, stood=False))
    assert "moved less than" in message


def test_a_primitive_that_moves_the_robot_the_wrong_way_is_refused():
    """A firmware whose lateral axis does not follow the documented convention.

    Nothing else in this repository would catch it: the magnitude is perfectly plausible
    and the robot strafes into the side of the lane nobody cleared.
    """
    message = _refuses(lambda: _measure(
        "lateral_positive", {"lateral_positive": 0.25}, lateral_convention=-1))
    assert "WRONG WAY" in message
    assert "to its right" in message


def test_a_drifting_control_refuses_the_run_on_the_axis_under_test():
    message = _refuses(lambda: _measure(
        "forward_positive", {"forward_positive": 0.4}, drift_m_s=0.5))
    assert "control segments drifted" in message


def test_a_profile_carrying_no_linear_primitive_is_refused():
    empty = _profile(primitives=dict.fromkeys(_RAW))
    message = _refuses(lambda: primitives_in(empty))
    assert "carries no linear primitive at all" in message


def test_only_the_primitives_the_profile_carries_are_run():
    partial = _profile(primitives={"forward_positive": 32767,
                                   "forward_negative": None,
                                   "lateral_positive": None,
                                   "lateral_negative": None})
    assert [p.name for p in primitives_in(partial)] == ["forward_positive"]


def test_a_control_is_interleaved_before_every_treatment():
    """Without them the drift check has nothing to check and the baseline is silently 0.

    Blocked controls would not do: a warm robot walks differently from a cold one, and a
    trend across the run is exactly what a contemporaneous control is there to separate
    from the primitive's delivery.
    """
    plan = plan_for(_BY_NAME["forward_positive"], 0.05, 3)
    roles = [role for role, _vx, _vy in plan]
    assert roles == ["control", "forward_positive"] * 3 + ["control"]


def test_a_plan_needs_at_least_two_treatments_to_show_a_spread():
    message = _refuses(lambda: plan_for(_BY_NAME["forward_positive"], 0.05, 1))
    assert "at least 2 treatments" in message


# ── what the number is ──────────────────────────────────────────────────────────────────
def test_the_declared_number_is_the_maximum_and_not_the_mean():
    """It is compared against a safety ceiling, so it is the fastest sample, not the mean."""
    profile = _profile()
    primitive = _BY_NAME["forward_positive"]
    loco = SignOnlyLocomotion(profile, {"forward_positive": 0.30})
    plan = plan_for(primitive, 0.05, 3)
    segments = _run(loco, plan)
    # Replace one treatment with a faster one; the report must follow the fast sample.
    faster = []
    seen = False
    for segment in segments:
        if segment.role == primitive.name and not seen:
            seen = True
            faster.append(dataclasses.replace(segment, forward_m=segment.forward_m * 2.0))
        else:
            faster.append(segment)
    result = analyse(faster, primitive, printer=lambda _line: None)
    assert result["declare_m_s"] > result["mean_m_s"]
    assert abs(result["declare_m_s"] - 0.60) < 1e-6


def test_a_lateral_primitive_is_measured_on_the_lateral_axis():
    """Measuring it forward would report 0.000 m/s for a primitive that worked perfectly."""
    result = _measure("lateral_negative", {"lateral_negative": 0.22})
    assert result["axis"] == "lateral"
    assert abs(result["declare_m_s"] - 0.22) < 1e-6


def test_the_control_baseline_is_subtracted_from_the_delivery():
    profile = _profile()
    primitive = _BY_NAME["forward_positive"]
    loco = SignOnlyLocomotion(profile, {"forward_positive": 0.30}, drift_m_s=0.01)
    segments = _run(loco, plan_for(primitive, 0.05, 3))
    speeds = delivered_speeds(segments, primitive)
    assert all(abs(speed - 0.30) < 1e-6 for speed in speeds)


# ── the profile comparison ──────────────────────────────────────────────────────────────
def test_a_profile_that_under_declares_a_primitive_is_called_out():
    """The dangerous direction: the envelope gate passed a robot faster than it checked."""
    declared = _profile(measured={"forward_positive": 0.30})
    notes = compare_with_profile({"forward_positive": {"declare_m_s": 0.47}}, declared)
    assert any("⚠️" in note and "1.57x" in note for note in notes)


def test_a_profile_that_over_declares_is_not_called_a_problem():
    declared = _profile(measured={"forward_positive": 0.60})
    notes = compare_with_profile({"forward_positive": {"declare_m_s": 0.47}}, declared)
    assert not any("⚠️" in note for note in notes)
    assert any("Not under-declared" in note for note in notes)


def test_a_primitive_with_nothing_declared_says_so():
    notes = compare_with_profile({"forward_positive": {"declare_m_s": 0.47}}, _profile())
    assert any("nothing was declared" in note for note in notes)


def test_the_pasteable_block_is_the_json_the_profile_actually_takes():
    block = measured_block({"forward_positive": {"declare_m_s": 0.4213},
                            "lateral_negative": {"declare_m_s": 0.2}})
    parsed = json.loads("{" + block + "}")
    assert parsed["measured_m_s"] == {"forward_positive": 0.421, "lateral_negative": 0.2}
    # And the profile loader accepts it, which is the only thing that matters about it.
    merged = _profile_dict(measured=parsed["measured_m_s"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "p.json"
        path.write_text(json.dumps(merged), encoding="utf-8")
        assert AxisProfile.load(path).measured_speeds["forward_positive"] == 0.421


# ── the room ────────────────────────────────────────────────────────────────────────────
def test_the_lane_is_checked_against_the_operators_upper_bound():
    plan = plan_for(_BY_NAME["forward_positive"], 0.05, 3)
    assert abs(planned_excursion_m(plan, 2.0, 0.5) - 3.0) < 1e-9


def test_a_short_lane_is_refused_before_anything_moves():
    message = _refuses(lambda: _quiet(lambda: main([
        *_CONTEXT, *_AXIS, "--axis-profile", _written(), "--assume-up-to", "0.8",
        "--lane-metres", "1.0", "--lane-width-metres", "5.0"])))
    assert "--lane-metres says there is 1.0 m" in message


def test_a_narrow_lane_is_refused_against_the_WIDTH_not_the_length():
    """The lateral primitives eat width, and this robot cannot see to either side.

    A room check that measured every primitive against ``--lane-metres`` would wave a
    strafe through a corridor 40 m long and 0.5 m wide, which is the one geometry this
    platform has no sensor for.
    """
    message = _refuses(lambda: _quiet(lambda: main([
        *_CONTEXT, *_AXIS, "--axis-profile", _written(), "--assume-up-to", "0.8",
        "--lane-metres", "40.0", "--lane-width-metres", "0.5"])))
    assert "--lane-width-metres says there is 0.5 m" in message
    assert "lateral" in message


# ── the CLI ─────────────────────────────────────────────────────────────────────────────
def _quiet(callable_):
    with contextlib.redirect_stdout(io.StringIO()):
        return callable_()


_WRITTEN: list = []


def _written(**kwargs) -> str:
    directory = tempfile.mkdtemp()
    path = Path(directory) / "profile.json"
    path.write_text(json.dumps(_profile_dict(**kwargs)), encoding="utf-8")
    _WRITTEN.append(path)
    return str(path)


def test_a_dry_run_opens_no_socket_and_returns_zero():
    code = _quiet(lambda: main([
        *_CONTEXT, *_AXIS, "--axis-profile", _written(), "--assume-up-to", "0.8",
        "--lane-metres", "20", "--lane-width-metres", "8"]))
    assert code == 0


def test_this_probe_is_refused_on_the_transport_that_has_no_primitives():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = run_main(lambda: main([*_CONTEXT, "--assume-up-to", "0.8",
                                      "--lane-metres", "20",
                                      "--lane-width-metres", "8"]), "axis")
    assert code == 2
    assert "carries the commanded magnitude to the wire" in buffer.getvalue()


def test_the_axis_transport_requires_a_profile():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = run_main(lambda: main([*_CONTEXT, *_AXIS, "--assume-up-to", "0.8",
                                      "--lane-metres", "20",
                                      "--lane-width-metres", "8"]), "axis")
    assert code == 2
    assert "requires --axis-profile" in buffer.getvalue()


def test_the_room_and_the_bound_have_no_defaults():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = run_main(lambda: main([*_CONTEXT, *_AXIS,
                                      "--axis-profile", _written()]), "axis")
    assert code == 2
    output = buffer.getvalue()
    for flag in ("--assume-up-to", "--lane-metres", "--lane-width-metres"):
        assert flag in output


def test_the_record_names_the_profile_these_numbers_belong_to():
    """A speed is a property of one profile's raw values, not of the robot in general.

    Re-point the profile at a different raw value and every number here is stale. The
    hash is what stops the numbers being re-attached to a profile they never described --
    the same failure as copying a measurement between the two Ventures, one level down.
    """
    profile_path = _written()
    args = build_parser().parse_args([
        *_CONTEXT, *_AXIS, "--axis-profile", profile_path, "--assume-up-to", "0.8",
        "--lane-metres", "20", "--lane-width-metres", "8"])
    profile = AxisProfile.load(Path(profile_path))
    context = record_context(args, profile, primitives_in(profile))
    assert context["axis_profile_sha256"] == hashlib.sha256(
        Path(profile_path).read_bytes()).hexdigest()
    assert context["primitives"] == [p.name for p in PRIMITIVES]
    # And the commanded magnitude is recorded, because it is arbitrary and a reader has
    # to be able to see that it was not fitted to anything.
    assert context["commanded_m_s"] == 0.05 * COMMAND_MARGIN


def test_a_profile_that_cannot_be_hashed_is_refused_rather_than_recorded_blank():
    message = _refuses(lambda: _profile_sha256("/nonexistent/profile.json"))
    assert "cannot hash axis profile" in message


def test_the_parser_offers_no_default_for_anything_physical():
    defaults = build_parser().parse_args([*_CONTEXT, *_AXIS])
    assert defaults.assume_up_to is None
    assert defaults.lane_metres is None
    assert defaults.lane_width_metres is None
    assert defaults.axis_profile is None


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"axis_primitive_probe: {len(tests)}/{len(tests)} passed")
