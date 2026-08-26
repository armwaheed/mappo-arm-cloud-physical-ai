#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 motor-temperature probe.

Three properties, and each one has a way of going wrong that would look fine in a report:

* **A silent link must not be reported as an absent channel.** Both look like "no
  temperatures arrived", and only one of them settles the vendor question.
* **The twelve-element decoy must not be mistaken for temperatures.** ``JointState``
  carries exactly twelve doubles, and they are joint angles in radians. A shape-based
  search finds them and reports a robot whose motors are all at 0.4 degrees.
* **A partial set must be refused.** A thermal gate fed eleven of twelve motors has a hole
  in it, and the hole is invisible once the number is written down.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning.measurement import Refusal
from deep_robotics.lite3.commissioning.motor_temperature_probe import (
    DECOY_FIELD,
    MOTOR_COUNT,
    normalise,
    scan_frames,
    verdict,
)

_JOINT_NAMES = tuple(f"J{index}" for index in range(12))


def _robot_state(**extra):
    frame = {"kind": "robot_state", "code": 2305, "size": 208, "cons_code": 0,
             "timestamp_s": 1.0, "battery_level": 87.5, "error_state": 0,
             "vel_body": [0.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]}
    frame.update(extra)
    return frame


def _joint_state(values=None):
    values = values if values is not None else [0.4] * 12
    return {"kind": "joint_state", "code": 2306, "size": 96, "cons_code": 0,
            "timestamp_s": 1.0,
            DECOY_FIELD: dict(zip(_JOINT_NAMES, values))}


# ── the scan ────────────────────────────────────────────────────────────────────────────
def test_the_scan_reports_the_wire_fields_and_not_the_probes_own_bookkeeping():
    evidence = scan_frames([_robot_state(), _joint_state()])
    assert "battery_level" in evidence["field_names"]
    for bookkeeping in ("kind", "code", "size", "cons_code", "timestamp_s"):
        assert bookkeeping not in evidence["field_names"], bookkeeping


def test_the_scan_counts_frames_by_kind():
    evidence = scan_frames([_robot_state(), _robot_state(), _joint_state()])
    assert evidence["frames"] == 3
    assert evidence["kinds"] == {"robot_state": 2, "joint_state": 1}


def test_the_scan_records_the_decoy_range_so_radians_can_be_told_from_celsius():
    evidence = scan_frames([_joint_state([-1.2] + [0.4] * 11)])
    assert evidence["decoy_range"] == [-1.2, 0.4]


# ── the verdict ─────────────────────────────────────────────────────────────────────────
def test_twelve_joint_angles_are_not_reported_as_twelve_temperatures():
    """A shape-based search would find this field and report motors at 0.4 degrees."""
    result = verdict(scan_frames([_robot_state(), _joint_state()]))
    assert result["channel"] == "absent"
    assert result["temperatures_c"] is None
    assert DECOY_FIELD in result["evidence"]["field_names"]


def test_a_flowing_stream_with_no_temperature_field_is_a_finding():
    result = verdict(scan_frames([_robot_state() for _ in range(50)]))
    assert result["channel"] == "absent"
    assert result["evidence"]["frames"] == 50


def test_a_silent_link_settles_nothing_and_is_refused():
    """Otherwise a disconnected laptop 'proves' the vendor question for us."""
    try:
        verdict(scan_frames([]))
    except Refusal as refusal:
        assert "network.toml" in str(refusal)
    else:
        raise AssertionError("no frames at all must refuse, not report absence")


def test_a_documented_temperature_field_is_read_when_it_exists():
    values = [30.0 + index for index in range(12)]
    frame = _robot_state(motor_temperatures=values)
    result = verdict(scan_frames([frame]))
    assert result["channel"] == "present"
    assert result["source"] == "udp:motor_temperatures"
    assert result["temperatures_c"] == values


def test_a_ros_reading_takes_precedence_and_is_labelled_as_such():
    result = verdict(scan_frames([_robot_state()]), [40.0] * 12)
    assert result["channel"] == "present"
    assert result["source"] == "ros:/motor_temperatures"


# ── the partial-set refusal ─────────────────────────────────────────────────────────────
def test_eleven_values_are_refused_rather_than_padded():
    try:
        normalise([30.0] * 11)
    except Refusal as refusal:
        assert "11 values" in str(refusal)
    else:
        raise AssertionError("a partial thermal set must refuse")


def test_thirteen_values_are_refused_too():
    try:
        normalise([30.0] * 13)
    except Refusal:
        return
    raise AssertionError("more than twelve is as wrong as fewer")


def test_a_non_finite_value_is_refused():
    try:
        normalise([30.0] * 11 + [math.nan])
    except Refusal as refusal:
        assert "non-finite" in str(refusal)
    else:
        raise AssertionError("NaN cannot exceed an abort limit and must not be accepted")


def test_a_non_numeric_field_is_refused():
    try:
        normalise(["warm"] * MOTOR_COUNT)
    except Refusal:
        return
    raise AssertionError("a non-numeric temperature field must refuse")


def test_exactly_twelve_finite_values_are_accepted():
    assert normalise([f"{value}" for value in range(12)]) == [float(v) for v in range(12)]


def test_a_partial_ros_reading_refuses_the_whole_verdict():
    try:
        verdict(scan_frames([_robot_state()]), [40.0] * 6)
    except Refusal:
        return
    raise AssertionError("six ROS values must not become a present channel")


def test_the_probe_module_contains_no_send_path():
    """Structural, same idiom as test_lite3_state_probe: this must stay receive-only."""
    import ast

    forbidden = {"send", "sendto", "sendall", "sendmsg", "connect", "connect_ex",
                 "set_velocity"}
    tree = ast.parse((_HERE / "motor_temperature_probe.py").read_text(encoding="utf-8"))
    offenders = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr in forbidden]
    assert offenders == [], f"the temperature probe gained a transmit path: {offenders}"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"motor_temperature_probe: {len(tests)}/{len(tests)} passed")
