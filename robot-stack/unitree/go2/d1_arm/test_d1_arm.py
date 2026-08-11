#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the D1 arm command protocol + joint clamping. No DDS, no arm.

Exercises the pure :class:`ArmProtocol` encoder, the feedback parser, and the joint
clamps against the envelope the FIRMWARE enforces.

TWO CONVENTIONS ARE PINNED HERE BECAUSE GETTING EITHER WRONG IS SILENT. Both were
corrected on hardware after the first binding was written against the spec sheet, and
both produce a command the arm accepts and ignores rather than an error:

  * **Angles are DEGREES.** The first binding used radians. Feeding it a radian value
    commands a number ~57x too small — a 3 deg correction becomes 0.05 deg, which is
    below this arm's 0.1 deg reporting resolution. It looks exactly like "the arm
    ignored my command".
  * **Joint ids are 0-BASED** (J0 base-yaw .. J5 wrist-roll, 6 = jaw). The first binding
    was 1-based, so every command landed on the NEXT joint along. Asking for base yaw
    moved the shoulder.

Together they turned a "recentre the base yaw by 3 degrees" command into "move the
shoulder by 0.05 degrees", and the arm dutifully did nothing observable.

Run: ``python3 test_d1_arm.py``
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_arm import (  # noqa: E402
    DAMP_ENABLE,
    DAMP_RELEASE,
    GRIPPER_INDEX,
    GRIPPER_LIMIT_DEG,
    JOINT_LIMITS_DEG,
    NUM_SERVOS,
    ArmProtocol,
    clamp_gripper,
    clamp_joint,
    parse_feedback,
)


def _decode(data_str):
    return json.loads(data_str)


# ── The two conventions that were wrong ────────────────────────────────────────
def test_angles_are_degrees_not_radians():
    """A radian-valued API would clamp 135.0 as out of range; a degree one passes it."""
    assert clamp_joint(0, 135.0) == 135.0, "135 deg is IN range for J0"
    assert clamp_joint(0, 200.0) == 135.0, "and 200 deg is the clamp"
    # The tell-tale: under the old radian API, pi rad would have been the whole envelope.
    assert JOINT_LIMITS_DEG[0] == (-135.0, 135.0)


def test_joint_ids_are_zero_based_and_j0_is_the_base_yaw():
    """J0 is base yaw — the sway axis safety.STOWED_YAW_DEG gates.

    Under the old 1-based ids, asking for joint 0 was an error and asking for joint 1
    moved the shoulder while the caller believed it was correcting the base yaw.
    """
    assert 0 in JOINT_LIMITS_DEG, "J0 must be addressable"
    assert 6 not in JOINT_LIMITS_DEG, "6 is the jaw, not a joint"
    assert GRIPPER_INDEX == 6
    assert JOINT_LIMITS_DEG[0] == (-135.0, 135.0), "J0 base yaw, the widest joint"
    assert JOINT_LIMITS_DEG[1] == (-90.0, 90.0), "J1 shoulder is narrower"
    d = _decode(ArmProtocol.set_joint(0, 3.0))
    assert d["data"]["id"] == 0, "the id must go out as given, not shifted"


def test_damp_mode_is_binary_not_a_stiffness():
    """funcode 4/5 ``mode`` is 0 (discharge) or 1 (enable), NOT a 0..80000 stiffness.

    The first binding sent 1000 as "hold", which is outside the accepted range. A latch
    issued that way does not latch — and an arm that was not going to move anyway shows
    zero drift either way, so the failure is invisible to a drift check.
    """
    assert (DAMP_RELEASE, DAMP_ENABLE) == (0, 1)
    assert _decode(ArmProtocol.enable_all(DAMP_ENABLE))["data"] == {"mode": 1}
    assert _decode(ArmProtocol.enable_all(DAMP_RELEASE))["data"] == {"mode": 0}


# ── ArmProtocol encoding ───────────────────────────────────────────────────────
def test_reset_has_no_data_field():
    d = _decode(ArmProtocol.reset())
    assert d["funcode"] == 7 and d["address"] == 1 and "data" not in d


def test_set_joint_shape():
    d = _decode(ArmProtocol.set_joint(3, 12.0, delay_ms=50, seq=9))
    assert d["funcode"] == 1 and d["seq"] == 9
    assert d["data"] == {"id": 3, "angle": 12.0, "delay_ms": 50.0}


def test_set_all_joints_has_seven_angles():
    angles = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]
    d = _decode(ArmProtocol.set_all_joints(angles, mode=1))
    assert d["funcode"] == 2 and d["data"]["mode"] == 1
    for i, a in enumerate(angles):
        assert d["data"][f"angle{i}"] == a
    assert "angle6" in d["data"] and "angle7" not in d["data"]


def test_set_all_joints_rejects_wrong_count():
    try:
        ArmProtocol.set_all_joints([0.0] * 6)  # missing the jaw
    except ValueError:
        return
    raise AssertionError("set_all_joints must require exactly 7 angles")


def test_enable_all_shape():
    d = _decode(ArmProtocol.enable_all(DAMP_ENABLE))
    assert d["funcode"] == 5 and d["data"] == {"mode": 1}


def test_emitted_json_is_reparseable():
    # every command must be valid JSON the arm can json.loads (no NaN/trailing junk)
    for s in (ArmProtocol.reset(), ArmProtocol.set_joint(0, 0.0),
              ArmProtocol.set_all_joints([0.0] * 7),
              ArmProtocol.enable_joint(2, DAMP_RELEASE)):
        json.loads(s)


# ── feedback parsing ───────────────────────────────────────────────────────────
def test_parse_feedback_tags_joint_angles():
    """(2,1) is the frame that actually arrives most of the time — ArmStowMonitor
    filters on exactly this, because feedback() returns whatever came last."""
    fb = parse_feedback('{"address":2,"funcode":1,"data":{"angle0":1.5}}')
    assert fb["kind"] == "joint_angles"
    assert fb["data"]["angle0"] == 1.5


def test_parse_feedback_tags_exec_ack():
    fb = parse_feedback('{"address":3,"funcode":2,"data":1}')
    assert fb["kind"] == "exec_ack" and fb["data"] == 1


def test_parse_feedback_unknown_passthrough():
    fb = parse_feedback('{"address":9,"funcode":9}')
    assert fb["kind"] == "unknown" and fb["address"] == 9


# ── clamping ───────────────────────────────────────────────────────────────────
def test_clamp_joint_respects_firmware_limits():
    assert clamp_joint(1, 200.0) == JOINT_LIMITS_DEG[1][1]
    assert clamp_joint(1, -200.0) == JOINT_LIMITS_DEG[1][0]
    assert clamp_joint(1, 12.0) == 12.0, "in-range passes through"


def test_clamp_gripper_bounded():
    assert clamp_gripper(1000.0) == GRIPPER_LIMIT_DEG[1]
    assert clamp_gripper(-1000.0) == GRIPPER_LIMIT_DEG[0]


def test_num_servos_is_seven():
    assert NUM_SERVOS == 7  # 6 joints + jaw


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"d1_arm: {len(tests)}/{len(tests)} passed")
