#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Go2 deploy joint-order utilities. No DDS, no robot.

The load-bearing Go2-specific logic here is the SDK(per-leg)↔Isaac(per-level) joint
reorder — a wrong permutation commands the wrong motors, so it is tested directly.

Run: ``python3 test_go2_robot_io.py``
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_robot_io import (  # noqa: E402
    GO2_JOINT_ORDER, GO2_ISAAC_JOINT_ORDER, NUM_MOTORS, reorder,
)


def test_orders_are_same_12_joint_set():
    assert len(GO2_JOINT_ORDER) == NUM_MOTORS == 12
    assert set(GO2_JOINT_ORDER) == set(GO2_ISAAC_JOINT_ORDER)
    assert len(set(GO2_JOINT_ORDER)) == 12  # no duplicates


def test_sdk_order_is_per_leg():
    # first three entries are one leg's hip/thigh/calf (per-leg grouping)
    assert GO2_JOINT_ORDER[:3] == ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"]


def test_isaac_order_is_per_level():
    # first four entries are all four hips (per-level grouping)
    assert GO2_ISAAC_JOINT_ORDER[:4] == [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]


def test_reorder_isaac_to_sdk_roundtrip():
    # give each joint a value equal to its SDK index, in Isaac order, then reorder to
    # SDK order — the result must be [0, 1, 2, ...].
    sdk_index = {name: i for i, name in enumerate(GO2_JOINT_ORDER)}
    isaac_vec = np.array([sdk_index[name] for name in GO2_ISAAC_JOINT_ORDER], dtype=float)
    sdk_vec = reorder(isaac_vec, GO2_ISAAC_JOINT_ORDER, GO2_JOINT_ORDER)
    assert np.allclose(sdk_vec, np.arange(12))


def test_reorder_is_invertible():
    v = np.arange(12, dtype=float) + 0.5
    there = reorder(v, GO2_JOINT_ORDER, GO2_ISAAC_JOINT_ORDER)
    back = reorder(there, GO2_ISAAC_JOINT_ORDER, GO2_JOINT_ORDER)
    assert np.allclose(back, v)


def test_reorder_rejects_mismatched_sets():
    try:
        reorder(np.zeros(12), GO2_JOINT_ORDER, ["x"] * 12)
    except ValueError:
        return
    raise AssertionError("reorder must reject a different joint set")


def test_reorder_rejects_wrong_length():
    try:
        reorder(np.zeros(6), GO2_JOINT_ORDER, GO2_ISAAC_JOINT_ORDER)
    except ValueError:
        return
    raise AssertionError("reorder must reject a wrong-length vector")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"go2_robot_io: {len(tests)}/{len(tests)} passed")
