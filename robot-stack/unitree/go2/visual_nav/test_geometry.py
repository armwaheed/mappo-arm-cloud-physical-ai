#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the shared angle helper.

Small, but it pins the property four call sites depend on: that a heading difference
straddling +-pi comes back as the SHORT way round. Getting that wrong reads a 10 deg
sweep as 350 deg, which is what ``yaw_span_deg`` exists to guard against and what
``_return_to_yaw`` would turn the long way to correct.

Run: ``python3 test_geometry.py``
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import wrap_pi


def test_wraps_into_the_interval():
    for angle, expected in ((0.0, 0.0), (math.pi / 2, math.pi / 2),
                            (-math.pi / 2, -math.pi / 2), (2.0 * math.pi, 0.0),
                            (5.0 * math.pi / 2, math.pi / 2)):
        got = float(wrap_pi(angle))
        assert abs(got - expected) < 1e-9, (angle, got, expected)


def test_the_antipode_lands_on_pi_with_either_sign():
    """At exactly +-pi the sign is not contractual, and callers must not rely on it.

    ``arctan2`` decides it from a sine that is ~1e-16 rather than 0, so an odd multiple
    of pi can come back as either end of the range. Every caller here takes ``abs()`` or
    compares a magnitude, so this is documented rather than forced.
    """
    for angle in (math.pi, -math.pi, 3.0 * math.pi, -3.0 * math.pi):
        assert abs(abs(float(wrap_pi(angle))) - math.pi) < 1e-9, angle


def test_the_branch_cut_takes_the_short_way_round():
    """The property every caller actually relies on.

    Two headings 10 deg apart but sitting either side of +-pi differ by 10 deg, not by
    350. A raw subtraction says 350 and would send `_return_to_yaw` the long way round
    — with the robot on an Ethernet tether.
    """
    a, b = math.radians(175.0), math.radians(-175.0)
    assert abs(math.degrees(float(wrap_pi(b - a))) - 10.0) < 1e-9
    assert abs(math.degrees(float(wrap_pi(a - b))) + 10.0) < 1e-9


def test_arrays_are_wrapped_elementwise():
    """The planner wraps a whole candidate set at once; the tracker wraps scalars."""
    angles = np.array([0.0, 3.0 * math.pi, -3.0 * math.pi, math.pi / 4])
    wrapped = wrap_pi(angles)
    assert wrapped.shape == angles.shape
    assert np.all(wrapped > -math.pi - 1e-9) and np.all(wrapped <= math.pi + 1e-9)
    assert abs(float(wrapped[3]) - math.pi / 4) < 1e-9


def test_wrapping_is_idempotent():
    """Already-wrapped angles must survive untouched — callers wrap defensively."""
    for angle in (-3.0, -1.0, 0.0, 1.0, 3.0):
        once = float(wrap_pi(angle))
        assert abs(float(wrap_pi(once)) - once) < 1e-12, angle


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"geometry: {len(tests)}/{len(tests)} passed")
