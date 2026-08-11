# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Angle arithmetic shared by the planner, the tracker, the map and the calibration tool.

Small, but it earns a module of its own because modules that must not import each other
all need it: ``avoidance`` (a planner), ``tracker`` (a filter), ``static_map`` (a map)
and ``calibrate_camera`` (a tool). Routing any of those through another to borrow a
helper would put an edge in the dependency graph the design does not want — a camera
calibration script has no business importing a motion planner, and the tracker and the
map must be able to occlude each other in both directions without one owning the other.

It was previously hand-inlined as ``atan2(sin(x), cos(x))`` at four call sites. That is
correct but easy to get subtly wrong, and it was: ``collect_spin`` and ``_return_to_yaw``
both wrote ``atan2(sin(f() - a), cos(f() - a))``, calling the pose accessor TWICE. On a
turning robot the two calls return different headings, so the sine came from one instant
and the cosine from another — an error that only appears while moving, which is the only
time either function runs.

Pure numpy. Tests: ``python3 test_geometry.py``.
"""

from __future__ import annotations

import math

import numpy as np


def angular_shadow(robot_x: float, robot_y: float, robot_yaw: float,
                   x: float, y: float, radius_m: float
                   ) -> tuple[float, float, float] | None:
    """The cone a disc hides behind it: ``(bearing_rad, half_angle_rad, range_m)``.

    Bearing is relative to the robot's nose. Anything FURTHER than ``range_m`` and
    within ``half_angle_rad`` of ``bearing_rad`` is hidden by this disc.

    Returns ``None`` when the robot is inside the disc. That case has no meaningful
    answer — the disc subtends everything — and returning a half-angle of pi would
    silence every miss in the scene rather than one bearing.

    Lives here because occlusion runs BOTH ways and neither side may own the other: a
    mapped bin hides a person from the tracker, and a person standing in front of the
    bin hides it from the map. ``static_map`` importing ``tracker`` is already an edge;
    the reverse would close a cycle.
    """
    dx, dy = x - robot_x, y - robot_y
    distance = math.hypot(dx, dy)
    if distance <= radius_m:
        return None
    return (float(wrap_pi(math.atan2(dy, dx) - robot_yaw)),
            math.asin(radius_m / distance),
            distance)


def hidden_by(bearing_rad: float, range_m: float, shadows) -> bool:
    """Whether a target at this bearing and range falls inside any shadow.

    ``shadows`` are :func:`angular_shadow` triples. The range test is strict: something
    in FRONT of an occluder is in plain sight, and something at the same range is beside
    it rather than behind it.
    """
    for shadow_bearing, half_angle, shadow_range in shadows:
        if range_m <= shadow_range:
            continue
        if abs(wrap_pi(bearing_rad - shadow_bearing)) <= half_angle:
            return True
    return False


def wrap_pi(angle):
    """Wrap angle(s) in radians to ``[-pi, pi]``.

    At exactly +-pi the sign is not contractual: ``arctan2`` decides it from a sine of
    order 1e-16 rather than 0, so an odd multiple of pi may come back either way. Every
    caller here compares a magnitude, so this is documented rather than forced.

    Accepts a scalar or an array and returns the matching shape — the planner wraps a
    whole candidate set at once, while the tracker and the calibration sweep wrap
    single headings.

    ``atan2(sin, cos)`` rather than ``angle - 2*pi*round(angle/(2*pi))`` because it is
    exact at the branch cut and needs no special case for negative input. Note that the
    argument is evaluated ONCE by the caller and passed in; that is the whole reason
    this is a function rather than an idiom (see the module docstring).
    """
    return np.arctan2(np.sin(angle), np.cos(angle))
