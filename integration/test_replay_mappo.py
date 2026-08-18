#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the replay tool's own measurements.

``replay_mappo.py`` produces the numbers this project quotes as evidence, and it had no
tests at all until issue #17 showed one of those numbers was measuring the wrong object.
A tool that generates evidence needs its measurements pinned at least as tightly as the
code it measures.

These use a stub controller rather than the real checkpoint: :func:`policy_sight` is pure
arithmetic over an observation vector, and a test whose expected value depends on what a
set of weights happens to produce is not a test of the arithmetic.

Run: ``python3 test_replay_mappo.py``
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_mappo import OBSERVATION_STATE_WIDTH, policy_sight


@dataclass(frozen=True)
class _Config:
    lidar_range_vmas: float = 0.35
    meters_per_vmas_unit: float = 2.5

    @property
    def lidar_range_m(self) -> float:
        return self.lidar_range_vmas * self.meters_per_vmas_unit


class _Controller:
    """Only the two attributes :func:`policy_sight` is allowed to read."""

    def __init__(self, observation) -> None:
        self.last_observation = observation
        self.cfg = _Config()


def _observation(lidar, state=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
    """A full observation: six state entries then the fan, as the checkpoint records."""
    return list(state) + list(lidar)


# ── Visibility comes from the observation, not from anywhere else ───────────
def test_an_empty_fan_is_not_visible():
    visible, surface = policy_sight(_Controller(_observation([0.0] * 12)))
    assert visible is False
    assert surface == float("inf")


def test_a_controller_that_has_not_stepped_is_not_visible():
    """``last_observation`` is ``None`` before the first step and on a tick with no goal.
    Returning "visible" there would invent an obstacle out of missing data."""
    visible, surface = policy_sight(_Controller(None))
    assert visible is False
    assert surface == float("inf")


def test_a_positive_ray_is_visible_and_inverts_the_proximity_encoding():
    """``lidar`` is PROXIMITY — ``lidar_range_vmas - range_vmas`` — so the LARGEST entry
    is the NEAREST thing, and the surface range is the encoding read backwards.

    0.15 of proximity against a 0.35 range leaves 0.20 VMAS of range, which at 2.5 m per
    unit is 0.50 m. Getting the sign of this backwards yields a plausible number rather
    than an error, which is why it is pinned.
    """
    fan = [0.0] * 12
    fan[3] = 0.15
    visible, surface = policy_sight(_Controller(_observation(fan)))
    assert visible is True
    assert math.isclose(surface, 0.50, abs_tol=1e-9), surface


def test_the_nearest_ray_wins_not_the_first_or_the_last():
    fan = [0.0] * 12
    fan[1], fan[7] = 0.05, 0.25          # 0.25 is the closer one
    _, surface = policy_sight(_Controller(_observation(fan)))
    assert math.isclose(surface, (0.35 - 0.25) * 2.5, abs_tol=1e-9), surface


def test_the_state_entries_are_not_read_as_rays():
    """The fan starts at :data:`OBSERVATION_STATE_WIDTH`. Slicing at the wrong offset
    would read the robot's own position — routinely much larger than any proximity — as
    the nearest obstacle, and report a confident, wrong range instead of failing."""
    state = (99.0, 99.0, 0.0, 0.0, 99.0, 99.0)
    visible, surface = policy_sight(_Controller(_observation([0.0] * 12, state=state)))
    assert visible is False, "position entries must not count as a detection"
    assert surface == float("inf")


def test_the_state_width_matches_the_checkpoints_recorded_layout():
    """``[x, y, vx, vy, x-gx, y-gy, *lidar]`` — six state entries. Pinned so a checkpoint
    with a different layout fails here rather than silently mis-slicing."""
    assert OBSERVATION_STATE_WIDTH == 6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"replay_mappo: {len(tests)}/{len(tests)} passed")
