#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the static obstacle map.

The tests that earn their place are the ones that separate a MAP from a TRACKER: a
landmark must sharpen rather than drift under repeated sightings, must survive leaving
the field of view, and must still be forgettable when the evidence says it was never
there. Pure numpy — no robot, no OpenCV.

Run: ``python3 test_static_map.py``
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from static_map import (
    ASSOCIATION_GATE_M,
    CONFIRM_SIGHTINGS,
    MAX_MISSES,
    POSITION_SIGMA_FLOOR_M,
    StaticObstacleMap,
)
from tracker import Observation

ORIGIN = (0.0, 0.0, 0.0)


def _map(**kwargs):
    kwargs.setdefault("radii", {"bin": 0.15})
    return StaticObstacleMap(**kwargs)


def _sighting(bearing_deg, range_m, label="bin", pose=ORIGIN):
    return Observation.from_bearing_range(
        math.radians(bearing_deg), range_m, pose[0], pose[1], pose[2], label=label)


def _see(mapping, times, bearing_deg=7.1, range_m=2.15, pose=ORIGIN, start=0.0):
    for index in range(times):
        mapping.observe([_sighting(bearing_deg, range_m, pose=pose)],
                        start + index * 0.14, *pose)


# ── Acquisition ─────────────────────────────────────────────────────────────
def test_one_sighting_is_not_enough_to_plan_against():
    """A single flash of the right colour must not stop the robot."""
    mapping = _map()
    _see(mapping, 1)
    assert len(mapping.landmarks) == 1
    assert mapping.confirmed() == []


def test_two_agreeing_sightings_confirm_a_landmark():
    mapping = _map()
    _see(mapping, CONFIRM_SIGHTINGS)
    assert len(mapping.confirmed()) == 1


def test_a_confirmed_landmark_lands_where_the_measurement_put_it():
    """The staged bin: 2.15 m at +7.1 deg is (2.13, +0.27) in odom."""
    mapping = _map()
    _see(mapping, 4)
    landmark = mapping.confirmed()[0]
    assert abs(landmark.x - 2.134) < 0.03, landmark.x
    assert abs(landmark.y - 0.266) < 0.03, landmark.y


def test_repeated_sightings_sharpen_rather_than_move_it():
    """The point of an information filter: evidence accumulates.

    A tracker would instead read the jitter as velocity, roll it forward 2.5 s and
    swerve around a phantom — which is the reason this module exists.
    """
    mapping = _map()
    _see(mapping, 2)
    early = mapping.landmarks[0].position_sigma
    _see(mapping, 20, start=1.0)
    late = mapping.landmarks[0]
    assert late.position_sigma <= early
    assert abs(late.x - 2.134) < 0.03, "more evidence must not move it"


def test_position_sigma_never_claims_more_precision_than_the_odometry():
    mapping = _map()
    _see(mapping, 200)
    assert mapping.landmarks[0].position_sigma == POSITION_SIGMA_FLOOR_M


def test_the_planning_radius_carries_the_uncertainty():
    mapping = _map()
    _see(mapping, 3)
    landmark = mapping.confirmed()[0]
    assert landmark.planning_radius_m > landmark.radius_m
    assert (landmark.planning_radius_m
            == landmark.radius_m + landmark.position_sigma)


def test_a_bin_is_not_inflated_to_a_persons_footprint():
    """Per-class radii: 0.35 m for a bin is what turns a passable gap into a stop."""
    mapping = _map()
    _see(mapping, 3)
    assert mapping.confirmed()[0].radius_m == 0.15


def test_an_unnamed_label_falls_back_to_the_default_radius():
    mapping = StaticObstacleMap(radii={"bin": 0.15}, default_radius_m=0.4)
    mapping.observe([_sighting(0.0, 2.0, label="crate")], 0.0, *ORIGIN)
    assert mapping.landmarks[0].radius_m == 0.4


# ── Association ─────────────────────────────────────────────────────────────
def test_two_props_far_apart_stay_two_landmarks():
    mapping = _map()
    for index in range(4):
        mapping.observe([_sighting(7.1, 2.15), _sighting(-20.0, 4.2)],
                        index * 0.14, *ORIGIN)
    assert len(mapping.confirmed()) == 2


def test_a_biased_range_still_associates_instead_of_spawning_a_twin():
    """The reason the gate is Euclidean and fixed rather than Mahalanobis.

    After many sightings a landmark's covariance collapses, so a Mahalanobis gate
    collapses with it — and any systematic bias, a size prior a few per cent off say,
    would then fail to associate and spawn a fresh landmark EVERY frame.
    """
    mapping = _map()
    _see(mapping, 30)                       # drive the covariance right down
    before = len(mapping.landmarks)
    for index in range(5):                  # now a 6% biased range, well beyond sigma
        mapping.observe([_sighting(7.1, 2.28)], 5.0 + index * 0.14, *ORIGIN)
    assert len(mapping.landmarks) == before, mapping.landmarks


def test_a_prop_beyond_the_gate_is_a_new_landmark():
    mapping = _map()
    _see(mapping, 3)
    far = 2.15 + ASSOCIATION_GATE_M + 0.15 + 0.3
    mapping.observe([_sighting(7.1, far)], 1.0, *ORIGIN)
    assert len(mapping.landmarks) == 2


def test_labels_do_not_cross_associate():
    mapping = StaticObstacleMap(radii={"bin": 0.15, "crate": 0.2})
    mapping.observe([_sighting(0.0, 2.0, label="bin")], 0.0, *ORIGIN)
    mapping.observe([_sighting(0.0, 2.0, label="crate")], 0.1, *ORIGIN)
    assert len(mapping.landmarks) == 2


def test_the_nearer_of_two_candidates_claims_a_landmark():
    mapping = _map()
    _see(mapping, 3)
    mapping.observe([_sighting(7.1, 2.60), _sighting(7.1, 2.16)], 1.0, *ORIGIN)
    # The close one matched; the far one spawned its own.
    assert len(mapping.landmarks) == 2
    assert abs(mapping.landmarks[0].x - 2.134) < 0.05


# ── Persistence and forgetting ──────────────────────────────────────────────
def test_a_landmark_survives_leaving_the_field_of_view():
    """THE reason this is not a tracker.

    Rounding an obstacle is exactly when it leaves a 120-degree cone, so a tracked bin
    would be pruned at the moment the robot is beside it. Here the robot turns 90 deg
    away and keeps seeing nothing for a long time; the bin must still be there.
    """
    mapping = _map()
    _see(mapping, 3)
    turned_away = (0.0, 0.0, math.radians(90.0))
    for index in range(60):
        mapping.observe([], 10.0 + index * 0.14, *turned_away)
    assert len(mapping.confirmed()) == 1, "a bin does not walk away"


def test_a_landmark_the_camera_is_staring_at_is_eventually_forgotten():
    """Nothing expires on time, but everything expires on disagreement."""
    mapping = _map()
    _see(mapping, 3)
    for index in range(MAX_MISSES + 2):
        mapping.observe([], 10.0 + index * 0.14, *ORIGIN)
    assert mapping.landmarks == []


def test_a_landmark_out_of_range_is_not_penalised():
    mapping = StaticObstacleMap(radii={"bin": 0.15}, max_range_m=3.0)
    _see(mapping, 3, range_m=2.15)
    walked_back = (-4.0, 0.0, 0.0)          # bin now 6.1 m off, past the detector
    for index in range(MAX_MISSES + 5):
        mapping.observe([], 10.0 + index * 0.14, *walked_back)
    assert len(mapping.confirmed()) == 1


def test_a_re_sighting_clears_the_miss_count():
    mapping = _map()
    _see(mapping, 3)
    for index in range(MAX_MISSES - 1):
        mapping.observe([], 10.0 + index * 0.14, *ORIGIN)
    _see(mapping, 1, start=20.0)
    for index in range(MAX_MISSES - 1):
        mapping.observe([], 30.0 + index * 0.14, *ORIGIN)
    assert len(mapping.confirmed()) == 1


def test_a_landmark_created_this_call_is_not_missed_by_it():
    """Ordering: spawning must come after miss-scoring, as in the tracker."""
    mapping = _map()
    mapping.observe([_sighting(7.1, 2.15)], 0.0, *ORIGIN)
    assert mapping.landmarks[0].misses == 0


# ── Occlusion shadows ───────────────────────────────────────────────────────
def test_a_landmark_casts_a_shadow_on_its_own_bearing():
    mapping = _map()
    _see(mapping, 3)
    shadows = mapping.occluders(*ORIGIN)
    assert len(shadows) == 1
    bearing, half_angle, range_m = shadows[0]
    assert abs(math.degrees(bearing) - 7.1) < 1.0
    assert abs(range_m - 2.15) < 0.05
    # 0.23 m of planning radius at 2.15 m is about 6 degrees of half-angle.
    assert 0.03 < half_angle < 0.20, half_angle


def test_an_unconfirmed_landmark_casts_no_shadow():
    """A shadow silences misses, so an unproven one would hide people for free."""
    mapping = _map()
    _see(mapping, 1)
    assert mapping.occluders(*ORIGIN) == []


def test_standing_inside_a_landmark_casts_no_shadow():
    """asin(radius/distance) is undefined there, and a half-angle of pi would silence
    every miss in the scene rather than one bearing."""
    mapping = _map()
    _see(mapping, 3)
    on_top_of_it = (2.134, 0.266, 0.0)
    assert mapping.occluders(*on_top_of_it) == []


def test_a_wider_uncertainty_casts_a_wider_shadow():
    sharp = _map()
    _see(sharp, 30)
    fuzzy = _map()
    _see(fuzzy, CONFIRM_SIGHTINGS)
    assert fuzzy.occluders(*ORIGIN)[0][1] > sharp.occluders(*ORIGIN)[0][1]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"static_map: {len(tests)}/{len(tests)} passed")
