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
import numpy as np

from static_map import (
    ASSOCIATION_GATE_M,
    CONFIRM_SIGHTINGS,
    MAX_LANDMARKS_PER_LABEL,
    MAX_MISSES,
    MAX_PLANNING_SIGMA_M,
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


# ── Guards learned from a failed live run ───────────────────────────────────
def test_a_landmark_the_map_is_unsure_of_is_not_planned_against():
    """Measured live: a landmark reached sigma 0.52 m, so the planner saw a 0.69 m disc.

    Once the uncertainty is that wide the landmark has stopped carrying direction as
    well as position — it blocks a bearing the robot could have taken exactly as hard as
    one it could not — so planning against it is worse than ignoring it.
    """
    mapping = _map()
    _see(mapping, CONFIRM_SIGHTINGS)
    landmark = mapping.landmarks[0]
    assert mapping.confirmed() == [landmark], "a tight landmark is planned against"
    landmark.covariance = np.eye(2) * (MAX_PLANNING_SIGMA_M + 0.2) ** 2
    assert mapping.confirmed() == [], "a vague one is not"
    assert mapping.landmarks == [landmark], "but it is KEPT, and may re-converge"


def test_one_prop_cannot_be_planned_against_in_four_places():
    """Measured live: four landmarks for one bin, up to 5.6 m away, boxing the robot in.

    Odometry that reads "stationary" while the robot is physically dragged projects every
    sighting to a different odom point, and each becomes its own landmark.
    """
    mapping = _map()
    for bearing in (7.1, -25.0, 40.0, -45.0):
        _see(mapping, CONFIRM_SIGHTINGS, bearing_deg=bearing, range_m=2.15)
    assert len(mapping.landmarks) == 4, "the map still holds what it saw"
    assert len(mapping.confirmed()) == MAX_LANDMARKS_PER_LABEL


def test_the_best_evidenced_duplicate_is_the_one_kept():
    """A duplicate spawned by bad odometry has few sightings and a wide covariance; the
    real landmark has many and a tight one. Rank on that, not on arrival order."""
    mapping = _map()
    _see(mapping, 2, bearing_deg=-30.0)          # the ghost, seen twice
    _see(mapping, 12, bearing_deg=7.1)           # the real bin, seen twelve times
    _see(mapping, 2, bearing_deg=35.0)           # another ghost
    confirmed = mapping.confirmed()
    assert len(confirmed) == MAX_LANDMARKS_PER_LABEL
    best = max(confirmed, key=lambda lm: lm.sightings)
    assert best.sightings == 12, best.sightings
    assert abs(best.x - 2.134) < 0.05, "the real bin must survive the cull"


def test_the_cap_is_per_label_not_global():
    """A bin and a crate compete for their own slots, not with each other."""
    mapping = StaticObstacleMap(radii={"bin": 0.15, "crate": 0.2})
    # All six in EVERY cycle: observing them in separate passes would leave the others
    # unmatched-but-visible each time, and they would be missed to death rather than
    # capped, which is a different rule and not the one under test.
    for index in range(6):
        mapping.observe(
            [_sighting(b, 2.2) for b in (7.1, -25.0, 40.0)]
            + [_sighting(b, 2.2, label="crate") for b in (14.0, -33.0, 47.0)],
            index * 0.14, *ORIGIN)
    labels = [lm.label for lm in mapping.confirmed()]
    assert labels.count("bin") == MAX_LANDMARKS_PER_LABEL, labels
    assert labels.count("crate") == MAX_LANDMARKS_PER_LABEL, labels


def test_a_far_landmark_still_confirms_once_it_has_been_seen_enough():
    """The sigma gate must not permanently exclude anything acquired at range.

    Range sigma is proportional to distance, so far landmarks start vague. One first seen
    at 4.5 m is above the cap on two sightings and has to earn its way in — otherwise the
    robot walks at obstacles it can see perfectly well, just because it noticed them
    early. The staged bin at 2.15 m is admitted immediately, which is the case that
    matters and is covered above.
    """
    mapping = _map()
    _see(mapping, CONFIRM_SIGHTINGS, range_m=4.5)
    assert mapping.confirmed() == [], "two sightings at 4.5 m is not yet certain enough"
    _see(mapping, 10, range_m=4.5, start=1.0)
    assert len(mapping.confirmed()) == 1, "more evidence must let it in"


def test_confirmed_preserves_map_order():
    """The overlay and the log read this list; ranking order would make them jump."""
    mapping = _map()
    _see(mapping, 3, bearing_deg=-20.0)
    _see(mapping, 9, bearing_deg=7.1)
    confirmed = mapping.confirmed()
    positions = [mapping.landmarks.index(lm) for lm in confirmed]
    assert positions == sorted(positions), positions


def test_a_landmark_survives_the_dropout_that_deleted_one_on_hardware():
    """The live stall of 2026-08-19, as a regression test.

    The colour detector loses a bin as the robot closes on it — at 0.71 m a 0.3 m bin
    fills 51% of the frame, is clipped by its bottom edge, and falls outside `min_fill`
    and the aspect band. ``is_visible`` still calls it visible, correctly, so a miss is
    scored on every one of those cycles. At the old MAX_MISSES of 8 that deleted a
    landmark with 32 sightings of converged evidence after about 1.2 s, and the next
    detection spawned a fresh one 0.216 m away carrying an unconverged 0.518 m radius.
    The gap between two bins collapsed from 0.23 m to 0.06 m and the run stalled.

    Twelve blind cycles is ~1.7 s at the measured 5-7 Hz, longer than any dropout seen
    before this one and comfortably past the old limit. Set MAX_MISSES back to 8 and the
    landmark is gone and the identity with it.
    """
    mapping = _map()
    _see(mapping, 30)
    landmark = mapping.landmarks[0]
    identity, radius = landmark.landmark_id, landmark.planning_radius_m

    for index in range(12):
        mapping.observe([], 10.0 + index * 0.14, *ORIGIN)

    assert len(mapping.landmarks) == 1, "a converged landmark must outlive the dropout"
    assert mapping.landmarks[0].landmark_id == identity, "and keep its identity"
    assert math.isclose(mapping.landmarks[0].planning_radius_m, radius), \
        "and its converged radius, which is what the identity was protecting"


def test_the_re_sighting_after_a_dropout_does_not_spawn_a_duplicate():
    """The consequence, stated the way the hardware showed it: one bin, two landmarks.

    Association is on plain euclidean distance within ASSOCIATION_GATE_M, so a
    re-sighting fuses — but only into a landmark that still EXISTS. Deletion is what
    turned a fusable observation into a second bin.
    """
    mapping = _map()
    _see(mapping, 30)
    identity = mapping.landmarks[0].landmark_id
    radius = mapping.landmarks[0].planning_radius_m
    for index in range(12):
        mapping.observe([], 10.0 + index * 0.14, *ORIGIN)
    _see(mapping, 4, start=12.0)

    # COUNTING LANDMARKS CANNOT SEE THIS BUG. Delete the landmark and the re-sighting
    # spawns exactly one replacement, so the count reads 1 either way and the test passes
    # while the failure is fully present. The identity is what changes, and the radius
    # that came back with it is what stalled the robot.
    assert len(mapping.landmarks) == 1, "the same bin must not become two"
    assert mapping.landmarks[0].landmark_id == identity, \
        "a re-sighting must fuse into the landmark, not replace it"
    # Compared against what a REPLACEMENT would actually have carried, not a tolerance
    # picked to pass. The surviving landmark's radius does grow across the blind period
    # — uncertainty should grow while nothing is observed — but it stays near its
    # converged value instead of reopening to a fresh prior. On hardware the replacement
    # arrived at 0.518 m and collapsed a 0.23 m gap to 0.06 m.
    replacement = _map()
    _see(replacement, 2)
    fresh = replacement.landmarks[0].planning_radius_m
    settled = mapping.landmarks[0].planning_radius_m
    assert settled - radius < 0.05, "the survivor stays near its converged radius"
    assert settled < 0.5 * (radius + fresh), \
        f"a survivor at {settled:.3f} m must not look like a fresh spawn at {fresh:.3f} m"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"static_map: {len(tests)}/{len(tests)} passed")
