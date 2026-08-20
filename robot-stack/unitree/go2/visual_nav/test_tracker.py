#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the moving-obstacle tracker.

Synthetic measurements only. The important ones:

  * **Ego-motion rejection.** A stationary person, measured from a robot that is
    turning hard, must come out with ~zero velocity. Tracking in the body frame would
    give them about a metre per second of phantom sideways motion — which is exactly
    the signal the planner would swerve on, so this test is the reason the tracker
    works in odom at all.
  * **Coasting out of view.** A person who leaves the field of view must survive, not
    be deleted for being invisible.

Run: ``python3 test_tracker.py``
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import (
    COAST_TIMEOUT_S,
    MAX_MISSES,
    UNMEASURED_SOURCE_SIGMA_SCALE,
    WIDTH_SOURCE_SIGMA_SCALE,
    Observation,
    ObstacleTracker,
    _range_sigma_scale,
    observation_from,
)

DT = 0.15  # a realistic perception period
ORIGIN = (0.0, 0.0, 0.0)


def _observe(tracker: ObstacleTracker, person_xy, robot_pose, now):
    """Feed one perfect measurement of a person at ``person_xy`` from ``robot_pose``."""
    robot_x, robot_y, robot_yaw = robot_pose
    dx, dy = person_xy[0] - robot_x, person_xy[1] - robot_y
    range_m = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - robot_yaw
    obs = Observation.from_bearing_range(bearing, range_m, robot_x, robot_y, robot_yaw)
    tracker.update([obs], now, robot_x, robot_y, robot_yaw)


def test_stationary_person_has_no_velocity():
    tracker = ObstacleTracker()
    now = 0.0
    for _ in range(15):
        tracker.predict(DT)
        now += DT
        _observe(tracker, (3.0, 0.0), (0.0, 0.0, 0.0), now)
    track = tracker.confirmed_tracks()[0]
    assert track.speed < 0.15, f"phantom velocity {track.speed:.3f} m/s"


def test_ego_rotation_does_not_create_phantom_velocity():
    """THE test: the robot spins, the person does not move, the estimate must agree."""
    tracker = ObstacleTracker()
    now, yaw = 0.0, 0.0
    for _ in range(20):
        tracker.predict(DT)
        now += DT
        yaw += 0.6 * DT                      # 0.6 rad/s — a brisk turn
        _observe(tracker, (3.0, 0.0), (0.0, 0.0, yaw), now)
    track = tracker.confirmed_tracks()[0]
    # In the BODY frame this person sweeps ~1.8 m/s across the view. In odom: nothing.
    assert track.speed < 0.20, f"ego-motion leaked in as {track.speed:.3f} m/s"
    assert abs(track.state[0] - 3.0) < 0.25 and abs(track.state[1]) < 0.25


def test_ego_translation_does_not_create_phantom_velocity():
    tracker = ObstacleTracker()
    now, robot_x = 0.0, 0.0
    for _ in range(20):
        tracker.predict(DT)
        now += DT
        robot_x += 0.4 * DT                  # robot walks forward at 0.4 m/s
        _observe(tracker, (5.0, 1.0), (robot_x, 0.0, 0.0), now)
    track = tracker.confirmed_tracks()[0]
    assert track.speed < 0.20, f"ego-motion leaked in as {track.speed:.3f} m/s"


def test_crossing_person_velocity_converges():
    tracker = ObstacleTracker()
    now, person_y = 0.0, -2.0
    speed = 1.0                              # m/s, crossing left across the robot's nose
    for _ in range(25):
        tracker.predict(DT)
        now += DT
        person_y += speed * DT
        _observe(tracker, (3.0, person_y), (0.0, 0.0, 0.0), now)
    track = tracker.confirmed_tracks()[0]
    assert abs(track.state[3] - speed) < 0.25, f"vy={track.state[3]:.3f}, want {speed}"
    assert abs(track.state[2]) < 0.25, f"vx={track.state[2]:.3f}, want 0"


def test_track_needs_two_hits_to_confirm():
    tracker = ObstacleTracker()
    _observe(tracker, (3.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    assert tracker.tracks and not tracker.confirmed_tracks()
    tracker.predict(DT)
    _observe(tracker, (3.0, 0.0), (0.0, 0.0, 0.0), DT)
    assert tracker.confirmed_tracks()


def test_out_of_view_track_coasts_instead_of_dying():
    tracker = ObstacleTracker(fov_rad=math.radians(120.0))
    now = 0.0
    for _ in range(5):
        tracker.predict(DT)
        now += DT
        _observe(tracker, (1.0, 0.0), (0.0, 0.0, 0.0), now)
    assert tracker.confirmed_tracks()

    # The robot turns away; the person is now well outside the 120 deg view. Far more
    # empty updates than MAX_MISSES, but none of them count.
    yaw = math.radians(120.0)
    for _ in range(MAX_MISSES * 3):
        tracker.predict(DT)
        now += DT
        tracker.update([], now, 0.0, 0.0, yaw)
    assert tracker.confirmed_tracks(), "a track outside the FOV must not be deleted"


def test_in_view_track_dies_after_repeated_misses():
    tracker = ObstacleTracker()
    now = 0.0
    for _ in range(5):
        tracker.predict(DT)
        now += DT
        _observe(tracker, (3.0, 0.0), (0.0, 0.0, 0.0), now)
    for _ in range(MAX_MISSES + 1):
        tracker.predict(DT)
        now += DT
        tracker.update([], now, 0.0, 0.0, 0.0)      # in view, and not there
    assert not tracker.tracks


def test_coast_expires_eventually():
    tracker = ObstacleTracker(fov_rad=math.radians(120.0))
    _observe(tracker, (1.0, 0.0), (0.0, 0.0, 0.0), 0.0)
    tracker.predict(DT)
    _observe(tracker, (1.0, 0.0), (0.0, 0.0, 0.0), DT)
    yaw = math.radians(120.0)
    tracker.update([], DT + COAST_TIMEOUT_S + 0.1, 0.0, 0.0, yaw)
    assert not tracker.tracks, "coasting must not be indefinite"


def test_two_people_keep_their_identities():
    tracker = ObstacleTracker()
    now = 0.0
    left, right = [3.0, 1.2], [3.0, -1.2]
    for _ in range(12):
        tracker.predict(DT)
        now += DT
        left[1] -= 0.3 * DT                  # converging, but never within the gate
        right[1] += 0.3 * DT
        robot = (0.0, 0.0, 0.0)
        observations = []
        for person in (left, right):
            dx, dy = person[0], person[1]
            observations.append(Observation.from_bearing_range(
                math.atan2(dy, dx), math.hypot(dx, dy), *robot))
        tracker.update(observations, now, *robot)
    confirmed = tracker.confirmed_tracks()
    assert len(confirmed) == 2, f"expected 2 tracks, got {len(confirmed)}"
    assert len({t.track_id for t in confirmed}) == 2


def test_position_sigma_grows_while_coasting():
    tracker = ObstacleTracker()
    now = 0.0
    for _ in range(6):
        tracker.predict(DT)
        now += DT
        _observe(tracker, (3.0, 0.0), (0.0, 0.0, 0.0), now)
    fresh = tracker.confirmed_tracks()[0].position_sigma
    for _ in range(4):
        tracker.predict(DT)
    stale = tracker.confirmed_tracks()[0].position_sigma
    assert stale > fresh, f"{stale:.3f} should exceed {fresh:.3f}"


def test_observation_covariance_is_anisotropic_and_rotates():
    # Straight ahead: range error lies along odom x, bearing error along odom y.
    obs = Observation.from_bearing_range(0.0, 4.0, 0.0, 0.0, 0.0)
    cov = obs.covariance()
    assert cov[0, 0] > cov[1, 1] * 4, "range should be far noisier than bearing"
    # Rotate the robot 90 deg: the same measurement's noise must rotate with it.
    turned = Observation.from_bearing_range(0.0, 4.0, 0.0, 0.0, math.pi / 2.0)
    turned_cov = turned.covariance()
    assert turned_cov[1, 1] > turned_cov[0, 0] * 4
    assert abs(cov[0, 0] - turned_cov[1, 1]) < 1e-9


def test_speed_is_clamped():
    tracker = ObstacleTracker()
    now = 0.0
    _observe(tracker, (2.0, 0.0), (0.0, 0.0, 0.0), now)
    for _ in range(6):
        tracker.predict(DT)
        now += DT
        # A wild range jump — the sort a truncated box can produce.
        _observe(tracker, (2.0 + 20.0 * now, 0.0), (0.0, 0.0, 0.0), now)
    assert tracker.tracks[0].speed <= 3.0 + 1e-6


# ── Occlusion ───────────────────────────────────────────────────────────────
# A shadow is (bearing_rad, half_angle_rad, range_m), as static_map.occluders returns
# it: the staged bin at 2.15 m subtends about 6 degrees.
BIN_SHADOW = ((math.radians(7.1), math.radians(6.2), 2.15),)


def _miss(tracker, robot_pose, now, occluders=()):
    """One perception cycle that saw nothing at all."""
    tracker.update([], now, *robot_pose, occluders=occluders)


def test_a_person_hidden_behind_a_bin_is_not_deleted():
    """THE asymmetry, and it ran the wrong way.

    Losing someone out of the 120-degree cone bought 3 s of coast; losing them behind
    something INSIDE the cone was read as "visible, therefore absent" and deleted in
    about half a second. Observed live: a volunteer stepped into a doorway, the track
    was dropped, the robot read the lane as clear and accelerated, and they reappeared
    0.19 m inside the hard gap.
    """
    tracker = ObstacleTracker()
    now = 0.0
    for _ in range(4):                       # confirm a track behind where the bin is
        _observe(tracker, (3.2, 0.40), ORIGIN, now)
        now += DT
    assert tracker.confirmed_tracks()
    for _ in range(MAX_MISSES + 3):
        now += DT
        _miss(tracker, ORIGIN, now, occluders=BIN_SHADOW)
    assert tracker.confirmed_tracks(), "hidden is not absent"


def test_the_same_person_with_no_occluder_is_still_deleted():
    """The counter-example. Without it the test above would pass on a tracker that
    simply never deletes anything."""
    tracker = ObstacleTracker()
    now = 0.0
    for _ in range(4):
        _observe(tracker, (3.2, 0.40), ORIGIN, now)
        now += DT
    for _ in range(MAX_MISSES + 3):
        now += DT
        _miss(tracker, ORIGIN, now)
    assert not tracker.tracks


def test_someone_in_front_of_the_bin_is_still_expected_to_be_seen():
    """A shadow falls BEHIND its caster. Silencing misses in front of it would hide
    exactly the people closest to the robot."""
    tracker = ObstacleTracker()
    now = 0.0
    for _ in range(4):
        _observe(tracker, (1.2, 0.15), ORIGIN, now)     # nearer than the bin's 2.15 m
        now += DT
    for _ in range(MAX_MISSES + 3):
        now += DT
        _miss(tracker, ORIGIN, now, occluders=BIN_SHADOW)
    assert not tracker.tracks


def test_someone_beside_the_bin_is_still_expected_to_be_seen():
    """Further away, but off the shadow's bearing."""
    tracker = ObstacleTracker()
    now = 0.0
    for _ in range(4):
        _observe(tracker, (3.0, -1.6), ORIGIN, now)     # well clear of +7.1 deg
        now += DT
    for _ in range(MAX_MISSES + 3):
        now += DT
        _miss(tracker, ORIGIN, now, occluders=BIN_SHADOW)
    assert not tracker.tracks


def test_the_coast_timeout_still_bounds_an_occluded_track():
    """Occlusion suspends the MISS count, not the clock. Someone who steps behind a bin
    and stays there must not be planned against for ever."""
    tracker = ObstacleTracker()
    now = 0.0
    for _ in range(4):
        _observe(tracker, (3.2, 0.40), ORIGIN, now)
        now += DT
    now += COAST_TIMEOUT_S + 0.5
    _miss(tracker, ORIGIN, now, occluders=BIN_SHADOW)
    assert not tracker.tracks


def test_a_constant_is_trusted_less_than_a_weak_measurement():
    """Three tiers, not two. A height-prior range is the good measurement; a width-prior
    one is weak because body yaw swings apparent width nearly 2:1; a constant is not a
    measurement at all and cannot move when the robot does. Collapsing the last two into
    one tier is what let a fixed 0.719 m reading hold a landmark still for five seconds
    on 2026-08-19."""
    assert _range_sigma_scale("height") == 1.0
    assert _range_sigma_scale("width") == WIDTH_SOURCE_SIGMA_SCALE
    for constant in ("frame-fill", "width-capped"):
        assert _range_sigma_scale(constant) == UNMEASURED_SOURCE_SIGMA_SCALE
        assert _range_sigma_scale(constant) > WIDTH_SOURCE_SIGMA_SCALE * 2


def test_a_constant_observation_keeps_its_bearing_and_gives_up_its_range():
    """The point of the inflation. The bearing stayed accurate through both hardware
    failures — it tracked +13 to +25 deg while the range was frozen — so the filter must
    keep using the cross axis and stop leaning on the along axis."""
    good = observation_from(0.3, 2.0, "height", "bin", (0.0, 0.0, 0.0))
    constant = observation_from(0.3, 2.0, "width-capped", "bin", (0.0, 0.0, 0.0))
    assert constant.sigma_cross == good.sigma_cross, "the bearing is still good"
    assert constant.sigma_along > good.sigma_along * 5, "the range is not"


def test_a_constant_is_still_an_observation():
    """Distrusting a measurement and discarding it are different things. Discarded, the
    landmark goes unmatched, misses accrue and it is deleted and respawned with an
    unconverged radius — the failure fixed in static_map's MAX_MISSES. It has to keep
    associating."""
    constant = observation_from(0.0, 0.719, "width-capped", "bin", (0.0, 0.0, 0.0))
    assert math.isfinite(constant.x) and math.isfinite(constant.y)
    assert math.isfinite(constant.sigma_along)
    assert math.isclose(constant.x, 0.719), "it still reports where it thinks the bin is"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"tracker: {len(tests)}/{len(tests)} passed")
