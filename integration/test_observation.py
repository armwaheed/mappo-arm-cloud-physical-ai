#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the telemetry reader and the observation adapter.

These run against ``../evidence/sample_telemetry.jsonl`` — a real 12 s run on the robot —
as well as synthetic ticks, so the contract is checked against the thing that actually
produced it rather than only against a fixture written to agree with the code.

Run: ``python3 test_observation.py``
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from observation import (
    observation_from_tick,
    range_vector,
    reliable_range_m,
    to_body_frame,
    wrap_pi,
)
from telemetry_reader import Run, SchemaError, read_run

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "evidence", "sample_telemetry.jsonl")


def _tick(**overrides):
    """A tick with the robot at the origin facing +x, and the staged bin ahead."""
    tick = {
        "type": "tick", "t": 1.0,
        "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "goal": {"x": 3.195, "y": 0.398, "distance_m": 3.22},
        "obstacles": [{"label": "bin", "x": 2.134, "y": 0.266,
                       "vx": 0.0, "vy": 0.0, "radius_m": 0.23}],
        "command": {"vx": 0.35, "vy": 0.0, "wz": 0.0, "reason": "goal",
                    "gap_m": 0.75, "feasible": 330, "evaluated": 330},
        "perception": {"seq": 1, "frame_age_s": 0.3, "detect_ms": 120.0,
                       "video_frame": 0},
        "posture": "standing", "live": False, "health": None,
    }
    tick.update(overrides)
    return tick


# ── Frame transform ─────────────────────────────────────────────────────────
def test_a_point_ahead_of_a_rotated_robot_is_ahead_in_the_body_frame():
    """The transform that is easiest to get backwards, so it gets its own test."""
    pose = {"x": 1.0, "y": 2.0, "yaw": math.radians(90.0)}
    # Two metres along the robot's nose, which points at odom +y.
    bx, by = to_body_frame(pose, 1.0, 4.0)
    assert abs(bx - 2.0) < 1e-9, bx
    assert abs(by) < 1e-9, by


def test_a_point_to_the_robots_left_has_positive_y():
    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    _, by = to_body_frame(pose, 0.0, 1.0)
    assert by > 0.0


# ── The range fan ───────────────────────────────────────────────────────────
def test_the_ray_straight_ahead_stops_at_the_bins_near_face():
    """Exact ray-versus-disc, not the distance to the centre.

    The bin is 2.15 m away with a 0.23 m radius, so a ray through its centre must report
    ~1.92 m. Reporting 2.15 would put the policy's obstacle a whole radius too far away.
    """
    tick = _tick(obstacles=[{"label": "bin", "x": 2.15, "y": 0.0,
                             "vx": 0.0, "vy": 0.0, "radius_m": 0.23}])
    ranges, bearings = range_vector(tick, rays=1, fov_rad=1e-9)
    assert abs(bearings[0]) < 1e-9
    assert abs(ranges[0] - (2.15 - 0.23)) < 1e-6, ranges[0]


def test_a_clear_bearing_reports_the_max_range():
    tick = _tick(obstacles=[])
    ranges, _ = range_vector(tick, rays=8, max_range_m=6.0)
    assert all(r == 6.0 for r in ranges), ranges


def test_an_obstacle_behind_the_robot_does_not_shorten_a_forward_ray():
    """A disc behind the ray origin has two negative roots and must be a miss."""
    tick = _tick(obstacles=[{"label": "bin", "x": -2.0, "y": 0.0,
                             "vx": 0.0, "vy": 0.0, "radius_m": 0.23}])
    ranges, _ = range_vector(tick, rays=1, fov_rad=1e-9, max_range_m=6.0)
    assert ranges[0] == 6.0


def test_the_robot_standing_inside_a_disc_reports_zero_not_a_miss():
    """Degenerate but reachable — the map's radius includes an uncertainty margin that
    can swallow the robot's own position. Zero is the safe reading; a miss is not."""
    tick = _tick(obstacles=[{"label": "bin", "x": 0.05, "y": 0.0,
                             "vx": 0.0, "vy": 0.0, "radius_m": 0.5}])
    ranges, _ = range_vector(tick, rays=1, fov_rad=1e-9)
    assert ranges[0] == 0.0, ranges[0]


def test_a_16_ray_360_degree_fan_is_blind_to_the_staged_bin():
    """MEASURED, and the reason DEFAULT_FOV_RAD is the camera's FOV and not 2*pi.

    16 rays over a full circle sit 22.5 deg apart. The bin at 2 m subtends 13.2 deg, so
    it fits entirely between two rays and the policy is handed open floor where the only
    obstacle in the scene is. Nothing about the adapter is wrong here — the fan is simply
    too coarse — which is exactly why it needs to be pinned rather than discovered on a
    robot.
    """
    tick = _tick(obstacles=[{"label": "bin", "x": 2.0, "y": 0.0,
                             "vx": 0.0, "vy": 0.0, "radius_m": 0.23}])
    ranges, _ = range_vector(tick, rays=16, fov_rad=2.0 * math.pi)
    assert all(r == 6.0 for r in ranges), "if this now sees it, re-derive the limit"
    assert reliable_range_m(0.23, 16, 2.0 * math.pi) < 2.0


def test_the_default_fan_does_see_the_staged_bin():
    """The same 16 rays over the camera's 85.27 deg are 5.3 deg apart, and catch it."""
    tick = _tick(obstacles=[{"label": "bin", "x": 2.0, "y": 0.0,
                             "vx": 0.0, "vy": 0.0, "radius_m": 0.23}])
    ranges, _ = range_vector(tick, rays=16)
    blocked = [r for r in ranges if r < 6.0]
    assert blocked, "the default fan must see a bin 2 m dead ahead"
    assert min(blocked) < 2.0


def test_reliable_range_matches_the_geometry_it_claims():
    """r / sin(half-spacing): at exactly that range the object subtends one ray gap."""
    for radius, rays, fov in ((0.23, 16, math.radians(85.27)), (0.35, 32, math.pi)):
        limit = reliable_range_m(radius, rays, fov)
        subtended = 2.0 * math.asin(radius / limit)
        assert abs(subtended - fov / rays) < 1e-9, (radius, rays, fov)


def test_the_default_fan_covers_the_detectors_whole_band():
    """A limit shorter than the detector's range would waste perception the robot has."""
    assert reliable_range_m(0.23) > 4.5


def test_rays_span_the_requested_field_of_view():
    _, bearings = range_vector(_tick(), rays=8, fov_rad=math.radians(85.27))
    assert all(abs(b) <= math.radians(85.27) / 2.0 + 1e-9 for b in bearings)


def test_the_fan_moves_with_the_robots_heading():
    """The obstacle is fixed in ODOM, so turning must slide it across the fan.

    A small turn, deliberately: 20 deg leaves the bin inside the camera's +-42.6 deg, so
    this isolates the transform rather than the field-of-view edge below.
    """
    ahead = _tick()
    turned = _tick(pose={"x": 0.0, "y": 0.0, "yaw": math.radians(20.0)})
    ranges_ahead, _ = range_vector(ahead, rays=16)
    ranges_turned, _ = range_vector(turned, rays=16)
    assert min(ranges_ahead) < 6.0 and min(ranges_turned) < 6.0
    assert ranges_ahead != ranges_turned, "turning must reshuffle the fan"


def test_turning_away_reports_clear_because_the_camera_cannot_look_there():
    """The blind-spot limitation, made explicit rather than left to be discovered.

    Turn 90 deg and the bin leaves the 85.27 deg camera cone, so every ray reads
    max_range — "clear". That is honest about the sensor and OPTIMISTIC as an input: a
    policy trained on a 360 deg scan will read this as open floor and may reverse into
    something the robot has simply stopped looking at. The stack itself never commands
    reverse for the same reason.
    """
    turned = _tick(pose={"x": 0.0, "y": 0.0, "yaw": math.radians(90.0)})
    ranges, _ = range_vector(turned, rays=16)
    assert all(r == 6.0 for r in ranges), ranges


# ── The observation ─────────────────────────────────────────────────────────
def test_the_goal_comes_back_in_body_polar():
    observation = observation_from_tick(_tick())
    assert abs(observation.goal_range_m - 3.22) < 0.01
    assert abs(math.degrees(observation.goal_bearing_rad) - 7.1) < 0.2


def test_a_tick_with_no_goal_yields_no_observation():
    """The goal-search phase is recorded on purpose; it must not become a zero vector
    that reads as a goal sitting on the robot."""
    assert observation_from_tick(_tick(goal=None)) is None


def test_the_vector_is_flat_and_the_right_length():
    observation = observation_from_tick(_tick(), rays=16)
    vector = observation.as_vector()
    assert len(vector) == 2 + 16 + 3
    assert all(isinstance(v, float) for v in vector)


def test_wrap_pi_is_symmetric_about_zero():
    assert abs(wrap_pi(math.pi * 3)) - math.pi < 1e-9
    assert abs(wrap_pi(0.5) - 0.5) < 1e-12


# ── Against the real recording ──────────────────────────────────────────────
def test_the_sample_run_parses_and_carries_a_pose_on_every_tick():
    """The point of the whole exercise: the console log managed ONE pose per run."""
    run = read_run(SAMPLE)
    assert run.ticks, "the sample should not be empty"
    assert all(t["pose"]["x"] is not None for t in run.ticks)
    assert run.completed, "the sample run wrote an outcome line"


def test_every_goal_bearing_tick_of_the_real_run_yields_an_observation():
    run = read_run(SAMPLE)
    with_goal = [t for t in run.ticks if t.get("goal")]
    assert with_goal
    observations = [observation_from_tick(t) for t in with_goal]
    assert all(o is not None for o in observations)
    assert all(len(o.as_vector()) == 2 + 16 + 3 for o in observations)


def test_the_real_run_actually_ranges_the_bin_on_some_rays():
    """If nothing ever blocks a ray the adapter is wired up but useless."""
    run = read_run(SAMPLE)
    blocked_ticks = 0
    for tick in run.ticks:
        if not tick.get("goal"):
            continue
        ranges, _ = range_vector(tick)
        if min(ranges) < 6.0:
            blocked_ticks += 1
    assert blocked_ticks > 0, "the bin should shorten a ray on at least one tick"


def test_an_unknown_schema_major_is_refused_rather_than_guessed():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "future.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "header",
                                     "schema": "go2.visual_nav.telemetry/9"}) + "\n")
        try:
            read_run(path)
        except SchemaError:
            return
    raise AssertionError("accepted a schema major it does not understand")


def test_a_file_with_no_header_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "headless.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(_tick()) + "\n")
        try:
            read_run(path)
        except SchemaError:
            return
    raise AssertionError("accepted a stream with no header")


def test_a_truncated_last_line_is_tolerated():
    """A killed run is the normal case, and losing the whole file over its last few
    bytes would defeat the writer's per-line flush."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "killed.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "header",
                                     "schema": "go2.visual_nav.telemetry/1"}) + "\n")
            handle.write(json.dumps(_tick()) + "\n")
            handle.write('{"type": "tick", "pose": {"x": 1.0')     # cut mid-write
        run = read_run(path)
    assert len(run.ticks) == 1
    assert not run.completed


# ── A command block can be PRESENT AND PARTIAL ──────────────────────────────
def test_a_command_carrying_only_a_reason_does_not_raise():
    """``mappo_policy.tick_from_state`` builds ``{"reason": ...}`` with no velocity in it,
    so this shape is produced by this repository, not hypothesised.

    The old guard was ``tick.get("command") or {"vx": 0.0, ...}``, which reads as though
    it handles a missing velocity and does not: the fallback only fires when the block is
    absent or empty, never when it is present and short, which is the case that occurs.
    It raised ``KeyError: 'vx'`` from two public entry points."""
    observation = observation_from_tick(_tick(command={"reason": "goal"}))
    assert observation is not None
    assert observation.speed == (0.0, 0.0, 0.0), observation.speed


def test_a_partial_command_is_not_counted_as_movement():
    """Same shape, through the reader's public ``moving_ticks``. A tick recording only a
    reason commanded no velocity this reader can see, which is the same answer as an
    absent block — but it used to be a ``KeyError`` instead of an answer."""
    run = Run(header={"schema": "go2.visual_nav.telemetry/1"},
              ticks=[_tick(command={"reason": "goal"}),
                     _tick(command={"vx": 0.35, "vy": 0.0, "wz": 0.0})])
    assert len(run.moving_ticks()) == 1


def test_a_null_velocity_component_is_not_movement_either():
    """JSON has no infinity, so the writer emits null for a non-finite value. ``abs(None)``
    is a ``TypeError``, and the reader must not fall over on its own format."""
    run = Run(header={"schema": "go2.visual_nav.telemetry/1"},
              ticks=[_tick(command={"vx": None, "vy": None, "wz": None})])
    assert run.moving_ticks() == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"observation: {len(tests)}/{len(tests)} passed")
