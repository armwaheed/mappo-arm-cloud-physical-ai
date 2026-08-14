#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the closed-loop simulation.

Most of these pin a bug that was actually in the first version, and the docstring says
which — because a simulation is the one kind of code that reports a confident, plausible,
wrong number when it is broken. The first version reported the shipped planner "arriving"
at a 2.6 m goal twice in ten runs while parked and commanding zero velocity.

Needs the policy package (numpy) and the vendored planner.
Run: ``python3 test_closed_loop_sim.py``
"""
from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import replace

# BOTH paths go in before ANY sibling import. `ruff --fix` sorts imports into
# contiguous blocks and will hoist `from avoidance import ...` above a sys.path line that
# sits between the blocks — which is exactly how this file went from passing to
# ModuleNotFoundError without anybody touching a test.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "robot-stack", "unitree", "go2",
                                "visual_nav"))
from avoidance import STATIC_HARD_GAP_M, STATIC_SOFT_GAP_M, PlannerConfig

from closed_loop_sim import (
    PlannerController,
    PolicyController,
    Scenario,
    SimConfig,
    SimObstacle,
    SupervisedController,
    _actuate,
    _pose_at,
    _to_planner,
    _visible,
    run_once,
    scenarios,
)
from mappo_policy import rollout_is_feasible

QUIET = replace(SimConfig(), velocity_noise_mps=0.0, yaw_noise_radps=0.0)
BIN = SimObstacle(1.3, 0.0, 0.23, "static", "landmark-1")
STRAIGHT = Scenario("straight", start=(0.0, 0.0, 0.0), goal=(2.6, 0.0),
                    obstacles=(BIN,))


# ── The actuator, where the worst bug lived ─────────────────────────────────
def test_a_commanded_stop_stays_stopped():
    """THE ONE THAT INVALIDATED THE FIRST RESULTS. Noise was applied to the achieved
    velocity unconditionally, so a robot commanding zero performed a random walk — and
    the shipped planner, deadlocked and stationary, "arrived" at the goal by drifting
    there. A simulation that lets a parked robot travel is not measuring navigation."""
    rng = random.Random(0)
    config = SimConfig()          # noise ON, deliberately
    assert config.velocity_noise_mps > 0.0
    velocity = (0.0, 0.0, 0.0)
    for _ in range(600):
        velocity = _actuate(velocity, (0.0, 0.0, 0.0), config, 0.1, rng)
        assert velocity == (0.0, 0.0, 0.0)


def test_the_robot_delivers_only_the_measured_fraction_of_the_command():
    """0.45, fitted against the POSE. Charging the shortfall to noise instead — which is
    what ``measured - command`` alone suggests — is what produced the random walk above."""
    config = replace(QUIET, actuator_gain=0.45)
    velocity = (0.0, 0.0, 0.0)
    for _ in range(200):          # long enough to be past the acceleration ramp
        velocity = _actuate(velocity, (0.35, 0.0, 0.0), config, 0.1, random.Random(0))
    assert math.isclose(velocity[0], 0.35 * 0.45, abs_tol=1e-6)


def test_the_acceleration_limit_binds_on_the_first_tick():
    """Otherwise a step command teleports the robot to top speed and the whole notion of
    a dynamic window is decoration."""
    got = _actuate((0.0, 0.0, 0.0), (0.35, 0.0, 0.0), QUIET, 0.1, random.Random(0))
    assert math.isclose(got[0], QUIET.limits.accel_x * 0.1, abs_tol=1e-9)


def test_the_rng_is_drawn_from_even_when_the_command_is_zero():
    """The ablated control is only a control if it sees the same numbers on the same
    step. Draw only while moving and the noise sequence depends on the trajectory, which
    is the thing being compared."""
    a, b = random.Random(7), random.Random(7)
    _actuate((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), SimConfig(), 0.1, a)
    b.gauss(0.0, 1.0), b.gauss(0.0, 1.0), b.gauss(0.0, 1.0)
    assert a.random() == b.random(), "a zero command must still consume three draws"


# ── The planner obstacle, where the second-worst bug lived ──────────────────
def test_a_static_obstacle_carries_the_gaps_visual_nav_gives_it():
    """WITHOUT THIS THE PLANNER NEVER MOVED. A mapped bin inheriting a PERSON's 1.20 m
    soft gap makes the clearance term outweigh the goal term from 1.4 m away, and
    standing still becomes the planner's best option. It scored zero arrivals and it
    looked like a policy result."""
    static = _to_planner(BIN)
    assert static.soft_gap_m == STATIC_SOFT_GAP_M
    assert static.hard_gap_m == STATIC_HARD_GAP_M
    assert static.soft_gap_m < PlannerConfig().soft_gap_m
    mover = _to_planner(SimObstacle(1.0, 0.0, 0.5, "tracked", "track-1"))
    assert mover.soft_gap_m is None and mover.hard_gap_m is None


def test_the_planner_reaches_the_goal_past_the_bin():
    """The control has to work, or nothing measured against it means anything."""
    result = run_once(STRAIGHT, PlannerController(QUIET), QUIET, seed=0)
    assert result.outcome == "arrived", result
    assert result.min_clearance_m > 0.0


# ── The veto ────────────────────────────────────────────────────────────────
def test_the_veto_refuses_a_command_that_drives_into_the_obstacle():
    """Remove the veto and this returns True. Straight ahead at top speed, from 1.3 m
    out, ends inside a 0.23 m disc well within the 2.5 s horizon."""
    planner = PlannerController(QUIET).planner
    obstacles = [_to_planner(BIN)]
    assert not rollout_is_feasible(planner, (0.0, 0.0, 0.0), (0.35, 0.0, 0.0), obstacles)


def test_the_veto_allows_a_command_that_clears_the_obstacle():
    """A veto that refuses everything is not a safety feature, it is a brake."""
    planner = PlannerController(QUIET).planner
    obstacles = [_to_planner(BIN)]
    assert rollout_is_feasible(planner, (0.0, 0.0, 0.0), (0.30, 0.0, 0.55), obstacles)
    assert rollout_is_feasible(planner, (0.0, 0.0, 0.0), (0.35, 0.0, 0.0), [])


def test_the_veto_still_reaches_the_planner_internals_it_depends_on():
    """``rollout_is_feasible`` uses ``_rollout``, ``_gaps`` and ``_hard_gaps`` because the public
    ``plan`` scores a whole sampled window and cannot be asked about one command.
    ``robot-stack/`` is vendored, so a re-vendor could move them. This is the tripwire."""
    planner = PlannerController(QUIET).planner
    for name in ("_rollout", "_gaps", "_hard_gaps"):
        assert callable(getattr(planner, name, None)), f"planner lost {name}"


def test_supervision_removes_the_collisions_the_raw_policy_has():
    """The result the whole file exists for, on the scenario set, at the shipped scale.
    Raw policy collides; under the veto it does not. Both numbers are reported in the PR
    rather than only the flattering one."""
    scenes = scenarios(random.Random(20260813), 10)
    config = replace(SimConfig(), max_run_s=40.0)
    raw = [run_once(s, PolicyController(config), config, seed)
           for seed, s in enumerate(scenes)]
    guarded = [run_once(s, SupervisedController(config), config, seed)
               for seed, s in enumerate(scenes)]
    raw_hits = sum(1 for r in raw if r.outcome == "collision")
    guarded_hits = sum(1 for r in guarded if r.outcome == "collision")
    assert raw_hits > 0, "the scenario set must be hard enough to separate the two"
    assert guarded_hits < raw_hits, f"veto did not help: {guarded_hits} vs {raw_hits}"


# ── Perception ──────────────────────────────────────────────────────────────
def test_the_camera_cannot_see_behind_the_robot():
    """Widen the cone to a full circle and this passes for the wrong reason. The stack's
    blindness astern is the optimistic direction and the simulation has to keep it."""
    assert _visible((0.0, 0.0, 0.0), SimObstacle(1.0, 0.0, 0.2), QUIET)
    assert not _visible((0.0, 0.0, 0.0), SimObstacle(-1.0, 0.0, 0.2), QUIET)
    assert not _visible((0.0, 0.0, 0.0), SimObstacle(0.0, 1.0, 0.2), QUIET)


def test_an_obstacle_past_the_detector_band_is_not_seen():
    assert not _visible((0.0, 0.0, 0.0),
                        SimObstacle(QUIET.detect_range_m + 1.0, 0.0, 0.2), QUIET)


def test_perception_is_delayed_and_never_reads_the_future():
    history = [(0.0, (0.0, 0.0, 0.0)), (0.1, (0.1, 0.0, 0.0)), (0.2, (0.2, 0.0, 0.0))]
    assert _pose_at(history, 0.15) == (0.1, 0.0, 0.0)
    assert _pose_at(history, -5.0) == (0.0, 0.0, 0.0), "before the run, use the start"
    assert _pose_at(history, 99.0) == (0.2, 0.0, 0.0)


def test_a_static_obstacle_stays_mapped_once_it_leaves_the_camera():
    """The odom map is what makes a landmark persist on the robot; a simulation that
    forgot it would credit the policy with dodging something it could still see."""
    scene = Scenario("passing", start=(0.0, 0.0, 0.0), goal=(2.6, 0.0),
                     obstacles=(SimObstacle(0.9, 0.6, 0.2, "static", "landmark-1"),))
    result = run_once(scene, PlannerController(QUIET), QUIET, seed=0)
    # If the landmark were dropped when it left the cone the planner would cut the corner
    # and graze it. Clearance staying positive is the observable consequence.
    assert result.min_clearance_m > 0.0


# ── The harness itself ──────────────────────────────────────────────────────
def test_the_same_seed_gives_the_same_run():
    a = run_once(STRAIGHT, PlannerController(SimConfig()), SimConfig(), seed=3)
    b = run_once(STRAIGHT, PlannerController(SimConfig()), SimConfig(), seed=3)
    assert (a.outcome, a.ticks, round(a.path_m, 9)) == \
           (b.outcome, b.ticks, round(b.path_m, 9))


def test_an_ablated_run_has_no_obstacles_and_therefore_no_clearance():
    result = run_once(STRAIGHT, PlannerController(QUIET), QUIET, seed=0, ablated=True)
    assert not math.isfinite(result.min_clearance_m)
    assert result.outcome == "arrived"


def test_a_controller_is_reset_between_runs():
    """They are reused across seeds for speed. A controller carrying the previous run's
    run-local frame would place the whole episode at an offset — and it would still
    produce a plausible-looking trajectory."""
    controller = PlannerController(QUIET)
    first = run_once(STRAIGHT, controller, QUIET, seed=0)
    second = run_once(STRAIGHT, controller, QUIET, seed=0)
    assert (first.outcome, first.ticks) == (second.outcome, second.ticks)


class _Straight:
    """Full ahead, always. Exists so the collision METRIC can be tested on its own,
    without a controller's avoidance deciding the answer for it."""

    name = "straight"

    def reset(self) -> None:
        pass

    def command(self, t, pose, goal, obstacles, measured):
        return (0.35, 0.0, 0.0), "straight", 0.0


def test_collision_is_judged_at_the_configured_robot_radius():
    """0.25 m, per issue #5 — not the trained 0.10 VMAS radius, which would let the
    simulation pass a robot that had walked through the bin. Two radii, one scenario:
    the same drive-straight run collides at 0.25 and clears at 0.10."""
    scene = Scenario("grazing", start=(0.0, 0.0, 0.0), goal=(2.0, 0.0),
                     obstacles=(SimObstacle(1.0, 0.40, 0.20, "static", "l1"),))
    # 0.40 m of lateral offset against 0.20 + 0.25 = 0.45 m of radii: overlapping.
    planning = run_once(scene, _Straight(), replace(QUIET, robot_radius_m=0.25), seed=0)
    assert planning.outcome == "collision" and planning.min_clearance_m < 0.0
    # ...and against 0.20 + 0.10 = 0.30 m it is 0.10 m clear, so it drives past.
    trained = run_once(scene, _Straight(), replace(QUIET, robot_radius_m=0.10), seed=0)
    assert trained.outcome == "arrived" and trained.min_clearance_m > 0.0


def test_the_scenario_set_puts_the_obstacle_where_it_has_to_be_dealt_with():
    """A prop set that misses the path measures nothing. Every generated obstacle is
    close enough to the straight line that ignoring it costs clearance."""
    for scene in scenarios(random.Random(1), 12):
        obstacle = scene.obstacles[0]
        goal_distance = math.hypot(*scene.goal)
        lateral = abs(obstacle.x * scene.goal[1] - obstacle.y * scene.goal[0]) \
            / goal_distance
        assert lateral < obstacle.radius_m + 0.25 + 0.20, scene
        assert 0.5 < math.hypot(obstacle.x, obstacle.y) < goal_distance


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"closed_loop_sim: {len(tests)}/{len(tests)} passed")
