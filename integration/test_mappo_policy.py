#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared policy runner and the heading servo.

Needs the policy package (numpy). Run: ``python3 test_mappo_policy.py``
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mappo_policy import HeadingServo, PolicyRunner, tick_from_state

#: Inside the 0.875 m sensing horizon, which is measured to the SURFACE: 0.90 - 0.23 =
#: 0.67 m. Put it at 1.2 m instead and the fan reads all-zero, which looks exactly like a
#: bridge that dropped the obstacle.
BIN = {"label": "bin", "kind": "static", "id": "landmark-1",
       "x": 0.9, "y": 0.0, "vx": 0.0, "vy": 0.0, "radius_m": 0.23}
WALKER = {"label": "person", "kind": "tracked", "id": "track-1",
          "x": 1.5, "y": 0.4, "vx": 0.5, "vy": 0.0, "radius_m": 0.5}


# ── The heading servo ───────────────────────────────────────────────────────
def test_the_servo_turns_towards_the_direction_of_travel():
    servo = HeadingServo()
    assert servo.yaw_rate(0.2, 0.2) > 0.0, "travel to the LEFT should turn left"
    assert servo.yaw_rate(0.2, -0.2) < 0.0


def test_the_servo_holds_still_inside_its_deadband():
    """Without a deadband a heading error of a couple of degrees becomes a permanent
    twitch, and a rolling-shutter camera pays for every one of them."""
    servo = HeadingServo(deadband_rad=math.radians(8.0))
    assert servo.yaw_rate(0.3, 0.3 * math.tan(math.radians(5.0))) == 0.0
    assert servo.yaw_rate(0.3, 0.3 * math.tan(math.radians(20.0))) != 0.0


def test_the_servo_does_not_chase_the_heading_of_a_stopped_robot():
    """A zeroed command has no direction of travel. Spinning on the spot because the
    residual is 0.001 m/s to the left is the failure this prevents."""
    assert HeadingServo().yaw_rate(0.0, 0.0) == 0.0
    assert HeadingServo().yaw_rate(0.001, 0.001) == 0.0


def test_the_servo_is_capped_below_the_planner_own_yaw_rate():
    """This turn is a convenience. A fast one smears the frames the policy is fed, and
    perception is already 0.31 s behind."""
    servo = HeadingServo(max_wz=0.4)
    assert abs(servo.yaw_rate(0.1, 5.0)) <= 0.4
    assert abs(servo.yaw_rate(0.1, -5.0)) <= 0.4
    assert servo.max_wz < 0.70


# ── The runner ──────────────────────────────────────────────────────────────
def _runner(**kwargs) -> PolicyRunner:
    return PolicyRunner(**kwargs)


def test_the_first_tick_resets_the_run_and_later_ones_do_not():
    """``reset_run`` re-fixes the run-local frame. Sending it every tick would pin the
    robot at the origin of its own observation for the whole run, which looks entirely
    reasonable in a log and means the policy never sees itself move."""
    runner = _runner()
    first = runner.step(tick_from_state(0.0, (5.0, 5.0, 0.0), (7.0, 5.0), []))
    assert first is not None
    assert runner.controller.last_observation[0] == 0.0
    runner.step(tick_from_state(0.1, (5.5, 5.0, 0.0), (7.0, 5.0), []))
    moved = runner.controller.last_observation[0] * runner.config.meters_per_vmas_unit
    assert math.isclose(moved, 0.5, abs_tol=1e-5), "the second tick must not re-origin"


def test_a_tick_with_no_goal_returns_none_rather_than_a_goal_at_the_origin():
    assert _runner().step(tick_from_state(0.0, (0.0, 0.0, 0.0), None, [])) is None


def test_the_runner_maps_through_the_bridge_so_the_frame_is_body():
    """The simulator and the robot must share one mapping. If this runner built a
    ``RobotInput`` itself, a frame fixed in ``mappo_bridge`` would not reach the
    simulation, and the closed-loop result would be about a mapping nothing uses."""
    runner = _runner()
    runner.step(tick_from_state(0.0, (0.0, 0.0, 0.0), (2.0, 0.0), [],
                                measured=(0.3, 0.0, 0.0)))
    runner.step(tick_from_state(0.1, (0.0, 0.0, math.pi / 2), (2.0, 0.0), [],
                                measured=(0.3, 0.0, 0.0)))
    scale = runner.config.meters_per_vmas_unit
    # Body-frame (0.3, 0) at 90 degrees of turn is run-local (0, 0.3). Read as odom it
    # would still be (0.3, 0), which is the silent failure the bridge exists to prevent.
    assert math.isclose(runner.controller.last_observation[3] * scale, 0.3, abs_tol=1e-6)


def test_a_stationary_obstacle_reaches_the_policy_and_a_mover_does_not():
    """The agreed division of labour: static objects become the policy's lidar input,
    movers stay with the existing stop/wait logic and arrive as ``external_hold``."""
    runner = _runner()
    runner.step(tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 0.0), [BIN]))
    assert max(runner.controller.last_observation[6:]) > 0.0

    runner = _runner()
    runner.step(tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 0.0), [WALKER]))
    assert max(runner.controller.last_observation[6:]) == 0.0


def test_a_hold_for_a_mover_is_external_but_a_hold_for_the_bin_is_not():
    """Forwarding every planner hold zeroes the policy in the one scene it exists for."""
    runner = _runner()
    for_bin = runner.step(tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 0.0), [BIN],
                                          reason="hold"))
    assert for_bin.status == "COMMAND"

    runner = _runner()
    for_mover = runner.step(tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 0.0),
                                            [BIN, WALKER], reason="hold"))
    assert for_mover.status == "STOP_EXTERNAL_HOLD"


def test_the_intent_survives_a_stop_even_though_the_command_does_not():
    """A held tick with no recorded intent is indistinguishable from a policy that had
    nothing to say, and a shadow run is nothing but the record of what it wanted."""
    runner = _runner()
    step = runner.step(tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 0.0),
                                       [BIN, WALKER], reason="hold"))
    assert step.status == "STOP_EXTERNAL_HOLD"
    assert (step.vx_mps, step.vy_mps, step.wz_radps) == (0.0, 0.0, 0.0)
    assert step.intent_bearing_rad is not None


def test_the_intent_is_body_frame_and_survives_the_robot_turning():
    """The action is in the run-local frame. Reported without rotating it back, a robot
    that has turned 90 degrees reports an intent 90 degrees out — and it would look
    perfectly plausible, because the goal moved in the same log."""
    runner = _runner()
    ahead = runner.step(tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 0.0), []))
    assert abs(ahead.intent_bearing_rad) < math.radians(30.0)
    turned = runner.step(tick_from_state(0.1, (0.0, 0.0, math.pi / 2), (3.0, 0.0), []))
    # Same goal, robot now facing left: the goal is 90 degrees to its RIGHT, and so is
    # the intent. In the run-local frame the action barely moved.
    assert turned.intent_bearing_rad < -math.radians(50.0)


def test_the_servo_is_off_unless_one_is_supplied():
    """Off is the honest default for a shadow run: a logged ``wz`` implies the recorded
    command is what the robot would have done, and in shadow nothing of the sort is true.
    The policy itself never commands yaw."""
    plain = _runner().step(tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 1.0), []))
    assert plain.wz_radps == 0.0
    turning = _runner(servo=HeadingServo()).step(
        tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 1.0), []))
    assert turning.wz_radps != 0.0


def test_the_observation_is_reported_so_an_action_can_be_checked_afterwards():
    step = _runner().step(tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 0.0), [BIN]))
    assert len(step.observation) == 18
    assert all(isinstance(v, float) for v in step.observation)


def test_tick_from_state_accepts_a_planner_obstacle_as_well_as_a_dict():
    """The drive path has ``avoidance.Obstacle`` instances; the simulator has dicts.
    Requiring the caller to convert is how a field gets dropped on one of the two paths."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "robot-stack", "unitree", "go2", "visual_nav"))
    from avoidance import Obstacle
    tick = tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 0.0),
                           [Obstacle(x=1.0, y=0.0, vx=0.0, vy=0.0, radius_m=0.2,
                                     kind="static", object_id="landmark-9")])
    assert tick["obstacles"][0]["kind"] == "static"
    assert tick["obstacles"][0]["id"] == "landmark-9"
    assert tick["obstacles"][0]["radius_m"] == 0.2


def test_reset_starts_a_new_run_local_frame():
    runner = _runner()
    runner.step(tick_from_state(0.0, (0.0, 0.0, 0.0), (3.0, 0.0), []))
    runner.step(tick_from_state(0.1, (1.0, 0.0, 0.0), (3.0, 0.0), []))
    assert runner.controller.last_observation[0] != 0.0
    runner.reset()
    runner.step(tick_from_state(0.2, (1.0, 0.0, 0.0), (3.0, 0.0), []))
    assert runner.controller.last_observation[0] == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"mappo_policy: {len(tests)}/{len(tests)} passed")
