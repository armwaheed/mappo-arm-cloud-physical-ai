#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for the delivered policy package.

Two kinds of test live here and they are worth telling apart:

  * **The corrections.** Each of the deltas listed in ``PROVENANCE.md`` has a test that
    FAILS if the delta is reverted — which is the only thing that stops the next
    re-vendor from quietly putting the delivered behaviour back. All three of the
    corrections were silent failures, so none of them would show up any other way.
  * **The conventions.** Frame, polarity, ray geometry and scale. These were verified
    against the weights before they were written down (see ``integration/replay_mappo.py``
    and issue #4); the tests keep them from drifting.

Needs numpy and the checkpoint. Run: ``python3 test_physical_ai_mappo.py``
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from physical_ai_mappo import (
    CLOCK_TOLERANCE_S,
    COMMAND,
    N_RAYS,
    OBS_DIM,
    STOP_CLOCK_ERROR,
    STOP_EXTERNAL_HOLD,
    STOP_GOAL_REACHED,
    STOP_STALE_INPUT,
    Config,
    MappoController,
    RobotInput,
    StationaryObject,
)

CONFIG = ROOT / "config.json"


def _controller(**overrides) -> MappoController:
    """A controller on the shipped config, optionally with fields overridden."""
    if not overrides:
        return MappoController(CONFIG)
    data = json.loads(CONFIG.read_text())
    data.update(overrides)
    data["model_path"] = str(ROOT / data["model_path"])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(data, handle)
        path = handle.name
    try:
        return MappoController(path)
    finally:
        Path(path).unlink()


def _input(**overrides) -> RobotInput:
    fields = {
        "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0,
        "vx_mps": 0.0, "vy_mps": 0.0,
        "goal_x_m": 2.0, "goal_y_m": 0.0,
        "reset_run": True,
    }
    fields.update(overrides)
    return RobotInput(**fields)


# ── CORRECTION 1: the velocity frame ────────────────────────────────────────
def test_the_shipped_config_declares_the_body_frame():
    """The delivered package defaulted to ``"odom"``. ``measured`` on this stack is the
    Go2 estimator's BODY-frame velocity, and the two agree exactly at the start heading —
    so the wrong default is invisible until the robot turns."""
    assert Config.load(CONFIG).velocity_frame == "body"


def test_omitting_the_frame_takes_the_config_and_that_changes_the_observation():
    """The regression this guards: ``velocity_frame`` unset used to mean odom.

    Constructed so the two readings genuinely differ — the robot has turned 90 degrees
    since the run reset, so a body-frame (0.3, 0) is a run-local (0, 0.3). If the default
    reverts to odom, the observed velocity comes out as (0.3, 0) and this fails.
    """
    controller = _controller()
    controller.step(_input(reset_run=True))
    controller.step(_input(reset_run=False, yaw_rad=math.pi / 2, vx_mps=0.3, vy_mps=0.0))
    scale = controller.cfg.meters_per_vmas_unit
    vx_vmas, vy_vmas = controller.last_observation[2], controller.last_observation[3]
    assert math.isclose(vx_vmas * scale, 0.0, abs_tol=1e-6)
    assert math.isclose(vy_vmas * scale, 0.3, abs_tol=1e-6)


def test_a_caller_may_still_override_the_frame_per_call():
    """The config is the default, not a lock: the field stayed on ``RobotInput`` so a
    caller whose velocities really are odom can say so."""
    controller = _controller()
    controller.step(_input(reset_run=True))
    controller.step(_input(reset_run=False, yaw_rad=math.pi / 2, vx_mps=0.3,
                           velocity_frame="odom"))
    scale = controller.cfg.meters_per_vmas_unit
    assert math.isclose(controller.last_observation[2] * scale, 0.3, abs_tol=1e-6)


def test_an_unknown_frame_is_rejected_when_the_config_loads_not_when_the_robot_runs():
    """A bad frame used to raise from inside ``step``, i.e. on the first control tick of a
    live run, with the robot already standing."""
    try:
        _controller(velocity_frame="inertial")
    except ValueError as exc:
        assert "velocity_frame" in str(exc)
    else:
        raise AssertionError("an unknown velocity_frame was accepted")


# ── CORRECTION 2: the clock guard ───────────────────────────────────────────
def test_a_wall_clock_timestamp_stops_the_robot_instead_of_being_ignored():
    """THE ONE THAT FAILED OPEN. ``timestamp_s`` is compared against ``time.monotonic()``.

    Hand it an epoch and the age is about -1.8e9 s, which is under every threshold — so
    the staleness gate could never fire and the controller kept driving on whatever world
    model it last had. Remove the guard and this test gets ``COMMAND``.
    """
    result = _controller().step(_input(timestamp_s=time.time()))
    assert result.status == STOP_CLOCK_ERROR, result.status
    assert (result.vx_mps, result.vy_mps, result.vyaw_radps) == (0.0, 0.0, 0.0)
    # Against the TOLERANCE, not a hard-coded -1e8. The magnitude of the age is a
    # property of the machine's real-time clock, not of the guard: the Go2's Jetson has
    # no battery-backed RTC and boots at the epoch, so `time.time()` there reads about
    # 3e5 and the age comes out near -3e5 — negative, correctly caught, and nowhere near
    # -1e8. The old bound turned a working guard into a red suite on the one machine
    # this package ships to, which is the wrong way round for a test to fail.
    assert result.age_s < -CLOCK_TOLERANCE_S, (
        f"an epoch stamp must read as an age below the tolerance, got {result.age_s}")


def test_the_clock_tolerance_cannot_swallow_the_failure_it_guards():
    """A tolerance sized so the real fault still passes is the classic dead gate. This
    one has ten orders of magnitude of headroom: it absorbs a few ms of clock coarseness
    and nothing else."""
    controller = _controller()
    fresh = controller.step(_input(timestamp_s=time.monotonic() + 0.01))
    assert fresh.status == COMMAND, "10 ms of clock slack must not trip the guard"
    assert controller.step(_input(timestamp_s=time.monotonic() + 1.0)).status == \
        STOP_CLOCK_ERROR


def test_a_genuinely_old_input_is_stale_and_says_so_rather_than_bad_clock():
    """The two failures need different fixes — a slow producer versus a wiring mistake —
    so they must not share a status."""
    controller = _controller()
    old = time.monotonic() - (controller.cfg.stale_input_timeout_s + 0.5)
    result = controller.step(_input(timestamp_s=old))
    assert result.status == STOP_STALE_INPUT, result.status
    assert result.age_s > controller.cfg.stale_input_timeout_s


def test_the_clock_check_outranks_the_external_hold():
    """Precedence matters for the log, not the legs: both stop the robot, but a run whose
    every tick reads EXTERNAL_HOLD while the real fault is a broken clock is a debugging
    session nobody needs."""
    result = _controller().step(_input(timestamp_s=time.time(), external_hold=True))
    assert result.status == STOP_CLOCK_ERROR


def test_no_timestamp_means_the_controller_stamps_its_own_and_is_fresh():
    result = _controller().step(_input(timestamp_s=None))
    assert result.status == COMMAND
    assert 0.0 <= result.age_s < 0.05


# ── CORRECTION 3: the config is checked against the checkpoint ──────────────
def test_a_lidar_range_the_checkpoint_was_not_trained_with_is_refused():
    """``lidar_range_vmas`` is the one config field the POLICY constrains: the proximity
    convention is ``range - measured``, measured against exactly this number. Disagree and
    every value stays finite and in range, so nothing downstream notices — the robot just
    steers wrongly."""
    try:
        _controller(lidar_range_vmas=0.5)
    except ValueError as exc:
        assert "0.35" in str(exc) and "meters_per_vmas_unit" in str(exc), \
            "the error must name the trained value and point at the right knob"
    else:
        raise AssertionError("a config that contradicts the checkpoint was accepted")


def test_the_checkpoint_states_its_own_training_constants():
    """These are the numbers issue #4's calibration is derived from, and they are in the
    npz rather than in a message. @spsagar13 confirmed the same values in AIDP-567."""
    meta = _controller().actor.metadata
    assert meta["training_lidar_range_vmas"] == 0.35
    assert meta["training_agent_radius_vmas"] == 0.1
    assert meta["training_n_agents"] == 3
    assert meta["actor_input_dim"] == OBS_DIM
    assert sum(1 for n in meta["observation_layout"] if n.startswith("lidar")) == N_RAYS


def test_the_observation_layout_is_the_one_the_checkpoint_declares():
    """Positions 0-5 are x, y, vx, vy, x-gx, y-gy and 6-17 are the fan. Pinned against the
    checkpoint's own list rather than a comment, so the two cannot drift apart."""
    controller = _controller()
    controller.step(_input(x_m=1.0, y_m=2.0, goal_x_m=4.0, goal_y_m=6.0))
    layout = controller.actor.metadata["observation_layout"]
    scale = controller.cfg.meters_per_vmas_unit
    obs = controller.last_observation
    assert len(obs) == len(layout) == OBS_DIM
    # The run reset AT (1, 2), so the run-local pose is the origin and the goal is the
    # offset. That is the property being pinned: absolute odom never reaches the network.
    assert math.isclose(obs[layout.index("x_vmas")], 0.0, abs_tol=1e-6)
    assert math.isclose(obs[layout.index("y_vmas")], 0.0, abs_tol=1e-6)
    assert math.isclose(obs[layout.index("robot_x_minus_goal_x_vmas")] * scale, -3.0,
                        abs_tol=1e-5)
    assert math.isclose(obs[layout.index("robot_y_minus_goal_y_vmas")] * scale, -4.0,
                        abs_tol=1e-5)


def test_an_unknown_config_key_is_named_rather_than_dropped():
    """A misspelled key is worse than an unknown one: the field it was meant to set keeps
    its default, and the config file reads as if it took effect."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"meters_per_vmas_units": 3.0}, handle)   # note the typo
        path = handle.name
    try:
        Config.load(path)
    except ValueError as exc:
        assert "meters_per_vmas_units" in str(exc)
    else:
        raise AssertionError("a misspelled config key was silently ignored")
    finally:
        Path(path).unlink()


def test_a_scale_of_zero_is_refused_by_the_config_not_by_the_network():
    """Zero divides the whole observation into infinities, and the delivered package
    surfaced that as "observation must contain 18 finite values" from inside the actor —
    several frames from the config that caused it."""
    for bad in (0.0, -1.0, float("nan")):
        try:
            Config(meters_per_vmas_unit=bad)
        except ValueError as exc:
            assert "meters_per_vmas_unit" in str(exc)
        else:
            raise AssertionError(f"meters_per_vmas_unit={bad} was accepted")


# ── The calibration (issue #4) ──────────────────────────────────────────────
def test_the_shipped_scale_matches_the_robot_to_the_trained_agent():
    """2.5 m/unit is not a preference: it is the live runs' 0.25 m planner radius divided
    by the trained 0.10 VMAS agent radius. The delivered 1.5 matched the ROOM to the
    trained spawn region instead, which is the wrong end to fix.

    If the planner is ever run without ``--robot-radius 0.25`` this number is stale — the
    vendored default is 0.40 m, which would want 4.0. ``deploy/README.md`` pins the flag.
    """
    controller = _controller()
    assert controller.cfg.meters_per_vmas_unit == 2.5
    assert math.isclose(controller.agent_radius_m, 0.25, abs_tol=1e-9)
    assert math.isclose(controller.cfg.lidar_range_m, 0.875, abs_tol=1e-9)


def test_the_shipped_command_scale_can_actually_cross_the_arena():
    """0.6, raised from the delivered 0.3, and the arithmetic is the whole argument.

    **This robot delivers about 0.45 of the velocity it is commanded** — fitted against
    the pose over the recorded run, 2.09 m travelled against 4.32 m commanded. At 0.3 the
    top speed on the floor is 0.047 m/s, which is 2.8 m in the 60 s run budget: less than
    the 3 m arena is wide. In the closed-loop simulation that showed up as a navigation
    failure rather than as the arithmetic it is, which is exactly why it is asserted here
    rather than left as a config value nobody re-derives.
    """
    controller = _controller()
    measured_actuator_gain = 0.45
    arena_m, budget_s = 3.0, 60.0
    floor_speed = controller.cfg.max_vx_mps * controller.cfg.command_scale \
        * measured_actuator_gain
    assert controller.cfg.command_scale == 0.6
    assert floor_speed * budget_s > arena_m, (
        f"{floor_speed:.3f} m/s reaches only {floor_speed * budget_s:.1f} m in "
        f"{budget_s:.0f} s — the robot cannot cross a {arena_m} m arena")
    # It is a SPEED knob, not the safety envelope: `mappo_drive` clamps to the control
    # stack's Limits, which is what --derate scales. Pinned so raising this is never
    # mistaken for raising the ceiling.
    assert controller.cfg.max_vx_mps * controller.cfg.command_scale < 0.35


# ── The conventions, verified against the weights before being written down ──
def test_the_lidar_is_proximity_and_clear_space_reads_zero():
    """``lidar = range_max - range``: bigger means CLOSER. An empty scene is all zeros,
    which is the value a naive "no reading" fill would also produce — hence the second
    half of this test, which is the one that distinguishes them."""
    controller = _controller()
    controller.step(_input())
    assert np.allclose(controller.last_observation[6:], 0.0)

    controller.step(_input(reset_run=True, stationary_objects=[
        StationaryObject(distance_m=0.3, bearing_rad=0.0, radius_m=0.1)]))
    fan = controller.last_observation[6:]
    assert fan[0] > 0.0, "the ray pointing at the obstacle should read closest"
    assert fan[0] == max(fan)
    assert np.allclose(fan[3:9], 0.0), "rays pointing away should read clear"


def test_an_obstacle_beyond_the_horizon_is_invisible_not_faint():
    """The response is a threshold, not a ramp — measured at 1.8 degrees mean deflection
    outside the horizon against 96.6 inside. Nothing about that is fixable here; it is
    recorded because a reader will otherwise assume the fan fades in."""
    controller = _controller()
    horizon = controller.cfg.lidar_range_m
    controller.step(_input(stationary_objects=[
        StationaryObject(distance_m=horizon + 0.5, bearing_rad=0.0, radius_m=0.1)]))
    assert np.allclose(controller.last_observation[6:], 0.0)


def test_the_rays_do_not_turn_with_the_robot():
    """The trained VMAS agent never rotates — ``state.rot`` stays zero for the whole
    navigation task — so its 12 rays are at fixed angles in the world, and the run-local
    frame is this deployment's stand-in for that world. Confirmed by @spsagar13.

    Turning the robot on the spot therefore must not move the obstacle between rays. The
    velocity is zero so that the frame conversion cannot be what keeps the two equal.
    """
    controller = _controller()
    obstacle = [StationaryObject(distance_m=0.4, bearing_rad=0.0, radius_m=0.1,
                                 object_id="bin")]
    facing = controller.step(_input(stationary_objects=obstacle))
    fan_facing = controller.last_observation[6:].copy()

    # Same place, turned 90 degrees, re-seeing the same object off to its right.
    turned = controller.step(_input(reset_run=False, yaw_rad=math.pi / 2,
                                    stationary_objects=[StationaryObject(
                                        distance_m=0.4, bearing_rad=-math.pi / 2,
                                        radius_m=0.1, object_id="bin")]))
    assert np.allclose(fan_facing, controller.last_observation[6:], atol=1e-6)
    assert math.isclose(facing.action_x, turned.action_x, abs_tol=1e-6)
    # ...and the BODY command does rotate, because that half is body-relative.
    assert not math.isclose(facing.vy_mps, turned.vy_mps, abs_tol=1e-3)


def test_the_run_local_frame_is_fixed_at_the_reset_not_re_derived_each_tick():
    controller = _controller()
    controller.step(_input(x_m=5.0, y_m=-3.0, yaw_rad=1.1, reset_run=True))
    assert np.allclose(controller.last_observation[:2], 0.0)
    controller.step(_input(x_m=5.0, y_m=-3.0, yaw_rad=1.1, reset_run=False,
                           goal_x_m=5.0, goal_y_m=-3.0))
    assert np.allclose(controller.last_observation[:2], 0.0)


def test_reset_run_forgets_the_obstacles_the_previous_run_mapped():
    """They are held in the OLD run-local frame, so carrying them over would place them
    at a wrong offset rather than merely stale."""
    controller = _controller()
    controller.step(_input(stationary_objects=[
        StationaryObject(distance_m=0.3, bearing_rad=0.0, radius_m=0.1)]))
    assert controller._obstacles
    controller.step(_input(reset_run=True))
    assert not controller._obstacles
    assert np.allclose(controller.last_observation[6:], 0.0)


def test_two_objects_closer_than_the_association_radius_merge_without_ids():
    """The documented limit, pinned so it is a known cost rather than a surprise. It is
    also why ``id`` was added to the telemetry in the first place."""
    controller = _controller()
    controller.step(_input(stationary_objects=[
        StationaryObject(distance_m=0.40, bearing_rad=0.0, radius_m=0.05),
        StationaryObject(distance_m=0.60, bearing_rad=0.0, radius_m=0.05)]))
    assert len(controller._obstacles) == 1


def test_distinct_ids_keep_two_close_objects_apart():
    """CORRECTION 4, and the one that made adding ``id`` to the telemetry worthwhile.

    The delivered fallback ran on EVERY mapped obstacle, so a detection carrying ``"b"``
    was absorbed by the obstacle already identified as ``"a"`` whenever the two were
    within 0.45 m — which is the situation ids exist to resolve. Silent, too: the merged
    disc keeps the larger radius, so the range vector stays plausible. Revert the fix and
    this reads 1.
    """
    controller = _controller()
    controller.step(_input(stationary_objects=[
        StationaryObject(distance_m=0.40, bearing_rad=0.0, radius_m=0.05, object_id="a"),
        StationaryObject(distance_m=0.60, bearing_rad=0.0, radius_m=0.05, object_id="b")]))
    assert len(controller._obstacles) == 2
    assert {o.object_id for o in controller._obstacles} == {"a", "b"}


def test_an_identified_detection_still_adopts_an_anonymous_obstacle_it_matches():
    """The fallback is narrowed, not removed: an obstacle mapped before its producer had
    an identity is the same object, and it takes the id on the next sighting. Without
    this, turning ids on mid-run would double every landmark."""
    controller = _controller()
    controller.step(_input(stationary_objects=[
        StationaryObject(distance_m=0.40, bearing_rad=0.0, radius_m=0.05)]))
    controller.step(_input(reset_run=False, stationary_objects=[
        StationaryObject(distance_m=0.42, bearing_rad=0.0, radius_m=0.05,
                         object_id="landmark-1")]))
    assert len(controller._obstacles) == 1
    assert controller._obstacles[0].object_id == "landmark-1"


def test_an_obstacle_is_forgotten_once_its_ttl_expires():
    controller = _controller(static_obstacle_ttl_s=1.0)
    base = time.monotonic()
    controller.step(_input(timestamp_s=base, stationary_objects=[
        StationaryObject(distance_m=0.3, bearing_rad=0.0, radius_m=0.1)]))
    assert len(controller._obstacles) == 1
    # Two seconds later, unseen. The timestamp is the controller's clock, so the age is
    # measured on the same monotonic base the guard above checks.
    controller.step(_input(reset_run=False, timestamp_s=base + 2.0))
    assert not controller._obstacles


# ── The stop conditions ─────────────────────────────────────────────────────
def test_arriving_stops_and_says_which_authority_stopped_it():
    result = _controller().step(_input(goal_x_m=0.1, goal_y_m=0.0))
    assert result.status == STOP_GOAL_REACHED
    assert (result.vx_mps, result.vy_mps) == (0.0, 0.0)


def test_an_external_hold_zeroes_the_command_but_still_reports_the_intent():
    """The action is what a shadow run records. Zeroing it as well would make every held
    tick indistinguishable from a policy that had nothing to say."""
    result = _controller().step(_input(external_hold=True, stationary_objects=[
        StationaryObject(distance_m=0.3, bearing_rad=0.5, radius_m=0.1)]))
    assert result.status == STOP_EXTERNAL_HOLD
    assert (result.vx_mps, result.vy_mps) == (0.0, 0.0)
    assert (result.action_x, result.action_y) != (0.0, 0.0)


def test_the_policy_never_commands_yaw():
    """It outputs a 2-D force and nothing else, so the robot crabs rather than turns.

    That is a real limitation of this checkpoint and not a bug here: the camera is a
    forward 85-degree cone, so a robot that never turns never looks anywhere new.
    ``integration/mappo_policy.py`` adds a heading servo for the drive path; this pins
    that the policy itself supplies no yaw.
    """
    result = _controller().step(_input(stationary_objects=[
        StationaryObject(distance_m=0.3, bearing_rad=1.0, radius_m=0.1)]))
    assert result.vyaw_radps == 0.0


def test_the_command_is_inside_the_configured_envelope():
    controller = _controller()
    result = controller.step(_input(stationary_objects=[
        StationaryObject(distance_m=0.3, bearing_rad=2.0, radius_m=0.2)]))
    ceiling_x = controller.cfg.max_vx_mps * controller.cfg.command_scale
    ceiling_y = controller.cfg.max_vy_mps * controller.cfg.command_scale
    assert abs(result.vx_mps) <= ceiling_x + 1e-9
    assert abs(result.vy_mps) <= ceiling_y + 1e-9


def test_the_same_input_gives_the_same_action():
    """The deployment takes ``tanh(loc)`` rather than sampling the ``TanhNormal``, so
    there is no run-to-run variation to chase when a live run misbehaves."""
    first = _controller().step(_input(stationary_objects=[
        StationaryObject(distance_m=0.4, bearing_rad=0.3, radius_m=0.1)]))
    second = _controller().step(_input(stationary_objects=[
        StationaryObject(distance_m=0.4, bearing_rad=0.3, radius_m=0.1)]))
    assert (first.action_x, first.action_y) == (second.action_x, second.action_y)


def test_a_malformed_detection_is_dropped_rather_than_mapped_to_a_ghost():
    controller = _controller()
    controller.step(_input(stationary_objects=[
        StationaryObject(distance_m=float("nan"), bearing_rad=0.0, radius_m=0.1),
        StationaryObject(distance_m=-1.0, bearing_rad=0.0, radius_m=0.1),
        StationaryObject(distance_m=0.3, bearing_rad=float("inf"), radius_m=0.1)]))
    assert not controller._obstacles


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"physical_ai_mappo: {len(tests)}/{len(tests)} passed")
