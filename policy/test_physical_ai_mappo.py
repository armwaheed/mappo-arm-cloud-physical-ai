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

import contextlib
import io
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


@contextlib.contextmanager
def _stderr():
    """Capture stderr, and hand back a callable returning what was written.

    ``MappoController`` warns on stderr the first time :data:`STOP_GOAL_REACHED` fires.
    That is signal on a live run and noise in a suite that drives the branch on purpose,
    and a warning a reader learns to scroll past is a warning that is not there.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        yield buffer.getvalue


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
    """The shipped 1.0 both clears the measured gait floor and crosses the arena.

    This Go2 delivered 0.70 of command at full speed and only 0.45 when derated. More
    importantly, its gait did not sustain at 0.6 x 0.35 = 0.21 m/s in five runs. A test
    that checked distance alone used to approve that unwalkable setting, so both physical
    constraints are explicit here.
    """
    controller = _controller()
    measured_actuator_gain = 0.70
    arena_m, budget_s = 3.0, 60.0
    commanded_speed = controller.cfg.max_vx_mps * controller.cfg.command_scale
    achieved_speed = commanded_speed * measured_actuator_gain
    assert controller.cfg.command_scale == 1.0
    assert commanded_speed >= 0.35, "the shipped policy command is below the gait floor"
    assert achieved_speed * budget_s > arena_m, (
        f"{achieved_speed:.3f} m/s reaches only {achieved_speed * budget_s:.1f} m in "
        f"{budget_s:.0f} s — the robot cannot cross a {arena_m} m arena")
    # It is a SPEED knob, not the safety envelope: `mappo_drive` clamps to the control
    # stack's Limits, which is what --derate scales. Pinned so raising this is never
    # mistaken for raising the ceiling.
    assert controller.cfg.max_vx_mps * controller.cfg.command_scale <= 0.35


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
    with _stderr():                       # the second step is AT the goal; see below
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


def test_a_named_obstacle_takes_the_radius_the_producer_reports_now():
    """CORRECTION 6. The delivered ``max`` made the mapped radius a high-water mark.

    The control stack's radius is ``radius_m + position_sigma`` — an estimate that starts
    large and CONVERGES as sightings accumulate, which is the whole reason a long approach
    was prescribed as the fix for a gap that looked too narrow. Taking ``max`` meant none
    of that convergence ever reached the policy: it planned for the rest of the run
    against the map's least certain moment. Measured on the 2026-08-18 two-bin runs, every
    landmark converged to 0.230 m while the controller held 0.379-0.472 m.

    Revert to ``max(match.radius, radius)`` and the radius reads 0.40.
    """
    controller = _controller()
    controller.step(_input(stationary_objects=[
        StationaryObject(distance_m=0.50, bearing_rad=0.0, radius_m=0.40,
                         object_id="landmark-1")]))
    assert math.isclose(controller._obstacles[0].radius, 0.40)
    controller.step(_input(reset_run=False, stationary_objects=[
        StationaryObject(distance_m=0.50, bearing_rad=0.0, radius_m=0.15,
                         object_id="landmark-1")]))
    assert len(controller._obstacles) == 1
    assert math.isclose(controller._obstacles[0].radius, 0.15)


def test_a_shrinking_radius_reopens_a_gap_the_fan_can_see():
    """The behavioural half of CORRECTION 6, and the reason it is not cosmetic.

    Two discs at +/-15 deg and 1.00 m, straddling the ray at 0 rad. Mapped at 0.40 m each
    subtends 23.6 deg, the two shadows overlap across the gap, and the ray reads blocked;
    at the converged 0.23 m each subtends 13.3 deg, the shadows part, and the same ray
    reads clear. Same objects, same ray count, same pose — only the map's certainty
    changed. Under ``max`` the first reading is latched and the second never happens.
    """
    controller = _controller()
    far = [StationaryObject(distance_m=1.00, bearing_rad=math.radians(+15),
                            radius_m=0.40, object_id="a"),
           StationaryObject(distance_m=1.00, bearing_rad=math.radians(-15),
                            radius_m=0.40, object_id="b")]
    near = [StationaryObject(distance_m=1.00, bearing_rad=math.radians(+15),
                             radius_m=0.23, object_id="a"),
            StationaryObject(distance_m=1.00, bearing_rad=math.radians(-15),
                             radius_m=0.23, object_id="b")]
    controller.step(_input(stationary_objects=far))
    blocked = controller.last_observation[OBS_DIM - N_RAYS]
    controller.step(_input(reset_run=False, stationary_objects=near))
    clear = controller.last_observation[OBS_DIM - N_RAYS]
    assert blocked > 0.0, "the inflated discs should close the ray at 0 rad"
    assert clear == 0.0, "the converged discs should leave it open"


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
    with _stderr():
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


# ── Issue #25: what the goal overrun actually is ────────────────────────────
#
# MEASURED, from the seven arriving runs this repository has recorded, across four
# sessions and two `--arrive` tolerances:
#
#   | run                          | arrive | stopped | over   | that tick moved |
#   | 2026-08-14 hero              |  0.80  | 0.7680  | 0.0320 | 0.0509          |
#   | 2026-08-17 run0 baseline     |  0.80  | 0.7959  | 0.0041 | 0.0159          |
#   | 2026-08-18 runA veto-on      |  0.80  | 0.7503  | 0.0497 | 0.0536          |
#   | 2026-08-18 runB veto-off     |  0.80  | 0.7471  | 0.0529 | 0.0530          |
#   | 2026-08-18 run11 two bins    |  0.80  | 0.7713  | 0.0287 | 0.0337          |
#   | 2026-08-25 contrast          |  0.30  | 0.2976  | 0.0024 | 0.0050          |
#   | 2026-08-25 hero              |  0.30  | 0.2595  | 0.0405 | 0.0616          |
#
# The overrun is SMALLER THAN THE DISTANCE THAT TICK TRAVELLED in 7 of 7. It correlates
# with that distance at r=+0.92 and with the tick interval at r=+0.80; a braking distance
# would not be bounded by a tick's travel at all, and would not depend on the interval.
# It is a threshold sampled once per tick, and nothing else — the commanded speed over
# the four ticks before the stop is FLAT in every run (0.340 -> 0.343, 0.352 -> 0.352,
# 0.339 -> 0.336, 0.350 -> 0.350), so there is no deceleration to attribute it to.
#
# The two tests below pin that in the policy's own code, because the same mechanism sets
# the residual past `goal_stop_distance_m`.

#: Commanded speed on the tick before the threshold was crossed, m/s — the median of the
#: seven runs above. Used to drive a synthetic approach at a realistic pace.
MEASURED_APPROACH_M_S = 0.34
#: Median tick interval of those runs, seconds. The loop is nominally 10 Hz and holds
#: 3.8-4.1 Hz live; that gap is issue #18 and it is what sizes the residual below.
MEASURED_TICK_S = 0.26


def _approach(controller, start_distance_m, speed_m_s, tick_s):
    """Walk straight at a goal 2 m along +x and return the distance at the first stop.

    A grid of samples along a line, which is what a control loop is: the stop test sees
    the pose at tick boundaries and nothing in between.
    """
    distance = start_distance_m
    reset = True
    while distance > 0.0:
        result = controller.step(_input(x_m=2.0 - distance, y_m=0.0,
                                        vx_mps=speed_m_s, goal_x_m=2.0, goal_y_m=0.0,
                                        reset_run=reset))
        reset = False
        if result.status == STOP_GOAL_REACHED:
            return distance
        distance -= speed_m_s * tick_s
    return None


def test_the_overrun_past_the_goal_stop_is_one_ticks_travel_not_the_thresholds_value():
    """🔴 THE MEASUREMENT ISSUE #25 ASKED FOR. Moving the stop distance does not move the
    overrun; the tick interval does.

    Issue #25 attributes ~5 cm to "deceleration plus perception lag" and proposes backing
    the stack's ``--arrive`` off by 10 cm. Backing it off moves where the robot stops, and
    this shows it leaves the overrun exactly where it was: the residual past a threshold
    tested once per tick is bounded by that tick's travel and is independent of the
    threshold's value. Halving the interval halves it. That is a loop-rate property —
    issue #18 — and no value of ``goal_stop_distance_m`` or ``--arrive`` can reach it.

    Swept over eight sub-tick start offsets so the result is the shape of the residual
    rather than one lucky alignment of the sample grid with the threshold.
    """
    def residuals(stop_m, tick_s):
        step_m = MEASURED_APPROACH_M_S * tick_s
        out = []
        for k in range(8):
            controller = _controller(goal_stop_distance_m=stop_m)
            with _stderr():
                stopped = _approach(controller, 1.30 + k * step_m / 8.0,
                                    MEASURED_APPROACH_M_S, tick_s)
            assert stopped is not None, (stop_m, tick_s, k)
            out.append(stop_m - stopped)
        return out

    one_tick = MEASURED_APPROACH_M_S * MEASURED_TICK_S
    worst = {}
    for stop_m in (0.20, 0.50, 0.90):
        measured = residuals(stop_m, MEASURED_TICK_S)
        assert all(0.0 <= r < one_tick for r in measured), (stop_m, measured)
        worst[stop_m] = max(measured)

    # The bound is reached, so "smaller than one tick" is not vacuous...
    assert min(worst.values()) > 0.9 * one_tick, worst
    # ...and it is the SAME bound at 0.20 m and at 0.90 m. A 4.5x change in the threshold
    # moves the worst case by less than a tenth of a tick's travel.
    assert max(worst.values()) - min(worst.values()) < 0.1 * one_tick, worst

    # The one thing that does move it. Half the interval, half the residual.
    half = max(residuals(0.20, MEASURED_TICK_S / 2.0))
    assert half < 0.6 * worst[0.20], (half, worst[0.20])


def test_the_goal_stop_is_a_hard_zero_and_there_is_no_ramp_to_shorten():
    """Why raising the threshold trades a 5 cm overrun for a failed run.

    On the last tick outside the stop radius the policy is still commanding most of its
    forward envelope; on the first tick inside it, zero. There is no deceleration in
    between — so a bigger stop radius does not make the approach gentler, it just stops
    the robot further out, and ``visual_nav`` does not end a run on this status.

    The Go2 could not use a ramp anyway: ``avoidance.MIN_GAIT_COMMAND_M_S`` is 0.35 and
    equals this robot's forward ceiling, so ``mappo_drive._at_least_walking_pace`` scales
    any slower command back UP to the envelope. This robot has one approach speed.
    """
    controller = _controller()
    last_moving = None
    distance = 1.30
    reset = True
    while distance > 0.0:
        with _stderr():
            result = controller.step(_input(x_m=2.0 - distance, y_m=0.0,
                                            vx_mps=MEASURED_APPROACH_M_S,
                                            goal_x_m=2.0, goal_y_m=0.0, reset_run=reset))
        reset = False
        if result.status == STOP_GOAL_REACHED:
            break
        last_moving = result
        distance -= MEASURED_APPROACH_M_S * MEASURED_TICK_S

    assert result.status == STOP_GOAL_REACHED
    assert (result.vx_mps, result.vy_mps) == (0.0, 0.0)
    assert last_moving is not None
    # Most of the envelope, one tick before a full stop. `max_vx_mps` is 0.35.
    assert last_moving.vx_mps > 0.6 * controller.cfg.max_vx_mps, last_moving
    # And the action itself never ramped to nothing — the stop is the `if`, not the net.
    assert abs(last_moving.action_x) > 0.5, last_moving


def test_the_distance_the_goal_stop_tests_is_the_one_the_caller_already_tested():
    """Why ``goal_stop_distance_m`` has fired on zero ticks of seven arriving runs, and
    why that is structural rather than lucky.

    ``_local_state`` maps odom into the run-local frame by a rotation about the reset
    pose and a translation — a rigid transform, which preserves distance. So the scalar
    this stop compares is the SAME one ``visual_nav`` compares against ``--arrive``, from
    the same pose on the same tick, and the stack breaks its loop first whenever
    ``--arrive`` is the larger of the two. Reachable only by making it larger, which
    ``test_the_goal_stop_is_a_hard_zero_...`` shows is worse.

    Pinned by giving the controller a start pose that is translated and rotated: if the
    stop were testing a frame-dependent quantity, these would disagree.
    """
    stop_m = _controller().cfg.goal_stop_distance_m
    inside, outside = stop_m * 0.9, stop_m * 1.1

    for x0, y0, yaw0 in [(0.0, 0.0, 0.0), (7.5, -3.25, 2.4), (-1.0, 12.0, -0.7)]:
        for offset, expected in ((inside, STOP_GOAL_REACHED), (outside, COMMAND)):
            controller = _controller()
            controller.step(_input(x_m=x0, y_m=y0, yaw_rad=yaw0,
                                   goal_x_m=x0 + 4.0, goal_y_m=y0, reset_run=True))
            # Placed at `offset` from the goal along a direction unrelated to the reset
            # heading, so a frame-dependent test would land somewhere else entirely.
            with _stderr():
                result = controller.step(_input(
                    x_m=x0 + 4.0 - offset * math.cos(0.9),
                    y_m=y0 - offset * math.sin(0.9), yaw_rad=yaw0 + 1.3,
                    goal_x_m=x0 + 4.0, goal_y_m=y0, reset_run=False))
            assert result.status == expected, (x0, y0, yaw0, offset, result.status)


def test_the_goal_stop_says_so_on_stderr_the_first_time_it_fires_and_only_then():
    """The loud half of issue #25's third proposal, at the only place this module can be
    loud: a refusal at load would need ``--arrive``, which is a per-run CLI flag on the
    other side of the adapter and is not in :class:`Config`.

    It has to fire on the branch and not on the config, and it has to fire ONCE — the
    condition holds for every remaining tick of the run.
    """
    controller = _controller()
    with _stderr() as written:
        controller.step(_input(goal_x_m=0.05, goal_y_m=0.0))
        first = written()
    assert "STOP_GOAL_REACHED" in first
    assert "goal_stop_distance_m" in first
    assert "timeout" in first, "the failure a reader has to recognise is a timeout"
    assert first.count("STOP_GOAL_REACHED") == 1

    with _stderr() as written:
        for _ in range(5):
            controller.step(_input(goal_x_m=0.05, goal_y_m=0.0, reset_run=False))
        assert written() == "", "five more held ticks must not print five more lines"

    # A NEW RUN re-arms it. Two runs in one process are two runs, and the second
    # operator is owed the same sentence the first got.
    with _stderr() as written:
        controller.step(_input(goal_x_m=0.05, goal_y_m=0.0, reset_run=True))
        assert "STOP_GOAL_REACHED" in written()

    # And it stays silent on a run that never reaches the threshold, which is every run
    # this repository has recorded.
    with _stderr() as written:
        _controller().step(_input(goal_x_m=2.0, goal_y_m=0.0))
        assert written() == ""


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"physical_ai_mappo: {len(tests)}/{len(tests)} passed")
