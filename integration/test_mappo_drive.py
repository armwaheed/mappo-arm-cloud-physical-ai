#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the drive path — the one file that can put the policy in charge of legs.

``visual_nav`` needs OpenCV and a robot, so it is never imported here. What IS tested is
everything that decides what the legs are told: the veto, the envelope clamp, the stop
mapping, and the patch guard that refuses to run against a stack whose globals have moved.
A stub module stands in for ``visual_nav`` in the patch tests, which is the right size of
fake — the thing under test is the substitution, not the stack.

Needs the policy package (numpy) and the vendored planner.
Run: ``python3 test_mappo_drive.py``
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# BOTH paths go in before ANY sibling import. `ruff --fix` sorts imports into
# contiguous blocks and will hoist `from avoidance import ...` above a sys.path line that
# sits between the blocks — which is exactly how this file went from passing to
# ModuleNotFoundError without anybody touching a test.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "robot-stack", "unitree", "go2",
                                "visual_nav"))
from avoidance import (
    MIN_GAIT_COMMAND_M_S,
    PLANNER_REASONS,
    Command,
    Limits,
    Obstacle,
    PlannerConfig,
    base_reason,
)

import mappo_bridge
from mappo_bridge import external_hold
from mappo_drive import (
    _STOP_REASONS,
    SUB_FLOOR_PROGRESS_FRACTION,
    SUB_FLOOR_WINDOW_S,
    MappoPlanner,
    _add_arguments,
    _record_refusal,
    peer_navigator,
    platform_gait_floor,
    split_argv,
)
from mappo_policy import (
    DEFAULT_PACKAGE,
    GOAL,
    TRAVEL,
    HeadingServo,
    PolicyRunner,
)
from peer_source import PEER_TIMEOUT_S, Alignment, PeerSource, spool_document, write_spool
from replay_mappo import derived_config

BIN = Obstacle(x=1.0, y=0.0, vx=0.0, vy=0.0, radius_m=0.23, kind="static",
               object_id="landmark-1")
WALKER = Obstacle(x=1.4, y=0.3, vx=0.6, vy=0.0, radius_m=0.5, kind="tracked",
                  object_id="track-1")


class _Loco:
    """Just the one method :meth:`MappoPlanner.attach` needs."""

    def __init__(self, velocity=(0.0, 0.0, 0.0)):
        self._velocity = velocity

    def velocity(self):
        return self._velocity


class _StubRunner:
    """A policy that always proposes the same thing.

    The veto tests use this rather than the real checkpoint, because a test whose branch
    coverage depends on what a set of weights happens to decide is not a test of the
    branch. It has already flipped once: at the delivered ``command_scale`` of 0.3 the
    real policy's rollout could not even reach a bin 1.0 m away, so nothing was ever
    vetoed; at the later 0.6 experiment it was vetoed every time. Injecting the command
    keeps the branch under the test's control, and
    ``test_the_real_checkpoint_needs_the_veto_at_the_shipped_command_scale`` records what
    the weights actually do, separately and by name.
    """

    def __init__(self, command=(0.35, 0.0, 0.0), status: str = "COMMAND"):
        self._command, self._status = command, status
        # Wraps a real runner rather than faking its whole surface: the planner reads
        # `controller.actor.metadata` for the radius calibration check and `config` for
        # the error message, and a stub that faked those would be asserting they exist
        # rather than that they are right. Only `step` is substituted.
        real = PolicyRunner(DEFAULT_PACKAGE)
        self.controller, self.config = real.controller, real.config

    def step(self, tick, monotonic_s=None):
        from mappo_policy import PolicyStep
        return PolicyStep(status=self._status, vx_mps=self._command[0],
                          vy_mps=self._command[1], wz_radps=self._command[2],
                          action_x=1.0, action_y=0.0, intent_bearing_rad=0.0,
                          age_s=0.0, observation=())


def _planner(supervised: bool = True, limits: Limits | None = None,
             servo: HeadingServo | None = None, runner=None,
             gait_floor_m_s: float = 0.0,
             platform_floor_m_s: float = 0.0) -> MappoPlanner:
    # Built with stdout swallowed: the constructor states how the floor and the envelope
    # compose, which is a run-log line and would otherwise interleave with the `  ok  `
    # lines this suite's output is read as. The announcement itself is asserted by
    # `test_a_floor_above_the_envelope_is_announced_when_the_planner_is_built`, which
    # constructs the planner directly for exactly that reason.
    with contextlib.redirect_stdout(io.StringIO()):
        planner = MappoPlanner(limits or Limits(), PlannerConfig(robot_radius_m=0.25),
                               runner or PolicyRunner(DEFAULT_PACKAGE, servo=servo),
                               supervised=supervised, gait_floor_m_s=gait_floor_m_s,
                               platform_floor_m_s=platform_floor_m_s)
    planner.attach(_Loco())
    return planner


def _drive_sub_floor(planner: MappoPlanner, speed: float, ticks: int,
                     delivered: float, dt: float = 0.33) -> str:
    """Feed ``ticks`` commands of ``speed`` m/s, moving the pose at ``delivered`` m/s.

    Straight into :meth:`MappoPlanner._note_sub_floor` rather than through ``plan()``,
    because what is under test is the JUDGEMENT and a scripted pose is the only way to
    replay a measured run — the planner would otherwise re-decide the command from a
    scene, which is the thing that has to be held constant. ``plan()``'s own arrival at
    this method is pinned separately, on each of its branches, by the tests below.

    Returns everything the planner printed while it was judging.
    """
    now, x = 100.0, 0.0
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for _ in range(ticks):
            # `plan()` counts the tick before it judges it, and `report()` prints the
            # sub-floor count as a fraction of it. Counting here keeps that denominator
            # true, so a report string asserted in a test is the one a run would print.
            planner.counts["ticks"] += 1
            planner._note_sub_floor(Command(speed, 0.0, 0.0, reason="veto-avoid",
                                            gap_m=1.0), (x, 0.0, 0.0), now)
            x += delivered * dt
            now += dt
    return out.getvalue()


# ── The veto ────────────────────────────────────────────────────────────────
def test_the_veto_fires_on_a_command_that_drives_into_the_obstacle():
    """Full ahead from 1.0 m out, at a 0.23 m disc, with 0.25 m of robot radius. The
    2.5 s rollout ends well inside it."""
    planner = _planner(supervised=True, runner=_StubRunner((0.35, 0.0, 0.0)))
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [BIN])
    assert planner.counts["vetoed"] == 1, planner.counts
    assert command.reason.startswith("veto-")
    assert command.reason != "veto-", "the planner's own reason must survive"


def test_the_same_command_is_not_vetoed_when_supervision_is_off():
    """Pins that the veto is what makes the difference, not the scene. If both modes
    behaved the same, ``--policy-mode`` would be a placebo and the simulation's
    "supervised did not collide" would be about nothing."""
    raw = _planner(supervised=False, runner=_StubRunner((0.35, 0.0, 0.0)))
    command = raw.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [BIN])
    assert raw.counts["vetoed"] == 0 and raw.counts["policy"] == 1
    assert command.reason == "policy" and command.vx == 0.35


def test_a_veto_issues_the_planner_command_rather_than_a_stop():
    """Falling back to zero would make every veto a hold, and a hold beside a static
    obstacle is exactly the deadlock the policy is there to break."""
    planner = _planner(supervised=True, runner=_StubRunner((0.35, 0.0, 0.0)))
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [BIN])
    assert (command.vx, command.vy, command.wz) != (0.35, 0.0, 0.0)
    assert command.gap_m is not None, "the fallback carries the planner's own gap"


def test_a_command_that_clears_the_obstacle_is_not_vetoed():
    """A veto that refuses everything is not a safety feature, it is a brake."""
    planner = _planner(supervised=True, runner=_StubRunner((0.0, 0.20, 0.0)))
    planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [BIN])
    assert planner.counts["vetoed"] == 0, planner.counts


def test_an_empty_scene_is_never_vetoed():
    planner = _planner(supervised=True, runner=_StubRunner((0.35, 0.0, 0.0)))
    planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [])
    assert planner.counts["vetoed"] == 0


def test_the_reason_a_veto_writes_is_one_base_reason_can_read():
    """⚠️ THE PRODUCER HALF OF ISSUE #118. The veto QUALIFIES the planner's reason
    rather than replacing it, which is right — one string then says both what the
    planner decided and that the policy's command was refused, and the telemetry needs
    both. It is only safe while every consumer can get the planner's word back out.

    Three could not: ``visual_nav``'s rest-after-blocked dispatch, both of the planner's
    Schmitt triggers, and ``mappo_bridge.external_hold``. This drives every branch of
    ``_choose`` that can produce a qualified reason and pins the round trip.

    Change the separator, or qualify with a word ``PLANNER_REASONS`` does not name, and
    this goes red at the point of writing rather than in a run nobody is watching.
    """
    close = Obstacle(x=0.6, y=0.0, vx=0.0, vy=0.0, radius_m=0.23, kind="static",
                     object_id="landmark-1")
    qualified = {}
    for name, obstacles, last in (("hold", [close], "goal"),
                                  ("avoid", [BIN], "goal"),
                                  ("avoid-continuing", [BIN], "veto-avoid")):
        planner = _planner(supervised=True, runner=_StubRunner((0.35, 0.0, 0.0)))
        command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), obstacles,
                               last_reason=last)
        assert planner.counts["vetoed"] == 1, (name, planner.counts)
        qualified[name] = command.reason

    assert qualified["hold"] == "veto-hold", qualified
    assert set(qualified.values()) >= {"veto-hold", "veto-avoid"}, (
        f"the scenes must produce more than one qualified word or the round trip below "
        f"is tested on one case: {qualified}")
    for name, reason in qualified.items():
        assert reason.startswith("veto-"), (name, reason)
        assert base_reason(reason) in PLANNER_REASONS, (
            f"{name}: `{reason}` does not strip back to a word the planner issues, so "
            f"every consumer comparing it will silently never match")
        assert base_reason(reason) == reason[len("veto-"):], (name, reason)


def test_a_supervised_approach_parks_under_a_veto_that_is_not_a_hold():
    """WHY ``visual_nav.blocked_stop`` MEASURES THE LEGS AND NOT ONLY THE LABEL (#118).

    A prefix-aware fix that still asked ``base_reason(...) == "hold"`` would leave the
    robot braced in the scene the demo actually runs, and this is the run that says so.
    A policy walking straight at a bin 2.6 m ahead, pose integrated at the 0.45 factor
    this robot delivers, 10 Hz: the veto starts firing at about 1.1 m of surface gap,
    and within half a second the command it substitutes is EXACTLY zero and stays there
    — because the planner's stopping-distance cap has ratcheted the dynamic window down
    until the best sampled candidate is zero, and ``avoid`` is a label applied after
    that choice.

    So the terminal state of this approach is a robot stopped dead, indefinitely, under
    a reason whose planner word is ``avoid``. The bounds below are loose because they
    are describing a trajectory rather than pinning a constant; what is not loose is the
    two facts the design rests on: it is a stop, and it is not a hold.
    """
    planner = _planner(supervised=True, runner=_StubRunner((0.35, 0.0, 0.0)))
    bin_ = Obstacle(x=2.6, y=0.0, vx=0.0, vy=0.0, radius_m=0.23, kind="static",
                    object_id="landmark-1")
    pose, last, last_reason, dt = [0.0, 0.0, 0.0], (0.0, 0.0, 0.0), "goal", 0.1
    trail = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(120):
            command = planner.plan(tuple(pose), (4.0, 0.0), last, [bin_],
                                   control_dt=dt, last_reason=last_reason)
            trail.append(command)
            last, last_reason = (command.vx, command.vy, command.wz), command.reason
            # The measured delivery factor, not the command: this robot walks about 0.45
            # of what it is told, which is why the veto has time to fire at all.
            pose[0] += command.vx * 0.45 * dt
            pose[1] += command.vy * 0.45 * dt
            pose[2] += command.wz * dt

    assert any(c.reason == "policy" for c in trail), (
        "the policy never drove, so this is not the scene the test describes")
    parked = [c for c in trail if c.is_stop]
    # It parks at about tick 71 of 120 and never moves again. A handful of stopped
    # ticks would be a deceleration; thirty of them in a row is the braced robot.
    assert len(parked) >= 30, (
        f"the approach did not park, so it says nothing about a braced robot: "
        f"{len(parked)} stopped ticks of {len(trail)}")
    assert all(c.is_stop for c in trail[-30:]), (
        "the stop must be terminal, not a pause in the middle of an approach")

    final = trail[-1]
    assert final.is_stop, final
    assert final.reason.startswith("veto-"), final
    assert base_reason(final.reason) != "hold", (
        f"this run's terminal reason IS a hold, so the scene no longer demonstrates "
        f"what it was added for — re-measure before relaxing `blocked_stop`: {final}")
    assert base_reason(final.reason) in PLANNER_REASONS, final

    surface_gap = math.hypot(bin_.x - pose[0], bin_.y - pose[1]) - bin_.radius_m - 0.25
    assert 0.8 < surface_gap < 1.4, (
        f"the robot parked {surface_gap:.2f} m off the bin's surface, which is outside "
        f"the band this scene was measured in")


def test_the_bridges_copy_of_the_reason_rule_matches_the_planners():
    """``mappo_bridge`` carries its own ``base_reason`` because it is stdlib-only and
    ``avoidance`` is numpy from its first line. A copy is only safe while something
    compares the two, so this does — over the vocabulary, over both qualifiers anyone
    writes, and over the near-misses a substring test would get wrong.

    Built from ``PLANNER_REASONS`` rather than written out, so a fifth word added to the
    vocabulary is covered here without anyone remembering to extend a list.
    """
    assert mappo_bridge.PLANNER_REASONS == PLANNER_REASONS, (
        f"the bridge names a different vocabulary: {mappo_bridge.PLANNER_REASONS}")
    words = [*PLANNER_REASONS, "policy", "threshold", "holding_pattern",
             "withhold", "", "-", "veto-"]
    cases = [*words, *(f"{q}-{w}" for q in ("veto", "peer", "") for w in words)]
    for reason in cases:
        assert mappo_bridge.base_reason(reason) == base_reason(reason), reason
    # Anti-vacuity: the comparison has to run over cases where the rule actually does
    # something, or two broken copies agreeing would pass.
    assert sum(1 for r in cases if base_reason(r) != r) >= len(PLANNER_REASONS), cases


def test_the_real_checkpoint_needs_the_veto_at_the_shipped_command_scale():
    """The counterpart to the stub tests, and it changed sign when `command_scale` went
    above the delivered 0.3 — which is worth pinning rather than
    forgetting.

    At 0.3 the policy's command covers 0.35 x 0.3 x 2.5 = 0.26 m over the planner's
    horizon, which cannot reach a bin 1.0 m away, so nothing was ever vetoed and the veto
    looked dormant. At the shipped 1.0 the rollout reaches 0.88 m and the swerve no
    longer clears the disc over the full horizon, so the veto fires. The closed-loop
    measurement agrees:
    **near the obstacle the policy's proposal is infeasible more often than not** — 61%
    of the ticks that have one. That is the number to check on the day, and
    ``MappoPlanner.report()`` prints it.
    """
    planner = _planner(supervised=True)
    for _ in range(5):
        planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [BIN])
    assert planner.counts["vetoed"] == 5, planner.counts
    assert "5 vetoed" in planner.report()


# ── The radius calibration, which is a trap rather than a bug ───────────────
def test_the_planner_default_radius_is_refused_because_it_breaks_the_calibration():
    """THE TRAP. ``meters_per_vmas_unit`` is the planner's robot radius divided by the
    checkpoint's trained agent radius, but only one of those two lives in the policy's
    config. The vendored planner's default is 0.40 m and every recorded run passed
    ``--robot-radius 0.25``, so a run at the default gives the policy a 1.4 m sensing
    horizon instead of 0.875 m — every closed-loop number stops applying and nothing
    anywhere says so. Now it refuses, before a leg moves."""
    log = Path(tempfile.mkdtemp()) / "refusals.jsonl"
    try:
        MappoPlanner(Limits(), PlannerConfig(robot_radius_m=0.40),
                     PolicyRunner(DEFAULT_PACKAGE), refusal_log=log)
    except SystemExit as exc:
        message = str(exc)
        assert "REFUSING TO RUN" in message
        # The message has to carry BOTH numbers and BOTH escape hatches, or the operator
        # is left guessing which of the two knobs is the wrong one.
        assert "0.400" in message and "2.50" in message and "4.00" in message
        assert "--robot-radius" in message and "--policy-scale" in message
    else:
        raise AssertionError("a mismatched robot radius was accepted")
    finally:
        record = json.loads(log.read_text().splitlines()[0])
        log.unlink()
        log.parent.rmdir()
    assert record["reason"] == "robot_radius_scale_mismatch"
    assert record["planner_robot_radius_m"] == 0.40
    assert record["implied_scale"] == 4.0 and record["configured_scale"] == 2.5


def test_a_deliberate_policy_scale_override_runs_instead_of_being_refused():
    """The escape hatch has to open the gate it is documented to open.

    ``--policy-scale`` is applied to the config BEFORE the calibration check runs, so it
    moves ``configured`` and creates exactly the mismatch the guard fires on: using the
    documented escape hatch tripped the guard, and the refusal then advised passing
    ``--policy-scale 2.50``, i.e. undoing the change the operator had deliberately made.
    Narrowing the pass distance is done with this knob, so a gate that blocks it blocks
    the one tuning job the flag exists for. The refusal is for a SILENT mismatch — the
    test above — not for a number the operator typed."""
    log = Path(tempfile.mkdtemp()) / "refusals.jsonl"
    try:
        with derived_config(DEFAULT_PACKAGE / "config.json",
                            meters_per_vmas_unit=2.0) as config:
            runner = PolicyRunner(DEFAULT_PACKAGE, config)
            planner = MappoPlanner(Limits(), PlannerConfig(robot_radius_m=0.25), runner,
                                   refusal_log=log, scale_override=True)
        assert planner.config.robot_radius_m == 0.25
        # The override still leaves a trace: a run that passed closer than the evidence
        # base describes must be identifiable afterwards from the log alone.
        record = json.loads(log.read_text().splitlines()[0])
        assert record["reason"] == "robot_radius_scale_override"
        assert record["implied_scale"] == 2.5 and record["configured_scale"] == 2.0
    finally:
        log.unlink()
        log.parent.rmdir()


def test_a_scale_override_still_shortens_the_horizon_it_was_asked_for():
    """The knob has to move the thing the operator is aiming at.

    The pass distance is set by where the policy first sees the obstacle, which is
    ``lidar_range_vmas * meters_per_vmas_unit`` — and ``lidar_range_vmas`` is pinned to
    the checkpoint, so the scale is the only half that can move."""
    with derived_config(DEFAULT_PACKAGE / "config.json",
                        meters_per_vmas_unit=2.0) as config:
        assert PolicyRunner(DEFAULT_PACKAGE, config).config.lidar_range_m == 0.7
    assert PolicyRunner(DEFAULT_PACKAGE).config.lidar_range_m == 0.875


def test_a_slow_policy_command_is_scaled_up_to_the_gait_floor_keeping_direction():
    """The Go2 has a minimum speed and the checkpoint has never heard of it.

    Threading a 0.93 m gap on 2026-08-18 the policy went strongly lateral, its forward
    component collapsed to 0.14 m/s, and the robot stood still for 4 s reporting no
    fault. The policy's magnitude is not a safety decision the way the planner's is — the
    network outputs a DIRECTION — so the vector is scaled, not the forward axis alone.
    Scaling vx only would rotate the command toward straight ahead, which near an
    obstacle is the one direction it was steering away from."""
    planner = _planner(gait_floor_m_s=0.35)
    vx, vy, wz = planner._at_least_walking_pace((0.14, -0.09, 0.2))
    # On the ellipse, not on a circle of radius 0.35: the floor is 0.35 forward but only
    # 0.20 lateral, and a command 32.7 deg off the nose cannot be delivered at 0.35 m/s
    # by this robot at all. Asserting hypot == 0.35 here is what the CIRCLE model claimed,
    # and it was reachable only by rotating the command — see the next test.
    assert math.isclose(math.hypot(vx / 0.35, vy / 0.20), 1.0, rel_tol=1e-6), \
        "projected onto the envelope ellipse, which is where the floor lives"
    assert math.isclose(math.atan2(vy, vx), math.atan2(-0.09, 0.14), rel_tol=1e-6), \
        "direction must survive the scaling"
    assert wz == 0.2, "yaw has no gait floor and must not be touched"
    assert planner.counts["speed_raised"] == 1


def test_the_floor_does_not_rotate_a_command_toward_the_obstacle():
    """THE BUG THIS REPLACED. Scaling to a circular floor and then clamping each axis
    trims only the component that overshot, and trimming one component of a vector turns
    it — toward straight ahead, which near an obstacle is the direction the policy was
    steering away from.

    (0.05, 0.108) is 65.2 deg off the nose. The old code scaled by 0.35/0.119 = 2.94 to
    (0.147, 0.318), the max_vy clamp cut vy to 0.20 and left vx at 0.147, and the command
    left at 53.7 deg — 11.5 deg closer to the obstacle than the policy asked for. At
    80 deg the same arithmetic turns the command by 34.8 deg.

    Restore `scale = floor / speed` with per-axis clamps and this fails.
    """
    planner = _planner(gait_floor_m_s=0.35)
    proposed = (0.05, 0.108)
    vx, vy, _ = planner._at_least_walking_pace((*proposed, 0.0))
    asked = math.degrees(math.atan2(proposed[1], proposed[0]))
    got = math.degrees(math.atan2(vy, vx))
    assert abs(got - asked) < 1e-6, f"command rotated {abs(got - asked):.1f} deg"
    assert abs(vy) <= planner.limits.max_vy + 1e-9, "still inside the envelope"


def test_a_lateral_command_that_already_walks_is_left_alone():
    """A pure strafe at the lateral floor walks — 0.20 m/s, three repeats out of three —
    even though 0.20 is below the FORWARD floor of 0.35. Scaling it would be scaling a
    command that needed no help, and the circular test `speed >= floor` misses that
    because it compares a lateral speed against the forward number."""
    planner = _planner(gait_floor_m_s=0.35)
    assert planner._at_least_walking_pace((0.0, 0.20, 0.1)) == (0.0, 0.20, 0.1)
    assert planner.counts["speed_raised"] == 0


def test_a_commanded_stop_is_never_turned_into_a_walk():
    """A zeroed status tick — hold, stale input, goal reached — must stay stopped. The
    floor scales a command the policy meant as MOTION, and nothing else."""
    planner = _planner(gait_floor_m_s=0.35)
    assert planner._at_least_walking_pace((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    assert planner.counts["speed_raised"] == 0


def test_a_switched_off_strafe_axis_does_not_divide_by_zero():
    """``--max-vy 0`` is how ``robot-stack/deep_robotics/lite3/DEPLOYMENT-SOP.md``
    disables the strafe axis on a Lite3, and ``--policy-gait-floor`` is what reaches the
    ellipse at all — ``deploy/run-peer-supervised.sh`` ships it at 0.35. Together they
    raised ``ZeroDivisionError`` on the drive path, from ``0.0 / 0.0``: the envelope
    clamp has already zeroed ``vy``, so it is the degenerate ratio and not a stray
    lateral command that divides.

    The surviving axis still gets the floor, at the direction asked for — with no
    lateral axis, straight ahead is the only direction there is.
    """
    planner = _planner(limits=Limits(max_vx=0.35, max_vy=0.0), gait_floor_m_s=0.35)
    vx, vy, wz = planner._at_least_walking_pace((0.05, 0.0, 0.1))
    assert math.isclose(vx, 0.35, abs_tol=1e-9), "forward should reach the forward floor"
    assert (vy, wz) == (0.0, 0.1)
    assert planner.counts["speed_raised"] == 1


def test_a_switched_off_forward_axis_leaves_the_lateral_floor_intact():
    """The guard is symmetric. ``--max-vx 0`` is not documented as a thing anyone does,
    but a one-sided guard is a second rule to remember, and this pins that the surviving
    axis is still scaled to ITS OWN measured floor rather than to the forward one."""
    planner = _planner(limits=Limits(max_vx=0.0, max_vy=0.20), gait_floor_m_s=0.35)
    assert planner._at_least_walking_pace((0.0, 0.108, 0.0)) == (0.0, 0.20, 0.0)


def test_a_plan_with_no_lateral_axis_completes():
    """The unit above pins the arithmetic; this pins that the whole drive path survives
    the configuration, because the crash was reached through ``plan()`` and a guard that
    only the helper's own test exercises is a guard nobody proved the caller reaches."""
    planner = _planner(supervised=False, limits=Limits(max_vx=0.55, max_vy=0.0),
                       gait_floor_m_s=0.35)
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 1.0), (0.0, 0.0, 0.0), [])
    assert command.vy == 0.0, "an envelope with no lateral axis must command no strafe"
    assert abs(command.vx) <= 0.55 + 1e-9


def test_the_gait_floor_scaling_is_off_unless_asked_for():
    """Default off: it is a deliberate override of what the policy asked for, and the
    runs that arrived on 2026-08-18 did not need it."""
    planner = _planner()
    assert planner._at_least_walking_pace((0.14, -0.09, 0.0)) == (0.14, -0.09, 0.0)


def test_the_scaled_command_still_respects_the_envelope():
    """The floor may not become a way to out-run --derate. A command scaled up is still
    clamped to the stack's limits, which are the safety envelope."""
    planner = _planner(limits=Limits(max_vx=0.20, max_vy=0.05), gait_floor_m_s=0.35)
    vx, vy, _ = planner._at_least_walking_pace((0.10, -0.04, 0.0))
    assert abs(vx) <= 0.20 and abs(vy) <= 0.05


def test_the_policy_can_never_command_reverse():
    """THE HAZARD. The vendored planner refuses to sample reverse because this Go2 has no
    rear-facing sensing — backing up means moving blind into space the pipeline has never
    observed. That rule lived in the planner's sampling and NOT on this path, so the
    policy, a holonomic agent with no notion of where the sensors point, could command up
    to -0.35 m/s. Measured live 2026-08-18: v=(-0.35, -0.03) with the goal distance
    growing 2.64 -> 2.73 m."""
    planner = _planner(runner=_StubRunner(command=(-0.30, -0.05, 0.0)))
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [BIN])
    assert command.vx >= 0.0, f"reverse reached the legs: {command.vx}"


def test_a_reverse_command_is_never_scaled_up_to_the_gait_floor():
    """The scaling multiplies the whole vector, so without a sign guard a timid backward
    drift becomes a committed full-speed reverse. Belt and braces behind the clamp above:
    if a future change lets a negative vx through, it must not also be amplified."""
    planner = _planner(gait_floor_m_s=0.35)
    assert planner._at_least_walking_pace((-0.03, -0.02, 0.0)) == (-0.03, -0.02, 0.0)
    assert planner.counts["speed_raised"] == 0


def test_a_pure_strafe_is_scaled_because_the_lateral_floor_is_lower():
    """The guard used to read ``vx <= 0.0``, which refused a PURE STRAFE as well as a
    reverse, on the stated grounds that a strafe "cannot reach the floor anyway
    (max_vy 0.20 < 0.35)". That compares max_vy against the FORWARD gait floor.

    The lateral floor is a different number and had never been measured. Measured
    2026-08-19 in open floor against a forward control in the same session: 0.15 m/s does
    not walk this robot; 0.20 m/s does, three repeats of three, 0.076-0.087 m of travel
    each. So the shipped envelope can strafe, and the guard was refusing the one command
    that would have helped.

    Three live runs stalled on exactly this. Every escape was v=(+0.000,-0.150) — vx
    EXACTLY zero, so ``vx <= 0.0`` held, nothing was scaled, 0.150 m/s went out, and
    0.150 is below the lateral floor. The robot stood inside its own hard gap with an
    escape available and no way to walk it.

    Revert to ``<=`` and this reads unscaled.
    """
    planner = _planner(gait_floor_m_s=0.35)
    vx, vy, _wz = planner._at_least_walking_pace((0.0, -0.150, 0.0))
    assert planner.counts["speed_raised"] == 1, "a sideways step is not a reverse"
    assert vx == 0.0, "scaling must not invent a forward component"
    assert abs(vy) > 0.150, "the whole point is that it leaves faster than it arrived"
    # Clamped by the envelope to max_vy, which is the value measured to walk.
    assert math.isclose(abs(vy), planner.limits.max_vy, rel_tol=1e-6)


def test_a_reverse_is_still_refused_after_the_strafe_correction():
    """The correction is `< 0.0`, not the removal of the guard. A backward drift must
    still never be amplified — that one was observed live as a -0.03 m/s twitch scaled
    into a committed 0.35 m/s reverse into space this robot cannot sense."""
    planner = _planner(gait_floor_m_s=0.35)
    assert planner._at_least_walking_pace((-0.001, -0.150, 0.0)) == (-0.001, -0.150, 0.0)
    assert planner.counts["speed_raised"] == 0


def test_the_radius_the_live_runs_used_is_accepted():
    """A gate that refuses the correct configuration is a gate nobody leaves switched on."""
    planner = MappoPlanner(Limits(), PlannerConfig(robot_radius_m=0.25),
                           PolicyRunner(DEFAULT_PACKAGE))
    assert planner.config.robot_radius_m == 0.25


def test_a_refused_run_leaves_a_trace_because_it_writes_no_telemetry():
    """The refusal happens before the telemetry writer exists, so without this a run that
    never started leaves nothing at all behind — and "why did the 14:32 run not happen" is
    asked an hour later, on a demo day, by someone who was not at the terminal."""
    log = Path(tempfile.mkdtemp()) / "nested" / "refusals.jsonl"
    for _ in range(2):
        with contextlib.suppress(SystemExit):
            MappoPlanner(Limits(), PlannerConfig(robot_radius_m=0.40),
                         PolicyRunner(DEFAULT_PACKAGE), refusal_log=log)
    lines = log.read_text().splitlines()
    assert len(lines) == 2, "refusals append; a second one must not overwrite the first"
    assert all(json.loads(line)["wall_time"] > 0 for line in lines)
    log.unlink()
    log.parent.rmdir()
    log.parent.parent.rmdir()


def test_a_refusal_survives_an_unwritable_log():
    """The refusal is the safety mechanism and the log is the audit trail. A full disk
    must not turn a refusal into a traceback that reads like a bug, or worse, into a
    run."""
    unwritable = Path("/proc/definitely-not-writable/refusals.jsonl")
    try:
        MappoPlanner(Limits(), PlannerConfig(robot_radius_m=0.40),
                     PolicyRunner(DEFAULT_PACKAGE), refusal_log=unwritable)
    except SystemExit as exc:
        assert "REFUSING TO RUN" in str(exc)
    else:
        raise AssertionError("the run was accepted when the refusal log was unwritable")


# ── The envelope ────────────────────────────────────────────────────────────
def test_the_policy_cannot_out_run_a_derated_envelope():
    """``--derate`` and ``--max-vx`` are the operator's ceiling and the policy has never
    heard of them: its own ``max_vx_mps`` comes from its config. A run derated for a hot
    motor or a short tether must stay derated."""
    tiny = Limits(max_vx=0.05, max_vy=0.02, max_wz=0.10)
    planner = _planner(supervised=False, limits=tiny,
                       servo=HeadingServo(max_wz=0.7))
    for _ in range(4):
        command = planner.plan((0.0, 0.0, 0.0), (3.0, 1.5), (0.0, 0.0, 0.0), [])
        assert abs(command.vx) <= tiny.max_vx + 1e-9, command
        assert abs(command.vy) <= tiny.max_vy + 1e-9, command
        assert abs(command.wz) <= tiny.max_wz + 1e-9, command


# ── Stopping ────────────────────────────────────────────────────────────────
def test_a_hold_for_a_mover_stops_and_uses_the_reason_the_stack_acts_on():
    """``hold`` is not cosmetic in the vendored loop: it starts the rest-after-blocked
    timer that puts the robot prone instead of standing braced, and with the D1 arm on its
    back standing is the expensive posture. Emitting ``stop_external_hold`` instead would
    leave it standing for the whole wait, and nothing would report that."""
    planner = _planner()
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0),
                           [BIN, WALKER], last_reason="hold")
    assert command.reason == "hold"
    assert (command.vx, command.vy, command.wz) == (0.0, 0.0, 0.0)
    assert planner.counts["stopped"] == 1


def test_a_hold_for_the_bin_alone_does_not_stop_the_policy():
    """The whole reason the policy is here. The planner emits ``hold`` for a static
    obstacle too, and forwarding that would zero the policy in its one scene."""
    planner = _planner(supervised=False)
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [BIN],
                           last_reason="hold")
    assert command.reason == "policy"
    assert planner.counts["stopped"] == 0


def test_every_stop_status_maps_to_a_reason_the_vendored_loop_understands():
    """A status the map does not know falls back to ``hold``, which stops the robot. The
    test is that the four known ones are all present, because a missing entry would be
    silently correct-looking and would skip the prone rest."""
    assert set(_STOP_REASONS) == {"STOP_EXTERNAL_HOLD", "STOP_STALE_INPUT",
                                  "STOP_CLOCK_ERROR", "STOP_GOAL_REACHED"}
    assert set(_STOP_REASONS.values()) <= {"hold", "arrived"}


def test_the_measured_velocity_comes_from_the_robot_not_from_the_command():
    """A factor of two on this robot: it delivers about 0.45 of what it is told. Feed the
    policy the command and its observation says it is moving twice as fast as it is."""
    planner = _planner()
    planner.attach(_Loco((0.30, 0.0, 0.0)))
    planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [])
    scale = planner._runner.config.meters_per_vmas_unit
    observed = planner._runner.controller.last_observation[2] * scale
    assert math.isclose(observed, 0.30, abs_tol=1e-6)


def test_a_planner_with_no_locomotion_client_counts_the_gap():
    """It assumes zero rather than crashing mid-run, and says so at the end. Silence here
    would be a policy told it is stationary for a whole run."""
    runner = PolicyRunner(DEFAULT_PACKAGE)
    planner = MappoPlanner(Limits(), PlannerConfig(robot_radius_m=0.25), runner)
    planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [])
    assert planner.counts["velocity_unavailable"] == 1
    assert "no measured velocity" in planner.report()


# ── The substitution ────────────────────────────────────────────────────────
def test_the_vendored_stack_still_offers_the_seam_this_file_needs():
    """THE TRIPWIRE. `robot-stack/` is vendored, and this file no longer monkey-patches
    precisely because upstream grew `main(planner_factory=...)` and a public
    `is_feasible`. A re-vendor that dropped either would otherwise fail at the robot —
    or worse, leave the SHIPPED planner driving a run the operator believes the policy is.

    `visual_nav` is not imported here (it needs OpenCV and a robot), so the seam it owns
    is checked through the planner, which is importable, plus the module source.
    """
    from avoidance import DynamicWindowPlanner as Shipped
    assert callable(getattr(Shipped, "is_feasible", None)), \
        "the planner lost its public feasibility predicate"

    source = (Path(_HERE).parent / "robot-stack" / "unitree" / "go2" / "visual_nav"
              / "visual_nav.py").read_text()
    assert "def main(argv" in source and "planner_factory" in source, \
        "visual_nav.main lost the planner seam"
    assert "planner = planner_factory(limits=limits, config=planner_config)" in source, \
        "the seam is present but main() no longer routes through it"


def test_the_veto_calls_the_planners_own_public_predicate():
    """Not its own copy of the geometry. Two implementations of a safety check is two
    implementations that will disagree, and this one had exactly that for a while."""
    planner = _planner(supervised=True, runner=_StubRunner((0.35, 0.0, 0.0)))
    from avoidance import Obstacle as _Obstacle
    assert planner.is_feasible((0.0, 0.0, 0.0), (0.0, 0.20, 0.0), [BIN])
    assert not planner.is_feasible((0.0, 0.0, 0.0), (0.35, 0.0, 0.0), [BIN])
    assert isinstance(BIN, _Obstacle)


def test_the_policy_flags_do_not_collide_with_the_stack_flags():
    """Both parsers are real here, so a name clash is an argparse error rather than a
    surprise at the robot."""
    import argparse
    parser = _add_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--policy-mode", "raw", "--policy-scale", "2.5"])
    assert args.policy_mode == "raw" and args.policy_scale == 2.5
    assert parser.parse_args([]).policy_mode == "supervised", "supervised is the default"


def test_the_stack_path_does_not_shadow_the_d1_arm_package():
    """`d1_arm` is a namespace package holding `d1_arm.py`. Put its own directory on
    sys.path and `import d1_arm` binds the MODULE, so `d1_arm._arm_idl` is "not a
    package" — which is how the arm-stow monitor died at import, before the pre-flight
    that uses it could refuse anything. An import error reads as a missing dependency,
    not as a disabled safety check, so nothing about it looked like a safety regression.
    """
    import importlib.util

    from mappo_drive import _STACK

    assert str(_STACK / "d1_arm") not in sys.path
    assert str(_STACK) in sys.path, "go2/ is what makes d1_arm and locomotion packages"

    spec = importlib.util.find_spec("d1_arm")
    assert spec is not None and spec.submodule_search_locations is not None, \
        "d1_arm resolved as a module, so safety.py's import of _arm_idl cannot work"
    assert importlib.util.find_spec("d1_arm._arm_idl") is not None


def test_the_policy_flags_are_stripped_before_the_vendored_parser_sees_them():
    """The regression that ate a live run: `--policy-mode supervised` reached
    `visual_nav.build_parser()`, which exits 2 on an option it does not know. The test
    above passes a BARE parser and so never touched this — the stack's parser has to be
    the one that reads the surviving argv, or the check is about nothing.
    """
    import visual_nav
    argv = ["--live", "--goal-class", "chair", "--goal-height", "1.067",
            "--robot-radius", "0.25", "--no-latch-arm", "--max-seconds", "45",
            "--policy-mode", "raw", "--policy-command-scale", "0.8",
            "--heading-servo", "goal"]
    args, vendored = split_argv(argv, visual_nav.build_parser())

    assert args.policy_mode == "raw" and args.policy_command_scale == 0.8
    assert args.heading_servo == "goal"
    for flag in ("--policy-mode", "--policy-command-scale", "--heading-servo",
                 "raw", "0.8", "goal"):
        assert flag not in vendored, f"{flag} survived into the vendored argv"

    # The vendored parser must accept what is left, and it must still be the whole run.
    stack = visual_nav.build_parser().parse_args(vendored)
    assert stack.live and stack.goal_class == "chair" and stack.no_latch_arm
    assert stack.robot_radius == 0.25 and stack.max_seconds == 45.0


# ── The heading servo is opt-in (issue #16) ────────────────────────────────
def _servo_for(argv: list):
    """The servo ``main()`` would build for this command line, without running a robot."""
    import visual_nav
    args, _ = split_argv([*argv, "--goal-class", "chair", "--goal-height", "1.067"],
                         visual_nav.build_parser())
    return (None if args.heading_servo == "off"
            else HeadingServo(mode=args.heading_servo))


def test_a_drive_command_that_names_no_servo_gets_no_servo():
    """Issue #16. The servo was opt-OUT, so the configuration an operator got by not
    thinking about it was the one that saturated the yaw rate and drove into a wall on
    three runs out of four. The runbook's own copy-pasteable command passed no flag."""
    assert _servo_for([]) is None


def test_the_retired_spelling_is_consumed_and_still_means_off():
    """``--no-heading-servo`` is in operator command lines and in ``deploy/``. It always
    meant off, and off is now the default, so honouring it costs nothing — whereas
    exiting 2 on it would break a runbook that is currently correct.

    Asserting the servo alone would pass whether or not the option exists: ``split_argv``
    uses ``parse_known_args``, so a flag nobody declared is silently left in the argv and
    ``heading_servo`` keeps its default of ``off`` either way. The bite is that the
    leftover then reaches ``visual_nav``'s parser, which exits 2 on an option it does not
    know — so what has to be checked is that the flag was CONSUMED.
    """
    import visual_nav
    argv = ["--goal-class", "chair", "--goal-height", "1.067", "--no-heading-servo"]
    args, vendored = split_argv(argv, visual_nav.build_parser())
    assert args.heading_servo == "off"
    assert "--no-heading-servo" not in vendored, \
        "the retired spelling reached the vendored parser, which exits 2 on it"
    visual_nav.build_parser().parse_args(vendored)


def test_the_servo_can_be_asked_for_by_name_and_the_default_law_is_the_goal_bearing():
    assert _servo_for(["--heading-servo", "goal"]).mode == GOAL
    assert _servo_for(["--heading-servo", "travel"]).mode == TRAVEL
    assert HeadingServo().mode == GOAL, \
        "a servo built with no mode must not be issue #16's law"


def test_an_unknown_servo_law_is_refused_by_the_parser():
    """Not silently ignored, and not passed through to the vendored parser, which would
    exit 2 with a message about a flag the operator did not type."""
    import visual_nav
    try:
        split_argv(["--heading-servo", "bearing", "--goal-class", "chair",
                    "--goal-height", "1.067"], visual_nav.build_parser())
    except SystemExit:
        return
    raise AssertionError("an unknown --heading-servo law must be refused")


# ── The planner's search counts reach the telemetry (issue #20) ─────────────
def test_the_planners_search_counts_survive_a_policy_driven_tick():
    """``feasible=0 evaluated=0`` is not an absent value.

    It reads as "the planner sampled nothing and found nothing feasible" — the robot was
    boxed in. The recorded runs of 2026-08-17 say exactly that on all 58 policy-driven
    ticks of the successful one, while the vetoed ticks beside them record 330 of 330.
    The incumbent is computed on every tick regardless of which branch is taken, so the
    numbers exist and were simply being dropped.
    """
    planner = _planner(supervised=True, runner=_StubRunner((0.05, 0.0, 0.0)))
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [BIN])
    assert command.reason == "policy", command.reason
    assert command.evaluated > 0, "the planner ran a search; its size must survive"
    assert command.feasible > 0, "candidates cleared; that must not read as zero"


def test_the_search_counts_survive_a_stopped_tick_too():
    """The third branch. A stop is where an operator is most likely to ask what the
    planner thought, so it is the worst branch to zero the counters on."""
    planner = _planner(supervised=True,
                       runner=_StubRunner(status="STOP_EXTERNAL_HOLD"))
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [BIN])
    assert command.reason == _STOP_REASONS["STOP_EXTERNAL_HOLD"]
    assert command.evaluated > 0, planner.counts


# ── The refusal log really cannot raise ─────────────────────────────────────
def test_a_refusal_is_recorded_even_when_the_detail_is_not_serialisable():
    """``_record_refusal`` promises it never raises, and used to catch only ``OSError``
    while ``json.dumps`` raises ``TypeError`` on anything unserialisable. The promise
    held by luck of the current call site passing floats. If it ever broke, a traceback
    would replace the refusal message that exists so a demo-day operator can see why the
    run did not start."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "refusals.jsonl"
        _record_refusal(path, "unserialisable_detail", {"where": Path("/tmp/x")})
        record = json.loads(path.read_text().strip())
    assert record["reason"] == "unserialisable_detail"
    assert record["where"] == "/tmp/x", "recorded approximately, not lost"


# ── Peer robots off the mesh ────────────────────────────────────────────────
_PEER_ID = "mappo-go2-peer"
_PEER_NOW = 98_765.0


def _peer_spool(tmpdir, received=_PEER_NOW, written=_PEER_NOW, x=1.6, y=0.4):
    path = os.path.join(tmpdir, "peers.json")
    write_spool(path, spool_document(
        {_PEER_ID: {"received_monotonic_s": received, "x": x, "y": y, "yaw": 0.0,
                    "vx": 0.0, "vy": 0.0, "radius_m": 0.40}},
        (_PEER_ID,), domain="test-boot", written_monotonic_s=written))
    return PeerSource(path, Alignment(), domain="test-boot")


def test_a_lost_peer_stops_the_robot_even_if_the_policy_never_looks_at_the_hold():
    """BELT AND BRACES, and this is the braces.

    The peer hold also travels the ordinary route — into ``RobotInput.external_hold`` via
    ``mappo_bridge`` — which is what keeps the policy's own view of the world honest. But
    that route runs a SAFETY property through the vendored policy package, and nothing in
    this repository decides whether a future checkpoint or runner honours it.
    ``_StubRunner`` is exactly such a runner: it returns a full-ahead COMMAND whatever it
    is told. The legs must stop anyway.

    Made to fail by deleting the ``peer_link.get("lost")`` branch in ``plan()``: the stub's
    0.35 m/s goes straight out.
    """
    with tempfile.TemporaryDirectory() as tmp:
        planner = _planner(supervised=False, runner=_StubRunner((0.35, 0.0, 0.0)))
        planner.attach_peers(_peer_spool(tmp, written=_PEER_NOW + 5.0))
        planner._peers.read(_PEER_NOW + 5.0)         # what PeerNavigator does each tick
        command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [])
        assert (command.vx, command.vy, command.wz) == (0.0, 0.0, 0.0), command
        assert command.reason == "hold", "'hold' is what starts the rest-when-blocked timer"
        assert planner.counts["peer_held"] == 1


def test_the_policy_is_TOLD_the_link_is_gone_and_not_just_overruled():
    """The belt, to the test above's braces — and it needs its own test, because the
    braces hide it: with the direct check in place the robot stops whether or not the
    policy was ever informed, so the bridge route can be deleted and nothing goes red.
    Found by mutation-testing this file.

    It matters because the alternative is silently handing the policy a world with one
    fewer robot in it. Its obstacle memory associates by position across ticks, and a peer
    that vanishes from the observation is indistinguishable, to it, from a peer that
    moved. ``external_hold`` is how it is told the difference.
    """
    ticks = []

    class _Recording(_StubRunner):
        def step(self, tick, monotonic_s=None):
            ticks.append(tick)
            return _StubRunner.step(self, tick, monotonic_s)

    with tempfile.TemporaryDirectory() as tmp:
        planner = _planner(supervised=False, runner=_Recording((0.35, 0.0, 0.0)))
        planner.attach_peers(_peer_spool(tmp, written=_PEER_NOW + 5.0))
        planner._peers.read(_PEER_NOW + 5.0)
        planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [])
    assert ticks and ticks[0]["peer_link"]["lost"] is True
    assert external_hold(ticks[0]) is True, "the policy was not told the link was gone"
    assert "peer link" in ticks[0]["peer_link"]["reason"]


def test_a_fresh_peer_leaves_the_policy_in_charge():
    """The other side of the same branch — without it the test above would pass on a
    planner that simply never drives."""
    with tempfile.TemporaryDirectory() as tmp:
        planner = _planner(supervised=False, runner=_StubRunner((0.35, 0.0, 0.0)))
        planner.attach_peers(_peer_spool(tmp))
        planner._peers.read(_PEER_NOW)
        command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [])
        assert command.reason == "policy", command
        assert planner.counts["peer_held"] == 0


def test_a_peer_source_that_has_never_been_read_holds_rather_than_reading_as_absent():
    """"Peers are configured and I have not looked yet" and "there are no peers" are the
    same empty snapshot and opposite situations. Reading the first as the second is the
    fail-open direction: the robot drives with peers configured and none of them modelled.

    Found by mutation-testing this file — deleting the ``last is None`` guard left every
    other test green, because they all read the source before planning, which a navigator
    that had been swapped out would not.
    """
    with tempfile.TemporaryDirectory() as tmp:
        planner = _planner(supervised=False, runner=_StubRunner((0.35, 0.0, 0.0)))
        planner.attach_peers(_peer_spool(tmp))       # attached, deliberately never read
        command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [])
        assert command.reason == "hold", command
        assert planner.counts["peer_held"] == 1


def test_a_planner_with_no_peer_link_is_unchanged():
    """Peer avoidance is opt-in, and every run recorded so far was made without it."""
    planner = _planner(supervised=False, runner=_StubRunner((0.35, 0.0, 0.0)))
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [])
    assert command.reason == "policy"
    assert planner.counts["peer_held"] == 0


class _FakeNavigator:
    """Enough of ``VisualNavigator`` for the seam: an ``_obstacles`` to extend."""

    def __init__(self, *args, **kwargs):
        self.args = args

    def _obstacles(self, now: float) -> list:
        return [BIN]


def test_the_peer_joins_the_one_obstacle_list_the_whole_loop_reads():
    """``_obstacles`` is the seam, not ``plan()``, and this is why: the vendored loop
    builds the list ONCE per tick and hands the same object to the planner, the recorder,
    the console log and the telemetry writer. A peer added here is in all four, on every
    path through the loop — including the stale-frame and goal-search ticks where
    ``plan()`` is never called and a peer appended there would go unrecorded."""
    with tempfile.TemporaryDirectory() as tmp:
        peers = _peer_spool(tmp)
        navigator = peer_navigator(_FakeNavigator, peers)("loco", "perception", "planner")
        obstacles = navigator._obstacles(_PEER_NOW)
        assert [o.object_id for o in obstacles] == ["landmark-1", f"peer-{_PEER_ID}"]
        peer = obstacles[-1]
        # A planner Obstacle, so the veto's rollout and the telemetry writer both work on
        # it with no special case — which is the entire argument for this shape.
        assert isinstance(peer, Obstacle)
        assert peer.kind == "tracked" and peer.radius_m >= 0.40
        assert peer.soft_gap_m is None and peer.hard_gap_m is None, (
            "a peer takes the planner's person-sized gaps: unlike a bin it can step "
            "sideways into the space the robot has committed to")


def test_the_obstacle_list_and_the_hold_are_one_ticks_decision():
    """``plan()`` reads the snapshot ``_obstacles`` took rather than re-reading the spool.
    Re-reading would let the policy be handed a peer the hold has already given up on,
    half a spool-write apart."""
    with tempfile.TemporaryDirectory() as tmp:
        peers = _peer_spool(tmp, written=_PEER_NOW + 5.0)
        navigator = peer_navigator(_FakeNavigator, peers)("loco", "perception", "planner")
        obstacles = navigator._obstacles(_PEER_NOW + 5.0)
        assert [o.object_id for o in obstacles] == ["landmark-1"], "a lost peer is no disc"
        assert peers.last.holds is True


def test_the_transform_is_the_enabling_flag_so_it_can_never_be_absent():
    """Two robots' odom frames start at their own power-on poses and have no relationship
    until one is measured. A separate on/off switch would create a combination — peers on,
    frames undeclared — whose only sane answer is a refusal; making the transform itself
    the switch removes the state instead of validating it."""
    parser = _add_arguments(argparse.ArgumentParser())
    assert parser.parse_args([]).peer_odom_align is None
    assert parser.parse_args(["--peer-odom-align", "2,1,180"]).peer_odom_align == "2,1,180"
    assert parser.parse_args([]).peer_timeout == PEER_TIMEOUT_S


# ── The floor is checked on the command, not just on the envelope (issue #26) ──
def test_the_floor_is_checked_on_the_vetoed_command_and_not_only_on_the_policys():
    """THE FINDING. ``_at_least_walking_pace`` lives inside ``plan()``'s POLICY branch,
    so the command that gets issued when the veto fires — the planner's own — has never
    been compared to the floor at all. That is the command issue #26 measured crawling
    at a 0.137 m/s mean over 32 ticks through a 0.93 m gap while the stall gate blamed
    the tether.

    A 0.93 m gap between two bins, closed to the width that makes the policy's proposal
    infeasible. The planner's own answer is 0.035 m/s — a tenth of the floor, and below
    ``visual_nav.PROGRESS_MIN_COMMAND_M_S`` too, so the stall gate calls it "not asking
    it to go anywhere" and NOTHING in the stack has an opinion about it.

    Delete the ``_note_sub_floor`` call from ``plan()`` and this reads zero.
    """
    bins = [Obstacle(x=1.15, y=+0.50, vx=0.0, vy=0.0, radius_m=0.23, kind="static",
                     object_id="bin-left"),
            Obstacle(x=1.15, y=-0.43, vx=0.0, vy=0.0, radius_m=0.23, kind="static",
                     object_id="bin-right")]
    planner = _planner(supervised=True, runner=_StubRunner((0.35, 0.0, 0.0)),
                       platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    command = planner.plan((0.0, 0.0, 0.0), (4.0, 0.0), (0.20, 0.0, 0.0), bins,
                           control_dt=0.33)
    assert command.reason.startswith("veto-"), command.reason
    assert 0.0 < math.hypot(command.vx, command.vy) < MIN_GAIT_COMMAND_M_S, \
        f"the scene must produce a sub-floor PLANNER command: {command}"
    assert planner.counts["sub_floor"] == 1, planner.counts
    assert planner.counts["speed_raised"] == 0, \
        "the planner's command must be reported, never scaled — issue #26 measured that"


def test_the_floor_is_checked_on_a_policy_driven_command_too():
    """The other branch, so "on every path" is two observations rather than one. A
    policy command below the floor with the raising knob OFF — the default, and what
    every recorded run used — is still counted."""
    planner = _planner(supervised=False, runner=_StubRunner((0.10, 0.0, 0.0)),
                       platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    command = planner.plan((0.0, 0.0, 0.0), (3.0, 0.0), (0.0, 0.0, 0.0), [])
    assert command.reason == "policy" and command.vx == 0.10
    assert planner.counts["sub_floor"] == 1, planner.counts


def test_a_sub_floor_command_that_goes_nowhere_is_named_before_the_tether_can_be():
    """The measured stall: ``commanded 0.13 m/s for 4.1s and moved 0.04 m of an expected
    0.54 m``, and the stack then said *"Something is holding the robot — check the
    tether"*. It was not the tether.

    Two things are pinned. First that it fires at all on those numbers. Second that it
    fires INSIDE ``visual_nav.PROGRESS_WINDOW_S`` (4.0 s), because that gate ENDS THE
    RUN with the tether message and an explanation printed after the outcome line is an
    explanation nobody reads.

    Raise ``SUB_FLOOR_WINDOW_S`` to 4.0 or above and this fails on the ordering even
    though the diagnosis is still right."""
    planner = _planner(platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    printed = _drive_sub_floor(planner, speed=0.137, ticks=8, delivered=0.010)
    assert planner.counts["sub_floor_stalled"] == 1, planner.counts
    assert SUB_FLOOR_WINDOW_S < 4.0, \
        "the stack's stall gate fires at 4.0 s and ends the run; this must precede it"
    assert "NOT THE TETHER" in printed, printed
    assert "0.137 m/s" in printed and "gait floor" in printed, printed
    assert "8/8 ticks commanded below the gait floor" in planner.report(), \
        planner.report()
    assert "(1 2s window of it covering no ground — THE GAIT FLOOR, NOT THE TETHER)" \
        in planner.report(), planner.report()


def test_the_run_that_arrived_below_the_floor_is_counted_and_not_faulted():
    """THE FALSIFIER, and it is in issue #26's own body. Run C of 2026-08-18 sustained a
    mean of 0.295 m/s with **54 of 54 ticks below the 0.35 floor**, minimum 0.189, and
    walked 3 m to the goal. Any rule that faulted on "sub-floor" alone would have killed
    an arriving run, which is why this gate is not one: it triggers on *commanded to move
    and not moving*, and uses the floor only to pick which explanation to print.

    The count still rises to 54. That is the point of counting it separately — issue
    #26's proposal 2 is to measure the real floor, and this pair of numbers across runs
    is the measurement. 0.35 is "the lowest speed observed to work", not a threshold.

    Drop the ``moved >= SUB_FLOOR_PROGRESS_FRACTION * travel`` test and this run is
    faulted."""
    planner = _planner(platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    # 0.240 m/s delivered of 0.295 commanded is the 0.70 ratio measured on this robot
    # at full command; run C walked 3 m.
    printed = _drive_sub_floor(planner, speed=0.295, ticks=54, delivered=0.240)
    assert planner.counts["sub_floor"] == 54, planner.counts
    assert planner.counts["sub_floor_stalled"] == 0, \
        "a run that arrived must not be faulted for the speed it arrived at"
    assert printed == "", printed
    assert "54/54 ticks commanded below the gait floor" in planner.report(), \
        planner.report()
    assert "NOT THE TETHER" not in planner.report(), planner.report()


def test_the_progress_bar_is_the_stacks_own_and_a_crawl_that_delivers_clears_it():
    """The boundary, so ``SUB_FLOOR_PROGRESS_FRACTION`` is load-bearing rather than
    decorative. At exactly the fraction the run passes; a hair under it does not."""
    over = _planner(platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    _drive_sub_floor(over, speed=0.200, ticks=10,
                     delivered=0.200 * SUB_FLOOR_PROGRESS_FRACTION * 1.05)
    assert over.counts["sub_floor_stalled"] == 0, over.counts
    under = _planner(platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    _drive_sub_floor(under, speed=0.200, ticks=10,
                     delivered=0.200 * SUB_FLOOR_PROGRESS_FRACTION * 0.95)
    assert under.counts["sub_floor_stalled"] >= 1, under.counts


def test_the_commanded_travel_is_what_was_asked_for_over_each_interval():
    """A planner that is actively slowing down commands a DIFFERENT speed every tick, and
    which one is credited to which interval decides the verdict.

    The loop issues a command and then sleeps, so the command in force over
    ``[previous, now]`` is the one decided at ``previous``. Crediting this tick's command
    to the last tick's interval is an off-by-one, and on a collapsing command it is not a
    rounding error: asked for 0.34 m/s and then 0.01 for seven ticks, the honest total is
    0.132 m and the off-by-one says 0.023 m — 5.7x apart, and the robot moved 0.015 m,
    which is between them. Correct accounting faults it; the off-by-one lets it pass.

    Replace ``held`` with ``speed`` in the integration and this reads zero.
    """
    planner = _planner(platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    speeds = [0.34] + [0.01] * 7
    dt, now, x = 0.33, 100.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for i, speed in enumerate(speeds):
            planner._note_sub_floor(Command(speed, 0.0, 0.0, reason="veto-avoid",
                                            gap_m=1.0), (x, 0.0, 0.0), now)
            # 0.015 m of travel in total, all of it while the command was still 0.34.
            x = 0.015 if i == 0 else x
            now += dt
    assert planner.counts["sub_floor_stalled"] == 1, planner.counts


def test_a_commanded_stop_is_not_a_sub_floor_command():
    """A held robot is stationary on purpose, and a stop has no speed to be under a
    floor. Counting it would fault every peer hold, every stale-input tick and every
    arrival — and the peer hold is the one branch of ``plan()`` that exists to stop."""
    planner = _planner(platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    printed = _drive_sub_floor(planner, speed=0.0, ticks=30, delivered=0.0)
    assert planner.counts["sub_floor"] == 0 and planner.counts["sub_floor_stalled"] == 0
    assert printed == ""


def test_a_strafe_at_the_lateral_floor_is_not_reported_as_sub_floor():
    """0.20 m/s of pure strafe walks this robot — three repeats of three, 0.076-0.087 m
    each — and 0.20 is below the FORWARD floor of 0.35. Judging the speed against the
    forward number would report every legal crab step as sub-floor, which is the same
    axis confusion the ``vx <= 0.0`` guard had. The ellipse is what makes it right."""
    planner = _planner(platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    now = 100.0
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(20):
            planner._note_sub_floor(Command(0.0, 0.20, 0.0, reason="policy", gap_m=1.0),
                                    (0.0, 0.0, 0.0), now)
            now += 0.33
    assert planner.counts["sub_floor"] == 0, planner.counts


def test_the_sub_floor_window_re_arms_so_a_second_stall_is_reported_too():
    """A GATE THAT FIRES ONCE IS A GATE THAT MISSES THE SECOND FAILURE. The window is
    judged and restarted whatever the verdict, so a robot that stalls, is nudged, and
    stalls again is caught twice; only the banner is latched, because eight copies of it
    would bury the run log it is printed into.

    Latch the whole check on ``_sub_floor_announced`` and the count stops at one.

    The count is ONE PER WINDOW, not one per tick, which is what makes it a duration: 40
    ticks of 0.33 s is 13.2 s, five whole windows of 2.0 s. Drop the ``_sub_floor = None``
    that restarts the window and every tick after the first stall re-judges the whole run
    and is counted again — 34 rather than 5, from the same 13.2 s of standing still.
    """
    planner = _planner(platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    ticks, dt = 40, 0.33
    printed = _drive_sub_floor(planner, speed=0.137, ticks=ticks, delivered=0.0,
                               dt=dt)
    windows = int(ticks * dt / SUB_FLOOR_WINDOW_S)
    assert planner.counts["sub_floor_stalled"] in (windows - 1, windows), \
        f"{planner.counts['sub_floor_stalled']} verdicts over {windows} windows"
    assert f"({planner.counts['sub_floor_stalled']} 2s windows" in planner.report(), \
        planner.report()
    assert printed.count("NOT THE TETHER") == 1, "the banner is printed once per run"


# ── The floor ellipse is the FLOOR's, not the envelope's (issues #26 + #103) ──
def test_the_measured_go2_pair_survives_the_floor_projection_exactly():
    """The projection was written when the floor and the envelope were provably the same
    curve — ``MIN_GAIT_COMMAND_M_S`` 0.35 == ``max_vx`` and the lateral floor 0.20 ==
    ``max_vy``, which ``avoidance`` says is not a coincidence. Every recorded run was
    made there, so the generalisation has to reduce to it exactly rather than nearly."""
    planner = _planner(gait_floor_m_s=MIN_GAIT_COMMAND_M_S)
    assert planner._floor_axes(MIN_GAIT_COMMAND_M_S) == (0.35, 0.20, False)
    # No floor is the envelope, unclipped. Both callers guard against reaching here with
    # zero, so this is the contract rather than a live path — pinned because the wrong
    # answer for it, `(0, 0)`, reads downstream as "a direction the envelope cannot
    # reach" and would count a normal command as unreachable.
    assert planner._floor_axes(0.0) == (0.35, 0.20, False)


def test_a_floor_inside_the_envelope_is_projected_onto_the_floor_not_the_ceiling():
    """ISSUE #103 BROKE THE PREMISE THE PROJECTION RESTED ON. The envelope is now a
    per-robot number a Lite3 must STATE on a live run, and
    ``robot-stack/deep_robotics/lite3/DEPLOYMENT-SOP.md`` states ``--max-vx 0.55`` beside
    a measured ``--gait-floor 0.30``. The floor is then strictly inside the envelope and
    they are different curves.

    Projecting onto the ENVELOPE turns a 0.050 m/s policy command into 0.550 m/s — an
    11x amplification of a command the policy meant as a crawl, with the veto then
    validating the sprint. The floor it should reach is 0.30.

    Restore ``_axis_reach(vx, self.limits.max_vx)`` and this reads 0.55.
    """
    planner = _planner(limits=Limits(max_vx=0.55, max_vy=0.0), gait_floor_m_s=0.30)
    vx, vy, wz = planner._at_least_walking_pace((0.05, 0.0, 0.1))
    assert math.isclose(vx, 0.30, abs_tol=1e-9), f"raised to {vx:.3f}, not to the floor"
    assert (vy, wz) == (0.0, 0.1)
    assert planner.counts["raised_below_floor"] == 0, "the envelope was above the floor"


def test_a_command_that_already_reaches_the_floor_is_not_amplified_to_the_envelope():
    """The same defect seen from the other side, and the more common one: a command that
    already walks. 0.40 m/s is above the Lite3 SOP's 0.30 floor and below its 0.55
    ceiling, so it needs no help — projecting onto the envelope would raise it anyway."""
    planner = _planner(limits=Limits(max_vx=0.55, max_vy=0.0), gait_floor_m_s=0.30)
    assert planner._at_least_walking_pace((0.40, 0.0, 0.0)) == (0.40, 0.0, 0.0)
    assert planner.counts["speed_raised"] == 0, planner.counts


def test_the_projection_still_preserves_direction_on_a_floor_inside_the_envelope():
    """Scaling both components by one scalar is what keeps the direction, and shrinking
    the ellipse must not change that — near an obstacle, rotating a command toward
    straight ahead is rotating it toward the thing being avoided."""
    planner = _planner(limits=Limits(max_vx=0.55, max_vy=0.30), gait_floor_m_s=0.30)
    vx, vy, _ = planner._at_least_walking_pace((0.05, 0.03, 0.0))
    assert abs(math.degrees(math.atan2(vy, vx))
               - math.degrees(math.atan2(0.03, 0.05))) < 1e-6
    assert math.isclose(math.hypot(vx / 0.30, vy / (0.30 * 0.30 / 0.55)), 1.0,
                        rel_tol=1e-6), "on the FLOOR ellipse, not the envelope's"
    assert abs(vx) <= 0.55 and abs(vy) <= 0.30, "still inside the envelope"


def test_a_floor_above_the_envelope_is_not_reported_as_having_reached_it():
    """``--derate 0.6`` on the shipped Go2 profile is exactly 0.21 m/s, the setting
    measured to stall 5 runs of 5. The projection can only reach the envelope, so the
    result is still sub-floor — and reporting it as "scaled up to the gait floor" is a
    false sentence about a safety number. It is counted apart and said out loud."""
    planner = _planner(limits=Limits().scaled(0.6), gait_floor_m_s=MIN_GAIT_COMMAND_M_S)
    vx, _vy, _wz = planner._at_least_walking_pace((0.10, 0.0, 0.0))
    assert math.isclose(vx, 0.21, abs_tol=1e-9), vx
    assert vx < MIN_GAIT_COMMAND_M_S, "the envelope cannot reach the floor here"
    assert planner.counts["raised_below_floor"] == 1, planner.counts
    assert "below the floor" in planner.report(), planner.report()


def test_a_derated_run_is_reported_as_sub_floor_on_every_tick():
    """``--derate 0.6`` on the shipped Go2 profile is exactly 0.21 m/s, the setting
    measured to stall 5 runs of 5 across two controllers. EVERY command such a run can
    make is below the floor, including the fastest one.

    A sub-floor test that clipped the floor to the envelope would call a command at the
    0.21 ceiling "at the floor" and report ``0/N ticks below the floor`` for a run that
    was under it from the first tick to the last — the exact reading that sent five runs
    after tethers and walls. ``_floor_axes(..., clip=False)`` is what stops it.

    Pass ``clip=True`` from ``_note_sub_floor`` and this reads zero.
    """
    planner = _planner(limits=Limits().scaled(0.6),
                       platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
    assert planner.limits.max_vx == 0.21, planner.limits
    now = 100.0
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(5):
            planner.counts["ticks"] += 1
            # The FASTEST command this envelope allows, and it still does not walk.
            planner._note_sub_floor(Command(0.21, 0.0, 0.0, reason="policy", gap_m=1.0),
                                    (0.0, 0.0, 0.0), now)
            now += 0.33
    assert planner.counts["sub_floor"] == 5, planner.counts
    assert "5/5 ticks commanded below the gait floor" in planner.report(), \
        planner.report()


# ── Which floor, and whose (issue #26) ──────────────────────────────────────
class _NoFloorBindings:
    """A binding with no measured floor — a Lite3 before commissioning."""

    @staticmethod
    def gait_floor(_args):
        return None


class _MeasuredFloorBindings:
    """A binding whose floor is the operator's measurement, as a Lite3's is."""

    @staticmethod
    def gait_floor(args):
        return args.gait_floor


def test_the_floor_judged_against_is_the_robots_own_and_needs_no_flag():
    """A Go2 gets the per-tick check from ``MIN_GAIT_COMMAND_M_S`` with nothing typed,
    which is what makes it apply to every command line already in the runbooks. Reading
    it off ``--policy-gait-floor`` instead would leave it off wherever it matters, since
    that flag defaults to 0 and only ``deploy/run-peer-supervised.sh`` passes it."""
    import visual_nav
    args, _ = split_argv(["--goal-class", "chair", "--goal-height", "1.067"],
                         visual_nav.build_parser())
    assert args.policy_gait_floor == 0.0, "the raising knob is off by default"
    with contextlib.redirect_stdout(io.StringIO()):
        assert platform_gait_floor(visual_nav.Go2Bindings(), args) \
            == MIN_GAIT_COMMAND_M_S


def test_an_unmeasured_floor_judges_nothing_and_says_so():
    """An absent measurement is not a floor of zero and must not become the Go2's 0.35
    by default — that substitution is issues #83, #96, #101 and #103. What it becomes is
    a printed absence, because a check that silently did not run is the other half of
    the same defect."""
    args = argparse.Namespace(policy_gait_floor=0.0)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert platform_gait_floor(_NoFloorBindings(), args) == 0.0
    assert "no gait floor is known" in out.getvalue(), out.getvalue()

    planner = _planner(platform_floor_m_s=0.0)
    _drive_sub_floor(planner, speed=0.137, ticks=20, delivered=0.0)
    assert planner.counts["sub_floor"] == 0, "nothing may be judged against no floor"
    assert "no gait floor was known" in planner.report(), planner.report()


def test_a_policy_gait_floor_that_is_not_this_robots_is_named():
    """Two numbers that both call themselves the gait floor, set in two places.
    ``deploy/run-peer-supervised.sh`` hard-codes ``--policy-gait-floor 0.35``, which is
    the Go2's; a Lite3 measures 0.30. Until now nothing compared them, and the script is
    the thing somebody copies onto the next robot."""
    args = argparse.Namespace(policy_gait_floor=0.35, gait_floor=0.30)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert platform_gait_floor(_MeasuredFloorBindings(), args) == 0.30
    assert "0.350" in out.getvalue() and "0.300" in out.getvalue(), out.getvalue()

    agreed = io.StringIO()
    args = argparse.Namespace(policy_gait_floor=0.30, gait_floor=0.30)
    with contextlib.redirect_stdout(agreed):
        platform_gait_floor(_MeasuredFloorBindings(), args)
    assert "⚠️" not in agreed.getvalue(), \
        "a gate that warns about the correct configuration is a gate people switch off"


def test_a_floor_above_the_envelope_is_announced_when_the_planner_is_built():
    """Before the legs move, and in the drive path's own terms. The stack warns when the
    envelope ceiling is under the floor, but nothing said what that means for a run that
    is about to hand the policy the legs: NO command it can make will reach the floor."""
    def built(limits):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            MappoPlanner(limits, PlannerConfig(robot_radius_m=0.25),
                         PolicyRunner(DEFAULT_PACKAGE),
                         platform_floor_m_s=MIN_GAIT_COMMAND_M_S)
        return out.getvalue()

    derated = built(Limits().scaled(0.6))
    assert "ABOVE THIS RUN'S ENVELOPE" in derated, derated
    shipped = built(Limits())
    assert "ABOVE THIS RUN'S ENVELOPE" not in shipped, shipped
    assert "gait floor 0.350" in shipped, shipped


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"mappo_drive: {len(tests)}/{len(tests)} passed")
