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

import contextlib
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
from avoidance import Limits, Obstacle, PlannerConfig

from mappo_drive import (
    _STOP_REASONS,
    MappoPlanner,
    _add_arguments,
    _record_refusal,
    split_argv,
)
from mappo_policy import DEFAULT_PACKAGE, HeadingServo, PolicyRunner
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
             gait_floor_m_s: float = 0.0) -> MappoPlanner:
    planner = MappoPlanner(limits or Limits(), PlannerConfig(robot_radius_m=0.25),
                           runner or PolicyRunner(DEFAULT_PACKAGE, servo=servo),
                           supervised=supervised, gait_floor_m_s=gait_floor_m_s)
    planner.attach(_Loco())
    return planner


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
    assert math.isclose(math.hypot(vx, vy), 0.35, rel_tol=1e-6), "scaled to the floor"
    assert math.isclose(math.atan2(vy, vx), math.atan2(-0.09, 0.14), rel_tol=1e-6), \
        "direction must survive the scaling"
    assert wz == 0.2, "yaw has no gait floor and must not be touched"
    assert planner.counts["speed_raised"] == 1


def test_a_commanded_stop_is_never_turned_into_a_walk():
    """A zeroed status tick — hold, stale input, goal reached — must stay stopped. The
    floor scales a command the policy meant as MOTION, and nothing else."""
    planner = _planner(gait_floor_m_s=0.35)
    assert planner._at_least_walking_pace((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    assert planner.counts["speed_raised"] == 0


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
            "--no-heading-servo"]
    args, vendored = split_argv(argv, visual_nav.build_parser())

    assert args.policy_mode == "raw" and args.policy_command_scale == 0.8
    assert args.no_heading_servo is True
    for flag in ("--policy-mode", "--policy-command-scale", "--no-heading-servo",
                 "raw", "0.8"):
        assert flag not in vendored, f"{flag} survived into the vendored argv"

    # The vendored parser must accept what is left, and it must still be the whole run.
    stack = visual_nav.build_parser().parse_args(vendored)
    assert stack.live and stack.goal_class == "chair" and stack.no_latch_arm
    assert stack.robot_radius == 0.25 and stack.max_seconds == 45.0


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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"mappo_drive: {len(tests)}/{len(tests)} passed")
