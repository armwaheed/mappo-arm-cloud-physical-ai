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

from mappo_drive import _STOP_REASONS, MappoPlanner, _add_arguments
from mappo_policy import DEFAULT_PACKAGE, HeadingServo, PolicyRunner

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
    vetoed; at the shipped 0.6 it is vetoed every time. Injecting the command keeps the
    branch under the test's control, and
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
             servo: HeadingServo | None = None, runner=None) -> MappoPlanner:
    planner = MappoPlanner(limits or Limits(), PlannerConfig(robot_radius_m=0.25),
                           runner or PolicyRunner(DEFAULT_PACKAGE, servo=servo),
                           supervised=supervised)
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
    from the delivered 0.3 to the shipped 0.6 — which is worth pinning rather than
    forgetting.

    At 0.3 the policy's command covers 0.35 x 0.3 x 2.5 = 0.26 m over the planner's
    horizon, which cannot reach a bin 1.0 m away, so nothing was ever vetoed and the veto
    looked dormant. At 0.6 the rollout reaches 0.52 m and the swerve no longer clears the
    disc over the full horizon, so the veto fires. The closed-loop measurement agrees:
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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"mappo_drive: {len(tests)}/{len(tests)} passed")
