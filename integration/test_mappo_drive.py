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

import math
import os
import sys
import types

# BOTH paths go in before ANY sibling import. `ruff --fix` sorts imports into
# contiguous blocks and will hoist `from avoidance import ...` above a sys.path line that
# sits between the blocks — which is exactly how this file went from passing to
# ModuleNotFoundError without anybody touching a test.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "robot-stack", "unitree", "go2",
                                "visual_nav"))
from avoidance import Limits, Obstacle, PlannerConfig

from mappo_drive import _STOP_REASONS, MappoPlanner, _add_arguments, _install
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
def _stub_module():
    module = types.ModuleType("stub_visual_nav")
    module.DynamicWindowPlanner = object
    module.VisualNavigator = lambda *a, **k: ("navigator", a, k)
    module.build_parser = lambda: __import__("argparse").ArgumentParser()
    return module


def test_the_substitution_replaces_all_three_globals():
    module = _stub_module()
    original = module.DynamicWindowPlanner
    _install(module, lambda **kwargs: "planner", lambda loco, planner: None)
    assert module.DynamicWindowPlanner is not original
    assert "--policy-mode" in module.build_parser().format_help()


def test_the_substitution_refuses_a_stack_whose_globals_have_moved():
    """``robot-stack/`` is vendored. A re-vendor that renames one of the three must be a
    loud failure here: patching what is present and skipping what is not would leave the
    SHIPPED planner driving a run the operator believes the policy is driving."""
    module = _stub_module()
    del module.VisualNavigator
    try:
        _install(module, lambda **kwargs: "planner", lambda loco, planner: None)
    except SystemExit as exc:
        assert "VisualNavigator" in str(exc)
    else:
        raise AssertionError("a missing global was patched over silently")


def test_the_navigator_wrapper_hands_the_locomotion_client_to_the_planner():
    """The seam that gets the MEASURED velocity into the policy. Without it every
    observation carries a zero velocity and nothing anywhere says so."""
    module = _stub_module()
    seen = []
    _install(module, lambda **kwargs: "planner",
             lambda loco, planner: seen.append((loco, planner)))
    module.VisualNavigator("loco", "perception", "planner", "extra")
    assert seen == [("loco", "planner")]


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
