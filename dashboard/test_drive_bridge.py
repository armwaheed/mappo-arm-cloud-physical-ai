#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SDK-env worker: the refusals, the stop, and the interface it stands in for.

Two of these are about what CANNOT happen. A worker that returns without stopping the robot
leaves ``SportClient.Move``'s last velocity latched, because that command has no dead-man
timeout — so ``test_the_robot_is_stopped_even_when_the_nudge_raises`` is the most important
test in the directory. And the gait-floor refusals are checked to fire without a platform at
all, which is the property that lets a bad speed be rejected without connecting to a robot.

The interface test is the one that would notice the real bindings drifting. ``SimLocomotion``
is a stand-in, and a stand-in that no longer implements what it stands in for produces green
tests and a broken first live run.

Pure stdlib, Python 3.8-compatible. ``python3 test_drive_bridge.py``.
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_bridge import (
    GAIT_FLOORS,
    MAX_REVERSE_SECONDS,
    MAX_SECONDS,
    MOTION_COMMANDS,
    BridgeError,
    SimLocomotion,
    build_parser,
    check_gait_floor,
    command_lie_down,
    command_pose_stream,
    load_platform,
    main,
    plan_nudge,
    run_nudge,
)


def _args(argv):
    return build_parser().parse_args(argv)


# ── the gate ─────────────────────────────────────────────────────────────────
def test_every_motion_command_is_refused_without_allow_motion():
    """A dashboard that can walk a robot the moment it is discovered is the thing to prevent.

    Checked through ``main`` rather than ``dispatch`` so the refusal is asserted on the JSON
    the driver actually parses, and on the non-zero exit status.
    """
    for command in ("stand", "stand-down", "walk", "strafe", "turn"):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([command, "--platform", "sim"])
        result = json.loads(buffer.getvalue().strip().splitlines()[-1])
        assert code == 1, command
        assert result["ok"] is False and result["refused"] is True, result
        assert "--allow-motion" in result["error"], result["error"]


def test_status_and_stop_are_never_gated():
    """A stop you cannot issue because motion is disabled is the wrong affordance."""
    for command in ("status", "stop"):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main([command, "--platform", "sim"])
        result = json.loads(buffer.getvalue().strip().splitlines()[-1])
        assert code == 0 and result["ok"] is True, (command, result)


# ── gait floors ──────────────────────────────────────────────────────────────
def test_a_speed_below_the_measured_floor_is_refused_without_a_robot():
    """0.21 m/s is the speed that cost five runs and two controllers. It never runs again."""
    try:
        check_gait_floor("go2", "forward", 0.21)
    except BridgeError as exc:
        assert "0.350" in str(exc) and "stands still" in str(exc), str(exc)
        return
    raise AssertionError("a sub-floor forward command was allowed")


def test_the_floor_can_be_overridden_but_says_so():
    warning = check_gait_floor("go2", "forward", 0.21, force=True)
    assert warning and "forced" in warning, warning


def test_an_axis_with_no_measured_floor_warns_instead_of_refusing():
    """The Go2's YAW floor has never been measured, and a Go2 that walks may still turn.

    Refusing outright would block a control whose behaviour is simply unknown on a robot
    whose gait is not. So: allow, and say on every press that it may do nothing. Kept on
    yaw because the lateral axis stopped being the example — see the test below.
    """
    warning = check_gait_floor("go2", "yaw", 0.70)
    assert warning is not None and "no yaw gait floor" in warning, warning
    assert GAIT_FLOORS["go2"]["yaw"] is None
    # the warn branch must not be the only branch a robot with a real floor can reach
    assert GAIT_FLOORS["go2"]["forward"] is not None


def test_the_lateral_floor_of_2026_08_19_belongs_to_the_go2_that_produced_it():
    """Issue #42's table: vy 0.15 travelled 0.010 m (no gait), vy 0.20 walked 3 of 3.

    That session is a Go2 session — same-evening control step at vx 0.35, the 85.27 deg
    HFOV and the 0.32 m camera height are the Go2's, and `integration/mappo_drive.py`
    already treats 0.20 as the Go2's lateral floor (`lateral floor 0.20 == max_vy`). The
    table used to record it against the LITE3, which has never moved at all, while telling
    the Go2 its own lateral floor had never been measured. This pins the owner.
    """
    assert GAIT_FLOORS["go2"]["lateral"] == 0.20
    assert check_gait_floor("go2", "lateral", 0.20) is None
    try:
        check_gait_floor("go2", "lateral", 0.15)
    except BridgeError as exc:
        assert "0.200" in str(exc), str(exc)
    else:
        raise AssertionError("0.15 m/s lateral was allowed on a Go2")
    assert GAIT_FLOORS["lite3"]["lateral"] is None


def test_no_lite3_axis_carries_a_number_it_did_not_produce():
    """Issue #13's measurements are all open and neither Venture has moved under this stack.

    The failure this pins is not "the value is wrong" — it is a value being present at all.
    Any float here, however plausible, came from somewhere else, because there is nowhere
    on a Lite3 it could have come from.
    """
    assert GAIT_FLOORS["lite3"] == {"forward": None, "lateral": None, "yaw": None}
    for axis, go2_value in GAIT_FLOORS["go2"].items():
        assert GAIT_FLOORS["lite3"][axis] != go2_value or go2_value is None, axis


def test_a_platform_with_nothing_measured_refuses_rather_than_warns():
    """The Lite3's own navigator answers a live run with no --gait-floor by refusing.

    A dashboard button carries no more authority than that, so it does not get a softer
    rule. The refusal has to fire at a HIGH speed too: the failure mode here is not "too
    slow", it is "nobody knows", and a fast command is no better informed than a slow one.
    """
    for axis, speed in (("forward", 0.35), ("lateral", 0.20), ("yaw", 0.70),
                        ("forward", 2.0)):
        try:
            check_gait_floor("lite3", axis, speed)
        except BridgeError as exc:
            assert "ever been measured on the lite3" in str(exc), (axis, str(exc))
            assert "issue #13" in str(exc), (axis, str(exc))
        else:
            raise AssertionError(f"a lite3 {axis} command at {speed} was allowed")


def test_the_uncommissioned_refusal_is_derived_from_the_table_not_a_second_list():
    """A hand-kept list of "platforms with nothing measured" is a gate that goes stale.

    The first real Lite3 measurement must switch this off by itself. Simulated here by
    putting one number into the row and checking the other axes drop back to warning —
    the same treatment the Go2's yaw gets.
    """
    original = dict(GAIT_FLOORS["lite3"])
    try:
        GAIT_FLOORS["lite3"] = dict(original, forward=0.30)
        assert check_gait_floor("lite3", "forward", 0.30) is None
        warning = check_gait_floor("lite3", "lateral", 0.20)
        assert warning is not None and "no lateral gait floor" in warning, warning
    finally:
        GAIT_FLOORS["lite3"] = original


def test_forcing_past_an_uncommissioned_platform_says_what_it_forced():
    warning = check_gait_floor("lite3", "forward", 0.35, force=True)
    assert warning and "forced" in warning, warning
    assert "never been measured" in warning or "no axis" in warning, warning


# ── planning ─────────────────────────────────────────────────────────────────
def test_reverse_is_capped_harder_than_forward_and_is_not_floor_checked():
    """Reverse gets a shorter leash and a statement, not a floor.

    The floor is a measurement of the FORWARD gait; applying it to reverse would be the same
    axis conflation again.
    """
    vx, _vy, _wz, seconds, warning = plan_nudge(
        _args(["walk", "--platform", "go2", "--vx", "-0.35", "--seconds", "5.0"]))
    assert vx < 0 and seconds == MAX_REVERSE_SECONDS, (vx, seconds)
    assert "REVERSE" in warning and "rear sensing" in warning, warning


def test_seconds_is_capped_for_every_command():
    """A web button must not be able to start a walk that outlives the operator's attention."""
    for command, flag, value in (("walk", "--vx", "0.35"),
                                 ("strafe", "--vy", "0.20"),
                                 ("turn", "--wz", "0.70")):
        _, _, _, seconds, _ = plan_nudge(
            _args([command, "--platform", "sim", flag, value, "--seconds", "999"]))
        assert seconds == MAX_SECONDS, (command, seconds)


def test_planning_touches_no_robot():
    """The refusal must be decidable from the table, so a bad speed costs no DDS connection.

    ``plan_nudge`` takes no locomotion object at all, which is how this is guaranteed rather
    than hoped for.
    """
    import inspect
    parameters = list(inspect.signature(plan_nudge).parameters)
    assert parameters == ["args"], parameters


# ── the stop ─────────────────────────────────────────────────────────────────
class _Recorder(SimLocomotion):
    """A sim that records every command and can be told to explode mid-nudge."""

    def __init__(self, explode_after=None):
        SimLocomotion.__init__(self)
        self.commands = []
        self.stops = 0
        self._explode_after = explode_after

    def set_velocity(self, vx, vy, vyaw):
        self.commands.append((vx, vy, vyaw))
        if self._explode_after is not None and len(self.commands) >= self._explode_after:
            raise RuntimeError("the bus went away mid-walk")
        SimLocomotion.set_velocity(self, vx, vy, vyaw)

    def stop(self):
        self.stops += 1
        SimLocomotion.stop(self)


def test_the_robot_is_stopped_even_when_the_nudge_raises():
    """THE test. ``Move`` persists until the next command, so a return without a stop walks.

    Made to fail by deleting the ``finally`` in ``run_nudge``: the exception propagates and
    ``stops`` stays at 0.
    """
    loco = _Recorder(explode_after=3)
    loco.connect()
    try:
        run_nudge(loco, 0.35, 0.0, 0.0, 2.0, sleep=lambda _s: None)
    except RuntimeError:
        assert loco.stops == 1, f"the robot was not stopped: stops={loco.stops}"
        return
    raise AssertionError("the injected failure did not propagate")


def test_a_normal_nudge_also_ends_in_exactly_one_stop():
    loco = _Recorder()
    loco.connect()
    ticks = [0.0]

    def clock():
        ticks[0] += 0.1
        return ticks[0]

    result = run_nudge(loco, 0.35, 0.0, 0.0, 1.0, sleep=lambda _s: None, clock=clock)
    assert loco.stops == 1, loco.stops
    assert result["ok"] is True
    assert loco.commands, "no velocity was ever commanded"
    assert all(c == (0.35, 0.0, 0.0) for c in loco.commands), loco.commands


def test_the_nudge_refreshes_rather_than_commanding_once():
    """``SportClient.Move`` has no dead-man, so a single call plus a sleep is not a walk —
    but it is also not obviously wrong, which is why this is pinned."""
    loco = _Recorder()
    loco.connect()
    ticks = [0.0]

    def clock():
        ticks[0] += 0.1
        return ticks[0]

    result = run_nudge(loco, 0.35, 0.0, 0.0, 1.0, sleep=lambda _s: None, clock=clock)
    assert result["ticks"] >= 5, result["ticks"]


def test_the_result_reports_what_moved_not_what_was_asked_for():
    """Every button is a measurement — that is the argument for open-loop nudges."""
    loco = SimLocomotion()
    loco.connect()
    result = run_nudge(loco, 0.35, 0.0, 0.0, 0.5)
    assert result["pose_before"] is not None and result["pose_after"] is not None
    assert result["travelled_m"] > 0.0
    assert "delivered_fraction" in result


def test_a_simulated_platform_keeps_the_real_platforms_rules():
    """--platform is the RULES; --backend is what is driven. Collapsing them breaks a demo.

    The first version of --simulate passed "sim" as the platform, which handed the demo the
    bench double's gait floors of zero — so a simulated Go2 happily accepted 0.21 m/s and the
    refusal that is this stack's most characteristic behaviour never fired. A demo that
    teaches the wrong number is worse than no demo.
    """
    refused = []
    try:
        plan_nudge(_args(["walk", "--platform", "go2", "--backend", "sim", "--vx", "0.21"]))
    except BridgeError as exc:
        refused.append(str(exc))
    assert refused and "0.350" in refused[0], (
        "a simulated go2 accepted a sub-gait-floor speed; --backend leaked into the rules")

    # ...and the real floor still lets a walkable speed through.
    vx, _vy, _wz, _s, warning = plan_nudge(
        _args(["walk", "--platform", "go2", "--backend", "sim", "--vx", "0.35"]))
    assert vx == 0.35 and warning is None, (vx, warning)


def test_the_backend_defaults_to_the_platform():
    """Without --backend nothing changes, so a real run is unaffected by the demo path."""
    args = _args(["status", "--platform", "go2"])
    assert args.backend == "", args.backend


# ── platform honesty ─────────────────────────────────────────────────────────
def test_lie_down_reports_that_a_lite3_was_not_laid_down():
    """A green tick for both platforms would tell the operator something untrue.

    ``Lite3Locomotion.stand_down`` only stops — posture there is operator-controlled through
    the vendor app.
    """
    go2 = command_lie_down(SimLocomotion(), "go2")
    lite3 = command_lie_down(SimLocomotion(), "lite3")
    assert go2["postured"] is True
    assert lite3["postured"] is False
    assert "not laid down" in lite3["note"] or "STOPPED" in lite3["note"], lite3["note"]


def test_the_sim_double_implements_what_the_real_bindings_do():
    """A stand-in that has drifted from the interface gives green tests and a broken run.

    Read with ``ast`` rather than imported. Importing ``go2_locomotion`` pulls in
    ``arm_dc_robotkit`` and, through it, the Unitree SDK — neither of which is on a
    workstation — so an import-based version of this test SKIPS everywhere it matters and
    only ever runs on the robot, where it is too late. Parsing the source needs nothing and
    runs on every machine, which is the difference between a check and a check that fires.

    Both real bindings are compared, because the worker drives both.
    """
    import ast

    here = os.path.dirname(os.path.abspath(__file__))
    bindings = {
        "Go2Locomotion": os.path.join(
            here, "..", "robot-stack", "unitree", "go2", "locomotion", "go2_locomotion.py"),
        "Lite3Locomotion": os.path.join(
            here, "..", "robot-stack", "deep_robotics", "lite3", "locomotion",
            "lite3_locomotion.py"),
    }
    # What the worker actually calls on a locomotion object. Kept here rather than derived,
    # because the point is to pin the worker's demands against both real classes.
    required = {"connect", "set_velocity", "stop", "pose", "velocity",
                "stand", "stand_down", "shutdown"}
    sim_members = set(dir(SimLocomotion))

    for class_name, path in bindings.items():
        path = os.path.normpath(path)
        assert os.path.isfile(path), f"{class_name} source not found at {path}"
        with open(path) as handle:
            tree = ast.parse(handle.read())
        node = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == class_name), None)
        assert node is not None, f"{class_name} not found in {path}"
        real = {m.name for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}

        # Everything the worker calls must exist on the real binding...
        missing_real = sorted(required - real)
        assert not missing_real, (
            f"the worker calls {missing_real} but {class_name} no longer defines them")
        # ...and on the double that stands in for it.
        missing_sim = sorted(required - sim_members)
        assert not missing_sim, f"SimLocomotion is missing {missing_sim}"


def test_the_lite3_cannot_be_laid_down_and_the_worker_knows_it():
    """Pinned against the vendored binding's source, not against a memory of it.

    ``Lite3Locomotion.stand_down`` delegates to ``stop``. If someone gives the Lite3 a real
    posture command upstream, this fails and ``command_lie_down`` needs revisiting — which
    is exactly when it should be revisited.
    """
    import ast

    path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "robot-stack", "deep_robotics",
        "lite3", "locomotion", "lite3_locomotion.py"))
    with open(path) as handle:
        tree = ast.parse(handle.read())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "Lite3Locomotion")
    stand_down = next(m for m in node.body
                      if isinstance(m, ast.FunctionDef) and m.name == "stand_down")
    calls = {n.func.attr for n in ast.walk(stand_down)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert calls == {"stop"}, (
        f"Lite3Locomotion.stand_down now calls {sorted(calls)}, not just stop(). "
        f"command_lie_down reports postured=False for the Lite3 on the strength of that.")


def test_an_unknown_platform_is_named_rather_than_crashing():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(["status", "--platform", "sim"])
    assert code == 0
    try:
        _args(["status", "--platform", "nonesuch"])
    except SystemExit:
        return                                    # argparse rejects it at the boundary
    raise AssertionError("an unknown platform was accepted")


# ── the pose stream ──────────────────────────────────────────────────────────
def test_the_pose_stream_never_commands_a_velocity():
    """THE test for this command, and the reason it is allowed to outlive one result.

    The "one command, one process, exit" rule in this module is a guard against a LATCHED
    VELOCITY: ``Move`` has no dead-man timeout, so a worker that does not exit can leave
    one on the bus. A reader that never calls ``set_velocity`` has nothing to latch, which
    is what buys the exception — and this is what keeps that true. Fails the moment
    anything in the stream path touches the legs.
    """
    loco = _Recorder()
    buffer = io.StringIO()
    command_pose_stream(loco, "go2", 100.0, out=buffer, sleep=lambda _s: None, limit=5)
    assert loco.commands == [], f"the pose stream commanded {loco.commands}"
    assert loco.stops == 0, "a pose READER must not stop a peer that is walking"


def test_every_pose_line_carries_a_clock_reading_and_a_pose():
    """``mono_s`` is what the driver turns into an age. A line without one is dropped by
    ``robot_driver._pose_line``, because an undatable safety input is not one."""
    buffer = io.StringIO()
    command_pose_stream(SimLocomotion(), "go2", 100.0, out=buffer,
                        sleep=lambda _s: None, limit=3)
    lines = [json.loads(line) for line in buffer.getvalue().splitlines()]
    assert len(lines) == 3
    for line in lines:
        assert isinstance(line["mono_s"], float)
        assert set(line["pose"]) == {"x", "y", "yaw"}
        assert len(line["velocity"]) == 3
        assert line["ok"] is True


def test_a_throwing_estimator_is_reported_rather_than_skipped():
    """A skipped sample is indistinguishable, at the far end, from a network drop — and
    the far end's answer to a drop is to stop the other robot. Reporting it means the
    diagnosis names the estimator instead."""
    class _Broken(SimLocomotion):
        def pose(self):
            raise RuntimeError("the estimator went away")

    buffer = io.StringIO()
    command_pose_stream(_Broken(), "go2", 100.0, out=buffer, sleep=lambda _s: None,
                        limit=2)
    lines = [json.loads(line) for line in buffer.getvalue().splitlines()]
    assert len(lines) == 2, "a failed read produced no line at all"
    assert all(line["ok"] is False for line in lines)
    assert all("estimator went away" in line["error"] for line in lines)


def test_the_pose_stream_ends_quietly_when_the_driver_goes_away():
    """The parent is the only consumer, so a closed pipe is a shutdown it initiated. A
    traceback about a broken pipe would be the last thing in its log about it."""
    class _Closed(io.StringIO):
        def write(self, _text):
            raise BrokenPipeError(32, "Broken pipe")

    result = command_pose_stream(SimLocomotion(), "go2", 100.0, out=_Closed(),
                                 sleep=lambda _s: None, limit=100)
    assert result["ok"] is True
    assert result["streamed"] == 0
    assert "parent closed" in result["stopped"]


def test_the_pose_stream_runs_without_allow_motion():
    """A peer publishing where it is must not require the flag that lets it be walked.
    Motion is gated by membership of ``MOTION_COMMANDS``, so this asserts the membership
    rather than the behaviour it produces — that is where the decision actually lives."""
    assert "pose-stream" not in MOTION_COMMANDS
    assert "pose-stream" in build_parser().parse_args(["pose-stream"]).command


# ── the virtualenv guard ─────────────────────────────────────────────────────────────────
#
# What the guard DECIDES is tested in robot-stack/preflight/test_venv_guard.py against an
# injected environment and filesystem. These test that this worker calls it on the two
# platforms that open a vendor transport, that it does NOT call it for the bench double,
# and that a refusal comes back as JSON rather than as a stack trace — because the driver
# reads the last line of stdout and would otherwise surface only 400 characters of stderr.

def _guarded_load(platform, decision_refuses, message="[venv-guard] REFUSING TO RUN: x"):
    """Drive load_platform with the guard's verdict forced, and record that it was asked."""
    import drive_bridge

    calls = []

    class _Decision:
        refuse = decision_refuses
        message = None

    def fake_evaluate(component, reaching_hardware, **kwargs):
        calls.append((component, reaching_hardware))
        verdict = _Decision()
        verdict.message = message
        return verdict

    # The worker imports `evaluate` inside _require_virtualenv, so the module it resolves
    # is patched rather than a name already bound in drive_bridge's namespace.
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(drive_bridge.__file__)), "..", "robot-stack",
        "preflight"))
    import venv_guard

    original = venv_guard.evaluate
    venv_guard.evaluate = fake_evaluate
    try:
        error = None
        try:
            load_platform(platform)
        except BridgeError as exc:
            error = str(exc)
        except Exception as exc:  # an SDK import failure on a machine with no robot
            error = f"{type(exc).__name__}: {exc}"
        return calls, error
    finally:
        venv_guard.evaluate = original


def test_a_real_platform_asks_the_virtualenv_guard_before_importing_the_vendor_sdk():
    for platform in ("go2", "lite3"):
        calls, _error = _guarded_load(platform, decision_refuses=False)
        assert calls == [(f"drive_bridge --platform {platform}", True)], (platform, calls)


def test_the_bench_double_never_asks_because_it_opens_no_transport():
    calls, error = _guarded_load("sim", decision_refuses=False)
    assert calls == []
    assert error is None


def test_a_refused_interpreter_comes_back_as_a_bridge_error_not_a_traceback():
    """BridgeError is what `main` turns into {"ok": false, "refused": true, ...}.

    A bare SystemExit would exit with no JSON on stdout at all, and robot_driver reports
    that as "worker exited 1 with no JSON result" plus the last 400 characters of stderr —
    which truncates the refusal from the front, losing the line that says what happened.
    """
    calls, error = _guarded_load("go2", decision_refuses=True)
    assert calls == [("drive_bridge --platform go2", True)]
    assert error is not None
    assert "REFUSING TO RUN" in error


def test_the_refusal_survives_the_json_round_trip_main_puts_it_through():
    import drive_bridge

    original = drive_bridge.load_platform

    def refuse(*_a, **_k):
        raise BridgeError("[venv-guard] REFUSING TO RUN: system Python on a robot")

    drive_bridge.load_platform = refuse
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            code = main(["status", "--platform", "go2"])
    finally:
        drive_bridge.load_platform = original
    result = json.loads(buffer.getvalue().strip().splitlines()[-1])
    assert code == 1
    assert result["ok"] is False
    assert result["refused"] is True
    assert "REFUSING TO RUN" in result["error"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"drive_bridge: {len(tests)}/{len(tests)} passed")
