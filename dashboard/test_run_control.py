#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the run gate: what refuses, what is spelled, and what a stop can reach.

Two of these are the ones worth reading. ``test_a_live_run_cannot_be_built_without_motion``
is the motion gate — the same gate ``walk_forward`` has, on the path that hands the legs to
a policy indefinitely rather than for 5 s. ``test_no_command_this_module_builds_can_send_a
_sigkill`` is ``SAFETY.md`` §0, which was written from a robot that broke a window because a
process commanding motors was hard-killed.

The rest are about a command line a browser can influence and a robot will run. The values a
page supplies are checked against fixed tuples, everything crossing a shell is
``shlex.quote``-d, and the profile — which is a file on the operator's own machine — may not
spell any flag this module owns.

Pure stdlib, no Device Connect, no robot. ``python3 test_run_control.py``.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_control
from run_control import (
    MAX_RUN_SECONDS,
    RESERVED_FLAGS,
    RunProfile,
    RunRecord,
    RunRefused,
    build_run_argv,
    check_control_mode,
    clamp_seconds,
    describe,
    launch_command,
    load_profile,
    new_run_id,
    output_paths,
    pidfile_for,
    stop_command,
    unsupported,
)

#: A profile shaped like the real deployment: the driver is on a workstation, the run is on
#: the robot, and the only way across is ssh. Every field here is a value that appeared in
#: `deploy/README.md` or on the lab Go2 on 2026-08-26.
REMOTE = RunProfile(
    label="lab go2",
    workdir="/home/unitree/mappo-run/integration",
    python="/home/unitree/robotics-connect-envs/armwaheed/bin/python3",
    script="mappo_drive.py",
    package="/home/unitree/mappo-run/policy",
    env_setup="/home/unitree/mappo-run/robot-stack/unitree/go2/install/setup_env.sh",
    launch_prefix=("ssh", "-o", "BatchMode=yes", "unitree@192.168.123.18"),
    extra_args=("--robot-radius", "0.25", "--no-latch-arm"),
    output_dir="/home/unitree/dashboard-runs",
)

#: The bench-double shape: the run is a child of the driver, so there is no ssh and no pid
#: file. Correct for `--platform sim` and for nothing with legs.
LOCAL = RunProfile(label="bench double", workdir="../integration", package="../policy")


def _argv(profile=REMOTE, **kwargs):
    params = {"seconds": 30.0, "policy_mode": "supervised", "heading_servo": "off",
              "live": True, "allow_motion": True, "run_id": "RID"}
    params.update(kwargs)
    return build_run_argv(profile, **params)


# ── the motion gate ──────────────────────────────────────────────────────────
def test_a_live_run_cannot_be_built_without_motion_enabled():
    """⛔ THE gate. A live run hands the legs to a learned policy for its whole length.

    It is the same refusal ``walk_forward`` gives and the same one ``drive_bridge.dispatch``
    gives a motion command, and it is checked in both the driver and here on purpose: two
    processes with one shared assumption is one process with an unwritten contract.

    Made to fail by deleting the ``if live and not allow_motion`` block in
    ``build_run_argv``: the command line then comes back carrying ``--live`` from a device
    that was started status-and-checkpoints only.
    """
    try:
        _argv(allow_motion=False)
    except RunRefused as exc:
        assert "--allow-motion" in str(exc), str(exc)
        return
    raise AssertionError("a live run was built on a device without --allow-motion")


def test_the_gate_refuses_and_never_downgrades_to_a_dry_run():
    """A silent downgrade is the worse of the two failures.

    The operator asked for motion, watched a run start, and is now watching a robot that was
    never going to move — the same shape as ``mode='mcf'`` and as a sub-gait-floor command,
    both refused here rather than warned about. Made to fail by returning a non-live argv
    instead of raising.
    """
    try:
        _argv(allow_motion=False)
    except RunRefused as exc:
        assert "refused rather than quietly downgraded" in str(exc), str(exc)
        return
    raise AssertionError("the gate downgraded a live request instead of refusing it")


def test_the_default_run_needs_no_motion_flag_and_cannot_command_a_leg():
    """The scene check, which is what ``start_run`` does with no arguments.

    Perception, the policy, the veto and the telemetry, with no ``--live``. ``--live`` is
    the only flag in ``mappo_drive.py`` that commands a leg, so this is an absent capability
    rather than a checked permission — which is what makes it safe for a button with no
    flags. Gating it on ``--allow-motion`` would refuse a run that cannot move the robot.
    """
    argv = _argv(live=False, allow_motion=False)
    assert "--live" not in argv, argv
    assert "--policy-mode" in argv and "--max-seconds" in argv, argv


def test_the_only_thing_that_moves_a_robot_is_the_live_flag():
    """Keeps the gate above from becoming vacuous. If ``--live`` stopped being what moves
    the robot, the flag this module gates would no longer be the one that matters."""
    live = _argv(live=True)
    dry = _argv(live=False, allow_motion=False)
    assert [a for a in live if a not in dry] == ["--live"], (live, dry)


# ── SAFETY.md §0 ─────────────────────────────────────────────────────────────
def test_no_command_this_module_builds_can_send_a_sigkill():
    """⛔ The cardinal rule: a hard kill is the OPPOSITE of a stop.

    ``SAFETY.md`` §0 is written from an incident — a process commanding motors was
    ``kill -9``'d "to stop the commands", the last high-gain target stayed latched with no
    publisher left to update or damp it, and the robot spin-kicked and broke a window. A
    SIGKILL cannot be caught, so a process cannot damp on the way out.

    ``mappo_drive`` inherits ``visual_nav``'s teardown and the shared ``SafeStop``, both of
    which damp on SIGTERM and neither of which runs on SIGKILL. Made to fail by changing
    ``kill -TERM`` to ``kill -9`` in ``stop_command``.
    """
    for profile in (REMOTE, LOCAL):
        rendered = " ".join(launch_command(profile, _argv(profile),
                                           pidfile_for(profile, "RID")))
        stop = stop_command(profile, pidfile_for(profile, "RID"))
        rendered += " " + " ".join(stop or [])
        for forbidden in ("-9", "SIGKILL", "-KILL", "kill -s KILL", "pkill"):
            assert forbidden not in rendered, (
                f"{forbidden!r} appears in a command this module builds: {rendered}")
    assert "kill -TERM" in " ".join(stop_command(REMOTE, "/tmp/p.pid"))


# ── what a page may say ──────────────────────────────────────────────────────
def test_a_policy_mode_the_page_made_up_never_reaches_the_command_line():
    """This is the boundary between a browser and a robot's argv.

    Everything crossing the shell is quoted, so an invented value could not execute
    anything — it would fail several seconds and one SSH connection later, on the far end,
    as an argparse usage message nobody sees.
    """
    for mode in ("Supervised", "", "raw ", "--live", "supervised; reboot"):
        try:
            _argv(policy_mode=mode)
        except RunRefused as exc:
            assert "policy_mode" in str(exc)
            continue
        raise AssertionError(f"policy_mode={mode!r} reached the command line")


def test_a_heading_servo_the_page_made_up_never_reaches_the_command_line():
    for servo in ("Off", "none", "--live", "travel!"):
        try:
            _argv(heading_servo=servo)
        except RunRefused:
            continue
        raise AssertionError(f"heading_servo={servo!r} reached the command line")


def test_shell_metacharacters_in_a_deployment_path_cannot_break_out():
    """A remote run is a shell line, so every element is quoted.

    The path here is not an attack — it is a directory with a space in it, plus what would
    happen if one ever had a semicolon. The quoting has to hold for both.
    """
    nasty = "/home/unitree/my runs; rm -rf ~"
    profile = RunProfile(**{**REMOTE.__dict__, "workdir": nasty})
    remote = launch_command(profile, _argv(profile), "/tmp/p.pid")[-1]
    assert shlex.quote(nasty) in remote, remote
    assert "; rm -rf ~ " not in remote.replace(shlex.quote(nasty), ""), remote


# ── the flags are spelled, never inherited ───────────────────────────────────
def test_every_flag_this_driver_decides_is_spelled_even_at_its_default():
    """A run that omits ``--heading-servo`` has its control law decided by the age of the
    checkout it runs from.

    The servo became opt-in in #106; the tree on the lab Go2 predates that by 43 commits
    and still has the old behaviour, and issue #16's servo put this robot into a wall on
    three runs out of four. ``supervised`` and ``30`` are defaults too and are sent for the
    same reason. Made to fail by making any of the three conditional on being non-default.
    """
    argv = _argv(policy_mode="supervised", heading_servo="off", seconds=30.0)
    for flag, value in (("--policy-mode", "supervised"), ("--heading-servo", "off"),
                        ("--max-seconds", "30")):
        assert flag in argv, f"{flag} was omitted because it was the default"
        assert argv[argv.index(flag) + 1] == value, (flag, argv)


# ── the flag that does not exist on the tree that would run it ───────────────
def test_a_tree_that_predates_the_rename_is_sent_the_spelling_it_understands():
    """⚠️ Measured, not anticipated. On 2026-08-26 the lab Go2's ~/mappo-main answered its
    own ``--help`` with ``[--no-heading-servo]`` and no ``--heading-servo``: it is 67
    commits behind #106, which renamed the flag.

    Sending the new spelling there makes argparse exit 2 on the far end of an SSH
    connection, several seconds later, and the run simply never starts with no reason an
    operator can see. Made to fail by writing ``--heading-servo`` unconditionally.
    """
    legacy = RunProfile(**{**REMOTE.__dict__, "heading_servo_flag": "legacy"})
    argv = _argv(legacy, heading_servo="off")
    assert "--no-heading-servo" in argv and "--heading-servo" not in argv, argv


def test_the_servo_is_spelled_on_a_legacy_tree_and_never_merely_omitted():
    """Omitting it there is not "the default": it is issue #16's ``travel`` law, which
    saturated the yaw rate and put this robot into a cubicle panel or a cabinet on three
    runs out of four on 2026-08-17."""
    legacy = RunProfile(**{**REMOTE.__dict__, "heading_servo_flag": "legacy"})
    argv = _argv(legacy, heading_servo="off")
    assert any(a.endswith("heading-servo") for a in argv), (
        "the run would inherit whatever the tree's default is, which on that tree is #16")


def test_a_mode_a_legacy_tree_cannot_select_is_refused_here_not_on_the_far_end():
    """The rejection would otherwise arrive as a usage message across an SSH connection and
    read as "the run did not start"."""
    legacy = RunProfile(**{**REMOTE.__dict__, "heading_servo_flag": "legacy"})
    for servo in ("goal", "travel"):
        try:
            _argv(legacy, heading_servo=servo)
        except RunRefused as exc:
            assert "predates #106" in str(exc), str(exc)
            continue
        raise AssertionError(f"heading_servo={servo!r} was sent to a tree that cannot take it")


def test_a_legacy_tree_advertises_only_the_mode_it_can_be_told():
    """A page offering three modes when two of them are refusals is a page offering
    refusals."""
    legacy = RunProfile(**{**REMOTE.__dict__, "heading_servo_flag": "legacy"})
    assert describe(legacy, allow_motion=True)["heading_servos"] == ["off"]
    assert describe(REMOTE, allow_motion=True)["heading_servos"] == list(
        run_control.HEADING_SERVOS)


def test_a_spelling_nobody_has_heard_of_is_refused_at_startup():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"workdir": "/x", "script": "s.py",
                            "heading_servo_flag": "whatever"})
        try:
            load_profile(path)
        except ValueError as exc:
            assert "heading_servo_flag" in str(exc)
            return
    raise AssertionError("an unknown servo spelling loaded")


def test_a_profile_may_not_spell_a_flag_this_module_owns():
    """A profile that could set ``--live`` would be a second motion gate nobody would think
    to look for; one that could set ``--policy-mode`` would make the page's choice a
    suggestion. argparse takes the LAST occurrence, so the winner would be decided by
    string order in a build function."""
    for flag in RESERVED_FLAGS:
        profile = RunProfile(**{**REMOTE.__dict__, "extra_args": (flag, "x")})
        try:
            build_run_argv(profile, seconds=10, policy_mode="supervised",
                           heading_servo="off", live=False, allow_motion=True, run_id="R")
        except ValueError as exc:
            assert flag in str(exc)
            continue
        raise AssertionError(f"a profile smuggled {flag} onto the command line")


def test_the_reserved_flags_are_the_ones_this_module_actually_writes():
    """Keeps the list above from going stale in the harmless-looking direction.

    A flag this module writes but does not reserve can be written twice, and the profile's
    copy would sometimes win.
    """
    for profile in (REMOTE, RunProfile(**{**REMOTE.__dict__,
                                          "heading_servo_flag": "legacy"})):
        argv = _argv(profile=profile, live=True)
        written = {a for a in argv if a.startswith("--")} - set(profile.extra_args)
        assert written <= set(RESERVED_FLAGS), sorted(written - set(RESERVED_FLAGS))


# ── the mode that makes a robot ignore every command ─────────────────────────
def test_the_go2s_mcf_mode_refuses_a_run_rather_than_warning_about_it():
    """⚠️ The Go2 failure with no symptom, measured on the lab robot on 2026-08-26.

    It answered ``mode='mcf'``. ``SportClient.Move`` is accepted in ``normal`` and ``ai``
    and nowhere else, so a run started here would command velocities for its whole length
    and the robot would never step: no fall, no fault, no error code. It is the gait floor's
    shape, and the gait floor is refused rather than warned about for exactly this reason.

    Made to fail by returning a warning string instead of raising.
    """
    try:
        check_control_mode("go2", "mcf")
    except RunRefused as exc:
        assert "mcf" in str(exc), "the refusal does not name the mode it read"
        assert "ensure_sport_mode" in str(exc), "the refusal does not say how to fix it"
        return
    raise AssertionError("a run was permitted on a Go2 that would have ignored it")


def test_a_go2_in_a_sport_mode_is_not_held_up():
    for mode in run_control.SPORT_MODES:
        assert check_control_mode("go2", mode) is None


def test_a_mode_that_could_not_be_read_fails_closed():
    """``CheckMode`` returning nothing is not evidence that the mode is fine."""
    for mode in (None, "", "unknown"):
        try:
            check_control_mode("go2", mode)
        except RunRefused:
            continue
        raise AssertionError(f"mode={mode!r} was treated as good enough to run")


def test_the_bench_double_is_not_held_to_a_gate_about_a_controller_it_has_not_got():
    """A simulated Go2 keeps the Go2's RULES — its gait floors, its posture semantics — but
    its mode comes from ``SimLocomotion`` and reads ``'sim'``. Refusing that would make
    ``--simulate`` unusable for the demo it exists for."""
    assert check_control_mode("go2", "sim", simulated=True)
    assert check_control_mode("sim", "sim")


def test_the_lite3_gets_a_note_and_not_an_invented_gate():
    """Its posture and navigation mode are the operator's confirmation on the vendor
    interface, which is what ``--operator-ready`` asserts and which nothing here can check.
    A gate whose truth this cannot verify would be a gate that reports on nothing."""
    note = check_control_mode("lite3", None)
    assert note and "operator" in note, note


# ── a stop has to be able to reach the run ───────────────────────────────────
def test_a_remote_launch_records_a_pid_that_a_stop_can_signal():
    """SIGTERM to the local ssh client closes a socket. The process on the far end never
    hears about it, and the policy keeps driving.

    So the launch writes the remote shell's pid before ``exec``-ing the run — ``exec`` keeps
    the pid — and the stop is a second connection that signals it. Made to fail by dropping
    the ``echo $$`` clause: the stop command then has nothing to read.
    """
    pidfile = pidfile_for(REMOTE, "RID")
    remote = launch_command(REMOTE, _argv(), pidfile)[-1]
    assert f"echo $$ > {shlex.quote(pidfile)}" in remote, remote
    assert remote.index("echo $$") < remote.index("exec "), (
        "the pid is recorded after the exec that replaces the shell")
    stop = " ".join(stop_command(REMOTE, pidfile))
    assert shlex.quote(pidfile) in stop and "kill -TERM" in stop, stop


def test_a_remote_run_will_not_launch_without_somewhere_to_record_its_pid():
    """An unstoppable run is worse than a run that did not start."""
    try:
        launch_command(REMOTE, _argv(), "")
    except RunRefused as exc:
        assert "pidfile" in str(exc)
        return
    raise AssertionError("a remote run was launched with no way to signal it")


def test_a_local_run_has_no_shell_between_the_signal_and_the_run():
    """A ``/bin/sh -c`` wrapper would take the SIGTERM and exit, leaving the run.

    A local run is spawned as its own argv, so the signal lands on the process that is
    commanding the robot. It also needs no pidfile: the driver holds the handle.
    """
    command = launch_command(LOCAL, _argv(LOCAL), "")
    assert command == _argv(LOCAL), command
    assert "sh" not in command[0], command
    assert stop_command(LOCAL, "/tmp/p.pid") is None, (
        "a local run was given a scripted stop; it is signalled directly")


def test_the_remote_line_will_not_run_the_command_from_the_wrong_directory():
    """``cd dir; exec ...`` runs the exec wherever the login shell left us if the cd failed.
    Made to fail by joining the clauses with ``;``."""
    remote = launch_command(REMOTE, _argv(), "/tmp/p.pid")[-1]
    assert " && " in remote and "; exec" not in remote, remote


# ── bounds ───────────────────────────────────────────────────────────────────
def test_a_run_is_bounded_because_a_nudge_is():
    """A motion nudge is capped at 5 s so a web button cannot start a walk that outlives the
    attention of the person who pressed it. A run needs a much larger number and still needs
    one. Made to fail by removing the ``min`` in ``clamp_seconds``."""
    assert clamp_seconds(10_000) == MAX_RUN_SECONDS
    assert clamp_seconds(45) == 45.0
    argv = _argv(seconds=10_000)
    assert argv[argv.index("--max-seconds") + 1] == f"{MAX_RUN_SECONDS:g}"


def test_a_length_nobody_means_literally_becomes_the_default():
    for value in (0, -5, None, "soon", float("nan")):
        assert clamp_seconds(value) == run_control.DEFAULT_RUN_SECONDS, value


# ── evidence ─────────────────────────────────────────────────────────────────
def test_two_runs_started_in_the_same_second_do_not_share_a_telemetry_file():
    """A fixed ``--telemetry`` in a profile silently overwrites, and the file you go looking
    for afterwards is the one that got overwritten. Made to fail by making ``new_run_id``
    the timestamp alone."""
    ids = {new_run_id() for _ in range(200)}
    assert len(ids) == 200, f"only {len(ids)} distinct run ids in 200"
    first, second = sorted(ids)[:2]
    assert output_paths(REMOTE, first) != output_paths(REMOTE, second)


def test_a_run_id_is_safe_in_a_path_and_readable_in_a_log():
    run_id = new_run_id()
    assert shlex.quote(run_id) == run_id, run_id
    assert run_id.startswith("20") and run_id.endswith(tuple("0123456789abcdef"))


def test_a_profile_with_nowhere_to_write_produces_a_run_with_no_evidence_flags():
    """Stated rather than defaulted: a run with no telemetry is a run that proves nothing,
    and inventing a path on somebody else's robot is worse than saying so."""
    profile = RunProfile(**{**REMOTE.__dict__, "output_dir": ""})
    argv = build_run_argv(profile, seconds=10, policy_mode="supervised", heading_servo="off",
                          live=False, allow_motion=True, run_id="RID")
    assert "--telemetry" not in argv and "--record" not in argv, argv


def test_video_is_opt_in_because_a_missing_codec_fails_the_whole_run():
    assert "--record" not in _argv()
    recorded = RunProfile(**{**REMOTE.__dict__, "record": True})
    assert "--record" in build_run_argv(recorded, seconds=10, policy_mode="supervised",
                                        heading_servo="off", live=False, allow_motion=True,
                                        run_id="RID")


# ── the profile is a startup decision ────────────────────────────────────────
def _write(tmp, data):
    path = os.path.join(tmp, "profile.json")
    with open(path, "w") as handle:
        json.dump(data, handle)
    return path


def test_a_profile_that_names_a_reserved_flag_is_refused_at_startup():
    """At load, not at the first press. A demo morning is the wrong time to find out."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"workdir": "/x", "script": "mappo_drive.py",
                            "extra_args": ["--live"]})
        try:
            load_profile(path)
        except ValueError as exc:
            assert "--live" in str(exc)
            return
    raise AssertionError("a profile carrying --live loaded cleanly")


def test_a_profile_that_could_publish_a_password_is_refused():
    """``launch_prefix`` is PUBLISHED — it goes into ``get_capabilities`` and therefore to
    the browser, and the rendered command goes onto the event stream and into every
    operator's log. Scrubbing a log afterwards does not un-log it.

    Refused rather than redacted: a redaction is a pattern that has to keep matching, and a
    refusal is a state the file cannot be in. Made to fail by dropping the
    ``check_no_credential`` call from ``load_profile``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cases = [
            {"launch_prefix": ["sshpass", "-e", "ssh", "unitree@192.168.123.18"]},
            {"launch_prefix": ["ssh", "-o", "PreferredAuthentications=password", "h"]},
            {"env": {"SSHPASS": "hunter2"}},
        ]
        for extra in cases:
            data = {"workdir": "/x", "script": "s.py"}
            data.update(extra)
            try:
                load_profile(_write(tmp, data))
            except ValueError as exc:
                assert "credential" in str(exc), str(exc)
                continue
            raise AssertionError(f"{extra} loaded and would have been published")


def test_a_mistyped_profile_key_is_refused_rather_than_ignored():
    """A key that silently does not apply is a setting the operator believes is on."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"workdir": "/x", "script": "s.py", "launch_prefx": ["ssh", "h"]})
        try:
            load_profile(path)
        except ValueError as exc:
            assert "launch_prefx" in str(exc)
            return
    raise AssertionError("a mistyped profile key was accepted and did nothing")


def test_a_profile_without_a_working_directory_is_refused():
    """There is no sensible default for where somebody else's robot keeps its code."""
    with tempfile.TemporaryDirectory() as tmp:
        for missing in ("workdir", "script"):
            data = {"workdir": "/x", "script": "s.py"}
            data.pop(missing)
            try:
                load_profile(_write(tmp, data))
            except ValueError as exc:
                assert missing in str(exc)
                continue
            raise AssertionError(f"a profile with no {missing} loaded")


def test_a_loaded_profile_is_one_this_module_can_actually_launch():
    """The round trip, so the JSON keys and the dataclass cannot drift apart."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"label": "go2", "workdir": "/home/unitree/mappo-run/integration",
                            "script": "mappo_drive.py", "python": "python3",
                            "package": "/home/unitree/mappo-run/policy",
                            "launch_prefix": ["ssh", "unitree@192.168.123.18"],
                            "extra_args": ["--robot-radius", "0.25"]})
        profile = load_profile(path)
        assert profile.source_path == path
        command = launch_command(profile, build_run_argv(
            profile, seconds=20, policy_mode="raw", heading_servo="goal", live=True,
            allow_motion=True, run_id="RID"), pidfile_for(profile, "RID"))
        assert command[0] == "ssh" and "--robot-radius 0.25" in command[-1]


#: The flags ``/home/unitree/run-smoke.sh`` passes on the robot, minus the ones this driver
#: spells for itself. That wrapper is the invocation known to work on the lab Go2, so the
#: shipped example profile has to still be it.
_SMOKE_FLAGS = ("--policy-config", "--calibration", "--static-prop", "--waypoint",
                "--confidence", "--robot-radius", "--arrive", "--policy-gait-floor",
                "--no-latch-arm")


def test_the_shipped_example_profile_is_still_the_invocation_that_works():
    """The example is copied from ``run-smoke.sh``, which is the only Go2 invocation anybody
    has seen work. A profile that drifts from it is a demo that starts and does nothing.

    Made to fail by dropping ``--calibration`` from the example: without it the camera model
    falls back to a nominal 120° FOV against the measured 85.27°, every range reads 29%
    short, nothing errors, and the run simply arrives early at a goal that was never there.
    """
    example = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "run-profile.example.json")
    profile = load_profile(example)
    for flag in _SMOKE_FLAGS:
        assert flag in profile.extra_args, f"the example no longer passes {flag}"
    # ...and it must not restate what the driver decides.
    for flag in RESERVED_FLAGS:
        assert flag not in profile.extra_args, flag
    assert any(pair.startswith("PYTHONPATH=") for pair in profile.env), (
        "without PYTHONPATH the Go2's SDK segfaults; the example has stopped setting it")
    assert profile.env_setup.endswith("setup_env.sh")

    argv = build_run_argv(profile, seconds=10, policy_mode="supervised",
                          heading_servo="off", live=False, allow_motion=True, run_id="RID")
    command = launch_command(profile, argv, pidfile_for(profile, "RID"))
    assert command[0] == "ssh" and "BatchMode=yes" in command
    assert "--live" not in command[-1], "the example's default press would move the robot"
    # The lab tree predates #106, so the servo is spelled the way THAT tree spells it —
    # and it is spelled, because omitting it there selects issue #16's control law.
    assert "--no-heading-servo" in argv and "--heading-servo" not in argv, argv


def test_the_example_profile_holds_no_credential():
    """Robot passwords do not go in this repository, in an issue, or in a file beside a
    command line that would use one. ``BatchMode=yes`` makes ssh fail fast rather than sit
    on a prompt nobody can see."""
    example = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "run-profile.example.json")
    with open(example) as handle:
        text = handle.read()
    profile = load_profile(example)
    # The COMMAND, not the prose: the file's own comment is allowed to say the word
    # "password" while telling you not to put one here, and a test that forbade the word
    # would forbid the warning along with the thing it warns about.
    command = " ".join(profile.launch_prefix) + " " + " ".join(profile.extra_args)
    for smell in ("sshpass", "PreferredAuthentications=password", "PubkeyAuthentication=no",
                  "--password", "-p "):
        assert smell not in command, f"{smell!r} is in the shipped profile's command"
    assert "BatchMode=yes" in command, (
        "without BatchMode ssh sits on a password prompt nobody can see, and a run start "
        "hangs instead of failing")
    assert "sshpass" not in text


# ── what the page is told ────────────────────────────────────────────────────
def test_the_default_preview_is_never_a_live_command():
    """``command_preview`` is what a press with no arguments runs, on every driver."""
    for allow in (True, False):
        assert "--live" not in describe(REMOTE, allow_motion=allow)["command_preview"]


def test_only_a_motion_enabled_driver_advertises_an_armed_command():
    """Showing an armed command on a device that will refuse it would be the one line on
    the page telling the operator a gate is not there. Made to fail by building ``armed``
    unconditionally."""
    enabled = describe(REMOTE, allow_motion=True)
    assert "--live" in enabled["armed_command_preview"]
    assert describe(REMOTE, allow_motion=False)["armed_command_preview"] is None
    assert unsupported()["armed_command_preview"] is None


# ── the environment the SDK actually needs ───────────────────────────────────
def test_the_run_environment_is_exported_after_the_setup_file_is_sourced():
    """The order is the known-good wrapper's: source, then export, then cd.

    It matters in both directions. The source is what puts the SDK's venv on ``PATH`` and
    ``LD_LIBRARY_PATH`` in the environment — the second being what stops the SDK segfaulting
    at rc 139, because this Jetson ships two builds of ``libddsc`` and ``ldconfig`` picks the
    wrong one. The exports are what add ``PYTHONPATH``, which the source does **not** set and
    which ``unitree_sdk2py`` needs because it lives outside site-packages. Emitting the
    exports first would let the source overwrite them.

    Made to fail by emitting the exports before the source, or by dropping the ``cd``.
    """
    pair = "PYTHONPATH=/home/unitree/deps:/home/unitree/unitree_sdk2_python"
    spaced = "MAPPO_TAG=first run"
    profile = RunProfile(**{**REMOTE.__dict__, "env": (pair, spaced)})
    remote = launch_command(profile, _argv(profile), "/tmp/p.pid")[-1]
    assert f"export {pair}" in remote, remote
    # A value with a space stays one assignment rather than becoming a second command.
    assert f"export {shlex.quote(spaced)}" in remote, remote
    assert remote.index(". /home") < remote.index("export "), remote
    assert remote.index("export ") < remote.index("cd "), remote


def test_a_local_run_gets_the_same_variables_without_a_shell():
    """There is no shell in the local path, so they are handed to the spawn instead."""
    profile = RunProfile(**{**LOCAL.__dict__, "env": ("PYTHONPATH=/deps",)})
    env = run_control.local_env(profile, base={"HOME": "/home/x", "PYTHONPATH": "/old"})
    assert env["PYTHONPATH"] == "/deps" and env["HOME"] == "/home/x"
    assert run_control.local_env(LOCAL) is None, (
        "a profile with no env forced an environment onto the spawn")


def test_an_environment_name_that_is_not_one_is_refused_at_startup():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"workdir": "/x", "script": "s.py",
                            "env": {"PYTHON PATH": "/deps"}})
        try:
            load_profile(path)
        except ValueError as exc:
            assert "PYTHON PATH" in str(exc)
            return
    raise AssertionError("a shell syntax error was accepted into a profile")


def test_a_driver_with_no_run_profile_answers_in_the_same_shape():
    """"This robot cannot start a run" and "this key is missing" must not look alike: the
    second is also what a dropped field looks like."""
    absent, present = unsupported(), describe(REMOTE, allow_motion=True)
    assert absent["supported"] is False and present["supported"] is True
    assert absent["reason"], "a robot that cannot start a run does not say why"
    for key in ("max_seconds", "policy_modes", "heading_servos", "named_commit",
                "tree_note", "supported"):
        assert key in absent and key in present, key


def test_nothing_advertised_claims_to_know_which_commit_is_running():
    """The deployed tree is not a checkout: no ``.git``, so no branch and no commit. On
    2026-08-26 the lab Go2's ``~/mappo-run`` matched no single commit on ``main`` —
    ``integration/`` 43 behind, ``dashboard/`` 50, and they were different commits.

    So this reports the COMMAND and states that it cannot name the code. Made to fail by
    filling ``named_commit`` in with anything at all.
    """
    for payload in (describe(REMOTE, allow_motion=True), unsupported()):
        assert payload["named_commit"] is None
        assert "not a git checkout" in payload["tree_note"]


# ── the record the page reads ────────────────────────────────────────────────
def test_the_snapshot_reports_a_duration_and_never_a_timestamp():
    """Same rule as ``peer_pose``'s ``sample_age_s`` and the same measured reason: the lab
    Go2 has no working RTC and its clock was 56 years behind on 2026-08-26, so no instant
    taken near it means anything to a second machine. A duration does."""
    record = RunRecord(run_id="RID", argv=["a"], command=["a"], live=True,
                       policy_mode="supervised", heading_servo="off", seconds=30.0,
                       remote=True, started_at=100.0)
    snapshot = record.snapshot(now=142.5)
    assert snapshot["elapsed_s"] == 42.5
    assert "started_at" not in snapshot and "ts" not in snapshot, snapshot


def test_the_snapshot_carries_what_ran_and_not_only_what_was_asked_for():
    """On a remote run the argv and the command are different things, and reporting only
    the argv would describe a local run that never happened."""
    argv = _argv()
    command = launch_command(REMOTE, argv, pidfile_for(REMOTE, "RID"))
    record = RunRecord(run_id="RID", argv=argv, command=command, live=True,
                       policy_mode="supervised", heading_servo="off", seconds=30.0,
                       remote=True, started_at=0.0)
    snapshot = record.snapshot(now=1.0)
    assert snapshot["command"][0] == "ssh" and snapshot["argv"][0] != "ssh"
    assert snapshot["named_commit"] is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"run_control: {len(tests)}/{len(tests)} passed")
