#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the pre-flight the two walking probes share.

This is the gate that stands between an operator with a laptop and a robot with legs, so
every branch of it is tested by making it fire. Two of them are less obvious than they
look:

* **a stale state stream is refused even though the transport would still accept the
  command.** The transport's own gate asks "may this one datagram go out"; this one asks
  "is a multi-minute measurement worth beginning", and it is deliberately tighter.
* **a gate that cannot be applied is reported, not passed.** ``error_state()`` arrives on
  the UDP transport with PR #74. On ``main`` it is absent, and the pre-flight says which
  check it could not run rather than printing a pass it never performed.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning import robot_link
from deep_robotics.lite3.commissioning.measurement import Refusal
from deep_robotics.lite3.locomotion.lite3_axis_locomotion import Lite3AxisLocomotion
from deep_robotics.lite3.locomotion.lite3_udp_locomotion import Lite3UdpLocomotion


class _Implementation:
    """A transport with only the accessors a given build actually has.

    ``omit`` shadows a method with ``None`` on the instance, which is what an older
    build looks like to ``robot_link``: it asks with ``getattr(..., name, None)`` and
    treats a missing accessor as "this gate cannot be applied here".
    """

    def __init__(self, *, age=0.05, battery=80.0, error_state=0, mode=(6, 0, 0, 0),
                 omit=()):
        self._age, self._battery = age, battery
        self._error_state, self._mode = error_state, mode
        for name in omit:
            setattr(self, name, None)

    def state_age(self):
        return self._age

    def battery_level(self):
        return self._battery

    def error_state(self):
        return self._error_state

    def mode(self):
        return self._mode


def _args(**overrides):
    values = {"operator_ready": True, "battery_abort": 20.0}
    values.update(overrides)
    return argparse.Namespace(**values)


def _link(**kwargs):
    return robot_link.Link(locomotion=object(), implementation=_Implementation(**kwargs))


def _refusal(link, args):
    try:
        robot_link.preflight(link, args, printer=lambda _line: None)
    except Refusal as refusal:
        return str(refusal)
    return None


# ── the authority gate ──────────────────────────────────────────────────────────────────
def test_without_operator_ready_nothing_is_commanded():
    message = _refusal(_link(), _args(operator_ready=False))
    assert message and "--operator-ready" in message
    assert "indistinguishable" in message, "the operator needs to know WHY, not just that"


# ── the link gate ───────────────────────────────────────────────────────────────────────
def test_a_silent_link_is_refused_before_the_first_command():
    message = _refusal(_link(age=None), _args())
    assert message and "silent" in message and "network.toml" in message


def test_a_stale_link_is_refused_even_though_the_transport_would_still_send():
    stale = robot_link.PREFLIGHT_STATE_MAX_AGE_S * 2
    message = _refusal(_link(age=stale), _args())
    assert message and "stale" in message


def test_a_fresh_link_passes_the_staleness_gate():
    assert _refusal(_link(age=robot_link.PREFLIGHT_STATE_MAX_AGE_S / 2), _args()) is None


def test_the_staleness_threshold_is_the_thing_that_decides():
    """Mutation guard: widening PREFLIGHT_STATE_MAX_AGE_S must change this outcome."""
    just_inside = robot_link.PREFLIGHT_STATE_MAX_AGE_S * 0.99
    just_outside = robot_link.PREFLIGHT_STATE_MAX_AGE_S * 1.01
    assert _refusal(_link(age=just_inside), _args()) is None
    assert _refusal(_link(age=just_outside), _args()) is not None


# ── the health gates ────────────────────────────────────────────────────────────────────
def test_a_flat_battery_is_refused_because_it_would_be_measuring_the_battery():
    message = _refusal(_link(battery=15.0), _args())
    assert message and "measurement of the battery" in message


def test_the_battery_limit_boundary_refuses_at_the_limit_not_only_below_it():
    assert _refusal(_link(battery=20.0), _args(battery_abort=20.0)) is not None
    assert _refusal(_link(battery=20.1), _args(battery_abort=20.0)) is None


def test_a_missing_battery_reading_means_the_link_is_wrong_not_the_field_absent():
    message = _refusal(_link(battery=None), _args())
    assert message and "the link is wrong" in message


def test_a_robot_already_reporting_a_fault_is_refused():
    message = _refusal(_link(error_state=3), _args())
    assert message and "error_state=3" in message


def test_a_nonsensical_battery_abort_limit_is_refused():
    for value in (0.0, -5.0, float("nan")):
        assert _refusal(_link(), _args(battery_abort=value)) is not None, value


# ── gates this build cannot apply ───────────────────────────────────────────────────────
def test_a_build_without_error_state_says_which_check_it_could_not_run():
    lines = []
    health = robot_link.preflight(_link(omit=("error_state",)), _args(),
                                  printer=lines.append)
    assert health["error_state"] is None
    assert any("could not be applied" in line for line in lines), lines


def test_the_preflight_returns_what_it_checked_so_it_lands_in_the_record():
    health = robot_link.preflight(_link(battery=77.0), _args(),
                                  printer=lambda _line: None)
    assert health["battery_pct"] == 77.0
    assert health["mode"] == [6, 0, 0, 0]
    assert health["state_age_s"] is not None


# ── argument surface ────────────────────────────────────────────────────────────────────
def test_the_issue_13_context_flags_are_all_required():
    parser = argparse.ArgumentParser()
    robot_link.add_context_arguments(parser)
    for missing in (["--firmware", "V1", "--payload", "none"],
                    ["--robot-id", "A", "--payload", "none"],
                    ["--robot-id", "A", "--firmware", "V1"]):
        try:
            # argparse writes its own usage to stderr on the way to SystemExit; that is
            # the behaviour under test, not output the suite should print.
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(missing)
        except SystemExit:
            continue
        raise AssertionError(f"{missing} should not have parsed")


def test_a_non_moving_probe_is_offered_no_authority_to_move():
    parser = argparse.ArgumentParser()
    robot_link.add_link_arguments(parser, moving=False)
    flags = {option for action in parser._actions for option in action.option_strings}
    assert "--live" not in flags and "--operator-ready" not in flags


def test_a_moving_probe_is_offered_the_authority_flags():
    parser = argparse.ArgumentParser()
    robot_link.add_link_arguments(parser, moving=True)
    flags = {option for action in parser._actions for option in action.option_strings}
    assert {"--live", "--operator-ready", "--battery-abort"} <= flags


# ── which interface the probe actually ends up commanding through ───────────────────────
def _transport_args(**overrides):
    values = {"locomotion_transport": "udp", "axis_profile": None,
              "axis_local_port": 20001, "motion_host": "192.168.1.120",
              "command_port": 43893, "state_port": 43897}
    values.update(overrides)
    return argparse.Namespace(**values)


def _profile_file(directory) -> str:
    path = Path(directory) / "profile.json"
    path.write_text(json.dumps({
        "schema": "lite3-axis-profile/v1",
        "input_deadband": {"linear_m_s": 0.05, "yaw_rad_s": 0.1},
        "allowed_gait_states": [0],
        "evidence": {"forward_positive": "vendor V1.0.8 reference control script"},
        "measured_m_s": {}, "measured_rad_s": {},
        "primitives": {"forward_positive": 32767, "forward_negative": None,
                       "lateral_positive": None, "lateral_negative": None,
                       "yaw_positive": None, "yaw_negative": None},
    }), encoding="utf-8")
    return str(path)


def test_the_selected_transport_is_the_one_that_gets_built():
    """The bug this whole argument exists to prevent.

    A ``connect`` that ignored the choice would put every walking measurement back on the
    legacy velocity interface -- silently, and with a working-looking axis flag on the
    command line to say otherwise.
    """
    with tempfile.TemporaryDirectory() as directory:
        axis = robot_link._transport_factory(
            _transport_args(locomotion_transport="axis",
                            axis_profile=_profile_file(directory)))(
            cmd_vel_topic=None, odom_topic=None, stamped=None, node_name=None)
    assert isinstance(axis, Lite3AxisLocomotion)

    udp = robot_link._transport_factory(_transport_args())(
        cmd_vel_topic=None, odom_topic=None, stamped=None, node_name=None)
    assert isinstance(udp, Lite3UdpLocomotion)
    assert not isinstance(udp, Lite3AxisLocomotion)


def test_the_axis_transport_carries_the_profile_it_was_given():
    with tempfile.TemporaryDirectory() as directory:
        loaded = robot_link.load_axis_profile(
            _transport_args(locomotion_transport="axis",
                            axis_profile=_profile_file(directory)))
    assert loaded.forward_positive == 32767


def test_a_profile_handed_to_a_transport_that_ignores_it_is_refused():
    """It looks exactly like a profile that took effect, and it never reaches the wire."""
    with tempfile.TemporaryDirectory() as directory:
        try:
            robot_link.load_axis_profile(
                _transport_args(axis_profile=_profile_file(directory)))
        except Refusal as refusal:
            assert "does not read one" in str(refusal)
        else:
            raise AssertionError("expected a Refusal")


def test_the_axis_transport_without_a_profile_is_refused():
    try:
        robot_link.load_axis_profile(_transport_args(locomotion_transport="axis"))
    except Refusal as refusal:
        assert "requires --axis-profile" in str(refusal)
    else:
        raise AssertionError("expected a Refusal")


def test_an_unknown_transport_is_named_rather_than_keyerrored():
    try:
        robot_link.selected_transport(_transport_args(locomotion_transport="ros2"))
    except Refusal as refusal:
        assert "unknown --locomotion-transport" in str(refusal)
    else:
        raise AssertionError("expected a Refusal")


def test_only_one_transport_claims_to_have_walked_a_venture():
    """If a second one ever does, the ladder probes stop being unrunnable and this is why."""
    walked = {name for name, row in robot_link.TRANSPORTS.items() if row.walked}
    assert walked == {"axis"}
    assert not robot_link.TRANSPORTS["axis"].preserves_magnitude
    assert robot_link.TRANSPORTS["udp"].preserves_magnitude


# ── the invariant that has to survive a copy of this directory ──────────────────────────
#
# WHY THIS IS OVER THE SOURCE AND NOT OVER A LIST OF MODULE NAMES.
#
# The failure being pinned is not a bad edit, it is a PARTIAL COPY of this directory. This
# harness shipped once without any transport argument at all, hard-wired to the legacy
# velocity interface -- the one this robot does not move on -- and the fix that gave it
# ``--locomotion-transport`` and these three guards landed as a separate change. Anyone who
# copies this directory into another repository at the earlier of those two points gets a
# harness that runs, refuses nothing, and reports the bottom rung of its own ladder as this
# robot's gait floor.
#
# A hand-kept list of module names is exactly what such a copy silently shortens, so the
# set of modules under audit is derived from what each module ASKS FOR: taking
# ``moving=True`` is a module saying it intends to command a velocity, and that is the
# thing that has to be paired with asking which transport the velocity is going out on.
#
# THIS COPY IS THE ONE THAT TRAVELS, and it is not the one that protects a consumer. It is
# here so that a repository which takes this directory gets the invariant along with the
# code -- but the failure being pinned is a copy that REPLACES the directory, and such a
# copy arrives carrying its own version of this file. Only an audit OUTSIDE the directory
# survives that, and it has to be written in the repository being protected.
#
# The internal per-robot Lite3 repository is where that matters first: it held only the two
# receive-only scripts, so a sync of this directory at its pre-transport commit was a live
# trap there. It is being given a `test_commissioning_transport_guards.py` one directory up
# from its own `commissioning/`, and outside it. Do not read this comment as a statement
# that the check is already in place there; go and look.
#
# One thing that audit found and this one cannot: a population derived from a single call
# shape can be emptied by a rename. Pass `moving` positionally in all three probes here and
# every test in this file stays green while the audit checks nothing. The pin below is what
# makes that loud, and it is the reason the audit above it is not vacuous.

#: The two guards that answer "is the measurement this module makes even defined on the
#: selected transport". A moving module must call one of them.
DEFINEDNESS_GUARDS = ("require_magnitude_transport", "require_sign_only_transport")

#: The guard that answers "has any Venture ever walked on the selected transport". Every
#: moving module must call it: on a transport that does not actuate, every segment reports
#: 0.000 m/s, which reads exactly like a floor above the whole ladder.
WALKED_GUARD = "require_walked_transport"


def called_function_names(tree) -> set:
    """Every name this module calls, whether bare or through a module attribute."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute):
                names.add(function.attr)
            elif isinstance(function, ast.Name):
                names.add(function.id)
    return names


def takes_moving_authority(tree) -> bool:
    """True when the module calls ``add_link_arguments(..., moving=True)``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else \
            getattr(function, "id", None)
        if name != "add_link_arguments":
            continue
        for keyword in node.keywords:
            if keyword.arg == "moving" and isinstance(keyword.value, ast.Constant) \
                    and keyword.value.value is True:
                return True
    return False


def moving_modules(directory) -> list:
    """Non-test modules in the directory that take the authority to move, by name.

    Split out of :func:`audit_transport_guards` so the population can be asserted directly.
    The audit returns "no findings" both when every moving module is guarded and when there
    are no moving modules at all, and only one of those is this directory.
    """
    found = []
    for path in sorted(Path(directory).glob("*.py")):
        if path.name.startswith("test_"):
            continue
        if takes_moving_authority(ast.parse(path.read_text(encoding="utf-8"))):
            found.append(path.name)
    return found


def audit_transport_guards(directory) -> list:
    """Modules that take the authority to move without asking which transport they are on.

    Returns a list of human-readable findings, empty when the directory is whole. Reading
    the source rather than importing it is deliberate: a directory that has been partially
    copied may not import at all, and a guard that cannot run on the broken case is not a
    guard.
    """
    findings = []
    for path in sorted(Path(directory).glob("*.py")):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not takes_moving_authority(tree):
            continue
        called = called_function_names(tree)
        missing = []
        if WALKED_GUARD not in called:
            missing.append(WALKED_GUARD)
        if not any(guard in called for guard in DEFINEDNESS_GUARDS):
            missing.append(" or ".join(DEFINEDNESS_GUARDS))
        if missing:
            findings.append(
                f"{path.name} takes moving authority (add_link_arguments(moving=True)) "
                f"but never calls " + ", ".join(missing))
    return findings


def test_every_probe_that_may_move_the_robot_asks_which_transport_it_is_on():
    findings = audit_transport_guards(_HERE)
    assert findings == [], (
        "a commissioning probe can command a velocity without consulting "
        "robot_link.TRANSPORTS. On the axis transport that probe does not crash: the "
        "mapping is sign-only, so every rung of a ladder fires the same primitive, walks "
        "at the same speed, passes every anchor and drift control, and reports the bottom "
        "rung as this robot's gait floor. Findings: " + "; ".join(findings))


def test_the_audit_has_a_population_to_check():
    """A single-signal population can be emptied by a rename, and then this file is inert.

    ``takes_moving_authority`` matches one call shape: ``add_link_arguments`` with a
    ``moving=True`` KEYWORD. Pass it positionally in all three walking probes -- a
    refactor nobody would flag in review -- and the population is empty, the audit above
    reports nothing, and every test in this file still passes while checking no module at
    all. Verified by mutation.

    The three names are spelled out rather than counted, because the failure this pins is a
    partial copy and a partial copy shortens a count as easily as a list. A fourth walking
    probe is a deliberate addition to this list; three that became two is the bug.
    """
    assert moving_modules(_HERE) == ["actuator_gain_probe.py", "axis_primitive_probe.py",
                                     "gait_floor_probe.py"], moving_modules(_HERE)


def test_the_audit_fails_on_a_probe_that_skips_the_guards():
    """The audit above passes on a whole directory, so prove here that it can fail.

    Without this, the day the audit stops finding anything -- because a rename moved
    ``add_link_arguments``, or because someone shortened its argument -- looks identical
    to the day the directory is correct.
    """
    unguarded = ("import robot_link\n"
                 "def build_parser(parser):\n"
                 "    robot_link.add_link_arguments(parser, moving=True)\n")
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "new_walking_probe.py").write_text(unguarded, encoding="utf-8")
        findings = audit_transport_guards(directory)
    assert len(findings) == 1, findings
    assert "new_walking_probe.py" in findings[0]
    assert "require_walked_transport" in findings[0]
    assert "require_magnitude_transport" in findings[0]


def test_a_probe_that_asks_for_no_authority_to_move_is_not_audited():
    """``moving=False`` is the whole population this check must not fire on."""
    passive = ("import robot_link\n"
               "def build_parser(parser):\n"
               "    robot_link.add_link_arguments(parser, moving=False)\n")
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "passive_probe.py").write_text(passive, encoding="utf-8")
        assert audit_transport_guards(directory) == []


def test_the_three_guards_the_audit_names_all_exist():
    """The audit is over names, so a rename must break it loudly rather than silently.

    A guard renamed in ``robot_link`` and not here leaves ``audit_transport_guards``
    looking for a call nobody makes -- which reports every moving probe as broken, and is
    the safe direction. A guard DELETED from ``robot_link`` while the probes keep calling
    it fails at import instead. This pins the pairing either way.
    """
    for name in (*DEFINEDNESS_GUARDS, WALKED_GUARD):
        assert callable(getattr(robot_link, name)), name


def test_the_preflight_reports_which_transport_it_is_about_to_command_through():
    buffer = io.StringIO()
    robot_link.preflight(_link(), _args(locomotion_transport="axis"),
                         printer=lambda line: buffer.write(line + "\n"))
    assert "transport axis" in buffer.getvalue()
    assert "has walked a Venture" in buffer.getvalue()

    buffer = io.StringIO()
    robot_link.preflight(_link(), _args(), printer=lambda line: buffer.write(line + "\n"))
    assert "NO Venture has been seen to walk on this transport" in buffer.getvalue()


#: How the RUNBOOK's safety section renders a count, in both languages, INCLUDING the bold
#: markers. The emphasis is what makes these exact rather than incidental: the section also
#: contains "All three refuse" and "三者都必须", so a bare "three"/"三" would be satisfied by
#: prose that had nothing to do with the count. A rendering table, not a population -- the
#: population is derived from the source below.
_COUNT_WORDS = {1: ("**One**", "**一个**"), 2: ("**Two**", "**两个**"),
                3: ("**Three**", "**三个**"), 4: ("**Four**", "**四个**"),
                5: ("**Five**", "**五个**"), 6: ("**Six**", "**六个**"),
                7: ("**Seven**", "**七个**"), 8: ("**Eight**", "**八个**")}


def runbook_safety_section() -> str:
    """The text of RUNBOOK.md section 1, which is where an operator reads what moves."""
    text = (_HERE / "RUNBOOK.md").read_text(encoding="utf-8")
    start = text.index("## 1. Safety")
    return text[start:text.index("\n## 2.", start)]


def walking_modules(directory) -> list:
    """Filenames in ``directory`` that take the authority to move the robot."""
    return sorted(path.name for path in sorted(Path(directory).glob("*.py"))
                  if not path.name.startswith("test_")
                  and takes_moving_authority(ast.parse(path.read_text(encoding="utf-8"))))


def runbook_omissions(section: str, walkers) -> list:
    """Walking tools the operator's safety section does not name, plus a count mismatch."""
    findings = [f"{name} is not named" for name in walkers if name not in section]
    english, chinese = _COUNT_WORDS[len(walkers)]
    if english not in section or chinese not in section:
        findings.append(f"the count does not read {english!r} / {chinese!r}")
    return findings


def test_the_runbook_safety_section_names_every_tool_that_walks_the_robot():
    """The claim an operator acts on, checked against the directory rather than remembered.

    Section 1 said "Two of the six tools in this directory walk the robot" and named
    ``gait_floor_probe.py`` and ``actuator_gain_probe.py``. True when the harness was first
    written. ``axis_primitive_probe.py`` arrived with the transport fix and is the THIRD --
    and on these two Ventures it is the one an operator actually runs, because it is the
    walking probe for the only transport either robot has been seen to move on. The text is
    correct now; nothing made it stay correct, which is why it went wrong once already.

    Derived here instead: ``takes_moving_authority`` reads the source, so a fourth walking
    probe fails this test until the section an operator reads says so. The count has to
    match in BOTH halves of the bilingual section -- half a correction is how a translated
    safety document ends up disagreeing with itself.
    """
    walkers = walking_modules(_HERE)
    assert walkers, "no module in this directory takes moving authority -- check the audit"
    findings = runbook_omissions(runbook_safety_section(), walkers)
    assert findings == [], (
        "RUNBOOK.md section 1 is where an operator is told which tools move the robot, and "
        + "; ".join(findings) + ". A tool that walks and is not in that list is one nobody "
        "was told to hold the stop for.")


def test_the_runbook_check_fails_on_the_section_as_it_actually_read():
    """The audit above passes today, so prove the mechanism can fail -- on the real text.

    This is the section as it stood before the transport fix: the third walking tool
    missing and the count still reading Two/两个. Reconstructed from the live section rather
    than from a fixture, so a restructuring of RUNBOOK.md that made section 1 unfindable
    breaks this test instead of quietly making the one above vacuous.
    """
    walkers = walking_modules(_HERE)
    as_it_read = (runbook_safety_section()
                  .replace("`axis_primitive_probe.py`", "")
                  .replace("**Three**", "**Two**").replace("**三个**", "**两个**"))
    findings = runbook_omissions(as_it_read, walkers)
    assert len(findings) == 2, findings
    assert "axis_primitive_probe.py is not named" in findings
    assert findings[1] == "the count does not read '**Three**' / '**三个**'"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"robot_link: {len(tests)}/{len(tests)} passed")
