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
import contextlib
import io
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning import robot_link
from deep_robotics.lite3.commissioning.measurement import Refusal


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


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"robot_link: {len(tests)}/{len(tests)} passed")
