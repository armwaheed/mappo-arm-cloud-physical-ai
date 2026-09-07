#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the combined forward+yaw arc probe. No robot, no network."""
from __future__ import annotations

import atexit
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_robotics.lite3.commissioning.arc_probe import build_parser, main, plan_radius_m
from deep_robotics.lite3.commissioning.measurement import Refusal

_PROFILE = {
    "schema": "lite3-axis-profile/v1",
    "input_deadband": {"linear_m_s": 0.05, "yaw_rad_s": 0.1},
    "allowed_gait_states": [0],
    "evidence": {"forward_positive": "m", "yaw_positive": "m", "yaw_negative": "m"},
    "measured_m_s": {"forward_positive": 0.5362},
    "measured_rad_s": {"yaw_positive": 0.8566, "yaw_negative": 0.8563},
    "primitives": {"forward_positive": 32767, "forward_negative": None,
                   "lateral_positive": None, "lateral_negative": None,
                   "yaw_positive": 16000, "yaw_negative": -16000},
}
_fd, _path = tempfile.mkstemp(suffix=".json")
os.write(_fd, json.dumps(_PROFILE).encode())
os.close(_fd)
atexit.register(lambda: os.path.exists(_path) and os.unlink(_path))

_AXIS = ["--locomotion-transport", "axis", "--axis-profile", _path]
_ROOM = ["--seconds", "1.5", "--clear-radius-metres", "2.0", "--yaw-sign", "1"]


def _refuses(argv, because: str) -> None:
    try:
        main(argv)
    except Refusal as refusal:
        assert because in str(refusal), f"refused, but not for {because!r}: {refusal}"
        return
    except SystemExit:
        return
    raise AssertionError(f"accepted an argv it should have refused: {argv}")


def test_the_predicted_radius_is_forward_over_yaw():
    """v/omega, and nothing else. This is the number the probe exists to falsify: it comes
    from two speeds measured on DIFFERENT runs, one axis at a time."""
    assert math.isclose(plan_radius_m(0.5362, 0.8566), 0.5362 / 0.8566)
    assert math.isclose(plan_radius_m(0.536, 0.0), math.inf)


def test_the_parser_offers_no_default_for_anything_physical():
    """Read straight off the parser: every flag that describes the ROOM or the robot's
    travel must default to None, so a forgotten one refuses instead of inventing a run."""
    defaults = {a.dest: a.default for a in build_parser()._actions}
    for flag in ("seconds", "clear_radius_metres", "yaw_sign"):
        assert defaults[flag] is None, f"{flag} has a default: {defaults[flag]!r}"


def test_the_room_and_the_duration_have_no_defaults():
    """Same rule as every probe here: a number nobody measured is refused, not defaulted.
    Only the operator knows the room, and --seconds is what sets how far this travels."""
    for missing in ("--seconds", "--clear-radius-metres", "--yaw-sign"):
        argv = [*_AXIS, *_ROOM]
        index = argv.index(missing)
        del argv[index:index + 2]
        _refuses(argv, "MEASURED")


def test_a_dry_run_moves_nothing():
    """No --live means no socket, no preflight and no command -- the brief prints and the
    probe exits. The exit code is the assertion: a dry run that refused would be 2."""
    assert main([*_AXIS, *_ROOM]) == 0


def test_an_arc_longer_than_the_stated_room_is_refused():
    """The check is against TRAVEL ALONG THE ARC, not the chord: a robot that arcs 3 m
    through a 2 m room has left the room, however close the endpoints are."""
    _refuses([*_AXIS, "--seconds", "30", "--clear-radius-metres", "1.0", "--yaw-sign", "1"],
             "travel")


def test_the_magnitude_transport_is_refused_by_name():
    """This probe measures a SIGN-ONLY transport's combined primitives. On the udp
    transport the commanded magnitude reaches the wire, so there is no fixed pair to
    check a prediction against."""
    _refuses(["--locomotion-transport", "udp", *_ROOM], "commanded magnitude")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"arc_probe: {len(tests)}/{len(tests)} passed")
