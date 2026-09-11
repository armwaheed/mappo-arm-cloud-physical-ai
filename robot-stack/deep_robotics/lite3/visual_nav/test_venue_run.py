#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the dashboard-to-venue-mission bridge.

What matters here is that the flags a dashboard spells arrive at ``mappo_drive.py``
unchanged, and that nothing in this file can add motion that was not already authorised.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import look_behind as _look_behind
from venue_run import DEFAULTS, build_command

#: What ``run_control.build_run_argv`` produces, in its own order.
_FROM_THE_DASHBOARD = [
    "--package", "/home/user/stage/policy",
    "--policy-scale", "4.0",
    "--camera-rectify", "/home/user/stage/calibration/rectify.json",
    "--policy-mode", "supervised",
    "--heading-servo", "goal",
    "--max-seconds", "90",
    "--telemetry", "/home/user/stage/evidence/run.jsonl",
]


def _after_separator(command: list) -> list:
    return command[command.index("--") + 1:]


def test_both_layers_are_unbuffered_so_the_dashboard_sees_the_run():
    """run_control streams the script's stdout; a buffered layer shows a blank panel.

    The inner one matters one layer down for the same reason: mission.py reads the drive's
    stdout line by line to decide when to speak.
    """
    command = build_command(_FROM_THE_DASHBOARD, env={}, python="/usr/bin/python3")
    assert command[:3] == ["/usr/bin/python3", "-u", "mission.py"]
    assert _after_separator(command)[:3] == ["/usr/bin/python3", "-u", "mappo_drive.py"]


def test_every_dashboard_flag_reaches_the_drive_in_the_order_it_was_given():
    """Forwarded verbatim. Reordering or dropping one silently changes the run."""
    command = build_command(_FROM_THE_DASHBOARD, env={}, python="/usr/bin/python3")
    tail = _after_separator(command)
    assert tail[:3] == ["/usr/bin/python3", "-u", "mappo_drive.py"]
    assert tail[3:] == _FROM_THE_DASHBOARD


def test_the_supervisor_wraps_the_drive_rather_than_replacing_it():
    """mission.py before the separator, mappo_drive.py after: that is what supervises."""
    command = build_command(_FROM_THE_DASHBOARD, env={}, python="/usr/bin/python3")
    assert command[2] == "mission.py"
    assert command.index("mission.py") < command.index("--") < command.index("mappo_drive.py")


def test_it_never_adds_a_motion_flag_of_its_own():
    """--live comes from build_run_argv, which appends it only with --allow-motion.

    A shim that added it would be a second motion gate nobody would think to look for.
    """
    command = build_command(_FROM_THE_DASHBOARD, env={}, python="/usr/bin/python3")
    assert "--live" not in command


def test_a_live_run_is_passed_through_untouched():
    command = build_command([*_FROM_THE_DASHBOARD, "--live"], env={},
                            python="/usr/bin/python3")
    assert _after_separator(command)[-1] == "--live"


def test_an_unset_voice_directory_is_silent_on_purpose_not_by_accident():
    """A wrong --voice-dir is reported by Voice only once a cue is due, mid-run."""
    command = build_command(_FROM_THE_DASHBOARD, env={}, python="/usr/bin/python3")
    assert "--no-voice" in command
    assert "--voice-dir" not in command


def test_the_voice_directory_and_device_come_from_the_deployment():
    command = build_command(_FROM_THE_DASHBOARD,
                            env={"MAPPO_VOICE_DIR": "/home/user/stage/voice",
                                 "MAPPO_VOICE_DEVICE": "plughw:1,0"},
                            python="/usr/bin/python3")
    assert command[command.index("--voice-dir") + 1] == "/home/user/stage/voice"
    assert command[command.index("--voice-device") + 1] == "plughw:1,0"
    assert "--no-voice" not in command


def test_the_venue_settings_are_the_defaults_and_a_deployment_can_override_them():
    default = build_command([], env={}, python="/usr/bin/python3")
    assert default[default.index("--max-attempts") + 1] == DEFAULTS["MAPPO_MISSION_ATTEMPTS"]
    tuned = build_command([], env={"MAPPO_MISSION_ATTEMPTS": "3",
                                   "MAPPO_MISSION_COOLDOWN": "5"},
                          python="/usr/bin/python3")
    assert tuned[tuned.index("--max-attempts") + 1] == "3"
    assert tuned[tuned.index("--cooldown") + 1] == "5"


def test_the_flourish_is_off_unless_the_deployment_asks_for_it():
    """It moves the robot after the run everybody stopped watching. Off is the default."""
    cmd = build_command(["--goal", "x"], env={})
    assert "--flourish" not in cmd, cmd


def test_a_partial_flourish_answer_is_treated_as_no_answer():
    """There is no safe default lane width for a robot about to sweep its own footprint in
    a room this file cannot see, so a half-configured gesture must not fire."""
    env = {"MAPPO_FLOURISH": "1", "MAPPO_ROBOT_ID": "LITE3-A"}
    cmd = build_command(["--goal", "x"], env=env)
    assert "--flourish" not in cmd, cmd


def test_a_fully_answered_flourish_reaches_the_supervisor():
    env = {"MAPPO_FLOURISH": "1", "MAPPO_FLOURISH_LANE_WIDTH": "2.0",
           "MAPPO_ROBOT_ID": "LITE3-A", "MAPPO_FIRMWARE": "V1.0.8",
           "MAPPO_PAYLOAD": "none"}
    cmd = build_command(["--goal", "x"], env=env)
    assert "--flourish" in cmd, cmd
    assert cmd[cmd.index("--flourish-lane-width") + 1] == "2.0"
    assert cmd[cmd.index("--robot-id") + 1] == "LITE3-A"
    # ...and it goes to mission.py, NOT to the drive after the separator.
    assert cmd.index("--flourish") < cmd.index("--"), "the supervisor fires it, not the drive"


def test_the_off_switch_accepts_the_shapes_a_run_profile_actually_writes():
    for value in ("0", "", "false", "False"):
        cmd = build_command(
            ["--goal", "x"],
            env={"MAPPO_FLOURISH": value, "MAPPO_FLOURISH_LANE_WIDTH": "2.0",
                 "MAPPO_ROBOT_ID": "A", "MAPPO_FIRMWARE": "B", "MAPPO_PAYLOAD": "C"})
        assert "--flourish" not in cmd, (value, cmd)


_FLOURISH = {"MAPPO_FLOURISH": "1", "MAPPO_FLOURISH_LANE_WIDTH": "2.0",
             "MAPPO_ROBOT_ID": "LITE3-A", "MAPPO_FIRMWARE": "V1.0.8",
             "MAPPO_PAYLOAD": "none"}

#: The kind is taken from `look_behind.BACKWARD_KINDS` rather than written out, because
#: `test_look_behind.py`'s directory sweep forbids any file here from carrying a
#: travelling kind as a string constant -- and it is right to: a literal in a file on this
#: path is the only thing that could reach a command line by accident. Deriving it also
#: means this fixture tests whatever kinds are actually offered rather than one that was
#: offered when it was written.
_FLIP_KIND = _look_behind.BACKWARD_KINDS[0]

_FLIP = {"MAPPO_ARRIVAL_ACTION": _FLIP_KIND,
         "MAPPO_FLIP_REAR_CLEARANCE_M": "2.0",
         "MAPPO_FLIP_BATTERY_FLOOR_PCT": "60", "MAPPO_FLIP_HOLD_SECONDS": "7.0"}


def test_the_flip_is_off_when_nothing_asks_for_it():
    """The default an operator gets by not ticking the box, and by far the most important
    assertion in this file: the action travels ~1.5 m BACKWARD into the one direction this
    robot has no sensor pointing, and the person holding the abort stands there."""
    cmd = build_command(["--goal", "x"], env=dict(_FLOURISH))
    assert "--flip-on-arrival" not in cmd, cmd
    for flag in ("--flip-kind", "--flip-rear-clearance-metres",
                 "--flip-battery-floor-pct", "--flip-hold-seconds"):
        assert flag not in cmd, (flag, cmd)


def test_the_flip_off_switch_accepts_the_shapes_a_run_profile_actually_writes():
    """`MAPPO_ARRIVAL_ACTION=spin` is what the dashboard sends when the operator picks the
    360 degree dance. If `spin` -- or an unset value -- were read as a vendor action, the
    default arrival gesture would fire an uninterruptible opcode."""
    for value in ("spin", ""):
        env = {**_FLOURISH, **_FLIP, "MAPPO_ARRIVAL_ACTION": value}
        cmd = build_command(["--goal", "x"], env=env)
        assert "--flip-on-arrival" not in cmd, (value, cmd)


def test_a_partial_flip_answer_is_treated_as_no_answer():
    """The four settings are measurements of the ROOM -- rear clearance especially, which
    `look_behind` cannot replace because its detector finds VOC classes and cannot see a
    wall, a step or a stage edge. A half-configured flip must not fire."""
    for missing in _FLIP:
        if missing == "MAPPO_ARRIVAL_ACTION":
            continue
        env = {**_FLOURISH, **_FLIP}
        del env[missing]
        cmd = build_command(["--goal", "x"], env=env)
        assert "--flip-on-arrival" not in cmd, (missing, cmd)


def test_the_flip_needs_the_flourish_that_fires_it():
    """`mission.py` refuses `--flip-on-arrival` without `--flourish`. Building the flags
    anyway would turn an operator's tick into a refusal at the far end -- after the robot
    has been committed -- so it is caught here and degrades to no flip instead."""
    cmd = build_command(["--goal", "x"], env=dict(_FLIP))
    assert "--flip-on-arrival" not in cmd, cmd
    assert "--flourish" not in cmd, cmd


def test_a_fully_answered_flip_reaches_the_supervisor_with_every_measurement():
    env = {**_FLOURISH, **_FLIP}
    cmd = build_command(["--goal", "x"], env=env)
    assert "--flip-on-arrival" in cmd, cmd
    assert cmd[cmd.index("--flip-kind") + 1] == _FLIP_KIND
    assert cmd[cmd.index("--flip-rear-clearance-metres") + 1] == "2.0"
    assert cmd[cmd.index("--flip-battery-floor-pct") + 1] == "60"
    assert cmd[cmd.index("--flip-hold-seconds") + 1] == "7.0"
    # To mission.py, NOT to the drive after the separator. `venue_run` forwards its argv
    # verbatim to `mappo_drive.py`, which would exit 2 on a flag it does not know.
    assert cmd.index("--flip-on-arrival") < cmd.index("--"), \
        "the supervisor fires the flip, not the drive"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_venue_run: {len(tests)}/{len(tests)} passed")
