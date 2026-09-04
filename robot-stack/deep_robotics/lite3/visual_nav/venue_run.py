#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Start the venue mission from a dashboard ``start_run``, not from a terminal.

``run_control.build_run_argv`` spells one command:

    <python> -u <script> --package <pkg> <profile extra_args> --policy-mode ...
             --heading-servo ... --max-seconds ... --telemetry ... [--live]

and every one of those flags belongs to ``mappo_drive.py``. The demo, though, is
``mission.py`` **wrapping** ``mappo_drive.py``: the supervisor is what speaks to the room,
waits for a person to move, and tries again. Naming ``mission.py`` as the profile's script
does not work, because ``--package`` would land on the supervisor, which has never heard
of it.

So this is the script the profile names. It forwards its whole argv, untouched and in
order, to ``mappo_drive.py``, and wraps that in the ``mission.py`` invocation
``run-venue-demo.sh`` uses. One demo, two ways to start it, no second copy of the flags.

⚠️ **It adds no motion flag of its own.** ``--live`` is present only if it was in the argv
this was handed, which ``build_run_argv`` appends only when the driver was started with
``--allow-motion``. There is no path here from a dashboard button to motion that the
driver did not already authorise.

The deployment's own paths arrive as environment variables rather than being written here,
because they are properties of a robot's staging directory and not of this repository.
"""

from __future__ import annotations

import os
import sys

#: The venue settings, matching ``run-venue-demo.sh``. Overridable per deployment.
DEFAULTS = {
    "MAPPO_MISSION_PATIENCE": "4",
    "MAPPO_MISSION_COOLDOWN": "25",
    "MAPPO_MISSION_ATTEMPTS": "8",
    "MAPPO_MISSION_TOTAL": "900",
}

#: The flourish is OFF unless the deployment asks for it, and it is asked for HERE rather
#: than in the drive flags because it is the supervisor that fires it, after the drive has
#: exited. All five must be set: the gesture turns the robot in place, and issue #13's
#: context is required beside anything that moves a leg. A partial answer is treated as no
#: answer -- `mission.py` prints which one is missing and skips the gesture rather than
#: turning a robot on a guess about the room.
FLOURISH_SETTINGS = ("MAPPO_FLOURISH_LANE_WIDTH", "MAPPO_ROBOT_ID",
                     "MAPPO_FIRMWARE", "MAPPO_PAYLOAD")


def build_command(drive_args, env=None, python: str | None = None) -> list:
    """The ``mission.py`` command line that runs ``drive_args`` under supervision.

    ``drive_args`` is forwarded verbatim: this does not parse, reorder or drop any of it.
    A flag this file does not recognise is a flag ``mappo_drive.py`` may have gained since,
    and swallowing it here would be a silent downgrade of the run somebody asked for.
    """
    env = os.environ if env is None else env
    python = python or sys.executable
    settings = {name: env.get(name, default) for name, default in DEFAULTS.items()}

    voice_dir = env.get("MAPPO_VOICE_DIR", "")
    if voice_dir:
        voice = ["--voice-dir", voice_dir,
                 "--voice-device", env.get("MAPPO_VOICE_DEVICE", "pulse")]
    else:
        # Silent, and SAYING so. A demo whose whole point is that the robot asks the room
        # to move should not start mute because a path was unset, and a wrong --voice-dir
        # would be reported by Voice as "not a directory" only once a cue was due.
        voice = ["--no-voice"]

    # ``-u`` on BOTH, and it is not a nicety. ``run_control`` launches the profile's script
    # with ``-u`` and then streams its stdout to the dashboard; exec'ing a buffered
    # ``mission.py`` from here throws that away, and the operator watches a blank panel
    # while the robot walks. The inner one matters for the same reason one layer down:
    # ``mission.py`` reads the drive's stdout line by line to decide when to speak.
    # Enabled only when the deployment answered ALL of it. `env.get` and not `settings`,
    # because these have no defaults on purpose: there is no safe default lane width for a
    # robot that is about to sweep its own footprint in a room this file cannot see.
    flourish: list = []
    if env.get("MAPPO_FLOURISH", "").strip() not in ("", "0", "false", "False"):
        answered = {name: env.get(name, "").strip() for name in FLOURISH_SETTINGS}
        missing = [name for name, value in answered.items() if not value]
        if missing:
            print(f"[venue-run] MAPPO_FLOURISH is set but {', '.join(missing)} "
                  f"{'is' if len(missing) == 1 else 'are'} not; the run will NOT gesture.",
                  flush=True)
        else:
            flourish = ["--flourish",
                        "--flourish-lane-width", answered["MAPPO_FLOURISH_LANE_WIDTH"],
                        "--robot-id", answered["MAPPO_ROBOT_ID"],
                        "--firmware", answered["MAPPO_FIRMWARE"],
                        "--payload", answered["MAPPO_PAYLOAD"]]

    return [python, "-u", "mission.py", *voice, *flourish,
            "--patience", settings["MAPPO_MISSION_PATIENCE"],
            "--cooldown", settings["MAPPO_MISSION_COOLDOWN"],
            "--max-attempts", settings["MAPPO_MISSION_ATTEMPTS"],
            "--max-total-seconds", settings["MAPPO_MISSION_TOTAL"],
            "--", python, "-u", "mappo_drive.py", *drive_args]


def main(argv: list | None = None) -> int:
    drive_args = list(sys.argv[1:] if argv is None else argv)
    command = build_command(drive_args)
    if not os.environ.get("MAPPO_VOICE_DIR"):
        print("[venue-run] MAPPO_VOICE_DIR is unset: this run will be SILENT. Set it in "
              "the run profile's env to the robot's voice directory.", flush=True)
    print(f"[venue-run] {' '.join(command)}", flush=True)
    # exec, not spawn: run_control records ONE pid and stops it with SIGTERM. A shim that
    # stayed alive as a parent would take that signal itself and leave mission.py, and the
    # drive under it, running.
    os.execv(command[0], command)
    return 0  # unreachable; execv does not return


if __name__ == "__main__":
    raise SystemExit(main())
