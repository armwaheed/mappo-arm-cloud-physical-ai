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

#: The flip is OFF unless the deployment asks for it AND answers all four measurements,
#: and it is asked for here for the same reason the flourish is: the supervisor fires it
#: after the drive has exited.
#:
#: ⚠️ THESE ARE NOT DEFAULTABLE AND NEVER WILL BE. The action travels ~1.5 m BACKWARD into
#: the one direction this robot has no sensor of any kind pointing, and the operator
#: holding the abort is the person who stands there. ``look_behind`` turns the robot round
#: and looks first, but its detector finds VOC classes -- it cannot see a wall, a step or a
#: stage edge -- so a TAPE-MEASURED rear clearance is still required beside the look and is
#: not replaced by it. A default here would be this file guessing about a room it cannot
#: see, on the one manoeuvre where guessing puts the robot through a person.
FLIP_SETTINGS = ("MAPPO_FLIP_REAR_CLEARANCE_M",
                 "MAPPO_FLIP_BATTERY_FLOOR_PCT", "MAPPO_FLIP_HOLD_SECONDS")

#: What the robot does when it reaches the goal, chosen per run from the dashboard.
#:
#: ``spin`` is the 360 degree turn `mission.py` has always fired on arrival: an
#: ARRIVAL_KIND, which is flourish's word for a gesture that keeps the robot's centre where
#: it is, and therefore the one value here that needs no clearance and no operator gate.
#: Every other value is a VENDOR CANNED ACTION -- a single opcode the firmware executes
#: with nothing in this stack able to shape, slow, shorten or interrupt it -- so each one
#: goes through `--flip-on-arrival` and is checked against its OWN floor by
#: `flourish.check_rear`.
#:
#: ⚠️ `MAPPO_FLIP_KIND` IS NO LONGER READ. The kind is the operator's per-run choice now,
#: not a deployment constant, so it arrives in this variable instead. A profile still
#: carrying `MAPPO_FLIP_KIND` is ignored rather than obeyed -- silently honouring a stale
#: deployment default would be the dashboard's choice losing to a file.
ARRIVAL_SPIN = "spin"
ARRIVAL_ACTION = "MAPPO_ARRIVAL_ACTION"


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

    # THE FLIP, which is the flourish's rule with one extra clause: it needs the flourish
    # itself. `mission.py` refuses `--flip-on-arrival` without `--flourish` (its own check,
    # kept there because a hand-typed command line must hit it too), and building the flags
    # here anyway would turn an operator's tick into a refusal at the far end -- after the
    # robot has been committed. Checked here so it degrades to "no flip" with a reason on
    # the console instead.
    flip: list = []
    action = env.get(ARRIVAL_ACTION, "").strip()
    # ⛔ THE LEGACY BOOLEAN IS REFUSED, NOT TRANSLATED. `MAPPO_FLIP=1` meant one thing when
    # it was the only switch, and the tempting kindness is to keep honouring it as that
    # thing. It is the wrong call twice over: it would put a travelling kind's NAME in this
    # file, which `test_look_behind.py`'s sweep forbids for the good reason that a literal
    # here is the only thing that could reach a command line by accident -- and it would
    # mean a stale variable in a profile written weeks ago silently deciding what a robot
    # does at the goal. Saying so is the whole point: an operator reading this line knows
    # their old setting did nothing, which "it still works" would never tell them.
    if not action and env.get("MAPPO_FLIP", "").strip() not in ("", "0", "false", "False"):
        print(f"[venue-run] MAPPO_FLIP is set but {ARRIVAL_ACTION} is not. MAPPO_FLIP is "
              f"RETIRED and is being IGNORED -- it is not read as any action. Set "
              f"{ARRIVAL_ACTION} to one of the kinds in flourish.OPERATOR_ONLY_KINDS, or "
              f"to '{ARRIVAL_SPIN}'. This run will {ARRIVAL_SPIN}.", flush=True)
    if action and action != ARRIVAL_SPIN:
        answered = {name: env.get(name, "").strip() for name in FLIP_SETTINGS}
        missing = [name for name, value in answered.items() if not value]
        if not flourish:
            print(f"[venue-run] {ARRIVAL_ACTION}={action} but the flourish is not enabled, "
                  f"and the action is fired BY the flourish; the run will NOT do it. Set "
                  f"MAPPO_FLOURISH and its four settings too.", flush=True)
        elif missing:
            print(f"[venue-run] {ARRIVAL_ACTION}={action} but {', '.join(missing)} "
                  f"{'is' if len(missing) == 1 else 'are'} not; the run will NOT do it. "
                  f"These are measurements of the ROOM and have no safe default -- a "
                  f"vendor canned action is a single opcode nothing here can interrupt, "
                  f"and some of these travel ~1.5 m into the robot's blind side.",
                  flush=True)
        else:
            flip = ["--flip-on-arrival",
                    "--flip-kind", action,
                    "--flip-rear-clearance-metres",
                    answered["MAPPO_FLIP_REAR_CLEARANCE_M"],
                    "--flip-battery-floor-pct",
                    answered["MAPPO_FLIP_BATTERY_FLOOR_PCT"],
                    "--flip-hold-seconds", answered["MAPPO_FLIP_HOLD_SECONDS"]]

    return [python, "-u", "mission.py", *voice, *flourish, *flip,
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
