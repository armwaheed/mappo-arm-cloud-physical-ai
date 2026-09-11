#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Drive to the goal, and keep trying until it is reached.

WHAT THIS IS FOR. A single ``mappo_drive`` run ends the first time anything goes wrong:
a person stands in the path until the stall detector fires, the goal leaves the frame,
the map fills with furniture. That is the right behaviour for a measurement and the wrong
behaviour for a demo in front of an audience, where the failure everyone remembers is a
robot standing still with nobody able to say why.

So this supervises a run rather than replacing it. It reads the child's stdout, speaks
when the robot needs the room cleared, and starts a fresh attempt when one ends without
arriving. Each attempt re-acquires the goal from scratch, which is what makes a retry
meaningful rather than a repeat.

WHY IT WATCHES STDOUT INSTEAD OF LIVING IN THE CONTROL LOOP. The loop already carries
150-500 ms of perception latency on this robot, and every millisecond added to it is a
millisecond of staleness in the belief the planner acts on. A supervisor outside the
process cannot slow the loop down no matter how badly it is written -- and audio, which
takes seven seconds to say one bilingual sentence, is exactly the kind of thing that must
never be on that path.

⚠️ RETRIES ARE BOUNDED, AND DELIBERATELY. The brief for this was "the robot does not give
up until it reaches the goal". An unbounded retry loop is not that; it is a robot walking
until something breaks. This platform reports NO motor temperatures -- every run here
carries ``--accept-no-motor-temperatures`` -- so nothing in software can see heat build
across back-to-back attempts, and ``robot-stack/SAFETY.md`` and AGENTS.md both require
moving runs to stay bounded. The compromise is a high attempt cap, a real cooldown between
attempts, and a total wall-clock ceiling: the robot keeps trying for as long as an operator
would reasonably let it, and then stops and says so rather than deciding for itself.

⛔ THE GESTURES THIS FIRES ARE THE ONES THAT DO NOT TRAVEL, and that is a safety property
rather than a preference. `play_flourish` runs after a run has arrived, failed, or lost
sight of its goal -- by which point nobody is necessarily watching -- so every call site
here hands it a CONSTANT kind out of `flourish.ARRIVAL_KINDS`, all four of which keep the
robot's centre where it is. `test_flourish.py` reads those constants out of this file's
syntax tree and asserts it.

There is now exactly ONE path from an arrival to a vendor canned action that TRAVELS, and
it is not `play_flourish` and is not named in this file. `--flip-on-arrival` hands
`look_behind.arrival_flip` a runner, and that module turns the robot around, looks at the
space with the run's own camera and detector, turns back, and refuses unless the look
authorised it. Off by default; `look_behind.py`'s docstring carries the argument, including
what a VOC detector cannot see.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from voice import Voice  # noqa: E402

try:
    import look_behind
except Exception as _absent:  # pragma: no cover - a tree staged without locomotion/
    # ⚠️ GUARDED, AND NOT OUT OF TIDINESS. `look_behind` imports `flourish`, which lives
    # in the SIBLING `locomotion/` directory -- and `flourish_command` below already
    # anticipates a deployment where that file is not there, by checking `is_file()` and
    # skipping the gesture rather than failing. A bare import here would turn that
    # survivable absence into a crash on every run, including the runs that never wanted a
    # gesture at all. The flip is the only thing that becomes unreachable, and it says so
    # at the parser rather than at the goal.
    look_behind = None
    LOOK_BEHIND_ABSENT = repr(_absent)
else:
    LOOK_BEHIND_ABSENT = None

#: What :func:`run_gesture` returns when no command could be built. Mirrored here so this
#: module still has a value for it in a tree where ``look_behind`` did not import.
GESTURE_UNAVAILABLE = -1 if look_behind is None else look_behind.GESTURE_UNAVAILABLE


def flourish_kinds() -> tuple:
    """Every vendor canned action this tree can fire, or ``()`` if flourish is absent.

    Read from ``flourish.OPERATOR_ONLY_KINDS`` rather than listed here, for the reason
    ``look_behind`` gives about its own table: a second copy is a second thing to forget to
    update, and the one that goes stale is the one an operator is offered.
    """
    if look_behind is None:
        return ()
    return tuple(look_behind.flourish.OPERATOR_ONLY_KINDS)

#: Tick statuses that mean the robot is NOT making progress. Taken from the words
#: ``visual_nav`` already prints, so this cannot drift into inventing its own vocabulary.
HELD = frozenset({"hold", "veto-hold", "stop", "blocked", "goal-search"})

#: Statuses that mean it IS driving.
MOVING = frozenset({"policy", "exec-turn", "goal", "supervisor"})

#: Held for this long before asking the room to clear. Short enough to be useful, long
#: enough that a momentary veto while the planner re-solves does not start talking.
DEFAULT_PATIENCE_S = 4.0

#: Between attempts. Not politeness -- this is the only thermal margin the platform has,
#: because it reports no motor temperature at all.
DEFAULT_COOLDOWN_S = 25.0

DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_MAX_TOTAL_S = 900.0

_TICK = re.compile(r"^\[\s*[\d.]+s\]\s+(\S+)")
_OUTCOME = re.compile(r"outcome:\s*(.+?)\s*$")

#: The refusal that only a HUMAN can clear. ``assert_axis_state_ready`` refuses when the
#: robot is not in force-control standing, and no amount of retrying changes that by
#: itself -- so this is the one failure where speaking is not decoration but the entire
#: remedy. Matched on the refusal's own words rather than an exit code, because the exit
#: code is shared with every other SystemExit in the drive path.
#: The refusals a PERSON can clear by setting the robot's mode, as opposed to a fault.
#: `assert_axis_state_ready` has five gates and this matched exactly one of them, so the
#: other three operator-fixable ones fell through to "a fault has occurred" -- a sentence
#: that tells an operator standing next to the robot nothing they can act on. Measured
#: 2026-09-04 on robot 2: three attempts, every one refused with
#: `Lite3 gait_state=4; axis profile allows (0,)`, and all three announced a generic fault
#: while the fix was one control on the vendor app.
#:
#: `error_state` is deliberately NOT here. It is the one gate standing the robot up cannot
#: clear, and promising an operator that it can is worse than saying nothing.
#: The outcome a look-around can actually do something about. Measured 2026-09-04 on both
#: robots: `outcome: goal never sighted in 20s`, three attempts, then a fault -- while the
#: marker was in the room and the detector found it in the RAW frame, the RECTIFIED frame
#: and the RTSP stream the run itself reads. Nothing was broken; the robot was pointed the
#: wrong way and waited 20 s three times for a marker behind it.
_GOAL_NEVER_SIGHTED = re.compile(r"goal never sighted", re.IGNORECASE)

#: Where to look after each failed attempt, in degrees, applied BETWEEN attempts. The
#: robot keeps each heading rather than returning to the one that just failed -- a scan
#: that comes home leaves the next attempt facing exactly the direction that saw nothing.
#: +90 then -180 lands on start, +90, -90: three headings, and at the camera's 134 deg
#: that is roughly 400 degrees of arc looked at across a mission.
_LOOK_DEGREES = (+90.0, -180.0)

_NEEDS_STANDING = re.compile(
    r"basic_state=\d+.*force-control state"
    r"|policy_state=\d+.*moving mode"
    r"|gait_state=\d+.*axis profile allows"
    r"|motion_state=\d+.*stationary/stepping")


#: Flags whose value is a path this supervisor must keep unique per attempt.
EVIDENCE_FLAGS = ("--telemetry", "--record", "--record-raw")


def per_attempt(command: list[str], attempt: int) -> list[str]:
    """Give each attempt its own evidence files.

    Found by running it: the launcher computes one run id, so every retry wrote over the
    telemetry and video of the attempt before it -- destroying the recording of the
    failure that CAUSED the retry, which is the one a person would actually want to
    watch. The first attempt keeps the unsuffixed name so a single-attempt run reads
    exactly as it did before this existed.
    """
    if attempt <= 1:
        return list(command)
    out = list(command)
    for index, token in enumerate(out[:-1]):
        if token in EVIDENCE_FLAGS:
            path = Path(out[index + 1])
            out[index + 1] = str(path.with_name(
                f"{path.stem}-attempt{attempt}{path.suffix}"))
    return out


class Attempt:
    """What one run of the child did, in the terms the supervisor decides on."""

    def __init__(self) -> None:
        self.outcome: str | None = None
        self.arrived = False
        self.needs_standing = False
        self.held_ticks = 0
        self.moving_ticks = 0
        self.spoke_for_help = False
        #: How the drive ENDED, kept for the case where it printed no outcome at all.
        #: `outcome=None` on its own says the drive died without deciding anything and
        #: nothing about why; these two turn that into a diagnosis. See `_no_outcome`.
        self.exit_code: int | None = None
        self.last_line: str = ""


#: Set when a stop signal arrives. The retry loop reads it so that stopping a mission
#: stops the mission, rather than ending one attempt and starting the next.
_STOP = threading.Event()


def stop_requested() -> bool:
    """Whether a SIGTERM or SIGINT has been seen since :func:`main` started."""
    return _STOP.is_set()


def _terminate(process: subprocess.Popen) -> None:
    """Stop the child, SIGTERM only, and wait for it.

    ⛔ **There is no SIGKILL path here**, for the reason ``SAFETY.md`` §0 gives: the drive
    process damps its velocity on SIGTERM, and a hard kill leaves the last command latched
    on a robot that is still walking.
    """
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.terminate()
    process.wait()


def supervise(command: list[str], voice: Voice, *, patience_s: float,
              echo=print, clock=time.monotonic) -> Attempt:
    """Run ``command`` once, narrating it. Returns what happened.

    **A signal here has to reach the child, not just this process.** The child is what
    commands velocity; this only reads its stdout. ``run-venue-demo.sh`` hid that, because
    Ctrl-C reaches the whole foreground process group and the child got it anyway. A
    supervisor started by Device Connect does not have that luck: ``run_control``'s stop is
    ``kill -TERM`` against the ONE recorded pid, so without the handler below a stop would
    end this process and leave the robot driving.
    """
    attempt = Attempt()
    held_since: float | None = None
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _on_stop(_signum, _frame):
        _STOP.set()
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                process.terminate()

    previous: dict = {}
    for number in (signal.SIGTERM, signal.SIGINT):
        # ValueError: not the main thread. Worth continuing without the handler rather
        # than refusing to run at all; the finally below still stops the child.
        with contextlib.suppress(ValueError, OSError):
            previous[number] = signal.signal(number, _on_stop)
    try:
        for line in process.stdout:
            line = line.rstrip("\n")
            echo(line)
            if line.strip():
                # The last thing the drive said before it stopped saying anything. When it
                # exits without an outcome this is the only evidence of why, and it is
                # usually the refusal itself.
                attempt.last_line = line
            if _NEEDS_STANDING.search(line):
                attempt.needs_standing = True
                if voice.say("stand_request"):
                    attempt.spoke_for_help = True
            found = _OUTCOME.search(line)
            if found:
                attempt.outcome = found.group(1)
                attempt.arrived = attempt.outcome.startswith("arrived")
            tick = _TICK.match(line)
            if not tick:
                continue
            status = tick.group(1)
            if status in HELD:
                attempt.held_ticks += 1
                if held_since is None:
                    held_since = clock()
                # Ask ONCE per held stretch; Voice's own guard stops it repeating.
                elif clock() - held_since >= patience_s and voice.say("person_stop"):
                    attempt.spoke_for_help = True
            elif status in MOVING:
                attempt.moving_ticks += 1
                if held_since is not None and attempt.spoke_for_help:
                    # It only makes sense to thank somebody who was actually asked.
                    voice.say("person_thanks")
                    voice.say("resuming")
                held_since = None
    finally:
        for number, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(number, handler)
        # Not process.wait() alone. A child that closed stdout but is still running would
        # hang this for ever, and a child still running after this returns is a robot
        # nobody is reading the output of any more.
        _terminate(process)
        # After _terminate, which waits: the code is settled by here.
        attempt.exit_code = process.returncode
    return attempt


def _no_outcome(attempt: Attempt) -> str:
    """What to print when the drive decided nothing, in place of a bare ``outcome=None``.

    ⚠️ THIS IS THE LINE THAT WAS MISSING. Measured 2026-09-04: the run profile carried
    `--prop-radius` alongside `--static-profile`, which `visual_nav.static_profile`
    refuses, so every attempt on both robots exited 1 during its own banner. All the
    supervisor said was `outcome=None held=0 moving=0`, which describes a drive that
    decided nothing and says nothing about why. The refusal WAS on stdout, one line
    above, and it took two sessions to read it -- the driver's log truncates long lines
    to 20 characters, and `[visual_nav] --stati...(117 chars)` is not a legible refusal.

    So the supervisor now says the exit code and quotes the last line itself.
    `held=0 moving=0` means it never issued a tick, which separates "refused before it
    drove" from "drove and then died".
    """
    drove = attempt.held_ticks or attempt.moving_ticks
    # Built outside the f-string: the robots run Python 3.8, where an f-string expression
    # may not reuse the enclosing quote character.
    what = ("ran and then stopped" if drove else
            "never issued a tick, so it stopped before it drove")
    last = attempt.last_line or "(nothing at all)"
    return (f"[mission] the drive printed no outcome and exited {attempt.exit_code}. "
            f"It {what}. Its last line was:\n"
            f"[mission]     {last}")


#: Transport flags the gesture needs and the drive command already carries. Copied FROM
#: the drive command rather than restated, so a gesture is commanded through exactly the
#: interface the run was, and cannot be pointed at a different robot by a stale default --
#: which is the failure `--motion-host` defaulting to robot 1 already cost this fleet once.
#: ``--live`` obeys the same rule and is handled in `flourish_command`, separately only
#: because it carries no value to copy. Restating it there, rather than inheriting it, is
#: what let a dry run turn a robot; the comment at that line has the measurement.
_FLOURISH_PASSTHROUGH = ("--locomotion-transport", "--axis-profile", "--axis-local-port",
                         "--motion-host", "--command-port", "--state-port")

#: Where the gesture lives, relative to this file.
_FLOURISH = (Path(__file__).resolve().parents[1] / "locomotion" / "flourish.py")


def flourish_command(command: list[str], kind: str, args,
                     extra: tuple = ()) -> list[str] | None:
    """The gesture invocation, or ``None`` with a printed reason if it cannot be built.

    ``None`` is not an error. A gesture is decoration on the end of a run, and a run that
    ARRIVED has succeeded whether or not the robot then spun. Every path here that cannot
    produce a command says why and returns, and `main` carries on to its own exit code.
    """
    if not args.flourish:
        return None
    if not _FLOURISH.is_file():
        print(f"[mission] no flourish at {_FLOURISH}; skipping the {kind}")
        return None
    for name, value in (("--robot-id", args.robot_id), ("--firmware", args.firmware),
                        ("--payload", args.payload)):
        if not value:
            print(f"[mission] --flourish needs {name}; skipping the {kind}")
            return None
    if args.flourish_lane_width is None:
        print(f"[mission] --flourish needs --flourish-lane-width, because the gesture "
              f"turns in place and this robot has no lateral sensing; skipping the {kind}")
        return None
    out = [sys.executable, str(_FLOURISH), "--kind", kind, *extra,
           "--robot-id", args.robot_id, "--firmware", args.firmware,
           "--payload", args.payload,
           "--lane-width-metres", str(args.flourish_lane_width)]
    # ⛔ --live is INHERITED from the drive, never restated here. `--live` is the only flag
    # in `mappo_drive.py` that commands a leg, and the driver's whole contract for a scene
    # check rests on that: `start_run(arm_motion=False)` builds a drive command without it,
    # and reports `can_move=False` because a run without `--live` has no path to a leg --
    # an absent capability rather than a checked permission. Hard-coding it here handed the
    # gesture a path of its own, out of a SEPARATE process the driver never gated.
    # Measured 2026-09-04 on robot 1: a `start_run(arm_motion=False)` that reported
    # `can_move=False` went on to fire a live +90 deg `look` between attempts. Nothing
    # turned only because the robots were lying down and the vendor mode gate refused it
    # (`Lite3LinkLost: basic_state=1; axis motion requires documented force-control state
    # 6`). On a robot standing in state 6 the same scene check would have turned it.
    # Without `--live`, flourish.py prints its plan and returns 0, so the narration a
    # dry run is FOR survives; `--operator-ready` goes with `--live` because flourish.py
    # refuses `--live` without it.
    if "--live" in command:
        out += ["--live", "--operator-ready"]
    for flag in _FLOURISH_PASSTHROUGH:
        if flag in command:
            out += [flag, command[command.index(flag) + 1]]
    return out


def run_gesture(command: list[str], args, kind: str, extra: tuple = ()) -> int:
    """Run one gesture and hand back its EXIT CODE, rather than swallowing it.

    ``play_flourish`` is deliberately not this and is left alone. A gesture is decoration:
    its failure must never rewrite a run's verdict, so that one prints and returns nothing.
    The look-behind is the opposite case -- whether a turn actually completed is the whole
    difference between a measurement and a guess, and ``look_behind`` refuses without it.
    Same command builder, so the same ``--live`` inheritance and the same passthrough; only
    the return value differs.

    Never raises. A subprocess that could not be started is reported as
    ``GESTURE_UNAVAILABLE``, which ``look_behind`` reads as "nothing ran", which refuses.
    """
    argv = flourish_command(command, kind, args, extra)
    if argv is None:
        return GESTURE_UNAVAILABLE
    print(f"[mission] {kind}")
    try:
        return subprocess.run(argv, check=False, timeout=60).returncode
    except Exception as failure:
        print(f"[mission] the {kind} did not run ({failure!r})")
        return GESTURE_UNAVAILABLE


def play_flourish(command: list[str], kind: str, args, extra: tuple = ()) -> None:
    """Run the gesture, and never let it change the mission's verdict.

    ⚠️ THE ORDER MATTERS AND IT IS NOT OBVIOUS. This runs AFTER the drive process has
    exited, so nothing else holds the legs, and BEFORE `voice.close()`, so a cue that is
    still playing is not cut off by the gesture's own exit. It is bounded by the gesture's
    own aborts rather than by a timeout here, because a turn that is refused should say
    which gate refused it rather than being killed by a stopwatch that knows nothing.
    """
    argv = flourish_command(command, kind, args, extra)
    if argv is None:
        return
    print(f"[mission] {kind}")
    try:
        subprocess.run(argv, check=False, timeout=60)
    except Exception as failure:
        # Deliberately broad. Whatever went wrong turning the robot in place, the run's
        # outcome was decided before this was called and must not be rewritten by it.
        print(f"[mission] the {kind} did not run ({failure!r}); the outcome above stands")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        epilog="Everything after -- is the drive command to supervise.")
    parser.add_argument("--voice-dir", type=Path, default=None,
                        help="directory of rrd_*_zh.wav / rrd_*_en.wav cues")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--voice-device", default=None, metavar="ALSA",
                        help="ALSA device for the cues, e.g. plughw:0,0. Worth stating: "
                             "this robot's PulseAudio default sink is auto_null, so the "
                             "player's default device is silent even when it exits 0")
    gesture = parser.add_argument_group(
        "flourish (off unless --flourish; it turns the robot in place)")
    gesture.add_argument("--flourish", action="store_true",
                         help="spin on arrival, rock on surrender. OFF by default: it "
                              "moves the robot after the run everybody stopped watching")
    gesture.add_argument("--flourish-lane-width", type=float, default=None, metavar="M",
                         help="clear width BOTH SIDES for the turn; required by --flourish")
    gesture.add_argument("--robot-id", default=None)
    gesture.add_argument("--firmware", default=None)
    gesture.add_argument("--payload", default=None)
    # ⛔ THE FLIP. Off, and every flag below ships NO DEFAULT, which is `flourish.py`'s own
    # rule for these actions restated at the layer that now fires them. Nothing on the
    # dashboard path reaches this: `venue_run.py` builds no `--flip-*` flag from any
    # environment variable, so a flip is only ever armed by somebody typing it.
    flip = parser.add_argument_group(
        "vendor canned action on arrival (OFF. One opcode, executed by the firmware, with "
        "nothing here able to shape, slow, shorten or interrupt it once sent. Each kind is "
        "held to its OWN travel floor -- a kind that goes ~1.5 m BACKWARD to that, a kind "
        "whose travel nobody has measured to this platform's own footprint. Runbook: "
        "deploy/ARRIVAL_ACTIONS.md, which lists every gate that refuses one and why the "
        "rear clearance is easy to measure in the wrong direction)")
    flip.add_argument("--flip-on-arrival", action="store_true",
                      help="after arriving, look behind with the run's own camera and "
                           "detector and fire --flip-kind if -- and only if -- the look "
                           "authorises it. Needs --flourish, and every other --flip-* "
                           "flag. Read look_behind.py before using this: the detector "
                           "finds VOC classes and CANNOT SEE A WALL")
    flip.add_argument("--flip-kind", default=None,
                      choices=() if look_behind is None else sorted(
                          flourish_kinds() or look_behind.BACKWARD_KINDS),
                      help=f"which vendor canned action fires on arrival, one of "
                           f"{', '.join(sorted(flourish_kinds())) or '(none available)'}. "
                           f"Each is checked against its OWN travel floor. WAS "
                           "restricted to the BACKWARD kinds, because a look behind is no "
                           "evidence about a manoeuvre whose direction nobody has "
                           "observed -- true, and it stopped applying on 2026-09-07 when "
                           "the look was disabled. What gates every kind now is the "
                           "operator's --flip-rear-clearance-metres, which flourish "
                           "checks against each action's OWN floor, so a kind that needs "
                           "less is not thereby less checked")
    flip.add_argument("--flip-rear-clearance-metres", type=float, default=None, metavar="M",
                      help="clear floor BEHIND the robot, measured with a tape. STILL "
                           "REQUIRED and not replaced by the look: the detector does not "
                           "find walls, steps or stage edges. Passed unchanged to "
                           "flourish's own --rear-clearance-metres gate")
    flip.add_argument("--flip-battery-floor-pct", type=float, default=None, metavar="PCT",
                      help="refuse below this battery percentage; fed to flourish's "
                           "--acrobatic-battery-floor-pct")
    flip.add_argument("--flip-hold-seconds", type=float, default=None, metavar="S",
                      help="how long flourish holds while the firmware runs the action. "
                           "Measured 2026-09-07: the two flips took 5.4-5.9 s to return "
                           "the robot to force-control 6, and returning sooner hands "
                           "control back mid-manoeuvre")
    parser.add_argument("--patience", type=float, default=DEFAULT_PATIENCE_S,
                        help="seconds held before asking the room to clear")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S,
                        help="seconds between attempts; this platform reports no motor "
                             "temperature, so this is its only thermal margin")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--max-total-seconds", type=float, default=DEFAULT_MAX_TOTAL_S)
    parser.add_argument("drive", nargs=argparse.REMAINDER,
                        help="-- followed by the mappo_drive command line")
    args = parser.parse_args(argv)

    command = args.drive[1:] if args.drive and args.drive[0] == "--" else args.drive
    if not command:
        parser.error("give the drive command after --")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.flip_on_arrival and look_behind is None:
        parser.error(f"--flip-on-arrival cannot be honoured in this tree: the look-behind "
                     f"would not import ({LOOK_BEHIND_ABSENT}). Most likely the sibling "
                     f"locomotion/ directory was not staged, which also means there is no "
                     f"flourish.py to turn the robot with.")
    if args.flip_on_arrival and not args.flourish:
        # Said at the parser rather than discovered after the robot has arrived: every
        # gesture the look-behind commands is built by `flourish_command`, which returns
        # None without this, so a flip armed without it would turn nothing and refuse at
        # the end of a run somebody had staged a flip for.
        parser.error("--flip-on-arrival needs --flourish: the flip is fired through "
                     "flourish.py, and --flourish is what licenses that at all")
    if args.flip_on_arrival:
        # ⛔ VALIDATED HERE BECAUSE THE LOOK NO LONGER VALIDATES IT. Every one of these was
        # checked inside `look_behind.Clearance`, which refused with a sentence naming the
        # missing one. The direct flip path bypasses that, and formats three of them with
        # `:.4f` -- so a `None` would not refuse, it would raise TypeError at the end of a
        # run somebody had staged a flip for, on a robot standing at its goal. That is the
        # #214/#217 shape exactly: a value that is fine everywhere except at the one call
        # site no test reaches. Said at the parser, before the robot has moved at all.
        missing = [name for name, value in (
            ("--flip-kind", args.flip_kind),
            ("--flip-rear-clearance-metres", args.flip_rear_clearance_metres),
            ("--flip-battery-floor-pct", args.flip_battery_floor_pct),
            ("--flip-hold-seconds", args.flip_hold_seconds)) if value is None]
        if missing:
            parser.error(
                f"--flip-on-arrival needs {', '.join(missing)}. With the rear look "
                f"disabled, --flip-rear-clearance-metres is the ONLY check on the space "
                f"this manoeuvre travels ~1.5 m into, and this platform has no rear "
                f"sensor of any kind. There is no default for it and there will not be.")

    voice = Voice(args.voice_dir, enabled=not args.no_voice,
                  device=args.voice_device)
    print(f"[mission] {voice.describe()}")
    # Prove the device opens BEFORE a run starts. A silent demo that nobody
    # notices until the robot needs to speak is the failure this prevents.
    unheard = voice.probe() if voice.enabled else None
    if unheard:
        print(f"[mission] ⚠️  NOTHING WILL BE AUDIBLE: {unheard}")
        print("[mission]    on this platform: is the account in the 'audio' group, and is")
        print("[mission]    --voice-device set? A null sink exits 0 and makes no sound.")
    absent = voice.missing()
    if absent:
        # Named at start-up, not discovered when the robot tries to speak mid-run.
        print(f"[mission] ⚠️  missing cue files, those cues will be silent: "
              f"{', '.join(absent)}")
    print(f"[mission] up to {args.max_attempts} attempts, {args.cooldown:.0f}s cooldown "
          f"between them, {args.max_total_seconds:.0f}s total ceiling")

    started = time.monotonic()
    # Cleared here rather than at import, so a second call to main() in one process (the
    # tests do exactly that) does not inherit the previous run's stop.
    _STOP.clear()
    voice.say("greeting")
    for attempt_number in range(1, args.max_attempts + 1):
        elapsed = time.monotonic() - started
        if elapsed > args.max_total_seconds:
            print(f"[mission] STOPPING: {elapsed:.0f}s spent, past the "
                  f"{args.max_total_seconds:.0f}s ceiling, after {attempt_number - 1} "
                  f"attempt(s). The goal was not reached; this is a stop, not a success.")
            voice.close()
            return 2
        print(f"\n[mission] ── attempt {attempt_number} of {args.max_attempts} "
              f"({elapsed:.0f}s elapsed) ──")
        attempt = supervise(per_attempt(command, attempt_number), voice,
                            patience_s=args.patience)
        print(f"[mission] attempt {attempt_number}: outcome={attempt.outcome!r} "
              f"held={attempt.held_ticks} moving={attempt.moving_ticks} "
              f"asked_for_help={attempt.spoke_for_help}"
              + (" NEEDS-STANDING" if attempt.needs_standing else ""))
        if attempt.outcome is None:
            print(_no_outcome(attempt))
        if attempt.needs_standing:
            print("[mission] the robot is not in force-control standing. It has asked, in "
                  "Chinese and English, for somebody to set it. Retrying after the "
                  "cooldown; nothing else will clear this.")
        if attempt.arrived:
            print(f"[mission] ARRIVED on attempt {attempt_number} "
                  f"after {time.monotonic() - started:.0f}s")
            # ⛔⛔ ARRIVE, THEN FLIP, WITH NO LOOK AND NO SPIN -- the operator's explicit
            # instruction on 2026-09-07, given twice, after being shown what it removes.
            #
            # THIS FILE NOW NAMES A KIND THAT TRAVELS. It did not, and could not, before:
            # `play_flourish` was handed a CONSTANT kind at every call site and
            # `test_flourish.py` read those constants to prove they were all in
            # `ARRIVAL_KINDS` -- kinds that keep the robot's centre where it is. The route
            # to a travelling kind ran through `look_behind`, which would not name one
            # until a completed outward turn, a completed return turn and unanimous fresh
            # frames from the run's own detector said the space behind was clear. That
            # route is now bypassed, so those tests have been changed rather than deleted:
            # they now pin exactly which travelling kind may be reached from here and that
            # `flip_on_arrival` is what gates it. The boundary moved; it is still tested.
            #
            # WHAT IS LEFT GUARDING THE SPACE BEHIND IS ONE NUMBER: the operator's
            # `--flip-rear-clearance-metres` tape measurement, enforced by
            # `flourish.check_rear`, which refuses below the kind's own floor and refuses
            # outright if the number is absent. That gate is deliberately still in the
            # path and must stay -- with the look gone it is not one guard of two, it is
            # the only one. The platform has no rear camera, no ultrasonic and no bumper,
            # and the manoeuvre travels ~1.5 m into that space.
            #
            # The spin is skipped only on the flip path. A run that is not flipping keeps
            # its victory spin, which nobody asked to change.
            if args.flip_on_arrival and look_behind is not None:
                print("[mission] ⛔⛔ FLIP ON ARRIVAL, NO REAR LOOK. The turn-and-look was "
                      "disabled by operator instruction on 2026-09-07. Nothing observes "
                      "the space behind this robot -- there is no rear sensor to observe "
                      "it with -- and the only check is the "
                      f"{args.flip_rear_clearance_metres} m clearance you stated.")
                # ⛔⛔ `--operator-triggered` IS PASSED HERE, and this is the sentence that
                # has to justify it. `flourish.check_vendor` refuses without it and says
                # why: "Pass it only if you are standing there, looking at the robot, with
                # the emergency stop in your hand." Until now nothing automatic passed it
                # -- that was the flag's whole point -- and `look_behind` earned it with a
                # completed look.
                #
                # WHAT EARNS IT NOW IS A PER-RUN HUMAN ACT, and nothing weaker: the
                # dashboard's flip checkbox, which ships unticked, is never auto-ticked,
                # is disabled unless arm motion is on, and reaches this process as
                # `MAPPO_FLIP=1` for THIS run only. An operator ticking it and pressing
                # Start is standing at the robot watching it -- which is the condition the
                # refusal names. It is not ambient configuration: `MAPPO_FLIP=0` is sent
                # explicitly when the box is clear, so a profile carrying `MAPPO_FLIP=1`
                # cannot license a flip the operator did not ask for on the day.
                #
                # It is still a downgrade and it should be read as one. The flag used to
                # mean "a human is looking at this robot right now, and a look-behind
                # agreed". It now means "a human ticked a box before this run started".
                # `flourish_command` is deliberately NOT the place this is added --
                # `test_flourish.py` pins that the bare builder never grows this flag, so
                # the licence lives at the ONE call site that can argue for it.
                code = run_gesture(
                    command, args, args.flip_kind,
                    ("--operator-triggered",
                     "--rear-clearance-metres", f"{args.flip_rear_clearance_metres:.4f}",
                     "--acrobatic-battery-floor-pct", f"{args.flip_battery_floor_pct:.4f}",
                     "--action-hold-seconds", f"{args.flip_hold_seconds:.4f}"))
                # Reported, never fatal: the robot ARRIVED, and that verdict is not the
                # flip's to rewrite. `check_rear`'s refusal arrives here as a non-zero
                # code and has to be legible, because a silent no-flip looks identical to
                # a flip that never fired for a reason nobody will go looking for.
                if code != 0:
                    print(f"[mission] the {args.flip_kind} did not fire (exit {code}). "
                          f"The arrival still stands.")
            else:
                play_flourish(command, "spin", args)
            voice.close()
            return 0
        if _STOP.is_set():
            # Somebody asked this to stop. Retrying now would restart the robot the stop
            # was issued to halt, which is the opposite of what the button said.
            print(f"[mission] STOPPED on request during attempt {attempt_number}. The "
                  f"drive process was sent SIGTERM and has exited; the goal was not "
                  f"reached and no further attempt will be started.")
            voice.close()
            return 3
        if not attempt.needs_standing:
            # It has already asked for the specific thing that would fix this; following
            # it with "a fault has occurred" would bury the actionable sentence.
            voice.say("fault")
        # LOOK SOMEWHERE ELSE BEFORE TRYING AGAIN. Retrying a "goal never sighted" from
        # the heading that did not sight it is three identical 20 s waits, which is what
        # was measured. Only for this outcome: a run that saw its goal and was blocked has
        # a different problem, and turning away from a goal it CAN see would make it worse.
        looked = False
        if (attempt.outcome and _GOAL_NEVER_SIGHTED.search(attempt.outcome)
                and not attempt.needs_standing
                and attempt_number <= len(_LOOK_DEGREES)):
            degrees = _LOOK_DEGREES[attempt_number - 1]
            print(f"[mission] the goal was never sighted; looking {degrees:+.0f} deg "
                  f"before attempt {attempt_number + 1}")
            play_flourish(command, "look", args, ("--degrees", str(degrees)))
            looked = True
        if attempt_number < args.max_attempts:
            if not looked:
                print(f"[mission] cooling down {args.cooldown:.0f}s before the next attempt")
            else:
                print(f"[mission] cooling down {args.cooldown:.0f}s from the new heading")
            # Interruptible: a stop during the cooldown should not wait it out first.
            if _STOP.wait(args.cooldown):
                print("[mission] STOPPED on request during the cooldown.")
                voice.close()
                return 3

    print(f"[mission] STOPPING: {args.max_attempts} attempts did not reach the goal. "
          f"Not a success — read the outcomes above.")
    play_flourish(command, "shake", args)
    voice.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
