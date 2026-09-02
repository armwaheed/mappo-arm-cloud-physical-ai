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
_NEEDS_STANDING = re.compile(r"basic_state=\d+.*force-control state")


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
    return attempt


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
        if attempt.needs_standing:
            print("[mission] the robot is not in force-control standing. It has asked, in "
                  "Chinese and English, for somebody to set it. Retrying after the "
                  "cooldown; nothing else will clear this.")
        if attempt.arrived:
            print(f"[mission] ARRIVED on attempt {attempt_number} "
                  f"after {time.monotonic() - started:.0f}s")
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
        if attempt_number < args.max_attempts:
            print(f"[mission] cooling down {args.cooldown:.0f}s before the next attempt")
            # Interruptible: a stop during the cooldown should not wait it out first.
            if _STOP.wait(args.cooldown):
                print("[mission] STOPPED on request during the cooldown.")
                voice.close()
                return 3

    print(f"[mission] STOPPING: {args.max_attempts} attempts did not reach the goal. "
          f"Not a success — read the outcomes above.")
    voice.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
