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
import re
import subprocess
import sys
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


class Attempt:
    """What one run of the child did, in the terms the supervisor decides on."""

    def __init__(self) -> None:
        self.outcome: str | None = None
        self.arrived = False
        self.held_ticks = 0
        self.moving_ticks = 0
        self.spoke_for_help = False


def supervise(command: list[str], voice: Voice, *, patience_s: float,
              echo=print, clock=time.monotonic) -> Attempt:
    """Run ``command`` once, narrating it. Returns what happened."""
    attempt = Attempt()
    held_since: float | None = None
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            line = line.rstrip("\n")
            echo(line)
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
        process.wait()
    return attempt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        epilog="Everything after -- is the drive command to supervise.")
    parser.add_argument("--voice-dir", type=Path, default=None,
                        help="directory of rrd_*_zh.wav / rrd_*_en.wav cues")
    parser.add_argument("--no-voice", action="store_true")
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

    voice = Voice(args.voice_dir, enabled=not args.no_voice)
    print(f"[mission] {voice.describe()}")
    absent = voice.missing()
    if absent:
        # Named at start-up, not discovered when the robot tries to speak mid-run.
        print(f"[mission] ⚠️  missing cue files, those cues will be silent: "
              f"{', '.join(absent)}")
    print(f"[mission] up to {args.max_attempts} attempts, {args.cooldown:.0f}s cooldown "
          f"between them, {args.max_total_seconds:.0f}s total ceiling")

    started = time.monotonic()
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
        attempt = supervise(command, voice, patience_s=args.patience)
        print(f"[mission] attempt {attempt_number}: outcome={attempt.outcome!r} "
              f"held={attempt.held_ticks} moving={attempt.moving_ticks} "
              f"asked_for_help={attempt.spoke_for_help}")
        if attempt.arrived:
            print(f"[mission] ARRIVED on attempt {attempt_number} "
                  f"after {time.monotonic() - started:.0f}s")
            voice.close()
            return 0
        voice.say("fault")
        if attempt_number < args.max_attempts:
            print(f"[mission] cooling down {args.cooldown:.0f}s before the next attempt")
            time.sleep(args.cooldown)

    print(f"[mission] STOPPING: {args.max_attempts} attempts did not reach the goal. "
          f"Not a success — read the outcomes above.")
    voice.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
