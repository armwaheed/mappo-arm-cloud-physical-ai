#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the bilingual cues and the persistent mission supervisor.

No sound card and no robot: ``Voice`` is pointed at generated WAV files and its player at
a command that exits immediately, and the supervisor is fed a child process that prints
the same tick lines ``visual_nav`` prints. What is tested is the DECISIONS -- when it
speaks, when it stays quiet, and when it tries again -- because those are what an audience
sees and what no unit of the drive path covers.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from mission import Attempt, main, supervise
from voice import CUES, Voice


def _voice_dir(skip: tuple = ()) -> Path:
    """A directory of real (silent) WAV files for every cue but ``skip``."""
    directory = Path(tempfile.mkdtemp())
    header = (b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
              b"\x40\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    for pair in CUES.values():
        for name in pair:
            if name not in skip:
                (directory / name).write_bytes(header)
    return directory


#: A real command that accepts any arguments and makes no sound. Located rather than
#: hard-coded: it is /bin/true on Linux and /usr/bin/true on macOS, and hard-coding the
#: Linux path made every cue here "fail" silently -- which is voice.py behaving correctly
#: and the test lying about it.
SILENT_PLAYER = shutil.which("true")


def _quiet_voice(**kwargs) -> Voice:
    """A Voice that records decisions and makes no sound."""
    return Voice(_voice_dir(), player=SILENT_PLAYER, **kwargs)


def _emitter(lines: list[str]) -> list[str]:
    """A child process that prints ``lines`` -- the real subprocess path, faked data."""
    script = "\n".join(f"print({line!r}, flush=True)" for line in lines)
    return [sys.executable, "-c", script]


# ── Voice ───────────────────────────────────────────────────────────────────
def test_voice_is_silent_rather_than_fatal_when_it_cannot_speak():
    """A missing sound card must cost the demo its announcements and nothing else.
    Losing a run to a codec is a worse outcome than losing the sentence."""
    for kwargs in ({"directory": None}, {"directory": "/nonexistent-voice-dir"}):
        voice = Voice(**kwargs)
        assert not voice.enabled and voice.reason
        assert voice.say("person_stop") is False
        assert "SILENT" in voice.describe()


def test_a_cue_does_not_repeat_while_somebody_is_still_deciding_to_move():
    """Held for a minute by one stationary person, a robot that says 'excuse me' twenty
    times stops reading as polite. The guard is what makes the cue mean something."""
    voice = _quiet_voice(repeat_guard_s=10.0)
    assert voice.say("person_stop", now=100.0) is True
    assert voice.say("person_stop", now=105.0) is False, "repeated inside the guard"
    assert voice.say("person_stop", now=111.0) is True, "and speaks again after it"


def test_missing_cue_files_are_named_at_startup_not_discovered_mid_run():
    """Finding out the robot cannot speak at the moment it needs to is the one time the
    information is useless."""
    voice = Voice(_voice_dir(skip=("rrd_fault_zh.wav",)), player=SILENT_PLAYER)
    assert voice.missing() == ["rrd_fault_zh.wav"]
    assert voice.say("fault") is False, "a half-present cue is not spoken"
    assert voice.say("person_stop") is True, "and the others still work"


def test_chinese_is_played_before_english():
    """The demo is in Shanghai. The person being asked to move should not have to wait
    through an English sentence first."""
    for zh, en in CUES.values():
        assert zh.endswith("_zh.wav") and en.endswith("_en.wav")


# ── The supervisor ──────────────────────────────────────────────────────────
def test_it_asks_the_room_to_clear_when_the_robot_stays_held():
    """THE POINT OF THIS MODULE. A run that holds is a run nobody can explain from the
    outside; the robot saying so in Chinese and English is the whole feature."""
    voice = _quiet_voice()
    ticks = [f"[  {i}.0s] veto-hold v=(+0.00,+0.00,+0.00) obst=[personx1]" for i in range(8)]
    clock = iter([0.0] + [10.0] * 40)          # first tick starts the clock, rest are late
    attempt = supervise(_emitter(ticks), voice, patience_s=4.0,
                        echo=lambda _l: None, clock=lambda: next(clock))
    assert attempt.spoke_for_help is True
    assert attempt.held_ticks == len(ticks)
    assert attempt.arrived is False


def test_a_brief_veto_while_the_planner_re_solves_does_not_start_talking():
    """The planner vetoes for a tick or two routinely. Announcing that would make the
    robot chatter through a run that is going fine."""
    voice = _quiet_voice()
    ticks = ["[  0.1s] veto-hold v=(+0.00,+0.00,+0.00)",
             "[  0.2s] policy v=(+0.34,+0.00,+0.00)"]
    clock = iter([0.0, 0.5, 1.0, 1.5])
    attempt = supervise(_emitter(ticks), voice, patience_s=4.0,
                        echo=lambda _l: None, clock=lambda: next(clock))
    assert attempt.spoke_for_help is False
    assert attempt.moving_ticks == 1


def test_arrival_is_recognised_from_the_words_visual_nav_actually_prints():
    """Coupled to the real string, so a change to the drive path's vocabulary breaks this
    test rather than silently turning every arrival into a retry."""
    attempt = supervise(_emitter(["[visual_nav] outcome: arrived (0.42 m from goal)"]),
                        _quiet_voice(), patience_s=4.0, echo=lambda _l: None)
    assert attempt.arrived is True and attempt.outcome.startswith("arrived")


def test_a_stall_is_not_an_arrival():
    """The stall detector's message mentions the tether and reads like an explanation.
    It is still a failure, and must be retried rather than celebrated."""
    stall = "[visual_nav] outcome: stalled: commanded 0.30 m/s for 4.0s and moved 0.00 m"
    attempt = supervise(_emitter([stall]), _quiet_voice(), patience_s=4.0,
                        echo=lambda _l: None)
    assert attempt.arrived is False and attempt.outcome.startswith("stalled")


def test_it_retries_until_it_arrives_and_then_stops():
    """Attempt one stalls, attempt two arrives: the supervisor must run twice, not once
    and not three times."""
    marker = Path(tempfile.mkdtemp()) / "n"
    script = (f"import pathlib;p=pathlib.Path({str(marker)!r});"
              "n=int(p.read_text()) if p.exists() else 0;p.write_text(str(n+1));"
              "print('[visual_nav] outcome: ' + ('arrived (0.4 m from goal)' "
              "if n else 'stalled: commanded 0.30 m/s'), flush=True)")
    rc = main(["--voice-dir", str(_voice_dir()), "--no-voice", "--cooldown", "0",
               "--max-attempts", "5", "--", sys.executable, "-c", script])
    assert rc == 0
    assert marker.read_text() == "2", "it stopped as soon as it arrived"


def test_it_stops_and_says_so_rather_than_retrying_for_ever():
    """The brief was 'does not give up'. An unbounded loop on a platform that reports no
    motor temperature is not that -- it is a robot walking until something breaks."""
    never = "[visual_nav] outcome: stalled: commanded 0.30 m/s"
    rc = main(["--no-voice", "--cooldown", "0", "--max-attempts", "3",
               "--", sys.executable, "-c", f"print({never!r}, flush=True)"])
    assert rc == 1, "a run that never arrived must not report success"


def test_an_attempt_starts_out_having_neither_arrived_nor_spoken():
    """Guards against a default that would make a failed run look like a success."""
    fresh = Attempt()
    assert fresh.arrived is False and fresh.spoke_for_help is False
    assert fresh.outcome is None


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_mission: {len(tests)}/{len(tests)} passed")
