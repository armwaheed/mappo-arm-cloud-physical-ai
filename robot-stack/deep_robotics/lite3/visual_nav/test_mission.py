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

import contextlib
import shutil
import signal
import sys
import tempfile
import time
import wave
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from mission import Attempt, main, per_attempt, stop_requested, supervise
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


def test_the_probe_reports_silence_that_exits_zero():
    """THE LESSON FROM THE ROBOT. The account was not in the `audio` group, so the device
    would not open, PulseAudio's default sink was `auto_null` -- a black hole -- and aplay
    wrote into it and exited 0. Every check passed and nothing was audible. A zero exit
    code is not evidence of sound, so the probe reads stderr too."""
    quiet = _voice_dir()
    liar = Path(tempfile.mkdtemp()) / "liar.sh"
    liar.write_text("#!/bin/sh\necho 'aplay: main:852: audio open error: "
                    "No such device' >&2\nexit 0\n")
    liar.chmod(0o755)
    voice = Voice(quiet, player=str(liar))
    assert voice.enabled, "it looks configured, which is exactly the trap"
    assert voice.probe() is not None, "a 0 exit with an ALSA error is NOT audible"
    assert "audio open error" in voice.probe()


def test_the_probe_passes_when_the_device_really_opens():
    """The other side of it: a player that opens the device and says nothing must not be
    reported as broken, or the warning becomes noise an operator learns to ignore."""
    voice = _quiet_voice()
    assert voice.probe() is None


def test_only_one_utterance_plays_at_a_time_and_the_rest_queue():
    """FOUND BY THE OPERATOR, TWICE. First only the Chinese was heard; then, once both
    played, the English STARTED OVER THE TOP of the longer Chinese phrases. The second is
    the subtler bug: `aplay -D pulse` hands its buffer to PulseAudio and EXITS while the
    sink is still rendering, so chaining two invocations never serialised anything and no
    exit code could have revealed it. One queue, one worker, one voice."""
    voice = _quiet_voice()
    order: list = []
    voice._play_one = lambda path: (order.append(path.name), time.sleep(0.05))
    assert voice.say("person_stop") is True
    voice.close(timeout_s=10.0)
    assert [n.split("_")[-1] for n in order] == ["zh.wav", "en.wav"], "Chinese, then English"
    assert voice.pending() == 0


def test_the_player_returning_early_does_not_end_the_utterance():
    """The whole cause of the overlap. A player that exits immediately must still cost
    the wall time of the file, or the next utterance starts over the top of it."""
    directory = _voice_dir()
    voice = Voice(directory, player=SILENT_PLAYER)
    sample = directory / "rrd_person_stop_zh.wav"
    assert voice.duration_s(sample) >= 0.0
    long_enough = Path(tempfile.mkdtemp()) / "half.wav"
    with contextlib.closing(wave.open(str(long_enough), "wb")) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * 12000)          # exactly 0.5 s
    assert abs(voice.duration_s(long_enough) - 0.5) < 0.01
    started = time.monotonic()
    voice._play_one(long_enough)
    assert time.monotonic() - started >= 0.45, "returned before the sound could have ended"


def test_a_backlog_is_dropped_rather_than_queued_for_ever():
    """By the time a cue five deep is spoken, the situation it described is over. Saying
    it then is worse than not saying it."""
    voice = _quiet_voice(repeat_guard_s=0.0)
    voice._play_one = lambda path: time.sleep(0.4)
    accepted = sum(1 for i in range(8)
                   if voice.say("person_stop", now=100.0 + i))
    voice.close(timeout_s=15.0)
    assert accepted < 8, "an unbounded backlog was accepted"


def test_an_explicit_alsa_device_reaches_the_player():
    """The default device on this robot is a null sink, so naming the hardware is what
    makes the difference between audible and not."""
    voice = Voice(_voice_dir(), player=SILENT_PLAYER, device="plughw:0,0")
    assert voice._command(Path("/x/a.wav"))[1:3] == ["-D", "plughw:0,0"]
    assert "-D" not in Voice(_voice_dir(), player=SILENT_PLAYER)._command(Path("/x/a.wav"))


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


def test_it_asks_for_standing_mode_when_only_a_human_can_clear_the_refusal():
    """The one failure where speaking IS the remedy. assert_axis_state_ready refuses a
    robot that is not in force-control standing, and no amount of retrying changes that
    by itself -- so it must ask, in Chinese and English, for the specific thing needed."""
    voice = _quiet_voice()
    refusal = ("deep_robotics.lite3.locomotion.lite3_udp_locomotion.Lite3LinkLost: "
               "Lite3 basic_state=8; axis motion requires documented force-control state 6")
    attempt = supervise(_emitter([refusal]), voice, patience_s=4.0, echo=lambda _l: None)
    assert attempt.needs_standing is True
    assert attempt.spoke_for_help is True, "it must ask, not just record"
    assert attempt.arrived is False


def test_an_ordinary_stall_is_not_mistaken_for_a_posture_refusal():
    """Guards the regex: matching too broadly would make every failure ask for standing
    mode, which is the wrong instruction and would train the operator to ignore it."""
    stall = "[visual_nav] outcome: stalled: commanded 0.30 m/s for 4.0s and moved 0.00 m"
    attempt = supervise(_emitter([stall]), _quiet_voice(), patience_s=4.0,
                        echo=lambda _l: None)
    assert attempt.needs_standing is False


def test_a_retry_does_not_overwrite_the_evidence_of_the_failure_that_caused_it():
    """FOUND BY RUNNING IT. The launcher computes one run id, so before this every retry
    wrote over the telemetry and video of the attempt before -- destroying the recording
    of the failure that caused the retry, which is the one worth watching."""
    base = ["python3", "drive.py", "--telemetry", "/e/run.jsonl",
            "--record", "/e/run.mp4", "--record-raw", "/e/run-raw.mp4", "--live"]
    assert per_attempt(base, 1) == base, "a single-attempt run reads exactly as before"
    second = per_attempt(base, 2)
    assert "/e/run-attempt2.jsonl" in second
    assert "/e/run-attempt2.mp4" in second
    assert "/e/run-raw-attempt2.mp4" in second
    assert second[-1] == "--live", "flags without a path are untouched"
    assert per_attempt(base, 3).count("/e/run-attempt3.jsonl") == 1


def test_a_child_that_stops_talking_is_stopped_rather_than_waited_on_for_ever():
    """``process.wait()`` on its own hangs here until the child decides to exit.

    A supervisor blocked in ``wait`` is a robot whose output nobody is reading any more,
    for as long as the drive process feels like running.
    """
    # BOTH descriptors: stderr is a dup of the same pipe (stderr=STDOUT), so closing
    # only fd 1 leaves the write end open and the parent never sees EOF.
    script = ("import os, sys, time;"
              "print('[visual_nav] outcome: stalled: commanded 0.30 m/s', flush=True);"
              "sys.stdout.flush(); os.close(1); os.close(2); time.sleep(60)")
    started = time.monotonic()
    attempt = supervise([sys.executable, "-c", script], _quiet_voice(),
                        patience_s=1.0, echo=lambda _line: None)
    assert time.monotonic() - started < 20, "it waited on a child that had stopped talking"
    assert attempt.outcome is not None


def test_a_stop_signal_terminates_the_drive_and_not_only_the_supervisor():
    """``run_control``'s stop is ``kill -TERM`` against ONE pid: this process.

    The child is what commands velocity. Under ``run-venue-demo.sh`` Ctrl-C reached the
    whole foreground process group and the child died with it, which hid this. Started by
    Device Connect there is no process group to rely on, so a stop that does not reach the
    child is a stop that leaves the robot walking.
    """
    script = ("import time;"
              "print('[visual_nav] outcome: stalled: commanded 0.30 m/s', flush=True);"
              "time.sleep(60)")

    def stop_when_it_speaks(_line):
        # The handler is installed before the read loop, so this is delivered to it and
        # not to the test runner.
        signal.raise_signal(signal.SIGTERM)

    started = time.monotonic()
    try:
        supervise([sys.executable, "-c", script], _quiet_voice(),
                  patience_s=1.0, echo=stop_when_it_speaks)
        assert time.monotonic() - started < 20, "the child outlived the stop"
        assert stop_requested(), "the stop was not recorded for the retry loop"
    finally:
        import mission
        mission._STOP.clear()


def test_a_stop_does_not_start_the_next_attempt():
    """Retrying after a stop restarts the robot the stop was issued to halt."""
    marker = Path(tempfile.mkdtemp()) / "n"
    script = (f"import os, pathlib, signal, time;p=pathlib.Path({str(marker)!r});"
              "n=int(p.read_text()) if p.exists() else 0;p.write_text(str(n+1));"
              "print('[visual_nav] outcome: stalled: commanded 0.30 m/s', flush=True);"
              "time.sleep(0.5); os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)")
    rc = main(["--no-voice", "--cooldown", "0", "--max-attempts", "5",
               "--", sys.executable, "-c", script])
    assert rc == 3, "a stopped mission is neither an arrival nor an exhausted retry budget"
    assert marker.read_text() == "1", "it started another attempt after being stopped"


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
