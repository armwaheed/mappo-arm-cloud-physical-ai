#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bilingual spoken cues, Chinese first and then English.

WHY CHINESE FIRST, and it is not alphabetical. The demo is in Shanghai, the audience and
the venue staff are Chinese-speaking, and the cue that matters most -- "please step out of
my path" -- is useless if the person it is addressed to has to wait through an English
sentence to hear it. English follows for the visitors.

WHY THIS NEVER RAISES AND NEVER BLOCKS. It is called from a supervisor watching a live
run. A missing WAV, a busy sound card or an absent ``aplay`` must degrade to silence, not
end a run that is otherwise going fine -- losing the demo to a codec is a worse outcome
than losing the announcement. And playback is a SUBPROCESS that is not waited on: the
person-stop cue is 2.7 s of Chinese plus 4.3 s of English, and seven seconds of blocking
would be seven seconds of a robot not reacting to the person it is talking to.

The cue set is borrowed from the RRD DR02 repository (``assets/voice``), which already
carries matched zh/en pairs recorded from one voice. Cues naming a *table* are deliberately
not used here: this demo drives to a marker, and a robot announcing furniture it is not
looking for reads as a bug to anyone who speaks the language.
"""

from __future__ import annotations

import contextlib
import queue
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

#: Cue -> the two files that speak it, Chinese first.
CUES = {
    "person_stop": ("rrd_person_stop_zh.wav", "rrd_person_stop_en.wav"),
    "person_thanks": ("rrd_person_thanks_zh.wav", "rrd_person_thanks_en.wav"),
    "resuming": ("rrd_resuming_zh.wav", "rrd_resuming_en.wav"),
    "fault": ("rrd_fault_zh.wav", "rrd_fault_en.wav"),
    "greeting": ("rrd_greeting_zh.wav", "rrd_greeting_en.wav"),
    # Not from RRD: that set has no cue for "you need to change my posture", and its
    # nearest neighbour (fault) is true but tells nobody what to do about it. Generated
    # to the same 24 kHz mono format so the two sets are interchangeable.
    #   zh  我还没有站起来。请在遥控器上把我切换到站立模式。
    #   en  I am not standing yet. Please set me to standing mode on the controller.
    "stand_request": ("lite3_stand_request_zh.wav", "lite3_stand_request_en.wav"),
}

#: Longest a single utterance may hold the queue before it is abandoned. A cue that will
#: not finish is a cue blocking every cue behind it.
UTTERANCE_TIMEOUT_S = 30.0

#: Utterances allowed to wait. A backlog is worse than a dropped cue: by the time a cue
#: five deep is spoken, the situation it described has gone.
QUEUE_LIMIT = 4

#: A cue may not repeat inside this many seconds. Without it a robot held for a minute by
#: a stationary person says "excuse me" twenty times, which stops reading as politeness.
DEFAULT_REPEAT_GUARD_S = 12.0


class Voice:
    """Plays cue pairs, or silently does nothing when it cannot."""

    def __init__(self, directory: Path | str | None, *, enabled: bool = True,
                 repeat_guard_s: float = DEFAULT_REPEAT_GUARD_S,
                 player: str | None = None, device: str | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        self.device = device
        self._guard = float(repeat_guard_s)
        self._last: dict[str, float] = {}
        self._player = player or shutil.which("aplay") or shutil.which("paplay")
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        # Resolved once, so the reason for silence is reportable rather than mysterious.
        self.reason: str | None = None
        if not enabled:
            self.reason = "disabled"
        elif self.directory is None:
            self.reason = "no voice directory configured"
        elif not self.directory.is_dir():
            self.reason = f"{self.directory} is not a directory"
        elif self._player is None:
            self.reason = "no aplay or paplay on PATH"
        self.enabled = self.reason is None

    def _command(self, path: Path) -> list[str]:
        device = ["-D", self.device] if self.device else []
        return [self._player, *device, "-q", str(path)]

    @staticmethod
    def duration_s(path: Path) -> float:
        """How long the file actually is. Zero when it cannot be read."""
        try:
            with contextlib.closing(wave.open(str(path))) as handle:
                rate = handle.getframerate()
                return handle.getnframes() / rate if rate else 0.0
        except Exception:
            return 0.0

    def _play_one(self, path: Path) -> None:
        """Play ONE file and do not return until it has actually been heard.

        THE PLAYER RETURNING IS NOT THE SOUND ENDING. ``aplay -D pulse`` hands its buffer
        to PulseAudio and exits while the sink is still rendering, so chaining two
        invocations started the English over the top of the Chinese -- audible on the
        robot, and invisible to every exit code. The remaining wall time of the file is
        therefore slept off after the player returns."""
        started = time.monotonic()
        try:
            subprocess.run(self._command(path), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=UTTERANCE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return
        remaining = self.duration_s(path) - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)

    def _run(self) -> None:
        while True:
            path = self._queue.get()
            try:
                if path is None:
                    return
                self._play_one(path)
            finally:
                self._queue.task_done()

    def describe(self) -> str:
        if self.enabled:
            where = self.device or "the player's default device"
            return (f"voice: {self.directory} via {Path(self._player).name} -> {where}, "
                    f"zh then en")
        return f"voice: SILENT ({self.reason})"

    def probe(self, timeout_s: float = 20.0) -> str | None:
        """Actually open the device, and report why not if it will not open.

        THIS EXISTS BECAUSE A ZERO EXIT CODE MEANT NOTHING. On this robot the account
        running the demo was not in the ``audio`` group, so the device could not be
        opened, PulseAudio's default sink had fallen back to ``auto_null`` -- a black
        hole -- and ``aplay`` wrote into it and exited 0. Every check passed and the robot
        made no sound. So this plays a real file and reads STDERR: ALSA reports "audio
        open error" there while still exiting successfully.

        Returns ``None`` when sound will actually come out, else the reason it will not.
        """
        if not self.enabled:
            return self.reason
        sample = next((self.directory / pair[0] for pair in CUES.values()
                       if (self.directory / pair[0]).is_file()), None)
        if sample is None:
            return "no cue files to probe with"
        try:
            # Probe the PLAYER directly rather than through the shell chain: `sh -c`
            # would report the shell's success, not the player's.
            device = ["-D", self.device] if self.device else []
            done = subprocess.run([self._player, *device, "-q", str(sample)],
                                  capture_output=True, text=True, timeout=timeout_s)
        except (OSError, subprocess.SubprocessError) as exc:
            return f"{type(exc).__name__}: {exc}"
        noise = (done.stderr or "").strip()
        if "audio open error" in noise or "No such device" in noise:
            return noise.splitlines()[-1][:120]
        if done.returncode != 0:
            return f"exit {done.returncode}: {noise.splitlines()[-1][:100] if noise else ''}"
        return None

    def missing(self) -> list[str]:
        """Cue files that are configured but absent. Reported at start-up rather than
        discovered when the robot tries to speak and a run is already moving."""
        if not self.enabled:
            return []
        return sorted(name for pair in CUES.values() for name in pair
                      if not (self.directory / name).is_file())

    def say(self, cue: str, *, now: float | None = None) -> bool:
        """Speak ``cue`` if it is due. Returns whether playback was started."""
        if not self.enabled or cue not in CUES:
            return False
        moment = time.monotonic() if now is None else now
        if moment - self._last.get(cue, float("-inf")) < self._guard:
            return False
        files = [self.directory / name for name in CUES[cue]]
        if not all(f.is_file() for f in files):
            return False
        # Dropped rather than queued when a backlog has built: by the time a cue five
        # deep is spoken the situation it described is over, and saying it then is worse
        # than not saying it.
        if self._queue.qsize() >= QUEUE_LIMIT:
            return False
        self._last[cue] = moment
        if self._worker is None or not self._worker.is_alive():
            # Daemon: a stuck sound card must not keep the process alive at exit.
            self._worker = threading.Thread(target=self._run, daemon=True,
                                            name="voice")
            self._worker.start()
        for path in files:
            self._queue.put(path)
        return True

    def pending(self) -> int:
        """Utterances still waiting. One queue, one voice, strictly in order."""
        return self._queue.qsize()

    def close(self, timeout_s: float = 20.0) -> None:
        """Let what is queued finish, then stop the worker."""
        if self._worker is None:
            return
        deadline = time.monotonic() + timeout_s
        while self._queue.qsize() and time.monotonic() < deadline:
            time.sleep(0.1)
        self._queue.put(None)
        self._worker.join(timeout=max(0.5, deadline - time.monotonic()))
