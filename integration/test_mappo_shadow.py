#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the live shadow reader.

The tailing is what gets tested hardest, because it is the part that fails silently: a
reader that drops a half-written line loses exactly the tick where the interesting thing
happened, and the summary still prints a confident number.

Needs the policy package (numpy). Run: ``python3 test_mappo_shadow.py``
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mappo_shadow import follow, shadow

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence" / "sample_telemetry.jsonl"


def _temp(text: str) -> Path:
    """A telemetry file on disk. ``delete=False`` because the point is to reopen it by
    path, the way the reader does; each caller unlinks it in a ``finally``."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        handle.write(text)
        return Path(handle.name)


# ── Tailing ─────────────────────────────────────────────────────────────────
def test_a_finished_file_is_read_to_the_end():
    path = _temp('{"type": "header"}\n{"type": "tick", "t": 1}\n')
    try:
        assert [r["type"] for r in follow(path, False)] == ["header", "tick"]
    finally:
        path.unlink()


def test_a_half_written_final_line_is_not_consumed():
    """The writer is appending at 10 Hz while this reads. Taking half a line and moving
    on drops that tick permanently and reports nothing — so the read position is rewound
    instead, and a non-following reader simply stops before it."""
    path = _temp('{"type": "tick", "t": 1}\n{"type": "tick", "t": 2')
    try:
        assert [r["t"] for r in follow(path, False)] == [1]
    finally:
        path.unlink()


def test_a_line_completed_later_is_picked_up_when_following():
    """The same partial line, finished by the writer a moment later. This is the whole
    point of --follow, and the previous test's rewind is what makes it possible."""
    path = _temp('{"type": "tick", "t": 1}\n{"type": "tick", "t": 2')

    def finish():
        time.sleep(0.2)
        with path.open("a") as handle:
            handle.write('}\n{"type": "tick", "t": 3}\n')

    writer = threading.Thread(target=finish)
    writer.start()
    try:
        assert [r["t"] for r in follow(path, True, idle_timeout_s=1.0)] == [1, 2, 3]
    finally:
        writer.join()
        path.unlink()


def test_following_stops_after_the_writer_goes_quiet():
    path = _temp('{"type": "tick", "t": 1}\n')
    try:
        started = time.monotonic()
        list(follow(path, True, idle_timeout_s=0.3))
        elapsed = time.monotonic() - started
        assert 0.3 <= elapsed < 3.0, elapsed
    finally:
        path.unlink()


def test_an_unparseable_line_is_skipped_rather_than_ending_the_run():
    """A truncated write mid-file should cost one tick, not the rest of the session."""
    path = _temp('{"type": "tick", "t": 1}\nnot json\n{"type": "tick", "t": 2}\n')
    try:
        assert [r["t"] for r in follow(path, False)] == [1, 2]
    finally:
        path.unlink()


# ── The shadow itself ───────────────────────────────────────────────────────
def test_the_recorded_run_shadows_cleanly():
    result = shadow(EVIDENCE, quiet=True)
    assert len(result["rows"]) == 122
    statuses = {row["policy_status"] for row in result["rows"]}
    assert statuses <= {"COMMAND", "STOP_EXTERNAL_HOLD"}, statuses


def test_the_shadow_never_hands_the_policy_a_wall_clock():
    """It stamps ``time.monotonic()`` per tick. Pass the tick's own ``wall_time`` and
    every row comes back STOP_CLOCK_ERROR; pass its ``t`` and the age is meaningless in
    the other direction. Both would be invisible without the guard the policy now has."""
    result = shadow(EVIDENCE, quiet=True)
    assert not any(row["policy_status"] == "STOP_CLOCK_ERROR"
                   for row in result["rows"])


def test_the_written_record_carries_the_observation_and_the_config():
    """An action alone cannot be checked after the fact: the input that produced it is the
    half that can be wrong. The header records the scale, because a shadow file compared
    against one from a different calibration is worse than no comparison."""
    out = Path(tempfile.mkdtemp()) / "shadow.jsonl"
    shadow(EVIDENCE, out=out, quiet=True)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    header, rows = records[0], records[1:]
    assert header["schema"] == "go2.mappo.shadow/1"
    assert header["config"]["meters_per_vmas_unit"] == 2.5
    assert header["config"]["velocity_frame"] == "body"
    assert len(rows) == 122
    assert len(rows[0]["observation"]) == 18
    # Not every tick has one — six of this run's ticks carry no command at all, which is
    # a real state (searching, standing up, stale) and is recorded as such rather than
    # being dropped. What matters is that the column is populated where it exists.
    assert sum(1 for r in rows if r["stack_reason"] is not None) == 116
    out.unlink()
    out.parent.rmdir()


def test_the_shadow_reports_the_stack_command_beside_the_policy_one():
    """The comparison is the deliverable. A file with only the policy's numbers in it
    cannot answer the one question a shadow run is for."""
    result = shadow(EVIDENCE, quiet=True)
    row = result["rows"][10]
    assert row["stack"] != [None, None, None]
    assert row["policy"] != row["stack"]
    assert row["disagreement_deg"] is not None


def test_a_run_with_no_goal_shadows_to_nothing_rather_than_crashing():
    path = _temp('{"type": "header"}\n'
                 '{"type": "tick", "t": 1, "pose": {"x": 0, "y": 0, "yaw": 0}, '
                 '"goal": null, "obstacles": [], "measured": {"vx": 0, "vy": 0}}\n')
    try:
        assert shadow(path, quiet=True)["rows"] == []
    finally:
        path.unlink()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"mappo_shadow: {len(tests)}/{len(tests)} passed")
