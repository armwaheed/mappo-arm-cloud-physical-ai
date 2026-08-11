#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the telemetry record.

This file is a CONTRACT, not an implementation detail: something downstream parses it,
so the tests are about what a consumer can rely on rather than about how it is written.
The ones that matter are the ones a plausible implementation gets wrong — infinity
serialised as a token strict parsers reject, a truncated file after Ctrl-C, and ticks
that commanded nothing being dropped so the episode silently re-times.

Pure stdlib. Run: ``python3 test_telemetry.py``
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avoidance import Command, Obstacle
from telemetry import SCHEMA, TelemetryWriter

POSE = (1.25, -0.5, 0.3)
GOAL = (3.195, 0.398)


@dataclass
class _Health:
    max_motor_temp_c: float = 37.0
    battery_soc_pct: float = 60.0


def _clock():
    return 1_700_000_000.0


def _writer(directory):
    return TelemetryWriter(os.path.join(directory, "run.jsonl"), clock=_clock)


def _read(directory):
    with open(os.path.join(directory, "run.jsonl"), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tick(writer, **overrides):
    fields = {
        "elapsed_s": 1.5, "pose": POSE, "goal_xy": GOAL, "goal_distance_m": 2.0,
        "command": Command(0.35, -0.2, 0.1, reason="avoid", gap_m=0.8,
                           feasible=200, evaluated=330),
        "obstacles": [Obstacle(x=2.134, y=0.266, vx=0.0, vy=0.0, radius_m=0.23,
                               label="bin")],
        "frame_age_s": 0.31, "perception_seq": 42, "detect_ms": 124.5,
        "standing": True, "live": False, "video_frame": 7, "health": _Health(),
    }
    fields.update(overrides)
    writer.write_tick(**fields)


# ── The contract ────────────────────────────────────────────────────────────
def test_every_line_is_standalone_json():
    """JSONL: a consumer must be able to stream it without loading the whole file."""
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            writer.write_header(live=False)
            _tick(writer)
            writer.write_outcome("arrived (0.80 m from goal)")
        records = _read(directory)
    assert [r["type"] for r in records] == ["header", "tick", "outcome"]


def test_the_header_pins_a_schema_version():
    """A consumer should refuse a major it does not know rather than guess."""
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            writer.write_header(live=True)
        assert _read(directory)[0]["schema"] == SCHEMA
    assert SCHEMA.endswith("/1")


def test_a_tick_carries_the_pose_on_every_line():
    """The whole reason this module exists. The console log printed the pose ONCE, in a
    start-up banner, across 107 control ticks — an observation vector cannot be built
    from that, and the gap is invisible without counting lines."""
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            for index in range(5):
                _tick(writer, elapsed_s=index * 0.1)
        ticks = [r for r in _read(directory) if r["type"] == "tick"]
    assert len(ticks) == 5
    for tick in ticks:
        assert tick["pose"] == {"x": POSE[0], "y": POSE[1], "yaw": POSE[2]}


def test_obstacles_carry_geometry_not_just_a_count():
    """A count cannot become a range vector, and cannot tell a bin from a ghost."""
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer)
        obstacle = _read(directory)[0]["obstacles"][0]
    assert obstacle["label"] == "bin"
    assert abs(obstacle["x"] - 2.134) < 1e-9
    assert abs(obstacle["radius_m"] - 0.23) < 1e-9
    assert obstacle["vx"] == 0.0 and obstacle["vy"] == 0.0


def test_an_infinite_gap_is_null_not_a_bare_Infinity_token():
    """`gap_m` is inf whenever the lane is clear, i.e. on most ticks of a GOOD run.

    json.dump writes bare `Infinity`, which is valid JavaScript and invalid JSON — Python
    round-trips it and most other stacks reject the line. So the common case would hand a
    consumer a file their parser refuses.
    """
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer, command=Command(0.35, 0.0, 0.0, reason="goal",
                                          gap_m=math.inf, feasible=330,
                                          evaluated=330))
        path = os.path.join(directory, "run.jsonl")
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
        record = _read(directory)[0]
    assert "Infinity" not in raw, raw
    assert record["command"]["gap_m"] is None


def test_a_tick_that_commanded_nothing_is_still_written():
    """Holds, stale-perception skips and the goal search are part of the episode.

    A file containing only the interesting ticks silently re-times the whole run, and
    "the robot stood still for 1.4 s" is a training signal, not a gap to elide.
    """
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer, command=None, goal_xy=None, goal_distance_m=None,
                  video_frame=None)
        record = _read(directory)[0]
    assert record["type"] == "tick"
    assert record["command"] is None
    assert record["goal"] is None
    assert record["pose"]["x"] == POSE[0], "pose is still known while blind"


def test_the_video_frame_index_is_the_join_key_to_the_mp4():
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer, video_frame=7)
            _tick(writer, video_frame=None)
        records = _read(directory)
    assert records[0]["perception"]["video_frame"] == 7
    assert records[1]["perception"]["video_frame"] is None


def test_each_line_is_flushed_so_a_killed_run_is_still_readable():
    """Ctrl-C is the NORMAL way one of these ends, and the last few seconds are exactly
    the ones that explain why."""
    with tempfile.TemporaryDirectory() as directory:
        writer = _writer(directory)          # deliberately never closed
        writer.write_header(live=False)
        _tick(writer)
        records = _read(directory)           # read while the handle is still open
    assert len(records) == 2


def test_a_write_failure_does_not_take_down_the_run():
    """The run is the product; this is a record of it."""
    with tempfile.TemporaryDirectory() as directory:
        writer = _writer(directory)
        writer.close()
        writer.write_header(live=False)      # must not raise on a closed handle
        _tick(writer)


def test_the_outcome_line_distinguishes_a_finished_run_from_a_truncated_one():
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer)
            writer.write_outcome("health abort: motor 58C", perception_cycles=140)
        last = _read(directory)[-1]
    assert last["type"] == "outcome"
    assert last["outcome"] == "health abort: motor 58C"
    assert last["perception_cycles"] == 140


def test_records_counts_what_was_written():
    with tempfile.TemporaryDirectory() as directory, _writer(directory) as writer:
        writer.write_header(live=False)
        _tick(writer)
        _tick(writer)
        assert writer.records == 3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"telemetry: {len(tests)}/{len(tests)} passed")
