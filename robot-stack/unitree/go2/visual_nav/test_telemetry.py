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
                               label="bin", kind="static",
                               object_id="landmark-1")],
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


def test_an_obstacle_says_which_subsystem_produced_it():
    """`kind`, not `label`, separates a mapped prop from a tracked mover.

    A consumer has to split the two — a mover is the stop-and-wait logic's job and a
    landmark is something to path around — and every cue short of this field fails on
    the case that matters. `label` is a CLASS name: it worked only while the scene had
    exactly one mapped prop and one detector class. Velocity fails on a person who has
    STOPPED, which is precisely when the distinction decides the behaviour.
    """
    stopped_person = Obstacle(x=3.0, y=0.0, vx=0.0, vy=0.0, radius_m=0.35,
                              label="person", kind="tracked", object_id="track-7")
    landmark = Obstacle(x=2.134, y=0.266, vx=0.0, vy=0.0, radius_m=0.23,
                        label="bin", kind="static", object_id="landmark-1")
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer, obstacles=[stopped_person, landmark])
        obstacles = _read(directory)[0]["obstacles"]
    assert [o["kind"] for o in obstacles] == ["tracked", "static"]
    # The two are indistinguishable by every other field a consumer might reach for.
    assert all(o["vx"] == 0.0 and o["vy"] == 0.0 for o in obstacles)


def test_an_obstacle_carries_a_stable_identity():
    """Without an id a consumer re-associates by position and merges near neighbours.

    The policy package this feeds matches an unidentified object to whatever it already
    holds within 0.45 m, so two props closer than that become one — with the larger of
    their radii, in the average of their positions. Both producers already have an id;
    this only stops it being thrown away.
    """
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer)
        obstacle = _read(directory)[0]["obstacles"][0]
    assert obstacle["id"] == "landmark-1"


def test_an_obstacle_with_no_identity_says_so_rather_than_inventing_one():
    """`None` is a fact the consumer can act on; a synthesised index is a lie that is
    stable within a tick and meaningless across two."""
    anonymous = Obstacle(x=1.0, y=0.0, vx=0.0, vy=0.0, radius_m=0.2)
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer, obstacles=[anonymous])
        assert _read(directory)[0]["obstacles"][0]["id"] is None


def test_the_header_declares_which_frame_every_vector_is_in():
    """The one thing a consumer cannot recover from the data.

    Odom and body agree EXACTLY while the robot faces its start heading and diverge as
    it turns, so an integration that reads `measured` as odom passes every bench test
    and fails in the first corner. An integration note for the MAPPO policy package
    proposed exactly that, because nothing in the file contradicted it.
    """
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            writer.write_header(live=True)
        frames = _read(directory)[0]["frames"]
    assert frames["pose"] == "odom"
    assert frames["obstacles"] == "odom"
    assert frames["measured"] == "body"
    assert frames["command"] == "body"


def test_the_frame_declaration_cannot_be_omitted_by_a_caller():
    """It is a property of the schema, not of a run, so no call site gets to drop it."""
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            writer.write_header()
        assert "frames" in _read(directory)[0]


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


def test_the_measured_velocity_is_recorded_beside_the_command():
    """A run that commands 0.12 m/s and moves nothing reads, in the command alone,
    exactly like a run that is walking. Only this tells them apart."""
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer, measured=(0.0, 0.0, 0.0))
        record = _read(directory)[0]
    assert record["command"]["vx"] == 0.35, "commanded"
    assert record["measured"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}, "achieved"


def test_a_backend_that_cannot_report_velocity_is_not_fatal():
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer, measured=None)
        assert _read(directory)[0]["measured"] is None


def test_a_stale_tick_keeps_its_goal_and_says_it_was_blind():
    """A stale frame means the robot cannot SEE, not that it has forgotten where it is
    going. Recording a null goal there reads downstream as "goal lost"."""
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer, command=None, stale=True)
        record = _read(directory)[0]
    assert record["perception"]["stale"] is True
    assert record["goal"] is not None, "the goal is latched; blindness does not clear it"
    assert record["command"] is None


def test_a_normal_tick_is_not_marked_stale():
    with tempfile.TemporaryDirectory() as directory:
        with _writer(directory) as writer:
            _tick(writer)
        assert _read(directory)[0]["perception"]["stale"] is False


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
