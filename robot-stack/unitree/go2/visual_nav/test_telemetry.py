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

import io
import json
import math
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avoidance import Command, Obstacle
from telemetry import (
    SCHEMA,
    TICK_STAGES,
    TelemetryWriter,
    TickProfiler,
    read_run,
    report,
    summarise,
)

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


# --- the raw measurement, and the crop that produced it ---------------------------

class _Sighting:
    """A RangedDetection's shape, minus the import: what perception measured."""

    class _Box:
        def __init__(self, label, score):
            self.label, self.score = label, score
            self.x1, self.y1, self.x2, self.y2 = 0.0, 746.0, 648.0, 1079.0

    def __init__(self, range_m, bearing_rad, source, label="bin", score=0.77):
        self.range_m, self.bearing_rad, self.source = range_m, bearing_rad, source
        self.detection = self._Box(label, score)


def test_the_raw_sighting_is_recorded_beside_the_fused_estimate():
    """`obstacles` is the MAP's answer, in odom, after fusion. A range recomputed from it
    is the map re-derived, so it cannot audit the map — which is why two open questions
    stalled for want of this field: whether the size-prior range scale is right, and how
    a detection ranged at 0.8 m became a landmark 0.18 m from the robot."""
    with tempfile.TemporaryDirectory() as directory:
        writer = _writer(directory)
        writer.write_header(live=False)
        _tick(writer, sightings=[_Sighting(0.8, 0.457, "frame-fill")])
        tick = _read(directory)[1]

    assert len(tick["sightings"]) == 1
    sighting = tick["sightings"][0]
    assert sighting["range_m"] == 0.8
    assert sighting["source"] == "frame-fill"
    assert sighting["box"] == [0.0, 746.0, 648.0, 1079.0]
    # And the fused estimate is still there, separately. Both, or neither is useful.
    assert tick["obstacles"][0]["id"] == "landmark-1"


def test_a_fabricated_range_is_distinguishable_from_a_measured_one():
    """`source` is the point of the field. A frame-fill reading is a CONSTANT returned
    when the box is clipped on both axes, not a measurement, and a consumer that cannot
    tell it apart from a height-prior range will average the two and believe the
    result."""
    with tempfile.TemporaryDirectory() as directory:
        writer = _writer(directory)
        writer.write_header(live=False)
        _tick(writer, sightings=[_Sighting(2.34, -0.27, "height"),
                                 _Sighting(0.8, 0.457, "frame-fill")])
        sightings = _read(directory)[1]["sightings"]

    assert [s["source"] for s in sightings] == ["height", "frame-fill"]


def test_the_crop_the_goal_pass_actually_used_is_recorded():
    """It is no longer the flag the operator passed — it widens as the goal nears. A
    goal that jumps is the first thing anyone suspects; without this there is no way to
    separate a moving crop from a hopping detection."""
    with tempfile.TemporaryDirectory() as directory:
        writer = _writer(directory)
        writer.write_header(live=False)
        _tick(writer, goal_crop=0.89)
        assert _read(directory)[1]["perception"]["goal_crop"] == 0.89


def test_a_goal_source_without_a_crop_records_null_rather_than_failing():
    """An ArUco marker or a fixed waypoint has no crop. A telemetry field must never be
    the thing that ends a run."""
    with tempfile.TemporaryDirectory() as directory:
        writer = _writer(directory)
        writer.write_header(live=False)
        _tick(writer)
        assert _read(directory)[1]["perception"]["goal_crop"] is None
        assert _read(directory)[1]["sightings"] == []


# ── Where the tick went ─────────────────────────────────────────────────────
class _Clock:
    """A monotonic clock that only moves when told, in seconds."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, milliseconds: float) -> None:
        self.now += milliseconds / 1000.0


def test_the_profile_names_every_stage_including_the_ones_that_did_not_run():
    """A stale-perception hold runs no planner and records no frame. Emitting only the
    stages that fired makes every consumer decide for itself whether a missing key means
    "free" or "did not happen", and they will not all decide the same way."""
    clock = _Clock()
    profiler = TickProfiler(clock)
    profiler.begin()
    with profiler.stage("plan"):
        clock.advance(40.0)
    stages = profiler.snapshot()["stages"]
    assert sorted(stages) == sorted(TICK_STAGES), stages
    assert stages["plan"] == 40.0
    assert stages["record"] == 0.0 and stages["command"] == 0.0


def test_a_stage_entered_twice_in_one_tick_adds_rather_than_replaces():
    """`perceive` is entered on both sides of the `result is None` branch. Replacing
    would report whichever half ran last and silently halve the stage."""
    clock = _Clock()
    profiler = TickProfiler(clock)
    profiler.begin()
    with profiler.stage("perceive"):
        clock.advance(3.0)
    with profiler.stage("perceive"):
        clock.advance(7.0)
    assert profiler.snapshot()["stages"]["perceive"] == 10.0


def test_a_stage_name_that_is_not_in_the_list_is_refused():
    """A typo would otherwise create a stage nothing sums and nothing prints — a hole in
    the accounting that reads as a fast tick."""
    profiler = TickProfiler(_Clock())
    profiler.begin()
    try:
        with profiler.stage("planner"):
            pass
    except KeyError as error:
        assert "planner" in str(error), error
    else:
        raise AssertionError("an unknown stage name should be refused")


def test_tick_ms_is_measured_from_the_instant_the_caller_names():
    """The loop's own `tick_start`, so `tick_ms` and the trailing sleep are measured
    against the same instant. Taking a second timestamp inside `begin` would put the gap
    between them into neither."""
    clock = _Clock()
    profiler = TickProfiler(clock)
    started = clock.now
    clock.advance(5.0)                    # the loop did something before calling begin
    profiler.begin(started)
    clock.advance(20.0)
    assert profiler.snapshot()["tick_ms"] == 25.0


def test_the_unaccounted_remainder_is_what_no_stage_covered():
    """A large `other_ms` means the stage list has a hole in it, which is the failure
    mode of every hand-placed profiler. It has to be visible, not absorbed."""
    clock = _Clock()
    profiler = TickProfiler(clock)
    profiler.begin()
    with profiler.stage("plan"):
        clock.advance(30.0)
    clock.advance(12.0)                   # something the loop does that no stage wraps
    snapshot = profiler.snapshot()
    assert snapshot["tick_ms"] == 42.0
    assert snapshot["other_ms"] == 12.0


def test_the_telemetry_write_is_priced_on_the_following_tick():
    """A record cannot contain the time it took to write itself, and the field name has
    to say so rather than quietly attributing it to the wrong tick."""
    clock = _Clock()
    profiler = TickProfiler(clock)
    profiler.begin()
    assert profiler.snapshot()["write_prev_ms"] == 0.0
    profiler.wrote(1.75)
    profiler.begin()
    assert profiler.snapshot()["write_prev_ms"] == 1.75


def test_the_profile_reaches_the_record_and_is_null_without_one():
    """A file written by anything that does not profile itself must still parse. Every
    committed run predates this and carries `"profile": null`."""
    with tempfile.TemporaryDirectory() as directory:
        writer = _writer(directory)
        writer.write_header(live=False)
        profiler = TickProfiler(_Clock())
        profiler.begin()
        _tick(writer, profile=profiler.snapshot())
        _tick(writer)
        records = _read(directory)
    assert records[1]["profile"]["stages"]["plan"] == 0.0
    assert records[1]["profile"]["tick_ms"] == 0.0
    assert records[2]["profile"] is None


def test_the_perception_cycle_and_its_wait_sit_beside_detect_ms():
    """`detect_ms` covers the goal pass, the tiered detect and colour segmentation and
    nothing else, so on its own it cannot say whether that thread is compute-bound or
    camera-bound. Both are optional: a caller that cannot measure them writes null rather
    than a zero that reads as "instant"."""
    with tempfile.TemporaryDirectory() as directory:
        writer = _writer(directory)
        writer.write_header(live=False)
        _tick(writer, cycle_ms=311.2, wait_ms=104.9)
        _tick(writer)
        records = _read(directory)
    assert records[1]["perception"]["cycle_ms"] == 311.2
    assert records[1]["perception"]["wait_ms"] == 104.9
    assert records[2]["perception"]["cycle_ms"] is None
    assert records[2]["perception"]["wait_ms"] is None


# ── Reading one back ────────────────────────────────────────────────────────
def _run_file(directory, ticks, header=None):
    path = os.path.join(directory, "read.jsonl")
    lines = [json.dumps({"type": "header", "control_hz": 10.0, **(header or {})})]
    lines += [json.dumps(tick) for tick in ticks]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _synthetic_tick(t, video_frame, profile=None, stale=False):
    return {"type": "tick", "t": t, "profile": profile,
            "perception": {"seq": 1, "video_frame": video_frame, "detect_ms": 200.0,
                           "frame_age_s": 0.3, "stale": stale, "cycle_ms": None,
                           "wait_ms": None}}


def test_summarise_takes_the_rate_from_the_tick_times():
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_synthetic_tick(0.0, 0), _synthetic_tick(0.25, 1),
                                     _synthetic_tick(0.5, 2)])
        header, ticks = read_run(path)
    summary = summarise(header, ticks)
    assert summary["ticks"] == 3
    assert math.isclose(summary["measured_hz"], 4.0)
    assert summary["configured_hz"] == 10.0
    assert summary["interval_ms"] == [250.0, 250.0]


def test_a_truncated_last_line_costs_one_tick_and_not_the_run():
    """The normal way one of these ends is Ctrl-C or a safety abort, and that is exactly
    the window that explains why the run ended."""
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_synthetic_tick(0.0, 0)])
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"type": "tick", "t": 0.1, "percep')
        _header, ticks = read_run(path)
    assert len(ticks) == 1


def test_a_file_with_no_profile_is_split_by_the_recorder_gate_instead():
    """What the 26 committed runs can still be asked. The recording gate and the
    perception-consumption gate are the SAME gate, so this cannot separate the MP4 encode
    from the tracker update — and the report has to say so rather than name a cause."""
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_synthetic_tick(0.0, 0), _synthetic_tick(0.25, None),
                                     _synthetic_tick(0.35, 1), _synthetic_tick(0.60, None),
                                     _synthetic_tick(0.70, 2)])
        header, ticks = read_run(path)
    out = io.StringIO()
    assert report(header, ticks, out=out) == 0
    text = out.getvalue()
    assert "no per-stage profile" in text
    assert "wrote a frame" in text and "wrote none" in text
    assert "150.0 ms difference" in text, text
    assert "It cannot say which." in text


def test_a_file_with_a_profile_is_reported_stage_by_stage():
    profile = {"tick_ms": 250.0, "other_ms": 2.0, "write_prev_ms": 0.4,
               "stages": {"perceive": 4.0, "obstacles": 1.0, "plan": 98.0,
                          "command": 0.5, "record": 144.5}}
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_synthetic_tick(0.0, 0, profile),
                                     _synthetic_tick(0.25, 1, profile)])
        header, ticks = read_run(path)
    out = io.StringIO()
    assert report(header, ticks, out=out) == 0
    text = out.getvalue()
    assert "no per-stage profile" not in text
    for name in TICK_STAGES:
        assert name in text, name
    assert "144.5" in text and "98.0" in text


def test_the_committed_hero_run_still_measures_what_issue_18_recorded():
    """The number this instrumentation exists for, pinned against the file it came from.

    3.15 Hz against a declared 10 Hz, and a median tick of 250.9 ms against a 100 ms
    period. The split is the finding: the 57 ticks that wrote a video frame took 251.4 ms
    and the one that wrote none took 100.3 ms — the configured period — while `detect_ms`
    on the same ticks was 201.8 ms on a thread that is not this one.
    """
    path = (Path(os.path.abspath(__file__)).parents[4]
            / "evidence" / "2026-08-25-peer-runs" / "hero-run-telemetry.jsonl")
    header, ticks = read_run(path)
    summary = summarise(header, ticks)
    assert summary["ticks"] == 59 and summary["configured_hz"] == 10.0
    assert round(summary["measured_hz"], 2) == 3.15
    assert round(statistics.median(summary["interval_ms"]), 1) == 250.9
    assert round(statistics.median(summary["recorded_ms"]), 1) == 251.4
    assert [round(x, 1) for x in summary["plain_ms"]] == [100.3]
    assert round(statistics.median(summary["detect_ms"]), 1) == 201.8
    assert summary["profiles"] == [], "this run predates the profiler"


def test_the_committed_dry_run_holds_the_design_rate_with_the_higher_detect_ms():
    """The control. `dryrun-corridor-scene-check` passes no `--record`, holds 9.86 Hz for
    195 ticks, and carries a HIGHER median `detect_ms` (262.4 ms) than either live run —
    which is what rules detection out of the control tick rather than assuming it out."""
    path = (Path(os.path.abspath(__file__)).parents[4]
            / "evidence" / "2026-08-17-corridor-and-room-runs"
            / "dryrun-corridor-scene-check.jsonl")
    header, ticks = read_run(path)
    summary = summarise(header, ticks)
    assert round(summary["measured_hz"], 2) == 9.86
    assert round(statistics.median(summary["interval_ms"]), 1) == 100.7
    assert round(statistics.median(summary["detect_ms"]), 1) == 262.4
    assert summary["recorded_ms"] == [], "this run recorded no video"


def test_every_ranging_source_is_named_here():
    """⚠️ THE COMMENT BESIDE `sightings[].source` ENUMERATES A SET THAT KEEPS GROWING.

    `estimate_range`'s own docstring has said "height", "width" or "frame-fill" since it was
    written, while its code also returns "width-capped"; then `GroundRanger` added four more
    in one PR. The value matters to a consumer — "frame-fill" and "width-capped" are
    CONSTANTS and not measurements — so a new one appearing with nothing saying so is a
    consumer silently weighting a constant as a reading.

    So this walks the producer rather than trusting prose. The positive control is the
    literal count: with a broken pattern the assertion below would pass over an empty set
    forever, which is the shape of gate this repository has already shipped once.
    """
    source = (Path(os.path.dirname(os.path.abspath(__file__))) / "person_detector.py"
              ).read_text()
    produced = set(re.findall(r'return\s+[^,\n]+,\s*"([a-z][a-z-]*)"', source))
    assert len(produced) >= 8, f"the pattern found only {sorted(produced)}"
    comment = telemetry_source_comment()
    missing = sorted(name for name in produced if f'"{name}"' not in comment)
    assert not missing, (
        f"person_detector returns ranging source(s) {missing} that the comment beside "
        f"`sightings[].source` in telemetry.py does not name. A consumer cannot tell a "
        f"constant from a measurement without that list.")


def telemetry_source_comment() -> str:
    """The comment block above ``"sightings"`` in ``write_tick``, as text."""
    lines = (Path(os.path.dirname(os.path.abspath(__file__))) / "telemetry.py"
             ).read_text().splitlines()
    end = next(i for i, line in enumerate(lines) if '"sightings": [{' in line)
    block = []
    for line in reversed(lines[:end]):
        if not line.strip().startswith("#"):
            break
        block.append(line)
    assert len(block) > 5, "the comment block above `sightings` has gone"
    return "\n".join(block)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"telemetry: {len(tests)}/{len(tests)} passed")
