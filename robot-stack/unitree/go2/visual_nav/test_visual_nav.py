#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the navigator's own glue — no robot, no camera.

``avoidance`` and ``tracker`` are each tested in isolation, but the seam between them
is :meth:`VisualNavigator._obstacles`, and that is where the belief the planner acts on
is actually assembled. It has its own arithmetic — extrapolate to now, inflate for
uncertainty — that neither module's tests can see.

The tests below pin the one property that arithmetic must have: it extrapolates across
the PERCEPTION LATENCY, not across the age of the track. Getting that wrong integrates
a coasting track's uncertainty twice, and a person who walked out of shot turns into a
no-go disc metres across (see the method's docstring).

Run: ``python3 test_visual_nav.py``
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# The routing seam has two halves in two directories: this package decides
# `person_shaped` and writes it, and `integration/mappo_bridge.holds_the_robot` acts on
# it. Nothing else in this suite reaches across, and it is deliberate here — see
# `test_a_peer_reaches_the_policy_and_a_person_holds_end_to_end` for what a test on one
# side alone failed to catch. ⚠️ Both inserts must stay ABOVE the sibling imports below:
# `ruff --fix` will hoist them under it if they are separated (AGENTS.md).
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "..", "integration"))
import inspect

import numpy as np

# Not a sibling of this directory — it lives in `integration/`, which the second
# sys.path line above puts on the path. Its own block because that is where ruff's
# isort puts a non-first-party import, and this file is linted, not auto-fixed.
from mappo_bridge import holds_the_robot

import person_detector
import visual_nav
from avoidance import (
    STATIC_SOFT_GAP_M,
    Command,
    DynamicWindowPlanner,
    PlannerConfig,
)
from camera import Frame
from camera_model import GO2_CAMERA_HEIGHT_M, FisheyeCamera
from colour_detector import PROFILES
from goal import ArucoGoal, OdomWaypoint
from person_detector import PERSON_ASPECT_MIN, Detection, RangedDetection
from static_map import StaticObstacleMap
from telemetry import TICK_STAGES, TelemetryWriter
from tracker import (
    PROCESS_ACCEL_SIGMA,
    Observation,
    ObstacleTracker,
    observation_from,
)
from visual_nav import (
    STATIC_DETECT_LABEL,
    NavConfig,
    PerceptionResult,
    PerceptionWorker,
    VisualNavigator,
    blocked_stop,
    build_camera_model,
    build_parser,
    static_detect_prior,
)

CAMERA = FisheyeCamera.from_hfov(1920, 1080, 85.27)

PERCEPTION_DT = 1.0 / 7.0
LATENCY_S = 0.16          # a typical age for the newest consumed perception result


def _tracked_person(bearing_deg: float = 20.0, range_m: float = 4.0,
                    sightings: int = 3) -> tuple[ObstacleTracker, float]:
    """A confirmed track, plus the filter time it has been advanced to."""
    tracker = ObstacleTracker(fov_rad=math.radians(85.27))
    now = 0.0
    for _ in range(sightings):
        tracker.predict(PERCEPTION_DT)
        tracker.update([Observation.from_bearing_range(
            math.radians(bearing_deg), range_m, 0.0, 0.0, 0.0)], now, 0.0, 0.0, 0.0)
        now += PERCEPTION_DT
    return tracker, now


def _walking_person(speed_m_s: float = 1.0, sightings: int = 6
                    ) -> tuple[ObstacleTracker, float]:
    """A confirmed track with a real estimated velocity, crossing left to right.

    A stationary person cannot tell a latency-sized extrapolation from a coast-sized
    one — both advance the position by nothing — so anything testing the extrapolated
    POSITION needs a target that is actually moving.
    """
    tracker = ObstacleTracker(fov_rad=math.radians(85.27))
    now = 0.0
    for step in range(sightings):
        x, y = 4.0, -1.0 + speed_m_s * step * PERCEPTION_DT
        tracker.predict(PERCEPTION_DT)
        tracker.update([Observation.from_bearing_range(
            math.atan2(y, x), math.hypot(x, y), 0.0, 0.0, 0.0)], now, 0.0, 0.0, 0.0)
        now += PERCEPTION_DT
    return tracker, now


def _coast(tracker: ObstacleTracker, filter_time: float, seconds: float) -> float:
    """Advance the filter with the track OUT OF SHOT, as the navigator's loop does.

    Out of shot matters: an in-view track that goes unmatched accrues misses and is
    deleted within MAX_MISSES, so only a track the camera cannot see reaches the full
    coast timeout — which is exactly the case the inflation arithmetic gets wrong.
    """
    steps = round(seconds / PERCEPTION_DT)
    for _ in range(steps):
        tracker.predict(PERCEPTION_DT)
        tracker.update([], filter_time, 0.0, 0.0, math.radians(-70.0))
        filter_time += PERCEPTION_DT
    return filter_time


def _navigator(tracker: ObstacleTracker, filter_time: float,
               static_map=None, expansion=None) -> VisualNavigator:
    """A navigator wired to nothing but the planner and tracker ``_obstacles`` reads.

    ``_tracker_time`` is set directly because the loop that normally maintains it is
    the loop under test; everything else _obstacles touches is passed in properly.
    """
    navigator = VisualNavigator(
        loco=None, perception=None,
        planner=DynamicWindowPlanner(config=PlannerConfig()),
        tracker=tracker, goal_source=None, health=None, config=NavConfig(),
        static_map=static_map, expansion=expansion)
    navigator._tracker_time = filter_time
    return navigator


def test_inflation_covers_the_perception_latency():
    """A freshly seen track is inflated by the latency, and barely at all."""
    tracker, filter_time = _tracked_person()
    navigator = _navigator(tracker, filter_time)
    track = tracker.confirmed_tracks()[0]

    obstacle = navigator._obstacles(filter_time + LATENCY_S)[0]
    expected = 0.5 * PROCESS_ACCEL_SIGMA * LATENCY_S ** 2
    inflation = (obstacle.radius_m - PlannerConfig().obstacle_radius_m
                 - track.position_sigma)
    assert abs(inflation - expected) < 1e-9, inflation
    assert inflation < 0.02, f"{LATENCY_S}s of latency should be centimetres: {inflation}"


def test_a_coasting_track_is_not_extrapolated_twice():
    """The regression: inflation must not grow with the AGE of the track.

    predict() has already advanced this track and grown its covariance across the
    whole coast, so re-extrapolating from ``last_seen`` would count that interval a
    second time — quadratically.
    """
    tracker, filter_time = _tracked_person()
    filter_time = _coast(tracker, filter_time, seconds=2.0)
    navigator = _navigator(tracker, filter_time)
    track = tracker.confirmed_tracks()[0]
    assert filter_time - track.last_seen > 1.5, "the track should have coasted"

    obstacle = navigator._obstacles(filter_time + LATENCY_S)[0]
    inflation = (obstacle.radius_m - PlannerConfig().obstacle_radius_m
                 - track.position_sigma)
    # Latency-sized, not coast-sized. Extrapolating the ~2.2 s since last_seen would
    # put this at ~2.4 m instead.
    assert inflation < 0.02, f"coast age leaked into the inflation: {inflation:.2f} m"


def test_position_is_advanced_by_the_latency_not_the_coast():
    """Same error, seen in the position rather than the radius.

    Needs a MOVING target: extrapolating a stationary track by 0.16 s and by 2.2 s
    both land in the same place, so a still person cannot catch this.
    """
    tracker, filter_time = _walking_person(speed_m_s=1.0)
    filter_time = _coast(tracker, filter_time, seconds=2.0)
    navigator = _navigator(tracker, filter_time)
    track = tracker.confirmed_tracks()[0]
    speed = math.hypot(track.state[2], track.state[3])
    assert speed > 0.5, f"the track should be moving to make this test bite: {speed}"

    obstacle = navigator._obstacles(filter_time + LATENCY_S)[0]
    travelled = math.hypot(obstacle.x - track.state[0], obstacle.y - track.state[1])
    assert abs(travelled - speed * LATENCY_S) < 1e-9, travelled
    # Extrapolating the ~2.2 s since last_seen would move it more than a metre.
    assert travelled < 0.25, f"coast age leaked into the position: {travelled:.2f} m"


def test_obstacles_are_only_the_confirmed_tracks():
    """One sighting is not yet an obstacle — CONFIRM_HITS gates what the planner sees."""
    tracker, filter_time = _tracked_person(sightings=1)
    navigator = _navigator(tracker, filter_time)
    assert tracker.tracks, "a track should exist"
    assert navigator._obstacles(filter_time + LATENCY_S) == []


class _RejectAll:
    """Stand-in for ``ExpansionConsistency`` that rejects whatever it is asked about.

    A real gate needs a walking robot to say anything, and this test is about the
    WIRING: does a rejection reach the planner's obstacle list, and does the track
    survive it? ``expansion.py``'s own suite covers whether the verdict is right.
    """

    def __init__(self, rejected=None):
        self.rejected = rejected
        self.asked = []

    def rejects(self, track_ids):
        ids = list(track_ids)
        self.asked.append(ids)
        return set(ids) if self.rejected is None else set(self.rejected) & set(ids)


def test_a_rejected_track_is_withheld_from_the_planner():
    """The one behaviour the gate buys. Without it the whole module is an opinion
    nothing reads."""
    tracker, filter_time = _tracked_person()
    gate = _RejectAll()
    navigator = _navigator(tracker, filter_time, expansion=gate)
    assert tracker.confirmed_tracks(), "there is a confirmed track to withhold"
    assert navigator._obstacles(filter_time + LATENCY_S) == []
    assert gate.asked and gate.asked[0], "the gate was never consulted"


def test_a_rejected_track_is_withheld_and_NOT_deleted():
    """⚠️ Withholding and deleting are different, and only one of them is safe.

    A deleted track loses its hits, its velocity and its window, and the very next
    detection of the same box spawns a NEW id with a clean sheet — which re-confirms in
    CONFIRM_HITS frames and is planned against again. That create/destroy loop is the
    bug ``static_map``'s MAX_MISSES was written to break. Keeping the track means the
    evidence keeps accumulating and a wrong rejection can be undone."""
    tracker, filter_time = _tracked_person()
    before = [t.track_id for t in tracker.confirmed_tracks()]
    navigator = _navigator(tracker, filter_time, expansion=_RejectAll())
    navigator._obstacles(filter_time + LATENCY_S)
    after = [t.track_id for t in tracker.confirmed_tracks()]
    assert after == before, f"the tracker lost {set(before) - set(after)}"


def test_without_a_gate_nothing_changes():
    """The flag is off by default, so the no-gate path is the deployed one and has to
    be byte-for-byte the behaviour that shipped."""
    tracker, filter_time = _tracked_person()
    plain = _navigator(tracker, filter_time)._obstacles(filter_time + LATENCY_S)
    tracker2, filter_time2 = _tracked_person()
    kept = _navigator(tracker2, filter_time2,
                      expansion=_RejectAll(rejected=[]))._obstacles(
                          filter_time2 + LATENCY_S)
    assert len(plain) == 1 and len(kept) == 1
    assert plain[0].object_id == kept[0].object_id
    assert abs(plain[0].radius_m - kept[0].radius_m) < 1e-12


def _coast_budget_s(range_m: float, bearing_deg: float = 20.0) -> float:
    """How long the robot keeps MOVING after losing sight of a person at ``range_m``.

    Drives the real _obstacles() and the real planner, with the goal 2 m beyond the
    person so there is always a reason to keep going.

    Measures ``is_stop``, not ``reason == "hold"``. Those used to differ: a full-stop
    row was appended to every candidate window and competed on cost, and since a
    stationary rollout approaches nothing its clearance cost is zero by construction —
    so near an obstacle the planner returned v=(0,0,0) while still reporting
    ``reason="goal"``. Timing the label rather than the wheels made this budget read
    ~50% longer than the robot actually managed, in the over-promising direction.
    """
    tracker, filter_time = _tracked_person(bearing_deg=bearing_deg, range_m=range_m)
    navigator = _navigator(tracker, filter_time)
    planner = DynamicWindowPlanner(config=PlannerConfig())
    goal = (range_m + 2.0, 0.0)
    while tracker.confirmed_tracks():
        navigator._tracker_time = filter_time
        control_now = filter_time + LATENCY_S
        plan = planner.plan((0.0, 0.0, 0.0), goal, (0.30, 0.0, 0.0),
                            navigator._obstacles(control_now), 0.1)
        if plan.is_stop:
            return control_now - tracker.confirmed_tracks()[0].last_seen
        filter_time = _coast(tracker, filter_time, PERCEPTION_DT)
    return math.inf


def test_the_documented_coast_budget_still_holds():
    """Pins the table in the module docstring, so prose and behaviour cannot drift.

    Tolerances are loose because these are operating limits, not contracts — the point
    is that the numbers a reader plans around are the numbers the code produces.
    """
    for range_m, documented_s in ((1.0, 0.30), (2.0, 0.59), (4.0, 1.59), (6.0, 2.45)):
        measured = _coast_budget_s(range_m)
        assert abs(measured - documented_s) < 0.2, (range_m, measured, documented_s)


def test_the_coast_budget_is_shortest_for_the_CLOSEST_person():
    """The limitation is time-shaped, not a maximum range — pin the direction.

    It reads backwards, so it is worth stating outright: a person lost at 1 m parks the
    robot sooner than one lost at 6 m, because a smaller gap is swallowed sooner by the
    same growing radius. Anyone who "fixes" this with a maximum-range cut-off, or who
    writes down an effective range of N metres, will fail here.
    """
    budgets = [_coast_budget_s(r) for r in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    assert budgets == sorted(budgets), budgets
    assert budgets[0] < budgets[-1] / 2.0, budgets


def test_a_person_in_view_is_handled_across_the_detector_band():
    """No range-shaped degradation while the person is actually visible.

    Stopping for someone 1 m dead ahead is the correct answer, so the assertion is
    about what happens further out, where the robot should still be making progress.
    """
    planner = DynamicWindowPlanner(config=PlannerConfig())
    for range_m in (3.0, 4.0, 5.0, 6.0):
        tracker, filter_time = _tracked_person(bearing_deg=0.0, range_m=range_m,
                                               sightings=8)
        navigator = _navigator(tracker, filter_time)
        plan = planner.plan((0.0, 0.0, 0.0), (range_m + 2.0, 0.0), (0.30, 0.0, 0.0),
                            navigator._obstacles(filter_time + LATENCY_S), 0.1)
        assert plan.vx > 0.0, (range_m, plan)


# ── The control loop itself ─────────────────────────────────────────────────
class _FakePose:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw


class _FakeLoco:
    """Records every command, and how long the robot had been standing when it came."""

    def __init__(self, blocked_after: int | None = None):
        self.commands: list = []
        self.calls: list = []
        #: Tick count after which is_blocked starts reporting a stall, or None.
        self._blocked_after = blocked_after
        self.blocked_queries: list = []

    def pose(self):
        return _FakePose()

    def is_blocked(self, commanded_vx):
        """Stands in for the real stall gate, which integrates over consecutive ticks."""
        self.blocked_queries.append(commanded_vx)
        if self._blocked_after is None or commanded_vx <= 0.05:
            return None
        if len(self.blocked_queries) > self._blocked_after:
            return "stalled"
        return None

    def recover(self):
        self.calls.append("recover")

    def stand(self):
        self.calls.append("stand")

    def stand_down(self):
        self.calls.append("stand_down")

    def stop(self):
        self.calls.append("stop")

    def set_velocity(self, vx, vy, wz):
        self.commands.append((round(vx, 3), round(vy, 3), round(wz, 3)))


class _FakePerception:
    """Always has a fresh, empty result — the loop under test is the navigator's."""

    cycles, errors = 1, 0

    def __init__(self):
        self.result = PerceptionResult(seq=1, capture_time=time.monotonic(),
                                       pose=(0.0, 0.0, 0.0), observations=[], ranged=[])

    def alive(self):
        return True

    def latest(self):
        # Re-stamp so the frame never trips the staleness check while the fake
        # stand-up burns wall-clock.
        return PerceptionResult(seq=1, capture_time=time.monotonic(),
                                pose=(0.0, 0.0, 0.0), observations=[], ranged=[])


class _FakeHealth:
    """Healthy for ``ticks`` calls, then aborts — a deterministic end to the loop."""

    def __init__(self, ticks=4):
        self.remaining = ticks

    def abort_reason(self):
        self.remaining -= 1
        return None if self.remaining > 0 else "test over"

    def latest(self):
        return None


class _FakeGoal:
    description = "test goal"

    def goal_xy(self):
        return (10.0, 0.0)


class _FakePlanner:
    """Always wants to drive forward, so the navigator always wants to be standing."""

    config = PlannerConfig()

    def plan(self, *_args, **_kwargs):
        return Command(vx=0.30, vy=0.0, wz=0.0, reason="goal", gap_m=math.inf)


def _navigator_with(loco, live=True, ticks=6) -> VisualNavigator:
    """A navigator around a caller-supplied loco, for the stall tests."""
    return VisualNavigator(
        loco=loco, perception=_FakePerception(), planner=_FakePlanner(),
        tracker=ObstacleTracker(), goal_source=_FakeGoal(), health=_FakeHealth(ticks),
        config=NavConfig(live=live, control_hz=100.0))


def _live_navigator(ticks=4) -> tuple[VisualNavigator, _FakeLoco]:
    loco = _FakeLoco()
    navigator = VisualNavigator(
        loco=loco, perception=_FakePerception(), planner=_FakePlanner(),
        tracker=ObstacleTracker(), goal_source=_FakeGoal(), health=_FakeHealth(ticks),
        config=NavConfig(live=True, control_hz=100.0))
    return navigator, loco


def test_the_tick_that_stands_up_does_not_issue_its_stale_plan():
    """Standing blocks the loop for ~3 s; the plan made before it must be discarded.

    ``_stand_up`` sleeps through a RecoveryStand and a BalanceStand, so by the time it
    returns, the command computed at the top of that tick was decided three seconds ago
    — while the robot was still lying down, and while the person it is walking past was
    three seconds further back. Issuing it is the one moment in a run where the
    commanded velocity is guaranteed stale, and it lands the instant the legs become
    able to act on it.

    The fix gives up the tick and re-plans, so the FIRST command after standing is a
    stop and the second is a freshly planned one.
    """
    navigator, loco = _live_navigator()
    navigator.run()

    assert loco.calls[:2] == ["recover", "stand"], loco.calls
    assert loco.commands, "the navigator should have commanded something"
    assert loco.commands[0] == (0.0, 0.0, 0.0), (
        f"the stand-up tick issued a stale plan: {loco.commands[0]}")
    assert any(c[0] > 0.0 for c in loco.commands[1:]), (
        f"the navigator should drive on a LATER tick: {loco.commands}")


def test_a_detector_that_throws_does_not_kill_perception():
    """A failed cycle must be survivable, and must be visible.

    An unguarded throw killed the thread outright: ``latest()`` then returned the last
    good result forever and the run ended as a plain timeout, with nothing anywhere
    saying perception had stopped 80 s earlier.
    """

    class _Boom:
        def detect(self, _image):
            raise RuntimeError("detector exploded")

        def detect_tiered(self, _image):
            # The worker calls THIS. Without it the test passes on an AttributeError,
            # i.e. it proves the method is missing rather than that a throw survives.
            raise RuntimeError("detector exploded")

    class _OneFrameCamera:
        def __init__(self):
            self.seq = 0

        def wait_for_new(self, after_seq, timeout):
            self.seq += 1
            return Frame(image=None, capture_time=time.monotonic(), seq=self.seq,
                         stamp=(0.0, 0.0, 0.0))

    class _NoGoal:
        def update(self, _image, _pose):
            return None

    worker = PerceptionWorker(_OneFrameCamera(), _Boom(), None, _NoGoal(),
                              lambda: (0.0, 0.0, 0.0))
    worker.start()
    deadline = time.monotonic() + 2.0
    while worker.errors == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    still_alive = worker.alive()
    worker.stop()

    assert worker.errors > 0, "the failure should have been counted"
    assert still_alive, "the perception thread died on a detector exception"


def test_latching_the_arm_is_on_by_default():
    """The D1 latch is a hard requirement, so it must not be an opt-in flag.

    An unpowered D1 back-drives and 3.15 kg off the dorsal centreline unbalances the
    vendor gait controller, so the default has to be "locked" and the escape hatch has
    to be the thing you type deliberately.
    """
    assert NavConfig().latch_arm is True
    parser = build_parser()
    assert parser.parse_args([]).no_latch_arm is False
    assert parser.parse_args(["--no-latch-arm"]).no_latch_arm is True


# ── The substitution seam ───────────────────────────────────────────────────
def test_main_accepts_a_planner_factory_and_defaults_to_the_shipped_planner():
    """A CONTRACT TRIPWIRE, and worth being honest about what it is: `main()` connects to
    a robot, so what it does with the factory cannot be tested here. What can be tested
    is that the seam exists and has not been renamed, which is the failure a downstream
    consumer actually suffers — and suffers silently, if they were reaching around it.

    The seam exists because everything in `main()` around that one line is what makes a
    run safe: the arm-latch refusal, the health gate, the recorder's codec check, and a
    `finally` whose ordering matters. A consumer who copies it will drift, and the drift
    will be in the arm check.
    """
    parameters = inspect.signature(visual_nav.main).parameters
    assert "planner_factory" in parameters, "the substitution seam has gone"
    assert parameters["planner_factory"].default is DynamicWindowPlanner, \
        "the default must stay the shipped planner — every existing caller relies on it"


def test_main_takes_argv_so_a_wrapper_does_not_have_to_mutate_sys_argv():
    """The other half of the seam. Without it a consumer has to rewrite `sys.argv` before
    calling in, which breaks `--help` and makes the real argument list unrecoverable from
    a crash."""
    parameters = inspect.signature(visual_nav.main).parameters
    assert "argv" in parameters
    assert parameters["argv"].default is None, "no argv must still mean sys.argv"


def test_the_factory_is_called_with_the_keywords_it_is_documented_with():
    """`planner_factory(limits=..., config=...)`. Pinned by reading the call site,
    because a positional call here would silently break every factory written against
    the documented signature — and the docstring is the contract."""
    source = inspect.getsource(visual_nav.main)
    assert "planner_factory(limits=limits, config=planner_config)" in source


# ── CLI resolution ──────────────────────────────────────────────────────────
def _args(**overrides):
    args = visual_nav.build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_the_static_prop_override_does_not_mutate_the_shared_profile():
    """PROFILES is module-level. A run that edited the bin's height in place would
    change every LATER run in the same process, and the symptom is ranges being wrong
    in the next session rather than this one."""
    before = PROFILES["bin"].height_m
    adjusted = visual_nav.static_profile(_args(static_prop="bin", prop_height=0.5))
    assert adjusted.height_m == 0.5
    assert PROFILES["bin"].height_m == before


def test_overriding_the_prop_height_carries_the_width_with_it():
    """The width is the fallback prior once the base leaves frame. Leaving it at the
    old absolute value while the height doubles would silently mis-range close in."""
    profile = PROFILES["bin"]
    doubled = visual_nav.static_profile(_args(static_prop="bin",
                                              prop_height=profile.height_m * 2.0))
    assert abs(doubled.width_m - profile.width_m * 2.0) < 1e-9


def test_the_profile_is_returned_untouched_when_nothing_is_overridden():
    assert visual_nav.static_profile(_args(static_prop="bin")) is PROFILES["bin"]


def test_custom_static_profile_is_evidence_backed_and_path_safe_in_telemetry():
    profile = {
        "schema": "colour-profile/v1",
        "label": "brown-box-marker",
        "hue_lo": 75,
        "hue_hi": 90,
        "sat_min": 200,
        "val_min": 70,
        "height_m": 0.05,
        "width_m": 0.10,
        "radius_m": 0.168,
        "min_area_px": 400,
        "min_fill": 0.55,
        "min_aspect": 1.3,
        "max_aspect": 2.6,
        "evidence": {"rtsp_frames": "green-marker-rtsp"},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.json"
        path.write_text(json.dumps(profile))
        args = _args(static_prop=None, static_profile=path)
        resolved = visual_nav.static_profile(args)
        telemetry = visual_nav.static_profile_telemetry(args, resolved)
        assert resolved.label == "brown-box-marker"
        assert telemetry["label"] == "brown-box-marker"
        assert telemetry["evidence"]["rtsp_frames"] == "green-marker-rtsp"
        assert str(path) not in json.dumps(telemetry)


def test_custom_static_profile_refuses_geometry_override():
    profile = {
        "schema": "colour-profile/v1",
        "label": "brown-box-marker",
        "hue_lo": 75,
        "hue_hi": 90,
        "sat_min": 200,
        "val_min": 70,
        "height_m": 0.05,
        "width_m": 0.10,
        "radius_m": 0.168,
        "min_area_px": 400,
        "min_fill": 0.55,
        "min_aspect": 1.3,
        "max_aspect": 2.6,
        "evidence": {"rtsp_frames": "green-marker-rtsp"},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.json"
        path.write_text(json.dumps(profile))
        try:
            visual_nav.static_profile(_args(static_prop=None, static_profile=path,
                                            prop_radius=0.2))
        except SystemExit as error:
            assert "--static-profile" in str(error)
        else:
            raise AssertionError("overrode evidence-backed custom profile geometry")


def test_static_profile_cli_sources_are_mutually_exclusive():
    parser = visual_nav.build_parser()
    try:
        parser.parse_args(["--static-prop", "bin", "--static-profile", "profile.json"])
    except SystemExit:
        pass
    else:
        raise AssertionError("accepted both named and custom static profile sources")


def test_a_detected_goal_without_a_height_is_refused():
    """The range scales linearly on it, so there is no safe default to guess."""
    try:
        visual_nav.build_goal_source(_args(goal_class="chair", goal_height=None),
                                     CAMERA, lambda: (0.0, 0.0, 0.0))
    except SystemExit:
        return
    raise AssertionError("built a detected-object goal with no height")


def test_a_waypoint_beats_a_detected_goal():
    """Both flags set is operator error; resolving it silently either way is worse than
    picking the one that needs no perception at all."""
    source = visual_nav.build_goal_source(
        _args(waypoint=[3.0, 0.0], goal_class="chair", goal_height=1.0),
        CAMERA, lambda: (0.0, 0.0, 0.0))
    assert isinstance(source, OdomWaypoint)


def test_the_default_goal_is_still_the_marker():
    source = visual_nav.build_goal_source(_args(), CAMERA, lambda: (0.0, 0.0, 0.0))
    assert isinstance(source, ArucoGoal)


# ── Static landmarks reach the planner ──────────────────────────────────────
def test_a_mapped_landmark_becomes_a_zero_velocity_obstacle():
    tracker, filter_time = _tracked_person(sightings=0)
    mapping = StaticObstacleMap(radii={"bin": 0.15})
    for index in range(4):
        mapping.observe([Observation.from_bearing_range(
            math.radians(7.1), 2.15, 0.0, 0.0, 0.0, label="bin")],
            filter_time + index * 0.14, 0.0, 0.0, 0.0)
    navigator = _navigator(tracker, filter_time, static_map=mapping)
    obstacles = navigator._obstacles(filter_time + LATENCY_S)
    assert len(obstacles) == 1
    obstacle = obstacles[0]
    assert obstacle.label == "bin"
    assert (obstacle.vx, obstacle.vy) == (0.0, 0.0), "a bin does not have a velocity"
    assert obstacle.soft_gap_m == STATIC_SOFT_GAP_M
    # Not extrapolated over the perception latency: it has nothing to extrapolate ON.
    assert abs(obstacle.x - 2.134) < 0.05 and abs(obstacle.y - 0.266) < 0.05


def test_landmarks_and_movers_reach_the_planner_together():
    tracker, filter_time = _tracked_person()
    mapping = StaticObstacleMap(radii={"bin": 0.15})
    for index in range(4):
        mapping.observe([Observation.from_bearing_range(
            math.radians(-30.0), 2.15, 0.0, 0.0, 0.0, label="bin")],
            filter_time + index * 0.14, 0.0, 0.0, 0.0)
    navigator = _navigator(tracker, filter_time, static_map=mapping)
    labels = sorted(o.label for o in navigator._obstacles(filter_time + LATENCY_S))
    assert labels == ["bin", "person"], labels


def test_each_obstacle_says_which_subsystem_produced_it_and_which_object_it_is():
    """The wiring, not the serialisation.

    ``kind`` and ``object_id`` are what a downstream policy splits on — a mover belongs
    to the stop-and-wait logic and a landmark is something to path around — and the
    telemetry tests cannot pin them, because they hand the writer obstacles that already
    carry the fields. This is the only place that would notice if ``_obstacles`` stopped
    setting them. ``kind`` DEFAULTS to "tracked", so a landmark that lost the argument
    would be silently reclassified as a mover rather than raising anything.
    """
    tracker, filter_time = _tracked_person()
    mapping = StaticObstacleMap(radii={"bin": 0.15})
    for index in range(4):
        mapping.observe([Observation.from_bearing_range(
            math.radians(-30.0), 2.15, 0.0, 0.0, 0.0, label="bin")],
            filter_time + index * 0.14, 0.0, 0.0, 0.0)
    navigator = _navigator(tracker, filter_time, static_map=mapping)
    obstacles = {o.label: o for o in navigator._obstacles(filter_time + LATENCY_S)}
    assert obstacles["person"].kind == "tracked"
    assert obstacles["bin"].kind == "static"
    # Identities come from the producers' own counters, so they survive a tick in which
    # an object was not seen — which is exactly when re-associating by position fails.
    assert obstacles["person"].object_id.startswith("track-")
    assert obstacles["bin"].object_id.startswith("landmark-")
    assert obstacles["person"].object_id != obstacles["bin"].object_id


# ── The stall abort ─────────────────────────────────────────────────────────
#: Deliberately NOT a divisor of PROGRESS_WINDOW_S, and irregular. A tidy dt=0.5 made
#: every window land exactly on the 4.0 s boundary, which was the only case an
#: off-by-one in the pruning could pass — so the tests went green while the gate could
#: never fire on the robot, where ticks arrive at an irregular ~10 Hz.
_TICK_DTS = (0.093, 0.117, 0.104, 0.131, 0.098, 0.112)


def _walk(navigator, command, poses):
    """Feed the stall gate a pose history and return its last verdict."""
    verdict = None
    now = 0.0
    for index, (x, y) in enumerate(poses):
        verdict = navigator._blocked_reason(command, now, (x, y, 0.0))
        now += _TICK_DTS[index % len(_TICK_DTS)]
    return verdict


CRUISE = Command(vx=0.30, vy=0.0, wz=0.0, reason="goal", gap_m=math.inf)


def test_a_robot_that_goes_nowhere_under_command_ends_the_run():
    """Observed live, twice: the tether went taut, the robot veered into a cubicle wall,
    and the loop kept commanding 0.10-0.14 m/s forward and a FULL 0.20 m/s of strafe
    into it while the pose sat still — for 21 s, then 12 s. Nothing stopped it."""
    navigator = _navigator_with(_FakeLoco(), live=True)
    navigator._standing = True
    verdict = _walk(navigator, CRUISE, [(0.0, 0.0)] * 60)
    assert verdict is not None and "stalled" in verdict, verdict
    assert "tether" in verdict, "the message should say where to look"


def test_a_robot_that_is_walking_is_left_alone():
    navigator = _navigator_with(_FakeLoco(), live=True)
    navigator._standing = True
    poses = [(0.030 * i, 0.0) for i in range(60)]     # ~0.28 m/s at ~0.11 s ticks
    assert _walk(navigator, CRUISE, poses) is None


def test_shuffling_on_the_spot_does_not_count_as_progress():
    """THE reason this measures net displacement instead of deferring to
    Go2Locomotion.is_blocked, whose bar is a tenth of the commanded speed — one
    centimetre per second for a 0.10 m/s command. A quadruped trotting against a taut
    tether IS moving its legs; its body-velocity estimate is not zero. What it is not
    doing is getting anywhere."""
    navigator = _navigator_with(_FakeLoco(), live=True)
    navigator._standing = True
    jitter = [(0.01 * (i % 2), 0.01 * ((i + 1) % 2)) for i in range(60)]
    verdict = _walk(navigator, CRUISE, jitter)
    assert verdict is not None, "a centimetre of twitch is not a walk"


def test_a_slow_but_real_manoeuvre_is_not_called_a_stall():
    """A careful sidestep is slow on purpose and must survive the window."""
    navigator = _navigator_with(_FakeLoco(), live=True)
    navigator._standing = True
    creep = Command(vx=0.10, vy=0.05, wz=0.0, reason="avoid", gap_m=0.4)
    poses = [(0.010 * i, 0.005 * i) for i in range(60)]
    assert _walk(navigator, creep, poses) is None


def test_the_gate_waits_for_a_full_window_before_judging():
    """Two ticks after standing up is not evidence of anything."""
    navigator = _navigator_with(_FakeLoco(), live=True)
    navigator._standing = True
    assert _walk(navigator, CRUISE, [(0.0, 0.0)] * 5) is None


def test_a_stopped_robot_is_not_asked_to_have_moved():
    navigator = _navigator_with(_FakeLoco(), live=True)
    navigator._standing = True
    held = Command(vx=0.0, vy=0.0, wz=0.0, reason="hold", gap_m=0.2)
    assert _walk(navigator, held, [(0.0, 0.0)] * 60) is None


def test_a_dry_run_never_reports_a_stall():
    """No leg moves, so 'commanded to move and not moving' is true of every tick."""
    navigator = _navigator_with(_FakeLoco(), live=False)
    navigator._standing = True
    assert _walk(navigator, CRUISE, [(0.0, 0.0)] * 60) is None


def test_a_prone_robot_is_not_judged_for_standing_still():
    navigator = _navigator_with(_FakeLoco(), live=True)
    navigator._standing = False
    assert _walk(navigator, CRUISE, [(0.0, 0.0)] * 60) is None


def test_lying_down_clears_the_history():
    """Otherwise the ticks spent prone count against the first ticks after standing."""
    navigator = _navigator_with(_FakeLoco(), live=True)
    navigator._standing = False
    _walk(navigator, CRUISE, [(0.0, 0.0)] * 60)
    navigator._standing = True
    assert navigator._blocked_reason(CRUISE, 99.0, (0.0, 0.0, 0.0)) is None


def test_the_dynamic_window_is_sized_by_the_interval_that_actually_elapsed():
    """The bug this pins cost three hardware runs. The loop passed the NOMINAL period
    while running at 2.8 Hz, so the planner allowed 0.05 m/s of change per tick where
    the robot could deliver 0.18. The ramp from a standstill never reached the gait
    floor before the next stale hold reset it, and the robot stood still under a full
    forward command while the stall gate blamed the tether."""
    period = 0.1
    # A loop keeping up: unchanged from the behaviour before the fix.
    assert visual_nav.control_interval_s(10.0, 10.1, period) == period
    # A loop running slow: the window must open to match real elapsed time.
    assert math.isclose(visual_nav.control_interval_s(10.0, 10.36, period), 0.36,
                        abs_tol=1e-9)
    # A loop running FAST must not shrink its own window below the nominal period.
    assert visual_nav.control_interval_s(10.0, 10.02, period) == period
    # The first tick has no predecessor and falls back to the nominal period.
    assert visual_nav.control_interval_s(None, 10.0, period) == period


def test_one_very_long_tick_cannot_authorise_the_whole_envelope():
    """Capped, so a run that hung for ten seconds does not come back with a dynamic
    window wide enough to jump straight to any velocity the robot can reach."""
    assert visual_nav.control_interval_s(0.0, 10.0, 0.1) == visual_nav.MAX_CONTROL_DT_S
    assert visual_nav.MAX_CONTROL_DT_S * 0.50 < 0.35, (
        "the cap must stay under a full-envelope jump in forward speed")


# ── the raw recorder ─────────────────────────────────────────────────────────
class _FakeWriter:
    """Records what it was handed, and whether it was handed the same array twice."""

    def __init__(self) -> None:
        self.frames: list = []

    def write(self, frame) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        pass


def _recording_navigator(recorder=None, raw_recorder=None) -> VisualNavigator:
    return VisualNavigator(
        loco=None, perception=None,
        planner=DynamicWindowPlanner(config=PlannerConfig()),
        tracker=ObstacleTracker(), goal_source=OdomWaypoint((0.0, 0.0, 0.0), 4.0),
        health=_FakeHealth(ticks=999), config=NavConfig(),
        recorder=recorder, raw_recorder=raw_recorder)


def _perception_with_a_detection(seq: int = 1) -> PerceptionResult:
    """A frame with one detection in it, so the annotated path has a box to draw."""
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:] = 30                                   # not black, so a drawn pixel shows
    detection = person_detector.Detection(
        x1=20.0, y1=20.0, x2=100.0, y2=90.0, score=0.8, label="person")
    ranged = [person_detector.RangedDetection(
        detection=detection, range_m=1.5, bearing_rad=0.0, source="height")]
    return PerceptionResult(seq=seq, capture_time=time.monotonic(),
                            pose=(0.0, 0.0, 0.0), observations=[], ranged=ranged,
                            image=image, detect_ms=12.0)


def test_the_raw_recording_has_nothing_drawn_on_it():
    """The whole point. --record writes the label into the pixels; --record-raw does not.

    Compared against the annotated frame from the SAME cycle rather than against a
    hand-built expectation, so this cannot pass by agreeing with a stale idea of what
    the overlay draws.
    """
    annotated, raw = _FakeWriter(), _FakeWriter()
    navigator = _recording_navigator(recorder=annotated, raw_recorder=raw)
    result = _perception_with_a_detection()
    navigator._record(result, (0.0, 0.0, 0.0), Command(0.3, 0.0, 0.0, "goal", math.inf))

    assert len(annotated.frames) == 1 and len(raw.frames) == 1
    assert np.array_equal(raw.frames[0], result.image), "the raw frame was decorated"
    assert not np.array_equal(annotated.frames[0], raw.frames[0]), (
        "the annotated frame is identical to the raw one — the overlay drew nothing, "
        "so this test would pass whatever _record did")
    # and specifically: the box the detector drew is not in the raw frame
    changed = np.argwhere(np.any(annotated.frames[0] != raw.frames[0], axis=2))
    assert len(changed) > 50, f"only {len(changed)} pixels differ; expected an overlay"


def test_the_source_frame_is_not_mutated_by_the_annotated_path():
    """`result.image` is what the raw writer is handed, so the overlay must copy first.

    If the annotated path ever drew in place, the raw file would silently become a
    second copy of the annotated one — and the run would look fine.
    """
    raw = _FakeWriter()
    navigator = _recording_navigator(recorder=_FakeWriter(), raw_recorder=raw)
    result = _perception_with_a_detection()
    before = result.image.copy()
    navigator._record(result, (0.0, 0.0, 0.0), Command(0.3, 0.0, 0.0, "goal", math.inf))
    assert np.array_equal(result.image, before), "the overlay drew on the source frame"


def test_both_recorders_share_one_frame_index():
    """Frame n of the raw file must be frame n of the annotated one, and of the telemetry.

    Two counters would drift the moment one writer skipped a cycle, and the drift would
    be invisible: both files would still play, and every join would be off by one.
    """
    annotated, raw = _FakeWriter(), _FakeWriter()
    navigator = _recording_navigator(recorder=annotated, raw_recorder=raw)
    indices = []
    for seq in (1, 2, 3):
        indices.append(navigator._record(_perception_with_a_detection(seq),
                                         (0.0, 0.0, 0.0),
                                         Command(0.3, 0.0, 0.0, "goal", math.inf)))
    assert indices == [0, 1, 2], indices
    assert len(annotated.frames) == len(raw.frames) == 3


def test_the_raw_recorder_advances_on_perception_not_on_ticks():
    """Same cadence rule as --record: once per perception cycle, not once per control tick.

    A tick that consumes an already-recorded result must write nothing and return None,
    or the video plays back faster than the run happened.
    """
    raw = _FakeWriter()
    navigator = _recording_navigator(raw_recorder=raw)
    result = _perception_with_a_detection(seq=4)
    assert navigator._record(result, (0.0, 0.0, 0.0), None) == 0
    assert navigator._record(result, (0.0, 0.0, 0.0), None) is None
    assert navigator._record(result, (0.0, 0.0, 0.0), None) is None
    assert len(raw.frames) == 1


def test_the_raw_recorder_works_with_no_annotated_recorder():
    """--record-raw alone still produces the join key, or the frames cannot be labelled.

    The gate used to be `if self._recorder is None: return None`, which would have made
    --record-raw on its own write frames that telemetry indexed as None.
    """
    raw = _FakeWriter()
    navigator = _recording_navigator(raw_recorder=raw)
    assert navigator._record(_perception_with_a_detection(), (0.0, 0.0, 0.0), None) == 0
    assert len(raw.frames) == 1


def test_neither_recorder_means_no_work_and_no_index():
    navigator = _recording_navigator()
    assert navigator._record(_perception_with_a_detection(), (0.0, 0.0, 0.0), None) is None


def test_record_raw_is_off_by_default_and_is_its_own_flag():
    parser = build_parser()
    assert parser.parse_args([]).record_raw is None
    assert parser.parse_args([]).record is None
    args = parser.parse_args(["--record-raw", "raw.mp4"])
    assert args.record_raw == "raw.mp4" and args.record is None


def test_the_constructor_keeps_the_positional_prefix_the_vendored_call_passes():
    """`integration/mappo_drive.py` builds this class through `navigator_factory` and says
    in its own docstring that a new positional argument must not be added. `main` passes
    ten of them positionally; everything since — `raw_recorder`, `profiler` — comes after
    and is passed by keyword.

    THE PREFIX IS THE INVARIANT, and the proxy this test used before was not. It asserted
    `raw_recorder` was the LAST parameter, which fails the first time a second optional
    argument is appended perfectly correctly — as `profiler` was — while still passing if
    someone inserted a positional one in the middle, which is the defect it was written to
    catch. Pinning the prefix and the defaults catches that and nothing else.
    """
    parameters = inspect.signature(VisualNavigator.__init__).parameters
    names = list(parameters)
    assert names[:11] == ["self", "loco", "perception", "planner", "tracker",
                          "goal_source", "health", "config", "recorder", "static_map",
                          "telemetry"], names
    optional = names[11:]
    assert set(optional) >= {"stand_up_fn", "lie_down_fn", "expansion", "raw_recorder",
                             "profiler"}, optional
    for name in optional:
        assert parameters[name].default is not inspect.Parameter.empty, name

# ── The routing seam: shape decides, and it has to survive the whole chain ──
FRAME_W, FRAME_H = 1920, 1080

#: A Go2 Wheel broadside at mid range: 460 x 360 px, aspect 0.78, which is the MEDIAN of
#: the 1,159 unclipped boxes of the 2026-08-24 peer corpus. Unclipped on every edge.
PEER_BOX = Detection(x1=700.0, y1=560.0, x2=1160.0, y2=920.0, score=0.62,
                     label="motorbike")
#: A standing adult at the same sort of range: 150 x 510 px, aspect 3.40, which is the
#: repo's own 1.70/0.50 prior. Labelled `motorbike` ON PURPOSE — that is the mislabel
#: that made the old label-based rule hand a person to the policy.
PERSON_BOX = Detection(x1=880.0, y1=300.0, x2=1030.0, y2=810.0, score=0.55,
                       label="motorbike")


def _obstacle_for(box: Detection, sightings: int = 3):
    """Drive one box through the WHOLE producer chain and return what comes out.

    Detector shape verdict -> Observation -> spawn -> correct -> Track -> _obstacles.
    Every step is the real one; nothing is constructed with `person_shaped=` by hand,
    because hand-setting it is exactly what a broken chain would still let a test do.
    """
    ranged = RangedDetection(detection=box, range_m=3.0,
                             bearing_rad=math.radians(12.0), source="height")
    verdict = ranged.person_shaped(FRAME_W, FRAME_H)
    tracker = ObstacleTracker(fov_rad=math.radians(85.27))
    now = 0.0
    for _ in range(sightings):
        tracker.predict(PERCEPTION_DT)
        tracker.update([observation_from(ranged.bearing_rad, ranged.range_m,
                                         ranged.source, ranged.label, (0.0, 0.0, 0.0),
                                         person_shaped=verdict)],
                       now, 0.0, 0.0, 0.0)
        now += PERCEPTION_DT
    obstacles = _navigator(tracker, now)._obstacles(now + LATENCY_S)
    assert len(obstacles) == 1, "the fixture must produce exactly one confirmed track"
    return verdict, obstacles[0]


def _telemetry_record(obstacle) -> dict:
    """The obstacle dict `mappo_bridge` actually reads, written by the real writer."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "t.jsonl"
        writer = TelemetryWriter(str(path))
        writer.write_tick(elapsed_s=0.0, pose=(0.0, 0.0, 0.0), goal_xy=(4.0, 0.0),
                          goal_distance_m=4.0, command=None, obstacles=[obstacle],
                          frame_age_s=0.1, perception_seq=1, detect_ms=131.0,
                          standing=True, live=False)
        writer.close()
        line = json.loads(path.read_text().strip().splitlines()[-1])
    return line["obstacles"][0]


def test_a_peer_reaches_the_policy_and_a_person_holds_end_to_end():
    """⛔ THE PROPERTY THAT A CLEAN MERGE CAN SILENTLY BREAK.

    `person_shaped` is carried BESIDE `label` through five hand-offs — the detector's
    shape verdict, `observation_from`, `_spawn`, `_correct`, and `_obstacles` — and every
    one of them defaults it to True, which is the stopping side. Drop it at any single
    hand-off and the chain still type-checks, still writes telemetry, and quietly routes
    EVERY peer to the hold path: the policy is handed nothing and the robot stands in
    front of the obstacle the whole integration exists to steer around.

    Measured: deleting `person_shaped=` from `_obstacles`, from `_spawn`, or from
    `_correct` each leaves all 315 tests in this directory green. This is the test that
    goes red instead, and it is written against `holds_the_robot` — the real read site in
    `integration/` — rather than against the flag, so a rename on either side of the seam
    fails it too.
    """
    peer_verdict, peer = _obstacle_for(PEER_BOX)
    person_verdict, person = _obstacle_for(PERSON_BOX)

    # The premise: the two boxes sit either side of the threshold, and the LABEL is the
    # same on both, so nothing below can be passing on the label.
    assert PEER_BOX.height_px / PEER_BOX.width_px < PERSON_ASPECT_MIN
    assert PERSON_BOX.height_px / PERSON_BOX.width_px >= PERSON_ASPECT_MIN
    assert PEER_BOX.label == PERSON_BOX.label == "motorbike"
    assert (peer_verdict, person_verdict) == (False, True)

    # The chain carried it, all the way to the dict the bridge reads.
    assert peer.person_shaped is False and person.person_shaped is True
    peer_record = _telemetry_record(peer)
    person_record = _telemetry_record(person)
    assert peer_record["person_shaped"] is False
    assert person_record["person_shaped"] is True

    # And the read site routes on it: the peer reaches the policy, the person holds.
    assert holds_the_robot(peer_record) is False, "a peer must reach the policy"
    assert holds_the_robot(person_record) is True, "a person must hold the robot"


def test_the_expansion_filter_is_off_unless_asked_for_and_cannot_route():
    """Two separate properties, both of which have to hold for this PR to be inert.

    OFF BY DEFAULT: `--expansion-filter` is a store_true, so a navigator built the way
    every run builds one has `_expansion is None` and `_obstacles` subtracts nothing.

    AND IT DOES NOT TOUCH ROUTING: with a gate fitted that rejects nothing, the same peer
    and the same person come out with the same verdicts. The gate can only ever REMOVE a
    track from the planner; it must never flip one from the policy path to the hold path
    or the other way, because those two are decided by shape and it does not see shape.
    """
    assert build_parser().parse_args([]).expansion_filter is False

    class _RejectsNothing:
        def observe(self, *args, **kwargs) -> None:
            self.observed = True

        def retain(self, ids) -> None:
            list(ids)

        def rejects(self, ids) -> set:
            list(ids)
            return set()

    for box, expected_hold in ((PEER_BOX, False), (PERSON_BOX, True)):
        ranged = RangedDetection(detection=box, range_m=3.0,
                                 bearing_rad=math.radians(12.0), source="height")
        gate = _RejectsNothing()
        tracker = ObstacleTracker(fov_rad=math.radians(85.27), expansion=gate)
        now = 0.0
        for _ in range(3):
            tracker.predict(PERCEPTION_DT)
            tracker.update([observation_from(
                ranged.bearing_rad, ranged.range_m, ranged.source, ranged.label,
                (0.0, 0.0, 0.0),
                person_shaped=ranged.person_shaped(FRAME_W, FRAME_H))],
                now, 0.0, 0.0, 0.0)
            now += PERCEPTION_DT
        navigator = _navigator(tracker, now)
        navigator._expansion = gate
        obstacle = navigator._obstacles(now + LATENCY_S)[0]
        assert gate.observed, "a fitted gate must actually be fed the measurements"
        assert holds_the_robot(_telemetry_record(obstacle)) is expected_hold


def test_a_rejected_track_is_withheld_from_the_planner_but_only_when_asked():
    """The gate's one effect, pinned in both directions.

    Fitted and rejecting, the track leaves the obstacle set. NOT fitted — the shipped
    default — the identical tracker state still yields the obstacle. If the second half
    ever fails, the filter has become on-by-default, which is what this PR must not do.
    """
    ranged = RangedDetection(detection=PEER_BOX, range_m=3.0,
                             bearing_rad=math.radians(12.0), source="height")

    def _tracker_with(gate):
        tracker = ObstacleTracker(fov_rad=math.radians(85.27), expansion=gate)
        now = 0.0
        for _ in range(3):
            tracker.predict(PERCEPTION_DT)
            tracker.update([observation_from(
                ranged.bearing_rad, ranged.range_m, ranged.source, ranged.label,
                (0.0, 0.0, 0.0),
                person_shaped=ranged.person_shaped(FRAME_W, FRAME_H))],
                now, 0.0, 0.0, 0.0)
            now += PERCEPTION_DT
        return tracker, now

    class _RejectsEverything:
        def observe(self, *args, **kwargs) -> None:
            pass

        def retain(self, ids) -> None:
            list(ids)

        def rejects(self, ids) -> set:
            return set(ids)

    gate = _RejectsEverything()
    tracker, now = _tracker_with(gate)
    navigator = _navigator(tracker, now)
    navigator._expansion = gate
    assert navigator._obstacles(now + LATENCY_S) == []

    tracker, now = _tracker_with(None)
    navigator = _navigator(tracker, now)
    assert navigator._expansion is None, "the shipped default fits no gate"
    assert len(navigator._obstacles(now + LATENCY_S)) == 1


# ── the camera model's lens height ──────────────────────────────────────────
def _calibration_file(directory, name="cal.json", **overrides):
    """A calibration of the shape `FisheyeCamera.save` writes, with fields overridden.

    An override of `None` DELETES the key, so both shapes of "this file states no lens
    height" — the null the fitter now writes, and a key nobody ever wrote — come from one
    helper rather than from two hand-built dicts that could drift apart.
    """
    data = {"width": 1920, "height": 1080, "focal_px": 1290.1637909789656,
            "cx": 960.0, "cy": 540.0, "pitch_rad": 0.0,
            "height_m": GO2_CAMERA_HEIGHT_M, "method": "spin"}
    for key, value in overrides.items():
        if value is _DELETED:
            data.pop(key, None)
        else:
            data[key] = value
    path = os.path.join(directory, name)
    Path(path).write_text(json.dumps(data))
    return path


#: Sentinel for `_calibration_file`: leave the key out entirely rather than set it null.
_DELETED = object()


def test_a_calibration_without_a_lens_height_is_refused_before_the_run_starts():
    """`object_fit_range` refuses without a lens height, and the first horizontally
    clipped detection can be several metres into a live run. A refusal that can only fire
    mid-run is a robot that stops walking because the process died, so the field is read
    at start-up. `calibrate_camera.py` fits a focal length and cannot measure a lens
    height, so the file it writes carries none until somebody adds it."""
    with tempfile.TemporaryDirectory() as directory:
        null = _calibration_file(directory, "null.json", height_m=None)
        omitted = _calibration_file(directory, "omitted.json", height_m=_DELETED)
        for path in (null, omitted):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    build_camera_model(1920, 1080, path)
            except SystemExit as refusal:
                message = str(refusal)
            else:
                raise AssertionError(f"{path} states no lens height and must be refused")
            assert "REFUSING TO RUN" in message, message
            assert path in message, "the refusal has to name the file"
            assert str(GO2_CAMERA_HEIGHT_M) in message, "and the Go2's number"
            assert "--lens-height" in message, "and how a Lite3 gets one"


def test_a_calibration_that_states_a_lens_height_is_accepted_and_reported():
    with tempfile.TemporaryDirectory() as directory:
        path = _calibration_file(directory, height_m=0.31)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            model = build_camera_model(1920, 1080, path)
        assert model.height_m == 0.31
        assert "lens height 0.310m" in out.getvalue(), out.getvalue()


def test_the_uncalibrated_fallback_names_the_go2s_lens_height_rather_than_inheriting_it():
    """Every intrinsic on this path is already the Go2's published nominal spec, so the
    Go2's lens height belongs with it — but it is passed at the call site, not taken from
    a default, and the run says whose numbers they are."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        model = build_camera_model(1920, 1080, None)
    assert model.height_m == GO2_CAMERA_HEIGHT_M
    printed = out.getvalue()
    assert "NOMINAL" in printed and "GO2" in printed, printed
    assert "un-calibrated" in printed, printed


# ── The sub-threshold static detection tier ────────────────────────────────
class _TieredDetector:
    """A detector that offers one mover and one static candidate per frame."""

    def __init__(self, movers=(), statics=()):
        self.movers = list(movers)
        self.statics = list(statics)
        self.calls = 0

    def detect_tiered(self, _image):
        self.calls += 1
        return list(self.movers), list(self.statics)


class _OneFrame:
    def __init__(self):
        self.seq = 0

    def wait_for_new(self, after_seq, timeout):
        self.seq += 1
        return Frame(image=np.zeros((1080, 1920, 3), np.uint8),
                     capture_time=time.monotonic(), seq=self.seq,
                     stamp=(0.0, 0.0, 0.0))


class _NoGoalSource:
    def update(self, _image, _pose):
        return None


#: A box that clears every gate in `static_shaped`, at the corpus frame size.
_STATIC_BOX = person_detector.Detection(x1=700.0, y1=300.0, x2=1100.0, y2=880.0,
                                        score=0.12, label="chair")
_BOX_PRIOR = person_detector.SizePrior(height_m=0.40, width_m=0.35)


def _one_cycle(**kwargs) -> PerceptionResult:
    worker = PerceptionWorker(_OneFrame(), kwargs.pop("detector"), CAMERA,
                              _NoGoalSource(), lambda: (0.0, 0.0, 0.0), **kwargs)
    worker.start()
    deadline = time.monotonic() + 2.0
    while worker.latest() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    result = worker.latest()
    worker.stop()
    assert result is not None, "perception produced nothing"
    return result


def test_a_static_detection_reaches_the_map_and_never_the_tracker():
    """The whole routing argument in one assertion. A detection this close to the
    network's noise floor must not be differentiated into a velocity, must not be able
    to HOLD the robot, and must earn a second agreeing sighting before the planner sees
    it — all three of which follow from it being a landmark rather than a track."""
    result = _one_cycle(detector=_TieredDetector(statics=[_STATIC_BOX]),
                        static_prior=_BOX_PRIOR)
    assert result.observations == [], "a static candidate must not become a track"
    assert len(result.static_observations) == 1
    assert result.static_observations[0].label == STATIC_DETECT_LABEL


def test_the_tier_is_inert_without_a_measured_prior():
    """FAIL CLOSED. `range_detections` turns a size prior into metres; with no prior
    there is no defensible number to put in the map, and a default would put a landmark
    at a range nothing measured. Offering candidates is not enough to map them."""
    result = _one_cycle(detector=_TieredDetector(statics=[_STATIC_BOX]),
                        static_prior=None)
    assert result.static_observations == []
    assert result.observations == []


def test_movers_still_reach_the_tracker_with_the_tier_on():
    """The mover tier is unchanged. If enabling static detection could quietly divert a
    person into the map, the feature would disable giving way to people."""
    person = person_detector.Detection(x1=800.0, y1=200.0, x2=950.0, y2=900.0,
                                       score=0.8, label="person")
    result = _one_cycle(detector=_TieredDetector(movers=[person],
                                                 statics=[_STATIC_BOX]),
                        static_prior=_BOX_PRIOR)
    assert len(result.observations) == 1
    assert result.observations[0].label == "person"
    assert len(result.static_observations) == 1


def test_the_static_label_is_not_the_voc_label():
    """`StaticObstacleMap` keys its plan-view radii by label. The same recycling bin
    comes back `tvmonitor` in one frame and `chair` in the next, so carrying VOC through
    would split one object into two landmarks with two different radii."""
    other = person_detector.Detection(x1=700.0, y1=300.0, x2=1100.0, y2=880.0,
                                      score=0.13, label="tvmonitor")
    result = _one_cycle(detector=_TieredDetector(statics=[_STATIC_BOX, other]),
                        static_prior=_BOX_PRIOR)
    assert {o.label for o in result.static_observations} == {STATIC_DETECT_LABEL}


def test_the_tier_is_off_by_default():
    """Nothing that changes what a robot does may arrive enabled."""
    args = build_parser().parse_args([])
    assert args.static_detect is False
    assert static_detect_prior(args) is None


def test_the_tier_refuses_to_start_without_measured_dimensions():
    """An inferred width comes from a standing adult's aspect ratio. A cardboard box is
    squatter than a person, so the inference is wronger for it than for the peer it
    already misranged into the robot's own footprint."""
    for flags in (["--static-detect"],
                  ["--static-detect", "--static-detect-height", "0.4"],
                  ["--static-detect", "--static-detect-width", "0.35"]):
        args = build_parser().parse_args(flags)
        try:
            static_detect_prior(args)
        except SystemExit:
            continue
        raise AssertionError(f"started the tier with an unmeasured prior: {flags}")


def test_the_radius_defaults_to_the_objects_own_footprint():
    """Half the measured width, with no clearance added. The planner adds its own and
    double-counting it is how a prop grows a disc it does not have."""
    args = build_parser().parse_args(
        ["--static-detect", "--static-detect-height", "0.40",
         "--static-detect-width", "0.35"])
    prior, radius = static_detect_prior(args)
    assert math.isclose(prior.height_m, 0.40)
    assert math.isclose(prior.width_m, 0.35)
    assert math.isclose(radius, 0.175)


def test_a_nonsense_dimension_is_refused():
    """Every range scales linearly on the height, so a zero or a NaN does not degrade
    the estimate, it destroys it."""
    for value in ("0", "-0.4", "nan"):
        args = build_parser().parse_args(
            ["--static-detect", "--static-detect-height", value,
             "--static-detect-width", "0.35"])
        try:
            static_detect_prior(args)
        except SystemExit:
            continue
        raise AssertionError(f"accepted --static-detect-height {value}")


# ── Where the tick went (issue #18) ──────────────────────────────────────────
#: A stage cost the test can see against a real monotonic clock. `time.sleep` is a LOWER
#: bound, so the assertions below are all "at least"; ten times the period keeps a slow
#: machine's scheduler jitter well clear of the threshold either way.
SLOW_MS = 25.0


def _burn(_self=None, *_args, **_kwargs):
    time.sleep(SLOW_MS / 1000.0)


class _SlowWriter(_FakeWriter):
    """A recorder that costs what a 1920x1080 mp4v encode costs on the Jetson."""

    def write(self, frame) -> None:
        _burn()
        super().write(frame)


class _SlowPlanner(_FakePlanner):
    def plan(self, *args, **kwargs):
        _burn()
        return super().plan(*args, **kwargs)


class _SlowTracker(ObstacleTracker):
    """The OTHER thing gated on `result.seq > <last>`, made expensive on purpose.

    `_record` and the tracker update fire on exactly the same ticks, so a recorded file
    cannot tell them apart within one run — which is the confound both #112 and #116
    named — and they are the whole reason `record` and `tracker` are separate names.
    """

    def update(self, *args, **kwargs):
        _burn()
        return super().update(*args, **kwargs)


class _ImagePerception(_FakePerception):
    """One perception cycle carrying pixels, so the recorder has something to encode.

    The seq never advances, which is deliberate: `_record` writes once per PERCEPTION
    cycle, so exactly one tick of the run records and the rest do not. That is the shape
    of every live run in `evidence/` — 72 to 98% of ticks record, and the ones that do not
    land on the configured period.
    """

    def __init__(self, age_s: float = 0.0) -> None:
        self._age = age_s
        self._image = np.zeros((60, 80, 3), dtype=np.uint8)

    def latest(self):
        return PerceptionResult(seq=1, capture_time=time.monotonic() - self._age,
                                pose=(0.0, 0.0, 0.0), observations=[], ranged=[],
                                image=self._image, detect_ms=200.0,
                                cycle_ms=311.0, wait_ms=104.0)


def _profiled_run(directory, *, planner=None, recorder=None, perception=None, ticks=4,
                  tracker=None):
    """Run the real loop with a real telemetry writer; return the tick records."""
    path = os.path.join(directory, "profiled.jsonl")
    writer = TelemetryWriter(path)
    writer.write_header(control_hz=100.0)
    navigator = VisualNavigator(
        loco=_FakeLoco(), perception=perception or _ImagePerception(),
        planner=planner or _FakePlanner(), tracker=tracker or ObstacleTracker(),
        goal_source=_FakeGoal(), health=_FakeHealth(ticks),
        config=NavConfig(live=True, control_hz=100.0),
        recorder=recorder, telemetry=writer)
    with contextlib.redirect_stdout(io.StringIO()):
        navigator.run()
    writer.close()
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle
                if line.strip() and json.loads(line)["type"] == "tick"]


def test_every_tick_carries_a_stage_profile():
    """Issue #18 sat for eight days on a loop nobody could attribute, because there was
    no `perf_counter`, no `cProfile` and no stage timer anywhere in the tree. Always on,
    because a profiler behind a flag is off during every run worth explaining."""
    with tempfile.TemporaryDirectory() as directory:
        ticks = _profiled_run(directory)
    assert ticks, "the loop wrote no ticks"
    for tick in ticks:
        assert tick["profile"] is not None, tick
        assert sorted(tick["profile"]["stages"]) == sorted(TICK_STAGES)
        assert tick["profile"]["tick_ms"] >= 0.0


def test_the_recorder_lands_in_the_record_stage_and_the_other_ticks_stay_cheap():
    """THE FINDING THIS EXISTS TO CONFIRM ON A REAL RUN. Across the committed runs, every
    tick that wrote a video frame took 173-299 ms and every tick that did not took
    100.3-104.0 ms — the configured period. The corpus already attributes that to the
    recorder rather than to the tracker, because the two runs with no `--record` still
    consume results and still update the tracker and their new-result ticks measure
    100.6 ms (see `telemetry.TickProfiler`). This is the on-robot confirmation of it.
    """
    with tempfile.TemporaryDirectory() as directory:
        ticks = _profiled_run(directory, recorder=_SlowWriter())
    recording = [t for t in ticks if t["perception"]["video_frame"] is not None]
    rest = [t for t in ticks if t["perception"]["video_frame"] is None]
    assert len(recording) == 1 and rest, [t["perception"]["video_frame"] for t in ticks]
    assert recording[0]["profile"]["stages"]["record"] >= SLOW_MS * 0.8, recording[0]
    assert recording[0]["profile"]["tick_ms"] >= SLOW_MS * 0.8
    for tick in rest:
        assert tick["profile"]["stages"]["record"] < SLOW_MS * 0.4, tick


def test_the_tracker_is_timed_apart_from_the_recorder_they_share_a_gate_with():
    """THE CONFOUND, SEPARATED. `_record` and the tracker update fire on the same predicate
    — `result.seq > <last>` — so within one recorded run they are one indicator for two
    candidates. Offline the dry runs settle it (their new-result ticks cost 100.6 ms with
    the tracker running and no recorder). On the robot this is what settles it: a slow
    tracker must show up as `tracker` and not as `record`.
    """
    with tempfile.TemporaryDirectory() as directory:
        ticks = _profiled_run(directory, tracker=_SlowTracker(), recorder=_FakeWriter())
    consuming = [t for t in ticks if t["perception"]["video_frame"] is not None]
    rest = [t for t in ticks if t["perception"]["video_frame"] is None]
    assert len(consuming) == 1 and rest, [t["perception"]["video_frame"] for t in ticks]
    assert consuming[0]["profile"]["stages"]["tracker"] >= SLOW_MS * 0.8, consuming[0]
    assert consuming[0]["profile"]["stages"]["record"] < SLOW_MS * 0.4, consuming[0]
    # `other_ms` is deliberately not asserted here: the tick that consumes the first result
    # is also the one that stands the robot up, and that is ~3 s of posture change which
    # `other_ms` reports correctly. See `test_the_planner_lands_in_the_plan_stage...`.
    for tick in rest:
        # Timed INSIDE the gate, so a tick that consumed nothing shows a zero rather than
        # a median halfway between an update and a no-op.
        assert tick["profile"]["stages"]["tracker"] == 0.0, tick


def test_a_slow_recorder_does_not_land_in_the_tracker_stage():
    """The other direction of the same confound, and the one that would have kept issue #18
    open: an encode charged to `tracker` reads as "the filter is the problem"."""
    with tempfile.TemporaryDirectory() as directory:
        ticks = _profiled_run(directory, recorder=_SlowWriter())
    recording = [t for t in ticks if t["perception"]["video_frame"] is not None]
    assert len(recording) == 1, ticks
    assert recording[0]["profile"]["stages"]["tracker"] < SLOW_MS * 0.4, recording[0]


def test_the_planner_lands_in_the_plan_stage_and_not_in_the_remainder():
    """`plan` is the other candidate #18 named: `MappoPlanner.plan` runs the full DWA
    rollout, the policy step AND a second rollout in `is_feasible`, on every tick. A cost
    that fell into `other_ms` would read as an un-instrumented hole rather than as the
    planner."""
    with tempfile.TemporaryDirectory() as directory:
        ticks = _profiled_run(directory, planner=_SlowPlanner())
    # The stand-up tick is excluded by `command is not None`: standing blocks the loop
    # for ~3 s and the loop gives that tick up rather than issuing its now-stale plan, so
    # its `other_ms` is three seconds of posture change and says so correctly.
    planned = [t for t in ticks if t["command"] is not None]
    assert planned, [t["profile"] for t in ticks]
    for tick in planned:
        assert tick["profile"]["stages"]["plan"] >= SLOW_MS * 0.8, tick
        assert tick["profile"]["other_ms"] < SLOW_MS * 0.4, tick
        # Not zero: with no recorder attached `_record` still returns through the timer,
        # which is the point of timing the whole method rather than the encode alone.
        assert tick["profile"]["stages"]["record"] < SLOW_MS * 0.1, tick


def test_a_stale_perception_hold_is_profiled_and_planned_nothing():
    """The stale branch commands zero and returns without planning or recording, and it
    was 16-33% of ticks on 2026-08-17. A profiler that only covered the healthy path
    would leave exactly those ticks unexplained."""
    with tempfile.TemporaryDirectory() as directory:
        ticks = _profiled_run(directory, planner=_SlowPlanner(),
                              perception=_ImagePerception(age_s=5.0))
    assert ticks and all(t["perception"]["stale"] for t in ticks), ticks
    for tick in ticks:
        assert tick["profile"] is not None
        assert tick["profile"]["stages"]["plan"] == 0.0, tick
        assert tick["profile"]["stages"]["record"] == 0.0, tick


def test_the_telemetry_write_is_priced_on_the_tick_after_it():
    """A record cannot carry the time it took to write itself. The first tick has nothing
    to report and every later one does — which is also what proves the writer is being
    timed at all rather than the field being a constant."""
    with tempfile.TemporaryDirectory() as directory:
        ticks = _profiled_run(directory, ticks=5)
    assert len(ticks) >= 3, ticks
    assert ticks[0]["profile"]["write_prev_ms"] == 0.0
    assert any(t["profile"]["write_prev_ms"] > 0.0 for t in ticks[1:]), \
        [t["profile"]["write_prev_ms"] for t in ticks]


def test_the_camera_wait_is_kept_out_of_the_perception_cycle_time():
    """`cycle_ms` has to be the thread's COMPUTE. A cycle time that included the block on
    `wait_for_new` would read as a slow detector on a run whose real problem was a slow
    camera, and the two want opposite fixes."""

    class _SlowCamera:
        def __init__(self):
            self.seq = 0

        def wait_for_new(self, after_seq, timeout):
            self.seq += 1
            time.sleep(SLOW_MS / 1000.0)
            return Frame(image=np.zeros((40, 60, 3), dtype=np.uint8),
                         capture_time=time.monotonic(), seq=self.seq,
                         stamp=(0.0, 0.0, 0.0))

    class _QuickDetector:
        def detect_tiered(self, _image):
            return [], []

    class _NoGoal:
        def update(self, _image, _pose):
            return None

    worker = PerceptionWorker(_SlowCamera(), _QuickDetector(), CAMERA, _NoGoal(),
                              lambda: (0.0, 0.0, 0.0))
    worker._cycle(0)
    result = worker.latest()
    assert result.wait_ms >= SLOW_MS * 0.8, result.wait_ms
    assert result.cycle_ms < SLOW_MS * 0.4, result.cycle_ms
    assert result.detect_ms <= result.cycle_ms


# ── One sleep for every path, and the three passes priced apart ─────────────
class _ClockSpy:
    """Stands in for the ``time`` module inside ``visual_nav``, recording every sleep.

    A shim object rather than a patch of ``time.sleep`` itself: this module's loop reads
    ``time.monotonic`` on every tick and so does the profiler, and monkeypatching the stdlib
    module in place would leak into the tracker and every other importer of it.
    """

    def __init__(self) -> None:
        self.sleeps: list = []

    def monotonic(self):
        return time.monotonic()

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        time.sleep(min(seconds, 0.001))     # keep the suite fast; the VALUE is the point


@contextlib.contextmanager
def _clock_spy():
    spy = _ClockSpy()
    real, visual_nav.time = visual_nav.time, spy
    try:
        yield spy
    finally:
        visual_nav.time = real


class _SlowPoseLoco(_FakeLoco):
    """Odometry that costs more than a whole control period to read.

    Something in the tick has to overrun the period, or every path sleeps ~a period and
    "sleeps the remainder" is indistinguishable from "sleeps a period".
    """

    def __init__(self, cost_s: float) -> None:
        super().__init__()
        self._cost_s = cost_s

    def pose(self):
        time.sleep(self._cost_s)
        return super().pose()


class _StalePerception:
    """One result, permanently older than ``perception_timeout_s``."""

    cycles, errors = 3, 0

    def __init__(self) -> None:
        self._result = PerceptionResult(seq=1, capture_time=time.monotonic() - 5.0,
                                        pose=(0.0, 0.0, 0.0), observations=[], ranged=[])

    def alive(self):
        return True

    def latest(self):
        return self._result


def test_the_stale_hold_sleeps_the_remainder_and_not_a_whole_extra_period():
    """🔴 THE BRANCH THAT FIRES BECAUSE THE LOOP IS SLOW USED TO MAKE IT SLOWER.

    Four sleeps existed. The goal-search and walking paths slept
    ``max(0, period - elapsed)``; the stale-perception hold and the no-result skip slept a
    flat ``period``. On a tick that has already spent 250 ms — the median of the two
    2026-08-25 runs — the first two sleep 0 and the other two add another 100 ms on top, so
    the hold branch cost a whole period MORE than the walking branch, on exactly the ticks
    issue #18 is about.

    Here the pose read alone costs three periods, so the correct remainder is 0.0 and the
    old code asked for the full 0.01 s. Staleness fired on 0 of those 150 ticks, so this is
    not offered as a fix for the rate — what it removes is a tick cost that depended on
    which ``if`` it came through, which ``TickProfiler.snapshot`` would now attribute to the
    hold rather than to the sleep.
    """
    navigator = VisualNavigator(
        loco=_SlowPoseLoco(cost_s=0.030), perception=_StalePerception(),
        planner=_FakePlanner(), tracker=ObstacleTracker(), goal_source=_FakeGoal(),
        health=_FakeHealth(ticks=4), config=NavConfig(live=True, control_hz=100.0))
    with contextlib.redirect_stdout(io.StringIO()), _clock_spy() as spy:
        navigator.run()
    assert spy.sleeps, "the stale branch never slept, so this test proves nothing"
    assert max(spy.sleeps) == 0.0, (
        f"a tick that had already overrun still asked to sleep {max(spy.sleeps)} s; "
        f"the period was 0.01 s")


def test_no_path_through_the_loop_sleeps_a_bare_period():
    """The pin for the above, against the whole method rather than one branch.

    Four sleeps, two of them wrong, is the shape of defect that a test on one branch leaves
    behind. Every path now delegates to ``_sleep_out_the_period``, so the property to pin is
    that ``run`` contains no ``sleep`` call of its own at all.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(VisualNavigator.run)))
    # `ast.unparse` is 3.9+ and CI runs a 3.8 leg over this directory, so the callee is read
    # off the node rather than round-tripped through source.
    called = [getattr(node.func, "attr", getattr(node.func, "id", None))
              for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert "sleep" not in called, called
    body = inspect.getsource(VisualNavigator._sleep_out_the_period)
    assert "period - (time.monotonic() - tick_start)" in body


#: What each fake pass costs, in seconds. Deliberately unequal and in the same ORDER as the
#: module docstring's measured 114 / 69 / 10 ms, so a test that mixed two of them up would
#: have to disagree about which is biggest.
_GOAL_COST_S, _DETECT_COST_S, _COLOUR_COST_S = 0.006, 0.024, 0.003


class _SlowGoalSource:
    def update(self, _image, _pose):
        time.sleep(_GOAL_COST_S)


class _SlowDetector:
    def detect_tiered(self, _image):
        time.sleep(_DETECT_COST_S)
        return [], []


class _SlowColour:
    """A colour tier with a measured prior, so `static_ranged` runs as it would live."""

    profile = PROFILES["bin"]

    def detect(self, _image):
        time.sleep(_COLOUR_COST_S)
        return []


def test_the_perception_cycle_prices_its_three_passes_separately():
    """⛔ `detect_ms` IS ONE NUMBER FOR THREE PASSES, and the question issue #18 ends on
    needs it split.

    `_cycle` starts that clock before the goal pass and stops it after colour segmentation.
    Its median over the two 2026-08-25 runs is 194 and 201 ms, and the module docstring's
    own components — person 114 ms, colour 10 ms, goal 69 ms — sum to 193. Consistent, and
    useless as a lever: if the detector owns the 202 ms then the 300x300 SSD input is the
    thing to change, and if the throttled goal pass owns half of it then its cadence is.
    Nothing in the tree could tell those apart.

    This asserts both halves of the split: the three are priced separately, AND they still
    sum to `detect_ms`, so a consumer parsing that field is not handed a different number
    under the same name.
    """
    worker = PerceptionWorker(_OneFrame(), _SlowDetector(), CAMERA, _SlowGoalSource(),
                              lambda: (0.0, 0.0, 0.0), colour_detector=_SlowColour())
    worker._cycle(0)
    result = worker.latest()
    assert result is not None
    split = result.pass_ms
    assert set(split) == {"goal", "detect", "colour"}, split
    assert split["detect"] > split["goal"] > split["colour"], split
    assert abs(result.detect_ms - sum(split.values())) < 2.0, (
        f"detect_ms {result.detect_ms:.2f} is not goal+detect+colour {split}; it has "
        f"stopped being the sum of the three passes it has always spanned")
    assert result.cycle_ms >= result.detect_ms


class _SplitPerception:
    """A fresh result every tick, carrying a split no control stage could have produced."""

    cycles, errors = 1, 0
    #: Values with no relation to anything this test does, so a line carrying them can only
    #: have got them off the result the tick consumed. A tuple of pairs rather than a dict,
    #: because a mutable class attribute is one shared object and this is handed out.
    SPLIT = (("goal", 69.0), ("detect", 114.0), ("colour", 10.2))

    def __init__(self) -> None:
        self._seq = 0

    def alive(self):
        return True

    def latest(self):
        self._seq += 1
        return PerceptionResult(seq=self._seq, capture_time=time.monotonic(),
                                pose=(0.0, 0.0, 0.0), observations=[], ranged=[],
                                detect_ms=193.2, pass_ms=dict(self.SPLIT))


def test_the_split_travels_from_the_perception_thread_onto_the_line():
    """The wiring, not the writer. Dropping `pass_ms=result.pass_ms` from the one call site
    leaves `write_tick`'s own tests green and every telemetry file without the split — which
    is exactly the state this change is fixing, so it has to be the thing that goes red."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "run.jsonl"
        telemetry = TelemetryWriter(str(path))
        navigator = VisualNavigator(
            loco=_FakeLoco(), perception=_SplitPerception(), planner=_FakePlanner(),
            tracker=ObstacleTracker(), goal_source=_FakeGoal(),
            health=_FakeHealth(ticks=5), config=NavConfig(live=True, control_hz=100.0),
            telemetry=telemetry)
        with contextlib.redirect_stdout(io.StringIO()):
            navigator.run()
        telemetry.close()
        ticks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ticks = [t for t in ticks if t["type"] == "tick"]
    assert ticks, "no tick was written"
    for tick in ticks:
        assert tick["perception"]["pass_ms"] == dict(_SplitPerception.SPLIT), (
            tick["perception"])


def test_the_three_pass_split_reaches_the_telemetry_line():
    """A split that stays in a dataclass answers nothing after the run is over."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "run.jsonl"
        telemetry = TelemetryWriter(str(path))
        telemetry.write_tick(
            elapsed_s=0.0, pose=(0.0, 0.0, 0.0), goal_xy=(4.0, 0.0), goal_distance_m=4.0,
            command=None, obstacles=[], frame_age_s=0.1, perception_seq=1,
            detect_ms=193.2, standing=True, live=False,
            pass_ms={"goal": 69.0, "detect": 114.0, "colour": 10.2})
        telemetry.close()
        tick = json.loads(path.read_text().strip().splitlines()[-1])
    assert tick["perception"]["pass_ms"] == {"goal": 69.0, "detect": 114.0,
                                            "colour": 10.2}


def test_a_producer_that_does_not_split_writes_null_rather_than_an_empty_object():
    """`{}` and `null` read differently: one is "measured nothing", the other is "did not
    measure". Every run committed before this change is the second."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "run.jsonl"
        telemetry = TelemetryWriter(str(path))
        telemetry.write_tick(
            elapsed_s=0.0, pose=(0.0, 0.0, 0.0), goal_xy=(4.0, 0.0), goal_distance_m=4.0,
            command=None, obstacles=[], frame_age_s=0.1, perception_seq=1,
            detect_ms=193.2, standing=True, live=False)
        telemetry.close()
        tick = json.loads(path.read_text().strip().splitlines()[-1])
    assert tick["perception"]["pass_ms"] is None
    assert tick["perception"]["detect_ms"] == 193.2


def test_no_committed_run_carries_the_split_yet():
    """The gap this closes, asserted rather than assumed."""
    runs = (Path(_HERE).resolve().parents[3] / "evidence" / "2026-08-25-peer-runs")
    for name in ("hero-run-telemetry.jsonl", "contrast-run-telemetry.jsonl"):
        ticks = [json.loads(line) for line in (runs / name).read_text().splitlines()
                 if line.strip()]
        ticks = [t for t in ticks if t["type"] == "tick"]
        assert ticks and all("pass_ms" not in t["perception"] for t in ticks)


# ── Resting the legs when the way stays blocked (issue #118) ────────────────
class _OnePlanner:
    """Returns the SAME command every tick, so the loop's dispatch is what is measured.

    The scenes that produce a ``veto-hold`` or a stopped ``veto-avoid`` live in
    ``integration/test_mappo_drive.py`` and are measured there; driving the real policy
    from here would make this a test of what a checkpoint happens to decide.
    """

    config = PlannerConfig()

    def __init__(self, command: Command):
        self._command = command

    def plan(self, *_args, **_kwargs):
        return self._command


def _rest_run(command: Command, ticks: int = 40, rest_after_s: float = 0.05):
    """Drive the real control loop with one fixed command. Returns what got laid down.

    Starts STANDING and LIVE, because the rest branch is guarded on ``self._standing``
    and a dry run never reaches ``lie_down_fn`` at all — a version of this test that
    forgot either passed against a robot that was already prone.

    ``rest_after_s`` is 0.05 s against a 100 Hz loop, so the timer needs about five
    ticks of the forty available. A threshold the fixture's step divides exactly is how
    a gate comes to never fire, so it is left with 8x of margin rather than 1x.
    """
    loco = _FakeLoco()
    rested: list = []
    navigator = VisualNavigator(
        loco=loco, perception=_FakePerception(), planner=_OnePlanner(command),
        tracker=ObstacleTracker(), goal_source=_FakeGoal(), health=_FakeHealth(ticks),
        config=NavConfig(live=True, control_hz=100.0, initially_standing=True,
                         rest_after_s=rest_after_s),
        lie_down_fn=lambda _loco: rested.append("lie_down"))
    with contextlib.redirect_stdout(io.StringIO()):
        navigator.run()
    return rested, loco, navigator


def test_a_vetoed_hold_still_rests_the_legs():
    """⚠️ THE DEFECT IN ISSUE #118, at the site that decides the posture.

    ``integration/mappo_drive.py`` qualifies a vetoed tick's reason instead of replacing
    it, so the planner's own hold reaches this loop as ``veto-hold``. The dispatch tested
    ``command.reason == "hold"``, which is False for it, so ``rest_after_s`` never
    elapsed and the Go2 stood braced under a 3.15 kg arm for the rest of the run — with
    no fault raised, because both stall gates decline a zero command by construction.

    Put ``command.reason == "hold"`` back and this goes red.
    """
    rested, _loco, navigator = _rest_run(
        Command(0.0, 0.0, 0.0, reason="veto-hold", gap_m=0.2, feasible=0, evaluated=330))
    assert rested == ["lie_down"], (
        "a vetoed hold left the robot standing: the rest-after-blocked timer never "
        "started, and nothing else in the loop would have said so")
    assert not navigator._standing


def test_a_vetoed_stop_labelled_avoid_still_rests_the_legs():
    """The half a prefix-only fix would miss, and the one the staged scene produces.

    Measured on the drive path: a policy walking at a bin 2.6 m ahead is vetoed at
    1.11 m of surface gap, and from 1.06 m the command is ``v=(0.00, 0.00, 0.00)`` on
    every remaining tick — labelled ``veto-avoid``, because the stopping-distance cap
    had ratcheted the dynamic window down until the best sampled candidate was zero and
    ``avoid`` is a label applied after that choice. ``base_reason(...) == "hold"`` is
    False on all of them, so teaching the dispatch only about ``veto-hold`` leaves the
    robot braced in exactly the scene the demo runs.
    """
    rested, _loco, _navigator = _rest_run(
        Command(0.0, 0.0, 0.0, reason="veto-avoid", gap_m=1.06, feasible=330,
                evaluated=330))
    assert rested == ["lie_down"], (
        "a stopped robot labelled `veto-avoid` stayed standing; the label is not the "
        "wheels, and this is the scene the veto actually produces")


def test_a_moving_command_never_rests_the_legs():
    """The counter-example that gives the two above their meaning.

    Same loop, same forty ticks, same tiny ``rest_after_s`` — only the velocity changed.
    A rule that lay the robot down here would be worse than the defect it replaced: it
    would stop a run that was walking perfectly well.
    """
    rested, loco, navigator = _rest_run(
        Command(0.30, 0.0, 0.0, reason="veto-avoid", gap_m=1.5, feasible=300,
                evaluated=330))
    assert rested == [], "the robot was walking; nothing should have laid it down"
    assert navigator._standing
    assert any(c[0] > 0.0 for c in loco.commands), (
        f"the fixture never drove the legs, so nothing was proved: {loco.commands}")


def test_arriving_is_not_being_blocked():
    """``arrived`` is the one stop that is not a failure to get on.

    ``mappo_drive`` maps the policy's ``STOP_GOAL_REACHED`` to it, and that can land
    while the stack's own ``arrive_tolerance_m`` is not yet met — so it does reach this
    dispatch, and lying down on it would be the robot going prone because it thinks it
    has finished.
    """
    rested, _loco, navigator = _rest_run(
        Command(0.0, 0.0, 0.0, reason="arrived", gap_m=float("inf"), feasible=330,
                evaluated=330))
    assert rested == [], "an arrival is not a blocked hold"
    assert navigator._standing


def test_blocked_stop_is_decided_by_the_legs_and_by_one_word():
    """The unit behind the four loop tests above, and where the strictness lives.

    Two failure modes, both of which look right written inline. Comparing the whole
    string misses every qualified reason. Searching for a substring — ``"hold" in
    reason`` — fires on a reason that merely contains the letters, which is a different
    decision taken for a spelling coincidence.
    """
    def stop(reason):
        return Command(0.0, 0.0, 0.0, reason=reason, gap_m=1.0)

    assert blocked_stop(stop("hold"))
    assert blocked_stop(stop("veto-hold")), "the reason this issue exists"
    assert blocked_stop(stop("veto-avoid")), "a stop is a stop, whatever labelled it"
    assert blocked_stop(stop("policy")), "the policy commanding zero is still a stop"
    assert not blocked_stop(stop("arrived"))

    # Moving, however it is labelled. A `hold` that is not a stop cannot happen from
    # this planner — both hold branches command zero directly — but the rule has to be
    # the legs rather than the word, or a future producer can lie the robot down while
    # it is walking.
    assert not blocked_stop(Command(0.30, 0.0, 0.0, reason="hold", gap_m=1.0))
    assert not blocked_stop(Command(0.0, 0.12, 0.0, reason="veto-hold", gap_m=1.0))
    assert not blocked_stop(Command(0.0, 0.0, 0.35, reason="hold", gap_m=1.0))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"visual_nav: {len(tests)}/{len(tests)} passed")
