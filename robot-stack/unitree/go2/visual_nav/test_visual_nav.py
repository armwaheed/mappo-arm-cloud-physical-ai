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

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspect

import numpy as np

import person_detector
import visual_nav
from avoidance import (
    STATIC_SOFT_GAP_M,
    Command,
    DynamicWindowPlanner,
    PlannerConfig,
)
from camera import Frame
from camera_model import FisheyeCamera
from colour_detector import PROFILES
from goal import ArucoGoal, OdomWaypoint
from static_map import StaticObstacleMap
from tracker import PROCESS_ACCEL_SIGMA, Observation, ObstacleTracker
from visual_nav import (
    NavConfig,
    PerceptionResult,
    PerceptionWorker,
    VisualNavigator,
    build_parser,
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
               static_map=None) -> VisualNavigator:
    """A navigator wired to nothing but the planner and tracker ``_obstacles`` reads.

    ``_tracker_time`` is set directly because the loop that normally maintains it is
    the loop under test; everything else _obstacles touches is passed in properly.
    """
    navigator = VisualNavigator(
        loco=None, perception=None,
        planner=DynamicWindowPlanner(config=PlannerConfig()),
        tracker=tracker, goal_source=None, health=None, config=NavConfig(),
        static_map=static_map)
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


def test_raw_recorder_is_the_last_parameter_so_the_vendored_call_still_works():
    """`integration/mappo_drive.py` calls this constructor positionally through
    `navigator_factory` and states that a new positional argument must not be added.
    """
    parameters = list(inspect.signature(VisualNavigator.__init__).parameters)
    assert parameters[-1] == "raw_recorder", parameters
    assert parameters.index("static_map") < parameters.index("raw_recorder")
    assert parameters.index("telemetry") < parameters.index("raw_recorder")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"visual_nav: {len(tests)}/{len(tests)} passed")
