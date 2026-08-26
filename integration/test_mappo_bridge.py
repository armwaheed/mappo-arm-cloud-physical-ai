#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the telemetry -> policy-input mapping.

Every test here is a mapping that a reasonable person gets wrong, and the reason it is
wrong is written into the test. That is the point of the file: the mapping is four lines
of arithmetic and three decisions, and it is the three decisions that will bite.

These run against ``../evidence/sample_telemetry.jsonl`` — a real run on the robot — as
well as synthetic ticks, so the contract is checked against the thing that produced it.

Pure stdlib, no policy package needed. Run: ``python3 test_mappo_bridge.py``
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mappo_bridge import (
    HOLD_LABELS,
    VELOCITY_FRAME,
    BridgeReport,
    audit,
    external_hold,
    holds_the_robot,
    is_stationary,
    policy_objects,
    robot_input,
)
from telemetry_reader import read_run

EVIDENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "evidence", "sample_telemetry.jsonl")

BIN = {"label": "bin", "kind": "static", "id": "landmark-1",
       "x": 2.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "radius_m": 0.3}
WALKER = {"label": "person", "kind": "tracked", "id": "track-7",
          "x": 3.0, "y": 1.0, "vx": 0.6, "vy": 0.0, "radius_m": 0.5}
STOPPED_PERSON = {**WALKER, "vx": 0.0, "vy": 0.0}
# A peer quadruped: a track, but not person-SHAPED, and barely moving. Note the label
# is deliberately a plausible-but-wrong one — routing must not depend on it. On live
# frames this same peer came back labelled `person` 12 times out of 12.
PARKED_PEER = {"label": "lite3", "kind": "tracked", "id": "track-3",
               "person_shaped": False,
               "x": 2.5, "y": -0.4, "vx": 0.0, "vy": 0.0, "radius_m": 0.35}
SHUFFLING_PEER = {**PARKED_PEER, "vx": 0.10, "vy": 0.05}
# A PERSON the detector mislabelled. Across the 2026-08-24 corpus the peer came back as
# `motorbike` 613 times, so the label lands on people too — and `motorbike` is not in
# HOLD_LABELS. Shape is the only thing standing between this and the policy.
MISLABELLED_PERSON = {"label": "motorbike", "kind": "tracked", "id": "track-9",
                      "person_shaped": True,
                      "x": 2.0, "y": 0.2, "vx": 0.05, "vy": 0.0, "radius_m": 0.5}
CHARGING_PEER = {**PARKED_PEER, "vx": 0.60, "vy": 0.00}


def _tick(**overrides):
    tick = {
        "type": "tick", "t": 1.0, "wall_time": 1_786_492_453.0,
        "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "goal": {"x": 4.0, "y": 0.0, "distance_m": 4.0},
        "measured": {"vx": 0.12, "vy": -0.03, "wz": 0.0},
        "command": {"vx": 0.1, "vy": 0.0, "wz": 0.0, "reason": "goal"},
        "obstacles": [], "perception": {"stale": False},
    }
    tick.update(overrides)
    return tick


# ── The frame, which is the mapping most likely to be got wrong ─────────────
def test_the_measured_velocity_is_declared_as_body_frame():
    """The policy package's own integration note passes ``velocity_frame="odom"``.

    It is body: ``measured`` comes from ``Go2Locomotion.velocity()``, which reads
    ``SportModeState_.velocity``, the estimator's body-frame vector. Getting this wrong
    is invisible on a bench — the two frames agree EXACTLY at yaw 0 and diverge only as
    the robot turns — so nothing but a test or a corner catches it.
    """
    assert VELOCITY_FRAME == "body"
    assert robot_input(_tick())["velocity_frame"] == "body"


def test_a_tick_with_no_measured_velocity_does_not_substitute_the_command():
    """Commanded is not measured, and the difference is a whole class of failure: a run
    that commanded 0.12 m/s for fifty seconds and moved nothing reads, in the command
    alone, exactly like a run that was walking. Zero is at least honest about being an
    assumption, and :func:`audit` counts it."""
    mapped = robot_input(_tick(measured=None, command={"vx": 0.3, "vy": 0.0, "wz": 0.0,
                                                       "reason": "goal"}))
    assert mapped["vx_mps"] == 0.0 and mapped["vy_mps"] == 0.0
    assert audit(_tick(measured=None))["velocity_missing"] == 1


# ── The hold, which is the mapping most likely to make the whole thing inert ─
def test_a_hold_caused_only_by_the_static_obstacle_is_not_an_external_hold():
    """THE ONE THAT MAKES THE INTEGRATION A NO-OP IF IT IS WRONG.

    ``external_hold`` is specified as the existing moving-object stop/wait logic. The
    planner's ``reason`` does not carry that distinction: it emits ``hold`` whenever no
    candidate clears EVERY obstacle, mover or landmark. Forwarding a bin-caused hold
    zeroes the policy's command in exactly the scene the policy is there to solve.
    """
    assert external_hold(_tick(command={"vx": 0.0, "vy": 0.0, "wz": 0.0,
                                        "reason": "hold"},
                               obstacles=[BIN])) is False


def test_a_hold_with_a_mover_in_the_scene_is_an_external_hold():
    assert external_hold(_tick(command={"vx": 0.0, "vy": 0.0, "wz": 0.0,
                                        "reason": "hold"},
                               obstacles=[BIN, WALKER])) is True


def test_a_blind_tick_holds_even_though_the_planner_did_not():
    """A policy acting on a frozen world model is worse than one that does not act. The
    stack records this as ``perception.stale`` and keeps its goal, so the tick is
    otherwise complete and would replay as if nothing were wrong."""
    assert external_hold(_tick(perception={"stale": True})) is True


def test_a_normal_avoiding_tick_is_not_a_hold():
    assert external_hold(_tick(command={"vx": 0.2, "vy": 0.1, "wz": 0.0,
                                        "reason": "avoid"}, obstacles=[BIN])) is False


# ── Splitting movers from landmarks ─────────────────────────────────────────
def test_a_stopped_person_is_still_a_mover():
    """The case every velocity heuristic gets wrong, and the reason ``kind`` exists.

    A person who has stopped has exactly the velocity of a bin and the same claim on the
    lane, but handing them to the policy as something to path around is the opposite of
    the agreed division of labour — and they can step into the gap the policy commits to.
    """
    assert is_stationary(STOPPED_PERSON) is False
    assert is_stationary(BIN) is True


def test_a_log_written_before_kind_existed_still_maps():
    """Falls back to speed, because runs recorded before the field was added are the
    only runs that exist. It is a stopgap and :func:`audit` says so out loud."""
    legacy_bin = {k: v for k, v in BIN.items() if k not in ("kind", "id")}
    legacy_walker = {k: v for k, v in WALKER.items() if k not in ("kind", "id")}
    assert is_stationary(legacy_bin) is True
    assert is_stationary(legacy_walker) is False
    counted = audit(_tick(command={"vx": 0.0, "vy": 0.0, "wz": 0.0, "reason": "hold"},
                          obstacles=[legacy_bin]))
    assert counted["hold_classified_by_speed"] == 1
    assert counted["unidentified_objects"] == 1


def test_a_person_always_holds_the_robot_however_fast_they_are_going():
    """The tier boundary. A person is not something to path around with a policy trained
    on static discs, and a STOPPED person is still a person — they have a bin's velocity
    and a person's claim on the lane, which is exactly when the distinction decides
    behaviour. Label, not speed, makes this call."""
    objects = policy_objects(_tick(obstacles=[BIN, WALKER, STOPPED_PERSON]))
    assert [o["object_id"] for o in objects] == ["landmark-1"]
    assert external_hold(_tick(obstacles=[STOPPED_PERSON],
                               command={"reason": "hold"})) is True


def test_a_parked_peer_reaches_the_policy_instead_of_stopping_the_robot():
    """THE POINT OF THE CHANGE. A peer used to be a track, every track was a hold, and a
    hold is a single boolean meaning stop — so the peer avoidance in a MULTI-agent demo
    was 100% incumbent planner and 0% policy. A parked peer is a disc at a known place,
    which is precisely what the policy was trained to path around."""
    objects = policy_objects(_tick(obstacles=[BIN, PARKED_PEER]))
    assert [o["object_id"] for o in objects] == ["landmark-1", "track-3"]
    assert objects[1]["radius_m"] == 0.35, "the peer's own footprint, not a default"
    assert external_hold(_tick(obstacles=[PARKED_PEER],
                               command={"reason": "hold"})) is False


def test_a_shuffling_peer_still_reaches_the_policy():
    """Manoeuvring, not travelling: below the gait floor of any quadruped here, so it has
    barely moved by the time the command lands — which is the only condition under which
    an observation with no velocity channel can be right about it."""
    assert [o["object_id"] for o in policy_objects(
        _tick(obstacles=[SHUFFLING_PEER]))] == ["track-3"]


def test_a_fast_mover_holds_the_robot_whatever_it_is():
    """The speed gate is the safety half of the two-tier split, and it is not about
    class. The policy's observation has NO obstacle-velocity channel, so a mover enters
    as an instantaneous disc; with a 0.875 m horizon at 10 Hz, something crossing with
    intent is a problem it was never trained on. Anything above the threshold holds, even
    a peer robot, even though a slow one would not."""
    assert policy_objects(_tick(obstacles=[CHARGING_PEER])) == []
    assert external_hold(_tick(obstacles=[CHARGING_PEER],
                               command={"reason": "hold"})) is True


def test_a_peer_whose_pose_stopped_arriving_holds_the_robot():
    """A mesh peer link that has gone quiet is the same blindness as a stale camera, one
    input over — and unlike a stale camera the obstacle is GONE, because a position that
    can no longer be dated is not a position. ``peer_source`` drops the disc; this is the
    other half of that decision, and separating the two is what would make it unsafe.

    Note it holds with an EMPTY obstacle list and no planner hold at all, which is exactly
    the situation: there is nothing left to be blocked by."""
    assert external_hold(_tick(obstacles=[], command={"reason": "goal"},
                               peer_link={"lost": True,
                                          "reason": "peer link: peer-a is 0.9 s old"})) is True


def test_a_peer_link_that_is_healthy_does_not_hold():
    """Otherwise the test above would pass on a bridge that holds unconditionally."""
    assert external_hold(_tick(peer_link={"lost": False, "reason": ""})) is False


def test_a_tick_from_before_the_peer_link_existed_never_holds_for_one():
    """Every run in ``evidence/`` predates the field. Absent must read as "no peer link
    configured" and not as "lost", or replaying the recorded corpus reports a robot that
    should have been standing still for 122 ticks."""
    assert external_hold(_tick()) is False
    run = read_run(EVIDENCE)
    assert not any(t.get("peer_link") for t in run.ticks)
    assert not any(external_hold(t) and not t.get("obstacles") for t in run.ticks)


def test_a_mapped_landmark_never_holds_however_it_is_labelled():
    """A hold for the bin would zero the policy in the one scene it exists for."""
    assert external_hold(_tick(obstacles=[BIN], command={"reason": "hold"})) is False


# ── The geometry ────────────────────────────────────────────────────────────
def test_bearing_is_measured_from_the_nose_and_is_positive_to_the_left():
    """``StationaryObject`` documents +left / CCW. An object at odom +y with the robot
    facing +x is on its left, so the bearing is +90 degrees, not -90."""
    left = {**BIN, "x": 0.0, "y": 2.0}
    mapped = policy_objects(_tick(obstacles=[left]))[0]
    assert abs(mapped["bearing_rad"] - math.pi / 2) < 1e-9
    assert abs(mapped["distance_m"] - 2.0) < 1e-9


def test_the_bearing_follows_the_robot_rather_than_the_map():
    """Telemetry is odom so a landmark stays put while the robot walks; the policy wants
    it relative to the nose. Turn 90 degrees left and the same object moves to the right
    of the robot — if it does not, the yaw was dropped somewhere."""
    ahead = {**BIN, "x": 2.0, "y": 0.0}
    turned = _tick(pose={"x": 0.0, "y": 0.0, "yaw": math.pi / 2}, obstacles=[ahead])
    mapped = policy_objects(turned)[0]
    assert abs(mapped["bearing_rad"] + math.pi / 2) < 1e-9
    assert abs(mapped["distance_m"] - 2.0) < 1e-9


def test_the_radius_is_passed_through_rather_than_defaulted():
    """It already carries the map's position uncertainty and the planner treats it as a
    hard footprint. ``StationaryObject`` would otherwise default to 0.15 m, which is half
    the smallest radius this stack has ever reported for the staged bin."""
    assert policy_objects(_tick(obstacles=[BIN]))[0]["radius_m"] == 0.3


# ── Things that must not be invented ────────────────────────────────────────
def test_a_tick_with_no_goal_has_no_policy_input():
    """The robot is searching. A zero-filled input is a goal at the origin, and the
    policy would drive at it."""
    assert robot_input(_tick(goal=None)) is None
    assert audit(_tick(goal=None))["no_goal"] == 1


def test_the_wall_clock_is_never_passed_as_the_policy_timestamp():
    """A GATE THAT WOULD NEVER FIRE.

    The policy compares ``timestamp_s`` against its own ``time.monotonic()``. Hand it
    ``wall_time`` — a Unix epoch — and the computed age is about -1.8e9 seconds, under
    any threshold, so ``STOP_STALE_INPUT`` can never trigger. It fails OPEN, in the
    direction where a frozen world model keeps driving the legs.
    """
    assert robot_input(_tick())["timestamp_s"] is None
    assert robot_input(_tick(), monotonic_s=1234.5)["timestamp_s"] == 1234.5
    # The mapping does not read `wall_time` at all, so no future edit can reach for it
    # as a convenient default.
    assert robot_input(_tick(wall_time=0.0))["timestamp_s"] is None


def test_a_null_field_is_a_legal_recording_and_does_not_crash_the_mapping():
    """JSON has no infinity, so the writer emits ``null`` for anything non-finite.

    A pose or goal that arrives that way cannot become a policy input, and says so the
    same way a searching tick does. A velocity that arrives that way is substituted, and
    counted — the substitution is an assumption, not a measurement.
    """
    assert robot_input(_tick(pose={"x": None, "y": 0.0, "yaw": 0.0})) is None
    assert robot_input(_tick(goal={"x": 1.0, "y": None, "distance_m": None})) is None
    null_velocity = _tick(measured={"vx": None, "vy": None, "wz": None})
    assert robot_input(null_velocity)["vx_mps"] == 0.0
    assert audit(null_velocity)["velocity_missing"] == 1


def test_a_half_recorded_obstacle_is_dropped_rather_than_placed_at_the_origin():
    """``None`` coordinates would otherwise become an obstacle at the robot's own
    position, which every ray reports as a collision."""
    broken = {**BIN, "x": None}
    assert policy_objects(_tick(obstacles=[broken, BIN])) != []
    assert len(policy_objects(_tick(obstacles=[broken, BIN]))) == 1


def test_a_mistyped_counter_is_rejected_rather_than_silently_ignored():
    """A counter that stays at zero because its key was misspelled reads exactly like a
    clean mapping, which is the one thing this report must never claim falsely."""
    try:
        BridgeReport().merge(no_gaol=1)
    except TypeError:
        return
    raise AssertionError("accepted an unknown counter")


def test_reset_run_is_the_callers_decision():
    assert robot_input(_tick(), reset_run=True)["reset_run"] is True
    assert robot_input(_tick())["reset_run"] is False


# ── Against the real run ────────────────────────────────────────────────────
def test_every_tick_of_a_recorded_run_maps_without_a_gap():
    """The whole file, not a fixture written to agree with the code.

    121 of its 122 ticks carry a mapped bin and all 122 carry a goal and a measured
    velocity, so a mapping that silently dropped any of the three would show up here as
    a count rather than as a crash.
    """
    run = read_run(EVIDENCE)
    mapped = [robot_input(tick) for tick in run.ticks]
    assert len(run.ticks) == 122
    assert all(m is not None for m in mapped), "a tick lost its goal"
    assert all(m["velocity_frame"] == "body" for m in mapped)
    with_objects = [m for m in mapped if m["stationary_objects"]]
    assert len(with_objects) == 121
    # Every object in this run is the one mapped bin, seen from a moving robot.
    assert all(o["distance_m"] > 0.0 for m in with_objects
               for o in m["stationary_objects"])


def test_the_recorded_run_predates_kind_and_the_audit_says_so():
    """This file was written before the field existed, which is exactly the situation
    the fallback is for — and exactly the situation an integrator must be told about
    rather than left to infer from silence."""
    run = read_run(EVIDENCE)
    totals = {}
    for tick in run.ticks:
        for key, value in audit(tick).items():
            totals[key] = totals.get(key, 0) + value
    assert totals["ticks"] == 122
    assert totals["unidentified_objects"] == 192
    assert totals["velocity_missing"] == 0
    assert totals["no_goal"] == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"mappo_bridge: {len(tests)}/{len(tests)} passed")



def test_a_person_shaped_obstacle_holds_even_when_its_label_is_not_in_hold_labels():
    """THE HOLE THIS CLOSES. Every other fixture here has a label that agrees with its
    shape, so deleting the shape check from `holds_the_robot` used to leave the whole
    suite green — the routing rule was unpinned. This is a person the detector called
    `motorbike`, which is not in HOLD_LABELS: under the old label-only rule the robot
    would have handed them to the policy, silently, at 0.05 m/s.

    Delete the `person_shaped` branch in `holds_the_robot` and this fails.
    """
    assert MISLABELLED_PERSON["label"] not in HOLD_LABELS, "the premise of the test"
    assert holds_the_robot(MISLABELLED_PERSON) is True
    assert [o["object_id"] for o in policy_objects(
        _tick(obstacles=[MISLABELLED_PERSON]))] == [], "must never reach the policy"


def test_shape_defaults_to_holding_when_a_producer_omits_it():
    """An older telemetry writer, or any producer that does not judge shape, must land
    on the STOPPING side rather than being waved through. `holds_the_robot` reads the
    field with a default of True for exactly this."""
    legacy = {k: v for k, v in PARKED_PEER.items() if k != "person_shaped"}
    assert "person_shaped" not in legacy
    assert holds_the_robot(legacy) is True
