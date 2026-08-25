#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for peer poses arriving over the mesh, with a fake mesh source.

There is no robot and no Device Connect here. The fake source is a spool file written
with the same :func:`peer_source.write_spool` ``dashboard/peer_link.py`` calls, so what
these drive is the real reader, the real staleness arithmetic and the real geometry —
only the thing that fills the file is fake, and it is fake at the same seam the real one
writes at.

**Most of this file is about staleness**, which is deliberate. The obstacle geometry is
four lines of trigonometry and a reviewer can check it by eye; the failure that actually
gets a robot into another robot is a pose that stopped arriving and was treated as a peer
standing still. Every branch that could report "nothing to worry about" while not knowing
where the peer is has a test whose name says what it would cost.

Each test states what makes it FAIL, because a staleness test that cannot fail is the
specific way this repository has been bitten before — a latch check that proved an arm
was held by asserting its joints had stopped moving, which an unpowered arm passes.

Pure stdlib, no policy package. Run: ``python3 test_peer_source.py``
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mappo_bridge import HOLD_LABELS, external_hold, holds_the_robot, policy_objects
from mappo_policy import tick_from_state
from peer_source import (
    PEER_ACCEL_SIGMA_M_S2,
    PEER_LABEL,
    PEER_RADIUS_M,
    PEER_TIMEOUT_S,
    SCHEMA,
    Alignment,
    PeerSource,
    spool_document,
    write_spool,
)

#: A clock domain both sides agree on, so the boot-id guard is satisfied except where a
#: test is deliberately exercising it.
DOMAIN = "test-boot"

#: The navigator's own monotonic clock reading. An arbitrary large number on purpose:
#: monotonic clocks count from boot, so a test that used numbers near zero would agree
#: with a wall clock by accident and stop being able to tell them apart.
NOW = 98_765.0

PEER = "mappo-go2-peer"


def _spool(tmpdir, peers=None, expect=(PEER,), written=NOW, domain=DOMAIN):
    """Write a spool file the way ``peer_link.py`` writes one, and return its path."""
    path = os.path.join(tmpdir, "peers.json")
    write_spool(path, spool_document(peers or {}, expect, domain=domain,
                                     written_monotonic_s=written))
    return path


def _pose(received=NOW, x=2.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, **extra):
    record = {"received_monotonic_s": received, "x": x, "y": y, "yaw": yaw,
              "vx": vx, "vy": vy, "radius_m": PEER_RADIUS_M, "platform": "go2"}
    record.update(extra)
    return record


#: The identity transform: what a bench looks like, where both robots really do start on
#: the same spot. Never a default on the command line — see ``Alignment``.
IDENTITY = Alignment()


def _source(tmpdir, alignment=IDENTITY, **kwargs):
    return PeerSource(os.path.join(tmpdir, "peers.json"), alignment,
                      domain=DOMAIN, **kwargs)


# ── The shape, which is the whole point of the design ───────────────────────
def test_a_peer_arrives_in_the_shape_visual_nav_already_emits():
    """The claim the design rests on: nothing downstream changes. If a key here is
    renamed or dropped, ``mappo_bridge.policy_objects`` stops seeing the peer, the
    telemetry writer records a half obstacle, and neither fails loudly."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose()})
        link = _source(tmp).read(NOW)
        assert not link.holds, link.reason()
        peer = link.obstacles[0]
        # Exactly the fields `telemetry._obstacle_record` writes for a planner Obstacle.
        assert set(peer) == {"label", "kind", "id", "x", "y", "vx", "vy", "radius_m"}
        assert peer["id"] == f"peer-{PEER}"


def test_a_peer_is_tracked_and_not_static_or_the_speed_gate_disappears():
    """``kind="static"`` would make a peer a mapped landmark, and a landmark NEVER holds
    the robot — ``mappo_bridge.holds_the_robot`` returns False for one before it ever
    looks at speed. A peer charging across the lane would then be handed to a policy with
    no obstacle-velocity channel. Fails if the kind is changed to "static"."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(vx=0.60)})
        peer = _source(tmp).read(NOW).obstacles[0]
        assert peer["kind"] == "tracked"
        assert holds_the_robot(peer) is True, "a peer at 0.60 m/s must stop the robot"


def test_a_parked_peer_reaches_the_policy_rather_than_stopping_the_robot():
    """The other half of the same gate, and the reason the mesh route is worth building:
    a peer that is standing still is a disc at a known place, which is precisely what the
    policy was trained to path around."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(x=2.0, y=0.5)})
        peer = _source(tmp).read(NOW).obstacles[0]
        tick = {"pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "obstacles": [peer]}
        assert [o["object_id"] for o in policy_objects(tick)] == [f"peer-{PEER}"]


def test_the_peer_label_cannot_be_routed_to_the_stop_tier_by_a_platform_name():
    """``HOLD_LABELS`` is matched on ``label``. Publishing the platform name there would
    make the tier depend on a string arriving over a network, and a platform ever called
    "person" would silently stop the robot for every peer forever."""
    assert PEER_LABEL not in HOLD_LABELS
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(platform="person")})
        assert _source(tmp).read(NOW).obstacles[0]["label"] == PEER_LABEL


def test_the_radius_is_the_half_diagonal_and_not_the_half_length():
    """0.70 x 0.31 m gives a half-LENGTH of 0.35 and a half-DIAGONAL of 0.383. A disc of
    0.35 leaves the corners outside it, and nothing here controls which way a peer is
    facing. ``avoidance.NavConfig.robot_radius_m`` made the same call for this robot's own
    body and rounded up to 0.40; a peer gets the same treatment. The colour-prop path's
    0.15 m is a bin and would under-model a quadruped by more than half."""
    assert math.hypot(0.35, 0.155) < PEER_RADIUS_M, "a disc smaller than the diagonal"
    assert PEER_RADIUS_M > 2.5 * 0.15, "the colour-prop default is a bin, not a robot"
    with tempfile.TemporaryDirectory() as tmp:
        # A peer that publishes no footprint falls back to this file's, rather than to
        # anything the planner would default to.
        record = _pose()
        del record["radius_m"]
        _spool(tmp, {PEER: record})
        assert _source(tmp).read(NOW).obstacles[0]["radius_m"] == PEER_RADIUS_M


# ── Staleness: the safety property ──────────────────────────────────────────
def test_a_pose_that_stops_arriving_is_not_a_peer_standing_still():
    """THE test in this file. The spool is unchanged and its peer is at the same
    coordinates it always was; the only thing that happened is that time passed. Before
    the timeout that is a peer being tracked, after it that is a peer whose position is
    unknown — and the two must not produce the same obstacle list.

    Fails if the age check is removed, if it is computed from the peer's own clock, or if
    a lost peer keeps contributing a disc."""
    quiet = _pose(received=NOW)
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: quiet}, written=NOW)
        source = _source(tmp)

        fresh = source.read(NOW + 0.1)
        assert not fresh.holds and len(fresh.obstacles) == 1

        # peer_link is alive and still writing — its heartbeat advances. What stopped is
        # the PEER: the same record, at the same coordinates, not being replaced. That is
        # the case this whole file exists for, and it is not the same as a dead link.
        later = NOW + PEER_TIMEOUT_S + 0.01
        _spool(tmp, {PEER: quiet}, written=later)
        lost = source.read(later)
        assert lost.link_error == "", "the link is alive; only the peer went quiet"
        assert lost.holds is True
        assert lost.obstacles == [], "a peer that cannot be located is not a disc"
        assert PEER in lost.lost


def test_dropping_the_obstacle_and_holding_the_robot_are_one_decision():
    """These are separable in the code and must never be separated in behaviour: the
    dropped disc is only safe BECAUSE the legs stop. Swept across the whole age range so
    a future boundary change cannot open a window where a peer is neither modelled nor
    held."""
    quiet = _pose(received=NOW)
    with tempfile.TemporaryDirectory() as tmp:
        source = _source(tmp)
        for step in range(200):
            age = step * 0.01
            # A live link whose peer has gone quiet: heartbeat current, record frozen.
            _spool(tmp, {PEER: quiet}, written=NOW + age)
            link = source.read(NOW + age)
            modelled = [o for o in link.obstacles if o["id"].endswith(PEER)]
            assert modelled or link.holds, f"peer neither modelled nor held at {age:.2f}s"


def test_a_dead_link_holds_even_when_every_peer_record_looks_fresh():
    """The heartbeat is a separate question from any peer's age, and it has to be asked
    separately: a ``peer_link.py`` that died leaves a file whose contents look exactly
    like a link with nothing new to say. Constructed so the peer record is impossibly
    fresh and only the heartbeat is old — if the heartbeat check is deleted, this reads
    as a healthy link."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(received=NOW)}, written=NOW - 5.0)
        link = _source(tmp).read(NOW)
        assert link.holds is True
        assert "not running" in link.reason()
        assert link.obstacles == []


def test_a_missing_spool_holds_rather_than_reading_as_a_mesh_with_no_peers():
    """``peer_link.py`` never started looks identical, from the robot, to a mesh nobody
    else is on. Only one of those is safe to drive in, and the file cannot tell which, so
    it holds and says which process is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        link = _source(tmp).read(NOW)
        assert link.holds is True
        assert "peer_link.py is not running" in link.reason()


def test_a_spool_from_another_boot_is_refused_rather_than_read_as_the_future():
    """A monotonic reading means nothing outside its own boot. A spool that survived a
    reboot has stamps far AHEAD of the new clock, so every age is large and negative — and
    a negative age is under every threshold ever written, which is the fail-OPEN direction
    and exactly the -1.8e9 s bug ``mappo_bridge.robot_input`` documents.

    Fails if the boot-id comparison is dropped AND if negative ages are clamped to zero."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(received=NOW + 3600.0)}, written=NOW + 3600.0,
               domain="a-different-boot")
        link = _source(tmp).read(NOW)
        assert link.holds is True
        assert "different boot" in link.reason()


def test_a_heartbeat_from_the_future_holds_on_its_own():
    """The link's own timestamp needs the same sign check the per-peer ones get, and it
    needs its OWN test: with a plausible peer record beside it, a future heartbeat is
    otherwise caught by the peer check and the link check can be deleted without anything
    going red. Found by mutation-testing this file — the guard survived until this
    existed."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(received=NOW)}, written=NOW + 3600.0)
        link = _source(tmp).read(NOW)
        assert link.holds is True
        assert "future" in link.reason(), link.reason()
        assert link.obstacles == []


def test_a_future_timestamp_holds_even_where_there_is_no_boot_id_to_compare():
    """The boot id is best-effort — there is none to read outside Linux — so the sign of
    the age has to be a check in its own right. Same file as above with the domains
    agreeing, which is what a macOS workstation or a container without /proc looks like."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(received=NOW + 3600.0)}, written=NOW + 3600.0)
        link = _source(tmp).read(NOW)
        assert link.holds is True
        assert "future" in link.reason()


def test_the_peers_own_sampling_lag_is_charged_to_the_age():
    """``sample_age_at_emit_s`` is a duration measured on the peer, which is why it can
    cross the mesh at all. Ignoring it would make a pose that was already half a second
    stale when it was emitted look brand new on arrival.

    Constructed to sit on the wrong side of the limit ONLY once the lag is added, so it
    fails the moment the term is dropped."""
    lag = 0.2
    with tempfile.TemporaryDirectory() as tmp:
        arrival_age = PEER_TIMEOUT_S - lag + 0.01     # under the limit on its own
        _spool(tmp, {PEER: _pose(received=NOW, sample_age_at_emit_s=lag)},
               written=NOW + arrival_age)
        link = _source(tmp).read(NOW + arrival_age)
        assert link.holds is True, "the peer's own lag was not counted"


def test_an_expected_peer_that_has_never_reported_holds():
    """A roster entry with no record is a robot that was supposed to be there and is not.
    Reporting "no peers lost" for it would be the emptiest possible all-clear."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {}, expect=(PEER,))
        link = _source(tmp).read(NOW)
        assert link.holds is True
        assert link.lost[PEER] == "has never reported a pose"


def test_an_empty_roster_cannot_report_a_missing_peer_and_says_so():
    """Every check in this file is per-expected-peer, so an empty roster makes all of
    them vacuous and the link reports clear while watching nothing. That is a
    misconfiguration, not a peer-free run — peer avoidance is off by not passing the
    flag, not by watching an empty list."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {}, expect=())
        link = _source(tmp).read(NOW)
        assert link.holds is True
        assert "never report one missing" in link.reason()


def test_a_schema_this_reader_does_not_know_holds():
    """Additions are free and a rename bumps the schema; a reader that guessed at an
    unknown one would read a field that has moved and get a plausible number."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "peers.json")
        document = spool_document({PEER: _pose()}, (PEER,), domain=DOMAIN,
                                  written_monotonic_s=NOW)
        document["schema"] = "mappo.peer_link/99"
        write_spool(path, document)
        assert _source(tmp).read(NOW).holds is True


def test_a_torn_or_corrupt_spool_costs_one_tick_and_not_the_run():
    """``write_spool`` renames rather than truncating, so this should be unreachable —
    which is not a safety argument, so it is handled. A parse failure holds for that tick
    and the next good write recovers."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "peers.json")
        with open(path, "w") as handle:
            handle.write('{"schema": "mappo.pee')
        source = _source(tmp)
        assert source.read(NOW).holds is True
        _spool(tmp, {PEER: _pose()})
        assert source.read(NOW).holds is False


def test_a_pose_that_is_not_a_number_is_rejected_rather_than_planned_against():
    """NaN is the dangerous one and it is why ``_finite`` exists rather than a truth test:
    every comparison against NaN is False, so a NaN obstacle centre makes the planner's
    rollout report clear space in the exact direction of the peer."""
    for field in ("x", "y", "yaw", "vx", "vy"):
        with tempfile.TemporaryDirectory() as tmp:
            _spool(tmp, {PEER: _pose(**{field: None})})
            link = _source(tmp).read(NOW)
            assert link.holds is True, f"a null {field} was accepted"
            assert link.obstacles == []


def test_a_wall_clock_reading_holds_the_robot_rather_than_driving_it():
    """The other side of the clock trap. ``read()`` documents that it takes a monotonic
    reading; handing it ``time.time()`` computes ages of about 1.8e9 s. That is the SAFE
    direction of the same mistake — everything reads as lost and the robot stops — and
    this pins it there, because the alternative is a mis-signed subtraction that fails
    open."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(received=NOW)}, written=NOW)
        assert _source(tmp).read(1_786_492_453.0).holds is True


# ── The geometry, which is where a plausible wrong answer comes from ────────
def test_the_alignment_puts_the_peer_where_the_tape_measure_says():
    """A peer standing on its own origin, when it was switched on 2 m ahead and 1 m to the
    left of this robot, is at (2, 1) here. Without the transform it would be at (0, 0),
    which is under this robot's feet — a plausible number and a completely wrong one."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(x=0.0, y=0.0, yaw=0.0)})
        peer = _source(tmp, Alignment.parse("2.0,1.0,0")).read(NOW).obstacles[0]
        assert abs(peer["x"] - 2.0) < 1e-9 and abs(peer["y"] - 1.0) < 1e-9


def test_the_alignment_rotates_the_peers_frame_as_well_as_shifting_it():
    """A peer facing 180 degrees from this robot reports +x for what is, here, -x. A
    transform that only translated would put a peer walking towards this robot into the
    obstacle list as one walking away."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(x=1.0, y=0.0, yaw=0.0, vx=0.3, vy=0.0)})
        peer = _source(tmp, Alignment.parse("4.0,0,180")).read(NOW).obstacles[0]
        assert abs(peer["x"] - 3.0) < 1e-6, peer          # 4 - 1, not 4 + 1
        assert peer["vx"] < 0.0, "a peer approaching was recorded as receding"


def test_the_translation_is_never_applied_to_a_velocity():
    """A rotation and a translation are one call away from each other and applying the
    translation to a velocity produces a number that looks like a measurement: a peer
    parked 2 m away would report 2 m/s of drift and be sent to the stop tier by the speed
    gate on every tick of every run."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(x=0.0, y=0.0, vx=0.0, vy=0.0)})
        peer = _source(tmp, Alignment.parse("2.0,1.0,45")).read(NOW).obstacles[0]
        assert math.hypot(peer["vx"], peer["vy"]) < 1e-9, "a parked peer was given speed"
        assert holds_the_robot(peer) is False


def test_the_peers_velocity_is_rotated_out_of_its_own_body_frame_first():
    """``Go2Locomotion.velocity()`` is BODY frame — that is what ``VELOCITY_FRAME`` says
    and it is the mapping this repository has already got wrong once. An obstacle's
    velocity is odom. A peer facing +90 degrees in its own frame, walking forward at
    0.3 m/s, is moving along +y and not +x.

    Invisible at yaw 0, which is why the test uses a turned peer: the two frames agree
    exactly while a robot faces its start heading."""
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(yaw=math.pi / 2, vx=0.3, vy=0.0)})
        peer = _source(tmp).read(NOW).obstacles[0]
        assert abs(peer["vx"]) < 1e-9, "the peer's own yaw was dropped"
        assert abs(peer["vy"] - 0.3) < 1e-9


# ── The coast ───────────────────────────────────────────────────────────────
def test_the_coast_advances_the_peer_and_the_growth_is_bounded_by_the_timeout():
    """Between samples the peer is advanced on its last velocity and its radius grown for
    what an acceleration could have cost. The growth is only safe because the timeout ends
    it: the unbounded version of this formula is what let a track a person had walked out
    of reach an 11 m radius and hold the robot for a ghost 4 m off its path."""
    age = 0.4
    with tempfile.TemporaryDirectory() as tmp:
        _spool(tmp, {PEER: _pose(x=2.0, y=0.0, vx=0.2, vy=0.0, received=NOW)},
               written=NOW + age)
        peer = _source(tmp).read(NOW + age).obstacles[0]
        assert abs(peer["x"] - (2.0 + 0.2 * age)) < 1e-9, "the peer was not advanced"
        grown = peer["radius_m"] - PEER_RADIUS_M
        assert abs(grown - 0.5 * PEER_ACCEL_SIGMA_M_S2 * age * age) < 1e-9
    ceiling = 0.5 * PEER_ACCEL_SIGMA_M_S2 * PEER_TIMEOUT_S ** 2
    assert ceiling < PEER_RADIUS_M / 2.0, (
        f"the coast can grow the disc by {ceiling:.2f} m before the timeout ends it")


def test_the_coast_sigma_is_the_trackers_and_has_not_drifted_from_it():
    """``PEER_ACCEL_SIGMA_M_S2`` is a restatement of ``tracker.PROCESS_ACCEL_SIGMA``,
    because this file has to import in an environment with no robot-stack on the path. A
    restated constant that nothing pins is a constant that will diverge."""
    stack = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "robot-stack", "unitree", "go2", "visual_nav")
    sys.path.insert(0, stack)
    from tracker import PROCESS_ACCEL_SIGMA
    assert PEER_ACCEL_SIGMA_M_S2 == PROCESS_ACCEL_SIGMA


def test_the_timeout_is_the_stacks_own_blind_budget():
    """0.6 s is ``NavConfig.perception_timeout_s``, on purpose rather than by coincidence:
    a peer link that has gone quiet is the same blindness as a camera that has, and two
    staleness budgets kept in agreement by hand is one that will not be."""
    stack = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "robot-stack", "unitree", "go2", "visual_nav")
    sys.path.insert(0, stack)
    from visual_nav import NavConfig
    assert NavConfig().perception_timeout_s == PEER_TIMEOUT_S


# ── The hold, on the way to the policy ──────────────────────────────────────
def test_a_lost_link_reaches_the_policy_through_external_hold_and_not_a_new_path():
    """The peer hold uses the bridge's existing ``external_hold``, which is what the
    policy already consumes as ``RobotInput.external_hold``. A second mechanism would be a
    second thing to keep correct."""
    with tempfile.TemporaryDirectory() as tmp:
        later = NOW + PEER_TIMEOUT_S + 0.5
        _spool(tmp, {PEER: _pose(received=NOW)}, written=later)
        source = _source(tmp)
        link = source.read(later)
        tick = tick_from_state(1.0, (0.0, 0.0, 0.0), (4.0, 0.0), [],
                               peer_link={"lost": link.holds, "reason": link.reason()})
        assert external_hold(tick) is True


def test_a_tick_with_no_peer_link_block_is_unaffected():
    """Every recorded run in ``evidence/`` predates this, and ``closed_loop_sim`` has no
    peers. An absent block must mean "no peer link configured" and never "lost"."""
    tick = tick_from_state(1.0, (0.0, 0.0, 0.0), (4.0, 0.0), [])
    assert external_hold(tick) is False
    assert external_hold({"pose": {}, "obstacles": []}) is False


# ── The spool writer ────────────────────────────────────────────────────────
def test_the_spool_is_replaced_rather_than_truncated():
    """A control loop reading at 10 Hz must never see a half-written file. Checked by the
    absence of leftovers as much as by the content: a temp file left behind is a rename
    that did not happen."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "peers.json")
        write_spool(path, spool_document({PEER: _pose()}, (PEER,), domain=DOMAIN))
        write_spool(path, spool_document({}, (PEER,), domain=DOMAIN))
        assert os.listdir(tmp) == ["peers.json"], os.listdir(tmp)
        with open(path) as handle:
            assert json.load(handle)["schema"] == SCHEMA


def test_an_alignment_that_is_not_three_numbers_is_refused_at_parse_time():
    """It is the enabling flag, so a typo in it must stop the run before the legs move
    rather than silently placing the peer somewhere else."""
    for bad in ("", "1,2", "1,2,3,4", "1,2,north", "nan,0,0"):
        try:
            Alignment.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted as an alignment")
    assert Alignment.parse(" 2.0 , 1.0 , 180 ").dx == 2.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"peer_source: {len(tests)}/{len(tests)} passed")
