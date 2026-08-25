#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mesh half of peer avoidance, with a fake mesh and no robot.

The fake mesh is a list of dicts shaped like the messages
``device_connect_agent_tools`` hands back — ``{"method", "params", "_subject"}`` — which
is the same shape ``server.pump`` and ``server.camera_pump`` read. Everything downstream
of that is the real code.

The last test is the one worth reading: it takes fake mesh messages all the way through
the real spooler, the real spool file, and ``integration/peer_source``'s real reader, and
asserts a peer comes out where the tape measure says. That end-to-end is the only thing
that proves the writer and the reader agree about the format, and neither file's own
tests can establish it.

Runs without ``device-connect-agent-tools`` installed, the same way
``test_drive_bridge.py`` does — the mesh imports live inside ``main()``.
``python3 test_peer_link.py``.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile

# BOTH paths go in before ANY sibling import, in one block. `ruff --fix` sorts imports
# into contiguous blocks and will hoist one above a sys.path line sitting between them —
# see AGENTS.md, which records the two files that broke that way.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "integration"))
from peer_source import Alignment, PeerSource

from peer_link import (
    MAX_BACKDATE_S,
    PeerSpooler,
    device_from_subject,
    drain_forever,
    pose_record,
)

# Two of these deliberately trigger the module's warnings, and a suite whose "ok" lines
# are interleaved with them is harder to read than one that is not. The records still
# reach a handler a test installs itself, which is how the warning is asserted.
logging.getLogger("mappo-peer-link").addHandler(logging.NullHandler())
logging.getLogger("mappo-peer-link").propagate = False

PEER = "mappo-go2-peer"
SELF = "mappo-go2"
NOW = 98_765.0


def _message(device=PEER, x=2.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, **params):
    payload = {"pose": {"x": x, "y": y, "yaw": yaw}, "velocity": [vx, vy, 0.0],
               "radius_m": 0.40, "platform": "go2", "seq": 1, "sample_age_s": 0.01,
               "ts": "2026-08-24T12:00:00Z"}
    payload.update(params)
    return {"method": "peer_pose", "params": payload,
            "_subject": f"tenant.default.{device}.event.peer_pose"}


# ── identity ─────────────────────────────────────────────────────────────────
def test_the_device_id_comes_from_the_subject_and_not_the_payload():
    """A security property here, not only a tidiness one. The payload is whatever the
    emitting device chose to send; the subject is assigned by the transport. Reading the
    id out of the payload would let anything on a demo LAN with no PKI publish poses AS
    the peer, and the navigator would plan around a disc placed by whoever asked."""
    message = _message(device=PEER)
    message["params"]["device_id"] = "somebody-else"
    spooler = PeerSpooler([PEER], domain="test")
    spooler.accept([message], NOW)
    assert list(spooler.records) == [PEER]


def test_a_malformed_subject_yields_no_device_rather_than_a_wrong_one():
    for subject in ("", "short.subject", None, "a.b.c.d"):
        assert device_from_subject(subject) == ""


# ── the allow-list ───────────────────────────────────────────────────────────
def test_an_unlisted_device_is_ignored_and_named_in_the_log():
    """The cost of an allow-list is that an unlisted robot is invisible, which is the
    wrong direction to fail in — so the one thing it must do is say which id to add.
    Warned once per device: at 10 Hz a line per message would bury it."""
    spooler = PeerSpooler([PEER], domain="test")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    log = logging.getLogger("mappo-peer-link")
    log.addHandler(handler)
    try:
        spooler.accept([_message(device="stranger")] * 5, NOW)
    finally:
        log.removeHandler(handler)
    assert spooler.records == {}
    assert spooler.counts["ignored"] == 5
    assert len(records) == 1, "one warning per device, not per message"
    assert "stranger" in records[0].getMessage()


def test_the_navigators_own_pose_is_not_spooled_as_a_peer():
    """The navigator very likely publishes its own pose so that peers can avoid IT. Fed
    back in, that is an obstacle on top of the robot, and — because a peer inside the hard
    gap can never be cleared — a run that holds forever for a diagnosis nobody would
    guess. The allow-list makes self-exclusion the default rather than a step to
    remember."""
    spooler = PeerSpooler([PEER], domain="test")
    spooler.accept([_message(device=SELF), _message(device=PEER)], NOW)
    assert list(spooler.records) == [PEER]


def test_a_spooler_with_nothing_to_watch_is_refused_at_construction():
    """An empty roster makes every staleness check downstream vacuous — the reader has
    nothing to report missing. Peer avoidance is turned off by not passing the flag, not
    by watching an empty list."""
    try:
        PeerSpooler([], domain="test")
    except ValueError:
        return
    raise AssertionError("a spooler with no peers was accepted")


# ── the payload ──────────────────────────────────────────────────────────────
def test_the_payload_is_flattened_and_only_the_planar_velocity_is_kept():
    """``velocity`` is (vx, vy, wz) BODY frame. The consumer rotates a translation; yaw
    rate is not one and has no meaning as an obstacle velocity."""
    record = pose_record(_message(vx=0.3, vy=-0.1)["params"], NOW)
    assert record["x"] == 2.0 and record["yaw"] == 0.0
    assert record["vx"] == 0.3 and record["vy"] == -0.1
    assert "wz" not in record
    assert record["received_monotonic_s"] == NOW


def test_the_peers_sampling_lag_is_carried_as_a_duration():
    """It arrives as ``sample_age_s`` and is spooled as ``sample_age_at_emit_s``, and it
    is a DURATION rather than a timestamp for the only reason that matters: two robots on
    a demo LAN with no NTP share no clock, so an interval crosses the mesh and an instant
    does not."""
    record = pose_record(_message(sample_age_s=0.13)["params"], NOW)
    assert record["sample_age_at_emit_s"] == 0.13


def test_an_empty_pose_survives_the_trip_rather_than_being_dropped_here():
    """A driver whose estimator is throwing emits an empty pose ON PURPOSE. Dropping it
    here would make the reader say "has never reported a pose" — a true sentence pointing
    at the wrong machine, when the specific one is available for free."""
    record = pose_record({"pose": {}, "velocity": []}, NOW)
    assert record is not None
    assert "x" not in record and "vx" not in record


# ── the timestamps ───────────────────────────────────────────────────────────
class _Subscription:
    """A fake mesh subscription that hands over one batch per read."""

    def __init__(self, batches):
        self.batches = list(batches)

    def read(self):
        return self.batches.pop(0) if self.batches else []


class _Counter:
    def __init__(self, budget):
        self.left = budget + 1

    def __call__(self):
        self.left -= 1
        return self.left <= 0


def _drain(batches, times, spooler):
    """Run the real drain loop over fixed batches on a fixed clock."""
    clock = iter(times)
    drain_forever(_Subscription(batches), spooler, poll_s=0.0, heartbeat_s=0.0,
                  sleep=lambda _s: None, clock=lambda: next(clock),
                  stop=_Counter(len(batches)))


def test_a_batch_is_stamped_when_the_window_opened_not_when_it_closed():
    """A ``Subscription`` buffers between reads, so a message handed over now may have
    arrived at any point since the last one. Stamping it "now" credits it with up to a
    full poll interval of freshness it does not have — in the direction that makes a stale
    pose look fresh, which is the one direction this whole feature must not fail in.

    The window here opens at 100.0 and closes at 100.04; the stamp must be 100.0."""
    spooler = PeerSpooler([PEER], domain="test")
    _drain([[_message()]], [100.0, 100.04, 100.08], spooler)
    assert spooler.records[PEER]["received_monotonic_s"] == 100.0


def test_the_back_date_is_capped_so_a_blocking_read_cannot_invent_staleness():
    """Back-dating is only right if ``read()`` drains. If it ever blocks for seconds
    waiting for traffic, an uncapped back-date would date a sample that arrived
    milliseconds ago to seconds old and stop the robot for a link that is working."""
    spooler = PeerSpooler([PEER], domain="test")
    _drain([[_message()]], [100.0, 130.0, 130.1], spooler)
    stamped = spooler.records[PEER]["received_monotonic_s"]
    assert stamped == 130.0 - MAX_BACKDATE_S, stamped


def test_the_spool_is_written_even_when_no_pose_arrived():
    """The heartbeat is what lets the reader tell "no peer has moved" from "this process
    is dead" — the same file contents, opposite situations and opposite repairs."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "peers.json")
        spooler = PeerSpooler([PEER], spool_path=path, domain="test")
        _drain([[], [], []], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], spooler)
        assert spooler.counts["writes"] == 3
        assert os.path.exists(path)


def test_a_read_that_raises_is_survived_rather_than_ending_the_link():
    """The consequence of this loop stopping is that every robot avoiding this one holds.
    That should happen because the link is broken, not because one read hit a hiccup."""
    class _Angry(_Subscription):
        def read(self):
            if self.batches:
                self.batches.pop(0)
                raise RuntimeError("transport hiccup")
            return []

    spooler = PeerSpooler([PEER], domain="test")
    clock = iter([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    drain_forever(_Angry([[_message()]]), spooler, poll_s=0.0, heartbeat_s=0.0,
                  sleep=lambda _s: None, clock=lambda: next(clock), stop=_Counter(2))
    assert spooler.counts["writes"] >= 1, "the heartbeat stopped for one bad read"


# ── the whole way through ────────────────────────────────────────────────────
def test_a_mesh_message_becomes_an_obstacle_where_the_tape_measure_says():
    """END TO END, and the only test that proves the writer and the reader agree.

    Fake mesh messages -> the real spooler -> the real spool file -> the real reader.
    The peer reports being on its own origin; it was switched on 2 m ahead and 1 m to the
    left of this robot, so it belongs at (2, 1) here. A format change on either side that
    the other did not follow shows up as a peer at the wrong place or as a hold, and both
    of those are visible here and in neither file's own tests.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "peers.json")
        spooler = PeerSpooler([PEER], spool_path=path, domain="test")
        _drain([[_message(x=0.0, y=0.0, yaw=0.0, vx=0.25)]],
               [NOW, NOW + 0.01, NOW + 0.02], spooler)

        source = PeerSource(path, Alignment.parse("2.0,1.0,0"), domain="test")
        link = source.read(NOW + 0.02)
        assert not link.holds, link.reason()
        peer = link.obstacles[0]
        assert peer["id"] == f"peer-{PEER}"
        assert abs(peer["x"] - 2.0) < 0.02 and abs(peer["y"] - 1.0) < 1e-9
        assert abs(peer["vx"] - 0.25) < 1e-9
        assert peer["radius_m"] >= 0.40


def test_the_same_message_stops_the_robot_once_it_stops_arriving():
    """The other half of the round trip, and the property the design turns on: the file is
    not rewritten with a new pose, only its heartbeat advances, and the peer goes from a
    disc the policy can path around to a robot that must stop.

    Fails if the two sides disagree about which field carries the arrival time — the
    reader would then age from something that never moves, and this would stay green
    forever at "fresh"."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "peers.json")
        spooler = PeerSpooler([PEER], spool_path=path, domain="test")
        _drain([[_message()]], [NOW, NOW + 0.01, NOW + 0.02], spooler)
        source = PeerSource(path, Alignment(), domain="test")
        assert not source.read(NOW + 0.02).holds

        # peer_link is still alive and still writing; the PEER went quiet.
        spooler.write(NOW + 5.0)
        link = source.read(NOW + 5.0)
        assert link.holds is True
        assert link.obstacles == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"peer_link: {len(tests)}/{len(tests)} passed")
