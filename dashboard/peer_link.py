#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Listen for peer poses on the mesh and spool them where the control loop can read them.

Runs ON the navigator robot, in the same Python >= 3.11 environment as ``robot_driver.py``,
and writes one small JSON file that ``integration/peer_source.py`` reads at 10 Hz from the
SDK environment's Python 3.8. It is the mirror image of ``drive_bridge.py``: that one
exists because the Device Connect side cannot import the robot's SDK, this one because the
robot's SDK side cannot import Device Connect. Same wall, opposite direction.

## Why a file and not a socket

The control loop is a 10 Hz Python 3.8 process with no Device Connect and no asyncio, and
whatever it reads it must read without blocking. A file replaced with ``os.replace`` is
the smallest thing that satisfies that: the reader never blocks, never half-reads, gets
the newest state rather than a queue of old ones, and works when it starts after this
process or restarts under it. It is also the only option an operator can debug with
``cat`` while a robot is walking, which on a demo day is worth more than it sounds.

## The allow-list, and the one thing it costs

``--peer`` is required and repeatable, and only the device ids named there are spooled.
An allow-list rather than "everything on the mesh that publishes a pose", for two
reasons. The navigator very likely runs its OWN ``robot_driver.py`` with pose publishing
on — so that peers can avoid IT — and a denylist that forgot to exclude self would put an
obstacle on top of the robot and hold the run forever, with a diagnosis nobody would
guess. And this is a safety input: what a demo LAN with no PKI lets onto the obstacle list
should be a deliberate act, the same argument ``server.ALLOWED_FUNCTIONS`` makes for
invocations.

What it costs is that an unlisted robot is INVISIBLE, which is the wrong direction to fail
in. So every unlisted device seen publishing a pose is logged, once per device, at
warning level, naming the id to add.

## The device id comes from the subject, not the payload

Exactly as ``server._shape_event`` does it, and the reason is the same and it is a
security property here rather than only a tidiness one: the payload is whatever the
emitting device chose to send, while the subject is assigned by the transport. Reading the
id out of the payload would let any device on the mesh publish poses AS the peer, and the
navigator would drive around a disc placed by whoever asked.

## Timestamps: the batch is back-dated on purpose

Every spooled record carries ``received_monotonic_s``, read from THIS host's clock, which
is what makes an age computable at the other end at all — see ``peer_source``'s module
docstring for why no timestamp taken on the peer can be used.

It is stamped with the moment the PREVIOUS drain finished, not with the moment this one
returned. A ``Subscription`` buffers between reads, so a message handed over now may have
arrived at any point since the last read; stamping it "now" would credit it with up to a
full poll interval of freshness it does not have, and the whole file exists to stop a
pose being treated as newer than it is. Back-dating to the start of the window is the only
choice that cannot under-state an age.

The back-date is capped at :data:`MAX_BACKDATE_S`, because it is only correct if ``read()``
drains rather than blocks. If it ever blocks for seconds waiting for traffic, an unbounded
back-date would declare a sample that arrived milliseconds ago to be seconds old and stop
the robot for a link that is working.

## Run it

On the navigator, beside the driver:

    python3 peer_link.py --peer mappo-go2-peer

Then give the control loop the transform between the two robots' odom frames, which is
the one thing no amount of meshing can supply:

    python3 mappo_drive.py --live --peer-odom-align 2.0,1.0,180 ...

``python3 test_peer_link.py`` — runs without the Device Connect packages, which are
imported inside the functions that need them.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# One block, before the sibling import, and `ruff --fix` must not hoist that import above
# it — see AGENTS.md, which records the two files this has already broken.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integration"))

from peer_source import (
    DEFAULT_SPOOL,
    clock_domain,
    spool_document,
    write_spool,
)

log = logging.getLogger("mappo-peer-link")

#: How often the subscription is drained. The same 0.05 s the dashboard uses for camera
#: frames, and for the same reason: this is the high-rate lane, and the poll interval is
#: added to every pose's age (see the module docstring), so it is latency spent out of
#: ``peer_source.PEER_TIMEOUT_S``.
PEER_POLL_S = 0.05

#: How often the spool is rewritten even when nothing arrived.
#:
#: The heartbeat is what lets the reader tell "no peer has moved" from "this process is
#: dead", which are the same file contents and opposite situations. Without it a
#: peer_link killed mid-run leaves a spool that ages out one peer at a time and reports
#: three separate stale peers instead of one dead link.
HEARTBEAT_S = 0.1

#: Ceiling on how far a drained batch may be back-dated. Two poll intervals: enough that
#: an ordinary drain is stamped exactly, small enough that a blocking ``read()`` cannot
#: turn a fresh pose into a stale one.
MAX_BACKDATE_S = 2 * PEER_POLL_S

#: The event this listens for. Its own subject, so draining it cannot be delayed behind a
#: burst of camera frames — the same lane argument the dashboard makes for its own
#: camera subscription, applied to the one input that stops the legs.
PEER_EVENT = "peer_pose"


def device_from_subject(subject: str) -> str:
    """The emitting device's id, out of the transport's subject.

    Mirrors ``server._shape_event``. Kept as a named function rather than inlined so the
    one place a peer's identity is decided is testable without a mesh.
    """
    parts = str(subject or "").replace("/", ".").split(".")
    return parts[2] if len(parts) > 4 else ""


def pose_record(params: dict, received_monotonic_s: float) -> dict | None:
    """One spool record from one ``peer_pose`` payload, or ``None`` if it is unusable.

    The payload carries ``pose`` and ``velocity`` in the shape ``robot_state`` already
    uses — a dict and a three-element list — and the spool is flat. Flattening here means
    the mesh payload stays consistent with the driver's other event and the control loop
    reads one shape.

    Deliberately does NOT validate the numbers. ``peer_source._obstacle`` rejects a
    non-finite or missing pose and holds the robot for it, and that check has to live
    there anyway — it is what protects the loop from a spool written by an older version
    of this file. Duplicating it would mean a malformed pose is dropped silently at this
    end while the reader reports "has never reported a pose": a true sentence pointing at
    the wrong machine. A driver whose estimator is throwing emits an empty pose ON PURPOSE
    for exactly that reason, and it must survive the trip to be diagnosed at the far end.
    """
    if not isinstance(params, dict):
        return None
    record = {"received_monotonic_s": received_monotonic_s}
    pose = params.get("pose")
    if isinstance(pose, dict):
        for key in ("x", "y", "yaw"):
            if key in pose:
                record[key] = pose[key]
    velocity = params.get("velocity")
    # (vx, vy, wz) BODY frame, as `Go2Locomotion.velocity()` returns it. Only the two
    # planar components are spooled; the consumer needs a translation to rotate, and yaw
    # rate is not one. Indexed rather than unpacked so a three-element list and a longer
    # one both work, and a shorter one simply leaves the keys absent to be rejected.
    if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
        record["vx"], record["vy"] = velocity[0], velocity[1]
    for key in ("radius_m", "seq", "platform"):
        if key in params:
            record[key] = params[key]
    # The peer measures this between reading its estimator and emitting. A duration, not
    # a timestamp: it is two readings of the peer's own clock subtracted at the source, so
    # unlike a timestamp it means the same thing on this side of the mesh.
    if "sample_age_s" in params:
        record["sample_age_at_emit_s"] = params["sample_age_s"]
    return record


class PeerSpooler:
    """Accumulates the newest pose per allowed peer and writes the spool.

    Separate from the mesh loop so the whole of the interesting behaviour — the
    allow-list, the back-dating, the heartbeat — is testable with a list of dicts and no
    Device Connect installed.
    """

    def __init__(self, peers, spool_path=DEFAULT_SPOOL, domain: str | None = None) -> None:
        self.expect = sorted({str(p) for p in peers})
        if not self.expect:
            raise ValueError("peer_link needs at least one --peer device id to watch")
        self.spool_path = Path(spool_path)
        self.domain = clock_domain() if domain is None else domain
        self.records: dict = {}
        #: Device ids seen publishing a pose that are not on the allow-list. Warned about
        #: once each: an unlisted peer is one this robot cannot see, and a line per
        #: message at 10 Hz would bury the line that says so.
        self.unlisted: set = set()
        self.counts = {"messages": 0, "accepted": 0, "ignored": 0, "writes": 0}

    def accept(self, messages, received_monotonic_s: float) -> int:
        """Fold one drained batch in. Returns how many records were updated."""
        updated = 0
        for message in messages or ():
            self.counts["messages"] += 1
            device = device_from_subject(message.get("_subject", ""))
            if device not in self.expect:
                self.counts["ignored"] += 1
                if device and device not in self.unlisted:
                    self.unlisted.add(device)
                    log.warning("%s is publishing a pose and is NOT being watched — this "
                                "robot cannot see it. Add --peer %s to change that.",
                                device, device)
                continue
            record = pose_record(message.get("params") or {}, received_monotonic_s)
            if record is None:
                continue
            # Newest wins within a batch. A Subscription drains in arrival order per
            # subject, so the last message from a device is its newest pose, and keeping
            # an older one would spool a position the peer has already left.
            self.records[device] = record
            self.counts["accepted"] += 1
            updated += 1
        return updated

    def write(self, now_s: float | None = None) -> None:
        """Replace the spool, heartbeat included.

        ``now_s`` is passed in rather than read here so the heartbeat and every record's
        ``received_monotonic_s`` are stamped from ONE clock. They are compared against
        each other at the far end — the reader asks "is the link alive" and "is this peer
        fresh" out of the same file — and two clocks that agree only because both happen
        to call ``time.monotonic()`` is an agreement nothing enforces.
        """
        write_spool(self.spool_path,
                    spool_document(self.records, self.expect, domain=self.domain,
                                   written_monotonic_s=now_s))
        self.counts["writes"] += 1

    def report(self) -> str:
        counts = self.counts
        return (f"[peer_link] {counts['accepted']}/{counts['messages']} poses accepted, "
                f"{counts['ignored']} from unlisted devices, {counts['writes']} writes")


def drain_forever(subscription, spooler: PeerSpooler, *, poll_s: float = PEER_POLL_S,
                  heartbeat_s: float = HEARTBEAT_S, sleep=time.sleep,
                  clock=time.monotonic, stop=None) -> None:
    """Drain, spool, repeat. The clock and the sleep are injectable so a test can run it.

    A read that raises is logged and retried rather than fatal. The consequence of this
    loop stopping is that the spool's heartbeat ages out and the robot holds, which is the
    right outcome for a broken link — but it should happen because the link is broken, not
    because one read hit a transport hiccup.
    """
    window_started = clock()
    last_write = 0.0
    while stop is None or not stop():
        try:
            messages = subscription.read()
        except Exception as exc:      # broad: any transport hiccup, see the docstring
            log.warning("peer pose read failed: %r", exc)
            messages = ()
        drained_at = clock()
        # Anything in this batch arrived at or after the previous drain finished, so that
        # instant is the newest stamp that cannot over-state freshness. Capped, because
        # back-dating is only right if read() drains rather than blocks.
        received = max(window_started, drained_at - MAX_BACKDATE_S)
        window_started = drained_at
        spooler.accept(messages, received)
        if drained_at - last_write >= heartbeat_s:
            spooler.write(drained_at)
            last_write = drained_at
        sleep(poll_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spool peer robot poses from the Device Connect mesh for the control "
                    "loop to read.")
    parser.add_argument("--peer", action="append", default=[], metavar="DEVICE_ID",
                        help="a peer to watch. Repeatable, and REQUIRED: this is an "
                             "allow-list, so an unlisted robot is one the navigator "
                             "cannot see. Do not list this robot's own device id.")
    parser.add_argument("--spool", default=str(DEFAULT_SPOOL), metavar="PATH.json",
                        help="where the control loop will read peer poses from. Must "
                             "match mappo_drive.py's --peer-spool.")
    # No --messaging-url or --tenant. `server.py` is the other consumer-side process in
    # this repository and it calls `connect()` with no arguments at all, so those are
    # whatever the agent-tools package resolves them to. Adding flags here that the
    # consumer API has not been shown to accept would be inventing a knob.
    parser.add_argument("--require-tls", action="store_true",
                        help="Require TLS on the mesh. The default is insecure, which is "
                             "what a demo LAN with no PKI needs.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(levelname)-7s  %(message)s")
    if not args.peer:
        build_parser().error("--peer is required at least once; see --help for why it is "
                             "an allow-list rather than a default of everything")

    spooler = PeerSpooler(args.peer, spool_path=args.spool)
    # Written before the first message so the control loop can tell "the link is up and
    # has heard nothing" from "the link was never started" — those are the same empty
    # peer list and completely different repairs.
    spooler.write()
    log.info("watching %s; spooling to %s", ", ".join(spooler.expect), spooler.spool_path)

    if not args.require_tls:
        # Set before the first connection so the SDK sees it, exactly as server.Mesh does.
        os.environ.setdefault("DEVICE_CONNECT_ALLOW_INSECURE", "true")

    from device_connect_agent_tools import connect, disconnect
    from device_connect_agent_tools.tools import subscribe

    connect()
    # Its own subject, not `event(*)`: a subscription keeps one inbox per subject, so
    # subscribing to everything would put peer poses in the same drain as camera frames
    # and motion events. That is the lane argument `server.Mesh` makes for its camera
    # subscription, and it matters more here, because what is being delayed is the input
    # that decides whether the legs stop.
    subscription = subscribe(f"event({PEER_EVENT})")
    try:
        drain_forever(subscription, spooler)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        print(spooler.report())
        # The spool is deliberately LEFT BEHIND rather than deleted. Its heartbeat stops,
        # which is what tells the control loop the link is gone; removing the file reports
        # the same thing through a different message, and leaving it lets an operator read
        # the last known peer positions after the fact.
        try:
            disconnect()
        except Exception as exc:
            log.warning("disconnect failed: %r", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
