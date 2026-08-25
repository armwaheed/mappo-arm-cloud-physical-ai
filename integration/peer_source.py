# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Peer robots as obstacles, from the Device Connect mesh instead of from the camera.

⛔ **STALENESS IS THE SAFETY PROPERTY IN THIS FILE.** A peer pose that stops arriving is
not a peer standing still — it is a peer whose position is no longer known. Everything
below is arranged around that one sentence, and :func:`PeerSource.read` is written so
that every way of failing lands on ``holds`` rather than on an empty obstacle list.

## Why the mesh rather than the camera

Three routes to "where is the other robot" were tried. A detector fine-tuned on 1,343
real in-domain frames of the peer reaches **53% recall at 38% false positives** on
held-out footage and has no usable operating point at all — the ceiling is the frozen
backbone, not the data (``detector/FROZEN-FEATURE-CEILING.md``, on ``feat/train-the-new-class``).
A marker on the peer and a colour panel on the peer were both vetoed.

The route left is the one the trained policy actually describes. The VMAS agents this
checkpoint was trained against observed each other's **true positions**; they did not run
detectors on each other. Both robots are already on a Device Connect mesh, and a robot
knows its own pose to the accuracy of its own estimator. Publishing it is a strictly
better measurement than inferring it from 1920x1080 pixels — and it carries a velocity,
which no single-frame detector can.

⚠️ **The velocity does not reach the network.** The policy's 18-value observation has no
obstacle-velocity channel; a disc enters it as an instantaneous position. What the peer's
velocity DOES reach is the two places that matter almost as much: the planner's rollout
(``DynamicWindowPlanner._gaps`` advances every obstacle on its own ``vx, vy`` at each
step, so the veto scores the policy's command against where the peer WILL be), and
:func:`mappo_bridge.holds_the_robot`, which is the gate that decides whether a peer is
handed to the policy at all. Saying "the policy now sees velocity" would be wrong.

That gate — :data:`mappo_bridge.POLICY_MAX_MOVER_SPEED_MPS`, 0.25 m/s — is below this
robot's own 0.35 m/s gait floor, so a peer that is actually *walking* still stops the
navigator and only a parked or shuffling one reaches the policy. That is unchanged here
and it is the conservative behaviour. What the mesh changes is that the number is now
worth arguing about: the gate was set against a velocity estimated by a filter fed by a
detector, and it is now the peer's own estimator. A shadow run with a peer driven at
0.2/0.4/0.6 m/s is the measurement that would move it.

## The shape, and why nothing else changes

This produces obstacle records in exactly the shape ``visual_nav._obstacles`` produces:
odom-frame ``x, y, vx, vy``, a ``radius_m`` already inflated for uncertainty, ``kind``
and ``id``. Everything downstream — ``mappo_bridge.policy_objects``, the planner's veto,
the telemetry writer, the overlay, ``replay_mappo`` — then works unchanged, because none
of them ask where an obstacle came from. There is deliberately no second path into the
policy for peers; a peer is one more disc.

``kind`` is ``"tracked"``, not ``"static"``. That is load-bearing rather than cosmetic:
``mappo_bridge.is_stationary`` reads ``kind`` first, and a peer marked ``static`` would
be a landmark that never holds the robot **however fast it was moving**, which deletes
the speed gate that is the safety half of the two-tier split.

``label`` is the constant :data:`PEER_LABEL` and NOT the peer's platform name, because
:data:`mappo_bridge.HOLD_LABELS` is matched on ``label`` — a platform that was ever
called ``person`` would silently move every peer to the stop tier, and a platform name
arriving over a network is not something this file controls.

## The two clocks, and the one that is actually comparable

``mappo_bridge.robot_input`` already carries the hard-won version of this warning: the
policy's own staleness gate compares against ``time.monotonic()``, and handing it a wall
clock computes an age of about -1.8e9 s, which is under every threshold, so the gate
fails OPEN. The same trap is worse here, because there are now **two machines**.

A peer's ``time.time()`` and this robot's ``time.time()`` are minutes apart on a demo LAN
with no NTP, and their ``time.monotonic()`` clocks count from unrelated boots. **No
timestamp taken on the peer is comparable with any timestamp taken here.** So none is
used. Ages are computed entirely from clocks read on THIS host:

* ``received_monotonic_s`` — stamped by ``dashboard/peer_link.py``, which runs on this
  robot, at the moment the event arrived. Same machine, same ``CLOCK_MONOTONIC``, so the
  subtraction is meaningful across the two processes and survives either of them
  restarting.
* ``sample_age_at_emit_s`` — a **duration**, measured on the peer between reading its
  estimator and emitting. A duration is two readings of one clock subtracted at the
  source, so it carries across the mesh where a timestamp cannot. It is added to the age.

What remains unmeasured is the mesh transit itself, which is exactly the term that needs
two clocks. The age computed here is therefore an UNDER-estimate by that transit, and
:data:`PEER_TIMEOUT_S` is spent against a budget that does not include it. Measuring it
needs a round trip and hardware; until then it is a known bias in the unsafe direction
and is written down rather than assumed small.

:data:`CLOCK_DOMAIN_PATH` guards the assumption underneath all of that. A monotonic
reading only means anything against the boot it was taken in, so the writer records the
boot id and this refuses a spool written under a different one — otherwise a spool file
surviving a reboot yields a large NEGATIVE age, and a negative age passes every "is it
older than" test ever written. Negative ages are rejected outright as well, because the
boot id is best-effort and absent on any host without ``/proc``.

## What happens when it goes stale, and why it is a hold

Three options were on the table and two of them are wrong in the same direction.

**Expire the obstacle.** The robot then plans through the space the peer was last seen
in. That is the fail-open answer: the one moment the link matters most is the moment the
peer is close enough to interfere with the radio, or busy enough to drop samples.

**Freeze the disc where it was last seen.** Wrong twice over. If the peer moved away the
navigator is blocked forever by a ghost — this repository has already shipped that bug
once in the tracker, where a person who walked out of shot reached an 11 m radius and the
robot held for a ghost 4 m off its path. And if the peer moved TOWARDS the navigator, the
ghost is in the wrong place and the real peer is unguarded.

**Hold the robot.** This is what the file does, and it is the precedent already set one
directory over: ``mappo_bridge.external_hold`` treats stale perception as an external
hold, because "the robot is blind, and a policy acting on a frozen world model is worse
than a policy that does not act". A peer link that has stopped updating is the same
blindness about a different sensor.

So a lost peer contributes **no obstacle and a hold**, and those two are one decision:
dropping the disc is only safe *because* the legs stop. Anyone tempted later to keep the
obstacle-dropping and drop the hold has inverted the whole file.

A reachability escape hatch was considered and rejected with arithmetic: hold only if the
peer's reachable set could still contain the robot, ``r = last_radius + v_max * age``. A
Go2's published top speed is 3.7 m/s, so at the 0.6 s timeout the reachable set is 2.2 m
wider than the last sighting, and the demo corridor is a few metres long. The hatch would
never fire in the arena it was written for, which makes it complexity with no behaviour.

## Position is extrapolated; the extrapolation is bounded BY the timeout

Between samples the peer's last pose is advanced on its last velocity and its radius
grown by ``0.5 * sigma * age^2`` with ``sigma`` the tracker's process-acceleration sigma.
That is the same formula ``visual_nav._obstacles`` uses, for the same reason.

It is **not** the double-count that formula's docstring warns about. There, a Kalman
filter had already advanced every track to ``_tracker_time``, so extrapolating again from
``last_seen`` integrated the coast twice. Here there is no filter and nothing has advanced
anything: the record is a raw sample and the full age is the right interval. The two look
identical and mean opposite things, so this paragraph exists to stop the wrong one being
copied in.

The growth is bounded because :data:`PEER_TIMEOUT_S` bounds the age — 0.18 m of inflation
at the 0.6 s limit, against the peer's own 0.40 m. Unbounded inflation is what produced
the 11 m ghost; a growth term is only safe next to a timeout that ends it.

Python 3.8, pure stdlib: this runs on the Jetson beside the control loop, in the SDK
environment, which has neither Device Connect nor numpy. The mesh half lives in
``dashboard/peer_link.py`` and reaches this file only through a file on disk.

``python3 test_peer_source.py``.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Bump on any rename or removal. The writer stamps it and this refuses what it does not
#: know, rather than reading a field that has moved and getting a plausible number.
SCHEMA = "mappo.peer_link/1"

#: Where ``dashboard/peer_link.py`` writes and this reads. Beside the refusal log, and
#: for the same reason: a fixed path an operator can `cat` while a run is happening.
DEFAULT_SPOOL = Path.home() / ".mappo-peers.json"

#: How old a peer pose, or the link itself, may be before the robot is held.
#:
#: This is ``NavConfig.perception_timeout_s`` — 0.6 s — and it is the same number on
#: purpose rather than by coincidence. That is already this stack's answer to "the newest
#: thing I know about the world is too old to act on", it is already what stops the robot
#: when the camera falls behind, and a peer link is the same question about a different
#: sensor. Inventing a second staleness budget would mean two numbers that have to be
#: kept in agreement by hand.
#:
#: At the publisher's 10 Hz that is six consecutive samples lost before the robot stops,
#: which is a real hiccup rather than a jittery one. ⚠️ It is spent against an age that
#: does NOT include mesh transit (see the module docstring), so the true budget is
#: shorter by however long that is, and nobody has measured it.
PEER_TIMEOUT_S = 0.6

#: Process-acceleration sigma for the coast inflation, m/s^2.
#:
#: The tracker's ``PROCESS_ACCEL_SIGMA``, restated rather than imported because this file
#: has to load in an environment with no ``robot-stack`` on the path. ``test_peer_source``
#: pins the two together, so the copy cannot drift silently.
PEER_ACCEL_SIGMA_M_S2 = 1.0

#: The disc a peer quadruped is modelled as, metres.
#:
#: **This is the half-DIAGONAL, and the distinction is the whole point of the constant.**
#: A Go2 is 0.70 x 0.31 m, so the half-length is 0.35 and the half-diagonal is
#: ``hypot(0.35, 0.155) = 0.383``. ``avoidance.NavConfig.robot_radius_m`` already made
#: this call for the navigator's own body and its comment says so: "Go2 is ~0.70 x 0.31 m;
#: half-diagonal, rounded up" — 0.40. A peer gets the same treatment.
#:
#: The live runs plan the navigator's OWN disc at 0.25 instead, and that is not a
#: precedent for a peer: 0.25 is justified by the half-width plus margin, on the measured
#: grounds that "the planner never rotates, so its footprint stayed square". Nothing here
#: controls a peer's heading, so a peer is broadside whenever it feels like it and the
#: long axis is the one that has to fit.
#:
#: For scale: the colour-prop path defaults to 0.15 m, which is a bin. Using it for a
#: quadruped under-models it by more than half.
PEER_RADIUS_M = 0.40

#: What a peer obstacle is labelled. A constant, never the platform name — see the module
#: docstring. ``test_peer_source`` asserts it is not in ``mappo_bridge.HOLD_LABELS``.
PEER_LABEL = "peer"

#: Best-effort identity of the boot this host is running. A ``time.monotonic()`` reading
#: is only meaningful within one boot, so writer and reader compare theirs and a spool
#: from a different boot is discarded rather than believed. Absent outside Linux, where
#: the negative-age rejection is the only guard left.
CLOCK_DOMAIN_PATH = "/proc/sys/kernel/random/boot_id"


def clock_domain() -> str:
    """This host's boot id, or ``""`` where there is not one to read."""
    try:
        with open(CLOCK_DOMAIN_PATH) as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _finite(value) -> float | None:
    """A usable number, or ``None``.

    Same contract as ``mappo_bridge._finite`` and for the same reason, with one addition
    that matters more here: this input crossed a network. A peer that emitted a NaN pose,
    or a JSON writer that wrote ``null`` for a non-finite float, must produce a rejected
    record and therefore a hold — not a silent NaN that propagates into the planner's
    rollout, where every comparison against it is False and every obstacle looks clear.
    """
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value) if math.isfinite(value) else None


@dataclass(frozen=True)
class Alignment:
    """The rigid transform from the PEER's odom frame into THIS robot's odom frame.

    ⚠️ **There is no shared frame between two quadrupeds and this is the only thing that
    invents one.** Each robot's odometry starts at its own power-on pose, so a peer pose
    of ``(0, 0)`` means "where the peer was switched on" and says nothing about where that
    is relative to here. Consuming a peer pose without this transform puts the obstacle
    somewhere plausible and wrong, and plausible-and-wrong is the failure this repository
    keeps finding: it passes every bench test, because at the bench both robots start on
    the same spot facing the same way and the transform is the identity.

    So it is required rather than defaulted. ``0,0,0`` is a legitimate value and an
    operator is welcome to pass it — the point is that it has to be typed, because typing
    it is the claim that both robots started from the same origin.

    **What to measure.** ``dx, dy`` are where the peer was standing when it powered on,
    expressed in the navigator's own start frame: ``+x`` is straight ahead of the
    navigator's start heading, ``+y`` is to its left. ``dyaw_deg`` is the peer's start
    heading minus the navigator's, CCW positive. A peer parked 2 m ahead, 1 m left, facing
    back at the navigator is ``2.0,1.0,180``. A tape measure and a floor mark are the
    instrument; both are already needed to stage the goal.

    **What it cannot fix.** Both odometries drift, and this stack has measured what that
    costs: a run whose robot was being dragged by its tether finished with four landmarks
    for one bin, up to 5.6 m apart. Alignment measured at ``t=0`` decays for the whole
    run at the sum of two robots' drift rates, and nothing here observes that. It is one
    more reason :data:`PEER_RADIUS_M` is not shaved to the half-width.
    """

    dx: float = 0.0
    dy: float = 0.0
    dyaw_rad: float = 0.0

    @classmethod
    def parse(cls, text: str) -> Alignment:
        """``"DX,DY,DYAW_DEG"`` -> an :class:`Alignment`. Degrees, because the operator
        is reading a protractor or counting quarter-turns on a floor mark, and a run
        refused for a typo'd radian is a run lost to unit conversion."""
        parts = [p.strip() for p in str(text).split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"peer odom alignment must be DX,DY,DYAW_DEG (three values); got {text!r}")
        try:
            dx, dy, dyaw_deg = (float(p) for p in parts)
        except ValueError as exc:
            raise ValueError(
                f"peer odom alignment must be three numbers; got {text!r}") from exc
        if not all(math.isfinite(v) for v in (dx, dy, dyaw_deg)):
            raise ValueError(f"peer odom alignment must be finite; got {text!r}")
        return cls(dx=dx, dy=dy, dyaw_rad=math.radians(dyaw_deg))

    def point(self, x: float, y: float) -> tuple:
        """A position in the peer's odom frame, in this robot's odom frame."""
        cos, sin = math.cos(self.dyaw_rad), math.sin(self.dyaw_rad)
        return (self.dx + x * cos - y * sin, self.dy + x * sin + y * cos)

    def direction(self, x: float, y: float) -> tuple:
        """A vector in the peer's odom frame, in this robot's odom frame.

        Rotation only. Separate from :meth:`point` rather than a flag, because applying
        the translation to a velocity is a mistake that reads as a plausible number: a
        peer standing still 2 m away would report 2 m/s of drift and be routed to the
        stop tier by the speed gate, on every tick, for the whole run.
        """
        cos, sin = math.cos(self.dyaw_rad), math.sin(self.dyaw_rad)
        return (x * cos - y * sin, x * sin + y * cos)


@dataclass(frozen=True)
class PeerLink:
    """One reading of the spool: what is known about every expected peer, and how old.

    ``holds`` is the field the control loop acts on and the rest is for the record. A
    reading with any lost peer holds; so does a reading whose link is dead, unusable or
    absent. There is no state in which this reports "nothing to worry about" because it
    could not tell.
    """

    #: Obstacle records, telemetry-shaped, in this robot's odom frame.
    obstacles: list = field(default_factory=list)
    #: Device ids that were expected and are not currently usable, with why.
    lost: dict = field(default_factory=dict)
    #: Age in seconds of each peer that IS usable, for the log and the report.
    ages_s: dict = field(default_factory=dict)
    #: Why the whole link is unusable, or ``""`` when it is fine. Distinct from ``lost``:
    #: this one means nothing is known about ANY peer, which is a different repair job.
    link_error: str = ""
    #: Age of the spool file's own heartbeat, or ``None`` when it could not be computed.
    link_age_s: float | None = None

    @property
    def holds(self) -> bool:
        """Whether the peer link requires the robot to stop this tick."""
        return bool(self.link_error) or bool(self.lost)

    def reason(self) -> str:
        """One line saying what is wrong, for the operator and the telemetry."""
        if self.link_error:
            return f"peer link: {self.link_error}"
        if self.lost:
            return "peer link: " + "; ".join(
                f"{device} {why}" for device, why in sorted(self.lost.items()))
        return ""


class PeerSource:
    """Reads the spool ``dashboard/peer_link.py`` writes and produces obstacles.

    Constructed once per run and read once per control tick. Cheap enough for 10 Hz: the
    spool is a few hundred bytes and, on the robot, in the page cache.

    The expected roster comes from the SPOOL, not from a second command-line list. That
    is deliberate — ``peer_link.py`` is already told which device ids to accept, and a
    consumer with its own list is a second place to forget a robot. It does mean an
    absent or unreadable spool cannot name what is missing, which is why that case is a
    :attr:`PeerLink.link_error` and holds unconditionally.
    """

    def __init__(self, path, alignment: Alignment, *,
                 radius_m: float = PEER_RADIUS_M,
                 timeout_s: float = PEER_TIMEOUT_S,
                 domain: str | None = None) -> None:
        self.path = Path(path)
        self.alignment = alignment
        self.radius_m = float(radius_m)
        self.timeout_s = float(timeout_s)
        #: This host's clock domain, resolved once. Injectable so a test can drive the
        #: mismatch branch without rebooting.
        self.domain = clock_domain() if domain is None else domain
        #: The most recent reading. ``MappoPlanner`` reads this rather than re-deriving
        #: it, so that the hold and the obstacle list are the same tick's decision.
        self.last: PeerLink | None = None
        #: Counted over a run and printed at the end. A link that held for a third of the
        #: run and a link that never held are both worth knowing about, and neither is
        #: visible in a log of velocities.
        self.counts = {"reads": 0, "held": 0, "peers": 0}
        #: Worst age any peer was acted on at, over the run. Reported rather than a count
        #: over some threshold: a threshold here would be a second staleness number with
        #: no argument behind it, and the useful question after a run is "how close did
        #: the link get to the limit", which this answers directly.
        self.worst_age_s = 0.0

    def read(self, now_s: float) -> PeerLink:
        """The current peer obstacles, or a reason to hold.

        ``now_s`` MUST be a ``time.monotonic()`` reading on this host. The vendored loop
        already passes exactly that into ``_obstacles(now)``, which is where this is
        called from, so the correct value is the one already in scope — but it is worth
        the sentence, because handing it ``time.time()`` computes ages of about 1.8e9 s,
        every peer reads as lost, and the robot holds forever under an unexplained
        diagnosis. That is the safe direction of the same bug ``mappo_bridge.robot_input``
        documents, which is the only reason it is a footnote here and a banner there.
        """
        self.counts["reads"] += 1
        link = self._read(now_s)
        self.last = link
        if link.holds:
            self.counts["held"] += 1
        self.counts["peers"] = max(self.counts["peers"], len(link.obstacles))
        if link.ages_s:
            self.worst_age_s = max(self.worst_age_s, max(link.ages_s.values()))
        return link

    def _read(self, now_s: float) -> PeerLink:
        document = self._load()
        if isinstance(document, str):
            return PeerLink(link_error=document)

        written = _finite(document.get("written_monotonic_s"))
        if written is None:
            return PeerLink(link_error="the spool carries no heartbeat")
        link_age = now_s - written
        if link_age < 0.0:
            # The writer's clock is ahead of the reader's, which cannot happen within one
            # boot on one host. Rejecting rather than clamping to zero: a clamp turns the
            # one observation that proves the clocks are unrelated into a fresh reading.
            return PeerLink(link_age_s=link_age,
                            link_error=f"heartbeat is {-link_age:.1f} s in the future — "
                                       f"the spool was written by another boot or host")
        if link_age > self.timeout_s:
            return PeerLink(link_age_s=link_age,
                            link_error=f"no heartbeat for {link_age:.1f} s — peer_link.py "
                                       f"is not running")

        expected = document.get("expect")
        if not isinstance(expected, list) or not expected:
            # An empty roster would make every check below vacuous and report "no peers
            # lost" for a link watching nothing at all.
            return PeerLink(link_age_s=link_age,
                            link_error="the spool expects no peers, so it can never "
                                       "report one missing")

        peers = document.get("peers")
        peers = peers if isinstance(peers, dict) else {}
        obstacles, lost, ages = [], {}, {}
        for device in sorted(str(d) for d in expected):
            record = peers.get(device)
            if not isinstance(record, dict):
                lost[device] = "has never reported a pose"
                continue
            obstacle, age, why = self._obstacle(device, record, now_s)
            if obstacle is None:
                lost[device] = why
                continue
            obstacles.append(obstacle)
            ages[device] = age
        return PeerLink(obstacles=obstacles, lost=lost, ages_s=ages,
                        link_age_s=link_age)

    def _load(self):
        """The parsed spool, or a string saying why there is not one.

        Every failure to read is a hold, so they are all one return type. A missing file
        is the commonest and the most misleading: it means ``peer_link.py`` was never
        started, which looks from the robot exactly like a mesh with no peers on it.
        """
        try:
            with self.path.open() as handle:
                document = json.load(handle)
        except FileNotFoundError:
            return (f"no spool at {self.path} — peer_link.py is not running, "
                    f"so nothing on the mesh is being watched")
        except OSError as exc:
            return f"cannot read {self.path}: {exc}"
        except ValueError as exc:
            # A torn read. peer_link.py writes through os.replace so this should not
            # happen; it is handled anyway, because "should not happen" is not a safety
            # argument and the cost of handling it is one tick of holding.
            return f"{self.path} is not valid JSON: {exc}"
        if not isinstance(document, dict):
            return f"{self.path} is not a spool document"
        schema = document.get("schema")
        if schema != SCHEMA:
            return f"spool schema {schema!r}, expected {SCHEMA!r}"
        domain = document.get("clock_domain") or ""
        if self.domain and domain and domain != self.domain:
            return ("the spool was written under a different boot, so its monotonic "
                    "timestamps mean nothing here")
        return document

    def _obstacle(self, device: str, record: dict, now_s: float) -> tuple:
        """One peer's obstacle record, or ``(None, age, why it cannot be used)``."""
        received = _finite(record.get("received_monotonic_s"))
        if received is None:
            return (None, None, "carries no arrival time")
        # The peer's own sampling latency is a DURATION measured at the source, so it
        # crosses the mesh where a timestamp could not. Adding it is what makes this the
        # age of the MEASUREMENT rather than the age of the message.
        emit_lag = _finite(record.get("sample_age_at_emit_s")) or 0.0
        age = (now_s - received) + max(0.0, emit_lag)
        if age < 0.0:
            return (None, age, f"arrived {-age:.1f} s in the future")
        if age > self.timeout_s:
            return (None, age, f"pose is {age:.2f} s old (limit {self.timeout_s:.2f} s)")

        x, y, yaw = (_finite(record.get(k)) for k in ("x", "y", "yaw"))
        vx, vy = (_finite(record.get("vx")), _finite(record.get("vy")))
        if None in (x, y, yaw, vx, vy):
            return (None, age, "reported a pose this robot cannot use")

        # THE PEER'S VELOCITY IS BODY FRAME. ``Go2Locomotion.velocity()`` reads
        # ``SportModeState_.velocity``, the estimator's body-frame vector, which is why
        # ``mappo_bridge.VELOCITY_FRAME`` is "body" — and an obstacle's velocity is ODOM
        # (``telemetry.FRAMES``: "obstacles: odom, position AND velocity"). Two rotations
        # are needed, not one: out of the peer's body frame by the peer's own yaw, then
        # out of the peer's odom frame by the alignment. Skipping the first is invisible
        # while both robots face their start headings and wrong the moment either turns,
        # which is the same quiet failure the frame constant was written to prevent.
        cos, sin = math.cos(yaw), math.sin(yaw)
        peer_odom_v = (vx * cos - vy * sin, vx * sin + vy * cos)

        px, py = self.alignment.point(x, y)
        pvx, pvy = self.alignment.direction(*peer_odom_v)

        # Advance to now on the last velocity, and grow the radius by what the coast
        # could have cost if the peer accelerated. Same formula as visual_nav._obstacles;
        # NOT the double-count its docstring warns about, because nothing has advanced
        # this record already — see the module docstring.
        inflation = 0.5 * PEER_ACCEL_SIGMA_M_S2 * age * age
        radius = _finite(record.get("radius_m"))
        if radius is None or radius <= 0.0:
            radius = self.radius_m
        return ({
            "label": PEER_LABEL,
            # A mesh peer is a ROBOT, asserted by the peer itself rather than guessed
            # from a box, so it is never person-shaped. Stated explicitly because
            # `mappo_bridge.holds_the_robot` defaults this True for any producer that
            # does not judge shape, and a mesh peer that fell to that default would hold
            # the robot for exactly the object this path exists to route to the policy.
            "person_shaped": False,
            "kind": "tracked",
            "id": f"peer-{device}",
            "x": px + pvx * age,
            "y": py + pvy * age,
            "vx": pvx,
            "vy": pvy,
            "radius_m": radius + inflation,
            # soft_gap_m and hard_gap_m are left to the planner's defaults, which are a
            # person's. A landmark gets tighter ones because a bin cannot step sideways
            # into the gap the robot has committed to; a peer robot certainly can.
        }, age, "")

    def report(self) -> str:
        counts = self.counts
        if not counts["reads"]:
            return "[peer_source] never read"
        return (f"[peer_source] {counts['reads']} reads, {counts['peers']} peer(s), "
                f"{counts['held']} ticks held for a stale or missing peer, worst pose age "
                f"{self.worst_age_s:.3f} s against a {self.timeout_s:.2f} s limit")


def spool_document(peers: dict, expect, *, domain: str = "",
                   written_monotonic_s: float | None = None) -> dict:
    """Build a spool document. Here rather than in ``peer_link.py`` so the writer and the
    reader are pinned to one definition of the format and a test can exercise both."""
    return {
        "schema": SCHEMA,
        "clock_domain": domain or clock_domain(),
        "written_monotonic_s": (time.monotonic() if written_monotonic_s is None
                                else written_monotonic_s),
        "expect": sorted(str(d) for d in expect),
        "peers": peers,
    }


def write_spool(path, document: dict) -> None:
    """Replace the spool atomically.

    ``os.replace`` rather than writing in place: the reader is a control loop at 10 Hz and
    a torn read would be a peer at a half-written position. The reader handles a parse
    failure by holding, so a torn read costs one tick rather than a run — but a hold that
    can be designed out should be, and the temp-file-plus-rename is two lines.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(document, handle)
    os.replace(str(temp), str(path))
