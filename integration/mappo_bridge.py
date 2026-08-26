# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn one telemetry tick into the argument of ``MappoController.step``.

This is the whole integration surface. The policy package asks for one call per control
cycle with a ``RobotInput``; the control stack already writes one telemetry line per
control cycle. This module is the map between them, and it exists as code rather than as
a table in a message because three of the mappings are not the obvious ones:

  * **Velocity is BODY frame, not odom.** ``measured`` comes from the Go2's own state
    estimator via ``Go2Locomotion.velocity()``, which documents itself as "measured
    body-frame velocity". The integration note that came with the policy package passes
    ``velocity_frame="odom"``. That is wrong for this stack, and wrong in the quiet
    direction: the two frames agree exactly while the robot faces its start heading and
    diverge as it turns, so a bench test at yaw 0 cannot tell them apart. See
    :data:`VELOCITY_FRAME`.

  * **Obstacles are odom, the policy wants body-polar.** Telemetry records obstacles in
    odom because that is the only frame in which a mapped landmark stays put while the
    robot walks. ``RobotInput.stationary_objects`` wants distance + bearing from the
    robot's nose. The conversion is here, once, rather than open-coded at the call site.

  * **"Held" is not the same as "held by a mover".** ``external_hold`` is specified as
    "the existing moving-object stop/wait logic". The planner's ``reason`` field does not
    carry that: it emits ``hold`` whenever no candidate trajectory clears EVERY obstacle,
    and :class:`avoidance.Obstacle` deliberately treats a static one as "simply one with
    zero velocity" with no special case. So a bin blocking the lane and a person blocking
    the lane produce the identical record. Feeding that straight through hands the policy
    a permanent zero exactly when the static obstacle it is there to solve is in the way.
    :func:`external_hold` reads the per-obstacle ``kind`` field when the log has one and
    falls back to a speed threshold when it does not — see :data:`MOVER_SPEED_MPS` for
    why the fallback is a stopgap and not a fix.

  * **A peer that is not there is not the same as a peer that is not moving.** Peer poses
    arriving over the Device Connect mesh reach this file as ordinary ``kind="tracked"``
    obstacles and need no special case — that is the point of routing them through
    ``visual_nav._obstacles``' shape. What DOES need one is the link itself going quiet:
    :func:`external_hold` reads a ``peer_link`` block, because the producer drops the
    obstacle when it can no longer date it and a dropped obstacle with no hold is a robot
    planning through a peer. See ``peer_source``.

  * **Movers are not all the same.** Until now every track was a hold, which meant a peer
    robot — the whole point of a MULTI-agent demo — reached the policy as a single
    boolean meaning *stop*, and the avoidance was 100% incumbent planner. The split is now
    two-tier: a **person** holds, always, by label; anything moving faster than
    :data:`POLICY_MAX_MOVER_SPEED_MPS` holds, whatever it is; everything else, a parked or
    shuffling peer included, is handed to the policy as one more disc. See
    :func:`holds_the_robot`. ⚠️ The policy's observation carries no obstacle-velocity
    channel, so a mover enters it as an instantaneous disc — that speed gate is the only
    thing standing between the policy and a problem it was never trained on.

Pure stdlib and no numpy, matching :mod:`observation`: this has to be importable in a
test environment that has no policy package, and the numpy dependency belongs to the
policy, not to the mapping. :func:`robot_input` returns a plain dict whose keys are
``RobotInput`` field names, so the caller writes ``RobotInput(**robot_input(tick))``.

``python3 test_mappo_bridge.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from observation import to_body_frame, wrap_pi

#: What ``velocity_frame`` this stack's ``measured`` block is in. Body, per
#: ``locomotion/go2_locomotion.py``: ``SportModeState_.velocity`` is the estimator's
#: body-frame ``(vx, vy)``. Named rather than inlined because it is the single most
#: consequential constant here and the one most likely to be "corrected" by someone
#: reading the policy package's example instead of the robot's source.
VELOCITY_FRAME = "body"

#: Fallback classifier for logs written before obstacles carried a ``kind``. A mapped
#: static landmark is emitted with ``vx = vy = 0.0`` EXACTLY (``visual_nav._obstacles``
#: constructs it that way), while a track carries the filter's estimate. Over the
#: 122-tick reference run the separation is total: 192 bin detections at exactly 0.000
#: and 15 person detections at 0.143-0.696 m/s.
#:
#: It is still only a stopgap, because it misclassifies the one case that matters most —
#: a person who has STOPPED reads as a stationary object and gets handed to the policy as
#: something to path around, when the whole agreed division of labour is that movers are
#: the existing stop-and-wait logic's job. The fix is the ``kind`` field; this exists so
#: the bridge can read the runs that already exist.
MOVER_SPEED_MPS = 0.05

#: Labels that keep the STOP path no matter what. A person is not an obstacle to be
#: pathed around by a policy that was trained on static discs, has no obstacle-velocity
#: channel, and responds as a cliff rather than a ramp (1.8 deg of steering outside
#: 0.525 m, 96.6 deg inside). Holding for a person is the conservative behaviour and the
#: one the README describes; routing them into the policy would change what the demo
#: claims as well as what it does.
#:
#: This is the tier boundary, and it sits where the detector is strongest: ``person`` is
#: the one class MobileNet-SSD is actually good at, measured at 0.93-0.97 on this robot's
#: own footage.
HOLD_LABELS = ("person",)

#: A tracked object faster than this holds the robot regardless of its label.
#:
#: The policy's 18-value observation carries NO obstacle-velocity channel — a mover
#: enters as an instantaneous disc at wherever it happened to be — so it can only be
#: right about something that has barely moved by the time the command lands. With a
#: 0.875 m sensing horizon and a 10 Hz loop, a peer closing at 0.7 m/s relative leaves
#: about 1.2 s between first response and contact, against a response that saturates at
#: ~100 deg the moment it fires.
#:
#: So the split is not really person-versus-robot, it is slow-versus-fast: a parked or
#: shuffling peer is inside what the policy was trained for, and anything with intent is
#: not. 0.25 m/s is below this robot's own 0.35 m/s gait floor, i.e. slower than the
#: slowest walk a peer quadruped can produce, which makes "it is manoeuvring, not
#: travelling" the thing being tested. ⚠️ Chosen from the policy's measured horizon and
#: not from a sweep of peer speeds; a shadow run with a peer driven at 0.2/0.4/0.6 m/s
#: is what would justify it.
POLICY_MAX_MOVER_SPEED_MPS = 0.25

#: How far ahead a moving obstacle's disc is grown, seconds.
#:
#: THE POLICY CANNOT SEE MOTION, and this is how it is told anyway. Its 18-value
#: observation is ``[x, y, vx, vy, x-gx, y-gy, *12 lidar]`` where ``vx, vy`` are the
#: ROBOT'S OWN; there is no channel for an obstacle's velocity, so a crossing peer enters
#: as an instantaneous disc wherever it happened to be. Measured in simulation with the
#: peer's EXACT position handed over every tick, the policy sees it (ray proximity 0.176,
#: well inside the 0.875 m horizon) and responds by STOPPING — forward command to zero —
#: with a closest approach of 0.194 m at the worst crossing speed.
#:
#: A disc does not have to mean "where it is". Growing it by ``speed * horizon`` makes the
#: ray cast report where the peer WILL be, which is the one thing the policy's only input
#: can express. The planner's own rollout already reasons this way in ``avoidance._gaps``;
#: this gives the policy the same idea through the channel it has.
#:
#: Measured, same simulation, clearance against the TRUE disc:
#:
#:   ==========  ==============  =================
#:   peer m/s    disc as-is      grown at 1.5 s
#:   ==========  ==============  =================
#:   0.10        0.148 m         0.275 m
#:   0.20        0.194 m         0.505 m
#:   0.35        0.537 m         0.791 m
#:   0.50        0.790 m         1.004 m
#:   ==========  ==============  =================
#:
#: and mean steering deflection at 0.20 m/s rises 9.5 deg -> 24.2 deg, attributed against a
#: paired ablated control with the peer removed. That is the policy steering around the
#: peer rather than halting in front of it.
#:
#: ISOTROPIC, NOT SWEPT, and that was the surprise. Placing the disc at the peer's
#: predicted position instead — directional, and self-cancelling once the peer is past —
#: was measured WORSE at every speed (0.376 m against 0.505 m at 0.20 m/s). Growing
#: backwards as well keeps pushing the robot away during the approach, and that margin is
#: worth more than the false conservatism costs.
#:
#: BOUNDED BY THE GATE ABOVE, with no clamp needed. Only an obstacle slower than
#: :data:`POLICY_MAX_MOVER_SPEED_MPS` reaches the policy at all — anything faster holds the
#: robot — so the largest disc the policy can ever be shown is its true radius plus
#: ``0.25 * 1.5 = 0.375 m``. The two mechanisms compose; neither needs to know about the
#: other.
#:
#: 1.5 s rather than the planner's 2.5 s veto horizon: 2.5 was also measured and gained
#: nothing at the speeds that matter (0.460 m against 0.505 m at 0.20 m/s), while inflating
#: every disc further into space the robot might legitimately use.
#:
#: ⛔ NOTHING PASSES THIS YET, AND THAT IS DELIBERATE. It is the value a caller opts into
#: via ``policy_objects(..., motion_horizon_s=...)``; the default is 0.0, so every shipped
#: call site — :func:`robot_input`, and therefore ``mappo_drive``, ``mappo_shadow`` and
#: ``replay_mappo`` — still hands the policy the true disc. The encoding has been measured
#: in simulation ONLY. It has never run on a robot, and the swerve it commands is
#: 0.085-0.182 m/s, below this robot's lateral gait floor: it would not walk. Turning it on
#: before that floor is measured on hardware would change what the demo does without
#: changing what it can execute.
POLICY_MOTION_HORIZON_S = 1.5


@dataclass(frozen=True)
class BridgeReport:
    """What the mapping could not do cleanly, counted over a run.

    Every field here is a silent failure in the making, so the replay tool prints them
    and a live integration should log them once at the end of a run. A bridge that
    quietly substitutes zeros is how an integration passes a bench test and drives into
    a bin.
    """

    ticks: int = 0
    #: Ticks with no goal yet — the robot is still searching. Not an error.
    no_goal: int = 0
    #: Ticks where ``measured`` was absent and velocity had to be assumed zero.
    velocity_missing: int = 0
    #: Ticks where a hold had to be classified by speed because no ``kind`` was present.
    hold_classified_by_speed: int = 0
    #: Obstacles that carried no stable identity, so the policy had to re-associate them
    #: by position on every tick.
    unidentified_objects: int = 0

    def merge(self, **counts) -> BridgeReport:
        """Add one tick's counts. Rejects a key it does not know rather than dropping it
        — a mistyped counter that silently stays at zero reads as "no problems found"."""
        unknown = set(counts) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"BridgeReport has no counter(s) {sorted(unknown)}")
        return BridgeReport(**{field: getattr(self, field) + counts.get(field, 0)
                               for field in self.__dataclass_fields__})

    def lines(self) -> list:
        """Human-readable summary; empty list when the mapping was clean."""
        out = []
        if self.no_goal:
            out.append(f"{self.no_goal}/{self.ticks} ticks had no goal (searching)")
        if self.velocity_missing:
            out.append(f"{self.velocity_missing}/{self.ticks} ticks had no measured "
                       f"velocity — assumed zero, which the policy reads as 'stopped'")
        if self.hold_classified_by_speed:
            out.append(f"{self.hold_classified_by_speed} holds classified by SPEED, not "
                       f"by kind — a stopped person is indistinguishable from a bin")
        if self.unidentified_objects:
            out.append(f"{self.unidentified_objects} obstacle records had no stable id — "
                       f"the policy re-associates by position within 0.45 m")
        return out


def _finite(value) -> float | None:
    """A usable number, or ``None``.

    JSON has no infinity, so the writer emits ``null`` for any value that was not finite
    — which means a pose, a goal or a velocity can legitimately arrive as ``None`` in a
    well-formed file. Reaching for ``tick["pose"]["x"]`` and trusting it is how a
    consumer turns that into a ``TypeError`` several frames away from the cause, or
    worse, into a NaN that the policy's own finite-check rejects at the bottom of the
    call stack with nothing to say about which field it was.
    """
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value) if math.isfinite(value) else None


def is_stationary(obstacle: dict) -> bool:
    """Whether this obstacle is a MAPPED LANDMARK rather than a track.

    Prefers the explicit ``kind`` field and falls back to :data:`MOVER_SPEED_MPS`. Note
    this is no longer the same question as "does the policy see it" — see
    :func:`holds_the_robot` and :func:`policy_objects` for that split. It still answers
    "is this a mapped prop", which is what the audit counts and what ``replay_mappo``
    reports.
    """
    kind = obstacle.get("kind")
    if kind is not None:
        return kind == "static"
    return math.hypot(obstacle.get("vx", 0.0), obstacle.get("vy", 0.0)) < MOVER_SPEED_MPS


def holds_the_robot(obstacle: dict) -> bool:
    """Whether this obstacle stops the robot instead of being handed to the policy.

    Two tiers, and the boundary is deliberate.

    A **person-shaped** obstacle always holds. A **fast mover** always holds, by speed,
    whatever it is — see :data:`POLICY_MAX_MOVER_SPEED_MPS`. Everything else, including a
    parked or slowly-manoeuvring peer robot, goes to the policy as one more disc in its
    ray cast, which is exactly how the policy already treats every mapped landmark. It has
    never had a notion of what an obstacle *is*; only where it is and how big it is.

    A mapped landmark never holds: that is the situation the policy exists to solve, and
    holding for it would make the integration a no-op in the one scene it was built for.

    ⚠️ THE FIRST TIER IS SHAPE, NOT LABEL, AND THAT CHANGED FOR A REASON. It used to read
    ``obstacle.get("label") in HOLD_LABELS``, which needed the detector to tell a person
    from a robot. It cannot. On 12 consecutive live frames the Go2 Wheel was labelled
    ``person`` every time, and across the 2026-08-24 corpus the same peer came back as
    ``motorbike`` 613 times, ``chair`` 372, ``aeroplane`` 200 and ``person`` 109. So the
    old rule failed in BOTH directions at once: it held for the peer this integration
    exists to route, and it silently handed the policy anyone the detector called
    ``motorbike``. The human-safety property it looked like it provided, it did not.

    ``person_shaped`` is decided on box aspect by
    ``person_detector.RangedDetection.person_shaped`` — scale-free, so it needs no range,
    which matters because this robot has no independent one. It defaults to True here as
    well as at every producer, so a telemetry tick from an older writer, or any obstacle
    from a source that does not judge shape, lands on the stopping side.

    :data:`HOLD_LABELS` is kept as a backstop rather than deleted: a producer that still
    labels something ``person`` on purpose — the mesh, a future detector — keeps its hold.
    """
    if is_stationary(obstacle):
        return False
    if obstacle.get("person_shaped", True):
        return True
    if obstacle.get("label") in HOLD_LABELS:
        return True
    speed = math.hypot(obstacle.get("vx", 0.0) or 0.0, obstacle.get("vy", 0.0) or 0.0)
    return speed > POLICY_MAX_MOVER_SPEED_MPS


def external_hold(tick: dict) -> bool:
    """Whether the existing moving-object logic is holding the robot this tick.

    A hold counts as external only when a MOVER is what the planner could not clear.
    A hold caused solely by the mapped static obstacle is deliberately NOT propagated:
    that is the situation the policy is being asked to solve, and zeroing its command
    there would make the whole integration a no-op in exactly the scene it was built for.

    A stale-perception hold IS external — the robot is blind, and a policy acting on a
    frozen world model is worse than a policy that does not act.

    A LOST PEER LINK is external for exactly that reason, one input over. A peer pose
    that stopped arriving is not a peer standing still, it is a peer whose position is no
    longer known, and ``peer_source`` drops the obstacle when that happens — so the hold
    is not an extra caution on top of a disc that is still there, it is the only thing
    left. The two halves are one decision and separating them is how this becomes unsafe.
    Absent from a tick means "no peer link configured", which is every recorded run in
    ``evidence/`` and must stay false.
    """
    if (tick.get("perception") or {}).get("stale"):
        return True
    if (tick.get("peer_link") or {}).get("lost"):
        return True
    command = tick.get("command")
    if not command or command.get("reason") != "hold":
        return False
    return any(holds_the_robot(o) for o in tick.get("obstacles", []))


def policy_objects(tick: dict, *, motion_horizon_s: float = 0.0) -> list:
    """Obstacle kwargs for one tick, in the robot's body frame.

    NAMED FOR WHAT IT IS. These become the policy's ``stationary_objects``, because that
    is the field name the vendored package uses, but the set is no longer only stationary
    things: a peer robot that is parked or shuffling is in here too. The policy's ray
    caster never asked whether a disc was moving — it asks where the disc is — so the
    field name is the stale part, not the behaviour. Renaming it in ``policy/`` is a
    change to Sagar's package and belongs in a conversation with him, not in this commit.

    Bearing is measured from the robot's nose, positive to the left (CCW), which is the
    convention :class:`StationaryObject` documents. ``radius_m`` is passed through
    already inflated for position uncertainty — the planner treats it as a hard
    footprint and so should the policy.

    An obstacle the robot is standing INSIDE is still emitted. The range caster on the
    far side reports zero for it, which is the correct and conservative reading; dropping
    it here would report clear space instead.

    ``motion_horizon_s`` GROWS A MOVING OBSTACLE'S DISC by ``speed * motion_horizon_s``,
    which is how motion is expressed to a policy that has no obstacle-velocity channel —
    see :data:`POLICY_MOTION_HORIZON_S` for why, and for the measurements.

    ⛔ IT DEFAULTS TO 0.0, i.e. OFF, and no caller in this repository passes anything else.
    The encoding is measured in simulation and unrun on hardware, and the lateral command
    it produces is below the gait floor, so it is landed inert rather than enabled. A zero
    horizon is not a special case: the growth term is a multiplication, so it vanishes.

    ⚠️ When it IS enabled, the growth applies ONLY here. The planner and its feasibility
    veto keep the true radius and do their own rollout, so nothing double-counts, and
    every clearance this repository reports is still measured against the real disc.
    """
    pose = tick["pose"]
    out = []
    for obstacle in tick.get("obstacles", []):
        if holds_the_robot(obstacle):
            continue
        x, y, radius = (_finite(obstacle.get(k)) for k in ("x", "y", "radius_m"))
        if x is None or y is None or radius is None:
            continue                       # a half-recorded obstacle is not a detection
        body_x, body_y = to_body_frame(pose, x, y)
        # Grow the disc by where the obstacle is going. Falls out to zero for anything
        # stationary AND for the default horizon of 0.0, so a mapped landmark is
        # unaffected and the shipped behaviour is unchanged; no branch is needed.
        speed = math.hypot(_finite(obstacle.get("vx")) or 0.0,
                           _finite(obstacle.get("vy")) or 0.0)
        radius += speed * motion_horizon_s
        out.append({
            "distance_m": math.hypot(body_x, body_y),
            "bearing_rad": wrap_pi(math.atan2(body_y, body_x)),
            "radius_m": radius,
            # `id` is the stable per-object identity; `label` is a CLASS name and two
            # bins would share it. Passing a class where an identity is expected is how
            # two objects merge into one at the policy's 0.45 m association threshold.
            "object_id": obstacle.get("id"),
        })
    return out


def robot_input(tick: dict, *, reset_run: bool = False,
                monotonic_s: float | None = None,
                motion_horizon_s: float = 0.0) -> dict | None:
    """``RobotInput`` kwargs for one tick, or ``None`` if the tick has no goal.

    A tick without a goal is a real part of the episode — the robot is searching — but
    it has no policy input. Returning ``None`` makes the caller decide, rather than
    handing the policy a goal at the origin, which it would drive towards.

    ``monotonic_s`` MUST BE A ``time.monotonic()`` READING OR ABSENT. The policy compares
    it against its own ``time.monotonic()`` to decide ``STOP_STALE_INPUT``, and the two
    clocks share no epoch: hand it the telemetry's ``wall_time`` and the computed age is
    about -1.8e9 seconds, which is under any threshold, so the staleness gate silently
    never fires. It fails OPEN, in the direction where a frozen world model keeps
    driving the legs. Leaving it ``None`` lets the policy stamp its own clock, and the
    staleness that this stack can actually measure — perception age — is already carried
    into :func:`external_hold`.

    ``motion_horizon_s`` is handed straight to :func:`policy_objects` and defaults to 0.0.
    ⛔ No caller passes it; see :data:`POLICY_MOTION_HORIZON_S` for what it would do and
    for the hardware measurement that has to land before anything should.
    """
    if not tick.get("goal"):
        return None
    pose = tick.get("pose") or {}
    measured = tick.get("measured") or {}
    x, y, yaw = (_finite(pose.get(k)) for k in ("x", "y", "yaw"))
    goal_x, goal_y = (_finite(tick["goal"].get(k)) for k in ("x", "y"))
    vx, vy = (_finite(measured.get(k)) for k in ("vx", "vy"))
    if None in (x, y, yaw, goal_x, goal_y):
        # Not an error and not a searching tick: the run is recorded, but this one line
        # cannot be turned into a policy input. Same contract as a missing goal — the
        # caller decides, rather than being handed a pose at the origin.
        return None
    return {
        "x_m": x,
        "y_m": y,
        "yaw_rad": yaw,
        # A non-finite velocity is recorded as null. It becomes zero here because the
        # policy needs a number, but it is "not measured" rather than "not moving", and
        # :class:`BridgeReport` counts it as such.
        "vx_mps": 0.0 if vx is None else vx,
        "vy_mps": 0.0 if vy is None else vy,
        "velocity_frame": VELOCITY_FRAME,
        "goal_x_m": goal_x,
        "goal_y_m": goal_y,
        "external_hold": external_hold(tick),
        "stationary_objects": policy_objects(tick, motion_horizon_s=motion_horizon_s),
        "reset_run": reset_run,
        "timestamp_s": monotonic_s,
    }


def audit(tick: dict) -> dict:
    """Per-tick counts for :class:`BridgeReport`. Separate from :func:`robot_input` so
    the mapping stays a pure function of its input and the bookkeeping stays optional."""
    obstacles = tick.get("obstacles") or []
    held = tick.get("command") or {}
    measured = tick.get("measured") or {}
    # A `measured` block whose components are null counts as missing, because that is
    # what it means downstream: the substituted zero is an assumption either way, and a
    # counter that only noticed the absent block would under-report it.
    velocity = [_finite(measured.get(k)) for k in ("vx", "vy")]
    return {
        "ticks": 1,
        "no_goal": 0 if tick.get("goal") else 1,
        "velocity_missing": 1 if None in velocity else 0,
        "hold_classified_by_speed": (
            1 if held.get("reason") == "hold"
            and any(o.get("kind") is None for o in obstacles) else 0),
        "unidentified_objects": sum(1 for o in obstacles
                                    if is_stationary(o) and o.get("id") is None),
    }
