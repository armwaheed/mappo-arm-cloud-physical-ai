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
    """Whether this obstacle belongs in ``stationary_objects`` rather than a hold.

    Prefers the explicit ``kind`` field and falls back to :data:`MOVER_SPEED_MPS`.
    """
    kind = obstacle.get("kind")
    if kind is not None:
        return kind == "static"
    return math.hypot(obstacle.get("vx", 0.0), obstacle.get("vy", 0.0)) < MOVER_SPEED_MPS


def external_hold(tick: dict) -> bool:
    """Whether the existing moving-object logic is holding the robot this tick.

    A hold counts as external only when a MOVER is what the planner could not clear.
    A hold caused solely by the mapped static obstacle is deliberately NOT propagated:
    that is the situation the policy is being asked to solve, and zeroing its command
    there would make the whole integration a no-op in exactly the scene it was built for.

    A stale-perception hold IS external — the robot is blind, and a policy acting on a
    frozen world model is worse than a policy that does not act.
    """
    if (tick.get("perception") or {}).get("stale"):
        return True
    command = tick.get("command")
    if not command or command.get("reason") != "hold":
        return False
    return any(not is_stationary(o) for o in tick.get("obstacles", []))


def stationary_objects(tick: dict) -> list:
    """``StationaryObject`` kwargs for one tick, in the robot's body frame.

    Bearing is measured from the robot's nose, positive to the left (CCW), which is the
    convention :class:`StationaryObject` documents. ``radius_m`` is passed through
    already inflated for position uncertainty — the planner treats it as a hard
    footprint and so should the policy.

    An obstacle the robot is standing INSIDE is still emitted. The range caster on the
    far side reports zero for it, which is the correct and conservative reading; dropping
    it here would report clear space instead.
    """
    pose = tick["pose"]
    out = []
    for obstacle in tick.get("obstacles", []):
        if not is_stationary(obstacle):
            continue
        x, y, radius = (_finite(obstacle.get(k)) for k in ("x", "y", "radius_m"))
        if x is None or y is None or radius is None:
            continue                       # a half-recorded obstacle is not a detection
        body_x, body_y = to_body_frame(pose, x, y)
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
                monotonic_s: float | None = None) -> dict | None:
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
        "stationary_objects": stationary_objects(tick),
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
