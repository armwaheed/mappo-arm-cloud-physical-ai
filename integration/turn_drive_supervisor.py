#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn-drive execution supervisor: avoidance projected onto measured primitives.

THE MISMATCH THIS EXISTS FOR. The shipped MAPPO checkpoint is a holonomic VMAS agent: it
can ask for any ``(vx, vy, wz)``, and threading a gap is exactly where it asks for a
sideways component. The Lite3 it now drives has been measured to perform THREE motions
and no more — straight ahead, and a left or right turn in place
(``lite3_axis_profile_LITE3-A.json``: ``forward_positive`` and both yaw primitives
evidenced, lateral and reverse ``null``). The deployment SOP therefore runs with
``--max-vy 0``, which deletes the policy's lateral intent at the envelope clamp, and the
axis mapping is sign-only besides, so no magnitude survives either. What was observed on
2026-08-26 was the consequence: the robot walked straight at the bin it had detected,
because "detected" and "able to execute the avoidance the policy wants" are different
properties and only the first held.

This module is option B of the two legitimate fixes, chosen over commissioning an
unmeasured lateral primitive: restrict the AVOIDANCE to motions the physical execution
layer has actually demonstrated. When a mapped static obstacle blocks the straight line
from the robot to the goal, the supervisor replaces the policy's command with a
two-segment detour — out to a waypoint beside the obstacle, then on to the goal —
executed as a sequence of pure turns and pure straight drives. No segment ever asks for
a lateral component, and no segment combines forward with yaw, because a sign-only
mapping that snaps a diagonal to one of eight directions does not trace the arc the
combined command was planned as.

THE WAYPOINT IS A TANGENT-LINE INTERSECTION, and the reason is a measurement, not a
taste. The naive placement — the point beside the obstacle at the required gap,
perpendicular to the robot→goal line — is unreachable by a straight first leg: the leg
from the robot to that point passes CLOSER to the obstacle than the waypoint itself
(for a blocker 1.5 m ahead needing an 0.85 m gap, the leg dips to 0.79 of it), so every
candidate fails its own clearance check and the supervisor can never act. The corner of
the two tangent half-lines — from the robot to the required-clearance circle, and from
the goal back to it — is the closest point whose BOTH legs touch the circle at exactly
the required gap and stay outside it everywhere else. That is also why the leg check
compares with ``<`` and not ``<=``: a tangent leg's minimum distance IS the gap, and a
test that rejects equality rejects every detour this module can produce.

WHAT IT DELIBERATELY DOES NOT DO:

* It does not replace the policy in the open. With the line to the goal clear it
  returns ``None`` and the policy's command stands — the policy remains the controller
  this repository is evaluating, and a supervisor that drove the whole run would make
  every subsequent telemetry line evidence about itself.
* It does not reason about PEOPLE. A tracked mover on the detour path is the shared
  feasibility check's business, and ``mappo_drive`` runs that check on the supervisor's
  command exactly as it does on the policy's, after this module has answered. The
  blocker search below ignores anything that is not ``kind="static"`` so that a person
  crossing the goal line triggers the veto rather than a detour around a being that
  will have moved by the time the robot gets there.
* It is stateless. The detour is recomputed from the current pose, goal and map on
  every tick, so a stale waypoint, a re-planned goal or a re-observed obstacle cannot
  accumulate. The "phase" it reports is derived from the heading error of THIS tick,
  not remembered from the last one.

Pure stdlib. Run: ``python3 test_turn_drive_supervisor.py``
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

_STACK = (Path(__file__).resolve().parent.parent / "robot-stack" / "unitree" / "go2"
          / "visual_nav")
if str(_STACK) not in sys.path:
    sys.path.insert(0, str(_STACK))

from avoidance import STATIC_HARD_GAP_M  # noqa: E402
from geometry import wrap_pi  # noqa: E402

#: How squarely the nose must be on the next leg before driving it. Fifteen degrees is
#: wide enough that the sign-only yaw primitive (measured 0.86 rad/s, one tick of which
#: is ~0.09 rad at the loop's 10 Hz) does not oscillate across the boundary, and narrow
#: enough that driving starts with the robot genuinely pointing at the waypoint rather
#: than vaguely toward it.
HEADING_TOLERANCE_RAD = math.radians(15.0)

#: How close to the waypoint the robot must be before the SECOND leg is even
#: considered — a necessary condition, not a sufficient one: the goal becomes the
#: target only when the straight line from the current pose to it also keeps the
#: required gap (see :meth:`TurnDriveSupervisor.command`). Tighter would dither
#: against odometry noise; wider would make the wait for the clearance condition
#: longer, never wrong.
WAYPOINT_ARRIVAL_M = 0.30

#: Comparison epsilon on clearance tests. The tangent legs touch the required gap
#: exactly, by construction, and binary64 evaluation of two algebraically-equal
#: expressions can land either side of it by one ulp. A nanometre-scale band rejects
#: real violations and passes the tangent.
_CLEARANCE_EPS_M = 1e-9

#: Free space the detour keeps ON TOP OF the stack's static hard gap. The tangent
#: construction puts each leg's minimum clearance at exactly the required gap, and the
#: planner's veto compares with ``>=`` at that same number — so at zero margin the
#: accept/refuse decision is made by float rounding and the landing spot of the veto's
#: 0.125 s rollout samples, not by geometry. Two centimetres moves the detour strictly
#: inside the veto's acceptance region while staying far inside any corridor the demo
#: floor offers (the measured scene needs 0.85 m; the corridor is metres wide).
EXECUTION_MARGIN_M = 0.02

PHASES = ("turn", "drive")


@dataclass(frozen=True)
class SupervisorCommand:
    """One tick of supervisor output: a pure turn or a pure drive, and why.

    ``vy`` is always exactly ``0.0`` and ``vx`` and ``wz`` are never both nonzero — the
    two invariants that make the command executable by a Lite3 axis profile whose
    lateral and reverse primitives are ``null``.
    ``test_the_detour_never_requests_an_unmeasured_motion`` pins them, because a future
    edit that lets one slip ships a command the robot silently cannot perform.
    """

    vx: float
    vy: float
    wz: float
    phase: str                 # one of PHASES
    waypoint: tuple[float, float]
    side: str                  # "left" | "right" of the robot→goal line
    blocker_id: str | None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"supervisor phase must be one of {PHASES}")
        if self.vy != 0.0:
            raise ValueError("turn-drive commands have no lateral component")
        if self.vx != 0.0 and self.wz != 0.0:
            raise ValueError("turn-drive commands never combine drive with yaw")


def _point_to_segment_m(px: float, py: float,
                        ax: float, ay: float, bx: float, by: float) -> float | None:
    """Distance from P to the segment AB, or ``None`` when P is off the ends.

    ``None`` is the answer that matters for blockage: an obstacle whose nearest point
    on the LINE is behind the robot or beyond the goal is not standing on the way
    there, and projecting it onto the segment anyway would detour around something the
    straight path never goes near.
    """
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return None
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    if not 0.0 <= t <= 1.0:
        return None
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class TurnDriveSupervisor:
    """Decide whether the straight line is blocked, and if so, the executable way around.

    Args:
        robot_radius_m: the planner's own ``robot_radius_m`` — the supervisor plans with
            the same disc the veto enforces, so a detour it approves cannot be one the
            veto refuses on geometry alone.
        clearance_m: free space kept between the two discs, on top of both radii. The
            default is the stack's static hard gap plus :data:`EXECUTION_MARGIN_M` —
            deliberately NOT the bare hard gap, because the tangent legs sit at exactly
            the planned clearance and the veto refuses anything below the hard gap, so
            a zero-margin plan hands the accept/refuse decision to float rounding.
        drive_speed_m_s: the forward speed a "drive" phase commands. On a sign-only
            axis transport this number never reaches the wire — any forward command past
            the deadband executes at the primitive's measured speed — so pass the
            measured primitive speed (or the envelope ceiling above it), not a wish.
        turn_rate_rad_s: the same, for yaw.
    """

    def __init__(self, *, robot_radius_m: float,
                 clearance_m: float = STATIC_HARD_GAP_M + EXECUTION_MARGIN_M,
                 drive_speed_m_s: float, turn_rate_rad_s: float,
                 heading_tolerance_rad: float = HEADING_TOLERANCE_RAD,
                 waypoint_arrival_m: float = WAYPOINT_ARRIVAL_M) -> None:
        for name, value in (("robot_radius_m", robot_radius_m),
                            ("clearance_m", clearance_m),
                            ("drive_speed_m_s", drive_speed_m_s),
                            ("turn_rate_rad_s", turn_rate_rad_s),
                            ("heading_tolerance_rad", heading_tolerance_rad),
                            ("waypoint_arrival_m", waypoint_arrival_m)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self._robot_radius_m = robot_radius_m
        self._clearance_m = clearance_m
        self._drive_speed_m_s = drive_speed_m_s
        self._turn_rate_rad_s = turn_rate_rad_s
        self._heading_tolerance_rad = heading_tolerance_rad
        self._waypoint_arrival_m = waypoint_arrival_m

    def _required_gap_m(self, obstacle) -> float:
        """Centre-line clearance the straight path must keep from this obstacle."""
        return obstacle.radius_m + self._robot_radius_m + self._clearance_m

    def blocker(self, pose: tuple[float, float, float], goal: tuple[float, float],
                obstacles) -> object | None:
        """The static obstacle standing on the straight robot→goal line, or ``None``.

        Static only: a tracked mover is the shared veto's problem, and routing around
        one plans a detour whose target has walked away by the time it is executed.
        Nearest along the line wins when several block, because the far one is re-planned
        against once the near one is behind.
        """
        x0, y0, _ = pose
        blocked = []
        for obstacle in obstacles:
            if getattr(obstacle, "kind", "tracked") != "static":
                continue
            distance = _point_to_segment_m(obstacle.x, obstacle.y, x0, y0,
                                           goal[0], goal[1])
            if distance is None \
                    or distance >= self._required_gap_m(obstacle) - _CLEARANCE_EPS_M:
                continue
            along = math.hypot(obstacle.x - x0, obstacle.y - y0)
            blocked.append((along, obstacle))
        if not blocked:
            return None
        return min(blocked, key=lambda item: item[0])[1]

    @staticmethod
    def _tangent_angle(px: float, py: float, cx: float, cy: float, radius: float,
                       side: float) -> float | None:
        """Direction angle of the tangent line from P to the circle, on one side.

        ``side`` is +1 for the tangent reached turning counter-clockwise from the
        P→centre bearing, -1 for the other. ``None`` when P is inside the circle —
        from there every direction ends inside it, and no detour this module can
        command starts with the robot already inside the required gap.
        """
        dx, dy = cx - px, cy - py
        distance = math.hypot(dx, dy)
        if distance <= radius:
            return None
        return math.atan2(dy, dx) + side * math.asin(radius / distance)

    def _corner(self, pose, goal, obstacle, side: float):
        """Intersection of the two tangent lines on one side, or ``None``.

        The robot's tangent line at ``pose`` and the goal's tangent line at ``goal`` —
        each touching the blocker's required-clearance circle at exactly the gap —
        meet at the corner of the detour. The goal's line is unoriented: only its
        angle matters for the intersection, and a corner on its wrong ray is rejected
        by the leg clearance check in :meth:`_waypoint`, which asks the same question
        of both legs either way.
        """
        gap = self._required_gap_m(obstacle)
        theta_r = self._tangent_angle(pose[0], pose[1], obstacle.x, obstacle.y, gap,
                                      side)
        theta_g = self._tangent_angle(goal[0], goal[1], obstacle.x, obstacle.y, gap,
                                      -side)
        if theta_r is None or theta_g is None:
            return None
        urx, ury = math.cos(theta_r), math.sin(theta_r)
        ugx, ugy = math.cos(theta_g), math.sin(theta_g)
        determinant = urx * ugy - ury * ugx
        if abs(determinant) < 1e-12:
            return None
        dx, dy = goal[0] - pose[0], goal[1] - pose[1]
        t = (dx * ugy - dy * ugx) / determinant
        if t <= 0.0:
            # The corner is on the robot tangent's BACKWARD extension — behind the
            # robot is unobserved space, and no leg this module approves goes there.
            return None
        return (pose[0] + t * urx, pose[1] + t * ury)

    def _leg_clear(self, a, b, statics) -> bool:
        """Whether segment AB keeps the required gap from every static obstacle.

        The endpoint B itself must clear too: it is the endpoint of a leg, and
        point-to-segment distance reports ``None`` for an obstacle sitting just past
        it — a flanker 0.57 m from the endpoint is off the end of the segment and
        still squarely on the path.
        """
        for other in statics:
            if math.hypot(other.x - b[0], other.y - b[1]) \
                    < self._required_gap_m(other) - _CLEARANCE_EPS_M:
                return False
            distance = _point_to_segment_m(other.x, other.y, a[0], a[1], b[0], b[1])
            if distance is not None and distance < (self._required_gap_m(other)
                                                    - _CLEARANCE_EPS_M):
                return False
        return True

    def _waypoint(self, pose, goal, obstacle, obstacles):
        """``(waypoint, side)`` beside the blocker with both legs clear, or ``None``.

        Both legs — robot→corner and corner→goal — are checked against every static
        obstacle with the same clearance, so the detour does not solve one blocker by
        clipping another. The shorter valid side wins; ``None`` means no two-segment
        detour exists here and the caller must fall back to the planner's own (holding)
        answer rather than to an invented one.
        """
        statics = [o for o in obstacles
                   if getattr(o, "kind", "tracked") == "static"]

        def clear(a, b) -> bool:
            return self._leg_clear(a, b, statics)

        candidates = []
        for side, sign in (("left", 1.0), ("right", -1.0)):
            corner = self._corner(pose, goal, obstacle, sign)
            if corner is None:
                continue
            if clear((pose[0], pose[1]), corner) and clear(corner, goal):
                detour = (math.hypot(corner[0] - pose[0], corner[1] - pose[1])
                          + math.hypot(goal[0] - corner[0], goal[1] - corner[1]))
                candidates.append((detour, corner, side))
        if not candidates:
            return None
        _detour, corner, side = min(candidates, key=lambda item: item[0])
        return corner, side

    def command(self, pose: tuple[float, float, float], goal: tuple[float, float],
                obstacles) -> SupervisorCommand | None:
        """This tick's executable command, or ``None`` to leave the policy in charge.

        Recomputed from scratch every call — see the module docstring. The target of the
        current leg is the waypoint until the robot is BOTH within the arrival disc of
        it AND clear to run at the goal in a straight line (see below — the disc alone
        reaches into the blocker's clearance circle), then the goal; the phase is
        "turn" until the nose is within tolerance of that target's bearing, then
        "drive".
        """
        obstacle = self.blocker(pose, goal, obstacles)
        if obstacle is None:
            return None
        placed = self._waypoint(pose, goal, obstacle, obstacles)
        if placed is None:
            return None
        waypoint, side = placed

        x, y, yaw = pose
        # SWITCHING LEGS IS A CLEARANCE QUESTION, NOT A DISTANCE QUESTION. The arrival
        # disc extends up to ``waypoint_arrival_m`` BEFORE the corner, and the straight
        # line from a pre-corner point to the goal cuts THROUGH the blocker's clearance
        # circle — measured on the demo scene: 0.12 m short of the corner, the
        # goal-ward drive passes the bin with 0.102 m of free space against a 0.12 m
        # hard gap, so the shared veto refuses it and a distance-switched supervisor
        # deadlocks against its own safety layer (drive commanded, veto holds, repeat).
        # The goal becomes the target only when the line from HERE to it keeps the
        # required gap from every static obstacle — geometrically, once the robot has
        # reached the goal-side tangent, i.e. the corner itself.
        near = math.hypot(waypoint[0] - x, waypoint[1] - y) <= self._waypoint_arrival_m
        if near and self._leg_clear((x, y), goal, [o for o in obstacles
                                                   if getattr(o, "kind", "tracked")
                                                   == "static"]):
            target = goal
        else:
            target = waypoint
        error = wrap_pi(math.atan2(target[1] - y, target[0] - x) - yaw)
        blocker_id = getattr(obstacle, "object_id", None)
        # DRIVE ONLY FROM THE OUTWARD HALF OF THE TOLERANCE BAND. The bearing to the
        # waypoint IS the tangent angle along the first leg, so a nose pointed even one
        # degree short of it (toward the blocker) is a straight line that enters the
        # required gap — and the shared veto, which rolls the command forward 2.5 s,
        # refuses it. A symmetric band would therefore deadlock: supervisor drives,
        # veto holds, repeat. The outward half cannot deadlock: driving away from the
        # blocker keeps the whole rollout outside the tangent, and turning back toward
        # the leg is always available. ``sign`` is +1 for a left detour, whose blocker
        # sits to the RIGHT of the leg, so "outward" is error <= 0 (nose at or left of
        # the leg bearing); mirrored for a right detour.
        outward = error * (1.0 if side == "left" else -1.0) <= 0.0
        if abs(error) > self._heading_tolerance_rad or not outward:
            return SupervisorCommand(
                0.0, 0.0, math.copysign(self._turn_rate_rad_s, error),
                phase="turn", waypoint=waypoint, side=side, blocker_id=blocker_id)
        return SupervisorCommand(
            self._drive_speed_m_s, 0.0, 0.0,
            phase="drive", waypoint=waypoint, side=side, blocker_id=blocker_id)
