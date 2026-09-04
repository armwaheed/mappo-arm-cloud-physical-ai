#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Retreat 0.66 m back down the path already walked, without a reverse primitive.

    python3 reverse_along_path.py --robot-id LITE3-A --firmware V1.0.8 --payload none \\
        --locomotion-transport axis --axis-profile lite3-axis-LITE3-A.json \\
        --lane-metres 3.0 --lane-width-metres 2.0
    # prints the plan and exits. Add --live --operator-ready to actually walk.

WHY THIS TURNS AROUND INSTEAD OF REVERSING.

The obvious way to back off is a negative forward command, and on this transport that is
exactly what is not available: ``lite3-axis-profile/v1`` carries one evidenced raw value
per direction and ``forward_negative`` is ``null`` on both Ventures, so ``map_velocity``
raises ``AxisProfileError`` and the legs never receive anything. Issue #195 records the
consequence -- "a planner decision to back off will silently stall".

Filling that field in would mean inventing a raw value. The joystick reading, that full
stick back mirrors full stick forward, is plausible and is not evidence; the commissioning
probe exists precisely because a primitive can be dead, or can move the robot the way its
name does not claim, and ``axis_primitive_probe.py``'s own docstring records that the
LATERAL pair inverts between the navigator's frame and the vendor's. A guess there is a
guess about which way the robot lunges.

So this does not guess. Everything it commands is already measured on this robot:

    forward_positive  +32767  ->  0.5362 m/s    400 samples, error_state 0
    yaw_positive      +16000  ->  0.8566 rad/s  -45.87 deg heading change
    yaw_negative      -16000  ->  0.8563 rad/s  +46.05 deg heading change

Turn 180 degrees, walk forward, turn back. The robot ends up 0.66 m back down its own
path having used three primitives that each carry an evidence string.

⚠️ AND IT IS NOT MERELY EQUIVALENT -- IT IS BETTER SENSED. ``walk_back`` is open-loop into
space nothing is looking at: neither Venture has a rear camera and the planner never
samples that direction. After the first turn the camera faces the way the robot is
travelling, so the retreat is observed rather than blind. That is the real argument for
this shape, and it survives even if somebody later measures ``forward_negative``.

WHAT CLOSES THE LOOP, AND WHY IT HAS TO BE ODOMETRY.

The transport is SIGN-ONLY: every command past the profile's deadband emits the same raw
value at full scale, so a commanded speed is a direction and never a speed. There is no
"drive at 0.2 m/s for 3.3 s" available. The only way to travel a DISTANCE is to watch the
distance go by and stop, which is what :class:`ReverseAlongPath` does -- it decides a sign
each tick from the measured pose and returns ``None`` when the leg is done. A measured
speed that turns out to be 10% off changes how long a phase takes and not how far the
robot goes.

"BACK ALONG A PATH IT ALREADY TRAVERSED" IS A CAP, NOT A DECORATION. :class:`PathHistory`
keeps the breadcrumbs and the requested distance is clipped to the length actually walked.
A robot that has moved 0.2 m since it stood up retreats 0.2 m, not 0.66 m into a room it
has never been in and nothing has looked at.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Metres. The retreat the demo asks for. Not tuned -- it is the operator's number, chosen
#: to be shorter than the 0.69 m at which a cone centre makes straight-line travel
#: infeasible (issue #195), so a robot that backs off by this much has undone the approach
#: that produced the hold without reversing past the obstacle that caused it.
DEFAULT_RETREAT_M = 0.66

#: Radians. A half turn. Both yaw primitives are measured within 0.03% of each other
#: (0.8566 vs 0.8563 rad/s), so which way it turns is a clearance question, not a speed one.
HALF_TURN_RAD = math.pi

#: Radians of heading error the turn phases accept. At the measured 0.8565 rad/s one 10 Hz
#: tick carries 0.086 rad, so anything below that would be asking the robot to stop between
#: two commands it cannot issue. This is one tick plus a margin.
TURN_TOLERANCE_RAD = 0.12

#: Metres of distance error the drive phase accepts, for the same reason: one tick at the
#: measured 0.5362 m/s carries 0.054 m.
DRIVE_TOLERANCE_M = 0.06

#: Command magnitudes. ARBITRARY, and that is the point -- past the profile's deadband
#: (0.05 m/s linear, 0.1 rad/s yaw) every value produces the byte-identical datagram on a
#: sign-only transport. Stated well clear of the deadband so a profile with a slightly
#: larger one still commands, and never near a real speed, so nobody reads these as one.
COMMAND_VX = 0.30
COMMAND_WZ = 0.50

#: How much longer than the measured-speed prediction a phase may take before it is called
#: a stall. 2.5x covers the turn-in-place slip a legged robot has against flat odometry
#: without letting a robot that is not moving at all run to the operator's patience.
PHASE_TIMEOUT_MARGIN = 2.5

#: Metres, and radians, of progress a phase must make within :data:`STALL_WINDOW_S` or be
#: abandoned. Sized under one tick of the measured speeds so normal motion never trips it.
STALL_PROGRESS_M = 0.02
STALL_PROGRESS_RAD = 0.04
STALL_WINDOW_S = 1.5


def wrap_pi(angle: float) -> float:
    """``angle`` folded into (-pi, pi]. Local rather than imported from ``visual_nav``:
    this module is stdlib-only so it can run from the commissioning directory on a robot
    where the navigator's package is not on the path."""
    return math.atan2(math.sin(angle), math.cos(angle))


class Refusal(RuntimeError):
    """A precondition the operator can fix, told before anything moves."""


@dataclass
class PathHistory:
    """The breadcrumbs the retreat is allowed to walk back along.

    Length of the polyline, not displacement from the start: a robot that walked a metre
    out and half a metre back has traversed 1.5 m of floor, and every centimetre of it has
    been under the camera. Displacement would say 0.5 m and under-serve a legitimate
    retreat; it is the FLOOR THAT WAS LOOKED AT that licenses backing over it.
    """

    #: Metres a pose must move before it counts. This is a NOISE GATE, not a sampling
    #: interval, and that is why the distance below it is discarded rather than banked: a
    #: robot standing still on jittering odometry would otherwise accumulate metres of
    #: "traversed floor" it never crossed, and every one of those centimetres is licence to
    #: back into a room nothing has looked at. Erring toward a shorter retreat is the safe
    #: direction. 5 cm is finer than DRIVE_TOLERANCE_M, so the cap is never the coarse part.
    spacing_m: float = 0.05
    points: list = field(default_factory=list)
    traversed_m: float = 0.0

    def append(self, x: float, y: float) -> None:
        if not self.points:
            self.points.append((x, y))
            return
        last_x, last_y = self.points[-1]
        step = math.hypot(x - last_x, y - last_y)
        # The epsilon is not decoration. A robot walking in exact 5 cm increments produces
        # steps that are 0.049999999999999996 in binary, and a bare `<` then discards every
        # other breadcrumb -- measured 1.35 m against a true 1.45 m before this was here.
        # The gate is meant to reject NOISE, and floating-point representation is not noise.
        if step < self.spacing_m - 1e-9:
            return
        self.traversed_m += step
        self.points.append((x, y))

    def available_m(self, requested_m: float) -> float:
        """How far back this robot has actually earned the right to go."""
        return max(0.0, min(requested_m, self.traversed_m))


@dataclass
class AxisIntent:
    """One tick of intent, in the navigator's body frame. Signs only; see COMMAND_VX.

    Deliberately NOT called ``Command``. ``avoidance.Command`` is the planner's per-tick
    verdict and carries `feasible`, `evaluated`, `floor_reach_m_s` and `transport_refusal`
    -- the record of a search this manoeuvre does not perform. `test_avoidance.py` scans
    the tree for anything constructing a `Command` without those fields and named this
    file when the class was called that; the guard was right, and the answer is a
    different name rather than four fields that would each be a claim this has no
    measurement for.
    """

    vx: float
    vy: float
    wz: float
    phase: str


class ReverseAlongPath:
    """Turn 180, walk ``target_m``, turn back -- as a state machine over measured pose.

    Pure: it is handed a pose and a clock and returns a command or ``None``, so the whole
    manoeuvre is testable with no robot, no sockets and no legs. ``step`` returning ``None``
    means finished; a :class:`Refusal` means abandoned, and the caller must stop the robot.

    Heading is accumulated from wrapped tick-to-tick deltas rather than compared against
    the start. A half turn lands exactly on the +/-pi discontinuity, where a comparison
    flips sign and reads as "no rotation at all" -- the robot would spin until the timeout.
    """

    TURN_AWAY = "turn-away"
    DRIVE = "drive"
    TURN_BACK = "turn-back"

    def __init__(self, target_m: float, *, turn_sign: int = +1, turn_back: bool = True,
                 forward_speed_m_s: float, yaw_speed_rad_s: float) -> None:
        if target_m <= 0.0:
            raise Refusal(
                f"nothing to walk back: the retreat resolved to {target_m:.3f} m. A robot "
                f"that has not traversed any floor has none to retreat over.")
        if forward_speed_m_s <= 0.0 or yaw_speed_rad_s <= 0.0:
            raise Refusal(
                "this manoeuvre is timed against the profile's MEASURED speeds and one of "
                "them is missing. Run axis_primitive_probe.py first; an unmeasured speed "
                "is an absence, not a zero.")
        self.target_m = target_m
        self.turn_sign = +1 if turn_sign >= 0 else -1
        self.turn_back = turn_back
        self.forward_speed_m_s = forward_speed_m_s
        self.yaw_speed_rad_s = yaw_speed_rad_s
        self.phase = self.TURN_AWAY
        self._turned_rad = 0.0
        self._driven_m = 0.0
        self._last_pose: tuple | None = None
        self._phase_started_at: float | None = None
        self._progress_at = 0.0
        self._progress_mark = 0.0

    # ── the phases, and what each is waiting for ────────────────────────────
    def _budget_s(self) -> float:
        if self.phase == self.DRIVE:
            return (self.target_m / self.forward_speed_m_s) * PHASE_TIMEOUT_MARGIN
        return (HALF_TURN_RAD / self.yaw_speed_rad_s) * PHASE_TIMEOUT_MARGIN

    def _advance(self) -> None:
        order = [self.TURN_AWAY, self.DRIVE, self.TURN_BACK]
        nxt = order.index(self.phase) + 1
        self.phase = order[nxt] if nxt < len(order) else None
        self._phase_started_at = None
        self._progress_mark = 0.0

    def step(self, x: float, y: float, yaw: float, now: float) -> AxisIntent | None:
        """One control tick. ``None`` when the manoeuvre is complete."""
        if self.phase is None:
            return None
        if self._phase_started_at is None:
            self._phase_started_at = now
            self._progress_at = now
            self._progress_mark = 0.0

        if self._last_pose is not None:
            last_x, last_y, last_yaw = self._last_pose
            if self.phase == self.DRIVE:
                self._driven_m += math.hypot(x - last_x, y - last_y)
            else:
                self._turned_rad += abs(wrap_pi(yaw - last_yaw))
        self._last_pose = (x, y, yaw)

        progress = self._driven_m if self.phase == self.DRIVE else self._turned_rad
        floor = STALL_PROGRESS_M if self.phase == self.DRIVE else STALL_PROGRESS_RAD
        if progress - self._progress_mark >= floor:
            self._progress_mark = progress
            self._progress_at = now
        elif now - self._progress_at > STALL_WINDOW_S:
            raise Refusal(
                f"{self.phase} made {progress:.3f} of progress in {now - self._progress_at:.1f}s "
                f"and is not moving. Stopping rather than commanding a robot that is stuck, "
                f"held by its own gait gate, or reporting flat odometry while being dragged.")
        if now - self._phase_started_at > self._budget_s():
            raise Refusal(
                f"{self.phase} outran its budget of {self._budget_s():.1f}s at "
                f"{progress:.3f} of its target. The measured speeds this is timed against "
                f"say it should be done; something is delivering less than the profile "
                f"claims, and that is not a thing to keep commanding.")

        if self.phase == self.TURN_AWAY:
            if self._turned_rad >= HALF_TURN_RAD - TURN_TOLERANCE_RAD:
                self._turned_rad = 0.0
                self._advance()
                return self.step(x, y, yaw, now)
            return AxisIntent(0.0, 0.0, COMMAND_WZ * self.turn_sign, self.phase)

        if self.phase == self.DRIVE:
            if self._driven_m >= self.target_m - DRIVE_TOLERANCE_M:
                self._advance()
                if not self.turn_back:
                    self.phase = None
                    return None
                return self.step(x, y, yaw, now)
            return AxisIntent(COMMAND_VX, 0.0, 0.0, self.phase)

        if self._turned_rad >= HALF_TURN_RAD - TURN_TOLERANCE_RAD:
            self._advance()
            return None
        return AxisIntent(0.0, 0.0, COMMAND_WZ * -self.turn_sign, self.phase)

    @property
    def driven_m(self) -> float:
        return self._driven_m


def describe(target_m: float, requested_m: float, traversed_m: float,
             forward_speed_m_s: float, yaw_speed_rad_s: float, turn_back: bool) -> str:
    """The plan, in the shape the commissioning probes print theirs."""
    turns = 2 if turn_back else 1
    turn_s = HALF_TURN_RAD / yaw_speed_rad_s
    drive_s = target_m / forward_speed_m_s
    lines = [
        f"  retreat requested   {requested_m:.3f} m",
        ("  floor traversed     not observed -- this tool was not watching the walk out, "
         "so the\n                      lane check is what licenses this move"
         if math.isinf(traversed_m) else f"  floor traversed     {traversed_m:.3f} m"),
        f"  retreat resolved    {target_m:.3f} m"
        + ("  (CLIPPED to the floor already walked)" if target_m < requested_m - 1e-9 else ""),
        "",
        f"  turn away  180 deg at {yaw_speed_rad_s:.4f} rad/s   {turn_s:5.2f} s",
        f"  drive      {target_m:.3f} m at {forward_speed_m_s:.4f} m/s   {drive_s:5.2f} s",
    ]
    if turn_back:
        lines.append(f"  turn back  180 deg at {yaw_speed_rad_s:.4f} rad/s   {turn_s:5.2f} s")
    lines += [
        "",
        f"  total {turns * turn_s + drive_s:.2f} s of motion, "
        f"{turns} turns in place plus {target_m:.3f} m forward",
        "  every primitive used carries an evidence string; nothing here is a guessed "
        "raw value",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The commissioning front-end, reused rather than re-typed.

    ``robot_link`` already owns the arguments that decide whether a robot may be moved --
    the issue #13 context, the transport selection, ``--live``/``--operator-ready``, the
    battery abort and the unwalked-transport refusal. Declaring a subset of them here by
    hand is how a tool ends up with the authority to move and none of the gates: an
    earlier cut of this file did exactly that and would have called `connect()` with no
    `--battery-abort` and no `--command-port` at all.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from deep_robotics.lite3.commissioning import robot_link

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    robot_link.add_context_arguments(parser)
    robot_link.add_link_arguments(parser, moving=True)

    how = parser.add_argument_group("the retreat")
    how.add_argument("--retreat-metres", type=float, default=DEFAULT_RETREAT_M,
                     help=f"how far back to walk (default: {DEFAULT_RETREAT_M})")
    how.add_argument("--turn-sign", type=int, default=1, choices=(1, -1),
                     help="which way to turn away. A CLEARANCE choice: both yaw "
                          "primitives are measured within 0.03%% of each other")
    how.add_argument("--no-turn-back", action="store_true",
                     help="leave the robot facing the way it retreated")
    how.add_argument("--control-hz", type=float, default=10.0)

    room = parser.add_argument_group("the room")
    room.add_argument("--lane-metres", type=float, default=None,
                      help="clear length ahead of and behind the robot, in metres")
    room.add_argument("--lane-width-metres", type=float, default=None,
                      help="clear width, BOTH SIDES -- this robot TURNS IN PLACE here and "
                           "has no lateral sensing")
    return parser


def check_room(target_m: float, args) -> None:
    """Refuse a lane too small for a robot that TURNS IN PLACE here.

    The width matters more than it does for a straight probe and is checked against the
    same number as the length: during the turn the robot sweeps its own footprint in every
    direction, and it has no lateral sensing to notice a chair leg while it does.
    """
    if args.lane_metres is None or args.lane_width_metres is None:
        raise Refusal(
            "state --lane-metres and --lane-width-metres. This manoeuvre turns the robot "
            "through 360 degrees in total and then walks; it has no rear or lateral "
            "sensing, so the room is the operator's measurement and not a default.")
    # A half-diagonal turning circle plus the retreat, both ends, because the robot walks
    # away from where it started and must still fit when it turns back.
    needed = target_m + 2 * PLATFORM_HALF_DIAGONAL_M
    if args.lane_metres < needed:
        raise Refusal(
            f"lane is {args.lane_metres:.2f} m and this needs {needed:.2f} m: "
            f"{target_m:.2f} m of retreat plus the robot's own turning circle at each end.")
    if args.lane_width_metres < 2 * PLATFORM_HALF_DIAGONAL_M:
        raise Refusal(
            f"lane is {args.lane_width_metres:.2f} m wide and a turn in place sweeps "
            f"{2 * PLATFORM_HALF_DIAGONAL_M:.2f} m.")


#: Metres. Half the diagonal of the Lite3's footprint -- the radius it sweeps turning in
#: place. Same figure the driver publishes as its peer footprint.
PLATFORM_HALF_DIAGONAL_M = 0.45


def walk(args, target_m: float, forward_speed_m_s: float, yaw_speed_rad_s: float) -> int:
    """Command the manoeuvre on a real robot. ⛔ ``robot-stack/SAFETY.md`` governs this.

    The command is re-sent every tick because the vendor high-level interface is
    edge-triggered -- it applies what it last received -- so a single send is
    indistinguishable from a dropped datagram for the rest of the phase. Same reason
    ``measurement.run_segment`` re-sends.

    Pose comes from ``loco.pose()`` at every tick rather than from integrating
    ``loco.velocity()``: an integrated velocity carries its bias into the distance, which
    is the one number this manoeuvre exists to get right.
    """
    import contextlib

    from deep_robotics.lite3.commissioning import robot_link

    link = robot_link.connect(args)
    loco = link.locomotion
    robot_link.preflight(link, args)

    plan = ReverseAlongPath(target_m, turn_sign=args.turn_sign,
                            turn_back=not args.no_turn_back,
                            forward_speed_m_s=forward_speed_m_s,
                            yaw_speed_rad_s=yaw_speed_rad_s)
    tick_s = 1.0 / args.control_hz
    phase_seen = None
    started = time.monotonic()
    try:
        while True:
            # `Lite3Pose.yaw`, not `yaw_rad`. The sibling upstream repository names
            # the same field `yaw_rad`, and writing this against that one cost a
            # live AttributeError on a robot that had already finished its run.
            pose = loco.pose()
            intent = plan.step(pose.x, pose.y, pose.yaw, time.monotonic())
            if intent is None:
                break
            if intent.phase != phase_seen:
                phase_seen = intent.phase
                print(f"[reverse] {intent.phase}")
            loco.set_velocity(intent.vx, intent.vy, intent.wz)
            time.sleep(tick_s)
    finally:
        # Every exit stops the legs: the aborts inside `step`, a Ctrl-C, and the normal
        # finish. A manoeuvre that raises while the legs still hold their last command is
        # the one failure mode a retreat must not have, and `suppress` is here so that a
        # stop which itself fails cannot mask the reason the run ended.
        with contextlib.suppress(Exception):
            loco.stop()
    print(f"[reverse] done: {plan.driven_m:.3f} m walked back in "
          f"{time.monotonic() - started:.1f}s, robot stopped")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    requested = args.retreat_metres

    # `lite3_axis_locomotion` imports its UDP sibling by package path, so the directory
    # alone is not enough -- `robot-stack` itself has to be importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from deep_robotics.lite3.locomotion.lite3_axis_locomotion import AxisProfile

    if args.axis_profile is None:
        print("[reverse] --axis-profile is required: the measured speeds this manoeuvre "
              "is timed against live in it", file=sys.stderr)
        return 2
    profile = AxisProfile.load(Path(args.axis_profile))
    speeds = profile.measured_speeds
    forward = speeds.get("forward_positive", 0.0)
    yaw = min(speeds.get("yaw_positive", 0.0) or 0.0,
              speeds.get("yaw_negative", 0.0) or 0.0)

    print(f"[reverse] {args.robot_id}: retreat {requested:.3f} m by turning, not reversing")
    if not forward or not yaw:
        print("[reverse] REFUSED: this profile has no measured forward and yaw speed. "
              "Run axis_primitive_probe.py first.", file=sys.stderr)
        return 1

    # Without a live state stream there are no breadcrumbs, so the plan is printed against
    # the request and the clip is stated as the thing that will happen on the robot.
    print(describe(requested, requested, float("inf"), forward, yaw, not args.no_turn_back))

    if not args.live:
        print("\n[reverse] plan only. Add --live --operator-ready to walk it.")
        return 0
    if not args.operator_ready:
        print("[reverse] REFUSED: --live needs --operator-ready.", file=sys.stderr)
        return 1
    try:
        check_room(requested, args)
    except Refusal as refusal:
        print(f"[reverse] REFUSED: {refusal}", file=sys.stderr)
        return 1

    try:
        return walk(args, requested, forward, yaw)
    except Refusal as refusal:
        print(f"[reverse] REFUSED: {refusal}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
