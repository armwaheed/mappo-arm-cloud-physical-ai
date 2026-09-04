#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Say something with the legs: a spin on arrival, a shake on surrender.

    python3 flourish.py --robot-id LITE3-A --firmware V1.0.8 --payload none \\
        --locomotion-transport axis --axis-profile lite3-axis-LITE3-A.json \\
        --kind spin --lane-width-metres 2.0
    # prints the plan and exits. Add --live --operator-ready to actually turn.

WHY THIS IS TWO TURNS AND NOT A DANCE.

The obvious ask is to record the vendor's canned motion off the hand controller and replay
it. That cannot be done through this stack, and the reason is not effort. The axis
transport is SIGN-ONLY: a profile holds one evidenced raw value per direction and every
command past the deadband emits it at full scale. Three directions carry a value on the
Ventures measured so far --

    forward_positive  +32767  ->  0.5362 m/s
    yaw_positive      +16000  ->  0.8566 rad/s
    yaw_negative      -16000  ->  0.8563 rad/s

-- and `forward_negative`, `lateral_positive` and `lateral_negative` are all null. So a
recorded dance would replay as on/off, at one speed, in three directions: no sideways, no
backwards, no slow. And if the canned motion is a firmware gesture the remote triggers, it
is not a velocity stream at all and this stack never sees it.

What IS available is a turn in place, at a measured rate, in either direction. That reads
clearly on camera, it uses nothing that has not been measured on this robot, and it cannot
travel: both gestures here keep the robot's centre where it is, which is the property that
makes them safe to fire automatically at the end of a run nobody is steering any more.

⚠️ THE ROBOT STILL SWEEPS ITS OWN FOOTPRINT. Turning in place is not motionless -- the
corners of a 0.90 m diagonal move through a 0.90 m circle, and this platform has no
lateral sensing to notice a chair leg while they do. The lane width is checked before the
first command and is the operator's measurement, not a default.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reverse_along_path import (
    COMMAND_WZ,
    PHASE_TIMEOUT_MARGIN,
    PLATFORM_HALF_DIAGONAL_M,
    STALL_PROGRESS_RAD,
    STALL_WINDOW_S,
    Refusal,
    wrap_pi,
)

#: Turns the victory spin makes. One, deliberately: two is not twice as legible on camera,
#: it is the same gesture taking twice as long while an operator waits to reset the scene.
SPIN_REVOLUTIONS = 1.0

#: The fault gesture: a small alternating rock, not a spin. It has to be distinguishable
#: from the success one at a glance and from behind, which rules out "a spin, but shorter".
SHAKE_SWEEP_RAD = 0.45
SHAKE_COUNT = 3

#: Radians of slop each leg accepts. One 10 Hz tick at the measured 0.8565 rad/s carries
#: 0.086 rad, so a tighter tolerance would ask the robot to stop between two commands it
#: has no way to issue.
LEG_TOLERANCE_RAD = 0.12


class Flourish:
    """A gesture as a state machine over measured heading. Commands yaw and nothing else.

    Pure: handed a heading and a clock, it returns a yaw sign or ``None`` when finished, so
    the whole gesture is testable with no robot. Heading is ACCUMULATED from wrapped
    tick-to-tick deltas rather than compared against the start, for the reason a half turn
    already taught this code: a full revolution passes through the +/-pi discontinuity,
    where a comparison against the start reads as no rotation at all and the robot spins
    until its timeout.
    """

    SPIN = "spin"
    SHAKE = "shake"

    def __init__(self, kind: str, *, yaw_speed_rad_s: float, turn_sign: int = +1,
                 revolutions: float = SPIN_REVOLUTIONS, shakes: int = SHAKE_COUNT) -> None:
        if kind not in (self.SPIN, self.SHAKE):
            raise Refusal(f"unknown gesture {kind!r}; it is {self.SPIN} or {self.SHAKE}")
        if yaw_speed_rad_s <= 0.0:
            raise Refusal(
                "this gesture is timed against the profile's MEASURED yaw speed and it is "
                "missing. Run axis_primitive_probe.py first; an unmeasured speed is an "
                "absence, not a zero.")
        self.kind = kind
        self.yaw_speed_rad_s = yaw_speed_rad_s
        self.turn_sign = +1 if turn_sign >= 0 else -1
        #: Each leg is (radians to turn, sign). A spin is one long leg; a shake alternates.
        if kind == self.SPIN:
            self._legs = [(2.0 * math.pi * revolutions, self.turn_sign)]
        else:
            self._legs = []
            for index in range(shakes * 2):
                sign = self.turn_sign if index % 2 == 0 else -self.turn_sign
                # The first and last half-sweeps are half width, so the gesture starts and
                # ends on the heading it began on rather than drifting a sweep to one side.
                span = SHAKE_SWEEP_RAD * (0.5 if index in (0, shakes * 2 - 1) else 1.0)
                self._legs.append((span, sign))
        self._leg = 0
        self._turned = 0.0
        self._last_yaw: float | None = None
        self._leg_started_at: float | None = None
        self._progress_at = 0.0
        self._progress_mark = 0.0

    @property
    def done(self) -> bool:
        return self._leg >= len(self._legs)

    @property
    def turned_rad(self) -> float:
        return self._turned

    def step(self, yaw: float, now: float) -> float | None:
        """One control tick. Returns a yaw command, or ``None`` when the gesture is over."""
        if self.done:
            return None
        span, sign = self._legs[self._leg]
        if self._leg_started_at is None:
            self._leg_started_at = now
            self._progress_at = now
            self._progress_mark = 0.0

        if self._last_yaw is not None:
            self._turned += abs(wrap_pi(yaw - self._last_yaw))
        self._last_yaw = yaw

        if self._turned - self._progress_mark >= STALL_PROGRESS_RAD:
            self._progress_mark = self._turned
            self._progress_at = now
        elif now - self._progress_at > STALL_WINDOW_S:
            raise Refusal(
                f"the {self.kind} turned {self._turned:.3f} rad and then stopped moving. "
                f"Stopping rather than commanding a robot that is stuck, held by its own "
                f"gait gate, or reporting flat odometry while being dragged.")
        budget = (span / self.yaw_speed_rad_s) * PHASE_TIMEOUT_MARGIN
        if now - self._leg_started_at > budget:
            raise Refusal(
                f"the {self.kind} outran its budget of {budget:.1f}s having turned "
                f"{self._turned:.3f} of {span:.3f} rad. The measured yaw speed says it "
                f"should be done; something is delivering less than the profile claims.")

        if self._turned >= span - LEG_TOLERANCE_RAD:
            self._leg += 1
            self._turned = 0.0
            self._leg_started_at = None
            self._last_yaw = yaw
            return self.step(yaw, now)
        return COMMAND_WZ * sign


def describe(kind: str, yaw_speed_rad_s: float) -> str:
    """The plan, in the shape the commissioning probes print theirs."""
    if kind == Flourish.SPIN:
        span, legs = 2.0 * math.pi * SPIN_REVOLUTIONS, 1
        what = f"one full turn in place, {math.degrees(span):.0f} deg"
    else:
        span = SHAKE_SWEEP_RAD * (SHAKE_COUNT * 2 - 1)
        legs = SHAKE_COUNT * 2
        what = (f"{SHAKE_COUNT} alternating rocks of "
                f"{math.degrees(SHAKE_SWEEP_RAD):.0f} deg, ending on the start heading")
    return "\n".join([
        f"  gesture       {kind}",
        f"  {what}",
        f"  legs          {legs}",
        f"  yaw speed     {yaw_speed_rad_s:.4f} rad/s (measured)",
        f"  duration      ~{span / yaw_speed_rad_s:.2f} s",
        f"  travel        none -- the centre does not move; it sweeps "
        f"{2 * PLATFORM_HALF_DIAGONAL_M:.2f} m turning",
        "  every primitive used carries an evidence string; nothing here is a guess",
    ])


def build_parser() -> argparse.ArgumentParser:
    """The commissioning front-end, reused rather than re-typed. See reverse_along_path."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from deep_robotics.lite3.commissioning import robot_link

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    robot_link.add_context_arguments(parser)
    robot_link.add_link_arguments(parser, moving=True)
    parser.add_argument("--kind", choices=(Flourish.SPIN, Flourish.SHAKE),
                        default=Flourish.SPIN)
    parser.add_argument("--turn-sign", type=int, default=1, choices=(1, -1),
                        help="which way to turn. A CLEARANCE choice: both yaw primitives "
                             "are measured within 0.03%% of each other")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--lane-width-metres", type=float, default=None,
                        help="clear width, BOTH SIDES. This robot TURNS IN PLACE here and "
                             "has no lateral sensing")
    return parser


def check_room(args) -> None:
    if args.lane_width_metres is None:
        raise Refusal(
            "state --lane-width-metres. This gesture turns the robot through a full "
            "circle and it has no lateral sensing, so the room is the operator's "
            "measurement and not a default.")
    if args.lane_width_metres < 2 * PLATFORM_HALF_DIAGONAL_M:
        raise Refusal(
            f"lane is {args.lane_width_metres:.2f} m wide and a turn in place sweeps "
            f"{2 * PLATFORM_HALF_DIAGONAL_M:.2f} m.")


def perform(args, yaw_speed_rad_s: float) -> int:
    """Command the gesture. ⛔ ``robot-stack/SAFETY.md`` governs this."""
    import contextlib

    from deep_robotics.lite3.commissioning import robot_link

    link = robot_link.connect(args)
    loco = link.locomotion
    robot_link.preflight(link, args)

    gesture = Flourish(args.kind, yaw_speed_rad_s=yaw_speed_rad_s,
                       turn_sign=args.turn_sign)
    tick_s = 1.0 / args.control_hz
    try:
        while True:
            pose = loco.pose()
            wz = gesture.step(pose.yaw_rad, time.monotonic())
            if wz is None:
                break
            # Re-sent every tick: the vendor high-level interface is edge-triggered, so a
            # single send is indistinguishable from a dropped datagram for the rest of it.
            loco.set_velocity(0.0, 0.0, wz)
            time.sleep(tick_s)
    finally:
        # Every exit stops the legs -- the aborts inside `step`, a Ctrl-C, and the normal
        # finish. `suppress` so a stop that itself fails cannot mask why the run ended.
        with contextlib.suppress(Exception):
            loco.stop()
    print(f"[flourish] {args.kind} done, robot stopped")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from deep_robotics.lite3.locomotion.lite3_axis_locomotion import AxisProfile

    if args.axis_profile is None:
        print("[flourish] --axis-profile is required: the measured yaw speed this gesture "
              "is timed against lives in it", file=sys.stderr)
        return 2
    speeds = AxisProfile.load(Path(args.axis_profile)).measured_speeds
    yaw = min(speeds.get("yaw_positive", 0.0) or 0.0,
              speeds.get("yaw_negative", 0.0) or 0.0)
    if not yaw:
        print("[flourish] REFUSED: this profile has no measured yaw speed. Run "
              "axis_primitive_probe.py first.", file=sys.stderr)
        return 1

    print(f"[flourish] {args.robot_id}:")
    print(describe(args.kind, yaw))
    if not args.live:
        print("\n[flourish] plan only. Add --live --operator-ready to turn it.")
        return 0
    if not args.operator_ready:
        print("[flourish] REFUSED: --live needs --operator-ready.", file=sys.stderr)
        return 1
    try:
        check_room(args)
        return perform(args, yaw)
    except Refusal as refusal:
        print(f"[flourish] REFUSED: {refusal}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
