#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Say something with the legs: a spin on arrival, a shake on surrender, and -- by hand,
never automatically -- one of the vendor's two canned acrobatic actions.

    python3 flourish.py --robot-id LITE3-A --firmware V1.0.8 --payload none \\
        --locomotion-transport axis --axis-profile lite3-axis-LITE3-A.json \\
        --kind spin --lane-width-metres 2.0
    # prints the plan and exits. Add --live --operator-ready to actually turn.

    python3 flourish.py --robot-id LITE3-A --firmware V1.0.8 --payload none \\
        --locomotion-transport axis --axis-profile lite3-axis-LITE3-A.json \\
        --kind backflip --operator-triggered --lane-width-metres 2.0 \\
        --rear-clearance-metres 2.5 --acrobatic-battery-floor-pct 60 \\
        --action-hold-seconds 4
    # every one of those extra flags is a precondition NOBODY HERE HAS MEASURED. Read the
    # brief it prints before you add --live --operator-ready.

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
travel: the four TURN kinds -- spin, shake, sweep, look -- keep the robot's centre where it
is, which is the property that makes them safe to fire automatically at the end of a run
nobody is steering any more.

⚠️ THE ROBOT STILL SWEEPS ITS OWN FOOTPRINT. Turning in place is not motionless -- the
corners of a 0.90 m diagonal move through a 0.90 m circle, and this platform has no
lateral sensing to notice a chair leg while they do. The lane width is checked before the
first command and is the operator's measurement, not a default.

⛔ AND THEN THERE ARE THE TWO THAT DO NOT KEEP THE CENTRE WHERE IT IS.

`--kind backflip` and `--kind twist-jump` are not gestures this file composes out of
measured primitives. Each is a SINGLE VENDOR OPCODE on the same command channel -- one
button on the vendor's reference GUI -- and once it is sent this stack cannot shape, slow,
shorten, steer or interrupt whatever the firmware then does. Which means the paragraph
above stops being true, and it stops being true exactly where it was load-bearing:

    spin, shake, sweep, look   in place. The centre does not move. Sweeps 0.90 m turning.
    backflip                   TRAVELS ~1.5 m BACKWARD. Vendor figure. Never measured here.
    twist-jump                 travel UNKNOWN. Never sent from this repository, by anyone.

⛔ THE TRAVELLING KINDS ARE NOT SAFE TO FIRE UNATTENDED. The property that licensed firing
a gesture at the end of a run nobody is steering -- that the robot's centre stays put -- is
the one property these two do not have. They still need `--operator-triggered` on top of
everything the turns require, and `_validate_action` still refuses without it.

⚠️ "NO AUTOMATIC END-OF-RUN PATH PASSES IT" WAS TRUE UNTIL 2026-09-07 AND IS NO LONGER.
`mission.py` now passes it on ONE path: an arrival, when the operator ticked the dashboard's
flip checkbox for that run. The box ships unticked, is never auto-ticked, is disabled unless
arm motion is on, and arrives as `MAPPO_FLIP=1` for a single run -- so what licenses the
flag is a per-run human act rather than configuration. That is weaker than what this
paragraph used to promise and should be read as such: the flag now means "a human ticked a
box before this run started", not "a human is watching this robot and a look-behind agreed".
The operator asked for it twice on 2026-09-07 having been shown what it removes.
`test_flourish.py` still pins that no arrival path NAMES these kinds, that
`mission.flourish_command` never grows this flag on its own, and -- newly -- that the one
call site which does pass it also carries the rear clearance that is now the only check on
the space behind.

⛔ ZERO REAR SENSING. Not poor rear sensing: none. There is no rear camera, no rear
ultrasonic and no bumper, and the one camera on the platform looks FORWARD through 134
degrees. A backflip travels 1.5 m into the single direction this robot is completely blind
in, so the floor behind it is `--rear-clearance-metres`, measured by the operator, with no
default -- the same rule `--lane-width-metres` already applies to the sideways sweep, for
the same reason.

⛔ EVERYTHING ELSE ABOUT THESE TWO OPCODES IS UNVALIDATED. Neither has ever been sent by
this repository. Unknown, and printed in full before either one can fire: the minimum
battery, whether the robot must be standing or in some particular vendor mode, how long the
action runs, whether it self-terminates or needs the separate 结束动作 / END ACTION opcode,
and what state it leaves the robot in afterwards. Where a precondition is unknown this file
refuses until the operator states it -- `--acrobatic-battery-floor-pct` and
`--action-hold-seconds` ship no defaults for exactly that reason -- or says so, loudly, in
the brief. It does not pick a number and call it a precondition.
"""
from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
import time
from dataclasses import dataclass
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

#: The recovery scan: 90 degrees each side of the heading the robot got stuck on, then
#: back to it. Three legs rather than two so it ENDS where it started -- the next attempt
#: re-plans from the heading the operator aimed it at, not from wherever the scan happened
#: to stop. What it buys is a different view: the camera is the only sensor, it sees 134
#: degrees, and a robot that has held for seconds against something it cannot get around
#: has been looking at the same frame the whole time.
SWEEP_RAD = math.pi / 2

#: Radians of slop each leg accepts. One 10 Hz tick at the measured 0.8565 rad/s carries
#: 0.086 rad, so a tighter tolerance would ask the robot to stop between two commands it
#: has no way to issue.
LEG_TOLERANCE_RAD = 0.12

# ── The vendor canned actions ───────────────────────────────────────────────
#: Single opcodes on the same 12-byte command channel the axis stream already uses, taken
#: from the vendor reference GUI ``Lite3_All_control_v20153.py:310-320``, where each one is
#: a button. NOT velocity streams: nothing above can shape, slow, shorten, steer or
#: interrupt what the firmware does with them, and the only decision this file makes is
#: whether to put the twelve bytes on the wire at all.
BACKFLIP_CODE = 0x21010502        # 后空翻 -- the PLAIN backflip, listed in the 2.0.153 GUI
CARPET_BACKFLIP_CODE = 0x2101050C  # "Carpet Backflip" -- listed in NO vendor GUI we hold
TWIST_JUMP_CODE = 0x2101020D    # 扭身跳 -- twist jump
HELLO_CODE = 0x21010507         # 打招呼 -- GREET: the front-leg wave. On a quadruped this
                                # is what "arm wave" means; there is no manipulator on a
                                # Venture and `MAPPO_PAYLOAD` is `none` on both of ours.
END_ACTION_CODE = 0x21010C0B    # 结束动作 -- END ACTION, its own separate RED button there

#: Metres the backflip travels BACKWARD.
#:
#: ⚠️ THIS IS THE VENDOR'S FIGURE AND NOT A MEASUREMENT. No Venture in this repository has
#: ever done it, so nobody here knows what it is on this firmware, this payload or this
#: floor. It is used for one thing only: the floor that ``--rear-clearance-metres`` has to
#: clear. That is the direction it is safe to be wrong in -- if the real travel is further,
#: the operator's clearance was checked against a number that was too small, so this is a
#: minimum to refuse below and never a promise of where the robot will stop.
BACKFLIP_TRAVEL_M = 1.5

_VENDOR_COMMAND = struct.Struct("<3I")


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
    SWEEP = "sweep"
    LOOK = "look"

    def __init__(self, kind: str, *, yaw_speed_rad_s: float, turn_sign: int = +1,
                 revolutions: float = SPIN_REVOLUTIONS, shakes: int = SHAKE_COUNT,
                 look_rad: float = SWEEP_RAD) -> None:
        if kind not in (self.SPIN, self.SHAKE, self.SWEEP, self.LOOK):
            raise Refusal(
                f"unknown gesture {kind!r}; it is one of "
                f"{self.SPIN}, {self.SHAKE}, {self.SWEEP}, {self.LOOK}")
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
        elif kind == self.LOOK:
            # ONE LEG, AND IT DOES NOT COME BACK. That is the whole difference from
            # `sweep`, and it is the point: a sweep that returns to its start heading
            # leaves the next attempt facing exactly the direction that just failed. This
            # is used between attempts of a run that ended "goal never sighted", so the
            # robot has to STAY where it looked for the next attempt to see anything new.
            if look_rad == 0.0:
                raise Refusal("a look of zero degrees turns nowhere and sees nothing new")
            self._legs = [(abs(look_rad), +1 if look_rad > 0 else -1)]
        elif kind == self.SWEEP:
            # Out to one side, across to the other, back to the middle. The robot ends on
            # the heading it was stuck on, having pointed the camera 90 degrees either way.
            self._legs = [(SWEEP_RAD, self.turn_sign),
                          (2.0 * SWEEP_RAD, -self.turn_sign),
                          (SWEEP_RAD, self.turn_sign)]
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


@dataclass(frozen=True)
class VendorAction:
    """One vendor canned action: an opcode, and the two facts that decide the room.

    ``rear_clearance_m`` is the floor ``--rear-clearance-metres`` must clear, and it is
    deliberately not the same kind of number for the two actions: the backflip's is the
    vendor's own travel figure, and the twist jump's is this platform's OWN footprint --
    :data:`PLATFORM_HALF_DIAGONAL_M` doubled, the number ``check_room`` already uses -- for
    the honest reason that nobody knows how far a twist jump goes. Neither is a number
    invented here, and ``travel`` says out loud which is which.
    """

    kind: str
    code: int
    vendor_name: str
    travel: str
    rear_clearance_m: float
    rear_basis: str


VENDOR_ACTIONS = {
    "backflip": VendorAction(
        kind="backflip",
        code=BACKFLIP_CODE,
        vendor_name="后空翻 / backflip (GUI button: Carpet Backflip)",
        travel=f"TRAVELS ~{BACKFLIP_TRAVEL_M:.1f} m BACKWARD -- vendor figure, never "
               f"measured on either Venture",
        rear_clearance_m=BACKFLIP_TRAVEL_M,
        rear_basis=f"the vendor's own ~{BACKFLIP_TRAVEL_M:.1f} m backward travel",
    ),
    "carpet-backflip": VendorAction(
        kind="carpet-backflip",
        code=CARPET_BACKFLIP_CODE,
        vendor_name='"Carpet Backflip" -- in NO vendor GUI this repository holds',
        travel=f"TRAVELS BACKWARD. Distance still UNMEASURED: the vendor pose channel "
               f"freezes for the whole manoeuvre (2026-09-07, both Ventures: one distinct "
               f"pos_world value across the entire flip), so the ~{BACKFLIP_TRAVEL_M:.1f} m "
               f"figure remains the operator's, not a measurement",
        rear_clearance_m=BACKFLIP_TRAVEL_M,
        rear_basis=f"the operator's ~{BACKFLIP_TRAVEL_M:.1f} m figure, unmeasured because "
                   f"the pose channel does not run during the manoeuvre",
    ),
    "hello": VendorAction(
        kind="hello",
        code=HELLO_CODE,
        vendor_name="打招呼 / greet -- the front-leg wave",
        travel="travel UNKNOWN -- never sent from this repository, by anyone. A greeting "
               "SHOULD lift a front leg and put it back, and that is exactly the "
               "assumption this file is not allowed to make: nothing here has watched it, "
               "so it is treated as one that travels",
        rear_clearance_m=2 * PLATFORM_HALF_DIAGONAL_M,
        rear_basis=f"this platform's own {2 * PLATFORM_HALF_DIAGONAL_M:.2f} m footprint, "
                   f"because its real travel is unmeasured and no number here may be "
                   f"invented to stand in for it. Measure it and this can come down",
    ),
    "twist-jump": VendorAction(
        kind="twist-jump",
        code=TWIST_JUMP_CODE,
        vendor_name="扭身跳 / twist jump",
        travel="travel UNKNOWN -- never sent from this repository, by anyone, in any "
               "direction. It is NOT known to stay in place, so it is treated as one that "
               "travels",
        rear_clearance_m=2 * PLATFORM_HALF_DIAGONAL_M,
        rear_basis=f"this platform's own {2 * PLATFORM_HALF_DIAGONAL_M:.2f} m footprint, "
                   f"because its real travel is unmeasured and no number here may be "
                   f"invented to stand in for it",
    ),
}

#: The kinds an unattended end-of-run path may fire, and the kinds it may not.
#:
#: ⛔ THIS SPLIT IS THE SAFETY PROPERTY, not a tidy-up. Everything in ``ARRIVAL_KINDS``
#: keeps the robot's centre where it is; everything in ``OPERATOR_ONLY_KINDS`` either
#: travels or has never been observed well enough to claim it does not. ``mission.py``
#: fires the first set from `play_flourish` when a run arrives, fails or loses sight of its
#: goal, with nobody necessarily watching. It cannot reach the second: `_validate_action`
#: refuses without ``--operator-triggered``. Since 2026-09-07 ONE arrival path passes that
#: flag -- the operator's per-run flip checkbox; see the module docstring -- and
#: ``test_flourish.py`` asserts both that no arrival path names these kinds and that a
#: mission-shaped invocation of one is refused.
ARRIVAL_KINDS = (Flourish.SPIN, Flourish.SHAKE, Flourish.SWEEP, Flourish.LOOK)
OPERATOR_ONLY_KINDS = tuple(VENDOR_ACTIONS)

#: What this repository DOES NOT KNOW about these two opcodes, printed in full before
#: either one can fire. Not a disclaimer: it is the list the operator has to close by
#: discovering each item on a charged robot in a clear space, and every entry here is a way
#: an unattended firing could go wrong that nothing in this stack would catch.
UNKNOWNS = (
    "THE MINIMUM BATTERY. Still unmeasured, though the DRAW is now known to be small: three "
    "actions back to back cost 2 percentage points on LITE3-A (71->69) and one cost 1 point "
    "on LITE3-B. That bounds the drain, not the floor below which a flip fails. There is no "
    "default: state --acrobatic-battery-floor-pct, which is fed into robot_link's existing "
    "battery gate.",
    "[MEASURED 2026-09-07] THE ROBOT MUST BE IN FORCE-CONTROL 6. All five captured actions "
    "fired from basic_state 6 with error_state 0, on both Ventures. A robot at 98 (a normal "
    "post-boot state) must be re-zeroed and stood first. This path applies "
    "the axis transport's own `assert_axis_state_ready` -- force-control basic 6, policy 0, "
    "an allowed gait state, motion stationary/stepping -- because it is the only state gate "
    "in this repository with hardware evidence behind it. It was written for VELOCITY AXES. "
    "Whether it is the right gate for a canned action is not known: it may refuse a robot "
    "that would have flipped fine, and it may pass one that should not be asked to.",
    "HOW LONG THE ACTION RUNS. Hence --action-hold-seconds, with no default: the END ACTION "
    "opcode is sent after it, and a stop delivered mid-flip would be delivered to a robot "
    "that is upside down.",
    "[MEASURED 2026-09-07] THEY SELF-TERMINATE. The hand controller was packet-captured "
    "performing all three and sent NO end-action; the robot returned to force-control 6 "
    "unaided each time. Backflips took 5.4-5.9 s (basic 6->18->5->6); the twist jump took "
    "1.1 s and never left 6. This path no longer sends 0x21010C0B.",
    "[MEASURED 2026-09-07] IT IS LEFT STANDING IN 6, error_state 0. Odometry is frozen for "
    "the manoeuvre and resumes once the robot walks again -- so a run that flips and then "
    "navigates is fine, but a pose read in the seconds right after a flip is stale.",
    "WHETHER THE MOTION HOST WANTS THE 4 Hz HEARTBEAT alongside a canned action. The vendor "
    "GUI streams one continuously; this path sends the opcode and nothing else, because a "
    "heartbeat stream here would be invented rather than evidenced. If nothing happens at "
    "all, that is the first thing to suspect and it is a null result, not a safe one.",
    "HOW FAR THE TWIST JUMP TRAVELS, in any direction, at all.",
    f"WHETHER ~{BACKFLIP_TRAVEL_M:.1f} m IS THIS ROBOT. It is the vendor's number for a "
    f"Lite3, not a measurement of this one, on this firmware, with this payload, on this "
    f"floor.",
)


def vendor_packet(code: int) -> bytes:
    """One 12-byte vendor simple command, ``<3I`` of ``(code, 0, 0)``. Allowlisted.

    ``lite3_control_mode_udp.simple_packet`` and ``lite3_axis_udp.axis_packet`` both refuse
    a code they do not name, and this refuses for the same reason: on this channel an
    unrecognised opcode is not an error that comes back, it is whatever that opcode happens
    to mean on this firmware, executed by a robot standing in the room.
    """
    if code not in (BACKFLIP_CODE, CARPET_BACKFLIP_CODE, TWIST_JUMP_CODE, HELLO_CODE,
                    END_ACTION_CODE):
        raise Refusal(f"unsupported Lite3 vendor action code: {code:#010x}")
    return _VENDOR_COMMAND.pack(code, 0, 0)


def action_packets(action: VendorAction) -> list[tuple[str, int, bytes]]:
    """Every datagram this kind puts on the wire, in order. Pure, so it can be asserted.

    ONE datagram, and the absence of a second is now a MEASUREMENT rather than a guess.

    An earlier version of this function sent 结束动作 / END ACTION after every action, on
    the reasoning that the vendor gives it its own red button and therefore does not
    promise self-termination. On 2026-09-07 the hand controller was packet-captured
    performing all three of these actions on LITE3-A, and it sent NO end-action for any of
    them -- while the robot's own state stream showed it returning to force-control 6
    unaided every time (basic 6 -> 18 -> 5 -> 6 for both backflips in 5.4-5.9 s;
    motion_state 0 -> 4 -> 0 for the twist jump in 1.1 s, never leaving 6). The same was
    then reproduced on LITE3-B.

    So these actions self-terminate, and a stop sent afterwards is a command delivered to
    a robot that is already standing again for reasons nobody here has characterised.
    Matching the vendor's own controller is the evidenced choice; inventing a stop it
    never sends is not.
    """
    return [(action.kind, action.code, vendor_packet(action.code))]


def fire(sock, address, action: VendorAction, *, hold_s: float,
         sleep=time.sleep, printer=print) -> list[tuple[str, int, bytes]]:
    """Put :func:`action_packets` on the wire, then hold while the manoeuvre runs.

    Takes the socket rather than opening one, so the send order is testable with no robot
    and no network.

    The hold is AFTER the single opcode, not between two of them. It used to sit between
    the action and an end-action this function no longer sends (see
    :func:`action_packets`), but it is still the operator's number and still load-bearing:
    a backflip was measured taking 5.4-5.9 s to return the robot to force-control 6, and
    returning before then would close the socket and hand control back while the robot is
    still upside down.
    """
    sent: list[tuple[str, int, bytes]] = []
    for label, code, packet in action_packets(action):
        sock.sendto(packet, address)
        sent.append((label, code, packet))
        printer(f"[flourish] sent {label} {code:#010x}")
    sleep(hold_s)
    return sent


def describe(kind: str, yaw_speed_rad_s: float, look_rad: float = SWEEP_RAD) -> str:
    """The plan, in the shape the commissioning probes print theirs."""
    if kind in VENDOR_ACTIONS:
        # Falling through would print the SHAKE plan -- "the centre does not move" -- over
        # an opcode that travels 1.5 m backward. The one sentence in this file that must
        # never be printed about a backflip is the one the `else` branch below ends with.
        raise Refusal(
            f"{kind!r} is a vendor canned action, not a turn this file composes; its plan "
            f"comes from action_brief() and it does not have a yaw speed, a leg count or a "
            f"duration this stack can predict")
    if kind == Flourish.SPIN:
        span, legs = 2.0 * math.pi * SPIN_REVOLUTIONS, 1
        what = f"one full turn in place, {math.degrees(span):.0f} deg"
    elif kind == Flourish.LOOK:
        span, legs = abs(look_rad), 1
        what = (f"turn {math.degrees(look_rad):+.0f} deg and STAY there, so the next "
                f"attempt looks somewhere new")
    elif kind == Flourish.SWEEP:
        span, legs = 4.0 * SWEEP_RAD, 3
        what = (f"look {math.degrees(SWEEP_RAD):.0f} deg each side of the stuck heading, "
                f"then back to it")
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
    parser.add_argument("--kind",
                        choices=(*ARRIVAL_KINDS, *OPERATOR_ONLY_KINDS),
                        default=Flourish.SPIN)
    parser.add_argument("--degrees", type=float, default=90.0,
                        help="for --kind look: how far to turn, signed, and STAY there")
    parser.add_argument("--turn-sign", type=int, default=1, choices=(1, -1),
                        help="which way to turn. A CLEARANCE choice: both yaw primitives "
                             "are measured within 0.03%% of each other")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--lane-width-metres", type=float, default=None,
                        help="clear width, BOTH SIDES. This robot TURNS IN PLACE here and "
                             "has no lateral sensing")

    vendor = parser.add_argument_group(
        f"vendor canned actions (--kind {' / '.join(OPERATOR_ONLY_KINDS)}): "
        f"OPERATOR-TRIGGERED ONLY, and every flag below has NO DEFAULT")
    vendor.add_argument("--operator-triggered", action="store_true",
                        help="a HUMAN is firing this, now, watching this robot. Required "
                             "by the travelling kinds and by nothing else. No automatic "
                             "end-of-run path passes it, and that is what keeps a backflip "
                             "off the end of a run nobody is steering")
    vendor.add_argument("--rear-clearance-metres", type=float, default=None,
                        help="clear floor BEHIND the robot, measured with a tape. This "
                             "platform has NO REAR SENSING OF ANY KIND and the backflip "
                             f"travels ~{BACKFLIP_TRAVEL_M:.1f} m backward into it")
    vendor.add_argument("--acrobatic-battery-floor-pct", type=float, default=None,
                        help="refuse below this battery percentage. Nobody has measured "
                             "what a Lite3 flip draws and a brownout mid-flip is a fall, "
                             "so this is YOUR number; it is fed into robot_link's existing "
                             "--battery-abort gate rather than becoming a second one")
    vendor.add_argument("--action-hold-seconds", type=float, default=None,
                        help="how long to hold before this tool returns and closes the "
                             "socket. Measured 2026-09-07: a backflip takes 5.4-5.9 s to "
                             "return the robot to force-control 6, a twist jump 1.1 s. "
                             "Returning sooner hands control back mid-manoeuvre")
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


def check_rear(args, action: VendorAction) -> None:
    """⛔ THIS ROBOT HAS ZERO REAR SENSING, and this is the point of use.

    Not poor rear sensing: none. No rear camera, no rear ultrasonic, no bumper. The one
    camera on the platform looks FORWARD through 134 degrees, and every obstacle check
    anywhere in this stack is downstream of it. A backflip travels ~1.5 m into the single
    direction the robot cannot see at all, and nothing in software will notice the wall,
    the tripod leg or the person standing there.

    So the clearance behind is the operator's tape measure, stated as a number, with no
    default -- the identical rule ``check_room`` above already applies to the sideways
    sweep, for the identical reason. A default here would be this file guessing about
    floor it has never been told anything about.
    """
    if args.rear_clearance_metres is None:
        raise Refusal(
            f"state --rear-clearance-metres. THIS ROBOT HAS NO REAR SENSING AT ALL -- no "
            f"rear camera, no ultrasonic, no bumper -- and nothing in this stack can keep "
            f"{action.kind} out of the space behind the robot once the opcode is sent: "
            f"{action.travel}. The floor behind is the operator's measurement and this "
            f"file ships no default for it.")
    if args.rear_clearance_metres < action.rear_clearance_m:
        raise Refusal(
            f"--rear-clearance-metres says {args.rear_clearance_metres:.2f} m behind the "
            f"robot and {action.kind} needs at least {action.rear_clearance_m:.2f} m, "
            f"which is {action.rear_basis}. Nothing on this platform can see that space. "
            f"Clear more floor or measure again; there is no flag that waives this.")


def _refuse_unmeasured_and_nonpositive(**values: float) -> None:
    """``measurement``'s two refusals, with its ``Refusal`` converted to this file's.

    ⚠️ THERE ARE TWO CLASSES CALLED ``Refusal`` IN THIS TREE and they are unrelated:
    ``measurement.Refusal`` is an ``Exception`` and ``reverse_along_path.Refusal`` -- the
    one this file imports and catches -- is a ``RuntimeError``. Converting here means the
    caller keeps ONE ``except Refusal`` that prints the operator's sentence, instead of the
    helper's refusal falling through to the catch-all below it and being announced as
    ``REFUSED: Refusal: ...``, which reads like a code fault and sends an operator to look
    at the software instead of at the robot.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from deep_robotics.lite3.commissioning import measurement

    try:
        measurement.refuse_unmeasured(**values)
        measurement.require_positive_finite(**values)
    except measurement.Refusal as refusal:
        raise Refusal(str(refusal)) from None


def _validate_action(args, action: VendorAction) -> None:
    """Every precondition that can be checked without a robot, before anything opens.

    Runs on the DRY RUN too, which is deliberate: each of these costs nothing to discover
    at a laptop and costs a session to discover with a robot standing on a cleared floor.
    """
    if not args.operator_triggered:
        raise Refusal(
            f"--kind {action.kind} needs --operator-triggered. This is not a gesture that "
            f"may fire at the end of a run: it {action.travel}, and the property that "
            f"licensed spinning a robot nobody is watching is that the spin stays where it "
            f"is. Pass it only if you are standing there, looking at the robot, with the "
            f"emergency stop in your hand.")
    check_rear(args, action)
    if args.lane_width_metres is None:
        # Said here rather than in `check_room`, whose sentence is about a turn through a
        # full circle and would be a false description of a flip. The FLOOR it applies is
        # the right one and is reused unchanged below: whatever else these actions do, the
        # robot's own footprint has to fit, and nothing sees sideways either.
        raise Refusal(
            f"state --lane-width-metres. A {action.kind} is a whole-body manoeuvre this "
            f"stack cannot shape or interrupt, and the robot has no lateral sensing any "
            f"more than it has rear sensing. The width is the operator's measurement.")
    check_room(args)
    _refuse_unmeasured_and_nonpositive(**{
        "--rear-clearance-metres": args.rear_clearance_metres,
        "--lane-width-metres": args.lane_width_metres,
        "--acrobatic-battery-floor-pct": args.acrobatic_battery_floor_pct,
        "--action-hold-seconds": args.action_hold_seconds})
    # ⛔ ONE BATTERY GATE, AND IT IS robot_link's. The operator's acrobatic floor is folded
    # INTO `--battery-abort` rather than becoming a second check further down, so there is
    # exactly one place in this process that compares a battery reading against a limit and
    # it is the one every other moving probe in this tree already uses and tests. A second
    # gate would be a second thing to forget to apply.
    args.battery_abort = max(args.battery_abort, args.acrobatic_battery_floor_pct)


def perform(args, yaw_speed_rad_s: float) -> int:
    """Command the gesture. ⛔ ``robot-stack/SAFETY.md`` governs this."""
    import contextlib

    from deep_robotics.lite3.commissioning import robot_link

    link = robot_link.connect(args)
    loco = link.locomotion
    robot_link.preflight(link, args)

    gesture = Flourish(args.kind, yaw_speed_rad_s=yaw_speed_rad_s,
                       turn_sign=args.turn_sign,
                       look_rad=math.radians(args.degrees))
    tick_s = 1.0 / args.control_hz
    try:
        while True:
            # `Lite3Pose.yaw`, not `yaw_rad`. The sibling upstream repository names
            # the same field `yaw_rad`, and writing this against that one cost a
            # live AttributeError on a robot that had already finished its run.
            pose = loco.pose()
            wz = gesture.step(pose.yaw, time.monotonic())
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


def action_brief(action: VendorAction, args) -> str:
    """What is about to happen, what it needs, and -- at length -- what is NOT known.

    Written in the shape ``measurement.brief`` prints a commissioning probe's, and returned
    as a string rather than printed for the same reason ``describe`` is: it is the thing a
    test can read. The third section is the one that matters. This is the only thing in
    this repository that fires an opcode whose effect nobody here has ever observed, and an
    operator who cannot read the code has to be told that in the terminal, not in a
    docstring they will never open.
    """
    rule = "=" * 78
    stated = "?" if args.rear_clearance_metres is None else f"{args.rear_clearance_metres:.2f}"
    hold = "?" if args.action_hold_seconds is None else f"{args.action_hold_seconds:.1f}"
    floor = ("?" if args.acrobatic_battery_floor_pct is None
             else f"{args.acrobatic_battery_floor_pct:.0f}")
    lines = [
        rule,
        f"Lite3 VENDOR CANNED ACTION -- {action.vendor_name}",
        rule,
        "MOVES THE ROBOT: YES, ACROBATICALLY -- keep the emergency stop in your hand",
        "",
        "WHAT IT DOES",
        f"  sends ONE opcode, {action.code:#010x}, as {_VENDOR_COMMAND.size} bytes to "
        f"{args.motion_host}:{args.command_port},",
        f"  then holds {hold}s while the firmware runs it, and sends NOTHING else.",
        "  No END ACTION: the vendor's own hand controller was captured performing all",
        "  three of these and sent none, and the robot returned to force-control 6",
        "  unaided each time (measured on both Ventures, 2026-09-07).",
        "  That is the whole of this stack's involvement. It cannot shape, slow, shorten,",
        "  steer or interrupt what the firmware does in between: there is no velocity",
        "  stream here and no control loop, and no measured number times any of it.",
        "",
        "TRAVEL",
        f"  {action.travel}.",
        "  ⛔ THIS ROBOT HAS NO REAR SENSING AT ALL. No rear camera, no ultrasonic, no",
        "     bumper; the one camera looks FORWARD through 134 degrees. Nothing in",
        "     software will see what is behind the robot, before or during this.",
        f"  rear clearance stated: {stated} m",
        f"  rear clearance floor:  {action.rear_clearance_m:.2f} m, which is "
        f"{action.rear_basis}",
        "",
        "WHAT IT NEEDS FROM YOU",
        f"  - {action.rear_clearance_m:.2f} m or more of CLEAR FLOOR BEHIND the robot, "
        f"measured, and nobody in it",
        f"  - a clear width both sides too: {args.lane_width_metres or '?'} m stated",
        "  - the robot standing, handed over to this laptop, in the vendor state its own",
        "    gate accepts -- and see the second unknown below about that gate",
        f"  - a charged battery: --acrobatic-battery-floor-pct {floor} is YOUR floor and it",
        "    is checked by robot_link's existing battery gate before anything is sent",
        "  - your hand on the emergency stop, and your eyes on the robot, for all of it",
        "",
        "⛔ WHAT IS NOT KNOWN. NEITHER OPCODE HAS EVER BEEN SENT BY THIS REPOSITORY.",
        "   Nothing below has been guessed at, defaulted or filled in. It is the list you",
        "   are about to close by discovery, on a robot, in a clear space:",
    ]
    for unknown in UNKNOWNS:
        first, *rest = _wrap(unknown, 72)
        lines.append(f"   - {first}")
        lines.extend(f"     {line}" for line in rest)
    lines.append(rule)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """Wrap without importing textwrap, to keep this file's import list boring."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def perform_action(args, action: VendorAction, *, printer=print,
                   socket_factory=socket.socket, sleep=time.sleep) -> int:
    """Fire one vendor canned action. ⛔ ``robot-stack/SAFETY.md`` governs this.

    Shares nothing with :func:`perform` because they have nothing to share: that one closes
    a loop over measured heading at 10 Hz and can abandon the gesture halfway through, and
    this one puts twelve bytes on the wire and then has no further say in anything.
    """
    import contextlib

    from deep_robotics.lite3.commissioning import robot_link

    link = robot_link.connect(args)
    try:
        # THE BATTERY GATE IS robot_link's OWN, not a second one written here.
        # `--acrobatic-battery-floor-pct` was folded into `args.battery_abort` back in
        # `_validate_action`, so this single call applies the operator's acrobatic floor
        # along with the state-freshness and error_state checks every moving probe gets.
        robot_link.preflight(link, args, printer=printer)

        # The vendor state gate. See the second entry in UNKNOWNS: this was written for
        # velocity axes and whether it is the RIGHT gate for a canned action is not known.
        # It is applied anyway because it is the only state gate in this repository with
        # hardware evidence behind it, and because refusing a robot that is lying down or
        # reporting a fault is correct for a backflip whatever else turns out to be true.
        gate = getattr(link.implementation, "assert_axis_state_ready", None)
        if gate is None:
            raise Refusal(
                f"--kind {action.kind} needs --locomotion-transport axis. The vendor state "
                f"gate that refuses a robot which is lying down, faulted or in the wrong "
                f"mode exists only on that transport, and firing an unvalidated acrobatic "
                f"opcode with no state gate at all is not something this file will do.")
        gate()

        sock = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Bound to the SAME source port the axis stream sends its accepted commands
            # from, so the datagram looks like the ones this robot is known to act on
            # rather than like one from an ephemeral port nobody has tested. The bind is
            # also a lock: if another run already holds that port, this refuses instead of
            # firing an acrobatic action into a session somebody else is driving.
            try:
                sock.bind(("0.0.0.0", args.axis_local_port))
            except OSError as error:
                raise Refusal(
                    f"cannot bind the axis source port {args.axis_local_port}: {error}. "
                    f"Something else is already holding it -- most likely a live run in "
                    f"another terminal. Refusing to fire {action.kind} into a session "
                    f"somebody else is driving.") from None
            fire(sock, (args.motion_host, args.command_port), action,
                 hold_s=args.action_hold_seconds, sleep=sleep, printer=printer)
        finally:
            sock.close()
    finally:
        # `suppress` for the same reason `perform` suppresses its stop: a shutdown that
        # itself fails must not mask why the run ended.
        with contextlib.suppress(Exception):
            link.locomotion.shutdown()
    printer(f"[flourish] {action.kind} opcode sent, end-action opcode sent. WHAT THE ROBOT "
            f"DID WITH THEM IS NOT READ BACK BY ANYTHING HERE -- look at the robot, and "
            f"assume its mode is now whatever the firmware chose.")
    return 0


def action_main(args, *, printer=print) -> int:
    """``--kind backflip`` / ``--kind twist-jump``: one vendor opcode, fired by a human.

    A separate entry point from :func:`main`'s turn path, sharing none of its steps,
    because the two have almost nothing in common: a turn is composed here out of measured
    primitives and closed against measured heading, and this is an opcode whose effect
    nobody in this repository has observed. Sharing a code path would have meant sharing
    ``describe``'s "travel none -- the centre does not move".
    """
    action = VENDOR_ACTIONS[args.kind]
    printer(f"[flourish] {args.robot_id}:")
    printer(action_brief(action, args))
    try:
        # BEFORE the --live branch, so every one of these refusals happens on the dry run
        # too, at a laptop, rather than in front of a robot on a floor somebody cleared.
        _validate_action(args, action)
        if not args.live:
            printer("\n[flourish] plan only. Nothing was opened, nothing was sent, and no "
                    "socket exists in this process. Re-read the unknowns above; when you "
                    "have answers for them, add --live --operator-ready.")
            return 0
        if not args.operator_ready:
            raise Refusal("--live needs --operator-ready.")
        return perform_action(args, action, printer=printer)
    except Refusal as refusal:
        print(f"[flourish] REFUSED: {refusal}", file=sys.stderr)
        return 1
    except Exception as refusal:
        # The same reason `main` has this: what reaches here is the vendor mode gate --
        # `Lite3LinkLost: Lite3 basic_state=1; axis motion requires documented
        # force-control state 6` and its siblings -- which means the robot is in the wrong
        # mode, which is one control on the vendor app and not a software fault.
        print(f"[flourish] REFUSED: {type(refusal).__name__}: {refusal}", file=sys.stderr)
        return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    if args.kind in VENDOR_ACTIONS:
        # Handed off whole. Below this line is the turn path, which is timed against a
        # measured yaw speed the canned actions do not have and must not be made to fake.
        return action_main(args)
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
    print(describe(args.kind, yaw, math.radians(args.degrees)))
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
    except Exception as refusal:
        # A GESTURE MUST NOT RAISE AT AN OPERATOR. What reaches here is the vendor mode
        # gate -- `Lite3LinkLost: Lite3 basic_state=1; axis motion requires documented
        # force-control state 6` and its siblings -- and it means the robot is in the
        # wrong mode, which is one control on the vendor app. Measured 2026-09-04: every
        # attempt on both robots ended with the surrender gesture printing a traceback on
        # top of the mission's own explanation, which reads as a code fault and sends an
        # operator to look at the software instead of at the robot.
        print(f"[flourish] REFUSED: {type(refusal).__name__}: {refusal}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
