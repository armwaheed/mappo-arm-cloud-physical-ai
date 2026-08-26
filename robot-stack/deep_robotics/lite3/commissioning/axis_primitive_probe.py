#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""How fast does each axis primitive actually move THIS Lite3, and in which direction?

    python3 axis_primitive_probe.py --robot-id LITE3-A --firmware V1.0.8 --payload none \\
        --locomotion-transport axis --axis-profile lite3-axis-LITE3-A.json \\
        --lane-metres 8.0 --lane-width-metres 3.0
    # prints the plan and exits. Add --live --operator-ready to actually walk.

WHY THIS EXISTS, AND WHY IT IS NOT A GAIT FLOOR.

The transport both Ventures have actually walked on is the profile-gated simple-axis one,
and its mapping is **sign-only**: a profile holds one evidenced raw value per direction,
and every command past the profile's linear deadband emits that one value at full scale.
``--derate`` and ``--max-vx`` never reach the wire. A commanded speed on this transport is
a direction, not a speed.

So the two questions issue #13 asks -- "the lowest forward command that sustains a gait"
and "the achieved/commanded velocity ratio" -- have no answer here. There is no lowest
command: there is one command per direction. There is no ratio: the denominator never
left the laptop. Pointing ``gait_floor_probe.py`` at this transport does not fail, which
is the dangerous part -- every rung walks, at the same speed, and it reports the bottom
rung as the floor. Both ladder probes now refuse this transport by name.

What replaces them is this: **the speed each primitive delivers.** That number is not a
convenience. ``lite3-axis-profile/v1`` has a ``measured_m_s`` field, and the live
navigator's envelope gate is enforced against it -- a primitive declared at 0.30 m/s
passes ``--max-vx 0.35``, and if the robot actually delivers 0.47 m/s then the safety veto
was planned around a robot 1.6x slower than the one in the room. Until this probe runs,
that field is whatever somebody typed, and a primitive with no declared speed only prints
a warning that nobody checked.

⚠️ THE OTHER HALF OF THAT GATE IS NOW STATED RATHER THAN INHERITED.
``_validate_axis_profile_speeds`` compares the measured speed against
``--max-vx x --derate``. Until 2026-08-26 the Lite3's ``--max-vx`` defaulted to 0.35 m/s
-- the **Go2's** arm-fitted envelope out of ``unitree/go2/visual_nav/avoidance.py``, where
``robot_radius`` and ``gait_floor`` were already refused instead -- so measuring this side
carefully did not make the comparison a Lite3 one. ``Lite3Bindings`` now blanks all three
envelope flags the way it blanks ``--robot-radius``: a live run refuses until the operator
states ``--max-vx``/``--max-vy``/``--max-wz``, and a dry run says out loud whose numbers
it fell back on. State them anyway when you paste this record's numbers into a live run;
the refusal tells you if you forgot.

THE COMMANDED MAGNITUDE HERE IS ARBITRARY, AND THAT IS THE POINT. Every treatment is
commanded at ``COMMAND_MARGIN`` x the profile's own linear deadband. Any other value above
that deadband produces the byte-identical datagram, so the number is a property of the
probe and never of the robot. It is recorded in the artefact so a reader can see that it
was not fitted to anything.

WHAT IT CHECKS THAT NOTHING ELSE DOES.

* **Direction.** A primitive must move the robot the way its name claims. The lateral
  pair inverts on the way to the wire -- the navigator's +y is left and the vendor's
  positive raw value is right -- so a profile with those two swapped is a robot that
  strafes into the side of the lane the operator cleared least. Nothing else in this
  repository would notice.
* **A dead primitive.** A raw value below the firmware's dead zone, or simply wrong,
  produces no motion at all. Measured on its own axis, per primitive, so a dead one is
  refused rather than averaged into the others.
* **A profile that under-declares.** If the profile already carries ``measured_m_s``, the
  report says whether the robot was faster than the declaration -- which is the only
  direction that matters, because that is the envelope gate passing a robot it did not
  check.

YAW IS DELIBERATELY NOT MEASURED HERE. Two things are unresolved and neither is this
probe's to decide: issue #13 still lists confirming the angular-velocity unit against
pose-yaw change as its own item, and ``Segment.yaw_change_deg`` is an unwrapped endpoint
difference that a turn through pi would report backwards. ``measured_rad_s`` therefore
stays undeclared and the yaw envelope stays unenforced. Said out loud rather than
half-measured.

WHAT THE NUMBER IS. The **maximum** delivered speed across the repeats, not the mean.
This number's job is to be compared against a safety ceiling, and a mean hides the fast
sample that is the one the ceiling has to survive. The mean and the spread are both
reported next to it so the choice is visible.

⚠️ THE ROBOT CRABS SIDEWAYS INTO SPACE IT CANNOT SEE. There is no lateral sensing on this
platform at all. Clear both sides of the lane, not just the ends.

Read ``../../../SAFETY.md``. ``--live`` is the only flag that moves a leg, and an operator
stays on the emergency stop for all of it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_robotics.lite3.commissioning import robot_link
from deep_robotics.lite3.commissioning.measurement import (
    WALKED_MARGIN_M,
    Refusal,
    brief,
    check_controls_are_still,
    control_baseline,
    merge_measurement,
    new_record,
    paste_block,
    print_paste_block,
    refuse_unmeasured,
    require_positive_finite,
    run_main,
    run_segment,
    stopped_afterwards,
    write_record,
)

#: Control period. Matches the navigation stack's 10 Hz so the vendor gait sees the same
#: cadence during the measurement that it will see on a real run.
TICK_S = 0.1

#: Seconds per segment. Longer than the gait ladder's because there is no ladder to fit
#: in the lane here -- one speed per direction -- and a longer segment makes
#: ``WALKED_MARGIN_M`` a smaller fraction of the travel being measured.
SEGMENT_S = 2.0

#: Treatments per primitive. Three is the fewest that can show a spread rather than just
#: a difference, and the spread is what decides whether the maximum is a fluke.
REPEATS = 3

#: Commanded magnitude, as a multiple of the profile's own linear deadband. Any value
#: above the deadband emits the identical datagram, so this only has to clear it; it is
#: above 1.0 by enough that a float comparison at the boundary cannot decide the run.
COMMAND_MARGIN = 1.5


class Primitive:
    """One profile primitive, and how to make the transport emit it.

    ``vx_sign``/``vy_sign`` are the signs of the navigation command that map to this
    primitive, and ``axis``/``expected_sign`` are the body-frame travel it must then
    produce. The lateral pair is the reason this table is written out rather than
    derived: the navigator's +y is left, the vendor's positive raw axis value is right,
    and the transport inverts between them. So ``lateral_positive`` -- a vendor-positive,
    rightward primitive -- is fired by a **negative** commanded ``vy`` and must produce
    **negative** body-frame lateral travel. Deriving that from the field name is exactly
    the mistake this table exists to prevent.
    """

    def __init__(self, name: str, axis: str, vx_sign: float, vy_sign: float,
                 expected_sign: int, travels: str) -> None:
        self.name = name
        self.axis = axis
        self.vx_sign = vx_sign
        self.vy_sign = vy_sign
        self.expected_sign = expected_sign
        self.travels = travels


#: Every linear primitive a ``lite3-axis-profile/v1`` can hold, in the order they are run.
#: Forward first: a forward primitive is the one an operator has most likely already seen
#: work, so a run that is going to fail fails while the robot is still pointing down the
#: lane.
PRIMITIVES = (
    Primitive("forward_positive", "forward", +1.0, 0.0, +1, "forwards"),
    Primitive("forward_negative", "forward", -1.0, 0.0, -1, "backwards"),
    Primitive("lateral_negative", "lateral", 0.0, +1.0, +1, "to its left"),
    Primitive("lateral_positive", "lateral", 0.0, -1.0, -1, "to its right"),
)


def primitives_in(profile) -> list:
    """The primitives this profile actually carries a raw value for, in run order.

    A profile is allowed to hold only the directions a demo needs, and the transport
    refuses at command time for any it does not hold. There is nothing to measure for an
    absent one, and inventing a segment for it would put the robot in motion to learn
    nothing.
    """
    present = [primitive for primitive in PRIMITIVES
               if getattr(profile, primitive.name) is not None]
    if not present:
        raise Refusal(
            "this axis profile carries no linear primitive at all, so there is nothing "
            "for the robot to be asked to do. Fill in at least one of "
            + ", ".join(primitive.name for primitive in PRIMITIVES)
            + " with its evidence reference first."
        )
    return present


def plan_for(primitive: Primitive, deadband_m_s: float, repeats: int) -> list:
    """``(role, vx, vy)`` triples for one primitive: a control before every treatment.

    Controls are interleaved rather than recorded once at the start because a warm robot
    walks differently from a cold one, and a contemporaneous control is the only thing
    that separates this primitive's delivery from the odometry's drift.
    """
    if repeats < 2:
        raise Refusal("a primitive needs at least 2 treatments to show a spread")
    command = deadband_m_s * COMMAND_MARGIN
    plan = []
    for _ in range(repeats):
        plan.append(("control", 0.0, 0.0))
        plan.append((primitive.name, primitive.vx_sign * command,
                     primitive.vy_sign * command))
    plan.append(("control", 0.0, 0.0))
    return plan


def planned_excursion_m(plan: list, segment_s: float, delivered_m_s: float) -> float:
    """Worst-case travel along the axis under test, for the room check.

    ``delivered_m_s`` is what a primitive is *assumed* to deliver for planning only. The
    whole point of the run is that nobody knows it yet, so this takes it from
    ``--assume-up-to`` and the operator is told that is what the lane was checked against.
    """
    treatments = [row for row in plan if row[0] != "control"]
    return len(treatments) * segment_s * delivered_m_s


def execute(loco, plan, *, segment_s: float, tick_s: float, printer=print,
            clock=time.monotonic, sleep=time.sleep) -> list:
    """Walk one primitive's plan, printing each segment as it lands. Always ends stopped."""
    segments = []
    with stopped_afterwards(loco):
        for index, (role, vx, vy) in enumerate(plan, start=1):
            segment = run_segment(loco, role=role, vx=vx, vy=vy,
                                  duration_s=segment_s, tick_s=tick_s,
                                  clock=clock, sleep=sleep)
            segments.append(segment)
            printer(f"  [{index:>2}/{len(plan)}] {role:<17} vx={vx:+.3f} vy={vy:+.3f} "
                    f"-> forward {segment.forward_mps:+.3f} m/s  "
                    f"lateral {segment.lateral_mps:+.3f} m/s")
    return segments


def delivered_speeds(segments, primitive: Primitive) -> list:
    """Per-treatment delivered speed on this primitive's own axis, net of drift.

    Signed, and deliberately so: the sign is the direction check, and averaging it away
    before looking would hide a profile whose lateral pair is swapped.
    """
    baseline = control_baseline(segments, primitive.axis)
    return [getattr(segment, f"{primitive.axis}_mps") - baseline
            for segment in segments if segment.role == primitive.name]


def check_primitive_moved(segments, primitive: Primitive) -> None:
    """Refuse a primitive that did not move the robot on its own axis.

    This is the Go2's no-motion refusal, applied per primitive and on the right axis.
    ``Segment.travelled`` cannot do it: it reads forward displacement only, so a lateral
    primitive that worked perfectly looks dead to it and one that never fired looks the
    same as one that did.

    Unlike a gait ladder, every treatment here is commanded at full scale -- there is no
    rung that is *supposed* to produce nothing -- so a treatment that does not travel can
    only mean the primitive is wrong or the legs were not running. Either way there is no
    number.
    """
    treatments = [segment for segment in segments if segment.role == primitive.name]
    if not treatments:
        raise Refusal(f"no {primitive.name} treatment ran, so there is nothing to report")
    dead = [segment for segment in treatments
            if abs(getattr(segment, f"{primitive.axis}_m")) < WALKED_MARGIN_M]
    if dead:
        raise Refusal(
            f"{len(dead)} of {len(treatments)} {primitive.name} treatments moved less "
            f"than {WALKED_MARGIN_M:.2f} m {primitive.axis}, even though every one of "
            f"them commanded that primitive at full scale -- there is no low rung in "
            f"this probe that could legitimately produce nothing.\n"
            f"Either the raw axis value in the profile is below the firmware's dead zone "
            f"for this axis, or the legs were not running. A zero here is not a slow "
            f"primitive; it is an absent one."
        )


def check_primitive_direction(segments, primitive: Primitive) -> None:
    """Refuse a primitive that moved the robot the opposite way to its name.

    The lateral pair inverts between the navigator's frame and the vendor's raw axis, and
    a profile that has those two the wrong way round is a robot that strafes into the
    side of the lane nobody cleared. This is the only check in the repository that would
    catch it, and it has to run before the speed is written down: a swapped profile
    produces a perfectly plausible magnitude.
    """
    wrong = [speed for speed in delivered_speeds(segments, primitive)
             if speed * primitive.expected_sign <= 0.0]
    if wrong:
        raise Refusal(
            f"{len(wrong)} {primitive.name} treatment(s) moved the robot the WRONG WAY. "
            f"This primitive must travel {primitive.travels}, and it did the opposite.\n"
            f"On the lateral pair this is the expected mistake: the navigator's +y is "
            f"left and the vendor's positive raw axis value is right, so a profile with "
            f"lateral_positive and lateral_negative swapped strafes into the side of the "
            f"lane you cleared least. Fix the profile, do not record a speed for it."
        )


def analyse(segments, primitive: Primitive, printer=print) -> dict:
    """Refuse first, then report. The declarable number is the maximum, and it says why."""
    check_controls_are_still(segments, primitive.axis)
    check_primitive_moved(segments, primitive)
    check_primitive_direction(segments, primitive)
    speeds = [abs(speed) for speed in delivered_speeds(segments, primitive)]
    fastest, slowest = max(speeds), min(speeds)
    mean = sum(speeds) / len(speeds)
    printer("")
    printer(f"  {primitive.name}: travels {primitive.travels}")
    for index, speed in enumerate(speeds, start=1):
        printer(f"    treatment {index}: {speed:.3f} m/s")
    printer(f"    max {fastest:.3f}  mean {mean:.3f}  min {slowest:.3f}  "
            f"spread {fastest - slowest:.3f} m/s")
    return {
        "primitive": primitive.name,
        "axis": primitive.axis,
        "travels": primitive.travels,
        "declare_m_s": fastest,
        "mean_m_s": mean,
        "min_m_s": slowest,
        "spread_m_s": fastest - slowest,
        "samples": len(speeds),
        "control_baseline_m_s": control_baseline(segments, primitive.axis),
        "segments": [segment.as_dict() for segment in segments],
    }


def compare_with_profile(results: dict, profile) -> list:
    """Say whether the profile already declares a speed, and whether it under-declares.

    Only the under-declaring direction is called a problem. A profile that claims a
    primitive is *faster* than it turned out to be makes the envelope gate stricter than
    it needs to be, which costs a demo nothing it cannot get back. A profile that claims
    it is slower is the gate passing a robot it never checked.
    """
    declared = dict(getattr(profile, "measured_m_s", ()) or ())
    notes = []
    for name, result in results.items():
        was = declared.get(name)
        if was is None:
            notes.append(f"- `{name}`: nothing was declared before this run.")
            continue
        if result["declare_m_s"] > was:
            notes.append(
                f"- ⚠️ `{name}`: the profile declares {was:.3f} m/s and this robot "
                f"delivered {result['declare_m_s']:.3f} m/s, "
                f"{result['declare_m_s'] / was:.2f}x that. Every live run gated on this "
                f"profile checked its envelope against the smaller number."
            )
        else:
            notes.append(f"- `{name}`: profile declared {was:.3f} m/s, measured "
                         f"{result['declare_m_s']:.3f} m/s. Not under-declared.")
    return notes


def measured_block(results: dict) -> str:
    """The ``measured_m_s`` object, ready to paste into the profile that was just used."""
    rows = ",\n".join(f'    "{name}": {result["declare_m_s"]:.3f}'
                      for name, result in sorted(results.items()))
    return '  "measured_m_s": {\n' + rows + "\n  }"


def record_context(args, profile, present) -> dict:
    """Everything that has to travel beside these numbers for them to still mean something.

    The profile hash is the load-bearing one. A primitive's speed is a property of one
    specific raw axis value, not of the robot in general: re-point the profile at a
    different value and every number measured here is stale. Without the hash there is
    nothing stopping the numbers being re-attached to a profile they never described --
    which is the same failure as copying a measurement between the two Ventures, one
    level down.
    """
    return {
        "firmware": args.firmware,
        "payload": args.payload,
        "segment_s": args.segment,
        "tick_s": args.tick,
        "repeats": args.repeats,
        "commanded_m_s": profile.linear_deadband_m_s * COMMAND_MARGIN,
        "linear_deadband_m_s": profile.linear_deadband_m_s,
        "lane_metres": args.lane_metres,
        "lane_width_metres": args.lane_width_metres,
        "assume_up_to_m_s": args.assume_up_to,
        "primitives": [primitive.name for primitive in present],
        "axis_profile_sha256": _profile_sha256(args.axis_profile),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    robot_link.add_context_arguments(parser)
    robot_link.add_link_arguments(parser, moving=True)

    method = parser.add_argument_group("the method")
    method.add_argument("--repeats", type=int, default=REPEATS,
                        help=f"treatments per primitive (default: {REPEATS})")
    method.add_argument("--segment", type=float, default=SEGMENT_S,
                        help=f"seconds per segment (default: {SEGMENT_S})")
    method.add_argument("--tick", type=float, default=TICK_S,
                        help=f"command period, seconds (default: {TICK_S})")
    method.add_argument("--assume-up-to", type=float, default=None, metavar="M_S",
                        help="for the ROOM CHECK ONLY: the fastest this robot might turn "
                             "out to walk. Nobody knows it yet -- that is what this probe "
                             "is for -- so the lane is checked against your upper bound. "
                             "No default")

    room = parser.add_argument_group("the room")
    room.add_argument("--lane-metres", type=float, default=None,
                      help="clear length ahead of and behind the robot, in metres")
    room.add_argument("--lane-width-metres", type=float, default=None,
                      help="clear width, BOTH SIDES -- this robot has no lateral sensing")

    parser.add_argument("--no-prompt", action="store_true",
                        help="do not pause between primitives to reposition the robot")
    parser.add_argument("--artefact", default=None,
                        help="write the machine-readable record here "
                             "(default: lite3-axis-primitives-<robot-id>.json)")
    return parser


def _validate(args) -> None:
    robot_link.require_sign_only_transport(
        args,
        measures="the delivered speed of a fixed axis primitive",
        instead="a gait floor and an actuator gain, with gait_floor_probe.py and "
                "actuator_gain_probe.py",
    )
    robot_link.require_walked_transport(args)
    refuse_unmeasured(**{"--assume-up-to": args.assume_up_to,
                         "--lane-metres": args.lane_metres,
                         "--lane-width-metres": args.lane_width_metres})
    require_positive_finite(**{"--assume-up-to": args.assume_up_to,
                               "--segment": args.segment, "--tick": args.tick,
                               "--lane-metres": args.lane_metres,
                               "--lane-width-metres": args.lane_width_metres})
    if args.tick >= args.segment:
        raise Refusal("--tick must be shorter than --segment, or a segment carries at "
                      "most one command")


def _check_room(primitive: Primitive, plan: list, args, printer=print) -> None:
    """Refuse a lane that is too short for what this primitive is about to be asked to do."""
    needed = planned_excursion_m(plan, args.segment, args.assume_up_to)
    have = args.lane_metres if primitive.axis == "forward" else args.lane_width_metres
    flag = "--lane-metres" if primitive.axis == "forward" else "--lane-width-metres"
    printer(f"[{primitive.name}] up to {needed:.1f} m {primitive.travels} if this robot "
            f"turns out to walk at --assume-up-to {args.assume_up_to:.2f} m/s")
    if needed > have:
        raise Refusal(
            f"{primitive.name} could travel {needed:.1f} m {primitive.travels} and "
            f"{flag} says there is {have:.1f} m. Nothing here knows how fast this "
            f"primitive is yet -- that is the measurement -- so the lane is checked "
            f"against your --assume-up-to. Lengthen the lane, lower --repeats or "
            f"--segment, or lower --assume-up-to only if you can defend the smaller "
            f"number."
        )


def _paste(record, results: dict, notes: list) -> str:
    rows = [
        ("robot", record.robot_id),
        ("firmware", record.context.get("firmware")),
        ("payload", record.context.get("payload")),
        ("transport", "axis (sign-only; commanded magnitude never reaches the wire)"),
        ("axis profile SHA-256", record.context.get("axis_profile_sha256")),
        ("command envelope", f"{record.context.get('segment_s')} s per segment at "
                             f"{1.0 / record.context.get('tick_s', 1.0):.0f} Hz, "
                             f"{record.context.get('repeats')} treatments per primitive"),
    ]
    for name, result in sorted(results.items()):
        rows.append((f"`{name}` delivered",
                     f"{result['declare_m_s']:.3f} m/s max, {result['mean_m_s']:.3f} "
                     f"mean, spread {result['spread_m_s']:.3f}"))
    body = [
        "- The declarable number is the **maximum**, not the mean: it is compared "
        "against a safety ceiling, and a mean hides the sample the ceiling has to "
        "survive.",
        "- The commanded magnitude was "
        f"{record.context.get('commanded_m_s'):.3f} m/s, which is "
        f"{COMMAND_MARGIN}x this profile's linear deadband. On a sign-only transport any "
        "value above the deadband emits the identical datagram, so it was not fitted to "
        "anything.",
        "- `measured_rad_s` is **not** measured here: the angular-velocity unit is still "
        "an open item on #13 and the yaw envelope stays unenforced.",
        "- ⚠️ This number is only half the gate. `_validate_axis_profile_speeds` compares "
        "it against `--max-vx x --derate`. That right-hand side used to default to the "
        "**Go2's** 0.35 m/s; `Lite3Bindings` now refuses a live run that does not state "
        "`--max-vx`/`--max-vy`/`--max-wz`, so state them, and state them from something "
        "other than this repository's Go2 numbers.",
        "- Measured on this robot only. Do not copy to the other Venture.",
        "",
        "Paste into this robot's axis profile:",
        "",
        "```json",
        measured_block(results),
        "```",
    ]
    return paste_block("Lite3 axis primitive speeds", rows, notes + body)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _validate(args)
    profile = robot_link.load_axis_profile(args)
    present = primitives_in(profile)
    deadband = profile.linear_deadband_m_s
    command = deadband * COMMAND_MARGIN

    brief(
        "Lite3 axis primitives -- the speed each evidenced direction actually delivers",
        does=f"""
        Commands each of {len(present)} profile primitive(s) {args.repeats} times for
        {args.segment:.1f} s, with a zero-command control between every one, and measures
        what the body did on that primitive's OWN axis. Every treatment is commanded at
        {command:.3f} m/s, which is {COMMAND_MARGIN}x this profile's linear deadband --
        on this transport any value above the deadband sends the identical datagram, so
        the commanded number is not part of the answer.
        Displacement comes from the robot's POSE, never from its own velocity estimate.
        """,
        needs=[
            f"a clear lane {args.lane_metres or '?'} m long and "
            f"{args.lane_width_metres or '?'} m wide, clear on BOTH SIDES -- this robot "
            f"has no lateral sensing at all",
            "the robot STANDING, in the vendor's moving/AI state, handed over to this "
            "laptop",
            "your hand on the emergency stop for the whole run",
            "an axis profile whose raw values each already carry an evidence reference",
            "room BEHIND the robot too if this profile carries forward_negative -- "
            "the backwards primitive is measured the same way as the forward one",
        ],
        means="""
        Each number goes into this profile's measured_m_s, and that field is what the
        live navigator's envelope gate is enforced against. Until it is measured, that
        gate is checking a number somebody typed.
        A primitive that moves the robot the wrong way is REFUSED, not recorded: the
        lateral pair inverts on the way to the wire and a swapped profile strafes into
        the side of the lane you cleared least.
        """,
        moves=True,
    )

    for primitive in present:
        _check_room(primitive, plan_for(primitive, deadband, args.repeats), args, print)

    if not args.live:
        print("")
        print("[axis] DRY RUN. Nothing was opened, nothing was commanded, and no socket "
              "exists in this process.")
        print(f"[axis] Primitives this profile carries: "
              f"{', '.join(primitive.name for primitive in present)}.")
        print("[axis] Re-read the plan above. When the lane is clear on all four sides "
              "and the robot is standing in moving/AI state, add --live "
              "--operator-ready.")
        return 0

    record = new_record(args.robot_id, **record_context(args, profile, present))

    link = robot_link.connect(args)
    loco = link.locomotion
    results: dict = {}
    try:
        record.context["preflight"] = robot_link.preflight(link, args)
        loco.prepare_motion()
        for index, primitive in enumerate(present):
            if index and not args.no_prompt:
                _reposition(primitive)
            plan = plan_for(primitive, deadband, args.repeats)
            _check_room(primitive, plan, args, print)
            print(f"\n[axis] {primitive.name}, {len(plan)} segments")
            segments = execute(loco, plan, segment_s=args.segment, tick_s=args.tick)
            results[primitive.name] = analyse(segments, primitive, print)
    finally:
        loco.shutdown()

    for name, result in results.items():
        merge_measurement(record, f"axis_primitive_{name}", result)
    notes = compare_with_profile(results, profile)
    destination = Path(args.artefact or f"lite3-axis-primitives-{args.robot_id}.json")
    write_record(destination, record)
    print(f"\n[axis] artefact: {destination.resolve()}  (provenance: {record.provenance})")
    print_paste_block(_paste(record, results, notes))
    return 0


def _reposition(primitive: Primitive) -> None:
    try:
        input(f"\n[axis] walk the robot back to the start of the lane, then press Enter "
              f"to measure {primitive.name} (it travels {primitive.travels}) "
              f"(Ctrl-C to stop): ")
    except EOFError:
        raise Refusal(
            "this probe pauses between primitives so you can walk the robot back to the "
            "start of the lane, and there is nobody at the keyboard to answer. Run it "
            "from a terminal, or pass --no-prompt only if the lane is long enough for "
            "every primitive back to back -- including the backwards one."
        ) from None


def _profile_sha256(path) -> str:
    """The hash of the profile these numbers belong to, so they cannot be re-attached."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise Refusal(f"cannot hash axis profile {path}: {error}") from None


if __name__ == "__main__":
    raise SystemExit(run_main(lambda: main(), "axis"))
