#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""How much of what you ask for does THIS Lite3 actually deliver?

    python3 actuator_gain_probe.py --robot-id LITE3-A --firmware V1.0.8 --payload none \\
        --gait-floor 0.25 --envelope-vx 0.35 --lane-metres 6.0
    # prints the plan and exits. Add --live --operator-ready to actually walk.

Answers the issue #13 item "achieved/commanded velocity ratio at the exact demo envelope;
record it as ``--actuator-gain``". The result is a fitted ratio with its residual, not a
single division, because one segment's ratio is one sample of a noisy quantity and the
number gets multiplied into every ``--max-seconds`` budget downstream.

⚠️ THE FIT IS AGAINST THE POSE DERIVATIVE, NOT AGAINST THE PLATFORM'S OWN VELOCITY
ESTIMATE. On the G1 that distinction cost a whole evening: a real 0.45 actuator gain was
read as 0.17 m/s of velocity-estimate noise, and a parked robot then random-walked toward
its goal because the controller believed the estimate. This probe records the platform's
estimate alongside, fits it separately, and **prints both** -- so if the two disagree, you
find that out here, at walking pace on a clear lane, rather than in a live run. The number
this probe ships is always the pose one.

WHY THE SAMPLES DO NOT SPAN THE GAIT FLOOR. Below the floor the robot does not walk, so
the delivered speed is zero and the relationship is not a ratio at all; a fit that
straddles the floor drags the gain down by however many sub-floor points it swallowed.
``robot-stack/deep_robotics/lite3/README.md`` says it directly -- "do not interpolate
across the gait floor" -- so every commanded speed here is at or above the measured floor
and the probe refuses if one is not.

WHERE ``--gait-floor`` COMES FROM. ``gait_floor_probe.py``, measured on this same robot.
There is no default. The Go2's number is not this robot's number.

Read ``../../../SAFETY.md``. ``--live`` is the only flag that moves a leg.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_robotics.lite3.commissioning import robot_link
from deep_robotics.lite3.commissioning.measurement import (
    Refusal,
    brief,
    check_controls_are_still,
    check_every_segment_walked,
    fit_ratio,
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

TICK_S = 0.1
SEGMENT_S = 2.0

#: Commanded speeds sampled between the floor and the demo envelope, inclusive of both.
POINTS = 3

#: Repeats of the whole sweep. Two is the minimum that can show a trend across the run --
#: a warm robot walks differently from a cold one -- and the controls are interleaved
#: rather than blocked so that trend cannot masquerade as a gain.
REPEATS = 2

#: How far the platform's own velocity estimate may disagree with the pose derivative
#: before the report calls it out. Not a tolerance on the robot: a tolerance on whether
#: the two instruments are describing the same event at all.
ESTIMATOR_DISAGREEMENT = 0.15


def sample_speeds(gait_floor: float, envelope_vx: float, points: int) -> list:
    """``points`` commanded speeds from the floor up to the demo envelope, inclusive.

    Anchored at both ends on purpose. The floor end is where the gain is least linear and
    the envelope end is the only speed the demo will actually use, so a fit that omits
    either is extrapolating into the part that matters.
    """
    if points < 2:
        raise Refusal("a gain needs at least two commanded speeds to be a fit rather "
                      "than a division")
    if envelope_vx < gait_floor:
        raise Refusal(
            f"--envelope-vx {envelope_vx:.3f} m/s is below the measured gait floor "
            f"{gait_floor:.3f} m/s. The demo would be commanding a speed this robot does "
            f"not walk at, which looks exactly like a broken tether and reports no fault."
        )
    if envelope_vx == gait_floor:
        return [gait_floor] * points
    span = envelope_vx - gait_floor
    return [gait_floor + span * index / (points - 1) for index in range(points)]


def plan_for(speeds, repeats: int) -> list:
    """``(role, vx)`` pairs: a control before every treatment, repeats interleaved.

    Interleaved, not blocked. Three of this project's gait conclusions were overturned by
    trial-order confounds, and a contemporaneous control is the only thing that separates
    the treatment from the drift.
    """
    if repeats < 1:
        raise Refusal("--repeats must be at least 1")
    plan = []
    for repeat in range(repeats):
        ordered = speeds if repeat % 2 == 0 else list(reversed(speeds))
        for vx in ordered:
            plan.append(("control", vx))
            plan.append(("treatment", vx))
    return plan


def planned_forward_metres(plan, segment_s: float) -> float:
    """Upper bound on lane consumed. Controls walk too -- they hold the commanded speed."""
    return sum(abs(vx) for _role, vx in plan) * segment_s


def measured_pairs(segments) -> list:
    """``(commanded, delivered)`` from the pose derivative, treatments only."""
    return [(segment.commanded_vx, segment.forward_mps)
            for segment in segments if segment.role == "treatment"]


def estimator_pairs(segments) -> list:
    """The same pairs read off the platform's own body-velocity estimate.

    Fitted separately and never merged with the pose fit. Two instruments that agree are
    one observation; two that disagree are a finding, and the only way to have either is
    to keep them apart.
    """
    return [(segment.commanded_vx, segment.estimator_forward_mps)
            for segment in segments
            if segment.role == "treatment" and math.isfinite(segment.estimator_forward_mps)]


def compare_estimators(pose_fit: dict, estimator_fit) -> str:
    """One sentence on whether the two instruments describe the same robot."""
    if estimator_fit is None:
        return ("the platform reported no usable body-velocity estimate during this run, "
                "so there is nothing to cross-check the pose fit against")
    difference = abs(pose_fit["gain"] - estimator_fit["gain"])
    if difference <= ESTIMATOR_DISAGREEMENT:
        return (f"the platform's own velocity estimate fits a gain of "
                f"{estimator_fit['gain']:.3f}, within {difference:.3f} of the pose fit. "
                f"The two instruments agree; the pose number still ships.")
    return (f"⚠️ the platform's own velocity estimate fits a gain of "
            f"{estimator_fit['gain']:.3f}, which is {difference:.3f} away from the pose "
            f"fit of {pose_fit['gain']:.3f}. They are not describing the same event. Ship "
            f"the pose number and do not let anything downstream servo on the estimate: "
            f"on the G1 that exact gap was a real gain read as velocity noise, and a "
            f"parked robot random-walked to its goal.")


def execute(loco, plan, *, segment_s: float, tick_s: float, printer=print,
            clock=time.monotonic, sleep=time.sleep) -> list:
    segments = []
    with stopped_afterwards(loco):
        for index, (role, vx) in enumerate(plan, start=1):
            segment = run_segment(loco, role=role, vx=vx, vy=0.0,
                                  duration_s=segment_s, tick_s=tick_s,
                                  clock=clock, sleep=sleep)
            segments.append(segment)
            printer(f"  [{index:>2}/{len(plan)}] {role:<9} vx={vx:.3f} -> pose "
                    f"{segment.forward_mps:+.3f} m/s  estimate "
                    f"{segment.estimator_forward_mps:+.3f} m/s")
    return segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    robot_link.add_context_arguments(parser)
    robot_link.add_link_arguments(parser, moving=True)

    envelope = parser.add_argument_group("the envelope (measure the floor first)")
    envelope.add_argument("--gait-floor", type=float, default=None, metavar="M_S",
                          help="this robot's measured forward gait floor, from "
                               "gait_floor_probe.py. No default")
    envelope.add_argument("--envelope-vx", type=float, default=None, metavar="M_S",
                          help="the top forward speed the demo will actually command. "
                               "The gain is only claimed over floor..envelope")
    envelope.add_argument("--points", type=int, default=POINTS,
                          help=f"commanded speeds between floor and envelope, inclusive "
                               f"(default: {POINTS})")
    envelope.add_argument("--repeats", type=int, default=REPEATS,
                          help=f"sweeps, alternating direction (default: {REPEATS})")
    envelope.add_argument("--segment", type=float, default=SEGMENT_S,
                          help=f"seconds per segment (default: {SEGMENT_S})")
    envelope.add_argument("--tick", type=float, default=TICK_S,
                          help=f"command period, seconds (default: {TICK_S})")

    room = parser.add_argument_group("the room")
    room.add_argument("--lane-metres", type=float, default=None,
                      help="clear length ahead of the robot, in metres")

    parser.add_argument("--artefact", default=None,
                        help="write the machine-readable record here "
                             "(default: lite3-actuator-gain-<robot-id>.json)")
    return parser


def _validate(args) -> None:
    robot_link.require_magnitude_transport(
        args,
        measures="an actuator gain -- the ratio of delivered speed to COMMANDED speed",
        instead="the delivered speed of each evidenced primitive, which "
                "axis_primitive_probe.py measures",
    )
    robot_link.require_walked_transport(args)
    refuse_unmeasured(**{"--gait-floor": args.gait_floor,
                         "--envelope-vx": args.envelope_vx,
                         "--lane-metres": args.lane_metres})
    require_positive_finite(**{"--gait-floor": args.gait_floor,
                               "--envelope-vx": args.envelope_vx,
                               "--segment": args.segment, "--tick": args.tick,
                               "--lane-metres": args.lane_metres})
    if args.tick >= args.segment:
        raise Refusal("--tick must be shorter than --segment")


def analyse(segments, printer=print) -> dict:
    """Refuse a run that did not walk, then fit. Both fits, reported side by side."""
    check_every_segment_walked(segments)
    check_controls_are_still(segments, "lateral")
    pose_fit = fit_ratio(measured_pairs(segments))
    estimator_samples = estimator_pairs(segments)
    estimator_fit = fit_ratio(estimator_samples) if estimator_samples else None

    printer("")
    printer(f"{'commanded':>10} {'pose m/s':>10} {'estimate m/s':>13} {'pose ratio':>11}")
    for segment in segments:
        if segment.role != "treatment":
            continue
        printer(f"{segment.commanded_vx:>10.3f} {segment.forward_mps:>10.3f} "
                f"{segment.estimator_forward_mps:>13.3f} "
                f"{segment.forward_mps / segment.commanded_vx:>11.3f}")
    printer("")
    printer(f"  --actuator-gain  {pose_fit['gain']:.3f}   "
            f"residual {pose_fit['residual_rms_m_s']:.4f} m/s RMS over "
            f"{pose_fit['samples']} segments")
    printer(f"  per-segment ratios {pose_fit['ratio_min']:.3f} .. "
            f"{pose_fit['ratio_max']:.3f} (median {pose_fit['ratio_median']:.3f})")
    printer("")
    printer("  " + compare_estimators(pose_fit, estimator_fit))
    return {"pose_fit": pose_fit, "estimator_fit": estimator_fit,
            "estimator_note": compare_estimators(pose_fit, estimator_fit),
            "segments": [segment.as_dict() for segment in segments]}


def _paste(record, result) -> str:
    pose = result["pose_fit"]
    context = record.context
    rows = [
        ("robot", record.robot_id),
        ("firmware", context.get("firmware")),
        ("payload", context.get("payload")),
        ("command envelope",
         f"{context.get('gait_floor_m_s')}..{context.get('envelope_vx_m_s')} m/s, "
         f"{context.get('segment_s')} s segments at "
         f"{1.0 / context.get('tick_s', 1.0):.0f} Hz"),
        ("`--actuator-gain`", f"{pose['gain']:.3f}"),
        ("residual", f"{pose['residual_rms_m_s']:.4f} m/s RMS over {pose['samples']} "
                     f"segments"),
        ("per-segment ratio range",
         f"{pose['ratio_min']:.3f} .. {pose['ratio_max']:.3f}"),
    ]
    notes = [
        "- Fitted against the **pose derivative**, through the origin. The platform's own "
        "body-velocity estimate was recorded and fitted separately: " +
        result["estimator_note"],
        "- Claimed only over the floor..envelope band above. Do not interpolate across "
        "the gait floor.",
        "- Measured on this robot only. Do not copy to the other Venture.",
    ]
    return paste_block("Lite3 actuator gain", rows, notes)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _validate(args)
    speeds = sample_speeds(args.gait_floor, args.envelope_vx, args.points)
    plan = plan_for(speeds, args.repeats)
    distance = planned_forward_metres(plan, args.segment)

    brief(
        "Lite3 actuator gain -- commanded versus delivered, fitted",
        does=f"""
        Holds each of {len(speeds)} commanded speeds
        ({', '.join(f'{value:.3f}' for value in speeds)} m/s) for {args.segment:.1f} s,
        {args.repeats} times, alternating sweep direction, with a contemporaneous control
        before every treatment. Fits delivered = gain x commanded through the origin from
        the POSE derivative, and fits the platform's own velocity estimate separately for
        comparison.
        """,
        needs=[
            f"a clear lane of at least {distance:.1f} m -- the robot walks the whole "
            f"sweep forward",
            "the robot STANDING, in the vendor's high-level navigation mode",
            "your hand on the emergency stop",
            "--gait-floor already measured on THIS robot with gait_floor_probe.py",
        ],
        means="""
        The gain multiplies every distance and duration budget downstream: at gain 0.6 a
        2 m waypoint takes 1.7x as long as the command implies. If the two fits disagree,
        the platform's velocity estimate is not trustworthy for control on this unit --
        that is a finding, not a nuisance.
        """,
        moves=True,
    )
    print(f"[gain] {len(plan)} segments, up to {distance:.1f} m of lane")
    if distance > args.lane_metres:
        raise Refusal(
            f"this sweep needs up to {distance:.1f} m and you have said the lane is "
            f"{args.lane_metres:.1f} m. Lower --repeats or --segment, or find a longer "
            f"lane."
        )

    if not args.live:
        print("")
        print("[gain] DRY RUN. Nothing was opened and nothing was commanded.")
        print("[gain] Add --live --operator-ready when the lane is clear.")
        return 0

    link = robot_link.connect(args)
    loco = link.locomotion
    try:
        record = new_record(
            args.robot_id, firmware=args.firmware, payload=args.payload,
            gait_floor_m_s=args.gait_floor, envelope_vx_m_s=args.envelope_vx,
            segment_s=args.segment, tick_s=args.tick, points=args.points,
            repeats=args.repeats, commanded_speeds=speeds,
            preflight=robot_link.preflight(link, args),
        )
        loco.prepare_motion()
        segments = execute(loco, plan, segment_s=args.segment, tick_s=args.tick)
    finally:
        loco.shutdown()

    result = analyse(segments)
    merge_measurement(record, "actuator_gain", result)
    destination = Path(args.artefact or f"lite3-actuator-gain-{args.robot_id}.json")
    write_record(destination, record)
    print(f"\n[gain] artefact: {destination.resolve()}  (provenance: {record.provenance})")
    print_paste_block(_paste(record, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(lambda: main(), "gain"))
