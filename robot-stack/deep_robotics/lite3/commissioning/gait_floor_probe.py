#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""What is the lowest command that actually walks THIS Lite3, forward and sideways?

    python3 gait_floor_probe.py --robot-id LITE3-A --firmware V1.0.8 --payload none \\
        --ladder-top 0.50 --lateral-top 0.30 --lane-metres 6.0 --lane-width-metres 2.0
    # prints the plan and exits. Add --live --operator-ready to actually walk.

Answers the issue #13 item "lowest forward command that sustains a gait; use the
conservative working value as ``--gait-floor``", and it answers it for one named robot.

THERE IS NO DEFAULT HERE AND THERE IS NOT GOING TO BE ONE. The Go2's 0.35 m/s forward and
0.20 m/s lateral floors are measurements of a different robot with different legs, a
different mass and a different vendor gait controller. They are not in this file, they are
not fallbacks in this file, and a Lite3 run that has not measured its own floor refuses
rather than borrowing one.

TWO THINGS THE GO2 LEARNED THE HARD WAY, BOTH CARRIED HERE.

**A lateral floor is not a floor on a diagonal.** The Go2's 0.20 m/s lateral floor was
measured as a PURE strafe from standstill, and every design decision after it treated 0.20
as a hard floor on the lateral axis -- which produced a rule that a command had to be
nearly 30 degrees off the nose before any sideways travel happened at all. Then a robot
already walking forward delivered lateral travel proportionally from 0.05 m/s upward, and
the whole 30-degree argument dissolved. So this probe measures **both** cases and reports
them as two different numbers that must never be substituted for one another: a
``strafe`` phase from standstill, and a ``diagonal`` phase with the forward gait already
running at this robot's own measured forward floor.

**A probe that fails to stand reports 0.000 m/s on every axis, which reads exactly like a
floor that is real and total.** The Go2 probe once called only ``stand()`` where getting
up needs two calls; the robot stayed prone, every command was ignored, and the run
produced a beautiful table of zeros. So: the forward ladder carries **anchor** segments
commanded at the top of the ladder, a speed the operator has already seen this robot walk
at, and if an anchor does not travel the run is refused outright. The diagonal phase holds
forward velocity for every segment and gets the Go2's own refusal unmodified.

WHERE ``--ladder-top`` AND ``--lateral-top`` COME FROM. Not from a guess. Run
``lite3_state_probe.py`` first with the operator driving on the vendor remote: its
"Remote-commanded vs measured forward speed" table shows speeds this robot has already
been seen to walk at. Take one of those.

⚠️ THE ROBOT CRABS SIDEWAYS INTO SPACE IT CANNOT SEE. There is no lateral sensing on this
platform at all. Clear both sides of the lane, not just the ends.

Read ``../../../SAFETY.md``. ``--live`` is the only flag that moves a leg, and an operator
stays on the emergency stop for all of it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_robotics.lite3.commissioning import robot_link
from deep_robotics.lite3.commissioning.measurement import (
    WALKED_MARGIN_M,
    Refusal,
    brief,
    check_anchors_walked,
    check_controls_are_still,
    check_every_segment_walked,
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

PHASES = ("forward", "strafe", "diagonal")

#: Control period. Matches the navigation stack's 10 Hz, so the vendor gait sees the same
#: command cadence during the measurement that it will see on a real run. A probe that
#: commands at a different rate measures a different controller.
TICK_S = 0.1

#: Seconds per segment. Long enough that ``WALKED_MARGIN_M`` of displacement is a small
#: fraction of a walking segment's travel, short enough to fit a ladder in a corridor.
SEGMENT_S = 1.0

#: Rungs in each ladder. The step is ``top / rungs``, so this is the resolution of the
#: answer: 6 rungs off a 0.50 m/s top resolves the floor to about 0.08 m/s.
RUNGS = 6


def descending_rungs(top: float, rungs: int) -> list:
    """``rungs`` evenly spaced commands from ``top`` down to ``top / rungs``.

    Descending rather than ascending because the robot has to already be walking for the
    question to mean anything: an ascending ladder measures how hard it is to *start*,
    which is a different quantity and the one the strafe phase is for.
    """
    if rungs < 3:
        raise Refusal("a ladder needs at least 3 rungs to show whether it is monotonic")
    return [top * (index / rungs) for index in range(rungs, 0, -1)]


def forward_plan(top: float, rungs: int) -> list:
    """``(role, vx, vy)`` triples: anchors, and a control before every treatment."""
    ladder = descending_rungs(top, rungs)
    plan = [("anchor", top, 0.0)]
    for index, vx in enumerate(ladder):
        plan.append(("control", 0.0, 0.0))
        plan.append(("treatment", vx, 0.0))
        if index == len(ladder) // 2:
            # An anchor in the middle as well as at the ends: a robot that stops walking
            # part way through would otherwise pass both end anchors and silently turn
            # every rung after the failure into a false floor.
            plan.append(("anchor", top, 0.0))
    plan.append(("control", 0.0, 0.0))
    plan.append(("anchor", top, 0.0))
    return plan


def strafe_plan(lateral_top: float, rungs: int, anchor_vx: float) -> list:
    """Pure strafe from standstill, alternating sign so the robot returns to the middle.

    The anchors are forward segments, because the failure this whole harness is built
    around -- a robot that never stood up -- shows as zeros on the lateral axis too, and
    a lateral-only phase has nothing else that could catch it.
    """
    ladder = descending_rungs(lateral_top, rungs)
    plan = [("anchor", anchor_vx, 0.0)]
    for index, vy in enumerate(ladder):
        plan.append(("control", 0.0, 0.0))
        plan.append(("treatment", 0.0, vy if index % 2 == 0 else -vy))
    plan.append(("control", 0.0, 0.0))
    plan.append(("anchor", anchor_vx, 0.0))
    return plan


def diagonal_plan(forward_floor: float, lateral_top: float, rungs: int) -> list:
    """Forward velocity held for EVERY segment; ``vy`` is the only thing that changes.

    Copied in shape from the Go2's ``lateral_floor_probe.py`` with its one hard-coded
    number replaced by a measurement: forward is held at *this* robot's measured floor,
    not at 0.35 m/s. Dropping ``vx`` between treatments would confound "lateral does not
    execute" with "the robot stopped walking", which is the confound the Go2 probe was
    written to remove.
    """
    ladder = descending_rungs(lateral_top, rungs)
    plan = []
    for index, vy in enumerate(ladder):
        plan.append(("control", forward_floor, 0.0))
        plan.append(("treatment", forward_floor, vy if index % 2 == 0 else -vy))
    plan.append(("control", forward_floor, 0.0))
    return plan


def planned_forward_metres(plan, segment_s: float) -> float:
    """Upper bound on lane length consumed, assuming the robot delivers what it is asked.

    An over-estimate on purpose: a gain below 1 makes the real travel shorter, and the
    direction to be wrong in when sizing a corridor is the one that asks for more room.
    """
    return sum(abs(vx) for _role, vx, _vy in plan) * segment_s


def planned_lateral_excursion_m(plan, segment_s: float) -> float:
    """Worst-case sideways excursion from the lane centreline.

    The sign alternates, so the running sum returns toward zero; the excursion that
    matters is the furthest the sum ever gets, not where it ends up.
    """
    position = 0.0
    worst = 0.0
    for _role, _vx, vy in plan:
        position += vy * segment_s
        worst = max(worst, abs(position))
    return worst


def forward_floor_from(segments, step: float) -> dict:
    """Read a forward gait floor off the ladder, or refuse to name one.

    Three outcomes, and they are genuinely different:

    * the ladder brackets the floor -- some rung walked, a lower one did not;
    * every rung walked, so the floor is *below* the smallest rung tested and this run
      has not found it, only bounded it;
    * the walking rungs are not contiguous, which is a confound rather than a finding.
      A robot cannot walk at 0.20 and 0.40 but not at 0.30; something else changed.
    """
    rows = sorted((segment for segment in segments if segment.role == "treatment"),
                  key=lambda segment: segment.commanded_vx)
    if not rows:
        raise Refusal("the forward ladder recorded no treatment segments")
    walking = [row for row in rows if row.travelled]
    if not walking:
        raise Refusal(
            "no rung of the forward ladder travelled, but the anchors did. That is a "
            "contradiction -- the top rung and the anchors were commanded the same "
            "speed -- so something changed during the run. Do not read a floor out of "
            "this; repeat it."
        )
    first_walking = rows.index(walking[0])
    if any(not row.travelled for row in rows[first_walking:]):
        return {
            "bracketed": False, "monotonic": False,
            "lowest_walking_m_s": walking[0].commanded_vx,
            "conservative_m_s": None,
            "note": "non-monotonic ladder: a lower rung walked while a higher one did "
                    "not. That is a confound, not a floor. Find it before believing any "
                    "of these numbers.",
        }
    lowest = walking[0].commanded_vx
    if first_walking == 0:
        return {
            "bracketed": False, "monotonic": True,
            "lowest_walking_m_s": lowest, "conservative_m_s": lowest,
            "note": f"every rung walked, including the smallest tested ({lowest:.3f} "
                    f"m/s). The floor is BELOW that and this run has not found it. "
                    f"{lowest:.3f} m/s is a value known to walk, not a measured floor -- "
                    f"rerun with a lower --ladder-top to bracket it.",
        }
    return {
        "bracketed": True, "monotonic": True,
        "lowest_walking_m_s": lowest,
        "highest_still_m_s": rows[first_walking - 1].commanded_vx,
        "conservative_m_s": lowest + step,
        "note": f"the floor lies between {rows[first_walking - 1].commanded_vx:.3f} and "
                f"{lowest:.3f} m/s. Use the conservative value as --gait-floor: it is one "
                f"ladder step above the lowest command seen to walk, so the demo is not "
                f"sitting on the edge of the cliff.",
    }


def lateral_rows(segments) -> list:
    """Delivered lateral travel per commanded ``vy``, net of the contemporaneous control."""
    baseline = control_baseline(segments, "lateral")
    rows = []
    for segment in segments:
        if segment.role != "treatment" or segment.commanded_vy == 0.0:
            continue
        commanded = abs(segment.commanded_vy)
        # Signed against the command, so a robot that crabs the wrong way reports a
        # negative fraction rather than a plausible positive one.
        delivered = (segment.lateral_mps - baseline)
        signed = delivered if segment.commanded_vy > 0 else -delivered
        rows.append({"commanded_vy_m_s": commanded, "delivered_m_s": signed,
                     "fraction": signed / commanded,
                     "forward_mps": segment.forward_mps})
    rows.sort(key=lambda row: row["commanded_vy_m_s"])
    return rows


def lateral_verdict(rows, segment_s: float = SEGMENT_S) -> str:
    """Step, line, or confound -- the three shapes, named.

    ``segment_s`` is the actual segment length, because the "delivered essentially
    nothing" threshold is a SPEED derived from a displacement: a quarter of the speed at
    which a segment would just barely count as having travelled. Pinning it to the module
    default would leave the verdict silently mis-scaled whenever ``--segment`` is changed.
    """
    if len(rows) < 3:
        return "too few rungs to tell a step from a line"
    fractions = [row["fraction"] for row in rows]
    smallest = rows[0]
    if smallest["delivered_m_s"] < WALKED_MARGIN_M / segment_s / 4.0:
        return ("STEP-like: the smallest rung delivered essentially nothing, so a floor "
                "on this axis may be real. Check where it steps.")
    spread = max(fractions) - min(fractions)
    if spread < 0.25:
        return ("LINE through the origin: delivery is proportional from the smallest rung "
                "up, so there is NO floor on this axis in this condition.")
    return ("NEITHER: the fractions fan out by "
            f"{spread:.2f}. Non-monotonic is a confound, not a finding -- find it before "
            "believing these numbers.")


def execute(loco, plan, *, segment_s: float, tick_s: float, printer=print,
            clock=time.monotonic, sleep=time.sleep) -> list:
    """Walk the plan, printing each segment as it lands. Always ends stopped."""
    segments = []
    with stopped_afterwards(loco):
        for index, (role, vx, vy) in enumerate(plan, start=1):
            segment = run_segment(loco, role=role, vx=vx, vy=vy,
                                  duration_s=segment_s, tick_s=tick_s,
                                  clock=clock, sleep=sleep)
            segments.append(segment)
            printer(f"  [{index:>2}/{len(plan)}] {role:<9} vx={vx:+.3f} vy={vy:+.3f} "
                    f"-> forward {segment.forward_mps:+.3f} m/s  "
                    f"lateral {segment.lateral_mps:+.3f} m/s  "
                    f"yaw {segment.yaw_change_deg:+.1f} deg"
                    f"{'' if segment.travelled else '   (no forward travel)'}")
    return segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    robot_link.add_context_arguments(parser)
    robot_link.add_link_arguments(parser, moving=True)

    ladder = parser.add_argument_group("the ladder (measure these on the vendor remote first)")
    ladder.add_argument("--ladder-top", type=float, default=None, metavar="M_S",
                        help="a forward speed you have ALREADY watched this robot walk "
                             "at, from lite3_state_probe.py capture 2. No default: the "
                             "Go2's number is not this robot's number")
    ladder.add_argument("--lateral-top", type=float, default=None, metavar="M_S",
                        help="likewise for sideways. Required by the strafe and diagonal "
                             "phases")
    ladder.add_argument("--forward-floor", type=float, default=None, metavar="M_S",
                        help="measured forward floor for the diagonal phase. Omit it and "
                             "the forward phase's own result is used")
    ladder.add_argument("--rungs", type=int, default=RUNGS,
                        help=f"rungs per ladder; the step is top/rungs (default: {RUNGS})")
    ladder.add_argument("--segment", type=float, default=SEGMENT_S,
                        help=f"seconds per segment (default: {SEGMENT_S})")
    ladder.add_argument("--tick", type=float, default=TICK_S,
                        help=f"command period, seconds (default: {TICK_S}, the stack's "
                             f"10 Hz)")
    ladder.add_argument("--phase", action="append", choices=PHASES, default=None,
                        help="run one phase; repeatable. Default: all three, in order")

    room = parser.add_argument_group("the room (the human is the only sensor for this)")
    room.add_argument("--lane-metres", type=float, default=None,
                      help="clear length ahead of the robot, in metres")
    room.add_argument("--lane-width-metres", type=float, default=None,
                      help="clear width, EXCLUDING the robot's own footprint, in metres")
    room.add_argument("--no-prompt", action="store_true",
                      help="do not pause between phases to let you reposition the robot")

    parser.add_argument("--artefact", default=None,
                        help="write the machine-readable record here "
                             "(default: lite3-gait-floor-<robot-id>.json)")
    return parser


def _validate(args) -> None:
    robot_link.require_magnitude_transport(
        args,
        measures="a gait floor -- the lowest commanded speed that still walks",
        instead="the delivered speed of each evidenced primitive, which "
                "axis_primitive_probe.py measures",
    )
    robot_link.require_walked_transport(args)
    refuse_unmeasured(**{"--ladder-top": args.ladder_top,
                         "--lane-metres": args.lane_metres,
                         "--lane-width-metres": args.lane_width_metres})
    require_positive_finite(**{"--ladder-top": args.ladder_top,
                               "--segment": args.segment, "--tick": args.tick,
                               "--lane-metres": args.lane_metres,
                               "--lane-width-metres": args.lane_width_metres})
    if args.tick >= args.segment:
        raise Refusal("--tick must be shorter than --segment, or a segment carries at "
                      "most one command")
    phases = tuple(args.phase or PHASES)
    if "strafe" in phases or "diagonal" in phases:
        refuse_unmeasured(**{"--lateral-top": args.lateral_top})
        require_positive_finite(**{"--lateral-top": args.lateral_top})
    if "diagonal" in phases and "forward" not in phases:
        refuse_unmeasured(**{"--forward-floor": args.forward_floor})
        require_positive_finite(**{"--forward-floor": args.forward_floor})


def _plan_for(phase: str, args, forward_floor) -> list:
    if phase == "forward":
        return forward_plan(args.ladder_top, args.rungs)
    if phase == "strafe":
        return strafe_plan(args.lateral_top, args.rungs, args.ladder_top)
    if forward_floor is None:
        raise Refusal(
            "the diagonal phase needs a measured forward floor and the forward phase did "
            "not produce one. Fix the forward phase, or pass --forward-floor from a run "
            "that did."
        )
    return diagonal_plan(forward_floor, args.lateral_top, args.rungs)


def _check_room(phase: str, plan, args, printer) -> None:
    forward_m = planned_forward_metres(plan, args.segment)
    lateral_m = planned_lateral_excursion_m(plan, args.segment)
    printer(f"[{phase}] {len(plan)} segments x {args.segment:.1f}s: up to "
            f"{forward_m:.1f} m forward and {lateral_m:.1f} m to one side")
    if forward_m > args.lane_metres:
        raise Refusal(
            f"the {phase} phase needs up to {forward_m:.1f} m of lane and you have said "
            f"there is {args.lane_metres:.1f} m. Lower --rungs or --segment, or find a "
            f"longer lane. Nothing here will quietly shorten the run for you."
        )
    if 2.0 * lateral_m > args.lane_width_metres:
        raise Refusal(
            f"the {phase} phase swings up to {lateral_m:.1f} m to each side and you have "
            f"said there is {args.lane_width_metres:.1f} m of clear width. The robot has "
            f"no lateral sensing whatsoever; it will crab straight into whatever is "
            f"there."
        )


def _analyse(phase: str, segments, args, printer) -> dict:
    step = args.ladder_top / args.rungs
    if phase == "forward":
        check_anchors_walked(segments)
        check_controls_are_still(segments, "forward")
        result = forward_floor_from(segments, step)
        printer("")
        printer(f"{'commanded vx':>13} {'forward m/s':>12} {'walked':>8}")
        for segment in sorted((s for s in segments if s.role == "treatment"),
                              key=lambda s: -s.commanded_vx):
            printer(f"{segment.commanded_vx:>13.3f} {segment.forward_mps:>12.3f} "
                    f"{'yes' if segment.travelled else 'NO':>8}")
        printer("")
        printer("  " + result["note"])
        return result

    if phase == "strafe":
        check_anchors_walked(segments)
        check_controls_are_still(segments, "lateral")
    else:
        check_every_segment_walked(segments)
        check_controls_are_still(segments, "lateral")
    rows = lateral_rows(segments)
    printer("")
    printer(f"{'commanded vy':>13} {'delivered m/s':>14} {'delivered':>10} "
            f"{'forward m/s':>12}")
    for row in rows:
        printer(f"{row['commanded_vy_m_s']:>13.3f} {row['delivered_m_s']:>14.3f} "
                f"{row['fraction'] * 100:>9.0f}% {row['forward_mps']:>12.3f}")
    verdict = lateral_verdict(rows, args.segment)
    printer("")
    printer("  " + verdict)
    return {"rows": rows, "verdict": verdict}


def _paste(record, results) -> str:
    forward = results.get("forward") or {}
    rows = [
        ("robot", record.robot_id),
        ("firmware", record.context.get("firmware")),
        ("payload", record.context.get("payload")),
        ("command envelope", f"segment {record.context.get('segment_s')} s at "
                             f"{1.0 / record.context.get('tick_s', 1.0):.0f} Hz, ladder "
                             f"top {record.context.get('ladder_top_m_s')} m/s"),
    ]
    if forward:
        rows += [
            ("lowest forward command seen to walk",
             f"{forward.get('lowest_walking_m_s'):.3f} m/s"),
            ("conservative `--gait-floor`",
             "NOT BRACKETED" if forward.get("conservative_m_s") is None
             else f"{forward['conservative_m_s']:.3f} m/s"),
        ]
    notes = []
    for phase in ("strafe", "diagonal"):
        result = results.get(phase)
        if result:
            rows.append((f"lateral, {phase}", result["verdict"].split(":")[0]))
            notes.append(f"- **{phase}**: " + result["verdict"])
    if "strafe" in results and "diagonal" in results:
        notes.append("- These two are **not the same number and never substitute for one "
                     "another**. The strafe figure is a standing start; the diagonal "
                     "figure is a robot already in gait. On the Go2 the pure-strafe floor "
                     "did not describe a diagonal command at all.")
    notes.append("- Measured on this robot only. Do not copy to the other Venture.")
    return paste_block("Lite3 gait floor", rows, notes)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    phases = tuple(args.phase or PHASES)
    brief(
        "Lite3 gait floor -- the lowest command that actually walks THIS robot",
        does=f"""
        Walks a descending ladder of commanded speeds, {args.segment:.1f} s per rung,
        with a zero-command control between every rung and anchor segments at a speed
        you have already seen this robot walk at. Phases: {', '.join(phases)}.
        Displacement is measured from the robot's POSE, never from its own velocity
        estimate.
        """,
        needs=[
            f"a clear lane {args.lane_metres or '?'} m long and "
            f"{args.lane_width_metres or '?'} m wide, clear on BOTH SIDES -- this robot "
            f"has no lateral sensing at all",
            "the robot STANDING, in the vendor's high-level navigation mode, handed over "
            "to this laptop",
            "your hand on the emergency stop for the whole run",
            "--ladder-top and --lateral-top taken from a lite3_state_probe.py capture of "
            "you driving this robot on the vendor remote",
        ],
        means="""
        The conservative forward number is what goes into --gait-floor. Below it, a
        command can look exactly like a tether fault or a transport failure: the robot
        stands there reporting no error at all.
        The lateral numbers are TWO different measurements and neither one describes the
        other. Read the verdict line under each.
        """,
        moves=True,
    )
    _validate(args)

    record = new_record(
        args.robot_id, firmware=args.firmware, payload=args.payload,
        segment_s=args.segment, tick_s=args.tick, rungs=args.rungs,
        ladder_top_m_s=args.ladder_top, lateral_top_m_s=args.lateral_top,
        lane_metres=args.lane_metres, lane_width_metres=args.lane_width_metres,
        phases=list(phases),
    )

    forward_floor = args.forward_floor
    for phase in phases:
        if phase == "diagonal" and forward_floor is None:
            # The diagonal phase holds forward velocity at the MEASURED floor, and the
            # forward phase has not run yet, so its lane cost is genuinely not knowable
            # here. Refusing on the --ladder-top worst case would refuse almost every
            # real room -- 13 segments at the top of the ladder rather than at a floor a
            # third of it -- and a guard that refuses correct runs is a guard operators
            # learn to work around. So it is stated as a bound now and CHECKED for real
            # immediately before the phase runs, when the floor is known.
            worst = planned_forward_metres(
                _plan_for(phase, args, args.ladder_top), args.segment)
            print(f"[diagonal] lane cost is {len(_plan_for(phase, args, args.ladder_top))}"
                  f" x segment x the measured forward floor -- at most {worst:.1f} m if "
                  f"the floor turned out to be the whole ladder top. Checked for real "
                  f"once the forward phase has measured it.")
            continue
        _check_room(phase, _plan_for(phase, args, forward_floor), args, print)

    if not args.live:
        print("")
        print("[gait] DRY RUN. Nothing was opened, nothing was commanded, and no socket "
              "exists in this process.")
        print("[gait] Re-read the plan above. When the lane is clear and the robot is "
              "standing in navigation mode, add --live --operator-ready.")
        return 0

    link = robot_link.connect(args)
    loco = link.locomotion
    results = {}
    try:
        health = robot_link.preflight(link, args)
        record.context["preflight"] = health
        loco.prepare_motion()
        for index, phase in enumerate(phases):
            if index and not args.no_prompt:
                try:
                    input(f"\n[gait] reposition the robot at the start of the lane, "
                          f"then press Enter to begin the {phase} phase "
                          f"(Ctrl-C to stop): ")
                except EOFError:
                    raise Refusal(
                        "this probe pauses between phases so you can walk the robot back "
                        "to the start of the lane, and there is nobody at the keyboard "
                        "to answer. Run it from a terminal, or pass --no-prompt only if "
                        "the lane is long enough for every phase back to back."
                    ) from None
            plan = _plan_for(phase, args, forward_floor)
            # Re-checked here, not only at plan time: the diagonal phase's forward speed
            # is only known once the forward phase has measured it.
            _check_room(phase, plan, args, print)
            print(f"\n[gait] {phase} phase, {len(plan)} segments")
            segments = execute(loco, plan, segment_s=args.segment, tick_s=args.tick)
            results[phase] = _analyse(phase, segments, args, print)
            results[phase]["segments"] = [segment.as_dict() for segment in segments]
            if phase == "forward":
                forward_floor = results[phase].get("conservative_m_s") or forward_floor
    finally:
        loco.shutdown()

    for phase, result in results.items():
        merge_measurement(record, f"gait_floor_{phase}", result)
    destination = Path(args.artefact or f"lite3-gait-floor-{args.robot_id}.json")
    write_record(destination, record)
    print(f"\n[gait] artefact: {destination.resolve()}  (provenance: {record.provenance})")
    print_paste_block(_paste(record, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(lambda: main(), "gait"))
