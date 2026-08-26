#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""How big is THIS Lite3, standing, with the event payload on it?

    python3 loaded_radius_probe.py --robot-id LITE3-A --firmware V1.0.8 \\
        --payload 'stock + 0.6 kg camera mast' \\
        --front 0.42 --back 0.38 --left 0.24 --right 0.24 --stance-confirmed

Answers the issue #13 items "loaded plan-view planning radius, including legs and event
payload" and "derive ``--policy-scale`` as loaded radius / 0.10 m".

THIS ONE IS A TAPE MEASURE AND THIS SCRIPT IS HONEST ABOUT THAT. The robot cannot measure
its own footprint: it has one forward RGB camera, no LiDAR, and nothing at all that sees
its own legs. Any script claiming to derive a loaded radius from the robot's own sensors
would be deriving it from a model of the robot, which is exactly the number in dispute.
So this is a recorder and a calculator: you measure four half-extents with a tape, it
turns them into the circumscribing radius the planner needs, derives the policy scale, and
writes the artefact with the payload description attached -- because a radius without the
payload it was measured with is not transferable to tomorrow's configuration, let alone to
the other robot.

MEASURE IT STANDING AND LOADED, AND THE SCRIPT WILL NOT LET YOU PRETEND OTHERWISE. A prone
Lite3 has a different plan-view outline from a standing one, and the Go2 corpus already
paid for this lesson in a different guise: a peer check that passed every gate with the
robot prone failed all of them at once the moment the robot stood, because standing moved
the camera 0.166 m and the geometry the run actually had was never the geometry that was
checked. ``--stance-confirmed`` is the acknowledgement that the outline on the floor is
the one the demo will have.

WHERE TO PUT THE TAPE. Stand the robot in its normal demo stance with the payload fitted.
Drop a plumb line from the centre of the body -- the point it turns about, not the centre
of the chassis casting if those differ -- and mark it. Then measure from that mark to the
furthest point of the robot in each of four directions, at floor level, **including the
legs at their widest point in the gait and anything bolted on**:

    --front   nose direction        --back    tail direction
    --left    port                  --right   starboard

THE RADIUS THIS PRODUCES. The planner treats the robot as a disc, so the honest radius is
the one that encloses the whole outline: the corner distance
``sqrt(max(front, back)^2 + max(left, right)^2)``, not the largest single extent. A robot
that is 0.42 m long and 0.24 m wide is not 0.42 m in radius; it is 0.48 m, and the 0.06 m
difference is the corner that clips the door frame.

Nothing here moves the robot. There is no ``--live`` because there is nothing to authorise.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_robotics.lite3.commissioning import robot_link
from deep_robotics.lite3.commissioning.measurement import (
    POLICY_AGENT_RADIUS_M,
    Refusal,
    brief,
    merge_measurement,
    new_record,
    paste_block,
    print_paste_block,
    refuse_unmeasured,
    require_positive_finite,
    run_main,
    write_record,
)

#: Ratio between the long and short half-extents above which the script asks you to look
#: again. This is a check on the METHOD, not a physical property of any robot: the usual
#: cause of a lopsided pair is a tape referenced to a body edge rather than to the
#: rotation centre, which halves one side and doubles the other. Override it deliberately
#: with --asymmetric-confirmed if the payload really does hang off one side.
ASYMMETRY_RATIO = 2.0


def circumscribing_radius(front: float, back: float, left: float, right: float) -> float:
    """Radius of the smallest disc, centred on the turning point, containing the outline.

    The outline is treated as the rectangle bounding the four measured extents, so the
    radius is its furthest corner. Using the largest single extent instead would leave
    every corner of the robot outside the disc the planner is avoiding obstacles with.
    """
    return math.hypot(max(front, back), max(left, right))


def policy_scale(radius_m: float) -> float:
    """``radius / 0.10`` -- the trained VMAS agent radius the checkpoint carries.

    0.10 describes the policy, not the robot, which is why it is the one number in this
    file that is allowed to be a constant.
    """
    return radius_m / POLICY_AGENT_RADIUS_M


def check_symmetry(front: float, back: float, left: float, right: float,
                   *, confirmed: bool, ratio: float = ASYMMETRY_RATIO) -> None:
    """Refuse a lopsided pair unless the operator says it is real."""
    if confirmed:
        return
    for name, (first, second) in (("front/back", (front, back)),
                                  ("left/right", (left, right))):
        larger, smaller = max(first, second), min(first, second)
        if smaller > 0 and larger / smaller >= ratio:
            raise Refusal(
                f"the {name} extents differ by {larger / smaller:.1f}x "
                f"({larger:.3f} m vs {smaller:.3f} m). The usual cause is a tape "
                f"referenced to an edge of the body rather than to the point the robot "
                f"turns about, which halves one side and doubles the other. Re-drop the "
                f"plumb line. If the payload genuinely does hang off one side, say so "
                f"with --asymmetric-confirmed and the number will be recorded as measured."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    robot_link.add_context_arguments(parser)

    tape = parser.add_argument_group("tape measurements, in metres, robot STANDING and LOADED")
    for name, help_text in (("front", "plumb line to the furthest point ahead"),
                            ("back", "plumb line to the furthest point behind"),
                            ("left", "plumb line to the furthest point to port"),
                            ("right", "plumb line to the furthest point to starboard")):
        tape.add_argument(f"--{name}", type=float, default=None, metavar="M",
                          help=f"{help_text}, including legs and payload")
    tape.add_argument("--stance-confirmed", action="store_true",
                      help="confirm the robot was STANDING in its demo stance with the "
                           "event payload fitted when you measured. A prone outline is "
                           "not the outline the run will have")
    tape.add_argument("--asymmetric-confirmed", action="store_true",
                      help="the robot really is lopsided; skip the symmetry check")

    parser.add_argument("--artefact", default=None,
                        help="write the machine-readable record here "
                             "(default: lite3-loaded-radius-<robot-id>.json)")
    return parser


def _paste(record, result) -> str:
    rows = [
        ("robot", record.robot_id),
        ("firmware", record.context.get("firmware")),
        ("payload", record.context.get("payload")),
        ("front / back / left / right",
         " / ".join(f"{result['extents_m'][key]:.3f}"
                    for key in ("front", "back", "left", "right")) + " m"),
        ("`--robot-radius`", f"{result['radius_m']:.3f} m"),
        ("`--policy-scale` (radius / 0.10 m)", f"{result['policy_scale']:.2f}"),
    ]
    notes = [
        "- Circumscribing radius about the turning point, not the largest single extent: "
        "the corner is what clips the door frame.",
        "- Measured **standing, with the payload fitted**. A radius measured prone or "
        "unloaded describes a geometry the run will never have.",
        "- Measured on this robot with this payload only. Change the payload and this "
        "number expires.",
    ]
    return paste_block("Lite3 loaded planning radius", rows, notes)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    brief(
        "Lite3 loaded planning radius -- how much room this robot needs",
        does="""
        Turns four tape measurements into the circumscribing radius the planner uses, and
        derives --policy-scale from it. Nothing is read from the robot and nothing moves.
        """,
        needs=[
            "the robot STANDING in its demo stance, with the event payload fitted",
            "a plumb line dropped from the point the robot turns about, marked on the floor",
            "four tape measurements from that mark to the furthest point of the robot "
            "ahead, behind, left and right -- at floor level, legs at their widest",
            "the payload written down, because this number expires when the payload changes",
        ],
        means="""
        --robot-radius is the disc the planner keeps clear of every obstacle, and
        --policy-scale is that radius divided by the 0.10 m agent the checkpoint was
        trained with. Both are wrong, in the direction that collides, if the tape was on a
        prone or unloaded robot.
        """,
        moves=False,
    )
    refuse_unmeasured(**{"--front": args.front, "--back": args.back,
                         "--left": args.left, "--right": args.right})
    require_positive_finite(**{"--front": args.front, "--back": args.back,
                               "--left": args.left, "--right": args.right})
    if not args.stance_confirmed:
        raise Refusal(
            "pass --stance-confirmed. A plan-view outline measured on a prone or "
            "unloaded robot is not the outline the demo will have, and there is nothing "
            "in this data that would reveal the difference afterwards."
        )
    check_symmetry(args.front, args.back, args.left, args.right,
                   confirmed=args.asymmetric_confirmed)

    radius = circumscribing_radius(args.front, args.back, args.left, args.right)
    scale = policy_scale(radius)
    result = {
        "extents_m": {"front": args.front, "back": args.back,
                      "left": args.left, "right": args.right},
        "radius_m": radius,
        "policy_scale": scale,
        "policy_agent_radius_m": POLICY_AGENT_RADIUS_M,
        "method": "circumscribing radius about the turning point, standing and loaded",
        "stance_confirmed": True,
        "asymmetric_confirmed": args.asymmetric_confirmed,
    }
    print("")
    print(f"  --robot-radius   {radius:.3f} m")
    print(f"  --policy-scale   {scale:.2f}   (= {radius:.3f} / "
          f"{POLICY_AGENT_RADIUS_M:.2f} m trained agent radius)")
    print(f"  largest single extent was {max(args.front, args.back, args.left, args.right):.3f} m; "
          f"the corner adds {radius - max(args.front, args.back, args.left, args.right):.3f} m")

    record = new_record(args.robot_id, firmware=args.firmware, payload=args.payload)
    merge_measurement(record, "loaded_radius", result)
    destination = Path(args.artefact or f"lite3-loaded-radius-{args.robot_id}.json")
    write_record(destination, record)
    print(f"\n[radius] artefact: {destination.resolve()}  "
          f"(provenance: {record.provenance})")
    print_paste_block(_paste(record, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(lambda: main(), "radius"))
