#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Commission one Lite3 Venture: run the measurements in a safe order, in one command.

    # 1. everything that cannot move the robot (read-only + tape + marker calibration)
    python3 commission.py --robot-id LITE3-A --firmware V1.0.8 --payload none \\
        --front 0.42 --back 0.38 --left 0.24 --right 0.24 --stance-confirmed \\
        --camera-source 0 --marker 1.50 --marker-size 0.15 \\
        --lens-height 0.31 --lens-height-source 'tape, standing, floor to lens centre'

    # 2. the two that walk. Lane clear, robot standing, your hand on the stop
    python3 commission.py ... --live --operator-ready \\
        --ladder-top 0.50 --lateral-top 0.30 --lane-metres 6.0 --lane-width-metres 2.0 \\
        --envelope-vx 0.35

    # 3. a human reads the numbers and signs for them
    python3 commission.py --record lite3-commissioning-LITE3-A.json --review 'Your Name'

    # 4. only now can the flags for a live run be produced
    python3 commission.py --record lite3-commissioning-LITE3-A.json --emit-flags

THE ORDER IS THE SAFETY ARGUMENT. Motor temperatures and the state link are read-only.
The radius is a tape measure. A marker calibration opens a camera and nothing else. Only
then does anything walk -- and the gait floor is measured before the actuator gain,
because the gain is only claimed above the floor and a gain fitted across it is dragged
down by every sub-floor point it swallowed.

WHY THE ARTEFACT STARTS ``provisional``. Because a number that has been measured and a
number that has been *believed* are different things, and only a person can turn one into
the other. ``--emit-flags`` is the only path from this artefact to a live run's
``--gait-floor``/``--actuator-gain``/``--robot-radius``, and it refuses a provisional
record. This is the same shape as ``Lite3Bindings.validate_camera_calibration``, which
stops a run rather than warning when a calibration file is not what it claims to be.

Nothing here invents a value. Every stage refuses rather than defaulting, and the driver
stops at the first refusal instead of carrying on with a hole in the record.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_robotics.lite3.commissioning import (
    actuator_gain_probe,
    camera_calibration,
    gait_floor_probe,
    loaded_radius_probe,
    motor_temperature_probe,
)
from deep_robotics.lite3.commissioning.measurement import (
    REVIEWED,
    Refusal,
    merge_measurement,
    new_record,
    paste_block,
    print_paste_block,
    read_record,
    require_reviewed,
    run_main,
    utc_now,
    write_record,
)

#: Stages in the order they may be run. Read-only first, tape second, camera third, and
#: the two that move the robot last. Changing this order is a safety decision.
STAGES = ("temperatures", "radius", "camera", "gait", "gain")

#: Which stages command a leg. The driver refuses these without --live.
MOVING_STAGES = frozenset(("gait", "gain"))

#: issue #13 checkbox -> the measurement key that closes it.
ISSUE_ITEMS = (
    ("Loaded plan-view planning radius, including legs and event payload",
     "loaded_radius"),
    ("Lowest forward command that sustains a gait (`--gait-floor`)",
     "gait_floor_forward"),
    ("Achieved/commanded velocity ratio (`--actuator-gain`)", "actuator_gain"),
    ("Lite3-tagged focal length/HFOV calibration", "camera_calibration"),
    ("Derive `--policy-scale` as loaded radius / 0.10 m", "loaded_radius"),
    ("High-level health bridge: exactly 12 Celsius motor temperatures",
     "motor_temperatures"),
)


def stage_argv(stage: str, args, gait_floor=None) -> list:
    """Build the exact argument list for one stage's own parser.

    The stages are invoked through their own parsers rather than by reaching into their
    internals, so a stage's refusal is the same refusal an operator would get running it
    by hand -- and the command printed above each stage is one they can re-run verbatim.
    """
    common = ["--robot-id", args.robot_id, "--firmware", args.firmware,
              "--payload", args.payload]
    artefact = ["--artefact", f"{_stage_artefact(args, stage)}"]
    if stage == "temperatures":
        return [*common, "--seconds", f"{args.temps_seconds}",
                "--state-port", f"{args.state_port}", *artefact]
    if stage == "radius":
        return (common
                + _value("--front", args.front) + _value("--back", args.back)
                + _value("--left", args.left) + _value("--right", args.right)
                + _flag("--stance-confirmed", args.stance_confirmed)
                + _flag("--asymmetric-confirmed", args.asymmetric_confirmed)
                + artefact)
    if stage == "camera":
        return (common
                + _value("--camera-source", args.camera_source)
                + _value("--lens-height", args.lens_height)
                + _value("--lens-height-source", args.lens_height_source)
                + _value("--marker", args.marker)
                + _value("--marker-size", args.marker_size)
                + _flag("--camera-gstreamer", args.camera_gstreamer)
                + _flag("--run", args.camera_run)
                + artefact)
    if stage == "gait":
        return (common + _link_argv(args)
                + _value("--ladder-top", args.ladder_top)
                + _value("--lateral-top", args.lateral_top)
                + _value("--rungs", args.rungs) + _value("--segment", args.segment)
                + _value("--lane-metres", args.lane_metres)
                + _value("--lane-width-metres", args.lane_width_metres)
                + _flag("--no-prompt", args.no_prompt)
                + artefact + _authority_argv(args))
    return (common + _link_argv(args)
            + _value("--gait-floor", gait_floor)
            + _value("--envelope-vx", args.envelope_vx)
            + _value("--segment", args.gain_segment)
            + _value("--repeats", args.gain_repeats)
            + _value("--lane-metres", args.lane_metres)
            + artefact + _authority_argv(args))


def _value(flag: str, value) -> list:
    """One flag, or nothing at all when the value was never supplied.

    Passing ``--front None`` through would make argparse complain about a float it cannot
    parse, and the operator would read an argparse error instead of the probe's own
    "measure this on this robot, there is no default" refusal. The refusals are the point;
    burying one under a type error would waste them.
    """
    return [] if value is None else [flag, str(value)]


def _flag(flag: str, enabled) -> list:
    return [flag] if enabled else []


def _link_argv(args) -> list:
    return ["--motion-host", args.motion_host, "--command-port", f"{args.command_port}",
            "--state-port", f"{args.state_port}"]


def _authority_argv(args) -> list:
    argv = []
    if args.live:
        argv.append("--live")
    if args.operator_ready:
        argv.append("--operator-ready")
    return argv


def _stage_artefact(args, stage: str) -> Path:
    return Path(args.out).parent / f"{Path(args.out).stem}-{stage}.json"


STAGE_ENTRY = {
    "temperatures": motor_temperature_probe,
    "radius": loaded_radius_probe,
    "camera": camera_calibration,
    "gait": gait_floor_probe,
    "gain": actuator_gain_probe,
}


def gait_floor_from(record) -> float:
    """The conservative floor the gait stage measured, or refuse to run the gain stage.

    A gain fitted from an unbracketed floor is a gain fitted from a guess, and the whole
    point of running the floor first is that the gain is only claimed above it.
    """
    forward = record.measurements.get("gait_floor_forward")
    if not forward:
        raise Refusal("the gait stage produced no forward result, so the gain stage has "
                      "no floor to start from")
    floor = forward.get("conservative_m_s")
    if floor is None:
        raise Refusal(
            "the gait ladder did not bracket a floor -- " + str(forward.get("note")) +
            " The actuator gain must not be fitted across an unknown floor, so the gain "
            "stage is not being run."
        )
    return float(floor)


def merge_stage_artefacts(record, paths, printer=print):
    """Fold each stage's own artefact into the single commissioning record."""
    for path in paths:
        source = Path(path)
        if not source.exists():
            continue
        stage_record = read_record(source)
        if stage_record.robot_id != record.robot_id:
            raise Refusal(
                f"{source} was measured on {stage_record.robot_id!r} but this record is "
                f"for {record.robot_id!r}. Numbers do not transfer between the two "
                f"Ventures; that is the whole premise of issue #13."
            )
        for name, payload in stage_record.measurements.items():
            merge_measurement(record, name, payload)
            printer(f"[commission] merged {name} from {source.name}")
    return record


def issue_status(record) -> list:
    """Which issue #13 per-robot checkboxes this record closes, and which it does not."""
    rows = []
    for label, key in ISSUE_ITEMS:
        payload = record.measurements.get(key)
        if payload is None:
            rows.append((label, "not measured", None))
        elif key == "motor_temperatures" and payload.get("channel") != "present":
            rows.append((label, "measured ABSENT -- vendor question, with evidence",
                         payload))
        else:
            rows.append((label, "closed", payload))
    return rows


def live_flags(record) -> list:
    """The flags a live run needs, assembled from a REVIEWED record only.

    Refuses a hole rather than filling one. Every one of these has a live pre-flight in
    ``Lite3Bindings`` that would refuse it as missing anyway; producing a partial line
    here would just move the refusal to the robot's feet.
    """
    radius = record.measurements.get("loaded_radius")
    floor = record.measurements.get("gait_floor_forward")
    gain = record.measurements.get("actuator_gain")
    camera = record.measurements.get("camera_calibration")
    missing = [name for name, value in
               (("loaded_radius", radius), ("gait_floor_forward", floor),
                ("actuator_gain", gain), ("camera_calibration", camera))
               if not value]
    if missing:
        raise Refusal(
            "this record cannot produce live-run flags; it is missing " +
            ", ".join(missing) + ". Run those stages before asking for the flags."
        )
    conservative = floor.get("conservative_m_s")
    if conservative is None:
        raise Refusal("the gait ladder never bracketed a floor, so there is no "
                      "--gait-floor to emit")
    return [
        "--calibration", str(camera["calibration_path"]),
        "--gait-floor", f"{conservative:.3f}",
        "--actuator-gain", f"{gain['pose_fit']['gain']:.3f}",
        "--robot-radius", f"{radius['radius_m']:.3f}",
        "--policy-scale", f"{radius['policy_scale']:.2f}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None,
                        help="the single commissioning artefact "
                             "(default: lite3-commissioning-<robot-id>.json)")
    parser.add_argument("--record", default=None,
                        help="an existing artefact, for --review / --emit-flags / --status")

    modes = parser.add_argument_group("record management (no robot contact)")
    modes.add_argument("--review", metavar="NAME", default=None,
                       help="mark --record reviewed, in this person's name. Read the "
                            "numbers first; this is the signature, not a formality")
    modes.add_argument("--emit-flags", action="store_true",
                       help="print the live-run flags. Refuses a provisional record")
    modes.add_argument("--status", action="store_true",
                       help="which issue #13 checkboxes --record closes")
    modes.add_argument("--merge", nargs="+", metavar="ARTEFACT", default=None,
                       help="fold artefacts written by the individual probes into one "
                            "record at --out. Use this when the tasks were run one at a "
                            "time rather than through this driver")

    context = parser.add_argument_group("issue #13 context")
    context.add_argument("--robot-id", default=None)
    context.add_argument("--firmware", default=None)
    context.add_argument("--payload", default=None)

    control = parser.add_argument_group("what to run")
    control.add_argument("--stage", action="append", choices=STAGES, default=None,
                         help="run one stage; repeatable. Default: all five, in the safe "
                              "order")
    control.add_argument("--live", action="store_true",
                         help="DANGER: permit the two stages that walk the robot")
    control.add_argument("--operator-ready", action="store_true")
    control.add_argument("--motion-host", default="192.168.1.120")
    control.add_argument("--command-port", type=int, default=43893)
    control.add_argument("--state-port", type=int, default=43897)

    temps = parser.add_argument_group("motor temperatures")
    temps.add_argument("--temps-seconds", type=float, default=20.0)

    radius = parser.add_argument_group("loaded radius (tape, standing, loaded)")
    for name in ("front", "back", "left", "right"):
        radius.add_argument(f"--{name}", type=float, default=None, metavar="M")
    radius.add_argument("--stance-confirmed", action="store_true")
    radius.add_argument("--asymmetric-confirmed", action="store_true")

    camera = parser.add_argument_group("camera calibration")
    camera.add_argument("--camera-source", default=None)
    camera.add_argument("--camera-gstreamer", action="store_true")
    camera.add_argument("--marker", type=float, default=None, metavar="DISTANCE_M")
    camera.add_argument("--marker-size", type=float, default=None, metavar="M")
    camera.add_argument("--lens-height", type=float, default=None, metavar="M")
    camera.add_argument("--lens-height-source", default=None)
    camera.add_argument("--camera-run", action="store_true",
                        help="actually open the camera. Without it the camera stage is "
                             "a dry run, like every other stage without its own gate")

    gait = parser.add_argument_group("gait floor")
    gait.add_argument("--ladder-top", type=float, default=None, metavar="M_S")
    gait.add_argument("--lateral-top", type=float, default=None, metavar="M_S")
    gait.add_argument("--rungs", type=int, default=6)
    gait.add_argument("--segment", type=float, default=1.0)
    gait.add_argument("--lane-metres", type=float, default=None)
    gait.add_argument("--lane-width-metres", type=float, default=None)
    gait.add_argument("--no-prompt", action="store_true")

    gain = parser.add_argument_group("actuator gain")
    gain.add_argument("--envelope-vx", type=float, default=None, metavar="M_S")
    gain.add_argument("--gain-segment", type=float, default=2.0)
    gain.add_argument("--gain-repeats", type=int, default=2)
    return parser


def _review(args, printer=print) -> int:
    record = read_record(args.record)
    if not args.review.strip():
        raise Refusal("--review needs a name. An unsigned review is not a review.")
    printer("")
    for label, state, _payload in issue_status(record):
        printer(f"  [{'x' if state == 'closed' else ' '}] {label} -- {state}")
    record.provenance = REVIEWED
    record.reviewed_by = args.review.strip()
    record.reviewed_utc = utc_now()
    write_record(args.record, record)
    printer(f"\n[commission] {args.record} is now {REVIEWED} in the name of "
            f"{record.reviewed_by}.")
    printer("[commission] --emit-flags will now produce live-run flags from it.")
    return 0


def _status(args, printer=print) -> int:
    record = read_record(args.record)
    printer("")
    printer(f"{record.robot_id}  provenance={record.provenance}"
            + (f"  reviewed by {record.reviewed_by}" if record.reviewed_by else ""))
    rows = []
    for label, state, _payload in issue_status(record):
        printer(f"  [{'x' if state == 'closed' else ' '}] {label} -- {state}")
        rows.append((label, state))
    print_paste_block(paste_block(
        f"Lite3 commissioning status -- {record.robot_id}", rows,
        [f"- Artefact provenance: **{record.provenance}**"
         + (f", reviewed by {record.reviewed_by} at {record.reviewed_utc}"
            if record.reviewed_by else
            ". Nothing marked provisional may be used for live movement.")]),
        printer=printer)
    return 0


def _emit_flags(args, printer=print) -> int:
    record = require_reviewed(args.record)
    printer(" ".join(live_flags(record)))
    return 0


def _run_stages(args, printer=print) -> int:
    missing = [name for name, value in (("--robot-id", args.robot_id),
                                        ("--firmware", args.firmware),
                                        ("--payload", args.payload))
               if not value]
    if missing:
        raise Refusal("running stages needs " + ", ".join(missing) +
                      "; issue #13 requires them beside every number")
    args.out = args.out or f"lite3-commissioning-{args.robot_id}.json"
    stages = [stage for stage in STAGES if stage in (args.stage or STAGES)]
    blocked = [stage for stage in stages if stage in MOVING_STAGES and not args.live]
    record = new_record(args.robot_id, firmware=args.firmware, payload=args.payload,
                        stages=stages, live=bool(args.live))

    gait_floor = None
    artefacts = []
    for stage in stages:
        module = STAGE_ENTRY[stage]
        if stage == "gain" and gait_floor is None:
            if "gait" in stages and args.live:
                raise Refusal("the gait stage did not produce a floor, so the gain "
                              "stage cannot start; see its output above")
            printer("\n[commission] SKIPPING the gain stage: it needs a measured gait "
                    "floor and this run has not produced one. Run the gait stage --live "
                    "first, or run actuator_gain_probe.py by hand with a --gait-floor "
                    "you have measured.")
            continue
        argv = stage_argv(stage, args, gait_floor)
        printer("")
        printer("#" * 78)
        printer(f"# stage {stage}"
                + ("   (MOVES THE ROBOT)" if stage in MOVING_STAGES else "")
                + ("   -- dry, no --live" if stage in blocked else ""))
        printer(f"# python3 {Path(module.__file__).name} {' '.join(argv)}")
        printer("#" * 78)
        code = module.main(argv)
        if code != 0:
            raise Refusal(f"the {stage} stage exited {code}; stopping here rather than "
                          f"writing a record with a hole in it")
        artefacts.append(_stage_artefact(args, stage))
        if stage == "gait" and args.live:
            gait_floor = gait_floor_from(read_record(_stage_artefact(args, stage)))
            printer(f"[commission] the gain stage will start from the measured floor "
                    f"{gait_floor:.3f} m/s")

    merge_stage_artefacts(record, artefacts, printer=printer)
    write_record(args.out, record)
    printer("")
    printer(f"[commission] artefact: {Path(args.out).resolve()}")
    printer(f"[commission] provenance: {record.provenance} -- nothing here may be used "
            f"for live movement yet.")
    printer(f"[commission] read the numbers, then: python3 commission.py --record "
            f"{args.out} --review 'Your Name'")
    args.record = args.out
    return _status(args, printer=printer)


def _merge(args, printer=print) -> int:
    """Fold standalone probe artefacts into the single record, then report on it.

    The probes are usable one at a time -- that is how the runbook has an operator work
    through them, one measurement per setup change -- so the combined record has to be
    constructible after the fact as well as by this driver.
    """
    if not args.robot_id:
        raise Refusal(
            "--merge needs --robot-id, so that an artefact measured on the other Venture "
            "is refused rather than quietly folded in"
        )
    args.out = args.out or f"lite3-commissioning-{args.robot_id}.json"
    record = new_record(args.robot_id, firmware=args.firmware, payload=args.payload,
                        merged_from=[str(path) for path in args.merge])
    missing = [path for path in args.merge if not Path(path).exists()]
    if missing:
        raise Refusal("these artefacts do not exist: " + ", ".join(missing))
    merge_stage_artefacts(record, args.merge, printer=printer)
    if not record.measurements:
        raise Refusal("none of those artefacts carried a measurement")
    write_record(args.out, record)
    printer(f"\n[commission] artefact: {Path(args.out).resolve()}")
    printer(f"[commission] provenance: {record.provenance} -- nothing here may be used "
            f"for live movement yet.")
    args.record = args.out
    return _status(args, printer=printer)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.merge:
        return _merge(args)
    if args.review or args.emit_flags or args.status:
        if not args.record:
            raise Refusal("--review, --emit-flags and --status all need --record PATH")
        if args.review:
            return _review(args)
        if args.emit_flags:
            return _emit_flags(args)
        return _status(args)
    return _run_stages(args)


if __name__ == "__main__":
    raise SystemExit(run_main(lambda: main(), "commission"))
