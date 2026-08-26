#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lite3-tagged camera intrinsics: focal length, HFOV, and the lens height above the floor.

    python3 camera_calibration.py --robot-id LITE3-A --firmware V1.0.8 --payload none \\
        --camera-source 0 --marker 1.50 --lens-height 0.31 \\
        --lens-height-source 'tape, standing, floor to lens centre'
    # add --run to actually capture; without it this prints the delegated command

Answers the issue #13 item "Lite3-tagged focal length/HFOV calibration for the installed
RGB camera".

**THIS DOES NOT CONTAIN A SECOND CALIBRATION METHOD, AND THAT IS DELIBERATE.** The fitter
lives in ``robot-stack/unitree/go2/visual_nav/calibrate_camera.py`` and this script calls
it through ``Lite3Bindings``, exactly as ``visual_nav/calibrate_camera.py`` already does.
Two implementations of one intrinsic is how two robots end up with numbers that cannot be
compared. What this script adds is the three things that wrapper does not do, and each one
is a hole that has already cost somebody a run somewhere in this corpus:

**1. It asserts that the probe runs the same inference configuration production will.**
The Go2 corpus paid for this: a probe run at a 300-pixel detector input while production
ran at 224 set a detection threshold that described nothing at all, because the two were
not looking at the same image. The shared calibrator exposes no ``--input-size`` and no
``--confidence``, so its ``--object`` and ``--spin --spin-target object`` modes silently
use the detector's own defaults. This script reads those defaults out of the detector and
out of the production navigator's parser and refuses if they differ, rather than letting
the difference be discovered later as a threshold that describes nothing.

``--marker`` uses ArUco corner refinement and no neural detector at all, so it has no
inference configuration to mismatch. That is why it is the recommended mode here, and why
the assertion prints "not applicable" rather than passing quietly.

**2. It measures the lens height instead of inheriting the Go2's.**
Nothing on the shared fitting path asks for a lens height, so a Lite3-tagged calibration
file carries whatever ``FisheyeCamera.height_m`` happened to hold. What that is depends on
which copy of the shared tree the fit ran through, and neither answer is this robot's:

* where the field still carries its default, that default is **0.32 m** -- the height of
  the *Go2's* camera when the *Go2* stands. Checked 2026-08-26: the upstream per-robot Go2
  repository still declares ``height_m: float = 0.32`` in
  ``unitree/go2/visual_nav/camera_model.py``.
* where the default has been removed -- this repository's own copy, since #96, which names
  the constant ``GO2_CAMERA_HEIGHT_M`` and gives the field no default -- an unset height is
  written into the file as ``null`` rather than omitted.

Both arrive here, so both are reported, in words: ``describe_replaced`` renders an
inherited ``0.32`` and an unset ``null`` differently because they tell the next reader
different things. They are not interchangeable and neither is a Lite3 measurement.

It is not overlay-only: ``person_detector.object_fit_range`` uses the height to bound a
width-derived range, so a wrong lens height moves a range bound on a real robot. This
script requires ``--lens-height`` and ``--lens-height-source`` either way, and rewrites the
field after the fit, reporting what it replaced.

**3. It ties the result to a robot.** The shared calibrator stamps ``platform`` but not
which of the two Ventures it was. Issue #13 is explicit that nothing transfers between
them.

Nothing here moves the robot in ``--marker`` or ``--object`` mode. ``--spin`` does, and it
carries the shared calibrator's own ``--live``/``--operator-ready`` gates plus the Lite3
requirement to supply a measured ``--spin-rate`` -- the Go2's 0.8 rad/s is not this
robot's yaw deadband.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_robotics.lite3.commissioning import robot_link
from deep_robotics.lite3.commissioning.measurement import (
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

PLATFORM = "deep-robotics-lite3-venture"

#: Modes of the shared calibrator that put an image through the neural detector, and so
#: have an inference configuration that can disagree with production's.
DETECTOR_MODES = ("object", "spin-object")


def _common_path() -> Path:
    return Path(__file__).resolve().parents[3] / "unitree" / "go2" / "visual_nav"


def _import_common():
    """Put the shared visual-navigation package on the path and import it.

    Imported lazily so that ``--help`` and the dry-run path work on a laptop with no
    ``opencv-python`` -- which is most laptops in a commissioning session, and all of the
    ones running the tests.
    """
    robot_stack = str(Path(__file__).resolve().parents[3])
    common = str(_common_path())
    for entry in (robot_stack, common):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import calibrate_camera
    import person_detector
    import visual_nav
    from deep_robotics.lite3.visual_nav.robot_bindings import Lite3Bindings
    return calibrate_camera, visual_nav, person_detector, Lite3Bindings


def probe_inference_config(person_detector_module) -> dict:
    """What the shared calibrator's detector modes will actually run at.

    Read out of ``PersonDetector.__init__``'s own defaults rather than restated here: the
    calibrator passes neither, so its configuration *is* whatever those defaults are, and
    a copy of them in this file would go stale silently -- which is the same class of
    mistake this assertion exists to catch.
    """
    parameters = inspect.signature(person_detector_module.PersonDetector.__init__).parameters
    config = {}
    for name in ("input_size", "confidence"):
        parameter = parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise Refusal(
                f"PersonDetector.__init__ no longer has a default for {name!r}, so this "
                f"assertion can no longer tell what the calibrator runs at. Fix the "
                f"assertion rather than deleting it."
            )
        config[name] = parameter.default
    return config


def production_inference_config(visual_nav_module, bindings) -> dict:
    """What ``lite3_visual_nav.py`` / ``mappo_drive.py`` will run at.

    Taken from the production parser's own defaults, for the same reason as above.
    """
    parser = visual_nav_module.build_parser(bindings)
    defaults = {action.dest: action.default for action in parser._actions}
    missing = [name for name in ("input_size", "confidence") if name not in defaults]
    if missing:
        raise Refusal(
            f"the production navigator's parser no longer defines {missing}, so this "
            f"assertion cannot compare the two configurations. Fix the assertion."
        )
    return {"input_size": defaults["input_size"], "confidence": defaults["confidence"]}


def assert_matches_production(probe: dict, production: dict, *, mode: str,
                              printer=print) -> dict:
    """Refuse when the calibration would be fitted through a different detector.

    Returns what was compared, so the artefact records the assertion rather than only its
    verdict -- a check whose inputs are not written down is a check nobody can re-run.
    """
    if mode not in DETECTOR_MODES:
        printer(f"[camera] inference-config assertion: NOT APPLICABLE -- {mode} mode uses "
                f"no neural detector, so there is no configuration to mismatch")
        return {"applicable": False, "mode": mode}
    differences = {name: (probe[name], production[name]) for name in probe
                   if probe[name] != production.get(name)}
    if differences:
        detail = "; ".join(f"{name}: calibration would use {probe_value!r}, production "
                           f"uses {production_value!r}"
                           for name, (probe_value, production_value) in
                           sorted(differences.items()))
        raise Refusal(
            f"the calibration and the production navigator would run the detector "
            f"differently ({detail}). A calibration fitted through one inference "
            f"configuration and used under another describes neither. Either use "
            f"--marker, which uses no detector at all, or make the two match before "
            f"measuring anything."
        )
    printer(f"[camera] inference-config assertion: PASS -- calibration and production "
            f"both run the detector at {probe}")
    return {"applicable": True, "mode": mode, "probe": probe, "production": production}


def delegated_argv(args) -> list:
    """The exact argument list handed to the shared calibrator."""
    argv = ["--camera-source", str(args.camera_source), "--out", args.out]
    if args.camera_gstreamer:
        argv.append("--camera-gstreamer")
    if args.marker is not None:
        argv += ["--marker", f"{args.marker}", "--marker-size", f"{args.marker_size}"]
    elif args.object is not None:
        argv += ["--object", f"{args.object}", "--object-height",
                 f"{args.object_height}", "--object-class", args.object_class]
    else:
        argv += ["--spin", "--spin-target", args.spin_target, "--live"]
        if args.spin_rate is not None:
            argv += ["--spin-rate", f"{args.spin_rate}"]
    if args.operator_ready:
        argv.append("--operator-ready")
    if args.accept_no_motor_temperatures:
        argv.append("--accept-no-motor-temperatures")
    return argv


def selected_mode(args) -> str:
    if args.marker is not None:
        return "marker"
    if args.object is not None:
        return "object"
    return f"spin-{args.spin_target}"


def describe_replaced(value) -> str:
    """How to say what the fitter left in ``height_m``, including nothing at all.

    ``None`` here is not a missing key. The shared model writes an unset lens height as
    ``null`` rather than omitting it, precisely so this wrapper can read the absence back
    out, so ``None`` means "the fitter considered the height and had nothing to put
    there". Interpolated straight into a sentence it renders as "None m", which reads as a
    defect in this script rather than as a fact about the calibration -- and it is the one
    the operator is being asked to act on.
    """
    return "unset (null)" if value is None else f"{value} m"


def stamp_lens_height(path, lens_height_m: float, source: str, context: dict) -> dict:
    """Rewrite the calibration's ``height_m`` with the measured value, and say what it was.

    Reads the file the shared calibrator just wrote, refuses it if it is not Lite3-tagged,
    replaces the lens height, and records both the old value and where the new one came
    from. The old value is worth keeping in the file, and there are two of them: ``0.32``
    sitting there is how the next person learns that the fitter's default was the Go2's,
    and ``null`` is how they learn that the fitter considered the height and had nothing to
    put there. ``height_m_replaced`` records which it was, so the difference survives into
    the artefact.
    """
    destination = Path(path)
    try:
        data = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refusal(f"cannot read the calibration the fitter wrote at {destination}: "
                      f"{error}") from None
    if not isinstance(data, dict):
        raise Refusal(f"{destination} must contain a JSON object")
    if data.get("platform") != PLATFORM:
        raise Refusal(
            f"{destination} is tagged platform={data.get('platform')!r}, not "
            f"{PLATFORM!r}. This is not a Lite3 calibration and the Lite3 navigator would "
            f"refuse it too."
        )
    inherited = data.get("height_m")
    data["height_m"] = float(lens_height_m)
    data["height_m_replaced"] = inherited
    data["height_m_source"] = source
    data.update({f"commissioning_{key}": value for key, value in context.items()})
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return {"focal_px": data.get("focal_px"), "hfov_deg": data.get("hfov_deg"),
            "width": data.get("width"), "height": data.get("height"),
            "lens_height_m": float(lens_height_m), "lens_height_replaced": inherited,
            "lens_height_source": source, "calibration_path": str(destination.resolve()),
            "method": data.get("method"), "samples": data.get("samples")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    robot_link.add_context_arguments(parser)

    camera = parser.add_argument_group("the installed RGB endpoint")
    camera.add_argument("--camera-source", required=True,
                        help="V4L2 index, RTSP URI, or GStreamer pipeline. Use the SAME "
                             "string the demo will use: a calibration fitted on one "
                             "endpoint does not describe another")
    camera.add_argument("--camera-gstreamer", action="store_true",
                        help="open --camera-source with OpenCV's GStreamer backend")
    camera.add_argument("--out", default=None,
                        help="calibration JSON to write "
                             "(default: lite3_front_camera_<robot-id>.json)")

    target = parser.add_argument_group("target (choose one)")
    mode = target.add_mutually_exclusive_group(required=True)
    mode.add_argument("--marker", type=float, default=None, metavar="DISTANCE_M",
                      help="RECOMMENDED. Printed ArUco marker at this tape-measured "
                           "camera-to-marker distance. Uses no neural detector, so it "
                           "cannot disagree with production's inference config, and it "
                           "does not move the robot")
    mode.add_argument("--object", type=float, default=None, metavar="DISTANCE_M",
                      help="detected object of known height at this measured distance. "
                           "Least precise; an SSD box is a soft fit")
    mode.add_argument("--spin", action="store_true",
                      help="turn on the spot and fit against yaw odometry. Needs no tape "
                           "measure, but MOVES THE ROBOT and needs a measured --spin-rate")
    target.add_argument("--marker-size", type=float, default=None, metavar="M",
                        help="printed black square edge length, in metres")
    target.add_argument("--object-height", type=float, default=None, metavar="M")
    target.add_argument("--object-class", default="bottle")
    target.add_argument("--spin-target", choices=("marker", "object"), default="marker")
    target.add_argument("--spin-rate", type=float, default=None, metavar="RAD_S",
                        help="measured yaw rate above THIS robot's deadband. The Go2's "
                             "0.8 rad/s is not a Lite3 measurement")

    lens = parser.add_argument_group("lens height (nothing on the fitting path asks for this)")
    lens.add_argument("--lens-height", type=float, default=None, metavar="M",
                      help="optical centre above the floor with the robot STANDING. No "
                           "default here, and nothing worth inheriting from the fitter: "
                           "it leaves either the Go2's 0.32 m or nothing at all")
    lens.add_argument("--lens-height-source", default=None,
                      help="how you measured it, e.g. 'tape, standing, floor to lens "
                           "centre'. Recorded in the calibration file")

    parser.add_argument("--operator-ready", action="store_true",
                        help="required by --spin: robot STANDING in vendor high-level "
                             "navigation mode, emergency stop in hand")
    parser.add_argument("--accept-no-motor-temperatures", action="store_true",
                        help="required by --spin while the temperature channel is a "
                             "vendor question; see motor_temperature_probe.py")
    parser.add_argument("--run", action="store_true",
                        help="actually capture. Without it this prints the delegated "
                             "command and the assertions it would make, and exits")
    parser.add_argument("--artefact", default=None,
                        help="write the machine-readable record here "
                             "(default: lite3-camera-<robot-id>.json)")
    return parser


def _validate(args) -> None:
    refuse_unmeasured(**{"--lens-height": args.lens_height})
    require_positive_finite(**{"--lens-height": args.lens_height})
    if not args.lens_height_source or not args.lens_height_source.strip():
        raise Refusal(
            "--lens-height-source is required. A height with no method attached is a "
            "number somebody will later assume was measured, and the Go2's 0.32 m sitting "
            "in a calibration file because the shared fitter defaulted it is exactly what "
            "that looks like."
        )
    if args.marker is not None:
        refuse_unmeasured(**{"--marker-size": args.marker_size})
        require_positive_finite(**{"--marker": args.marker,
                                   "--marker-size": args.marker_size})
    elif args.object is not None:
        refuse_unmeasured(**{"--object-height": args.object_height})
        require_positive_finite(**{"--object": args.object,
                                   "--object-height": args.object_height})
    else:
        refuse_unmeasured(**{"--spin-rate": args.spin_rate})
        require_positive_finite(**{"--spin-rate": args.spin_rate})
        if not args.operator_ready:
            raise Refusal(
                "--spin turns the robot. Stand it in the vendor's high-level navigation "
                "mode, keep the emergency stop in your hand, then pass --operator-ready."
            )


def _paste(record, result, assertion) -> str:
    rows = [
        ("robot", record.robot_id),
        ("firmware", record.context.get("firmware")),
        ("payload", record.context.get("payload")),
        ("camera endpoint", f"`{record.context.get('camera_source')}`"),
        ("method", f"{result.get('method')}, {result.get('samples')} samples"),
        ("frame size", f"{result.get('width')}x{result.get('height')}"),
        ("focal length", f"{result['focal_px']:.1f} px"),
        ("HFOV", f"{result['hfov_deg']:.2f} deg"),
        ("lens height, standing", f"{result['lens_height_m']:.3f} m"),
        ("calibration file", f"`{Path(result['calibration_path']).name}`"),
    ]
    notes = [
        f"- Lens height was **{describe_replaced(result['lens_height_replaced'])}** in "
        f"the file the shared fitter wrote and has been replaced with the measured "
        f"{result['lens_height_m']:.3f} m ({result['lens_height_source']}). Nothing on "
        f"the fitting path asks for this value, so a calibration carries whatever the "
        f"shared camera model left in the field -- the Go2's 0.32 m where that default "
        f"is still in place, and `null` where it has been removed -- until it is "
        f"overwritten here.",
    ]
    if assertion.get("applicable"):
        notes.append(f"- Inference config asserted equal to production: "
                     f"`{assertion['probe']}`.")
    else:
        notes.append("- `--marker` uses no neural detector, so there is no inference "
                     "configuration that could differ from production's.")
    notes.append("- Measured on this robot's installed camera only. Do not copy to the "
                 "other Venture.")
    return paste_block("Lite3 camera calibration", rows, notes)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.out = args.out or f"lite3_front_camera_{args.robot_id}.json"
    mode = selected_mode(args)
    brief(
        "Lite3 camera calibration -- focal length, HFOV, and the lens height",
        does=f"""
        Runs the shared Go2 focal-length fitter through Lite3Bindings in {mode} mode,
        asserts that it uses the same detector configuration production will, then
        replaces the calibration's inherited lens height with the one you measured.
        """,
        needs=[
            "the SAME --camera-source string the demo will use",
            "for --marker: a printed ArUco marker, square-on, at a tape-measured "
            "distance. Measure at the longest distance it is still comfortably detected",
            "the lens height measured with the robot STANDING, floor to optical centre",
            "for --spin only: a clear turning circle, a measured --spin-rate above this "
            "robot's yaw deadband, and your hand on the emergency stop",
        ],
        means="""
        focal_px is pixels per radian off the optical axis and the whole perception
        pipeline scales on it: bearings steer the robot, and range is size/angle, so a
        20% focal error is a 20% range error.
        The lens height bounds a width-derived range in the detector, so it is not
        cosmetic.
        """,
        moves=args.spin,
    )
    _validate(args)

    context = {"robot_id": args.robot_id, "firmware": args.firmware,
               "payload": args.payload, "camera_source": str(args.camera_source),
               "provenance": "provisional"}
    print(f"[camera] would run: python3 "
          f"{_common_path() / 'calibrate_camera.py'} "
          f"{' '.join(delegated_argv(args))}")
    print("[camera]   (through Lite3Bindings, from this process -- shown so you can see "
          "exactly what is delegated)")

    if not args.run:
        print("")
        print("[camera] DRY RUN. No camera was opened. The inference-config assertion "
              f"for {mode} mode runs at capture time; add --run to perform it.")
        return 0

    calibrate_camera, visual_nav, person_detector, Lite3Bindings = _import_common()
    bindings = Lite3Bindings()
    assertion = assert_matches_production(
        probe_inference_config(person_detector),
        production_inference_config(visual_nav, bindings),
        mode=mode)

    calibrate_camera.main(argv=delegated_argv(args), bindings=bindings)
    result = stamp_lens_height(args.out, args.lens_height,
                               args.lens_height_source.strip(), context)
    result["inference_assertion"] = assertion

    print("")
    print(f"  focal      {result['focal_px']:.1f} px")
    print(f"  HFOV       {result['hfov_deg']:.2f} deg")
    print(f"  lens height {result['lens_height_m']:.3f} m "
          f"(replaced the fitter's {describe_replaced(result['lens_height_replaced'])})")

    record = new_record(args.robot_id, firmware=args.firmware, payload=args.payload,
                        camera_source=str(args.camera_source), mode=mode,
                        calibration_out=str(Path(args.out).resolve()))
    merge_measurement(record, "camera_calibration", result)
    destination = Path(args.artefact or f"lite3-camera-{args.robot_id}.json")
    write_record(destination, record)
    print(f"\n[camera] artefact: {destination.resolve()}  "
          f"(provenance: {record.provenance})")
    print_paste_block(_paste(record, result, assertion))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(lambda: main(), "camera"))
