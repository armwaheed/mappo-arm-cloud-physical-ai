#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure the front camera's focal length, instead of trusting the nominal spec.

The whole pipeline scales on one number — ``focal_px``, pixels per radian off the
optical axis — and this robot ships no calibration for its front camera. (Its VIO
calibration files describe a RealSense that is not fitted.) Everything downstream is
only as good as this: bearings steer the robot, and range is ``size / angle``, so a
20% focal error is a 20% range error.

Three ways to measure it, all reducing to a 1-D search over that single intrinsic.

``--spin`` (needs ``--live``; the robot turns on the spot) — THE ACCURATE ONE
    Leave any recognisable object in view and let the robot rotate slowly. Its own yaw
    odometry is the angular ruler: as it turns through a known angle, the object's
    bearing must change by the same angle, and only the true focal length makes that
    hold across the whole sweep.

    This is the method to trust, because it needs **no tape measure and no size
    prior**. Both static methods below inherit every error in the two numbers you feed
    them, and at short range those errors are large: measuring to the near face of a
    0.12 m-wide object rather than its axis moves the answer by 10%, and "distance to
    the robot" is ambiguous by however far the lens sits inside the nose. The spin
    method has no such term. It also samples the model right across the frame instead
    of at a single point, which is where an equidistant fit is most likely to show
    strain.

``--marker DISTANCE`` (no motion)
    A printed ArUco marker at a tape-measured distance. Corner refinement makes the
    pixel span precise; the distance is then the only weak link.

``--object DISTANCE --object-height H`` (no motion)
    Any detectable object of measured height, at a measured distance. The most
    convenient and the least precise: an SSD box is a soft fit, measured on this robot
    at ~8% taller than the object it bounds, and that error goes straight into the
    focal length. Prefer it for a sanity check, not for the number you ship.

For any static method, **measure at the longest distance where the object is still
comfortably detected**. Placement error is roughly constant in metres, so its relative
cost falls as distance grows: ±3 cm is 8% at 0.37 m and 2% at 1.5 m.

All three write a JSON model that ``visual_nav.py --calibration`` reads back.
``--spin`` runs the same arm-stowed and thermal pre-flight as the navigator, stands,
turns, and lies back down.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))       # sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # locomotion/, d1_arm/

from camera_model import DEFAULT_HFOV_DEG, FisheyeCamera, solve_focal_px
from geometry import wrap_pi
from goal import (
    DEFAULT_DICTIONARY,
    DEFAULT_MARKER_ID,
    DEFAULT_MARKER_SIZE_M,
    aruco_detector,
)
from lifecycle import run_cleanup
from robot_bindings import Go2Bindings

MIN_STATIC_SAMPLES = 3
MIN_SPIN_SAMPLES = 8
MIN_SPIN_YAW_DEG = 20.0   # below this the fit is unconstrained — see calibrate_from_spin
RECORD_FPS = 10.0

# This robot has a large yaw deadband, and a sweep inside it never turns far enough to
# constrain anything: it burns the whole sweep and then fails MIN_SPIN_YAW_DEG. Measured
# on this unit — 0.30 rad/s commanded achieved 0.02-0.04 (7-14%); 0.80 achieved 0.45-0.49
# (56-61%); 1.50 saturated at 0.55-0.58. So the default is the measured-good rate rather
# than the gentle-looking one; --derate has no equivalent here, this IS the knob.
SPIN_DEADBAND_RAD_S = 0.4   # below this the robot does not reliably initiate a turn
SPIN_RATE_RAD_S = 0.8


@dataclass(frozen=True)
class Sighting:
    """Where the calibration target was found in one frame.

    ``centre`` is all the fit uses; ``detection`` rides along only so a recorded
    sweep can draw the box that produced it.
    """

    centre: np.ndarray
    detection: object = None


# ── Locators: each returns pixel evidence for one frame, or None ────────────
def marker_corners(image: np.ndarray, detect, marker_id: int) -> np.ndarray | None:
    """Full-resolution corners of ``marker_id``, or ``None``."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids = detect(gray)
    if ids is None:
        return None
    for marker, found_id in zip(corners, ids.flatten()):
        if int(found_id) == marker_id:
            return marker.reshape(4, 2)
    return None


def best_detection(image: np.ndarray, detector, label: str):
    """Highest-scoring detection of ``label``, or ``None``."""
    best = None
    for detection in detector.detect(image):
        if detection.label == label and (best is None or detection.score > best.score):
            best = detection
    return best


def object_span(image: np.ndarray, detector, label: str):
    """Vertical span ``(top_centre, bottom_centre)`` of the best ``label`` box.

    Rejects a box touching the frame edge: the object runs out of shot, so its
    apparent height is a lower bound and would bias the focal length low.
    """
    best = best_detection(image, detector, label)
    if best is None or any(best.clipped(image.shape[1], image.shape[0])):
        return None
    centre_x, _ = best.centre
    return (centre_x, best.y1), (centre_x, best.y2)


# ── Fits ────────────────────────────────────────────────────────────────────
def subtended_angle(model: FisheyeCamera, points: np.ndarray) -> float:
    """Angle subtended by the marker's less-foreshortened edge pair.

    Matches ``goal.ArucoGoal`` exactly, so the calibration optimises the number the
    navigator will actually use.
    """
    pair_a = 0.5 * (model.angle_between(points[0], points[1])
                    + model.angle_between(points[2], points[3]))
    pair_b = 0.5 * (model.angle_between(points[1], points[2])
                    + model.angle_between(points[3], points[0]))
    return float(max(pair_a, pair_b))


def _model_with(width: int, height: int, focal_px: float) -> FisheyeCamera:
    return FisheyeCamera(width=width, height=height, focal_px=focal_px,
                         cx=width / 2.0, cy=height / 2.0)


def calibrate_from_spans(spans: list, width: int, height: int, physical_m: float,
                         distance_m: float) -> FisheyeCamera:
    """Focal length that makes a known span at a KNOWN distance measure that distance.

    ``spans`` is a list of ``(p1, p2)`` pixel pairs, each bounding the same physical
    length ``physical_m``.
    """

    def residual(focal_px: float) -> float:
        model = _model_with(width, height, focal_px)
        total = 0.0
        for p1, p2 in spans:
            estimate = model.range_from_span(p1, p2, physical_m)
            if math.isfinite(estimate):
                total += (estimate - distance_m) ** 2
        return total

    return _model_with(width, height, solve_focal_px(residual, width))


def calibrate_from_marker(samples: list, width: int, height: int,
                          marker_size_m: float, distance_m: float) -> FisheyeCamera:
    """Focal length that makes a marker at a KNOWN distance measure that distance."""

    def residual(focal_px: float) -> float:
        model = _model_with(width, height, focal_px)
        total = 0.0
        for points in samples:
            angle = subtended_angle(model, points)
            if angle > model.pixel_angle_rad:   # same floor as range_from_span
                estimate = marker_size_m / (2.0 * math.tan(angle / 2.0))
                total += (estimate - distance_m) ** 2
        return total

    return _model_with(width, height, solve_focal_px(residual, width))


def calibrate_from_spin(samples: list, width: int, height: int) -> FisheyeCamera:
    """Focal length that makes image bearing track measured yaw across a sweep.

    ``samples`` is ``(centre_pixel, robot_yaw)``. For an object fixed in the world,
    turning the robot left by ``dyaw`` must move its bearing right by exactly
    ``dyaw``. The unknown constant start bearing drops out: for each candidate focal
    length it is fitted in closed form as the mean offset, leaving a 1-D search.

    Needs a real yaw sweep to constrain anything — with the object parked near the
    optical axis every focal length fits equally well.
    """
    centres = np.array([centre for centre, _ in samples], dtype=float)
    yaws = np.array([yaw for _, yaw in samples], dtype=float)
    yaw_delta = np.unwrap(yaws) - yaws[0]

    def residual(focal_px: float) -> float:
        model = _model_with(width, height, focal_px)
        bearings, _ = model.bearing_elevation(centres[:, 0], centres[:, 1])
        # bearing_i = start_bearing - yaw_delta_i; the best start_bearing is the mean.
        predicted = bearings + yaw_delta
        return float(np.sum((predicted - predicted.mean()) ** 2))

    return _model_with(width, height, solve_focal_px(residual, width))


def yaw_span_deg(samples: list) -> float:
    """Total yaw swept across ``(centre_pixel, yaw)`` samples, in degrees.

    Unwrapped before differencing, exactly as :func:`calibrate_from_spin` fits. A raw
    ``max - min`` over wrapped yaw reads a sweep that straddles +-pi as a near-full
    turn — a genuine 10 deg sweep reports 359 deg — which would walk straight through
    the guard that exists to stop an unconstrained fit being written out as a
    calibration.
    """
    unwrapped = np.unwrap([yaw for _, yaw in samples])
    return float(math.degrees(unwrapped.max() - unwrapped.min()))


def spin_fit_quality(model: FisheyeCamera, samples: list) -> dict:
    """Decompose the fit error into target jitter versus systematic model misfit.

    A raw residual cannot tell you which you have, and the two call for opposite
    responses: jitter means take more samples or use a rigid target, misfit means no
    single focal length will ever work. Measured on this robot with a PERSON as the
    target, the residual was 3.13 deg and the jitter 3.12 deg — the equidistant model
    was blameless and a human simply is not a rigid fiducial.

    The split uses a second difference along the sample series. Any smooth sweep is
    locally linear in time, so differencing twice cancels it and leaves only
    high-frequency noise; for white noise of sigma, ``std(2nd diff) = sqrt(6)*sigma``.
    Whatever is left over after removing that is systematic.

    Returns degrees: ``residual_rms``, ``jitter``, ``systematic``, ``standard_error``.
    """
    centres = np.array([centre for centre, _ in samples], dtype=float)
    yaws = np.array([yaw for _, yaw in samples], dtype=float)
    bearings, _ = model.bearing_elevation(centres[:, 0], centres[:, 1])
    predicted = bearings + (np.unwrap(yaws) - yaws[0])
    error = predicted - predicted.mean()

    residual = float(math.degrees(np.sqrt(np.mean(error ** 2))))
    if len(error) >= 3:
        second = error[2:] - 2 * error[1:-1] + error[:-2]
        jitter = float(math.degrees(np.std(second) / math.sqrt(6.0)))
    else:
        jitter = 0.0
    # Variances subtract; clamp because noise estimates can exceed the total.
    systematic = float(math.sqrt(max(residual ** 2 - jitter ** 2, 0.0)))
    return {"residual_rms": residual, "jitter": jitter, "systematic": systematic,
            "standard_error": residual / math.sqrt(max(len(error), 1))}


# ── Capture ─────────────────────────────────────────────────────────────────
def collect_static(camera, locate, count: int, timeout_s: float) -> list:
    """Gather up to ``count`` sightings from a stationary robot."""
    samples: list = []
    seq = 0
    deadline = time.monotonic() + timeout_s
    while len(samples) < count and time.monotonic() < deadline:
        frame = camera.wait_for_new(seq, timeout=1.0)
        if frame is None:
            continue
        seq = frame.seq
        found = locate(frame.image)
        if found is not None:
            samples.append(found)
    return samples


def collect_spin(camera, loco, locate, pose_yaw, rate: float,
                 seconds: float, max_yaw_deg: float, recorder=None,
                 annotate=None) -> tuple[list, int]:
    """Sweep back and forth on the spot, gathering ``(centre_pixel, yaw)``.

    Bidirectional rather than one continuous rotation, for two reasons. This robot is
    **tethered by an Ethernet cable**, and a continuous spin winds it around the legs;
    the sweep returns to where it started, so the tether never accumulates a turn. And
    the target has to stay in frame — a full rotation loses it almost immediately,
    whereas a bounded sweep keeps it in view while still moving its bearing right
    across the frame, which is exactly what constrains the fit.

    Returns ``(samples, frames_seen)``. The frame count is what separates "the camera
    stalled" from "the target was never recognised" when a sweep comes back empty —
    without it, both look identical and the sweep has to be repeated to find out.
    """
    samples: list = []
    frames_seen = 0
    seq = 0
    start_yaw = pose_yaw()
    limit = math.radians(max_yaw_deg)
    direction = 1.0
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            # Sample the heading ONCE. Calling pose_yaw() separately for the sine and
            # the cosine reads a turning robot at two different instants, so the two
            # halves of the atan2 disagree — an error that only exists while moving,
            # which is the only time this loop runs.
            offset = float(wrap_pi(pose_yaw() - start_yaw))
            if abs(offset) >= limit:
                direction = -math.copysign(1.0, offset)   # turn back toward the start
            loco.set_velocity(0.0, 0.0, direction * rate)

            frame = camera.wait_for_new(seq, timeout=0.5)
            if frame is None:
                continue
            seq = frame.seq
            frames_seen += 1
            # Yaw at the camera adapter boundary, carried by stamp_fn and not sampled
            # after detection. This fit pairs an image bearing with a robot heading, so a
            # stale heading is a direct error in the thing being measured: at the
            # ~27 deg/s this sweep actually achieves, even 60 ms of frame age is 1.6
            # deg, and it flips sign with sweep direction, which is exactly the shape
            # that inflates the residual without moving the fitted value much.
            yaw_now = frame.stamp if frame.stamp is not None else pose_yaw()
            sighting = locate(frame.image)
            if sighting is not None:
                samples.append((sighting.centre, yaw_now))
            if recorder is not None and annotate is not None:
                yaw_offset = math.degrees(float(wrap_pi(yaw_now - start_yaw)))
                recorder.write(annotate(frame.image, sighting, yaw_offset, frames_seen,
                                        len(samples)))
    finally:
        loco.stop()   # the turn must end even if the locator throws
    # Leave the robot pointing where it started, so the tether is unwound and the
    # next run begins from the same heading.
    _return_to_yaw(loco, pose_yaw, start_yaw, rate)
    return samples, frames_seen


def _return_to_yaw(loco, pose_yaw, target_yaw: float, rate: float,
                   tolerance_deg: float = 4.0, timeout_s: float = 8.0) -> None:
    """Turn back to ``target_yaw``. Best-effort: never raises, always stops."""
    tolerance = math.radians(tolerance_deg)
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            # One heading sample per iteration — see collect_spin for why two is a bug.
            error = float(wrap_pi(target_yaw - pose_yaw()))
            if abs(error) <= tolerance:
                break
            loco.set_velocity(0.0, 0.0, math.copysign(rate, error))
            time.sleep(0.05)
    except Exception as exc:
        print(f"[calibrate] return-to-start failed: {exc!r}")
    finally:
        loco.stop()


def _target_in_view(camera, locate, seconds: float = 2.0) -> bool:
    """Whether the locator finds the target in any frame within ``seconds``."""
    seq = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = camera.wait_for_new(seq, timeout=0.5)
        if frame is None:
            continue
        seq = frame.seq
        if locate(frame.image) is not None:
            return True
    return False


# ── Modes ───────────────────────────────────────────────────────────────────
def run_marker(camera, args, width: int, height: int):
    detect = aruco_detector(args.dictionary)
    print(f"[calibrate] hold marker {args.marker_id} square-on at {args.marker:.3f} m")
    samples = collect_static(
        camera, lambda im: marker_corners(im, detect, args.marker_id),
        args.samples, 20.0)
    if len(samples) < MIN_STATIC_SAMPLES:
        raise SystemExit(f"[calibrate] only {len(samples)} sightings — check the marker "
                         f"id, the lighting, and that it is fully in frame")
    print(f"[calibrate] {len(samples)} sightings")
    model = calibrate_from_marker(samples, width, height, args.marker_size, args.marker)
    return model, {"method": "marker", "samples": len(samples),
                   "distance_m": args.marker, "marker_size_m": args.marker_size}


def run_object(camera, args, width: int, height: int):
    from person_detector import PersonDetector

    print(f"[calibrate] looking for a '{args.object_class}' "
          f"{args.object_height:.3f} m tall at {args.object:.3f} m")
    detector = PersonDetector(args.model_dir, classes=(args.object_class,))
    spans = collect_static(camera,
                           lambda im: object_span(im, detector, args.object_class),
                           args.samples, 20.0)
    if len(spans) < MIN_STATIC_SAMPLES:
        raise SystemExit(f"[calibrate] only {len(spans)} clean sightings of "
                         f"'{args.object_class}' — is it fully in frame and unclipped?")
    heights = [abs(p2[1] - p1[1]) for p1, p2 in spans]
    print(f"[calibrate] {len(spans)} sightings, box height {min(heights):.0f}-"
          f"{max(heights):.0f} px")
    print("[calibrate] NOTE: an SSD box is a soft fit (~8% tall on this robot's own "
          "test object); treat this as a sanity check, not a shipped calibration.")
    model = calibrate_from_spans(spans, width, height, args.object_height, args.object)
    return model, {"method": "object", "samples": len(spans),
                   "object_class": args.object_class,
                   "object_height_m": args.object_height, "distance_m": args.object}


def run_spin(camera, args, width: int, height: int, loco, pose_yaw,
             stand_up_fn, lie_down_fn):
    """Turn on the spot and fit bearing against odometry. Needs no measurements."""
    if args.spin_target == "marker":
        detect = aruco_detector(args.dictionary)

        def locate(image):
            points = marker_corners(image, detect, args.marker_id)
            return None if points is None else Sighting(centre=points.mean(axis=0))
        target = f"ArUco {args.marker_id}"
    else:
        from person_detector import PersonDetector

        detector = PersonDetector(args.model_dir, classes=(args.object_class,))

        def locate(image):
            # Only the box CENTRE is used, so the loose box edges that spoil the
            # static object fit do not matter here — centre error is far smaller
            # than edge error, and it is what the bearing depends on.
            best = best_detection(image, detector, args.object_class)
            return (None if best is None
                    else Sighting(centre=np.array(best.centre), detection=best))
        target = f"'{args.object_class}'"

    # When the target is a person, the operator who typed the command IS the target,
    # and they are standing at a laptop rather than 2.5 m in front of the robot. Give
    # them time to walk there — otherwise the pre-flight check below sees them
    # mid-departure, passes, and the sweep then runs against an empty room.
    if args.start_delay > 0:
        print(f"[calibrate] starting in {args.start_delay:.0f}s — get into position "
              f"now, facing the robot, and stay still")
        for remaining in range(int(args.start_delay), 0, -1):
            if remaining <= 5 or remaining % 5 == 0:
                print(f"  {remaining}…", flush=True)
            time.sleep(1.0)

    print("[calibrate] standing up")
    stand_up_fn(loco)

    # Confirm the target is visible FROM THE STANDING POSE before committing to the
    # sweep. Standing lifts this camera from ~0.10 m to ~0.32 m, which is enough to
    # look clean over a short object that filled the frame while the robot was lying
    # down — exactly what happened on this robot with a 0.25 m canister at 0.5 m.
    # Twenty seconds of motion to discover that is twenty seconds too many.
    if not _target_in_view(camera, locate, seconds=2.0):
        lie_down_fn(loco)
        # Two quite different causes, and naming the wrong one sends the operator off
        # to fix something that is not broken.
        hint = ("Is the person actually in position and facing the robot? Whoever runs "
                "this command is also the target — use --start-delay to give yourself "
                "time to walk there."
                if args.object_class == "person" else
                "The camera sits ~0.32 m up when STANDING and looks level, so a short "
                "object close in drops below the frame even though it filled it while "
                "the robot was lying down. Move the target back (a 0.25 m object needs "
                "roughly 1 m) or use a taller one.")
        raise SystemExit(f"[calibrate] {target} is not visible with the robot STANDING, "
                         f"so the sweep would find nothing. {hint}")

    print(f"[calibrate] sweeping +-{args.spin_max_yaw:.0f} deg at "
          f"{args.spin_rate:.2f} rad/s for {args.spin_seconds:.0f}s — keep {target} "
          f"in view throughout")

    recorder = annotate = None
    if args.record:
        import overlay

        recorder = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                   RECORD_FPS, (width, height))
        # A writer the build cannot open swallows every frame and leaves a 0-byte file,
        # so check before the robot turns rather than after the sweep is unrepeatable.
        if not recorder.isOpened():
            raise SystemExit(f"[calibrate] cannot open {args.record} for writing "
                             f"(mp4v) — refusing to sweep without the recording")

        def annotate(image, sighting, yaw_offset_deg, frame_index, sightings):
            canvas = image.copy()
            overlay.draw_calibration(
                canvas, None if sighting is None else sighting.detection,
                None if sighting is None else sighting.centre,
                yaw_offset_deg, frame_index, sightings)
            return canvas

    try:
        samples, frames_seen = collect_spin(
            camera, loco, locate, pose_yaw, args.spin_rate, args.spin_seconds,
            args.spin_max_yaw, recorder=recorder, annotate=annotate)
    finally:
        if recorder is not None:
            recorder.release()
            print(f"[calibrate] wrote {args.record}")

    if len(samples) < MIN_SPIN_SAMPLES:
        raise SystemExit(f"[calibrate] only {len(samples)} sightings in {frames_seen} "
                         f"frames — {target} must stay in view for the whole sweep "
                         f"(0 sightings with frames > 0 means the detector never fired; "
                         f"0 frames means the camera stalled)")
    spread = yaw_span_deg(samples)
    if spread < MIN_SPIN_YAW_DEG:
        raise SystemExit(f"[calibrate] only {spread:.1f} deg of yaw — too little to "
                         f"constrain the fit (need {MIN_SPIN_YAW_DEG:.0f} deg). Turn "
                         f"further or for longer.")
    print(f"[calibrate] {len(samples)} sightings over {spread:.1f} deg of yaw")
    model = calibrate_from_spin(samples, width, height)
    quality = spin_fit_quality(model, samples)
    print(f"[calibrate] fit residual {quality['residual_rms']:.2f} deg RMS = "
          f"{quality['jitter']:.2f} target jitter + {quality['systematic']:.2f} "
          f"systematic; standard error {quality['standard_error']:.2f} deg")
    if quality["systematic"] > 1.0:
        print("[calibrate] WARNING: the systematic part is large — the equidistant "
              "model may not describe this lens, and no single focal length will fix "
              "that. Check the residual against bearing before trusting this.")
    elif quality["jitter"] > 1.0:
        print("[calibrate] residual is dominated by TARGET JITTER, not the lens. The "
              "fit is sound (noise averages out), but a rigid target — an ArUco "
              "marker rather than a person — would tighten it.")
    return model, {"method": "spin", "samples": len(samples), "target": target,
                   "yaw_span_deg": round(spread, 2),
                   **{k: round(v, 4) for k, v in quality.items()}}


def build_parser(bindings=None) -> argparse.ArgumentParser:
    bindings = bindings or Go2Bindings()
    ap = argparse.ArgumentParser(
        description=f"Calibrate {bindings.platform_name}'s front-camera focal length.")
    ap.add_argument("--out", default=bindings.default_calibration_output,
                    help="model JSON to write")
    ap.add_argument("--samples", type=int, default=20,
                    help="sightings to average over (static modes)")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spin", action="store_true",
                      help="turn on the spot and fit against yaw odometry; needs no "
                           "measurements and is the accurate method (needs --live)")
    mode.add_argument("--marker", type=float, metavar="DISTANCE_M",
                      help="ArUco marker at this measured camera-to-marker distance")
    mode.add_argument("--object", type=float, metavar="DISTANCE_M",
                      help="detected object of known height at this measured distance")

    aruco = ap.add_argument_group("ArUco target")
    aruco.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    aruco.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE_M,
                       help="printed black square, in metres")
    aruco.add_argument("--dictionary", default=DEFAULT_DICTIONARY)

    obj = ap.add_argument_group("detected-object target")
    obj.add_argument("--object-class", default="bottle",
                     help="VOC class of the calibration object")
    obj.add_argument("--object-height", type=float, default=None,
                     help="the object's true height in metres (--object mode)")
    obj.add_argument("--model-dir", default=str(Path.home() / "go2_models"),
                     help="MobileNet-SSD directory")

    spin = ap.add_argument_group("spin")
    spin.add_argument("--live", action="store_true",
                      help="permit motion — required by --spin")
    spin.add_argument("--spin-target", choices=("object", "marker"), default="object",
                      help="what to track through the sweep")
    spin.add_argument("--spin-rate", type=float, default=SPIN_RATE_RAD_S,
                      help=bindings.spin_rate_help)
    spin.add_argument("--spin-seconds", type=float, default=12.0, help="sweep duration")
    spin.add_argument("--start-delay", type=float, default=0.0,
                      help="seconds to wait before standing, so the operator can walk "
                           "into position. Needed when the target is a PERSON: whoever "
                           "runs the command is also the thing being tracked")
    spin.add_argument("--record", default=None,
                      help="write an annotated MP4 of the sweep — the clearest way to "
                           "show the method, and evidence the run happened")
    spin.add_argument("--spin-max-yaw", type=float, default=30.0,
                      help="sweep this far either side of the start heading. Bounded "
                           "because the robot is tethered by Ethernet and because the "
                           "target must stay in frame")
    bindings.add_calibration_arguments(ap, spin)
    return ap


def main(argv=None, bindings=None) -> None:
    bindings = bindings or Go2Bindings()
    ap = build_parser(bindings)
    args = ap.parse_args(argv)
    if args.object is not None and args.object_height is None:
        ap.error("--object needs --object-height METRES (measure it; the whole metric "
                 "scale is proportional to this number)")
    if args.spin and not args.live:
        ap.error("--spin turns the robot; pass --live to authorise motion")

    loco = bindings.create_locomotion(args)
    health = bindings.create_health_monitor(args, live=args.spin)
    camera = None
    standing = False
    try:
        loco.connect()
        health.start()
        bindings.start(args)
        bindings.preflight_calibration(args, health)

        # Stamp each frame with yaw at the camera adapter boundary. The spin fit pairs
        # image bearing with heading, so sampling after detection would add avoidable
        # latency. Camera transport latency is still platform-specific (see collect_spin).
        camera = bindings.create_camera(args, lambda: loco.pose().yaw)
        camera.start()
        height, width = camera.latest().image.shape[:2]
        print(f"[calibrate] camera {width}x{height}, "
              f"nominal HFOV {DEFAULT_HFOV_DEG:.0f}deg")

        if args.spin:
            bindings.prepare_motion(args, loco)
            standing = True
            model, provenance = run_spin(camera, args, width, height, loco,
                                         lambda: loco.pose().yaw,
                                         bindings.stand_up, bindings.lie_down)
        elif args.object is not None:
            model, provenance = run_object(camera, args, width, height)
        else:
            model, provenance = run_marker(camera, args, width, height)

        provenance.update(bindings.calibration_provenance(args))
        model.save(args.out, **provenance)
        print(f"[calibrate] focal={model.focal_px:.1f}px  HFOV={model.hfov_deg:.2f}deg "
              f"(nominal was {DEFAULT_HFOV_DEG:.1f}deg)")
        print(f"[calibrate] wrote {args.out} — pass it as --calibration")
    finally:
        run_cleanup("calibrate", [
            ("platform rest", None if not standing else lambda: bindings.lie_down(loco)),
            ("camera stop", None if camera is None else camera.stop),
            ("platform shutdown", bindings.shutdown),
            ("health stop", health.stop),
            ("locomotion shutdown", loco.shutdown),
        ])


if __name__ == "__main__":
    main()
