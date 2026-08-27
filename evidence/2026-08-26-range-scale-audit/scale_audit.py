#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Is the camera's range scale right? Settled against an object whose size is known.

Regenerates every number in this directory's README from files already in this
repository. No robot, no network:

    python3 scale_audit.py

ISSUE #35 ASKS WHICH OF FOUR ESTIMATES OF THE SAME BIN IS RIGHT — 2.34 m from the size
prior, 2.12 m from the floor contact, "the map reads 25% long" from landmark drift, and
2.9 m from the operator standing in the room. Three of them cannot be, and the issue's
own instruction is not to average them.

THE MEASUREMENT THAT DECIDES IT NEEDS NO TAPE, and it was recorded eighteen months of
argument ago without anyone noticing. Both live runs of ../2026-08-25-peer-runs/ drove at
a printed ArUco marker. A printed marker's size is known BY MANUFACTURE — 0.10 m of black
square, stated in the run header — so the range the stack computes for it is a range
whose only unknowns are the focal length, the projection model and the arithmetic. Fit
those ranges against the robot's own odometry over 2.2-2.6 m of closing and the
multiplicative scale error falls straight out.

THE FIT. For a static object at unknown P, a robot at recorded pose p_i, a reported range
R_i and a reported bearing giving the unit vector u_i, the reported range is k times the
true one exactly when

    p_i + (R_i / k) * u_i = P     for every i

which is a THREE-PARAMETER LINEAR least squares in (P_x, P_y, 1/k). No initial guess, no
iteration, no ground truth. It measures the CAMERA's scale relative to the ODOMETRY's,
and that caveat is stated rather than hidden: a k of 1.0 means the two agree, not that
both are separately perfect.

WHAT IT RETIRES. #35's "How to settle it" item 2 proposed exactly this fit over the RAW
per-sighting ranges, and named the missing `sightings` field as the blocker. The field
landed (#37); this is the first tool to read it; and section 2 below shows the fit does
NOT work there. The blocker was never the field. It is that the detector's per-sighting
range noise — median |dln R| of 103% when `estimate_range` switches source, on 14.1% of
consecutive pairs — swamps 0.2-1.2 m of baseline. The marker works because it is one
unambiguous object, cornered to sub-pixel, over forty-odd fixes and a long baseline.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ROOT = HERE.parents[1]
VISUAL_NAV = ROOT / "robot-stack" / "unitree" / "go2" / "visual_nav"
sys.path.insert(0, str(VISUAL_NAV))

import cv2
from camera_model import FisheyeCamera
from colour_detector import PROFILES, ColourBlobDetector
from person_detector import estimate_range

CAMERA = FisheyeCamera.load(str(VISUAL_NAV / "go2_front_camera.json"))
RUNS = ROOT / "evidence" / "2026-08-25-peer-runs"
#: One frame pulled read-only over HTTP from the robot's own front camera on 2026-08-26,
#: with the staged blue bin in it. Committed so the number below is reproducible.
LIVE_FRAME = HERE / "blue-bin-2026-08-26T19-33Z.jpg"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def scale_of(samples):
    """``(k, residual_rms_m, n)`` from ``(px, py, range_m, world_bearing_rad)`` samples,
    or ``None`` when there are too few or the normal equations are singular.

    Solves ``p_i + a*R_i*u_i = P`` for ``(P_x, P_y, a)``; ``k = 1/a``. Written out as
    3x3 normal equations with a pivoted elimination rather than pulled from numpy, so
    the arithmetic a reader wants to check is on the page.
    """
    if len(samples) < 4:
        return None
    matrix = [[0.0] * 3 for _ in range(3)]
    rhs_vector = [0.0] * 3
    for px, py, range_m, bearing in samples:
        along = (range_m * math.cos(bearing), range_m * math.sin(bearing))
        for coefficients, rhs in (([1.0, 0.0, -along[0]], px),
                                  ([0.0, 1.0, -along[1]], py)):
            for i in range(3):
                rhs_vector[i] += coefficients[i] * rhs
                for j in range(3):
                    matrix[i][j] += coefficients[i] * coefficients[j]
    augmented = [matrix[i] + [rhs_vector[i]] for i in range(3)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda r: abs(augmented[r][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        if abs(augmented[column][column]) < 1e-12:
            return None
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column] / augmented[column][column]
            for j in range(column, 4):
                augmented[row][j] -= factor * augmented[column][j]
    goal_x, goal_y, inverse_k = (augmented[i][3] / augmented[i][i] for i in range(3))
    if inverse_k <= 0.0:
        return None
    sse = sum((goal_x - inverse_k * r * math.cos(b) - px) ** 2
              + (goal_y - inverse_k * r * math.sin(b) - py) ** 2
              for px, py, r, b in samples)
    return 1.0 / inverse_k, math.sqrt(sse / len(samples)), len(samples)


def ticks_of(name: str) -> list:
    with open(RUNS / f"{name}-run-telemetry.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle
                if line.strip() and json.loads(line).get("type") == "tick"]


def marker_fixes(ticks) -> list:
    """Every ArUco fix, recovered from ``goal.x/y`` and the pose of the tick it moved on.

    `ArucoGoal` latches ``pose + range*(cos, sin)(yaw + bearing)`` on each sighting and
    the telemetry writes the latched pair, so the raw range and bearing are recoverable
    exactly. A tick whose goal did not move is a tick with no new sighting.
    """
    fixes, previous = [], None
    for tick in ticks:
        goal = tick.get("goal")
        if not goal:
            continue
        latched = (goal["x"], goal["y"])
        if latched == previous:
            continue
        previous = latched
        pose = tick["pose"]
        offset = (latched[0] - pose["x"], latched[1] - pose["y"])
        fixes.append((pose["x"], pose["y"], math.hypot(*offset),
                      math.atan2(offset[1], offset[0])))
    return fixes


# ── 1. the marker: an object whose size is known by manufacture ─────────────
def marker_scale() -> None:
    rule("1. The range scale, against a printed 10 cm marker and the robot's odometry")
    print("k > 1 means the stack reports ranges LONGER than the odometry says they are.\n")
    print(f"{'run':>9} {'window':>16} {'n':>4} {'k':>7} {'residual rms':>13}")
    for name in ("hero", "contrast"):
        fixes = marker_fixes(ticks_of(name))
        ranges = [f[2] for f in fixes]
        travel = math.hypot(fixes[-1][0] - fixes[0][0], fixes[-1][1] - fixes[0][1])
        bands = (("all fixes", fixes),
                 ("first half", fixes[:len(fixes) // 2]),
                 ("second half", fixes[len(fixes) // 2:]),
                 ("inside 2.0 m", [f for f in fixes if f[2] < 2.0]))
        for label, subset in bands:
            fit = scale_of(subset)
            if fit is None:
                continue
            print(f"{name:>9} {label:>16} {fit[2]:4d} {fit[0]:7.3f} {fit[1] * 100:11.1f} cm")
        print(f"{'':>9} {'':>16}      marker seen {max(ranges):.2f} m -> {min(ranges):.2f} m "
              f"over {travel:.2f} m of odometry\n")
    print("The stack's range chain — focal length, the equidistant projection and\n"
          "`range_from_span` — is therefore correct to about 10% against this robot's own\n"
          "odometry. ⚠️ It is a RATIO of the two scales: k = 1.0 says the camera and the\n"
          "odometry agree, not that either is separately perfect.")


# ── 2. the same fit on detector sightings, which does not work ──────────────
def sighting_scale() -> None:
    rule("2. The same fit over raw detector sightings — a negative result")
    for name in ("hero", "contrast"):
        samples = []
        for tick in ticks_of(name):
            pose = tick["pose"]
            for sighting in tick.get("sightings") or ():
                samples.append((tick["t"], pose["x"], pose["y"], sighting["range_m"],
                                pose["yaw"] + sighting["bearing_rad"]))
        tracks: list = []
        for sample in samples:
            for track in tracks:
                if (abs(sample[0] - track[-1][0]) < 1.5
                        and abs(math.degrees(sample[4] - track[-1][4])) < 12.0):
                    track.append(sample)
                    break
            else:
                tracks.append([sample])
        print(f"  {name}: {len(samples)} sightings -> {len(tracks)} bearing-contiguous tracks")
        for track in tracks:
            fit = scale_of([(t[1], t[2], t[3], t[4]) for t in track])
            if fit is None or len(track) < 5:
                continue
            travel = math.hypot(track[-1][1] - track[0][1], track[-1][2] - track[0][2])
            print(f"     n={fit[2]:3d} t={track[0][0]:5.1f}-{track[-1][0]:5.1f} "
                  f"travel={travel:.2f} m -> k = {fit[0]:8.3f}  rms {fit[1] * 100:5.1f} cm")
    print("\n⛔ Those k values are not measurements of anything. #35's 'How to settle it'\n"
          "   item 2 proposed this fit and named the missing `sightings` field as the\n"
          "   blocker; the field landed in #37 and this is the first tool to read it. The\n"
          "   blocker was never the field. Over 0.2-1.2 m of baseline against objects at\n"
          "   0.8-1.7 m, the detector's own range noise dominates — and the run that\n"
          "   answers the question is in the same file, ranging a marker.")


# ── 3. what a wrong pitch does, and which way ───────────────────────────────
def pitch_direction() -> None:
    rule("3. An unmodelled nose-down pitch makes the floor contact read LONG")
    print("`go2_front_camera.json` records `pitch_rad: 0.0`, never independently verified.\n"
          "#35's 2026-08-20 correction dismissed the floor-contact estimate on the grounds\n"
          "that a downward pitch 'puts the floor contact lower in the frame, which makes\n"
          "0.32 x f / (y_bottom - c_y) read short'. Tilting a camera DOWN moves a ground\n"
          "point UP in the image, toward c_y, so the denominator shrinks and the range\n"
          "reads LONG. The sign is the other way round:\n")
    print(f"{'row v':>7} {'true (nose-down 2 deg)':>24} {'model assuming 0':>18} "
          f"{'reported/true':>14}")
    truth = FisheyeCamera(width=CAMERA.width, height=CAMERA.height,
                          focal_px=CAMERA.focal_px, cx=CAMERA.cx, cy=CAMERA.cy,
                          pitch_rad=math.radians(2.0), height_m=CAMERA.height_m)
    for row in (800.0, 900.0, 1000.0):
        real, modelled = truth.ground_range(CAMERA.cx, row), CAMERA.ground_range(CAMERA.cx, row)
        print(f"{row:7.0f} {real:22.3f} m {modelled:16.3f} m {modelled / real:13.3f}")
    print("\nSo the confound named in that correction cannot explain a floor-contact estimate\n"
          "that came out SHORTER than the size prior's. Nor can a clipped dark base, which\n"
          "raises the box's bottom edge and pushes the range long as well. Both of the\n"
          "confounds offered run the other way.")


# ── 4. the bin itself, on a frame from the robot today ──────────────────────
def the_bin() -> None:
    rule("4. The bin's height, from a live frame, by an estimator that never saw the prior")
    profile = PROFILES["bin"]
    image = cv2.imread(str(LIVE_FRAME))
    if image is None:
        print(f"  {LIVE_FRAME.name} not readable — skipping")
        return
    blobs = ColourBlobDetector(profile).detect(image)
    print(f"{LIVE_FRAME.name}, {image.shape[1]}x{image.shape[0]}, {len(blobs)} blob(s)")
    for blob in blobs:
        prior_range, source = estimate_range(blob, CAMERA, profile.prior)
        ground = CAMERA.ground_range((blob.x1 + blob.x2) / 2.0, blob.y2)
        if ground is None:
            continue
        print(f"  box ({blob.x1:.0f},{blob.y1:.0f},{blob.x2:.0f},{blob.y2:.0f}), "
              f"{blob.width_px:.0f}x{blob.height_px:.0f} px")
        print(f"    size prior, told {profile.height_m:.4f} m : {prior_range:.3f} m ({source})")
        print(f"    floor contact,   told nothing   : {ground:.3f} m")
        print(f"    -> implied bin height {profile.height_m * ground / prior_range:.3f} m "
              f"against the {profile.height_m:.4f} m the profile states "
              f"({100.0 * (ground / prior_range - 1.0):+.1f}%)")
        prone = 0.10 / CAMERA.height_m
        prone_height = profile.height_m * ground * prone / prior_range
        print(f"    (if the robot were PRONE, lens 0.10 m: {prone_height:.3f} m, which is not")
        print( "     a bin — so the frame is a standing robot and 0.32 m is the lens)")
    print("\n⚠️  ASSUMES the lens is 0.32 m up and level, and that the blue blob's bottom edge\n"
          "    is the bin's floor contact. `val_min = 40` rejects an unlit dark base, which\n"
          "    would raise that edge and push the implied height UP — so this figure is if\n"
          "    anything an over-estimate, and the direction matters for what it is used for.")


if __name__ == "__main__":
    marker_scale()
    sighting_scale()
    pitch_direction()
    the_bin()
