#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Does ranging from the floor contact point survive a walking robot? Measured.

Regenerates every number in this directory's README, and every number the
`camera_model` and `person_detector` docstrings quote for the ground-plane ranger,
from two recorded runs already in this repository. No robot:

    python3 ground_vs_prior.py

THE CLAIM UNDER TEST. `camera_model` argued that the ground-plane intersection is
unusable on this unit because a 2 deg trunk-pitch wobble swings a 3 m estimate from
2.3 m to 4.4 m. The arithmetic is right; this asks whether the conclusion is, at the
ranges a detection actually arrives from.

THE METHOD, and why it needs no ground truth. The `sightings` block of the telemetry
(#37) carries the raw per-detection BOX alongside the range the shipped size-prior
estimator produced. Two independent estimators can therefore be run over the same box:

  * the size prior, whose error is a multiplicative bias when the prior is wrong and is
    otherwise insensitive to pitch;
  * the floor contact point, whose error is the pitch, growing with range.

Within one track of one object the RATIO of the two is constant if both are behaving.
Its MEAN is the prior's error and says nothing about either method's precision. Its
SCATTER is the combined frame-to-frame noise of the two, and so is an upper bound on the
ground ranger's own. Neither reading needs the odometry, which is what makes this
measurable off a recording.

WHAT IT CANNOT SETTLE. Bearing-only triangulation against the robot's own odometry was
tried as a third opinion and is NOT reported as evidence: over these baselines
(0.3-0.6 m of travel against objects at 0.7-1.8 m, with most of the bearing sweep coming
from yaw rather than translation) it is ill-conditioned, and the two runs prefer odometry
scale factors of 1.9 and 0.5. That is a statement about the triangulation, not about
either estimator. Absolute accuracy is still unmeasured.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                      / "robot-stack" / "unitree" / "go2" / "visual_nav"))

from camera_model import FisheyeCamera
from person_detector import Detection, GroundRanger

RUNS = (
    ("hero", pathlib.Path(__file__).resolve().parent.parent
     / "2026-08-25-peer-runs" / "hero-run-telemetry.jsonl"),
    ("contrast", pathlib.Path(__file__).resolve().parent.parent
     / "2026-08-25-peer-runs" / "contrast-run-telemetry.jsonl"),
)

#: The stated pitch wobble the tables below are computed at. NOT a measurement of this
#: robot: it is the figure `camera_model` argues from, kept so the tables answer the
#: argument on its own terms.
WOBBLE_DEG = 2.0

#: What `tracker.RANGE_SIGMA_FRACTION` already budgets for the size-prior source the
#: filter trusts most, used here as the reference tolerance.
TOLERANCE = 0.18

#: A track needs this many sightings before its ratio scatter means anything.
MIN_TRACK = 4


def load(path):
    """``(camera, [(t, sighting), ...])`` for one telemetry file."""
    camera = None
    rows = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("type") == "header":
            spec = record["camera"]
            camera = FisheyeCamera(width=spec["width"], height=spec["height"],
                                   focal_px=spec["focal_px"], cx=spec["width"] / 2.0,
                                   cy=spec["height"] / 2.0, height_m=spec["height_m"])
        for sighting in record.get("sightings") or []:
            rows.append((record.get("t", 0.0), sighting))
    return camera, rows


def overlap(box_a, box_b) -> float:
    """Intersection over union of two boxes."""
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inner = max(0.0, right - left) * max(0.0, bottom - top)
    union = ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
             + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - inner)
    return inner / union if union > 0.0 else 0.0


def tracks(rows, iou_gate=0.4, dt_gate=0.6):
    """Group sightings of one object by box overlap and time, greedily.

    Deliberately crude: the point is to compare two estimators over the same object, and
    a track that merges two objects shows up as extra scatter, i.e. in the conservative
    direction for the conclusion being drawn.
    """
    grouped = []
    for time_s, sighting in rows:
        best, index = 0.0, -1
        for candidate, group in enumerate(grouped):
            last_t, last = group[-1]
            if time_s - last_t > dt_gate:
                continue
            score = overlap(last["box"], sighting["box"])
            if score > iou_gate and score > best:
                best, index = score, candidate
        if index >= 0:
            grouped[index].append((time_s, sighting))
        else:
            grouped.append([(time_s, sighting)])
    return grouped


def geometry_tables():
    """The error model, on the measured Go2 camera. No telemetry involved."""
    camera = FisheyeCamera(width=1920, height=1080, focal_px=1290.1637909789656,
                           cx=960.0, cy=540.0, height_m=0.32)
    wobble = math.radians(WOBBLE_DEG)
    print("== the near wall: where the floor contact leaves the frame ==")
    print(f"  centre column {camera.ground_range(camera.cx, camera.height - 1.0):.3f} m"
          f"   frame corner {camera.ground_range(0.0, camera.height - 1.0):.3f} m")
    print(f"\n== far-bound error at {WOBBLE_DEG:.0f} deg of pitch wobble ==")
    print(f"  {'range m':>8} {'near m':>8} {'far m':>8} {'far err':>8}")
    for range_m in (0.72, 1.0, 1.33, 1.5, 2.0, 2.7, 3.0):
        nearest, farthest = camera.ground_range_bounds(range_m, wobble)
        print(f"  {range_m:8.2f} {nearest:8.3f} {farthest:8.3f} "
              f"{(farthest / range_m - 1.0) * 100:7.1f}%")
    print("\n== usable ceiling, metres: stated pitch error against stated tolerance ==")
    tolerances = (0.18, 0.25, 0.30, 0.40, 0.50)
    print(f"  {'pitch':>7} " + " ".join(f"{'eps=' + format(eps, '.2f'):>9}"
                                        for eps in tolerances))
    near_wall = camera.ground_range(camera.cx, camera.height - 1.0)
    for degrees in (1.0, 1.6, 2.0, 3.0, 4.5, 6.0):
        cells = []
        for eps in tolerances:
            limit = camera.ground_range_limit(math.radians(degrees), eps)
            cells.append(f"{limit:9.2f}" if limit > near_wall else f"{'none':>9}")
        print(f"  {degrees:6.1f}d " + " ".join(cells))
    print("  (`none` = the ceiling is inside the near wall, so the band is empty and")
    print("   `GroundRanger` refuses to construct at all)")


def estimator_agreement():
    """The measurement: two independent estimators over the same recorded boxes."""
    angular = []
    print("\n== two estimators, one box, per track ==")
    print("  Both are run over the same recorded detection. `ratio` is ground/size-prior:")
    print("  its MEAN is the size prior's error, its SCATTER bounds the ground ranger's.")
    total = 0
    for name, path in RUNS:
        camera, rows = load(path)
        # Only vertically unclipped boxes: a clipped one has no contact point, and the
        # size prior has fallen back to width or to a constant, so neither estimator is
        # reporting a measurement.
        rows = [(t, s) for t, s in rows if s["source"] == "height"]
        wide = GroundRanger(camera, math.radians(1.6), 3.0)
        shipped = GroundRanger(camera, math.radians(1.6), TOLERANCE)
        print(f"\n  {name} run — {len(rows)} vertically unclipped sightings")
        for group in tracks(rows):
            items = []
            for time_s, sighting in group:
                box = sighting["box"]
                detection = Detection(box[0], box[1], box[2], box[3],
                                      sighting["score"], sighting["label"])
                ground, _source = wide(detection)
                if not math.isfinite(ground):
                    continue
                items.append((time_s, sighting, detection, ground))
            if len(items) < MIN_TRACK:
                continue
            total += len(items)
            ratio = np.array([g / s["range_m"] for _t, s, _d, g in items])
            heights = np.array([wide.implied_height_m(d) for _t, _s, d, _g in items])
            kept = [d for _t, _s, d, _g in items
                    if math.isfinite(shipped(d)[0])]
            labels = sorted({s["label"] for _t, s, _d, _g in items})
            print(f"    t={items[0][0]:5.2f}-{items[-1][0]:5.2f}  n={len(items):2d}  "
                  f"labels={','.join(labels)}")
            scatter = np.log(ratio).std(ddof=1) * 100.0
            print(f"      ratio mean {ratio.mean():5.3f}  log-sd {scatter:4.1f}%"
                  f"   implied height {heights.mean():5.3f} m "
                  f"(sd {heights.std(ddof=1):.3f})"
                  f"   kept by 1.6deg/{TOLERANCE:.0%}: {len(kept)}/{len(items)}")
            # Attribute the within-track residual entirely to the ground ranger's
            # elevation error. That over-attributes -- the size prior's own box-height
            # noise is in there too -- so what comes out is an UPPER BOUND.
            residual = np.log(ratio) - np.log(ratio).mean()
            for (_t, _s, _d, ground), r in zip(items, residual):
                sensitivity = ground / camera.height_m + camera.height_m / ground
                angular.append(abs(r) / sensitivity)
    bound = np.array(angular)
    print(f"\n== implied per-frame angular error, {len(bound)} sightings ==")
    print("  Upper bound, not a measurement of trunk pitch: the size prior's own")
    print("  box-height noise is inside this number and cannot be separated out.")
    print(f"  median {math.degrees(np.median(bound)):.2f} deg   "
          f"p90 {math.degrees(np.percentile(bound, 90)):.2f} deg   "
          f"max {math.degrees(bound.max()):.2f} deg")
    print(f"  ({total} sightings in tracks of at least {MIN_TRACK})")


def main() -> int:
    geometry_tables()
    estimator_agreement()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
