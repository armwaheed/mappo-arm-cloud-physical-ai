#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""What the robot should do when the camera can no longer say where anything is.

Regenerates every number in this directory's README from files already in this
repository. No robot, no network:

    python3 no_open_bearing.py

THREE THINGS ARE MEASURED HERE.

1. THE GEOMETRY OF ISSUE #72, from the shipped calibration rather than from the issue
   text. Below a range that depends only on the lens and the mounting height, a single
   object fills the cone and there is no bearing left to steer at.

2. WHAT THAT LOOKS LIKE IN A RECORDED RUN. The `sightings` block (#37) carries the raw
   per-detection box and the `source` that produced its range. Four of those sources are
   not measurements — two are constants the shipped estimator RETURNS, two are refusals
   the ground ranger makes — and the boxes carrying them are exactly the boxes the robot
   could not locate. How much of the cone they leave open is answerable from the box
   alone, with no range and no size prior, which is the point: it is answerable exactly
   when ranging is not.

3. WHY THE FIT NAMED `contact_row = 0.472 x box_height + 0.585` IS NOT THE ESTIMATOR.
   It is a data-augmentation placement rule from `../2026-08-26-checkpoint-sweep/`, and
   it predicts the contact row FROM the box height — which is a size prior, fitted, and
   over one object class. Read as a ranger it is worse than the one that already exists,
   and this section shows by how much.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "robot-stack" / "unitree" / "go2" / "visual_nav"))

from camera_model import FisheyeCamera
from person_detector import (
    UNRANGEABLE_SOURCES,
    Detection,
    steerable_bearing_rad,
    widest_open_bearing_rad,
)

CAMERA = FisheyeCamera.load(
    str(ROOT / "robot-stack" / "unitree" / "go2" / "visual_nav" / "go2_front_camera.json"))
RUNS = ROOT / "evidence" / "2026-08-25-peer-runs"
LABELS = ROOT / "detector" / "labels" / "peer_go2wheel_20260824.json"

#: Half-extent of a Go2 presenting broadside, from `integration/peer_source.py:192`
#: (`hypot(0.35, 0.155)`): 0.35 is the half-LENGTH, 0.155 the half-width.
PEER_HALF_EXTENT_M = 0.35
#: The planner's shipped `robot_radius_m` (`avoidance.PlannerConfig`).
ROBOT_RADIUS_M = 0.40


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ── 1. the two floors, from the calibration ─────────────────────────────────
def geometry() -> None:
    rule("1. Where the geometry runs out, from go2_front_camera.json")
    half_fov = math.radians(CAMERA.hfov_deg) / 2.0
    print(f"focal {CAMERA.focal_px:.2f} px, HFOV {CAMERA.hfov_deg:.3f} deg, "
          f"lens {CAMERA.height_m:.2f} m off the floor")

    blind = PEER_HALF_EXTENT_M / math.sin(half_fov)
    print(f"\nAPERTURE floor (issue #72): a {PEER_HALF_EXTENT_M} m half-extent fills the "
          f"cone at {blind:.3f} m")
    print(f"{'peer range':>11} {'arc subtended':>14} {'open either side':>17}")
    for range_m in (2.0, 1.5, 1.0, 0.76, blind, 0.45):
        arc = 2.0 * math.asin(min(PEER_HALF_EXTENT_M / range_m, 1.0))
        open_side = math.degrees(half_fov - arc / 2.0)
        print(f"{range_m:11.3f} {math.degrees(arc):13.1f}d {open_side:16.1f}d")

    centre = CAMERA.ground_range(CAMERA.cx, CAMERA.height - 1.0)
    corner = CAMERA.ground_range(CAMERA.width - 1.0, CAMERA.height - 1.0)
    print(f"\nCONTACT-POINT floor: the floor contact leaves the frame at {centre:.3f} m on "
          f"the centre-line\nand at {corner:.3f} m at the frame corner, where the ray is "
          f"shallower. The corner is the\none that binds, and it is 0.09 m further out "
          f"than the figure this repository quotes.")
    print(f"\nThe contact-point floor is the LARGER of the two, so a robot with working "
          f"ranging\nloses its range at {corner:.3f} m and never reaches {blind:.3f} m "
          f"with anything to steer by.")

    rule("   ...and the arc that is still worth steering at")
    print(f"{'robot_radius_m':>15} {'asin(r/near)':>13} {'2*asin(r/near)':>15}")
    for radius in (0.25, ROBOT_RADIUS_M):
        one = math.degrees(steerable_bearing_rad(CAMERA, radius))
        print(f"{radius:15.2f} {one:12.2f}d {2.0 * one:14.2f}d")
    print("\nThe shipped floor is the FIRST column, and the second is the stricter test it\n"
          "is not: this asks whether a steerable bearing exists, not whether the robot fits\n"
          "through it. Section 2 prices both against recorded data.")


# ── 2. the recorded runs ────────────────────────────────────────────────────
def _boxes(tick: dict) -> list:
    return [Detection(x1=s["box"][0], y1=s["box"][1], x2=s["box"][2], y2=s["box"][3],
                      score=s["score"], label=s["label"])
            for s in tick.get("sightings") or () if s["source"] in UNRANGEABLE_SOURCES]


def recorded() -> None:
    rule("2. Every unrangeable sighting in the two live runs of 2026-08-25")
    print(f"sources that are not a measurement: {', '.join(UNRANGEABLE_SOURCES)}")
    print("`ground-far` is deliberately excluded — it refuses a DISTANT object, and\n"
          "counting it would fire this on 22 of the hero run's 59 ticks instead of 1.\n")
    floors = {radius: steerable_bearing_rad(CAMERA, radius) for radius in (0.25, ROBOT_RADIUS_M)}
    totals = {"ticks": 0, "carrying": 0}
    rows = []
    for name in ("hero", "contrast"):
        with open(RUNS / f"{name}-run-telemetry.jsonl", encoding="utf-8") as handle:
            for line in handle:
                tick = json.loads(line)
                if tick.get("type") != "tick":
                    continue
                totals["ticks"] += 1
                boxes = _boxes(tick)
                if not boxes:
                    continue
                totals["carrying"] += 1
                rows.append((name, tick["t"], math.degrees(widest_open_bearing_rad(boxes, CAMERA)),
                             [s["source"] for s in tick["sightings"]
                              if s["source"] in UNRANGEABLE_SOURCES]))
    print(f"{'run':>9} {'t':>6} {'widest open bearing':>20}  sources")
    for name, t, open_deg, sources in sorted(rows, key=lambda r: r[2]):
        mark = "  <-- HELD" if open_deg < math.degrees(floors[ROBOT_RADIUS_M]) else ""
        print(f"{name:>9} {t:6.2f} {open_deg:19.2f}d  {','.join(sources)}{mark}")

    print(f"\n{totals['carrying']} of {totals['ticks']} ticks carry a sighting the stack "
          f"could not locate.")
    for radius, floor in floors.items():
        fired = [r for r in rows if r[2] < math.degrees(floor)]
        passed = [r for r in rows if r[2] >= math.degrees(floor)]
        strict = [r for r in rows if r[2] < 2.0 * math.degrees(floor)]
        print(f"  radius {radius:.2f} m -> floor {math.degrees(floor):5.2f}d : "
              f"holds {len(fired):2d}, drives {len(passed):2d}"
              f"   (the 2x variant would hold {len(strict)})")
    fired = [r[2] for r in rows if r[2] < math.degrees(floors[ROBOT_RADIUS_M])]
    passed = [r[2] for r in rows if r[2] >= math.degrees(floors[ROBOT_RADIUS_M])]
    print(f"\nSEPARATION at the shipped radius: held ticks leave "
          f"{min(fired):.2f}-{max(fired):.2f}d,\n"
          f"driven ticks leave {min(passed):.2f}-{max(passed):.2f}d, so the floor of "
          f"{math.degrees(floors[ROBOT_RADIUS_M]):.2f}d sits\nwith {min(passed) / max(fired):.2f}x "
          f"between the two populations. That is a real gap and it is also only\n16 ticks; the "
          f"floor is derived from the lens and the robot rather than chosen to fit it,\nwhich is "
          f"what makes the number auditable rather than tuned.")
    print("\n⚠️  NEITHER RUN CONTAINS THE COLLISION THIS GUARD IS FOR. `live05` — the peer at\n"
          "0.45 m with 0/91 open windows — is not committed to this repository. What is\n"
          "measured here is that the threshold discriminates on the runs that ARE committed,\n"
          "and that it does not stop the hero run's successful approach: of its 59 ticks it\n"
          "holds 1, at t=17.53, 0.9 s before the run arrived.")


# ── 3. the fit that is not a ranger ─────────────────────────────────────────
def _fit(boxes, frame_height: int = 1080):
    x = [(b[3] - b[1]) / frame_height for b in boxes]
    y = [b[3] / frame_height for b in boxes]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    slope = (sum((a - mx) * (c - my) for a, c in zip(x, y))
             / sum((a - mx) ** 2 for a in x))
    intercept = my - slope * mx
    sse = sum((c - (slope * a + intercept)) ** 2 for a, c in zip(x, y))
    sst = sum((c - my) ** 2 for c in y)
    return slope, intercept, 1.0 - sse / sst, n


def the_fit() -> None:
    rule("3. `contact_row = 0.472 x box_height + 0.585`, r2 = 0.702, n = 1,256")
    records = json.loads(LABELS.read_text())["records"]
    inside = [r["box"] for r in records if r["box"][3] < CAMERA.height]
    slope, intercept, r2, n = _fit(inside)
    print(f"reproduced: contact_row = {slope:.3f} x box_height + {intercept:.3f}   "
          f"r2 = {r2:.3f}, n = {n:,}")

    counts: dict = {}
    for box in inside:
        counts[tuple(box)] = counts.get(tuple(box), 0) + 1
    distinct = sorted(counts)
    print(f"\n⛔ n = {n:,} IS {len(distinct)} DISTINCT BOXES, and the four commonest "
          f"account for {sum(sorted(counts.values())[-4:]):,} of them.")
    print(f"   The single commonest box is repeated {max(counts.values())} times — one "
          f"observation, counted {max(counts.values())} times.")
    slope, intercept, r2, n = _fit(distinct)
    print(f"   on distinct boxes: {slope:.3f} x + {intercept:.3f}, r2 = {r2:.3f}, n = {n}")

    print("\n⛔ AND THE FIT IS BETWEEN CLIPS, NOT WITHIN THEM. Split by capture clip, the\n"
          "   relationship it claims to have measured is absent:")
    clips: dict = {}
    for record in records:
        if record["box"][3] < CAMERA.height:
            clips.setdefault("_".join(record["image"].split("_")[:-1]), []).append(record["box"])
    print(f"   {'clip':>34} {'n':>5} {'distinct':>9} {'slope':>8} {'r2':>8}")
    for clip, boxes in sorted(clips.items()):
        unique = len({tuple(b) for b in boxes})
        if unique < 3:
            print(f"   {clip:>34} {len(boxes):5d} {unique:9d} {'—':>8} {'—':>8}")
            continue
        slope, _intercept, r2, _n = _fit(boxes)
        print(f"   {clip:>34} {len(boxes):5d} {unique:9d} {slope:8.3f} {r2:8.3f}")

    print("\n   Read as a RANGER — predict the contact row from the box height, then\n"
          "   intersect that row with the floor — here is what it costs, against the row the\n"
          "   box actually has:")
    rows = []
    for box in inside:
        actual = CAMERA.ground_range((box[0] + box[2]) / 2.0, box[3])
        predicted_row = (0.472 * (box[3] - box[1]) / CAMERA.height + 0.585) * CAMERA.height
        predicted = CAMERA.ground_range((box[0] + box[2]) / 2.0, predicted_row)
        if actual and predicted and actual > 0.0:
            rows.append((actual, abs(predicted / actual - 1.0)))
    print(f"   {'true range':>16} {'n':>6} {'median error':>14} {'p90':>8} {'worst':>8}")
    for low, high in ((0.72, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.7), (2.7, 1e9)):
        band = sorted(e for r, e in rows if low <= r < high)
        if not band:
            continue
        label = f"{low:.2f}-{high:.2f} m" if high < 1e9 else f"{low:.2f} m +"
        print(f"   {label:>16} {len(band):6d} {band[len(band) // 2] * 100:13.1f}% "
              f"{band[int(0.9 * len(band))] * 100:7.1f}% {band[-1] * 100:7.1f}%")
    print("\n   0.72-1.00 m is the band that sets stopping distance, and it is the band the\n"
          "   fit is worst in. `camera_model.ground_range` needs no fit at all there: when the\n"
          "   contact row is in frame it is READ, and when it is not, no fit recovers it.")


if __name__ == "__main__":
    geometry()
    recorded()
    the_fit()
