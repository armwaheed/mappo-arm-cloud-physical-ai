#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recompute every number in this directory's README, from the committed JSON.

    python3 audit.py

THE CLAIMS UNDER TEST, in the order they decide what the footage is worth.

  1. The recordings are RAW and the telemetry's documented join does not exist:
     ``perception.video_frame`` is null in every tick of all six files, so the frame the
     detector saw has to be recovered from ``frame_age_s``.
  2. All six are TRIPOD SHOTS. "in different distance and angle" describes the subject,
     not the camera, so the set adds at most six viewpoints and no augmentation adds a
     seventh.
  3. The telemetry cannot label this set -- 3 boxes across 2,701 frames of quadruped
     footage -- and the reason is the CLASS FILTER, not the network. Re-running the same
     weights without it recovers boxes on the Lite3.
  4. The dim quadruped scene has to be dropped wholesale, because its illumination sweeps
     and the background-subtraction gate fails OPEN on it.

WHAT IT NEEDS. numpy, and the JSON files beside it. No video, no model, no network, no GPU.
``measure_scenes.py`` is the script that read the video; ``scene_measurements.json`` is what
it wrote. Any line here that disagrees with the README is a bug in one of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
#: person_detector.PERSON_ASPECT_MIN, restated so this script needs no import.
PERSON_ASPECT_MIN = 2.0


def rule(text: str) -> None:
    print(f"\n{text}\n" + "=" * len(text))


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def main() -> None:
    m = load("scene_measurements.json")
    hand = load("handcheck.json")
    scenes = m["scenes"]

    rule("1. What the six recordings are")
    print(f"  {'scene':46s}{'frames':>7s}{'ticks':>7s}{'sightings':>10s}"
          f"{'video_frame non-null':>22s}")
    total_frames = total_ticks = total_sight = total_vf = 0
    for name, block in scenes.items():
        t = block["telemetry"]
        print(f"  {name[:44]:46s}{block['n_frames']:>7d}{t['ticks']:>7d}"
              f"{t['sightings']:>10d}{t['video_frame_non_null']:>22d}")
        total_frames += block["n_frames"]
        total_ticks += t["ticks"]
        total_sight += t["sightings"]
        total_vf += t["video_frame_non_null"]
    print(f"  {'TOTAL':46s}{total_frames:>7d}{total_ticks:>7d}{total_sight:>10d}"
          f"{total_vf:>22d}")
    print(f"  -> {total_frames} frames, {total_ticks} ticks. `perception.video_frame` is "
          f"non-null in {total_vf} of {total_ticks} ticks:")
    print("     the documented join DOES NOT EXIST. frame = round((t - frame_age_s) * 15).")
    headers = {json.dumps(b["telemetry"]["header_classes"]) for b in scenes.values()}
    confs = {b["telemetry"]["header_confidence"] for b in scenes.values()}
    print(f"  -> every header reads classes={headers.pop()} confidence={confs.pop()}"
          "  = go2-navigator-default")

    rule("2. The camera does not move — all six are tripod shots")
    print(f"  {'scene':46s}{'median px':>11s}{'p90':>8s}{'max':>8s}")
    for name, block in scenes.items():
        d = np.array([v for v in block["camera_displacement_px"] if np.isfinite(v)])
        print(f"  {name[:44]:46s}{np.median(d):>11.2f}{np.percentile(d, 90):>8.2f}"
              f"{d.max():>8.2f}")
    print("  -> the previous clip measured 0.19 px median and was called a tripod shot.")
    print("     These are the same instrument on the same claim: 'different distance and")
    print("     angle' is the SUBJECT moving. The set holds at most SIX viewpoints.")

    rule("3. Illumination — why one scene is dropped and three are not trusted for motion")
    print(f"  {'scene':46s}{'mean':>7s}{'std':>7s}{'min':>7s}{'max':>7s}{'max step':>10s}")
    for name, block in scenes.items():
        lum = np.array(block["luminance"])
        print(f"  {name[:44]:46s}{lum.mean():>7.1f}{lum.std():>7.2f}{lum.min():>7.1f}"
              f"{lum.max():>7.1f}{np.abs(np.diff(lum)).max():>10.2f}")
    print("  -> the three dim scenes are not merely darker, they are UNSTABLE. A background")
    print("     median is not a background when the room's brightness triples inside the")
    print("     clip, and a motion gate built on it fails open.")

    rule("4. The class filter, not the network, is why there are no Lite3 boxes")
    print(f"  {'scene':46s}{'telemetry person':>18s}{'re-run person':>15s}"
          f"{'re-run chair/sofa':>19s}")
    for name, block in scenes.items():
        tel = sum(1 for s in block["telemetry_sightings"] if s["label"] == "person")
        person = sum(1 for f in block["frames"] for d in f["det"]
                     if d["cls"] == "person" and d["score"] >= 0.25)
        chair = sum(1 for f in block["frames"] for d in f["det"]
                    if d["cls"] in ("chair", "sofa") and d["score"] >= 0.25)
        print(f"  {name[:44]:46s}{tel:>18d}{person:>15d}{chair:>19d}")
    lite3_scenes = [n for n in scenes if "-lite3-" in n]
    tel_l3 = sum(len(scenes[n]["telemetry_sightings"]) for n in lite3_scenes)
    frames_l3 = sum(scenes[n]["n_frames"] for n in lite3_scenes)
    print(f"  -> across {frames_l3} frames of quadruped footage the telemetry carries "
          f"{tel_l3} boxes.")
    print("     The same weights over the same pixels, class filter removed, return the")
    print("     Lite3 as `chair` — which is what the previous audit predicted at 300 px.")

    rule("5. Distinct views — 5,854 frames are not 5,854 samples")
    dv = load("distinct_views.json")
    print(f"  {'scene':46s}{'frames':>8s}{'distinct':>10s}{'%':>7s}")
    for name, block in sorted(dv["scenes"].items()):
        print(f"  {name[:44]:46s}{block['frames']:>8d}{block['distinct']:>10d}"
              f"{100 * block['distinct'] / block['frames']:>7.1f}")
    print(f"  {'TOTAL':46s}{dv['total_frames']:>8d}{dv['total_distinct']:>10d}"
          f"{100 * dv['total_distinct'] / dv['total_frames']:>7.1f}")
    print(f"  -> rule: every {dv['method']['step']}th frame, {dv['method']['thumbnail']} grey,")
    print(f"     kept when it differs from the {dv['method']['compared_against']} by more")
    print(f"     than {dv['method']['threshold_mean_grey']} mean grey levels. The knobs are")
    print("     crude and the count moves with them; both are reported together.")

    rule("6. What the labeller produced, and what it is allowed to decide")
    sam = load("sam_labels.json")
    queries = load("scene_queries.json")
    print(f"  {'scene':46s}{'keyfrm':>8s}{'owl':>6s}{'labels':>8s}  by class")
    for name, block in sorted(sam["stats"].items()):
        print(f"  {name[:44]:46s}{block['keyframes']:>8d}{block['owl_boxes']:>6d}"
              f"{block['labels']:>8d}  {block['by_class']}")
    print(f"  -> {sam['count']} labels. detector {sam['detector']}, segmenter "
          f"{sam['segmenter']}.")
    print("  -> NEITHER NAMES A CLASS. Every label comes from scene_queries.json — the")
    print("     folder name plus the phrase prompted:")
    for name, block in sorted(queries["scenes"].items()):
        for query in block["queries"]:
            print(f"       {name[:40]:42s} {query['text']!r:38s} -> {query['label']}"
                  f"  @{query['threshold']}")

    rule("7. The phrase sweep — the natural phrase is the worst one")
    sweep = load("phrase_sweep.json")
    print(f"  max OWLv2 score over 5 keyframes per clip, threshold "
          f"{sweep['threshold']}, {sweep['model']}")
    print(f"  {'phrase':38s}{'light-lite3':>13s}{'dim-lite3':>12s}")
    for phrase in sweep["phrases"]:
        cells = []
        for scene in sorted(sweep["scenes"]):
            hits = [row["score"] for row in sweep["scenes"][scene][phrase]]
            cells.append(max(hits))
        # scenes sort dim before light; print light first to match the README table
        print(f"  {phrase:38s}{cells[1]:>13.3f}{cells[0]:>12.3f}")
    print("  -> a phrase that reads like the object's NAME scores 0.000; one that reads")
    print("     like its DESCRIPTION scores 0.31-0.63. That is the difference between 0")
    print("     and 60 labelled frames in light-lite3.")

    rule("8. What was looked at by hand — the number that decides everything")
    hand = load("handcheck.json")
    for check in hand["checks"]:
        print(f"  {check['route'][:58]:60s} {check['on_subject']:>3d}/{check['sampled']:<3d}")
        print(f"      {check['verdict']}")
    print(f"  -> shipped pipeline: {hand['totals']['shipped_pipeline_on_subject']}"
          f"/{hand['totals']['shipped_pipeline_sampled']} on-subject.")

    rule("9. Real, synthetic, and near-duplicate")
    train = load("lite3_train_20260827.json")
    aug = load("lite3_train_aug_20260827.json")
    ev = load("lite3_eval_20260827.json")
    person = load("person_20260827.json")
    negatives = load("negatives_20260827.json")
    drift = load("neighbour_drift.json")
    families: dict = {}
    for record in aug["records"]:
        families[record["derivation"]] = families.get(record["derivation"], 0) + 1
    keyframe_boxes = sum(b["by_class"].get("lite3", 0) for b in sam["stats"].values())
    print(f"  lite3 keyframe boxes               {keyframe_boxes:>6d}")
    print(f"  lite3 train (with +/-1 ride-along)  {train['count']:>6d}")
    print(f"  lite3 eval  (held-out time block)   {ev['count']:>6d}")
    print(f"  person boxes (not a training input) {person['count']:>6d}")
    print(f"  in-domain negatives                 {negatives['count']:>6d}")
    print(f"  train + synthetic                   {aug['count']:>6d}   "
          f"ratio {aug['source']['ratio']}")
    for name, count in sorted(families.items()):
        print(f"    {name:32s}{count:>6d}")
    added = aug["source"]["viewpoints_added_by_synthesis"]
    print(f"  viewpoints added by synthesis       {added:>6d}")
    print("  ride-along IoU against the keyframe's own box:")
    for offset, row in sorted(drift["by_offset"].items(), key=lambda kv: int(kv[0])):
        print(f"    +{offset} frame  n={row['n']:>3d}  median IoU {row['median_iou']:.3f}"
              f"  p10 {row['p10']:.3f}  below 0.75: {row['below_0_75']}")
    print("  -> +/-1 is shipped (median 0.954). +4 has a p10 of 0.496, i.e. one box in ten")
    print("     has moved off its object, so nothing wider rides along.")

    rule("10. What is same-session, and therefore unproven")
    print("  Every clip: 2026-08-27, one room, inside 13 minutes. Every clip a tripod shot.")
    print(f"  {dv['total_distinct']} distinct views is the ceiling; augmentation multiplies")
    print("  EXAMPLES, not viewpoints, and adds 0 rooms and 0 days.")
    print("  This project has measured 0/705 same-session against 60/159 cross-day for one")
    print("  model. The lite3 split here is same-session. It does not predict tomorrow.")


if __name__ == "__main__":
    main()
