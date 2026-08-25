#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Score a written ``.caffemodel`` on held-out frames, through the real DetectionOutput.

Everything here runs the candidate the way the robot does -- ``cv2.dnn.readNetFromCaffe``,
one forward pass, the prototxt's own NMS and ``keep_top_k`` -- because a metric computed
from a torch-side decode agrees with the trainer by construction and says nothing about
what gets deployed.

## The manifest carries three states, not two

``present: true`` with a box, ``present: false``, and ``present: null`` for a frame nobody
can honestly call either way. The third state exists because it was needed: of the fifteen
Aug-20 stills this repository has been calling "held-out peer frames", inspection at 4x
found a peer in six, no peer at all in eight, and one frame (``arrive_08``) showing only the
peer's lit LED bar against a shadowed pillar. Scoring the eight as positives is what
produced a 53% recall figure -- and the same model's false-positive rate was 38%, so most
of that "recall" was the false-positive rate arriving under a different name.

## Two recall definitions, and the gap between them is the interesting number

``fires`` is what the earlier work measured: the class scored above threshold ANYWHERE in
the frame. ``hits`` requires the detection to overlap the labelled box at IoU 0.5. On a
detector with a high false-positive rate the first is inflated by exactly that rate, and the
distance between the two columns is how much of the "recall" is the detector finding the
robot rather than finding something.

## --mask-overlay

The Aug-20 frames were recorded with the navigator's overlay composited in: a plan-view
radar inset top-right and a status plate bottom-left, neither of which appears in any
training frame. ``fix/ceiling-overlay-caveat`` measured the confound at four points of
false-positive rate. The flag paints both fixed regions with the frame's own median colour
so the number can be quoted both ways. It does NOT remove the overlay's detection
rectangles, which are drawn wherever the incumbent detector fired and cannot be masked
without masking the objects too -- so "masked" is an upper bound on the fix, not the fix.

Usage:

    eval_detector.py --proto m.prototxt --model m.caffemodel --manifest heldout.json
    eval_detector.py --proto m.prototxt --models 'runs/*.caffemodel' --manifest heldout.json
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import cv2
import numpy as np

from add_class import VOC_CLASSES

#: Square network input and the preprocessing baked into the weights.
INPUT_SIZE = 300
SSD_SCALE, SSD_MEAN = 1.0 / 127.5, 127.5

#: Thresholds the table reports. Nothing below 0.25 is reachable: the prototxt's
#: ``detection_output_param`` carries ``confidence_threshold: 0.25``, so DetectionOutput has
#: already discarded weaker boxes before ``forward()`` returns. Sweeping lower without
#: editing the deployed prototxt would report zeros and call them a result.
THRESHOLDS = (0.25, 0.50, 0.70, 0.90, 0.99)

#: Fixed overlay regions, as fractions of the frame, measured off the Aug-20 stills:
#: the plan-view radar inset (top-right) and the status plate (bottom-left).
RADAR_REGION = (0.834, 0.0, 1.0, 0.287)
PLATE_REGION = (0.0, 0.930, 0.215, 1.0)


def _iou(box: np.ndarray, other: np.ndarray) -> float:
    x0, y0 = max(box[0], other[0]), max(box[1], other[1])
    x1, y1 = min(box[2], other[2]), min(box[3], other[3])
    overlap = max(x1 - x0, 0) * max(y1 - y0, 0)
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (other[2] - other[0]) * (other[3] - other[1]) - overlap)
    return overlap / union if union > 0 else 0.0


def mask_overlay(image: np.ndarray) -> np.ndarray:
    """Paint the two fixed overlay regions with the frame's own median colour."""
    out = image.copy()
    height, width = out.shape[:2]
    fill = np.median(out.reshape(-1, 3), axis=0)
    for x0, y0, x1, y1 in (RADAR_REGION, PLATE_REGION):
        out[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)] = fill
    return out


def detections(net, image: np.ndarray) -> np.ndarray:
    """``(label, score, xmin, ymin, xmax, ymax)`` rows in PIXELS, from DetectionOutput."""
    height, width = image.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(image, SSD_SCALE, (INPUT_SIZE, INPUT_SIZE), SSD_MEAN))
    rows = net.forward()[0, 0]
    out = np.zeros((len(rows), 6), np.float32)
    out[:, 0] = rows[:, 1]
    out[:, 1] = rows[:, 2]
    out[:, 2:] = rows[:, 3:7] * [width, height, width, height]
    return out


def score_model(proto: Path, model: Path, frames: list, new_class: int,
                masked: bool) -> dict:
    """Per-frame detections for the new class plus ``person``'s best score."""
    net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
    person = VOC_CLASSES.index("person")
    results = []
    for frame in frames:
        image = cv2.imread(str(frame["image"]))
        if image is None:
            raise FileNotFoundError(frame["image"])
        if masked:
            image = mask_overlay(image)
        rows = detections(net, image)
        mine = rows[rows[:, 0] == new_class]
        results.append({
            "image": str(frame["image"]),
            "present": frame["present"],
            "box": frame.get("box"),
            "scores": mine[:, 1].tolist(),
            "boxes": mine[:, 2:].tolist(),
            "person": float(rows[rows[:, 0] == person][:, 1].max())
            if (rows[:, 0] == person).any() else 0.0,
        })
    return {"model": str(model), "frames": results}


def table(scored: dict, thresholds=THRESHOLDS) -> list:
    """Recall / false-positive rate / precision at each threshold, frame- and box-level."""
    positives = [f for f in scored["frames"] if f["present"] is True]
    negatives = [f for f in scored["frames"] if f["present"] is False]
    # ⚠️ THE TWO RECALLS HAVE DIFFERENT DENOMINATORS, and conflating them understates
    # localisation by whatever fraction of the positives carries no box. Presence/absence is
    # cheap to label over a whole clip; a box is not, so most positives here are labelled
    # `present` with `box: null`. Dividing hits by ALL positives made a model that localised
    # one of the two boxed frames report "8% recall".
    boxed = [f for f in positives if f["box"]]
    rows = []
    for threshold in thresholds:
        fires = hits = 0
        for frame in positives:
            pairs = zip(frame["scores"], frame["boxes"], strict=True)
            kept = [(s, b) for s, b in pairs if s >= threshold]
            if kept:
                fires += 1
            if frame["box"] and any(_iou(np.array(b), np.array(frame["box"])) >= 0.5
                                    for _, b in kept):
                hits += 1
        false = sum(1 for f in negatives if any(s >= threshold for s in f["scores"]))
        rows.append({
            "threshold": threshold,
            "positives": len(positives), "negatives": len(negatives),
            "boxed": len(boxed),
            "fires": fires, "hits": hits, "false_positive_frames": false,
            "fire_recall": fires / max(len(positives), 1),
            "hit_recall": hits / max(len(boxed), 1),
            "false_positive_rate": false / max(len(negatives), 1),
            "fire_precision": fires / max(fires + false, 1),
            "hit_precision": hits / max(hits + false, 1),
        })
    return rows


def person_shift(reference: dict, candidate: dict, floor: float = 0.5) -> dict:
    """How far ``person`` moved, on the frames where the REFERENCE model still sees one.

    ``person`` is the one class on this robot's stop path and the one class MobileNet-SSD is
    actually good at (0.93-0.97 on this robot's own footage). Unfreezing the backbone moves
    every class at once, so "the new class works" is only half a result: the other half is
    whether a fine-tune has quietly cost the robot its ability to stop for people.

    Frames where the reference scores below ``floor`` are excluded rather than counted as
    agreement -- averaging over frames with no person in them dilutes the answer with
    hundreds of near-zeros and would report "person barely moved" whatever happened.
    """
    pairs = [(r["person"], c["person"])
             for r, c in zip(reference["frames"], candidate["frames"], strict=True)
             if r["person"] >= floor]
    if not pairs:
        return {"frames": 0}
    before = np.array([p[0] for p in pairs])
    after = np.array([p[1] for p in pairs])
    return {"frames": len(pairs), "reference_mean": float(before.mean()),
            "candidate_mean": float(after.mean()),
            "worst_drop": float((before - after).max()),
            "lost": int((after < floor).sum())}


def render(rows: list, title: str) -> str:
    out = [f"### {title}", "",
           "| threshold | recall (fires) | recall (IoU>=0.5) | FP rate | precision (fires) |"
           " precision (IoU) |", "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        out.append(
            f"| {row['threshold']:.2f} "
            f"| {row['fire_recall']:.0%} ({row['fires']}/{row['positives']}) "
            f"| {row['hit_recall']:.0%} ({row['hits']}/{row['boxed']}) "
            f"| {row['false_positive_rate']:.0%} "
            f"({row['false_positive_frames']}/{row['negatives']}) "
            f"| {row['fire_precision']:.0%} | {row['hit_precision']:.0%} |")
    return "\n".join(out)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--proto", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--models", default="", help="glob, to score a run's checkpoints")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON: {'frames': [{'clip', 'index', 'present', 'box'}]}")
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="directory of <clip>_<index>.jpg; see the manifest's own "
                             "'extract' field for the command that fills it")
    parser.add_argument("--split", default="",
                        help="score only frames whose 'split' field matches")
    parser.add_argument("--class-index", type=int, default=21)
    parser.add_argument("--mask-overlay", action="store_true")
    parser.add_argument("--select-threshold", type=float, default=0.50,
                        help="threshold whose (hit recall - FP rate) ranks the checkpoints")
    parser.add_argument("--reference", type=Path, default=None,
                        help="model to compare person scores against; see person_shift")
    parser.add_argument("--out", type=Path, default=None, help="write raw detections here")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    frames = [f for f in manifest["frames"]
              if not args.split or f.get("split") == args.split]
    for frame in frames:
        frame["image"] = str(args.frames_dir / f"{frame['clip']}_{frame['index']:03d}.jpg")
    if not frames:
        parser.error(f"no frames in {args.manifest} with split {args.split!r}")

    models = [Path(p) for p in sorted(glob(args.models))] if args.models else [args.model]
    if not models or models == [None]:
        parser.error("give --model or --models")

    ranking, dump = [], []
    for model in models:
        scored = score_model(args.proto, model, frames, args.class_index, args.mask_overlay)
        rows = table(scored)
        dump.append({"model": str(model), "table": rows, "frames": scored["frames"]})
        pick = min(rows, key=lambda r: abs(r["threshold"] - args.select_threshold))
        ranking.append((pick["hit_recall"] - pick["false_positive_rate"], str(model), rows))
        if len(models) > 1:
            print(f"{model.name:<28} @{pick['threshold']:.2f} "
                  f"hit-recall {pick['hit_recall']:.0%} fires {pick['fire_recall']:.0%} "
                  f"FP {pick['false_positive_rate']:.0%}", flush=True)

    ranking.sort(reverse=True)
    best_score, best_model, best_rows = ranking[0]
    print()
    print(render(best_rows, f"{Path(best_model).name}"
                            f"{' (overlay masked)' if args.mask_overlay else ''}"))
    persons = [f["person"] for entry in dump if entry["model"] == best_model
               for f in entry["frames"]]
    print(f"\nbest by (IoU recall - FP) at {args.select_threshold:.2f}: {best_model} "
          f"({best_score:+.3f});  person best-score mean over the same frames "
          f"{np.mean(persons):.3f}")
    if args.reference:
        reference = score_model(args.proto, args.reference, frames, args.class_index,
                                args.mask_overlay)
        best = next(e for e in dump if e["model"] == best_model)
        shift = person_shift(reference, {"frames": best["frames"]})
        print(f"person, on the {shift['frames']} frames the reference still detects one: "
              f"{shift.get('reference_mean', 0):.3f} -> {shift.get('candidate_mean', 0):.3f}, "
              f"worst drop {shift.get('worst_drop', 0):.3f}, "
              f"lost below 0.5 on {shift.get('lost', 0)}")
    if args.out:
        args.out.write_text(json.dumps(dump, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
