#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Does the STOCK 21-class network already see the peer, if we stop asking WHICH class?

This is the parsimony test, and it is the script that ended the fine-tune. `train_new_class`
and `finetune_ssd` exist to add a `go2wheel` label. But nothing downstream consumes a label
as a label: a detection reaches the policy as geometry, and `person_detector.PersonDetector`
uses the name only to decide whether to forward the box at all — its default
`classes=DYNAMIC_CLASSES` is `("person",)`, so every other VOC label the network emits is
dropped before the tracker, the map or the policy can see it.

So: run the robot's own unmodified weights, accept EVERY label, and ask two questions.

1. **Recall.** On frames with a hand-labelled peer, does any detection land ON it?
2. **False alarm.** On peer-free frames, does anything fire at all?

A class-agnostic "any box is an obstacle" policy is viable exactly when (1) is high and (2)
is low. Measured 2026-08-25 on the Aug-24 capture: **64% and 18%**, against a fine-tune that
cost a day of GB10 time and does worse on both.

## Why same-session negatives are legitimate HERE and were not there

`FROZEN-FEATURE-CEILING.md` records a gate that scored 0 of 705 because its negatives came
from the same session as its training frames — it was measuring memorisation. That trap
cannot apply to a model that was not trained on the session. The stock weights have never
seen a frame of this corpus, so every frame in it is out-of-sample and the peer-free frames
are an honest false-alarm set.

## The threshold you ask for is not the threshold you get

The deployed prototxt carries ``confidence_threshold: 0.25`` in its ``detection_output_param``
and ``DetectionOutput`` applies it before ``forward()`` returns. Asking for 0.15 measures 0.25.
:data:`THRESHOLDS` starts below the floor deliberately, so that the first two rows printing
identical numbers is a visible check that the floor is where this says it is.

Usage:

    eval_class_agnostic.py --model-dir ~/go2_models \\
                           --labels labels/peer_go2wheel_20260824.json \\
                           --frames-dir PEERCAP
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from add_class import VOC_CLASSES

# The detector's preprocessing is NOT declared in this file. It comes from
# robot-stack/unitree/go2/visual_nav/inference_profile.py, which is the same object
# deploy/run-peer-supervised.sh takes the robot's own --input-size and --confidence from.
# See issue #129: this module used to hold `INPUT_SIZE = 300` while every peer run
# executed at 224, and a 94-checkpoint sweep was ranked against the difference.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot-stack" / "unitree"
                       / "go2" / "visual_nav"))
import inference_profile
from inference_profile import PreprocessingMismatch

#: 0.15 is below the prototxt's own floor and must print the same row as 0.25. If it ever
#: does not, the prototxt in ``--model-dir`` is not the one the robot loads.
THRESHOLDS = (0.15, 0.25, 0.40, 0.50)

#: A detection "lands on" the peer at this overlap. Deliberately looser than the 0.5 an mAP
#: convention would use: the question is whether the planner gets a box in the right place,
#: not whether the box is tight, and a `motorbike` fitted to two wheels is neither expected
#: nor required to cover the whole robot.
LANDS_ON_IOU = 0.30


def iou(box: np.ndarray, other: np.ndarray) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes in pixels."""
    x0, y0 = max(box[0], other[0]), max(box[1], other[1])
    x1, y1 = min(box[2], other[2]), min(box[3], other[3])
    overlap = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    union = ((box[2] - box[0]) * (box[3] - box[1])
             + (other[2] - other[0]) * (other[3] - other[1]) - overlap)
    return overlap / union if union > 0 else 0.0


def detections(net, image: np.ndarray, profile) -> list[tuple[str, float, np.ndarray]]:
    """``(label, score, box)`` for every class, in the image's own pixel coordinates."""
    height, width = image.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(image, profile.scale, profile.blob_size,
                                       profile.mean, swapRB=profile.swap_rb))
    out = []
    for row in net.forward()[0, 0]:
        class_id = int(row[1])
        label = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else f"?{class_id}"
        out.append((label, float(row[2]),
                    row[3:7] * np.array([width, height, width, height])))
    return out


def load_frames(labels: Path, frames_dir: Path) -> tuple[list, list]:
    """Split every JPEG in ``frames_dir`` into peer frames (with boxes) and peer-free ones."""
    boxes = defaultdict(list)
    for record in json.loads(labels.read_text())["records"]:
        boxes[record["image"]].append(np.asarray(record["box"], dtype=float))
    images = sorted(p.name for p in frames_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no .jpg under {frames_dir}")
    return ([(frames_dir / n, boxes[n]) for n in images if n in boxes],
            [(frames_dir / n, []) for n in images if n not in boxes])


def score(net, frames: list, profile) -> list:
    """Every detection on every frame, once, so the thresholds are a sweep and not N passes."""
    out = []
    for path, truth in frames:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  unreadable, skipped: {path.name}")
            continue
        out.append((detections(net, image, profile), truth))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="holds the robot's own MobileNetSSD_deploy.prototxt/.caffemodel")
    parser.add_argument("--labels", type=Path, required=True,
                        help="peer label manifest, e.g. labels/peer_go2wheel_20260824.json")
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="the JPEGs the manifest names; peer-free frames are the rest")
    parser.add_argument("--iou", type=float, default=LANDS_ON_IOU)
    inference_profile.add_arguments(parser)
    args = parser.parse_args(argv)

    try:
        profile, reason = inference_profile.resolve(args)
    except PreprocessingMismatch as refusal:
        parser.exit(2, f"\nREFUSED\n{refusal}\n\n")
    print(f"preprocessing: {profile.name} — {profile.input_size} px, confidence "
          f"{profile.confidence}, scale 1/{1.0 / profile.scale:.1f}, mean {profile.mean}"
          + ("" if profile.is_deployed else f"\n  RUN BY NO LAUNCHER: {reason}"))

    proto = args.model_dir / "MobileNetSSD_deploy.prototxt"
    weights = args.model_dir / "MobileNetSSD_deploy.caffemodel"
    for path in (proto, weights):
        if not path.is_file():
            raise SystemExit(f"{path} not found")
    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))

    positives, negatives = load_frames(args.labels, args.frames_dir)
    print(f"{len(positives) + len(negatives)} frames: {len(positives)} with a labelled "
          f"peer, {len(negatives)} peer-free")
    peer_frames = score(net, positives, profile)
    clean_frames = score(net, negatives, profile)

    print(f"\n{'conf':<6} {'box ON the peer':<22} {'anything fired':<22} "
          f"peer-free frames firing")
    for threshold in THRESHOLDS:
        landed, labels = 0, Counter()
        fired = 0
        for found, truth in peer_frames:
            kept = [d for d in found if d[1] >= threshold]
            fired += bool(kept)
            hit = False
            for label, _, box in kept:
                if max(iou(box, t) for t in truth) >= args.iou:
                    labels[label] += 1
                    hit = True
            landed += hit
        alarms = sum(1 for found, _ in clean_frames
                     if any(d[1] >= threshold for d in found))
        print(f"{threshold:<6.2f} "
              f"{_rate(landed, len(peer_frames)):<22} "
              f"{_rate(fired, len(peer_frames)):<22} "
              f"{_rate(alarms, len(clean_frames))}")
        print(f"       labels landing on the peer: {dict(labels.most_common(8))}")
    return 0


def _rate(count: int, total: int) -> str:
    return f"{count}/{total} = {100.0 * count / total:.0f}%" if total else "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
