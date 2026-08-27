#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""What the SHIPPED detector does with a Lite3, at production preprocessing.

THE CLAIM UNDER TEST. The demo needs a Lite3 to see another Lite3. Before any retraining
is worth doing, the incumbent MobileNet-SSD has to be measured on this robot, in this
room, at the input size production actually launches -- ``--input-size 224``
(``deploy/run-peer-supervised.sh``), not the ``INPUT_SIZE = 300`` every scorer in
``detector/`` hardcodes. A whole checkpoint sweep was invalidated by that mismatch
(issue #129), so both are reported here and the 224 column is the one that counts.

WHAT THIS MEASURES. Class-agnostic recall: whether ANY of the 21 VOC classes puts a box
on the peer at IoU >= 0.30, which is the same "lands on" convention
``detector/eval_class_agnostic.py`` uses, and which label it uses when it does. A peer the
network already localises under the wrong name is a relabelling problem; a peer it does
not localise at all is a detection problem, and they need different work.

READING THE OUTPUT. ``lands on`` is the fraction of frames with a box on the peer.
``fires`` is the fraction with that class anywhere in the frame -- inflated by exactly
the false-positive rate, which is why both columns are printed.

WHAT IT NEEDS. The stock ``MobileNetSSD_deploy.prototxt``/``.caffemodel``, the frames
named by the manifest, and OpenCV. No GPU, no torch.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

#: Preprocessing baked into the weights. Not a knob.
SSD_SCALE, SSD_MEAN = 1.0 / 127.5, 127.5

#: Production launches 224 (deploy/run-peer-supervised.sh); every detector/ scorer uses 300.
SIZES = (224, 300)

#: detector/eval_class_agnostic.py's convention, deliberately looser than the mAP 0.5 one:
#: the question is whether the network put a box ON the robot, not whether it fits well.
LANDS_ON_IOU = 0.30

#: Nothing below this is reachable -- the prototxt's DetectionOutput has already
#: discarded weaker boxes before forward() returns.
THRESHOLDS = (0.25, 0.40, 0.50, 0.70)

VOC = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
       "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
       "pottedplant", "sheep", "sofa", "train", "tvmonitor")


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0) * max(y1 - y0, 0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def detect(net, image: np.ndarray, size: int) -> np.ndarray:
    """(class, score, x0, y0, x1, y1) rows in PIXELS, from the real DetectionOutput."""
    h, w = image.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(image, SSD_SCALE, (size, size), SSD_MEAN))
    rows = net.forward()[0, 0]
    out = np.zeros((len(rows), 6), np.float32)
    out[:, 0], out[:, 1] = rows[:, 1], rows[:, 2]
    out[:, 2:] = rows[:, 3:7] * np.array([w, h, w, h])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    net = cv2.dnn.readNetFromCaffe(
        str(args.model_dir / "MobileNetSSD_deploy.prototxt"),
        str(args.model_dir / "MobileNetSSD_deploy.caffemodel"))
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    print(f"{len(records)} frames, label {manifest['label']!r}, "
          f"from {args.frames_dir}")

    dump = {}
    for size in SIZES:
        per_frame = []
        for rec in records:
            image = cv2.imread(str(args.frames_dir / rec["image"]))
            rows = detect(net, image, size)
            gt = rec["box"]
            per_frame.append([
                {"cls": int(r[0]), "score": float(r[1]),
                 "iou": iou(r[2:], gt)} for r in rows])
        dump[str(size)] = per_frame

        tag = "  <-- PRODUCTION" if size == 224 else ""
        print(f"\n=== input size {size}{tag} ===")
        print(f"{'conf':>5} | {'lands on peer':>14} | {'anything fires':>14} | best label on peer")
        for t in THRESHOLDS:
            lands = [f for f in per_frame
                     if any(d["score"] >= t and d["iou"] >= LANDS_ON_IOU for d in f)]
            fires = [f for f in per_frame if any(d["score"] >= t for d in f)]
            names = Counter()
            for f in per_frame:
                on = [d for d in f if d["score"] >= t and d["iou"] >= LANDS_ON_IOU]
                if on:
                    names[VOC[max(on, key=lambda d: d["score"])["cls"]]] += 1
            top = ", ".join(f"{k} {v}" for k, v in names.most_common(4)) or "-"
            print(f"{t:>5.2f} | {len(lands):>5}/{len(records)} = {len(lands)/len(records):>4.0%} "
                  f"| {len(fires):>5}/{len(records)} = {len(fires)/len(records):>4.0%} | {top}")

        best = [max((d["iou"] for d in f), default=0.0) for f in per_frame]
        print(f"best IoU with the peer box, any class, any score: "
              f"median {np.median(best):.3f}  max {max(best):.3f}")

    if args.out:
        args.out.write_text(json.dumps(dump))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
