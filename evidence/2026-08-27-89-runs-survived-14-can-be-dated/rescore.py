# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-score captured frames with the shipped weights, at both input sizes.

Production passes ``--input-size 224`` (``deploy/run-peer-supervised.sh``); every offline
scorer in ``detector/`` hardcodes 300 (issue #129). A recall number that does not say
which of those it used is not a number, so this reports both from one decode per frame.

    python3 rescore.py --frames /path/to/walk3-frames --model-dir /path/to/models

``--model-dir`` holds the stock ``MobileNetSSD_deploy.prototxt`` and ``.caffemodel``.

⚠️ **The score floor is in the prototxt, not in this script.** The ``detection_out``
layer ships ``confidence_threshold: 0.25``, so nothing below 0.25 leaves the network
whatever this script asks for, and a "sub-threshold" sweep that does not patch the
prototxt is measuring nothing. ``--subfloor`` patches it to 0.01 in a temporary copy.

Read the sub-floor output as noise rather than as near-misses: on walk 3 at 224 px it
reports a ``person`` in 535 of 535 frames, including roughly 300 in which the camera is
pressed against cardboard and no person is in shot. That is why the shipped floor is
where it is.
"""
import argparse
import glob
import json
import os
import re
import shutil
import tempfile

#: Verbatim from person_detector.py. Index is the SSD class id.
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
SSD_SCALE = 1.0 / 127.5
SSD_MEAN = 127.5

#: The two sizes in play: what the robot runs, and what every offline scorer assumes.
SIZES = (224, 300)

#: visual_nav's own default --confidence, i.e. the gate a detection must clear to reach
#: the planner at all. Distinct from the prototxt floor above.
OPERATING_THRESHOLD = 0.45


def _model(model_dir, subfloor, stack):
    """Load the net, optionally through a prototxt whose score floor is lowered."""
    import cv2
    proto = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
    weights = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")
    if subfloor:
        tmp = stack.enter_context(tempfile.TemporaryDirectory())
        with open(proto) as fh:
            text = fh.read()
        patched, n = re.subn(r"confidence_threshold:\s*[0-9.]+",
                             "confidence_threshold: 0.01", text)
        if n != 1:
            raise SystemExit(f"expected exactly one confidence_threshold, found {n}")
        proto = os.path.join(tmp, "MobileNetSSD_deploy.prototxt")
        with open(proto, "w") as fh:
            fh.write(patched)
        shutil.copy(weights, os.path.join(tmp, "MobileNetSSD_deploy.caffemodel"))
        weights = os.path.join(tmp, "MobileNetSSD_deploy.caffemodel")
    return cv2.dnn.readNetFromCaffe(proto, weights)


def main():
    import contextlib

    import cv2
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", required=True, help="directory of frame_*.jpg")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--label", default="person")
    ap.add_argument("--subfloor", action="store_true",
                    help="patch the prototxt floor from 0.25 to 0.01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames, "frame_*.jpg")))
    if not frames:
        raise SystemExit(f"no frame_*.jpg under {args.frames}")

    with contextlib.ExitStack() as stack:
        net = _model(args.model_dir, args.subfloor, stack)
        best = {size: [] for size in SIZES}
        for path in frames:
            image = cv2.imread(path)
            if image is None:
                continue
            for size in SIZES:
                net.setInput(cv2.dnn.blobFromImage(
                    image, SSD_SCALE, (size, size), SSD_MEAN))
                raw = net.forward()
                top = 0.0
                for _, class_id, score, _x1, _y1, _x2, _y2 in raw[0, 0]:
                    index = int(class_id)
                    name = VOC_CLASSES[index] if index < len(VOC_CLASSES) else "?"
                    if name == args.label:
                        top = max(top, float(score))
                best[size].append(top)

    report = {"frames": len(frames), "label": args.label,
              "prototxt_floor": 0.01 if args.subfloor else 0.25, "by_size": {}}
    for size in SIZES:
        scores = best[size]
        fired = sum(1 for s in scores if s >= OPERATING_THRESHOLD)
        report["by_size"][str(size)] = {
            "fired_at_operating_threshold": fired,
            "operating_threshold": OPERATING_THRESHOLD,
            "best_score": round(max(scores), 4) if scores else None,
            "frames_with_any_score": sum(1 for s in scores if s > 0),
        }
        print(f"{size} px: {fired}/{len(scores)} frames fired "
              f"{args.label} >= {OPERATING_THRESHOLD}, "
              f"best score {max(scores):.4f}, "
              f"{sum(1 for s in scores if s > 0)} frames scored anything")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
