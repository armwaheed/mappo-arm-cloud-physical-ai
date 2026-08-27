#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Score checkpoints at a STATED preprocessing, and say which one every number is at.

    python3 score_checkpoints.py --proto P --models 'runs/x/epoch*.caffemodel' \
        --lite3-manifest lite3_eval_20260827.json --lite3-frames DIR \
        --person-manifest peer_crossday_20260820.json --person-frames DIR \
        --preprocessing go2-peer-supervised --out scored.json

WHY THIS EXISTS RATHER THAN ``detector/eval_detector.py``. That scorer has no
``--input-size``: it is hardcoded at 300, which is issue #129, and the launcher this robot
actually runs (``deploy/run-peer-supervised.sh``) passes **224**. The previous Lite3 wave
found every checkpoint scoring 168/168 at 300 and **0/168 at 224**, so a number quoted
without its square is not a number. This takes the four configurations from
``robot-stack/unitree/go2/visual_nav/inference_profile.py`` — the one place the
preprocessing is written down — and refuses to run without being told which.

⚠️ **A FRESH ``cv2.dnn.Net`` PER CONFIGURATION, ALWAYS.** ``cv2.dnn.Net`` retains state
across an input-size change: measured on this project, 300-after-224 scored 42/60 where a
fresh net on the same weights scored 41/60. Reusing a net makes the SECOND configuration
scored a function of the first, which is how a ranking becomes an artefact of loop order.

TWO METRICS, AND THEY ARE NOT THE SAME KIND OF NUMBER.

``lite3``   is SAME-SESSION. The eval frames are a held-out *time block* of the same
            60-second tripod shot the training frames come from — same camera pose, same
            room, same morning. A temporal block rather than a random split, because at
            15 fps a random split puts frame n in train and n+1 in eval, which is the same
            photograph twice. It still does not generalise, and the README says so.
``person``  is CROSS-DAY: the 2026-08-20 Go2 manifest, a different day and a different
            building. It is the only number here that generalises, and it is the gate
            every candidate so far has failed.

Both are reported two ways, because the existing corpus uses the second and a class the
training produced is the first:

``class_hit``       the NEW class fires, at IoU >= 0.5 against the label.
``agnostic_hit``    ANY class fires with a non-person aspect, which is the convention
                    ``sweep_all.py`` used for every number in the wave-5/6 headers. It is
                    the comparable one; it is not a ``lite3`` detection rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

#: The four configurations, copied from inference_profile.py by NAME so a reader can
#: check them against it. Only input_size and confidence differ between them.
PROFILES = {
    "go2-peer-supervised":   {"input_size": 224, "confidence": 0.25},
    "go2-run-smoke":         {"input_size": 300, "confidence": 0.45},
    "go2-navigator-default": {"input_size": 300, "confidence": 0.40},
    "mobilenet-ssd-trained": {"input_size": 300, "confidence": 0.25},
}
SCALE, MEAN, SWAP_RB = 1.0 / 127.5, 127.5, False
#: person_detector.PERSON_ASPECT_MIN.
PERSON_ASPECT_MIN = 2.0
#: VOC index of `person` in the 21-class base net, which the fine-tune does not move.
PERSON_INDEX = 15


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def forward(net, image, size):
    height, width = image.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(image, SCALE, (size, size), MEAN, swapRB=SWAP_RB))
    raw = net.forward()
    out = []
    for i in range(raw.shape[2]):
        x1, y1, x2, y2 = (float(v) for v in raw[0, 0, i, 3:7])
        out.append((int(raw[0, 0, i, 1]), float(raw[0, 0, i, 2]),
                    [x1 * width, y1 * height, x2 * width, y2 * height]))
    return out


def score_lite3(net, records, frames_dir, profile, new_index):
    """Same-session recall on the held-out time block."""
    class_hit = agnostic_hit = 0
    for record in records:
        image = cv2.imread(str(frames_dir / record["image"]))
        if image is None:
            continue
        dets = forward(net, image, profile["input_size"])
        truth = record["box"]
        for index, score, box in dets:
            if score < profile["confidence"]:
                continue
            width, height = box[2] - box[0], box[3] - box[1]
            if width <= 0 or height <= 0:
                continue
            if index == new_index and iou(box, truth) >= 0.5:
                class_hit += 1
                break
        for _index, score, box in dets:
            if score < profile["confidence"]:
                continue
            width, height = box[2] - box[0], box[3] - box[1]
            if width <= 0 or height <= 0 or height / width >= PERSON_ASPECT_MIN:
                continue
            if iou(box, truth) >= 0.5:
                agnostic_hit += 1
                break
    return {"frames": len(records), "class_hit": class_hit,
            "agnostic_hit": agnostic_hit}


def score_person(net, manifest, frames_dir, profile):
    """Cross-day person retention: frames carrying a person-shaped `person` box."""
    kept = 0
    considered = 0
    for frame in manifest["frames"]:
        name = f"{frame['clip']}_{frame['index']:03d}.jpg"
        image = cv2.imread(str(frames_dir / name))
        if image is None:
            continue
        considered += 1
        for index, score, box in forward(net, image, profile["input_size"]):
            if index != PERSON_INDEX or score < profile["confidence"]:
                continue
            width, height = box[2] - box[0], box[3] - box[1]
            if width > 0 and height / width >= PERSON_ASPECT_MIN:
                kept += 1
                break
    return {"frames": considered, "people_kept": kept}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proto", type=Path, required=True)
    parser.add_argument("--models", required=True, help="glob over checkpoints")
    parser.add_argument("--lite3-manifest", type=Path)
    parser.add_argument("--lite3-frames", type=Path)
    parser.add_argument("--person-manifest", type=Path)
    parser.add_argument("--person-frames", type=Path)
    parser.add_argument("--preprocessing", required=True, choices=sorted(PROFILES),
                        help="which configuration every number below is at")
    parser.add_argument("--new-class-index", type=int, default=21)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    profile = PROFILES[args.preprocessing]
    lite3 = (json.loads(args.lite3_manifest.read_text())["records"]
             if args.lite3_manifest else [])
    person = (json.loads(args.person_manifest.read_text())
              if args.person_manifest else None)

    results = []
    root = Path(args.models).parent
    for model in sorted(str(p) for p in root.glob(Path(args.models).name)):
        # A FRESH net for every checkpoint AND every configuration. See the docstring.
        net = cv2.dnn.readNetFromCaffe(str(args.proto), model)
        row = {"model": Path(model).name, "preprocessing": args.preprocessing,
               "input_size": profile["input_size"], "confidence": profile["confidence"]}
        if lite3:
            row["lite3"] = score_lite3(net, lite3, args.lite3_frames, profile,
                                       args.new_class_index)
        if person:
            row["person"] = score_person(net, person, args.person_frames, profile)
        results.append(row)
        summary = ""
        if "lite3" in row:
            summary += (f"lite3 {row['lite3']['class_hit']}/{row['lite3']['frames']} "
                        f"(agnostic {row['lite3']['agnostic_hit']}) ")
        if "person" in row:
            summary += f"people {row['person']['people_kept']}/{row['person']['frames']}"
        print(f"  {Path(model).name:28s} @{args.preprocessing:22s} {summary}", flush=True)

    args.out.write_text(json.dumps(
        {"preprocessing": args.preprocessing, "profile": profile,
         "note": "lite3 is SAME-SESSION; person is CROSS-DAY. See the README.",
         "results": results}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
