#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Score Lite3 fine-tune checkpoints at BOTH input sizes, and check what they cost `person`.

THE CLAIM UNDER TEST. Two things, and the second is the one that decides shipping.
(1) Does a checkpoint fire on the Lite3 at ``--input-size 224``, which is what
``deploy/run-peer-supervised.sh`` launches? The incumbent does not: 0 of 168 frames, best
IoU 0.041. (2) Does it still see people? The Go2 gate was "lose zero of the people the
shipped network sees at 0.45", and that is measured here on the Aug-20 CROSS-DAY frames --
a different day and a different building from the Lite3 clip, so unlike the Lite3 column
it is not same-session.

⚠️ THE LITE3 COLUMN IS AN UPPER BOUND AND NOT A GENERALISATION. Its 168 training frames
and its 168 evaluation frames are the same 31-second block of one camera pose. A number
near 100% there means the network memorised one image, which is exactly what it should be
able to do and says nothing about a second room, a second day or a second pose.

READING THE OUTPUT. ``lite3@224`` is frames where class 21 lands on the labelled box at
IoU >= 0.5. ``person kept`` is how many of the people the BASE network scores >= 0.45 the
candidate still scores >= 0.45, on the cross-day frames.

WHAT IT NEEDS. OpenCV, the checkpoints, the Lite3 manifest and frames, and the Go2
cross-day frames for the person column.
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import cv2
import numpy as np

SSD_SCALE, SSD_MEAN = 1.0 / 127.5, 127.5
PERSON, LITE3 = 15, 21
PERSON_FLOOR = 0.45
HIT_IOU = 0.5


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0) * max(y1 - y0, 0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def rows_for(net, image, size):
    h, w = image.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(image, SSD_SCALE, (size, size), SSD_MEAN))
    r = net.forward()[0, 0]
    out = np.zeros((len(r), 6), np.float32)
    out[:, 0], out[:, 1] = r[:, 1], r[:, 2]
    out[:, 2:] = r[:, 3:7] * np.array([w, h, w, h])
    return out


def lite3_recall(net, frames, size, thr):
    hit = 0
    for image, box in frames:
        r = rows_for(net, image, size)
        mine = r[(r[:, 0] == LITE3) & (r[:, 1] >= thr)]
        if any(iou(m[2:], box) >= HIT_IOU for m in mine):
            hit += 1
    return hit


def person_scores(net, images, size):
    out = []
    for image in images:
        r = rows_for(net, image, size)
        p = r[r[:, 0] == PERSON]
        out.append(float(p[:, 1].max()) if len(p) else 0.0)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--proto", type=Path, required=True)
    ap.add_argument("--models", required=True, help="glob over checkpoints")
    ap.add_argument("--base", type=Path, required=True, help="the model before fine-tuning")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--person-frames", type=Path, required=True,
                    help="cross-day frames used only for the person column")
    ap.add_argument("--person-limit", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    frames = [(cv2.imread(str(args.frames_dir / r["image"])), r["box"])
              for r in man["records"]]
    person_paths = sorted(glob(str(args.person_frames / "*.jpg")))[:args.person_limit]
    people = [cv2.imread(p) for p in person_paths]
    print(f"{len(frames)} Lite3 frames (SAME SESSION), {len(people)} cross-day frames for person")

    base = cv2.dnn.readNetFromCaffe(str(args.proto), str(args.base))
    base_person = {s: person_scores(base, people, s) for s in (224, 300)}
    for s in (224, 300):
        n = int((base_person[s] >= PERSON_FLOOR).sum())
        print(f"  base: people >= {PERSON_FLOOR} at {s}px: {n}/{len(people)}")
        hits = lite3_recall(base, frames, s, args.threshold)
        print(f"  base: lite3 hits at {s}px: {hits}/{len(frames)}")

    results = []
    for m in sorted(glob(args.models)):
        net = cv2.dnn.readNetFromCaffe(str(args.proto), m)
        row = {"model": Path(m).name}
        for s in (224, 300):
            row[f"lite3@{s}"] = lite3_recall(net, frames, s, args.threshold)
            cand = person_scores(net, people, s)
            keep = int(((base_person[s] >= PERSON_FLOOR) & (cand >= PERSON_FLOOR)).sum())
            row[f"person_kept@{s}"] = keep
            row[f"person_base@{s}"] = int((base_person[s] >= PERSON_FLOOR).sum())
        results.append(row)
        print(f"  {row['model']:<22} "
              f"lite3@224 {row['lite3@224']:>3}/{len(frames)}  "
              f"lite3@300 {row['lite3@300']:>3}/{len(frames)}  "
              f"person kept @224 {row['person_kept@224']}/{row['person_base@224']}  "
              f"@300 {row['person_kept@300']}/{row['person_base@300']}", flush=True)

    if args.out:
        args.out.write_text(json.dumps({"threshold": args.threshold,
                                        "lite3_frames": len(frames),
                                        "person_frames": len(people),
                                        "rows": results}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
