#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rescore the class-agnostic detector against the frames that are ACTUALLY peer-free.

The published 18% false-alarm figure split the corpus by "is this filename in the label
file". Four clips are not, and only two of them are negatives: `p1b_close_broadside_
STANDING` (134 frames) and `smoke` (58) both contain a peer robot the labeller never
covered. This rescores against the 705 frames that are genuinely empty and prints, for
each threshold, what the two splits give.

Also reports the highest-scoring detection on those 705, which is the number that says
how much headroom the deployed 0.40 threshold has — and it is only meaningful when the
dump was taken with the PROTOTXT threshold lowered. `MobileNetSSD_deploy.prototxt` sets
`confidence_threshold: 0.25` in its `detection_out` layer, so a sweep below 0.25 against
the stock file measures nothing at all. Pass a dump taken with it lowered to see the
real floor; see this directory's README.

    python3 rescore.py detections.json [labels.json]
"""
import collections
import json
import os
import re
import sys


def _load_json(path):
    """Read a JSON file. A helper so these one-shot scripts still close their files."""
    with open(path) as handle:
        return json.load(handle)

DETECTIONS = sys.argv[1] if len(sys.argv) > 1 else "detections.json"
LABELS = (sys.argv[2] if len(sys.argv) > 2 else
          os.path.expanduser("~/ssdft/base/peer_go2wheel_20260824.json"))

#: The only two clips in the corpus with no peer in them. Named rather than derived,
#: because "absent from the label file" is exactly the inference that was wrong.
PEER_FREE_CLIPS = ("neg_prone", "neg_standing")

DETS = _load_json(DETECTIONS)
GT = collections.defaultdict(list)
for record in _load_json(LABELS)["records"]:
    GT[record["image"]].append(record["box"])


def clip(name):
    return re.sub(r"_?\d{4}\.jpg$", "", name)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


frames = sorted(DETS)
labelled = [f for f in frames if f in GT]
unlabelled = [f for f in frames if f not in GT]
empty = [f for f in unlabelled if clip(f) in PEER_FREE_CLIPS]
contaminated = [f for f in unlabelled if clip(f) not in PEER_FREE_CLIPS]

print(f"{len(frames)} frames: {len(labelled)} labelled, {len(unlabelled)} unlabelled")
print(f"  of the unlabelled, {len(empty)} are genuinely peer-free and "
      f"{len(contaminated)} are NOT:")
for name in sorted({clip(f) for f in contaminated}):
    print(f"    {name}: {sum(1 for f in contaminated if clip(f) == name)} frames")

def pct(n, total):
    return f"{n}/{total} = {100 * n / total:.1f}%"


print()
print(f"{'conf':<6}{'recall (IoU>=.3)':<24}{'FA as scored':<24}"
      f"{'FA on truly empty':<24}{'detected on contaminated'}")
for threshold in (0.15, 0.25, 0.40, 0.50, 0.60):
    hits = sum(1 for f in labelled
               if any(d[1] >= threshold and max(iou(d[2:6], g) for g in GT[f]) >= 0.30
                      for d in DETS[f]))

    def fired(group, t=threshold):
        return sum(1 for f in group if any(d[1] >= t for d in DETS[f]))

    print(f"{threshold:<6.2f}{pct(hits, len(labelled)):<24}"
          f"{pct(fired(unlabelled), len(unlabelled)):<24}"
          f"{pct(fired(empty), len(empty)):<24}"
          f"{pct(fired(contaminated), len(contaminated))}")

print()
scored = sorted(((d[1], f, d[0], d[2:6]) for f in empty for d in DETS[f]), reverse=True)
print(f"detections on the {len(empty)} peer-free frames: {len(scored)}")
if scored:
    print(f"  highest: {scored[0][0]:.3f} {scored[0][2]} in {scored[0][1]} "
          f"box={[round(v) for v in scored[0][3]]}")
    print("  classes:",
          dict(collections.Counter(row[2] for row in scored).most_common(10)))
    for floor in (0.05, 0.10, 0.15, 0.20, 0.25, 0.40):
        n = sum(1 for f in empty if any(d[1] >= floor for d in DETS[f]))
        print(f"  frames with ANY detection >= {floor:.2f}: {n}/{len(empty)} = "
              f"{100 * n / len(empty):.2f}%")

print()
print("what the detector calls the peer on the contaminated frames (>=0.25):",
      dict(collections.Counter(d[0] for f in contaminated for d in DETS[f]
                               if d[1] >= 0.25).most_common()))
