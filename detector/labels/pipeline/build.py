#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble annotations.json from the per-segment propagation results.

Frames come from ``PEERCAP_FRAMES``; ``peercap.py`` records where the corpus lives and
refuses with that list when it is not set.
"""
from __future__ import annotations

import json
import os

import peercap

SRC = peercap.frames_dir()
WORK = peercap.work_dir()
OUT = peercap.labelled_dir()

LABEL = "go2wheel"
SNAP = 3
TRACKED = [
    "p1_close_broadside", "p2_close_headon_stand", "p3_close_rearon_stand",
    "p5_1_far_left_stand", "p5_23_far_centre_then_right_stand", "p6_1_trunc_left_stand",
    "p6_2_trunc_right_stand", "p6_3_trunc_half_left_stand", "peer01",
]
P523 = "p5_23_far_centre_then_right_stand"
P523_MOVE_FROM = 183


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def clip(b: list[int]) -> list[int]:
    """Clamp to the frame, and snap an edge that lands within SNAP px of a border.

    On the truncated segments the object really is cut by the frame edge, and a 1-2 px
    tracker shift would otherwise leave a sliver of background between box and border.
    """
    out = [max(0, min(1920, b[0])), max(0, min(1080, b[1])),
           max(0, min(1920, b[2])), max(0, min(1080, b[3]))]
    if out[0] <= SNAP:
        out[0] = 0
    if out[1] <= SNAP:
        out[1] = 0
    if out[2] >= 1920 - SNAP:
        out[2] = 1920
    if out[3] >= 1080 - SNAP:
        out[3] = 1080
    return out


def p523_tail() -> dict:
    """The tail of p5_23, where the peer walks off to the right.

    NCC cannot follow a peer that small -- the background inside the box outvotes it -- so the
    box comes from the plate-difference blobs, gated above the floor glare streak, and is then
    re-sized to the anchor box: the difference only catches the peer's high-contrast parts.
    """
    blobs = load(WORK + f"blobs_{P523}.json")
    anchor = load(WORK + f"track_{P523}.json")[f"{P523}_0000.jpg"]
    w, h = anchor[2] - anchor[0], anchor[3] - anchor[1]
    out = {}
    for name, bl in blobs.items():
        if int(name.rsplit("_", 1)[1].split(".")[0]) < P523_MOVE_FROM:
            continue
        keep = [b for b in bl if b[3] < 745 and b[1] > 500 and b[4] > 200]
        if not keep:
            continue
        cx = (min(b[0] for b in keep) + max(b[2] for b in keep)) / 2
        cy = (min(b[1] for b in keep) + max(b[3] for b in keep)) / 2
        out[name] = [round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)]
    return out


def main() -> None:
    recs, counts = [], {}
    for tag in TRACKED:
        boxes = {k: v[:4] for k, v in load(WORK + f"track_{tag}.json").items()}
        if tag == P523:
            boxes.update(p523_tail())
        recs += [{"image": n, "label": LABEL, "box": clip(boxes[n])} for n in sorted(boxes)]
        counts[tag] = len(boxes)
    p4 = load(WORK + "p4_boxes.json")
    p4kept = [n for n in sorted(p4) if p4[n] is not None]
    recs += [{"image": n, "label": LABEL, "box": clip(p4[n])} for n in p4kept]
    counts["p4_mid_sweep_stand"] = len(p4kept)
    os.makedirs(OUT, exist_ok=True)
    doc = {"label": LABEL, "source_dir": SRC, "count": len(recs), "records": recs}
    with open(OUT + "annotations.json", "w") as fh:
        json.dump(doc, fh, indent=1)
    for k, v in sorted(counts.items()):
        print(f"{k:38s} {v}")
    print("TOTAL", len(recs))


if __name__ == "__main__":
    main()
