#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Anchor-template NCC tracker: propagate one hand-drawn box across a segment.

The camera never moves within a segment, so the peer's box can only translate.  Matching the
anchor patch -- not the previous frame's patch -- keeps the tracker from accumulating drift
over a long static run, and the reported peak correlation flags frames where the peer no
longer looks like its anchor and the box should be distrusted.

Frames come from ``PEERCAP_FRAMES``; ``peercap.py`` records where the corpus lives and
refuses with that list when it is not set.
"""
import glob
import json
import sys

import cv2
import peercap

SRC = peercap.frames_dir()
WORK = peercap.work_dir()


def track(tag: str, anchor_idx: int, box: tuple[int, ...], pad: int = 70) -> dict:
    files = sorted(glob.glob(SRC + tag + "_[0-9][0-9][0-9][0-9].jpg"))
    x0, y0, x1, y1 = box
    tpl = cv2.imread(files[anchor_idx], cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1]
    h, w = tpl.shape
    px, py = x0, y0
    out = {}
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        sx0, sy0 = max(0, px - pad), max(0, py - pad)
        sx1, sy1 = min(img.shape[1], px + w + pad), min(img.shape[0], py + h + pad)
        res = cv2.matchTemplate(img[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(res)
        px, py = sx0 + loc[0], sy0 + loc[1]
        out[f.rsplit("/", 1)[1]] = [px, py, px + w, py + h, round(float(peak), 4)]
    return out


def main() -> None:
    tag, anchor = sys.argv[1], int(sys.argv[2])
    res = track(tag, anchor, tuple(int(v) for v in sys.argv[3:7]))
    with open(WORK + f"track_{tag}.json", "w") as fh:
        json.dump(res, fh)
    peaks = [v[4] for v in res.values()]
    first = res[min(res)]
    shifts = [max(abs(v[0] - first[0]), abs(v[1] - first[1])) for v in res.values()]
    print(f"{tag:38s} n={len(res):4d} min peak={min(peaks):.3f} max shift={max(shifts)} px")


main()
