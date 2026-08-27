#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Build the contact sheet embedded in this directory's README. Needs the clip."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

RADAR, PLATE = (964, 16, 300, 300), (0, 630, 470, 90)
BOX = [543, 477, 809, 717]
RADAR_REGION = (0.834, 0.0, 1.0, 0.287)

def label(im, text, colour=(0, 0, 255)):
    cv2.rectangle(im, (0, 0), (im.shape[1], 40), (255, 255, 255), -1)
    cv2.putText(im, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour, 2)
    return im

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("lite3-clip-contact-sheet.jpg"))
    a = ap.parse_args()
    cap = cv2.VideoCapture(str(a.video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    tiles = []
    raw = frames[239].copy()
    tiles.append(label(raw, "1. as recorded: --record burns the HUD in"))

    ann = frames[239].copy()
    for (x, y, w, h) in (RADAR, PLATE):
        cv2.rectangle(ann, (x, y), (x + w, y + h), (255, 0, 255), 4)
    rx0, ry0 = int(RADAR_REGION[0]*1280), int(RADAR_REGION[1]*720)
    rx1, ry1 = int(RADAR_REGION[2]*1280), int(RADAR_REGION[3]*720)
    cv2.rectangle(ann, (rx0, ry0), (rx1-1, ry1-1), (0, 255, 255), 4)
    tiles.append(label(ann, "2. magenta = the real HUD; yellow = what --mask-overlay masks"))

    filled = frames[119].copy()
    fill = np.median(filled.reshape(-1, 3), axis=0)
    for (x, y, w, h) in (RADAR, PLATE):
        filled[y:y+h, x:x+w] = fill
    cv2.rectangle(filled, BOX[:2], BOX[2:], (255, 0, 255), 3)
    tiles.append(label(filled, "3. a training frame: HUD filled, peer labelled"))

    a2, b2 = frames[36].copy(), frames[203].copy()
    diff = cv2.absdiff(a2, b2)
    tiles.append(label(cv2.convertScaleAbs(diff, alpha=6.0),
                       "4. f0037 vs f0204, 31 s apart, x6 gain: nothing moved"))

    grid = np.vstack([np.hstack([cv2.resize(t, (640, 360)) for t in tiles[:2]]),
                      np.hstack([cv2.resize(t, (640, 360)) for t in tiles[2:]])])
    cv2.imwrite(str(a.out), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"wrote {a.out} {grid.shape}")

if __name__ == "__main__":
    main()
