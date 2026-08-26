#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crop / upscale / optionally CLAHE a frame, overlay candidate boxes and a labelled grid.

This is the seeding tool: box corners were read off these renders by eye.
usage: view.py NAME OUT x0 y0 x1 y1 [step] [scale] [enh] [bx0 by0 bx1 by1 ...]

Frames come from ``PEERCAP_FRAMES``; ``peercap.py`` records where the corpus lives and
refuses with that list when it is not set.
"""
import sys

import cv2
import peercap

SRC = peercap.frames_dir()


def main() -> None:
    a = sys.argv
    name, out = a[1], a[2]
    x0, y0, x1, y1 = (int(v) for v in a[3:7])
    step = int(a[7]) if len(a) > 7 else 100
    scale = float(a[8]) if len(a) > 8 else 1.0
    enh = a[9] if len(a) > 9 else "n"
    boxes = [int(v) for v in a[10:]]
    crop = cv2.imread(SRC + name)[y0:y1, x0:x1].copy()
    if enh == "y":
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[:, :, 0])
        crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for i in range(0, len(boxes), 4):
        bx0, by0, bx1, by1 = boxes[i:i + 4]
        cv2.rectangle(crop, (int((bx0 - x0) * scale), int((by0 - y0) * scale)),
                      (int((bx1 - x0) * scale), int((by1 - y0) * scale)), (0, 255, 0), 2)
    for x in range(x0 - x0 % step, x1 + 1, step):
        px = int((x - x0) * scale)
        if 0 <= px < crop.shape[1]:
            cv2.line(crop, (px, 0), (px, crop.shape[0]), (0, 0, 255), 1)
            cv2.putText(crop, str(x), (px + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for y in range(y0 - y0 % step, y1 + 1, step):
        py = int((y - y0) * scale)
        if 0 <= py < crop.shape[0]:
            cv2.line(crop, (0, py), (crop.shape[1], py), (0, 255, 255), 1)
            cv2.putText(crop, str(y), (2, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(out, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
