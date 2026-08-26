#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-frame peer blobs for one corridor-pose segment, by registered plate difference.

The plate is registered to the segment once (within-segment camera drift is under 2 px) and
every frame is then differenced against it.  Output is a raw blob list per frame; deciding
which blobs are the peer is left to the caller, because the polished floor also returns the
robot's reflection and a couple of fixed glare streaks.

Frames come from ``PEERCAP_FRAMES``; ``peercap.py`` records where the corpus lives and
refuses with that list when it is not set.
"""
from __future__ import annotations

import glob
import json
import sys

import cv2
import numpy as np
import peercap

SRC = peercap.frames_dir()
WORK = peercap.work_dir()


def register(plate: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Warp `plate` into `frame`'s coordinates; returns the warped plate and inlier ratio."""
    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=10)
    kp1, d1 = orb.detectAndCompute(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), None)
    kp2, d2 = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0, maxIters=20000, confidence=0.999)
    warped = cv2.warpPerspective(plate, h, (frame.shape[1], frame.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, float(inl.sum()) / max(len(good), 1)


def gain_match(img: np.ndarray, plate: np.ndarray) -> np.ndarray:
    """Scale the plate so its per-channel median matches the frame's."""
    med_p = np.median(plate.reshape(-1, 3).astype(np.float32), axis=0)
    med_i = np.median(img.reshape(-1, 3).astype(np.float32), axis=0)
    g = med_i / np.maximum(med_p, 1.0)
    return (plate.astype(np.float32) * g).clip(0, 255).astype(np.uint8)


def tolerant_diff(img: np.ndarray, plate: np.ndarray, rad: int = 7) -> np.ndarray:
    """Difference each pixel against the min/max of its neighbourhood in the plate.

    A couple of pixels of residual misregistration would otherwise light up every edge in the
    corridor; comparing against a local range absorbs that.
    """
    k = np.ones((rad, rad), np.uint8)
    lo = cv2.erode(plate, k).astype(np.int16)
    hi = cv2.dilate(plate, k).astype(np.int16)
    f = img.astype(np.int16)
    return np.maximum(f - hi, lo - f).clip(0).max(axis=2).astype(np.uint8)


def frame_blobs(img: np.ndarray, plate: np.ndarray, thr: int, min_area: int) -> list:
    d = cv2.GaussianBlur(tolerant_diff(img, plate), (7, 7), 0)
    mask = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = [(int(s[0]), int(s[1]), int(s[0] + s[2]), int(s[1] + s[3]), int(s[4]))
           for s in stats[1:] if s[4] >= min_area]
    return sorted(out, key=lambda b: -b[4])


def main() -> None:
    tag, thr, min_area = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    plate = np.load(WORK + "plate_corridor.npy")
    files = sorted(glob.glob(SRC + tag + "_[0-9][0-9][0-9][0-9].jpg"))
    mid = cv2.imread(files[len(files) // 2])
    reg, ratio = register(plate, mid)
    reg = gain_match(mid, reg)
    print(f"{tag}: registration inlier ratio {ratio:.2f}")
    recs = {f.rsplit("/", 1)[1]: frame_blobs(cv2.imread(f), reg, thr, min_area) for f in files}
    with open(WORK + f"blobs_{tag}.json", "w") as fh:
        json.dump(recs, fh)
    keys = sorted(recs)
    for k in keys[:: max(1, len(keys) // 8)]:
        print("  ", k.rsplit("_", 1)[1], recs[k][:3])


if __name__ == "__main__":
    main()
