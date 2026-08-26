#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a registered background plate for the 'corridor standing' camera-pose group.

Each corridor segment parks the peer somewhere different, so a per-pixel median over one
frame from each -- all warped to a common reference -- removes the peer and leaves the empty
corridor, including the marker chair, which never moves and is a useful hard negative.

Frames come from ``PEERCAP_FRAMES``; ``peercap.py`` records where the corpus lives and
refuses with that list when it is not set.
"""
import cv2
import numpy as np
import peercap

SRC = peercap.frames_dir()
WORK = peercap.work_dir()

REFNAME = "p6_1_trunc_left_stand_0070.jpg"
MEMBERS = [
    "p4_mid_sweep_stand_0000.jpg", "p4_mid_sweep_stand_0030.jpg", "p4_mid_sweep_stand_0200.jpg",
    "p5_1_far_left_stand_0050.jpg", "p5_23_far_centre_then_right_stand_0050.jpg",
    "p5_23_far_centre_then_right_stand_0190.jpg",
    "p6_1_trunc_left_stand_0070.jpg", "p6_2_trunc_right_stand_0070.jpg",
    "p6_3_trunc_half_left_stand_0070.jpg",
]


def ecc_warp(src_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    han = cv2.createHanningWindow((ref_gray.shape[1], ref_gray.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ref_gray.astype(np.float32) * han,
                                     src_gray.astype(np.float32) * han)
    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2], warp[1, 2] = -dx, -dy
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp = cv2.findTransformECC(ref_gray, src_gray, warp, cv2.MOTION_EUCLIDEAN,
                                       crit, None, 5)
    except cv2.error as exc:
        print("  ECC failed, phase correlation only:", exc.err)
    return warp


def main() -> None:
    ref = cv2.imread(SRC + REFNAME)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    layers = []
    for name in MEMBERS:
        img = cv2.imread(SRC + name)
        warp = ecc_warp(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), ref_gray)
        layers.append(cv2.warpAffine(img, warp, (ref.shape[1], ref.shape[0]),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE))
        print(f"{name:46s} t=({warp[0, 2]:+7.2f},{warp[1, 2]:+6.2f}) "
              f"rot={np.degrees(np.arctan2(warp[1, 0], warp[0, 0])):+.2f} deg")
    plate = np.median(np.stack(layers), axis=0).astype(np.uint8)
    cv2.imwrite(WORK + "plate_corridor.jpg", plate, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(WORK + "plate_corridor.npy", plate)


if __name__ == "__main__":
    main()
