#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Was the robot actually parked during the negative clips?

Phase-correlate consecutive frames (downscaled) and report the pixel shift.
A parked camera on a quadruped still breathes; a walking one translates.
"""
import collections
import glob
import os
import re

import cv2
import numpy as np

D = os.environ.get("PEER_DATASET",
                   os.path.expanduser("~/go2-peer-dataset-20260824"))

def pref(n): return re.sub(r"_?\d{4}\.jpg$", "", os.path.basename(n))

frames = collections.defaultdict(list)
for p in sorted(glob.glob(D + "/*.jpg")):
    frames[pref(p)].append(p)

for tag in ("neg_prone", "neg_standing", "p1b_close_broadside_STANDING",
            "p4_mid_sweep_stand", "p1_close_broadside", "p5_23_far_centre_then_right_stand"):
    paths = frames[tag]
    shifts = []
    prev = None
    for p in paths[:200]:
        g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (480, 270)).astype(np.float32)
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            shifts.append((dx, dy))
        prev = g
    a = np.array(shifts)
    mag = np.hypot(a[:, 0], a[:, 1])
    # cumulative drift = net camera motion over the clip, in 480-wide px
    net = np.abs(a.sum(axis=0))
    print(f"{tag:42s} n={len(a):4d} per-frame |shift| med={np.median(mag):6.3f} "
          f"p95={np.percentile(mag, 95):6.3f} max={mag.max():6.2f}  "
          f"net=({net[0]:7.1f},{net[1]:6.1f}) px@480w")
