#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smallest true/claimed range ratio the gate drops, by claimed range and ego speed.

Noiseless, full 30-sample window at the perception thread's ~7 Hz, so this is the
gate's reach given the measured sigma and nothing else moving.
"""
import os
import sys

#: Where the modules under test live. Overridable because these scripts run on the DGX
#: Spark, where the repo is not necessarily checked out next to the data.
VISUAL_NAV = os.environ.get(
    "VISUAL_NAV",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "robot-stack", "unitree", "go2", "visual_nav"))
sys.path.insert(0, VISUAL_NAV)
from expansion import ExpansionConsistency

DT = 0.143
STEPS = 30


def dropped(claimed, ratio, speed):
    g = ExpansionConsistency()
    k = 1.0 / ratio
    true = claimed * ratio
    v = None
    for i in range(STEPS):
        rx = speed * i * DT
        rep = k * (true - rx)
        if rep <= 0:
            break
        v = g.observe(1, time_s=100 + i * DT, range_m=rep, source="height",
                      odom_xy=(rx + rep, 0.0), robot_xy=(rx, 0.0))
    return v is not None and v.rejected


ratios = [1.0 + 0.05 * i for i in range(140)]
speeds = (0.35, 0.5, 0.8)
print("smallest true/claimed ratio dropped  (30 samples @ 7 Hz, no noise)")
print(f"{'claimed m':>10} " + " ".join(f"{'v=' + str(v):>12}" for v in speeds))
for claimed in (0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
    cells = []
    for v in speeds:
        hit = next((r for r in ratios if dropped(claimed, r, v)), None)
        cells.append(f"{hit:>12.2f}" if hit else f"{'never':>12}")
    print(f"{claimed:>10.1f} " + " ".join(cells))

print()
print("and the same as 'how far away must the thing REALLY be', in metres")
print(f"{'claimed m':>10} " + " ".join(f"{'v=' + str(v):>12}" for v in speeds))
for claimed in (0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
    cells = []
    for v in speeds:
        hit = next((r for r in ratios if dropped(claimed, r, v)), None)
        cells.append(f"{claimed * hit:>12.2f}" if hit else f"{'never':>12}")
    print(f"{claimed:>10.1f} " + " ".join(cells))
