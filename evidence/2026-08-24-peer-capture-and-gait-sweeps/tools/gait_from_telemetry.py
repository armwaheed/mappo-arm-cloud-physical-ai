#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recover the gait floor from telemetry already recorded, instead of from new sweeps.

Every tick this stack writes pairs a COMMANDED velocity with the estimator's MEASURED
velocity. Five sessions of live runs are therefore thousands of naturalistic trials of the
question "does this command move the robot", spanning days, surfaces, positions and both
postures — conditions no staged sweep can re-create, because they are gone.

This matters because the staged sweeps that prompted it were confounded. Trials ran as a
walk down a corridor, so the forward ones carried the robot 0.5-0.75 m between samples and
every trial landed somewhere new; position and trial-order could not be separated. The
recorded runs have the opposite property: the confounds are uncorrelated with the command,
because nobody was choosing commands to test a floor.

## ⚠️ WHAT THIS CORPUS CAN AND CANNOT ANSWER

**It answers the SUSTAINING floor well.** 514 samples above 0.25 m/s with the robot already
moving, 88-95% of which produced motion. That is a lot of independent evidence that 0.35 is
not the floor for a robot already under way.

**It cannot answer the STARTING floor**, and the reason is a selection effect worth naming:
this directory holds the runs that were worth SAVING, which means the ones that went wrong.
After excluding dry runs, the remaining from-rest non-movers at 0.35 m/s come from
`run1-mappo-corridor-wall-contact` and `run4-room-cabinet-contact` — a robot pressed against
a wall does not move at any command, and that is an obstruction, not a floor — plus
`sample_telemetry.jsonl` and `live_run_telemetry.jsonl`, which are the same committed
example run counted twice.

Strip those and a handful of real from-rest samples remain. The number this script prints
for that column should be read as "not established", and the designed experiment — position
fixed, motion state varied, null control between every trial — is still what settles it.

## Two things the analysis has to get right

**LATENCY.** A command issued at tick i does not move the robot at tick i. The loop runs at
~10 Hz and the vendor controller has its own lag, so motion is read :data:`LOOK_AHEAD`
ticks later. Reading the same tick measures the PREVIOUS command and would smear every
band.

**STATE.** The whole hypothesis under test is that a command's effect depends on whether
the robot was already moving. So each sample is labelled by the measured speed BEFORE the
command, and the bands are reported separately. Pooling them is what makes a single scalar
floor look real.

Samples are only taken where the command held roughly steady across the look-ahead window
(:data:`STEADY_TOL`); a command that changed mid-flight tells you nothing about either
value.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict

#: Ticks to wait before reading the motion a command produced. ~200 ms at 10 Hz.
LOOK_AHEAD = 2

#: A command must not change by more than this across the window for the sample to count.
STEADY_TOL = 0.02

#: Measured speed above which the robot is "moving". The zero-command null control on
#: 2026-08-24 recorded 0.001 m of travel and ~0.010 m/s of estimator noise while standing,
#: so this sits well clear of the noise floor without being so high it only counts brisk
#: walking.
MOVING_MPS = 0.05

#: Commanded-speed bins, m/s.
BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50]


def ticks(path):
    """Ticks from one telemetry file, skipping headers and half-written lines."""
    out = []
    with open(path) as handle:
        lines = handle.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue                     # a run killed mid-write leaves a partial line
        if record.get("type") == "tick":
            out.append(record)
    return out


def speed(block):
    if not block:
        return None
    vx, vy = block.get("vx"), block.get("vy")
    if vx is None or vy is None:
        return None
    return math.hypot(vx, vy)


def samples(run):
    """``(commanded, measured_after, was_moving_before)`` for each usable tick.

    ⚠️ DRY RUNS ARE EXCLUDED, and skipping this filter is not a small error. Without
    ``--live`` the navigator perceives, plans and writes telemetry, but never enables the
    legs — so every tick reads as "commanded and did not move" and the answer is
    predetermined. Measured on this corpus: two dry runs supplied 198 of 301 from-rest
    samples above 0.30 m/s, 66% of them, all of them non-moving. Pooled in, they made a
    robot that starts reliably look like one that starts 9% of the time.
    """
    for i in range(1, len(run) - LOOK_AHEAD):
        if not run[i].get("live"):
            continue                     # legs were never enabled; the tick proves nothing
        command = run[i].get("command")
        if not command:
            continue                     # searching tick: no command was issued
        commanded = speed(command)
        if commanded is None:
            continue
        # The command must have held across the window, or the motion cannot be attributed.
        window = [run[j].get("command") for j in range(i, i + LOOK_AHEAD + 1)]
        if any(not c or speed(c) is None
               or abs(speed(c) - commanded) > STEADY_TOL for c in window):
            continue
        after = speed(run[i + LOOK_AHEAD].get("measured"))
        before = speed(run[i - 1].get("measured"))
        if after is None or before is None:
            continue
        yield commanded, after, before > MOVING_MPS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--glob", default="evidence/**/*.jsonl")
    args = parser.parse_args(argv)

    tally = defaultdict(lambda: {"moving": [0, 0], "rest": [0, 0]})
    files = runs = 0
    for path in sorted(glob.glob(args.glob, recursive=True)):
        run = ticks(path)
        if not run:
            continue
        files += 1
        got = 0
        for commanded, after, was_moving in samples(run):
            band = max((b for b in BINS if b <= commanded), default=BINS[0])
            slot = tally[band]["moving" if was_moving else "rest"]
            slot[1] += 1
            slot[0] += after > MOVING_MPS
            got += 1
        runs += got > 0

    print(f"{files} telemetry files, {runs} with usable samples, "
          f"look-ahead {LOOK_AHEAD} ticks, steady tolerance {STEADY_TOL}\n")
    print(f"{'commanded':>12} | {'ALREADY MOVING':>22} | {'FROM REST':>22}")
    print(f"{'band m/s':>12} | {'moved':>8} {'n':>6} {'rate':>6} "
          f"| {'moved':>8} {'n':>6} {'rate':>6}")
    print("-" * 64)
    for band in BINS:
        if band not in tally:
            continue
        m, r = tally[band]["moving"], tally[band]["rest"]
        top = f"{band:>10.2f}+ |"
        for hit, total in (m, r):
            rate = f"{hit / total:>6.0%}" if total else f"{'-':>6}"
            top += f" {hit:>8} {total:>6} {rate} |"
        print(top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
