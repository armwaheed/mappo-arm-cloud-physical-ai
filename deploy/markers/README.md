<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Printable markers

`goal-aruco-4x4_50-id0-a4-140mm.png` — **print this one.** A4 portrait, 100% / actual
size. Tape it to the goal chair (or whatever object is standing in for the goal) with
the black square flat and facing down the approach lane.

Then run the navigator with the size that is printed on the sheet:

```sh
python3 visual_nav.py --calibration go2_front_camera.json --marker-size 0.14 ...
```

`goal-aruco-4x4_50-id0.png` is the bare pattern, for anyone who wants to lay it out at a
different size. `make_marker_sheet.py` regenerates both — it takes the pattern from the
robot stack's own `goal.write_marker`, so this directory cannot drift from the
dictionary and id the detector is actually looking for.

## Check the print before you trust a range

Every distance this beacon produces is proportional to the physical side of its black
square. A print dialogue left on "fit to page" rescales the sheet, and the navigator then
reports a confident, precise, **wrong** distance with nothing in the logs to say so.

The sheet carries a 100 mm bar for exactly this. Put a ruler on it. If it is not 100 mm,
the marker is not 140 mm either, and `--marker-size` is now a lie — either reprint, or
measure the black square and pass what you measured.

## Why 140 mm and not the 0.20 m default

`goal.DEFAULT_MARKER_SIZE_M` is 0.20, and its docstring claims that "fits on one sheet of
A4". **It does not.** A `DICT_4X4_50` marker is 6 modules across — 4 data modules plus a
one-module black border on each side — and detection needs a quiet zone of at least one
module of white outside that border. The sheet therefore has to carry

    s + 2 * (s / 6) = 4s / 3

which at s = 200 mm is **267 mm**, against A4's 210 mm width. Solving against a 190 mm
printable width (A4 less 10 mm of printer margin) gives **s ≤ 142.5 mm**.

Two refinements worth knowing. The quiet zone can be *unprinted paper* rather than printed
white, so bare A4 actually allows **157.5 mm**; 142.5 mm is the strict answer and the one
worth designing to, because it is the width a "fit to page" print scales into. And the
one-module quiet zone is a **convention, not a cliff** — measured, OpenCV needs none at all
against a mid-grey background and about 0.1 of a module against an ink-dark one. One module
is kept here because the background is not knowable in advance.

## What that costs, measured — a predictable 30%, not nothing

⚠️ **CORRECTED.** An earlier version of this file said 140 mm cost nothing when sharp. That
was a measurement error, and the error was visible in its own table: it reported 9.8 m for
140 mm against 9.4 m for 200 mm at the same `detect_scale`, which is **non-monotonic in
marker size** and therefore impossible. The sweep took the *maximum* range at which
detection ever succeeded, so it picked up isolated hits — up to 2 m — past the range where
detection is continuous.

Measured properly, detection range is **exactly linear in the printed side**. The ceiling
is a constant pixel floor: **23.7 full-frame px** at the shipped `detect_scale` of 0.5, and
19.2 px at 1.0, steady to a tenth of a pixel across sizes from 100 mm to 300 mm. So

    range(140 mm) = 0.70 x range(200 mm)

at every blur level tested (Gaussian sigma 0, 1, 1.5 and 2 px). **A4 costs a predictable
30% of range.** That is the honest trade, and it is still the right one for this demo: the
arena is ~3 m and a 140 mm sheet was detected on the robot's own camera at **6.43 m** in a
live frame today.

Print 200 mm on A3 if a run has to acquire the goal from across a large room.

## The peer robot is the more interesting marker

Nothing in this stack can see another quadruped: measured over 38 frames of two
peer-crossing runs, MobileNet-SSD returns **zero** response to a Go2 Walk at any
confidence down to 0.02, and YOLO11n on COCO does no better (`motorcycle` 0.14 at its
best, on a robot filling half the frame). A marker on the peer reuses everything here and
needs no detector work at all — but it needs an ArUco *obstacle* source, which does not
exist yet; `ArucoGoal` only latches a goal. See the repository README's peer-avoidance
notes.
