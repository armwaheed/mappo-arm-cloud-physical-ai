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

## What that costs, measured

Detection range, simulated against the measured focal length of this unit (1290.2 px) by
rendering the marker at the pixel size the sensor would actually see and running it back
through `goal.aruco_detector`:

| black square | `detect_scale` | sharp | blurred |
| --- | --- | --- | --- |
| 140 mm | 0.5 (**the shipped default**) | 9.8 m | 5.2 m |
| 140 mm | 1.0 | 10.7 m | 6.4 m |
| 200 mm | 0.5 | 9.4 m | 7.5 m |
| 200 mm | 1.0 | 11.9 m | 9.1 m |

Sharp, 140 mm costs nothing — it is inside the noise of the 200 mm figure, and both are
far beyond the ~3 m demo area. **Under blur the larger marker is genuinely better**
(7.5 m against 5.2 m), so if a run needs to acquire the goal at long range while the
robot is already trotting, print 200 mm on A3 rather than 140 mm on A4.

The blurred column is a Gaussian stand-in for motion blur, not a calibrated match to what
a trotting Go2 produces at 0.35 m/s — treat the ordering as sound and the absolute
numbers as indicative.

## The peer robot is the more interesting marker

Nothing in this stack can see another quadruped: measured over 38 frames of two
peer-crossing runs, MobileNet-SSD returns **zero** response to a Go2 Walk at any
confidence down to 0.02, and YOLO11n on COCO does no better (`motorcycle` 0.14 at its
best, on a robot filling half the frame). A marker on the peer reuses everything here and
needs no detector work at all — but it needs an ArUco *obstacle* source, which does not
exist yet; `ArucoGoal` only latches a goal. See the repository README's peer-avoidance
notes.
