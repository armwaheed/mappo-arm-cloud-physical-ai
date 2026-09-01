#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render a print-ready A4 ArUco GRID BOARD, for intrinsic camera calibration.

WHY A BOARD AND NOT THE GOAL SHEET. ``make_marker_sheet.py`` prints ONE marker, which is
the right target for a goal beacon and the wrong one for a calibration. A calibration has
to solve a focal length, a principal point and four radial distortion coefficients, and it
can only do that from MANY correspondences WITHIN each frame. One marker gives four
corners; this board gives eighty. That is the difference between a fit that converges in
twenty views and one that does not converge at all.

WHY THIS EXISTS AT ALL, measured on 2026-08-31. This Lite3's front camera was being
described by a single ``focal_px`` in an equidistant model. Three values were fitted from
three different anchors -- 469.63 shipped, 660 from a panel width at a taped range, 392.4
from a taped range to the goal marker -- and every one of them is right at the bearing it
was fitted at and wrong elsewhere, because the lens does not obey the model. In the live
run ``live-goal-avoid-20260831T110648Z`` the robot swept 63 degrees of yaw while
translating 0.08 m, and its range estimate to a stationary box collapsed from 1.27 m to
0.33 m -- a 281% swing on an obstacle that had not moved. The planner then veto-held,
correctly, on a number that was wrong. ``calibrate_camera.py --spin`` had already warned
this would happen: it reported a 4.37 deg SYSTEMATIC residual and said in as many words
that no single focal length would fix it.

WHY NOT A CHESSBOARD. It needs no explanation as an instrument, but it needs a flat rigid
board and someone on the floor holding it still, and this project has a printed ArUco
workflow already. A planar ArUco board is an equally valid calibration target: OpenCV
solves each view's pose rather than being told it, so nothing here is tape-measured and
nothing depends on the robot's odometry -- which matters, because odometry is what made
the ``--spin`` fit disagree with a tape by 47%.

The pattern comes from the robot stack's own dictionary via ``goal.DEFAULT_DICTIONARY``,
so this sheet cannot disagree with the detector about what it is printing.

IDS START AT 10, not 0. The goal beacon is id 0, and a calibration board carrying it would
be seen by a running navigator as a goal lying on the floor.

    python3 make_board_sheet.py
    # print A4 portrait at 100%, then check the 100 mm bar with a ruler
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_VISUAL_NAV = (Path(__file__).resolve().parents[2]
               / "robot-stack" / "unitree" / "go2" / "visual_nav")
sys.path.insert(0, str(_VISUAL_NAV))

from goal import DEFAULT_DICTIONARY  # noqa: E402
from make_marker_sheet import A4_MM, RULER_MM, _draw_caption, _px  # noqa: E402

#: Markers across and down. 4x5 fills A4 portrait while leaving room for the caption and
#: the verification bar, and 20 markers is 80 corners per view.
DEFAULT_COLS, DEFAULT_ROWS = 4, 5

#: Side of one printed marker's black square, millimetres, and the white gap between
#: neighbours. ``4*36 + 3*11 = 177 mm`` across and ``5*36 + 4*11 = 224 mm`` down, inside
#: A4's 210 x 297 with printer margins and the caption block.
DEFAULT_MARKER_MM, DEFAULT_SEPARATION_MM = 36.0, 11.0

#: First id. The goal beacon is id 0; see the module docstring.
DEFAULT_FIRST_ID = 10


def build_board_sheet(cols: int, rows: int, marker_mm: float, separation_mm: float,
                      first_id: int, dictionary_name: str) -> np.ndarray:
    """Compose the printable A4 sheet around a ``cv2.aruco.GridBoard``."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    ids = np.arange(first_id, first_id + cols * rows, dtype=np.int32)
    board = cv2.aruco.GridBoard((cols, rows), marker_mm, separation_mm, dictionary, ids)

    board_w_mm = cols * marker_mm + (cols - 1) * separation_mm
    board_h_mm = rows * marker_mm + (rows - 1) * separation_mm
    # generateImage renders at whatever pixel size it is given; asking for exactly the
    # physical size in pixels is what makes "print at 100%" mean the stated millimetres.
    image = board.generateImage((_px(board_w_mm), _px(board_h_mm)), marginSize=0)

    sheet = np.full((_px(A4_MM[1]), _px(A4_MM[0])), 255, dtype=np.uint8)
    x0 = (sheet.shape[1] - image.shape[1]) // 2
    y0 = _px(12.0)
    if x0 < _px(5.0) or y0 + image.shape[0] > sheet.shape[0]:
        raise ValueError(f"{cols}x{rows} at {marker_mm} mm does not fit on A4")
    sheet[y0:y0 + image.shape[0], x0:x0 + image.shape[1]] = image

    y = y0 + image.shape[0] + _px(10.0)
    y = _draw_caption(sheet, f"ArUco {dictionary_name}  {cols}x{rows} GRID BOARD  "
                             f"ids {first_id}-{first_id + cols * rows - 1}", x0, y,
                      scale=1.4)
    y = _draw_caption(sheet, f"marker {marker_mm:.0f} mm, gap {separation_mm:.0f} mm", x0,
                      y, scale=1.4)
    y = _draw_caption(sheet, "Print A4 portrait at 100% / actual size. NOT 'fit to page'.",
                      x0, y, scale=1.1)
    y = _draw_caption(sheet, "Calibration target -- NOT the goal beacon (that is id 0).",
                      x0, y, scale=1.1)

    bar_y, bar_px = y + _px(4.0), _px(RULER_MM)
    cv2.line(sheet, (x0, bar_y), (x0 + bar_px, bar_y), 0, 6)
    for end in (x0, x0 + bar_px):
        cv2.line(sheet, (end, bar_y - _px(3.0)), (end, bar_y + _px(3.0)), 0, 6)
    _draw_caption(sheet, f"check: this bar is {RULER_MM:.0f} mm", x0, bar_y + _px(11.0),
                  scale=1.1)
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_MM)
    parser.add_argument("--separation-mm", type=float, default=DEFAULT_SEPARATION_MM)
    parser.add_argument("--first-id", type=int, default=DEFAULT_FIRST_ID)
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY)
    args = parser.parse_args(argv)

    sheet = build_board_sheet(args.cols, args.rows, args.marker_mm, args.separation_mm,
                              args.first_id, args.dictionary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = (f"calib-board-{args.dictionary.lower().replace('dict_', '')}-"
            f"{args.cols}x{args.rows}-{args.marker_mm:.0f}mm-a4.png")
    out = args.out_dir / name
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}")
    print(f"  {args.cols}x{args.rows} = {args.cols * args.rows} markers, "
          f"{args.cols * args.rows * 4} corners per view")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
