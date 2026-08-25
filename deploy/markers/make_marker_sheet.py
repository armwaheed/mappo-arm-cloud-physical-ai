#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Render a print-ready A4 sheet for the ArUco goal beacon.

The pattern itself comes from ``goal.write_marker`` in the robot stack, so this script
cannot disagree with the detector about which dictionary or id it is producing — the one
failure that would be invisible until the robot is standing in front of the sheet not
seeing it.

WHY THIS EXISTS RATHER THAN "just print the PNG". Every range this beacon produces is
proportional to the physical side of its black square: ``range = size / (2 tan(theta/2))``
in :meth:`ArucoGoal._fix_from_corners`. A print dialogue left on "fit to page" or "scale
to fit" silently rescales the sheet, and the navigator then reports a confident, precise,
wrong distance to the goal with nothing anywhere in the logs to say so. The sheet
therefore carries its own ground truth: the size is printed on it, and a 100 mm scale bar
lets a ruler falsify a bad print in five seconds.

WHY 140 mm AND NOT THE 200 mm DEFAULT. ``goal.DEFAULT_MARKER_SIZE_M`` is 0.20 and its
docstring says that "fits on one sheet of A4". It does not. A ``DICT_4X4_50`` marker is
6 modules across (4 data + a one-module black border each side), and detection needs a
quiet zone of at least one module of white outside that border, so the sheet must carry
``s + 2*(s/6) = 4s/3``. At s = 200 mm that is 267 mm against A4's 210 mm width. Solving
the other way against a 190 mm printable width (A4 less 10 mm printer margins) gives
s <= 142.5 mm, hence 140. The cost is only range: detection distance is linear in size,
so the 6 m quoted for 0.20 m becomes ~4.2 m — still well beyond the 3 m demo area.

Print A4 portrait at 100% / "actual size", then pass the size to the navigator:

    python3 visual_nav.py --calibration go2_front_camera.json --marker-size 0.14 ...
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

from goal import DEFAULT_DICTIONARY, DEFAULT_MARKER_ID, write_marker  # noqa: E402

#: Print resolution of the emitted sheet. 300 dpi is what a desk printer resolves; the
#: marker's modules are ~28 mm across, so this is far above what detection needs and is
#: chosen for the legibility of the printed caption instead.
DPI = 300

#: A4 portrait, millimetres.
A4_MM = (210.0, 297.0)

#: Side of the printed BLACK SQUARE, millimetres. See the module docstring for why this
#: is not ``goal.DEFAULT_MARKER_SIZE_M``.
DEFAULT_SIZE_MM = 140.0

#: Length of the printed verification bar. 100 mm because that is one unambiguous ruler
#: reading — a bar the same length as the marker would invite measuring the wrong edge.
RULER_MM = 100.0


def _px(mm: float) -> int:
    """Millimetres to pixels at :data:`DPI`, rounded to the nearest whole pixel."""
    return round(mm * DPI / 25.4)


def _draw_caption(sheet: np.ndarray, text: str, x: int, y: int,
                  scale: float = 2.0) -> int:
    """Draw one left-aligned caption line; returns the baseline y for the next one.

    Stroke weight tracks ``scale`` rather than being a second knob: they were only ever
    set together, and letting them drift apart is how a caption ends up bold and tiny.
    """
    thickness = max(2, round(scale * 2))
    (_, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(sheet, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thickness,
                cv2.LINE_AA)
    return y + int(height * 2.0)


def build_sheet(marker: np.ndarray, size_mm: float, marker_id: int,
                dictionary_name: str) -> np.ndarray:
    """Compose the printable A4 sheet around ``marker``.

    ``marker`` arrives from :func:`goal.write_marker`, which has already added its own
    white border. That border is re-derived rather than assumed: the sheet needs to place
    the BLACK SQUARE at an exact physical size, and the black square is the marker minus
    whatever white surrounds it.
    """
    ink = np.argwhere(marker == 0)
    if ink.size == 0:
        raise ValueError("marker image has no black pixels")
    (top, left), (bottom, right) = ink.min(axis=0), ink.max(axis=0)
    black = marker[top:bottom + 1, left:right + 1]

    side_px = _px(size_mm)
    # INTER_NEAREST keeps the modules hard-edged. Any interpolation here greys the
    # module boundaries, and a grey boundary is exactly what the detector's adaptive
    # threshold has to guess at.
    black = cv2.resize(black, (side_px, side_px), interpolation=cv2.INTER_NEAREST)

    quiet_px = _px(size_mm / 6.0)          # one module, the minimum quiet zone
    sheet = np.full((_px(A4_MM[1]), _px(A4_MM[0])), 255, dtype=np.uint8)
    x0 = (sheet.shape[1] - side_px) // 2
    y0 = _px(25.0) + quiet_px
    if x0 < quiet_px or y0 + side_px + quiet_px > sheet.shape[0]:
        raise ValueError(f"{size_mm} mm marker plus its quiet zone does not fit on A4")
    sheet[y0:y0 + side_px, x0:x0 + side_px] = black

    y = y0 + side_px + quiet_px + _px(12.0)
    y = _draw_caption(sheet, f"ArUco {dictionary_name}  id={marker_id}", x0, y)
    y = _draw_caption(sheet, f"BLACK SQUARE = {size_mm:.0f} mm  ->  --marker-size "
                             f"{size_mm / 1000.0:.2f}", x0, y)
    y = _draw_caption(sheet, "Print A4 portrait at 100% / actual size. NOT 'fit to page'.",
                      x0, y, scale=1.2)

    # The verification bar. Ticked at both ends so there is no doubt which span to
    # measure, and captioned with its own length.
    bar_y, bar_px = y + _px(6.0), _px(RULER_MM)
    cv2.line(sheet, (x0, bar_y), (x0 + bar_px, bar_y), 0, 6)
    for end in (x0, x0 + bar_px):
        cv2.line(sheet, (end, bar_y - _px(3.0)), (end, bar_y + _px(3.0)), 0, 6)
    _draw_caption(sheet, f"check: this bar is {RULER_MM:.0f} mm", x0, bar_y + _px(12.0),
                  scale=1.2)
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY)
    parser.add_argument("--size-mm", type=float, default=DEFAULT_SIZE_MM,
                        help="side of the printed black square")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"goal-aruco-{args.dictionary.lower().replace('dict_', '')}-id{args.marker_id}"
    pattern_path = args.out_dir / f"{stem}.png"
    write_marker(str(pattern_path), marker_id=args.marker_id,
                 dictionary_name=args.dictionary, pixels=1200)

    marker = cv2.imread(str(pattern_path), cv2.IMREAD_GRAYSCALE)
    sheet = build_sheet(marker, args.size_mm, args.marker_id, args.dictionary)
    sheet_path = args.out_dir / f"{stem}-a4-{args.size_mm:.0f}mm.png"
    cv2.imwrite(str(sheet_path), sheet)

    print(f"pattern: {pattern_path}")
    print(f"sheet:   {sheet_path}")
    print(f"         print A4 portrait at 100%, then --marker-size "
          f"{args.size_mm / 1000.0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
