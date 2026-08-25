#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Will the blue on the peer actually be seen as a `bin`? Point this at a photo.

Runs the SHIPPED `ColourBlobDetector` with the SHIPPED `BLUE_BIN` profile, then — for
every blue contour it found, including the ones the profile threw away — says which gate
rejected it. A phone photo is good enough: the gates are on shape and saturation, not on
the camera.

    python3 check_blue_patch.py peer.jpg

Why the gates reject tape. `BLUE_BIN` wants a roughly SQUARE solid: `aspect` (w/h) must
be within 0.35..2.60, so a strip of tape is out by construction — a horizontal strip runs
3-8, a vertical one 0.1-0.3. The gate is not arbitrary: it is what separates a bin from a
strip of blue glazing on the corridor wall, which is the false positive that started the
whole colour path. Loosening it to admit tape re-admits the wall.

What to put on the peer instead: ONE roughly square blue patch, about 0.27 m wide by
0.30 m tall (the profile's `width_m`/`height_m`), on the flank that faces the robot's
approach. Both numbers are load-bearing — range is `size / subtended angle`, so a patch
half that size reads as twice the distance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_VISUAL_NAV = (Path(__file__).resolve().parents[2]
               / "robot-stack" / "unitree" / "go2" / "visual_nav")
sys.path.insert(0, str(_VISUAL_NAV))

from colour_detector import BLUE_BIN, ColourBlobDetector  # noqa: E402

#: Contours smaller than this are not reported at all. Well below the profile's own
#: `min_area_px` so that a patch failing ONLY on size still shows up as a near miss.
REPORT_AREA_PX = 80


def _verdict(area: float, aspect: float, fill: float) -> str:
    """Which gate this contour fails, or ``PASSES``. Order matches the detector's."""
    if area < BLUE_BIN.min_area_px:
        return f"too small (area {area:.0f} < {BLUE_BIN.min_area_px})"
    if not BLUE_BIN.min_aspect <= aspect <= BLUE_BIN.max_aspect:
        shape = "a wide strip" if aspect > BLUE_BIN.max_aspect else "a tall strip"
        return (f"ASPECT {aspect:.2f} outside {BLUE_BIN.min_aspect}-"
                f"{BLUE_BIN.max_aspect} — {shape}, not a square patch")
    if fill < BLUE_BIN.min_fill:
        return f"too ragged (fill {fill:.2f} < {BLUE_BIN.min_fill})"
    return "PASSES"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    image = cv2.imread(argv[1])
    if image is None:
        print(f"cannot read {argv[1]}", file=sys.stderr)
        return 2

    # Rebuild the detector's own mask so the near misses can be reported too. Kept in
    # step with ColourBlobDetector.detect by construction: same profile, same scale,
    # same morphology. If that method changes, this reads as a near miss and not as a
    # silent disagreement, because the PASSES line below is cross-checked against the
    # real detector's output.
    scale = 0.5
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.medianBlur(BLUE_BIN.mask(hsv), 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    blue_px = int(np.count_nonzero(BLUE_BIN.mask(hsv)))
    print(f"{argv[1]}: {small.shape[1]}x{small.shape[0]} at detect scale, "
          f"{blue_px} px inside the blue window "
          f"(hue {BLUE_BIN.hue_lo}-{BLUE_BIN.hue_hi}, sat>={BLUE_BIN.sat_min}, "
          f"val>={BLUE_BIN.val_min})")
    if blue_px == 0:
        print("\nNo pixel is inside the colour window at all — the blue is too pale or "
              "too dark.\nSaturation is the usual culprit: masking tape is much less "
              "saturated than a bin.")
        return 1

    rows = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < REPORT_AREA_PX:
            continue
        _, _, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        rows.append((area, w, h, area / float(w * h)))
    rows.sort(reverse=True)

    print(f"\n{len(rows)} blue region(s) worth reporting:\n")
    print(f"{'area':>8} {'box':>12} {'aspect':>7} {'fill':>6}   verdict")
    for area, w, h, fill in rows[:10]:
        print(f"{area:>8.0f} {f'{w}x{h}':>12} {w / h:>7.2f} {fill:>6.2f}   "
              f"{_verdict(area, w / h, fill)}")

    passing = ColourBlobDetector(BLUE_BIN).detect(image)
    print(f"\nthe shipped detector returns {len(passing)} detection(s)"
          + (":" if passing else " — the peer would be INVISIBLE to the policy."))
    for d in passing:
        print(f"    {d.label} score={d.score:.2f} "
              f"box=({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f})")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
