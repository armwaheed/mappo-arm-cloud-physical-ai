#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for colour-blob detection of static props.

Synthetic frames only — a solid rectangle of a known colour on a grey ground, which is
what the staged scene reduces to. The tests worth having are the ones a naive
hue-threshold would fail, because that is the whole risk of segmenting by colour: the
gates must reject a same-coloured WALL and keep a bin.

Run: ``python3 test_colour_detector.py``
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_model import FisheyeCamera
from colour_detector import BLUE_BIN, ColourBlobDetector, ColourProfile
from person_detector import estimate_range

#: The office the robot actually works in: grey carpet and grey partitions, i.e.
#: unsaturated. Saturation is what the profile leans on, so the ground matters.
GREY = (150, 150, 150)
#: BGR of the staged bin, sampled from the robot's own footage (hue ~108).
BIN_BLUE = (150, 70, 25)


def _frame(width=1920, height=1080, ground=GREY):
    return np.full((height, width, 3), ground, dtype=np.uint8)


def _rect(image, x, y, w, h, colour=BIN_BLUE):
    cv2.rectangle(image, (x, y), (x + w, y + h), colour, cv2.FILLED)
    return image


def test_a_solid_blue_rectangle_is_found():
    image = _rect(_frame(), 800, 500, 160, 180)
    found = ColourBlobDetector().detect(image)
    assert len(found) == 1, found
    box = found[0]
    assert box.label == "bin"
    # Half-scale segmentation, so a box is good to ~2 px of the original either way.
    assert abs(box.x1 - 800) <= 4 and abs(box.y1 - 500) <= 4, box
    assert abs(box.width_px - 160) <= 6 and abs(box.height_px - 180) <= 6, box


def test_grey_office_furniture_is_not_a_bin():
    """The ground the robot actually walks past must produce nothing."""
    image = _rect(_frame(), 700, 400, 300, 400, colour=(160, 160, 160))
    assert ColourBlobDetector().detect(image) == []


def test_a_tall_thin_strip_of_the_same_blue_is_rejected():
    """A glazed wall reflecting the same hue is the false positive that matters.

    Colour cannot separate it from the bin; SHAPE can. This is the aspect gate, and it
    is why the module does not simply take the largest blue blob — on the staged scene
    the wall strip is genuinely large.
    """
    image = _rect(_frame(), 150, 200, 80, 480)
    assert ColourBlobDetector().detect(image) == []


def test_a_ragged_blue_scatter_is_rejected():
    """Speckle across a big bounding box has a large area but a low FILL.

    Measured on the staged scene: the bin fills 0.85 of its box and the glazed wall
    0.35. This is the discriminating gate, so it gets a test that isolates it — the
    scatter below is square, so aspect cannot be what rejects it.
    """
    image = _frame()
    for row in range(12):
        for column in range(12):
            if (row + column) % 2 == 0:
                _rect(image, 700 + column * 24, 400 + row * 24, 12, 12)
    assert ColourBlobDetector().detect(image) == []


def test_two_bins_are_both_returned_largest_first():
    """A second bin down the corridor is an obstacle, not a false positive."""
    image = _rect(_frame(), 800, 500, 160, 180)
    _rect(image, 1300, 560, 70, 80)
    found = ColourBlobDetector().detect(image)
    assert len(found) == 2, found
    assert found[0].width_px > found[1].width_px, "largest first"


def test_max_blobs_caps_what_is_returned():
    image = _frame()
    for index in range(5):
        _rect(image, 200 + index * 300, 500, 150, 170)
    assert len(ColourBlobDetector(max_blobs=2).detect(image)) == 2


def test_the_measured_bin_ranges_to_its_measured_distance():
    """End-to-end against the numbers taken off the robot's own footage.

    On the staged scene the bin segmented to 164x183 px with the calibrated model
    (f=1290.2 px), and 0.3048 m of bin subtending 183 px is 2.15 m. This pins the whole
    chain — mask, box, prior, ranging — to a real measurement rather than to itself.
    """
    camera = FisheyeCamera(width=1920, height=1080, focal_px=1290.2, cx=960.0, cy=540.0)
    image = _rect(_frame(), 716, 559, 164, 183)
    found = ColourBlobDetector().detect(image)
    assert len(found) == 1, found
    range_m, source = estimate_range(found[0], camera, BLUE_BIN.prior)
    assert source == "height"
    assert abs(range_m - 2.15) < 0.05, range_m


def test_a_wrapped_hue_window_matches_both_sides_of_zero():
    """Red spans 170..179 and 0..10, so the window has to wrap."""
    red = ColourProfile(label="red", hue_lo=170, hue_hi=10, sat_min=90, val_min=40,
                        height_m=0.3, width_m=0.3, radius_m=0.15)
    hsv = np.zeros((10, 20, 3), dtype=np.uint8)
    hsv[:, :10] = (175, 200, 200)      # just below the wrap
    hsv[:, 10:] = (5, 200, 200)        # just above it
    assert red.mask(hsv).mean() == 255, "both sides of the wrap must match"


def test_an_unwrapped_window_does_not_match_outside_itself():
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[:] = (20, 200, 200)            # orange: outside the bin's 95..135
    assert BLUE_BIN.mask(hsv).max() == 0


def test_a_desaturated_blue_is_rejected():
    """Saturation is the gate that keeps the grey office out; prove it bites."""
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[:] = (110, BLUE_BIN.sat_min - 20, 200)   # right hue, washed out
    assert BLUE_BIN.mask(hsv).max() == 0


def test_boxes_are_returned_in_the_input_frames_pixels_at_any_scale():
    """The downscale is an implementation detail; callers must never see it."""
    image = _rect(_frame(), 800, 500, 160, 180)
    full = ColourBlobDetector(detect_scale=1.0).detect(image)[0]
    half = ColourBlobDetector(detect_scale=0.5).detect(image)[0]
    assert abs(full.x1 - half.x1) <= 4 and abs(full.width_px - half.width_px) <= 6


def test_the_profile_supplies_a_real_width_not_a_persons_aspect_ratio():
    """SizePrior.of_height would infer 0.09 m across for a 0.30 m bin."""
    assert abs(BLUE_BIN.prior.width_m - BLUE_BIN.width_m) < 1e-9
    assert BLUE_BIN.prior.width_m > 0.2, "a bin is not 9 cm wide"


def test_a_nonsense_profile_is_rejected_at_construction():
    for bad in ({"height_m": 0.0}, {"width_m": -1.0}, {"radius_m": 0.0},
                {"hue_lo": 200}, {"min_aspect": 3.0, "max_aspect": 1.0}):
        fields = {"label": "x", "hue_lo": 95, "hue_hi": 135, "sat_min": 90,
                  "val_min": 40, "height_m": 0.3, "width_m": 0.3, "radius_m": 0.15}
        fields.update(bad)
        try:
            ColourProfile(**fields)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_a_nonsense_detector_is_rejected_at_construction():
    for bad in ({"detect_scale": 0.0}, {"detect_scale": 1.5}, {"blur_px": 4}):
        try:
            ColourBlobDetector(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"colour_detector: {len(tests)}/{len(tests)} passed")
