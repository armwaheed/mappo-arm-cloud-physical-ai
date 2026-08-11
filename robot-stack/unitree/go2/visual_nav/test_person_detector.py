#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for detection ranging and frame-edge truncation.

The network itself is not exercised (it needs weights and a real image); what is
tested is the geometry around it, and specifically the case that is easy to get wrong
and dangerous when you do: a person close enough that the camera clips their head. A
clipped box is SHORTER than the person, so a naive height-prior ranger reports them
further away than they are — the one error direction that gets someone walked into.

Run: ``python3 test_person_detector.py``
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_model import PERSON_HEIGHT_M, FisheyeCamera
from person_detector import (
    FILLS_FRAME_RANGE_M,
    PERSON_PRIOR,
    Detection,
    SizePrior,
    estimate_range,
    object_fit_range,
)

WIDTH, HEIGHT = 1920, 1080
MODEL = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0)


def _box_for(distance_m: float, height_m: float = PERSON_HEIGHT_M) -> Detection:
    """A centred, untruncated box for a person of ``height_m`` at ``distance_m``."""
    half = MODEL.focal_px * math.atan2(height_m / 2.0, distance_m)
    width_half = MODEL.focal_px * math.atan2(0.50 / 2.0, distance_m)
    return Detection(x1=MODEL.cx - width_half, y1=MODEL.cy - half,
                     x2=MODEL.cx + width_half, y2=MODEL.cy + half,
                     score=0.9, label="person")


def test_untruncated_box_ranges_from_height():
    for distance in (2.5, 4.0, 6.0):
        detection = _box_for(distance)
        estimate, source = estimate_range(detection, MODEL)
        assert source == "height", source
        assert abs(estimate - distance) / distance < 0.01, (distance, estimate)


def test_range_scales_inversely_with_box_height():
    near, _ = estimate_range(_box_for(2.5), MODEL)
    far, _ = estimate_range(_box_for(5.0), MODEL)
    assert abs(far / near - 2.0) < 0.05, (near, far)


def test_clipped_returns_the_frame_fill_distance():
    full_frame = Detection(x1=0.0, y1=0.0, x2=WIDTH - 1.0, y2=HEIGHT - 1.0,
                           score=0.9, label="person")
    estimate, source = estimate_range(full_frame, MODEL)
    assert source == "frame-fill"
    assert estimate == FILLS_FRAME_RANGE_M


def test_vertically_clipped_falls_back_to_width():
    # Head and feet out of frame, body still fully in shot horizontally.
    detection = Detection(x1=800.0, y1=0.0, x2=1120.0, y2=HEIGHT - 1.0,
                          score=0.9, label="person")
    estimate, source = estimate_range(detection, MODEL)
    assert source == "width", source
    assert estimate <= object_fit_range(MODEL) + 1e-9


def test_clipped_box_is_never_reported_as_far_away():
    """The safety property: truncation caps the range instead of inflating it."""
    # A narrow silhouette (someone in profile) clipped at the top. Width alone would
    # say "miles away"; the cap must stop that.
    detection = Detection(x1=940.0, y1=0.0, x2=980.0, y2=900.0, score=0.9,
                          label="person")
    estimate, source = estimate_range(detection, MODEL)
    assert source == "width"
    assert estimate <= object_fit_range(MODEL) + 1e-9, estimate


def test_range_is_proportional_to_the_size_prior():
    """The whole metric scale rides on the prior, so an error in it scales the range.

    Pins that a custom prior is actually honoured — the path used when a hand-slid
    object stands in for a person.
    """
    detection = _box_for(3.0)
    with_person, _ = estimate_range(detection, MODEL)
    half_size, _ = estimate_range(detection, MODEL,
                                  SizePrior(height_m=PERSON_HEIGHT_M / 2.0))
    assert abs(half_size - with_person / 2.0) < 1e-9


def test_size_prior_of_height_keeps_the_person_aspect_ratio():
    prior = SizePrior.of_height(0.254)
    assert prior.height_m == 0.254
    assert abs(prior.width_m / prior.height_m
               - SizePrior().width_m / SizePrior().height_m) < 1e-9


def test_object_fit_range_matches_the_geometry():
    """Below this distance a standing person's head leaves the frame."""
    fit = object_fit_range(MODEL)
    assert 1.5 < fit < 3.0, fit
    # A person exactly at the fit distance should just about fill the upper half.
    half_vfov = (HEIGHT / 2.0) / MODEL.focal_px
    expected = (PERSON_HEIGHT_M - MODEL.height_m) / math.tan(half_vfov)
    assert abs(fit - expected) < 1e-9


def test_fit_range_grows_with_a_narrower_lens():
    narrow = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 70.0)
    assert object_fit_range(narrow) > object_fit_range(MODEL)


def test_clipped_flags():
    inside = _box_for(4.0)
    assert inside.clipped(WIDTH, HEIGHT) == (False, False)
    top = Detection(x1=800.0, y1=1.0, x2=1100.0, y2=900.0, score=0.9, label="person")
    assert top.clipped(WIDTH, HEIGHT) == (True, False)
    side = Detection(x1=0.0, y1=200.0, x2=400.0, y2=900.0, score=0.9, label="person")
    assert side.clipped(WIDTH, HEIGHT) == (False, True)


def test_detection_geometry_helpers():
    detection = Detection(x1=100.0, y1=200.0, x2=300.0, y2=600.0, score=0.5,
                          label="person")
    assert detection.width_px == 200.0
    assert detection.height_px == 400.0
    assert detection.centre == (200.0, 400.0)


# ── Objects shorter than the camera ─────────────────────────────────────────
def test_a_short_object_does_not_fit_at_zero_metres():
    """object_fit_range used to take only "when does the TOP leave the frame?".

    For anything shorter than the camera mount that difference is negative, clamped to
    0.0 — and estimate_range caps a width-derived range at this value with min(). So a
    0.30 m bin came back at ZERO metres: an obstacle inside the robot, a permanently
    negative gap, and a hold for the rest of the run. The binding end for a short object
    is its BASE.
    """
    camera = FisheyeCamera(width=1920, height=1080, focal_px=1290.2, cx=960.0, cy=540.0,
                           height_m=0.32)
    bin_prior = SizePrior(height_m=0.3048, width_m=0.27)
    fit = object_fit_range(camera, bin_prior)
    assert fit > 0.0, "a bin has to stop fitting SOMEWHERE, and it is not at 0 m"
    # 0.32 m of camera height over tan(half-VFOV) — the base leaving the bottom edge.
    assert abs(fit - 0.32 / math.tan((1080 / 2.0) / 1290.2)) < 1e-6, fit


def test_a_clipped_short_object_is_not_reported_at_zero_range():
    """The consequence, end to end."""
    camera = FisheyeCamera(width=1920, height=1080, focal_px=1290.2, cx=960.0, cy=540.0,
                           height_m=0.32)
    bin_prior = SizePrior(height_m=0.3048, width_m=0.27)
    clipped = Detection(x1=800.0, y1=700.0, x2=1100.0, y2=1079.0, score=0.9, label="bin")
    range_m, source = estimate_range(clipped, camera, bin_prior)
    assert source == "width"
    assert range_m > 0.05, f"a bin reported at {range_m:.3f} m is inside the robot"


def test_a_person_still_switches_estimator_where_they_always_did():
    """The fix must not move the threshold for the case it was designed around."""
    camera = FisheyeCamera(width=1920, height=1080, focal_px=1290.2, cx=960.0, cy=540.0,
                           height_m=0.32)
    fit = object_fit_range(camera, PERSON_PRIOR)
    assert abs(fit - (PERSON_HEIGHT_M - 0.32) / math.tan((1080 / 2.0) / 1290.2)) < 1e-6
    assert 2.5 < fit < 3.5, fit


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"person_detector: {len(tests)}/{len(tests)} passed")
