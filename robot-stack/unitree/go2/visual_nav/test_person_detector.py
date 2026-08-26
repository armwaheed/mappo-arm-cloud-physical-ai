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
import pytest

from camera_model import PERSON_HEIGHT_M, PERSON_WIDTH_M, FisheyeCamera
from person_detector import (
    FILLS_FRAME_RANGE_M,
    PERSON_ASPECT_MIN,
    PERSON_PRIOR,
    Detection,
    RangedDetection,
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
    # "width-capped", not "width": the cap BOUND here, so what comes back is the cap and
    # not the width span, and a consumer has to be able to tell. That is the whole point
    # of the separate name — see UNMEASURED_SOURCES in tracker.
    assert source == "width-capped", source
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
    assert source == "width-capped", source
    assert range_m > 0.05, f"a bin reported at {range_m:.3f} m is inside the robot"


def test_a_person_still_switches_estimator_where_they_always_did():
    """The fix must not move the threshold for the case it was designed around."""
    camera = FisheyeCamera(width=1920, height=1080, focal_px=1290.2, cx=960.0, cy=540.0,
                           height_m=0.32)
    fit = object_fit_range(camera, PERSON_PRIOR)
    assert abs(fit - (PERSON_HEIGHT_M - 0.32) / math.tan((1080 / 2.0) / 1290.2)) < 1e-6
    assert 2.5 < fit < 3.5, fit


def test_the_cap_is_reported_as_its_own_source_only_when_it_binds():
    """Tonight's numbers. Approaching a bin, the width span read 0.748-0.907 m across
    thirteen frames against a 0.719 m fit range, so every one was capped and the reported
    range was 0.719 m to three decimals for five seconds. A span BELOW the fit range is a
    real measurement and keeps the plain name."""
    camera = FisheyeCamera(width=1920, height=1080, focal_px=1290.2, cx=960.0, cy=540.0,
                           height_m=0.32)
    bin_prior = SizePrior(height_m=0.3048, width_m=0.27)
    fit = object_fit_range(camera, bin_prior)

    # 438 px wide, bottom-clipped: the box measured on hardware at t=7.9.
    capped = Detection(x1=340.0, y1=624.0, x2=778.0, y2=1079.0, score=0.77, label="bin")
    range_m, source = estimate_range(capped, camera, bin_prior)
    assert source == "width-capped"
    assert math.isclose(range_m, fit)

    # Twice as wide in frame, so the span is well inside the cap and it does not bind.
    near = Detection(x1=100.0, y1=624.0, x2=1000.0, y2=1079.0, score=0.77, label="bin")
    range_m, source = estimate_range(near, camera, bin_prior)
    assert source == "width", source
    assert range_m < fit


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"person_detector: {len(tests)}/{len(tests)} passed")


# ── Routing on shape rather than on the label ───────────────────────────────
def _ranged(x1, y1, x2, y2, label="person"):
    """A RangedDetection with a given box. Range and bearing are irrelevant here —
    the whole point of the shape rule is that it does not consult them."""
    return RangedDetection(
        detection=Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=0.9, label=label),
        range_m=2.0, bearing_rad=0.0, source="height")


FRAME_W, FRAME_H = 1920, 1080


def test_a_peer_shaped_box_is_not_person_shaped_whatever_voc_called_it():
    """THE REASON THIS EXISTS. On 12 consecutive live frames the Go2 Wheel came back
    labelled `person`, and `mappo_bridge.HOLD_LABELS` would have stopped the robot for
    it every time. Aspect settles it: the peer corpus tops out at 0.99 over 1,159
    unclipped boxes against a standing adult's 1.70/0.50 = 3.40."""
    peer = _ranged(1150, 410, 1490, 790, label="person")   # 340 x 380 -> aspect 1.12
    assert peer.person_shaped(FRAME_W, FRAME_H) is False


def test_a_person_shaped_box_holds_even_when_voc_calls_it_furniture():
    """The same failure in the other direction, and the more dangerous one: across the
    2026-08-24 corpus the peer was called `motorbike` 613 times and `chair` 372, so a
    real person landing on one of those labels used to be handed straight to the policy.
    Shape stops that."""
    upright = _ranged(900, 200, 1000, 800, label="motorbike")   # 100 x 600 -> 6.0
    assert upright.person_shaped(FRAME_W, FRAME_H) is True


def test_a_vertically_clipped_box_is_unclassifiable_and_must_hold():
    """A person whose head leaves the frame gives a SHORTER box, so the aspect falls and
    they start to look like a quadruped — the dangerous direction, at close range where
    being wrong costs most. Vertical clipping must fail safe."""
    topped = _ranged(900, 0, 1240, 380, label="person")
    assert topped.person_shaped(FRAME_W, FRAME_H) is True
    bottomed = _ranged(900, 700, 1240, 1080, label="person")
    assert bottomed.person_shaped(FRAME_W, FRAME_H) is True


def test_a_horizontally_clipped_peer_still_reaches_the_policy():
    """THE OPPOSITE CASE, AND IT COST US A LIVE RUN. Cutting width RAISES height/width,
    so a partly-out-of-frame object drifts towards the person verdict by itself and
    needs no separate branch. The first cut of this rule refused on horizontal clipping
    too: on the first live run the peer clipped the right edge as the robot swerved past
    it, flipped to person_shaped, and froze the robot beside it.

    Re-add `or horizontal` to the guard and this fails.
    """
    beside = _ranged(1600, 410, 1920, 790, label="person")   # 320 x 380 -> aspect 1.19
    assert beside.person_shaped(FRAME_W, FRAME_H) is False


def test_a_sliver_at_the_frame_edge_holds_on_aspect_alone():
    """The safety this keeps despite the above: an object cut down to a narrow strip has
    an aspect over the threshold and holds, without the rule needing to know it was
    clipped. A peer must lose roughly two thirds of its width to get here."""
    sliver = _ranged(1830, 300, 1920, 800, label="person")   # 90 x 500 -> aspect 5.6
    assert sliver.person_shaped(FRAME_W, FRAME_H) is True


def test_the_threshold_sits_between_the_two_measured_populations():
    """Pins PERSON_ASPECT_MIN against the numbers that chose it. The peer's worst
    observed box is 0.99 and a standing adult is 3.40; a threshold outside (1.0, 3.4)
    would collapse one population into the other."""
    assert 1.0 < PERSON_ASPECT_MIN < PERSON_HEIGHT_M / PERSON_WIDTH_M
    just_under = _ranged(500, 100, 600, 100 + int(100 * PERSON_ASPECT_MIN) - 10)
    just_over = _ranged(500, 100, 600, 100 + int(100 * PERSON_ASPECT_MIN) + 10)
    assert just_under.person_shaped(FRAME_W, FRAME_H) is False
    assert just_over.person_shaped(FRAME_W, FRAME_H) is True


def test_a_degenerate_box_holds_rather_than_dividing_by_zero():
    """A zero-width box would raise on the aspect division. It must fail safe, not
    crash the perception thread."""
    assert _ranged(700, 200, 700, 800).person_shaped(FRAME_W, FRAME_H) is True


def test_the_width_prior_is_not_inferred_when_it_is_measured():
    """`of_height` alone fills width from a PERSON's aspect ratio: 0.514 m of peer
    becomes 0.151 m wide against a real ~0.31 m. Width ranges a vertically clipped box,
    and 39% of peer boxes are clipped, so the inferred prior reported the peer at
    0.09-0.14 m — inside the robot's own footprint."""
    inferred = SizePrior.of_height(0.514)
    assert inferred.width_m == pytest.approx(0.514 * PERSON_WIDTH_M / PERSON_HEIGHT_M)
    assert inferred.width_m < 0.16, "the bug this documents"
    measured = SizePrior.of_height(0.514, 0.31)
    assert measured.width_m == pytest.approx(0.31)
    assert measured.height_m == pytest.approx(0.514)
