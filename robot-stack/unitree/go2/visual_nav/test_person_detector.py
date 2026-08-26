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

from camera_model import (
    GO2_CAMERA_HEIGHT_M,
    PERSON_HEIGHT_M,
    PERSON_WIDTH_M,
    FisheyeCamera,
)
from person_detector import (
    DEFAULT_CONFIDENCE,
    DEFAULT_STATIC_CONFIDENCE,
    FILLS_FRAME_RANGE_M,
    PERSON_ASPECT_MIN,
    PERSON_PRIOR,
    STATIC_ASPECT_MIN,
    STATIC_CLASSES,
    STATIC_MAX_AREA_FRAC,
    STATIC_MIN_AREA_FRAC,
    Detection,
    GroundRanger,
    RangedDetection,
    SizePrior,
    estimate_range,
    object_fit_range,
    prototxt_with_floor,
    range_detections,
    static_shaped,
)

WIDTH, HEIGHT = 1920, 1080
# The lens height is NAMED, not defaulted: `FisheyeCamera.height_m` has no default, so a
# fixture that wants the Go2's geometry has to say so. Everything below that touches
# `object_fit_range` is a statement about a Go2-shaped camera.
MODEL = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0, height_m=GO2_CAMERA_HEIGHT_M)


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
    narrow = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 70.0,
                                     height_m=GO2_CAMERA_HEIGHT_M)
    assert object_fit_range(narrow) > object_fit_range(MODEL)


def test_the_fit_range_refuses_a_camera_that_states_no_lens_height():
    """The cap this function computes is proportional to the lens height, and there is no
    safe number to assume.

    `FisheyeCamera.height_m` used to default to the Go2's 0.32 m. Any Lite3 calibration
    that did not pass one inherited it in silence, and this is the function it reaches: it
    caps a width-derived range, which `estimate_range` applies to the 39% of peer boxes
    that touch a frame edge. Neither fallback is available — `inf` disables the cap, and
    0.0 reports every object as standing inside the robot — so it refuses.
    """
    unmeasured = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0)
    assert unmeasured.height_m is None, "the model must not supply a platform's number"
    with pytest.raises(ValueError) as refusal:
        object_fit_range(unmeasured)
    message = str(refusal.value)
    assert "no lens height" in message, message
    assert "height_m=" in message, "the refusal has to say what to pass"
    assert str(GO2_CAMERA_HEIGHT_M) in message, "and whose number 0.32 is"


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


# ── The sub-threshold STATIC tier ──────────────────────────────────────────
#: The DetectionOutput stanza as the published MobileNet-SSD prototxt ships it. Trimmed
#: to the layer under test; the real file is ~1,900 lines and carries this exactly once.
_DETECTION_OUT = """layer {
  name: "detection_out"
  type: "DetectionOutput"
  detection_output_param {
    num_classes: 21
    nms_param {
      nms_threshold: 0.45
      top_k: 100
    }
    keep_top_k: 100
    confidence_threshold: 0.25
  }
}
"""

#: The cardboard box as MobileNet-SSD actually saw it, at input 300, on the only frame of
#: one that exists: a Lite3 dry run in the Shanghai office, 1280x720. Score 0.1221,
#: labelled `chair`, box [535,165,800,547]. ONE OBSERVATION — it demonstrates the
#: mechanism and sets nothing.
_LITE3_FRAME = (1280, 720)
_LITE3_BOX = (535.0, 165.0, 800.0, 547.0)
_LITE3_SCORE = 0.1221


def _det(x1, y1, x2, y2, label="chair", score=0.12):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, label=label)


def test_the_network_floor_is_lowered_not_the_python_one():
    """The 0.25 lives in DetectionOutput and is applied inside forward(), so a Python
    threshold can only ever discard what the layer already passed. Lowering the layer is
    the entire mechanism; if this substitution stops happening the tier sees nothing."""
    assert "confidence_threshold: 0.25" in _DETECTION_OUT
    patched = prototxt_with_floor(_DETECTION_OUT, 0.10)
    assert "confidence_threshold: 0.1" in patched
    assert "confidence_threshold: 0.25" not in patched
    # Everything else about the layer is untouched — nms_threshold is a number in the
    # same stanza and a sloppier pattern would eat it.
    assert "nms_threshold: 0.45" in patched
    assert "keep_top_k: 100" in patched
    assert "num_classes: 21" in patched


def test_the_floor_is_never_raised():
    """A caller asking for a permissive static tier must not be able to make the PERSON
    tier blinder as a side effect. Above the baked value the file comes back unchanged."""
    assert prototxt_with_floor(_DETECTION_OUT, 0.40) == _DETECTION_OUT
    assert prototxt_with_floor(_DETECTION_OUT, 0.25) == _DETECTION_OUT


def test_an_ambiguous_prototxt_is_refused_rather_than_guessed():
    """Two thresholds would leave one of them silently authoritative, and the run would
    look like it had lowered the floor while DetectionOutput still dropped the box."""
    with pytest.raises(ValueError):
        prototxt_with_floor(_DETECTION_OUT + _DETECTION_OUT, 0.10)
    with pytest.raises(ValueError):
        prototxt_with_floor("layer { type: \"DetectionOutput\" }", 0.10)


def test_the_one_box_that_exists_passes_the_gate():
    """The Lite3 cardboard box, at its measured geometry. This is a DEMONSTRATION that
    the mechanism reaches the object, not evidence that the thresholds are right: one
    frame cannot set a threshold and this test must never be read as if it had."""
    assert static_shaped(_det(*_LITE3_BOX), *_LITE3_FRAME) is True
    assert _LITE3_SCORE >= DEFAULT_STATIC_CONFIDENCE
    assert _LITE3_SCORE < DEFAULT_CONFIDENCE, "below the mover tier, which is the point"


def test_a_person_shaped_box_is_refused_entry_to_the_static_map():
    """THE SAFETY-CRITICAL HALF. A landmark is planned with person_shaped=False, so a
    person routed into the map is a person the robot never holds for — the one behaviour
    it ships. `chair` at close range on a real person's legs is a measured occurrence,
    not a hypothetical."""
    tall = _det(500.0, 100.0, 800.0, 100.0 + 300.0 * PERSON_ASPECT_MIN + 10.0)
    frac = (tall.width_px * tall.height_px) / (1920.0 * 1080.0)
    assert STATIC_MIN_AREA_FRAC <= frac <= STATIC_MAX_AREA_FRAC, "area must not decide"
    assert static_shaped(tall, 1920, 1080) is False


def test_a_vertically_clipped_box_is_refused_for_the_reason_person_shaped_holds():
    """Clipping the head SHORTENS the box, so the aspect falls toward furniture at exactly
    the range where being wrong costs most. `person_shaped` holds on that; this must
    refuse on it, or the two rules disagree and a close person becomes scenery."""
    clipped_top = _det(500.0, 0.0, 900.0, 700.0)
    clipped_bottom = _det(500.0, 300.0, 900.0, 1080.0)
    assert static_shaped(clipped_top, 1920, 1080) is False
    assert static_shaped(clipped_bottom, 1920, 1080) is False


def test_the_person_refusal_uses_the_same_threshold_person_shaped_does():
    """Not a parallel constant. If these drift apart a band opens between them that is
    neither held for nor mapped, or worse is both."""
    # Wide enough that the AREA gate is not what answers — this test is about the
    # aspect boundary and must not pass for the other gate's reason.
    width = 300.0
    just_under = _det(500.0, 100.0, 500.0 + width,
                      100.0 + width * (PERSON_ASPECT_MIN - 0.1))
    just_over = _det(500.0, 100.0, 500.0 + width,
                     100.0 + width * (PERSON_ASPECT_MIN + 0.1))
    assert static_shaped(just_under, 1920, 1080) is True
    assert static_shaped(just_over, 1920, 1080) is False
    ranged = RangedDetection(detection=just_over, range_m=2.0, bearing_rad=0.0,
                             source="height")
    assert ranged.person_shaped(1920, 1080) is True


def test_a_horizontal_slab_is_not_a_floor_standing_object():
    """A skirting rail, a window mullion, the wall-wide `aeroplane` at aspect 0.19 in the
    Lite3 frame. Wide and flat is the corridor, not something in it."""
    slab = _det(200.0, 400.0, 1500.0, 560.0)
    frac = (slab.width_px * slab.height_px) / (1920.0 * 1080.0)
    assert STATIC_MIN_AREA_FRAC <= frac <= STATIC_MAX_AREA_FRAC, "area must not decide"
    assert slab.height_px / slab.width_px < STATIC_ASPECT_MIN
    assert static_shaped(slab, 1920, 1080) is False


def test_a_small_distant_blob_is_below_the_area_floor():
    """The gate with margin. This pipeline has no depth sensor and ranges by a prior it
    must be told, so a small far blob is both the least trustworthy detection and the
    least urgent. Keeping the tier to the near field is what makes its false-alarm rate
    nil on empty corridor."""
    small = _det(900.0, 500.0, 1000.0, 620.0)
    frac = (small.width_px * small.height_px) / (1920.0 * 1080.0)
    assert frac < STATIC_MIN_AREA_FRAC
    assert static_shaped(small, 1920, 1080) is False


def test_a_box_covering_the_frame_is_the_detector_describing_the_corridor():
    """The 0.1475 `person` covering 81% of the Lite3 frame is the worked example. Mapped
    as a landmark it puts a disc on top of the robot."""
    huge = _det(200.0, 5.0, 1750.0, 1075.0)
    frac = (huge.width_px * huge.height_px) / (1920.0 * 1080.0)
    aspect = huge.height_px / huge.width_px
    # The AREA ceiling must be what refuses this. An earlier version of this box was
    # 0.585 wide-and-flat, so the slab rule rejected it first and the test passed while
    # STATIC_MAX_AREA_FRAC did nothing — green for a reason it was not testing.
    assert STATIC_ASPECT_MIN <= aspect < PERSON_ASPECT_MIN, "aspect must not decide"
    assert frac > STATIC_MAX_AREA_FRAC
    assert static_shaped(huge, 1920, 1080) is False


def test_a_degenerate_box_is_refused_rather_than_dividing_by_zero():
    """`person_shaped` fails SAFE by holding; this fails safe by declining to map. Both
    directions are 'do not quietly treat it as scenery'."""
    assert static_shaped(_det(700.0, 200.0, 700.0, 800.0), 1920, 1080) is False
    assert static_shaped(_det(700.0, 200.0, 900.0, 200.0), 1920, 1080) is False


def test_person_is_not_a_static_class():
    """A label saying `person` outright is a configuration error, not a shape question.
    The default set must never contain it, whatever else it holds."""
    assert "person" not in STATIC_CLASSES


# ── ranging with no prior at all: the floor contact point ───────────────────
#: The measured Go2 front camera — `calibrate_camera.py --spin`, 85.27 deg — rather than
#: the 120 deg nominal `MODEL` above. The near wall and the accuracy ceiling are both
#: statements about a real lens, and a nominal one would move both.
GO2 = FisheyeCamera(width=WIDTH, height=HEIGHT, focal_px=1290.1637909789656,
                    cx=WIDTH / 2.0, cy=HEIGHT / 2.0, height_m=GO2_CAMERA_HEIGHT_M)

#: 1.6 deg is the MEDIAN of the upper bound two recorded runs support (71 unclipped
#: sightings, six tracks, 2026-08-25), not a measurement of trunk pitch — see
#: `GroundRanger`. 0.18 is `tracker.RANGE_SIGMA_FRACTION`, i.e. what the filter already
#: budgets for the source it trusts most. Together they put the ceiling at ~1.70 m.
STATED_WOBBLE_RAD = math.radians(1.6)
STATED_TOLERANCE = 0.18


def _standing(range_m: float, height_m: float, width_px: float = 180.0,
              lateral_m: float = 0.0, camera: FisheyeCamera = GO2) -> Detection:
    """A box for an object of ``height_m`` STANDING ON THE FLOOR at ``range_m``.

    Built by projecting the object's real base and crown rather than by inverting the
    ranger, so a test using it is not assuming the answer.
    """
    forward = math.sqrt(max(range_m ** 2 - lateral_m ** 2, 0.0))
    base_u, base_v = camera.project((forward, lateral_m, -camera.height_m))
    _top_u, top_v = camera.project((forward, lateral_m, height_m - camera.height_m))
    # A fixture whose crown or base has left the frame is CLIPPED, and every ranger here
    # refuses a clipped box — so it would test the refusal while reading as a test of the
    # range. This robot's frame holds `0.32 + 0.4448·range` metres of object; refuse to
    # build one that does not fit rather than let a later edit fall off the top.
    assert top_v > 2.0, (f"a {height_m} m object at {range_m} m leaves the top of the "
                         f"frame (v={top_v:.0f}); this fixture would be clipped")
    assert base_v < camera.height - 2.0, (f"the base of a {range_m} m object is off the "
                                          f"bottom (v={base_v:.0f})")
    return Detection(x1=base_u - width_px / 2.0, y1=top_v,
                     x2=base_u + width_px / 2.0, y2=base_v, score=0.2, label="chair")


def _ranger(wobble_rad: float = STATED_WOBBLE_RAD,
            tolerance: float = STATED_TOLERANCE) -> GroundRanger:
    return GroundRanger(GO2, wobble_rad, tolerance)


def test_the_ground_ranger_recovers_a_staged_range_with_no_size_prior():
    """The property the whole class-agnostic route rests on. Nothing is told how big the
    object is, at three different heights and off-axis, and the range comes back."""
    ranger = _ranger()
    for range_m, height_m, lateral_m in ((0.9, 0.30, 0.0), (1.2, 0.60, 0.0),
                                         (1.5, 0.90, 0.0), (1.2, 0.60, 0.5)):
        got, source = ranger(_standing(range_m, height_m, lateral_m=lateral_m))
        assert source == "ground", (range_m, height_m, source)
        assert abs(got - range_m) < 1e-6, (range_m, height_m, lateral_m, got)


def test_the_ground_ranger_beats_a_size_prior_that_was_never_measured():
    """THE ARGUMENT FOR THIS ISSUE, as a test. `estimate_range` divides apparent size by
    an assumed one, so an unmeasured object is ranged by whatever prior happened to be
    configured — and the error is the ratio of the two sizes, unbounded.

    Not hypothetical: the two live runs of 2026-08-25 ranged every detection, including
    the goal chair, with the peer robot's 0.514 m prior. Here a 0.60 m object at 1.20 m
    is ranged with a 1.20 m prior and comes back at 2.40 m — a real obstacle reported at
    twice its distance, which is the direction that walks into it. The contact point
    needs no prior and is exact.
    """
    detection = _standing(1.20, 0.60)
    too_far, source = estimate_range(detection, GO2, SizePrior(height_m=1.20, width_m=0.3))
    assert source == "height"
    assert abs(too_far - 2.40) < 0.02, too_far
    from_ground, ground_source = _ranger()(detection)
    assert ground_source == "ground"
    assert abs(from_ground - 1.20) < 1e-6, from_ground


def test_the_ground_ranger_refuses_at_the_near_wall_and_does_not_substitute_a_constant():
    """Below ~0.72 m the contact point has left the frame. The two constants that already
    exist for that band — `FILLS_FRAME_RANGE_M` and the width-prior fit cap — between them
    deadlocked a live run for five seconds on 2026-08-19, because a planner cannot tell a
    constant from a measurement. This returns `inf`, which `range_detections` drops."""
    ranger = _ranger()
    nearest = GO2.ground_range(GO2.cx, HEIGHT - 1.0)
    assert 0.70 < nearest < 0.74, nearest
    clipped = Detection(x1=800.0, y1=200.0, x2=1100.0, y2=HEIGHT - 1.0,
                        score=0.2, label="chair")
    range_m, source = ranger(clipped)
    assert source == "ground-clipped"
    assert range_m == math.inf
    assert range_m != FILLS_FRAME_RANGE_M, "no fallback constant on this path"


def test_a_box_that_is_not_vertically_clipped_is_already_past_the_near_wall():
    """Why there is no separate minimum-range test: the vertical-clip refusal IS the near
    wall. A box whose bottom edge is strictly inside the frame has its contact point
    inside the frame too, so every range this reports is at least the near limit."""
    ranger = _ranger()
    nearest = GO2.ground_range(GO2.cx, HEIGHT - 1.0)
    for range_m in (0.75, 0.9, 1.4):
        got, source = ranger(_standing(range_m, 0.45))
        assert source == "ground"
        assert got >= nearest - 1e-9, (range_m, got, nearest)


def test_the_ground_ranger_refuses_past_its_accuracy_ceiling():
    """The far half of the tree's objection, enforced rather than argued. Beyond the
    ceiling the wobble alone exceeds the stated tolerance, so there is no measurement to
    report — and it is a refusal, not a widened radius."""
    ranger = _ranger()
    assert abs(ranger.range_limit_m - 1.696) < 0.01, ranger.range_limit_m
    inside, source = ranger(_standing(ranger.range_limit_m * 0.95, 0.6))
    assert source == "ground" and math.isfinite(inside)
    outside, far_source = ranger(_standing(ranger.range_limit_m * 1.10, 0.6))
    assert far_source == "ground-far"
    assert outside == math.inf


def test_the_ceiling_moves_with_the_two_numbers_the_caller_states():
    """Neither knob has a default, and both have to bite. A tighter tolerance or a bigger
    stated wobble must shrink the band."""
    assert _ranger(tolerance=0.30).range_limit_m > _ranger(tolerance=0.18).range_limit_m
    assert _ranger(wobble_rad=math.radians(3.0)).range_limit_m < _ranger().range_limit_m


def test_a_ranger_with_no_usable_band_refuses_at_construction():
    """A 2 deg wobble held to 5% has a ceiling of zero — inside the near wall, so it would
    refuse every detection forever and look from outside like a detector that stopped
    working. This repository has already shipped a gate that could never fire; a
    configuration that can never succeed fails where it is written instead."""
    with pytest.raises(ValueError) as refusal:
        GroundRanger(GO2, math.radians(2.0), 0.05)
    message = str(refusal.value)
    assert "no usable range" in message, message
    assert "max_error_frac" in message, "the refusal has to say what to change"
    assert "raise the camera" in message, "and that the mount is the other lever"


def test_the_ground_ranger_refuses_a_box_that_floats_above_the_horizon():
    """A box whose bottom edge is above the skyline has no floor intersection at all.
    That is junk rather than a near-field case, and it gets its own reason so a log can
    tell the two apart."""
    floating = Detection(x1=800.0, y1=100.0, x2=1000.0, y2=300.0, score=0.2, label="chair")
    range_m, source = _ranger()(floating)
    assert source == "ground-horizon"
    assert range_m == math.inf


def test_range_detections_is_byte_identical_without_a_ranger():
    """The default-off pin. `ranger=None` is the signature this function has always had,
    and every existing caller passes nothing."""
    detections = [_box_for(3.0), _box_for(1.5), _box_for(0.9)]
    with_default = range_detections(detections, MODEL)
    explicit = [estimate_range(d, MODEL, PERSON_PRIOR) for d in detections]
    assert len(with_default) == len(detections)
    for ranged, (range_m, source) in zip(with_default, explicit):
        assert ranged.range_m == range_m and ranged.source == source


def test_range_detections_with_a_ranger_ignores_the_prior_entirely():
    """A `ranger` replaces the size-prior estimator, so the prior must have NO effect.
    If it leaked through, the class-agnostic route would still be class-bound and the
    bug would be invisible — both answers look plausible."""
    detections = [_standing(1.2, 0.6), _standing(1.5, 0.9)]
    ranger = _ranger()
    tiny = range_detections(detections, GO2, SizePrior(height_m=0.05, width_m=0.05),
                            ranger=ranger)
    huge = range_detections(detections, GO2, SizePrior(height_m=5.0, width_m=5.0),
                            ranger=ranger)
    assert [r.range_m for r in tiny] == [r.range_m for r in huge]
    assert [round(r.range_m, 6) for r in tiny] == [1.2, 1.5]
    assert {r.source for r in tiny} == {"ground"}


def test_range_detections_drops_every_ground_refusal():
    """`inf` is how this path says "no measurement", and the existing `isfinite` filter is
    what makes that reach nothing downstream. A refusal that survived into the map would
    be a landmark at infinity."""
    ranger = _ranger()
    usable = _standing(1.2, 0.6)
    clipped = Detection(x1=800.0, y1=200.0, x2=1100.0, y2=HEIGHT - 1.0,
                        score=0.2, label="chair")
    too_far = _standing(ranger.range_limit_m * 1.5, 0.6)
    kept = range_detections([clipped, usable, too_far], GO2, ranger=ranger)
    assert len(kept) == 1, [(k.range_m, k.source) for k in kept]
    assert abs(kept[0].range_m - 1.2) < 1e-6


def test_implied_height_recovers_an_object_dimension_from_geometry_alone():
    """The diagnostic that validated this estimator and would size the refusal it does not
    have. Over the five sightings of the parked peer in the 2026-08-25 hero run that a
    1.6 deg / 18% ceiling keeps, it implies 0.494 m against the 0.514 m the run was told.

    Within a few per cent, not exact, and the residual is not noise: it is
    `range_from_span`'s assumption that the extent is BISECTED by the line of sight,
    which a short object standing below a 0.32 m lens is not. The bias is largest for the
    shortest object here (0.30 m reads 0.29 m) and vanishes as the object's mid-height
    approaches the lens. It is a bias on a diagnostic, not on the range.
    """
    ranger = _ranger()
    for range_m, height_m in ((0.9, 0.30), (1.2, 0.60), (1.5, 0.90)):
        implied = ranger.implied_height_m(_standing(range_m, height_m))
        assert abs(implied - height_m) / height_m < 0.05, (range_m, height_m, implied)
    # The bias has a direction and a cause; pin both so a real regression is separable.
    short = ranger.implied_height_m(_standing(0.9, 0.30))
    tall = ranger.implied_height_m(_standing(1.5, 0.90))
    assert short < 0.30, short
    assert abs(tall - 0.90) / 0.90 < abs(short - 0.30) / 0.30, (short, tall)


def test_implied_height_is_none_for_anything_the_ranger_refused():
    """A height read off a refusal would be a number with no measurement under it."""
    ranger = _ranger()
    clipped = Detection(x1=800.0, y1=200.0, x2=1100.0, y2=HEIGHT - 1.0,
                        score=0.2, label="chair")
    assert ranger.implied_height_m(clipped) is None
    assert ranger.implied_height_m(_standing(ranger.range_limit_m * 1.5, 0.6)) is None


def test_a_box_that_does_not_reach_the_floor_reads_far_and_nothing_here_catches_it():
    """The residual risk, pinned so it cannot be forgotten. Something standing on a table,
    or a box whose lower half the detector cut off, gives a shallower elevation and so a
    LONGER range — the unsafe direction. No threshold is invented to guess at it; what
    exists is `implied_height_m`, which reports an object taller than the thing could be,
    and the `static_shaped` gates upstream."""
    ranger = _ranger()
    truthful = _standing(1.0, 0.5)
    lifted = Detection(x1=truthful.x1, y1=truthful.y1, x2=truthful.x2,
                       y2=truthful.y2 - 120.0, score=0.2, label="chair")
    honest, _ = ranger(truthful)
    optimistic, source = ranger(lifted)
    assert source == "ground"
    assert optimistic > honest * 1.3, (honest, optimistic)
    assert ranger.implied_height_m(lifted) > 0.5, "the diagnostic is what would catch it"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"person_detector: {len(tests)}/{len(tests)} passed")
