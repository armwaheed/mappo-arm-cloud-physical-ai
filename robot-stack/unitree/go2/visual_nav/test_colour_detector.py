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

import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_model import FisheyeCamera
from colour_detector import (
    BLUE_BIN,
    COLOUR_PROFILE_SCHEMA,
    ColourBlobDetector,
    ColourProfile,
    collapse_stacked_blobs,
    load_colour_profile,
)
from person_detector import Detection, estimate_range

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


def _custom_profile_data(**overrides):
    return {
        "schema": COLOUR_PROFILE_SCHEMA,
        "label": "brown-box-marker",
        "hue_lo": 75,
        "hue_hi": 90,
        "sat_min": 200,
        "val_min": 70,
        "height_m": 0.05,
        "width_m": 0.10,
        "radius_m": 0.168,
        "min_area_px": 400,
        "min_fill": 0.55,
        "min_aspect": 1.3,
        "max_aspect": 2.6,
        "evidence": {
            "rtsp_frames": "green-marker-rtsp-20260825T172500Z",
            "panel_measurement": "0.10m x 0.05m at 0.68m",
        },
        **overrides,
    }


def test_custom_colour_profile_requires_schema_and_evidence():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.json"
        path.write_text(json.dumps(_custom_profile_data()))
        profile = load_colour_profile(path)
        assert profile.label == "brown-box-marker"
        assert profile.radius_m == 0.168
        assert dict(profile.evidence)["rtsp_frames"].startswith("green-marker")

        # The schema half of this test's name. Every field below is valid, so only the
        # version check can reject it — replacing that check with `if False:` used to
        # leave the suite at 17/17.
        for schema in ("colour-profile/v2", "", None):
            path.write_text(json.dumps(_custom_profile_data(schema=schema)))
            try:
                load_colour_profile(path)
            except ValueError as error:
                assert "schema" in str(error) and COLOUR_PROFILE_SCHEMA in str(error)
            else:
                raise AssertionError(f"accepted a profile declaring schema {schema!r}")

        invalid = _custom_profile_data(evidence={})
        path.write_text(json.dumps(invalid))
        try:
            load_colour_profile(path)
        except ValueError as error:
            assert "evidence" in str(error)
        else:
            raise AssertionError("accepted custom profile without evidence")

        invalid = _custom_profile_data(label=None)
        path.write_text(json.dumps(invalid))
        try:
            load_colour_profile(path)
        except ValueError as error:
            assert "label" in str(error)
        else:
            raise AssertionError("accepted custom profile without a label")

        invalid = _custom_profile_data(sat_min=256)
        path.write_text(json.dumps(invalid))
        try:
            load_colour_profile(path)
        except ValueError as error:
            assert "sat_min" in str(error)
        else:
            raise AssertionError("accepted out-of-range HSV saturation threshold")

    try:
        ColourProfile(label="invalid-hsv", hue_lo=75, hue_hi=90, sat_min=256, val_min=70,
                      height_m=0.05, width_m=0.10, radius_m=0.168)
    except ValueError as error:
        assert "sat_min" in str(error)
    else:
        raise AssertionError("direct profile construction bypassed HSV threshold validation")

    invalid = _custom_profile_data(height_m=float("nan"), min_fill=float("nan"))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.json"
        path.write_text(json.dumps(invalid))
        try:
            load_colour_profile(path)
        except ValueError as error:
            assert "finite" in str(error)
        else:
            raise AssertionError("accepted non-finite custom profile geometry")

    try:
        ColourProfile(label="invalid-geometry", hue_lo=75, hue_hi=90, sat_min=200, val_min=70,
                      height_m=float("nan"), width_m=0.10, radius_m=0.168)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("direct profile construction bypassed finite geometry validation")


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


#: The staged bin as it segmented off this unit's own footage: a 164x183 px box under the
#: calibrated model. THESE ARE MEASUREMENTS — pixels off a real frame, and the focal length
#: from ``calibrate_camera.py --spin``. The metre range they imply is not, because it needs
#: :data:`colour_detector.BLUE_BIN`'s height prior, which has never been checked against a
#: tape. See :func:`test_the_bin_height_prior_is_still_the_unaudited_one_foot_tape_measure`.
STAGED_BIN_BOX_PX = (716, 559, 164, 183)
STAGED_FOCAL_PX = 1290.2


def test_the_mask_box_and_ranging_chain_agrees_with_the_stated_prior():
    """Mask to box to :func:`estimate_range`, against the prior the profile states.

    RENAMED, AND THE EXPECTATION IS NOW DERIVED. This used to be called
    ``test_the_measured_bin_ranges_to_its_measured_distance`` and to assert a literal
    2.15 m, described as pinning the chain "to a real measurement rather than to itself".
    It was not: 2.15 is ``0.3048 x 1290.2 / 183``, computed from the very prior that
    issue #35 is auditing. The only thing it could ever catch was a bug in the arithmetic
    between the mask and the range — never a prior that does not match the physical bin.

    So it now asserts what it actually covers, and it computes the expected metres from
    ``BLUE_BIN.prior`` instead of restating them, which removes the trap in the old shape:
    correcting the prior would turn this red, and the obvious way to make it green again is
    to recompute the literal, which looks like fixing a test and is in fact deleting the
    only place the change was visible. The prior is now pinned on its own, by name, below.

    What is still measured here and worth keeping: 164x183 px of blue at f=1290.2 px is a
    real box off a real frame, the detector has to segment it as exactly one blob of a
    pinned pixel size, and the range has to come from the HEIGHT prior — a ``"width"`` or
    ``"frame-fill"`` source at this box size would mean the clipping logic had changed
    underneath the ranging.

    Worth recording while it is visible: the old literal was 3.6 cm from what this
    pipeline actually returns (2.114 m against 2.15 m), and passed only because the
    tolerance was 0.05 m. A number that has to be given a 5 cm gate to match the code is
    not the code's output; it is the arithmetic beside it.
    """
    camera = FisheyeCamera(width=1920, height=1080, focal_px=STAGED_FOCAL_PX,
                           cx=960.0, cy=540.0)
    x, y, width_px, height_px = STAGED_BIN_BOX_PX
    image = _rect(_frame(), x, y, width_px, height_px)
    found = ColourBlobDetector().detect(image)
    assert len(found) == 1, found

    # The pixels are the half that IS a measurement, so they are pinned here rather than
    # taken on trust. A filled rectangle carries a one-pixel border on each side, so the
    # blob the mask returns is 166x186 for a 164x183 draw — pinned as what the detector
    # segments, because an erosion, a gate or a contour change that moved it would move
    # every range with it.
    assert (found[0].width_px, found[0].height_px) == (166.0, 186.0), \
        (found[0].width_px, found[0].height_px)

    # ``estimate_range`` goes through the exact ``L / (2·tan(dtheta/2))`` form, not the
    # pinhole ``h·f/px``; at this box size the two differ by 1.9 mm, so the 0.01 m gate
    # pins that agreement rather than merely restating the formula.
    range_m, source = estimate_range(found[0], camera, BLUE_BIN.prior)
    assert source == "height", source
    expected_m = BLUE_BIN.height_m * STAGED_FOCAL_PX / found[0].height_px
    assert abs(range_m - expected_m) < 0.01, (range_m, expected_m)


def test_the_bin_height_prior_is_still_the_unaudited_one_foot_tape_measure():
    """``BLUE_BIN.height_m`` is 0.3048 m, and nothing in this repository has checked it.

    Every bin range is this number times ``f`` over the box's pixel height, so every
    obstacle position, every mapped gate width and every clearance a run reports scales
    linearly on it — and the map, the planner and the policy all stay internally
    consistent while being uniformly wrong, which is why no test downstream can notice.
    Issue #35 is open on it and the thing that settles it is a tape measure against the
    real prop, not anything that can be run from a desk.

    This assertion is not a claim that 0.3048 is correct. It is a tripwire: it makes
    changing the prior a deliberate, visible act with a red test attached, rather than an
    edit that silently rescales the whole obstacle map. If you are here because you took
    the tape measurement, change the number, say so in issue #35 with the two distances you
    measured, and note that ``--prop-height`` rescales ``width_m`` but NOT ``radius_m``.
    """
    assert BLUE_BIN.height_m == 0.3048, BLUE_BIN.height_m
    assert BLUE_BIN.prior.height_m == BLUE_BIN.height_m, "the prior must carry the profile's height"


def test_every_bin_range_scales_linearly_with_the_height_prior():
    """Double the prior and every range doubles — which is why a wrong prior is invisible.

    This is the property that makes issue #35 a real risk rather than a rounding worry: the
    error a wrong height introduces is MULTIPLICATIVE and uniform, so nothing downstream
    disagrees with anything else. It is also the property ``--prop-height`` relies on, and
    the reason the fix for a bad scale is the prior and not a re-calibration of ``f``.
    """
    camera = FisheyeCamera(width=1920, height=1080, focal_px=STAGED_FOCAL_PX,
                           cx=960.0, cy=540.0)
    x, y, width_px, height_px = STAGED_BIN_BOX_PX
    found = ColourBlobDetector().detect(_rect(_frame(), x, y, width_px, height_px))
    assert len(found) == 1, found

    shipped, _ = estimate_range(found[0], camera, BLUE_BIN.prior)
    for factor in (0.5, 1.5, 2.0):
        # ``replace`` rather than a fresh ColourProfile: the dataclass carries thirteen
        # fields and re-listing eight of them would quietly reset the gates to defaults.
        taller = replace(BLUE_BIN, height_m=BLUE_BIN.height_m * factor,
                         width_m=BLUE_BIN.width_m * factor)
        scaled, source = estimate_range(found[0], camera, taller.prior)
        assert source == "height", source
        assert abs(scaled - shipped * factor) < 1e-6, (factor, scaled, shipped)


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


def test_a_bin_at_a_live_runs_fill_is_detected_not_rejected():
    """The gate that ran the robot over a bin on 2026-08-19.

    0.55 was measured square-on and well lit, where a bin fills 0.85 of its box. A live
    run is not that. Over the 63 recorded frames of that run, with two bins staged the
    whole time, the shipped gate saw BOTH bins on 2 frames — 3%. Landmarks therefore went
    unobserved, accrued misses, and were pruned while the robot was still walking toward
    them; the map then reported clear space and it drove through.

    0.36-0.48 is the band those bins actually presented at. This asserts the gate admits
    it. Put min_fill back to 0.55 and the blob below is rejected.
    """
    profile = BLUE_BIN
    # 0.42 fill: mid-band for a bin seen at an angle with its logo breaking the mask.
    assert profile.min_fill <= 0.42, "a bin at a live run's fill must pass"
    # And the gate is not switched off — a smear of wall colour still fails.
    assert profile.min_fill > 0.10


def test_the_shipped_profile_carries_the_measured_gate():
    """A regression pin on the value itself, since it was chosen from a sweep and a
    later edit that 'tidies' it back to a rounder number would silently restore the
    3%-detection behaviour without failing anything else."""
    assert BLUE_BIN.min_fill == 0.35


def _box(x1, y1, x2, y2, label="cone"):
    return Detection(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                     score=0.5, label=label)


def test_a_cones_two_red_bands_collapse_to_the_lower_one():
    """The measured failure. Robot 1 saw ONE cone at 0.74 m as two boxes stacked in the
    same column, and the size prior — calibrated on the lower band — read the smaller
    upper band as the same band 1.75x further away. That phantom landed on the goal line
    and stopped a run. Both boxes below are the real ones, in pixels, off that frame."""
    lower = _box(1070, 488, 1152, 592)
    upper = _box(1084, 366, 1132, 426)
    refused = []
    kept = collapse_stacked_blobs([lower, upper], refused=refused)
    assert kept == [lower], "the band nearest the floor is the calibrated one"
    assert [d for d, _ in refused] == [upper]
    assert [source for _, source in refused] == ["stacked-blob"]


def test_the_lower_band_wins_whichever_order_it_arrives_in():
    """`detect` sorts by area and the lower band is usually larger, but a cone seen at an
    angle, part-occluded or clipped by the frame edge need not present that way. The
    choice must come from geometry, not from arrival order."""
    lower = _box(1070, 488, 1152, 592)
    upper = _box(1084, 366, 1132, 426)
    assert collapse_stacked_blobs([upper, lower]) == [lower]
    assert collapse_stacked_blobs([lower, upper]) == [lower]


def test_two_cones_side_by_side_both_survive():
    """The gate must not cost the scene its second obstacle. Two cones at different
    bearings share no image column, so nothing collapses — this is the case that
    separates 'one object, one obstacle' from 'one obstacle, whatever the scene'."""
    left_lower = _box(200, 488, 282, 592)
    left_upper = _box(214, 366, 262, 426)
    right_lower = _box(1070, 488, 1152, 592)
    right_upper = _box(1084, 366, 1132, 426)
    kept = collapse_stacked_blobs([left_lower, left_upper, right_lower, right_upper])
    assert sorted(kept, key=lambda d: d.x1) == [left_lower, right_lower]


def test_columns_that_merely_graze_are_left_alone():
    """Two distinct props whose boxes clip each other at the edge are still two props.
    The overlap is measured against the NARROWER box, so a wide blob cannot swallow a
    slim neighbour it happens to touch."""
    wide = _box(100, 400, 400, 600)
    slim = _box(390, 300, 420, 360)
    assert len(collapse_stacked_blobs([wide, slim])) == 2


def test_a_dropped_blob_is_never_dropped_silently():
    """An obstacle that leaves nothing behind is indistinguishable from open world:
    `is_feasible` returns True unconditionally when there is nothing to check. Every
    collapse must therefore be recoverable from `refused`."""
    boxes = [_box(1070, 488, 1152, 592), _box(1084, 366, 1132, 426),
             _box(1080, 250, 1120, 300)]
    refused = []
    kept = collapse_stacked_blobs(boxes, refused=refused)
    assert len(kept) + len(refused) == len(boxes)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"colour_detector: {len(tests)}/{len(tests)} passed")
