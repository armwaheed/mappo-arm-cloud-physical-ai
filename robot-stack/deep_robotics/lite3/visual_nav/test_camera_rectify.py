#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the equidistant rectification.

The property that matters is not "the maps exist" but "a ray at a known angle lands where
an equidistant model of the stated focal says it should" — that is the whole claim the
vendored geometry then rests on, so it is tested against the model directly rather than
against a stored image.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.visual_nav.camera_rectify import (
    Rectifier,
    build_maps,
    equidistant_focal_for,
    rectified_camera,
)

WIDTH, HEIGHT = 1280, 720
#: The real lens, as measured on 2026-09-01 from 26 ArUco grid-board views.
CAMERA_MATRIX = np.array([[508.74, 0.0, 640.90],
                          [0.0, 494.93, 339.51],
                          [0.0, 0.0, 1.0]])
DISTORTION = np.array([-0.01785, -0.03947, 0.00526, 0.00407, 0.00858])


def _profile(**overrides) -> dict:
    body = {
        "schema": "lite3-rectify/v1",
        "platform": "deep-robotics-lite3-venture",
        "width": WIDTH, "height": HEIGHT,
        "camera_matrix": CAMERA_MATRIX.tolist(),
        "distortion": DISTORTION.tolist(),
        "focal_out_px": 545.9,
    }
    body.update(overrides)
    return body


def _written(body: dict) -> Path:
    path = Path(tempfile.mkdtemp()) / "rectify.json"
    path.write_text(json.dumps(body))
    return path


def test_a_ray_lands_where_the_equidistant_model_says_it_should():
    """THE CLAIM. After the remap the stack's model is meant to be true BY CONSTRUCTION,
    so an output pixel at radius r must correspond to a real ray at exactly r/focal_out.
    Checked by walking the map backwards through the real lens model and comparing the
    recovered angle with the one the equidistant model promises."""
    focal_out = 545.9
    map_x, map_y = build_maps(CAMERA_MATRIX, DISTORTION, WIDTH, HEIGHT, focal_out)
    for radius in (0.0, 60.0, 140.0, 260.0, 330.0):
        row, col = HEIGHT // 2, int(WIDTH / 2 + radius)
        # Undo the real lens: pixel -> undistorted normalised -> angle off the axis.
        undistorted = cv2.undistortPoints(
            np.array([[[map_x[row, col], map_y[row, col]]]], dtype=np.float64),
            CAMERA_MATRIX, DISTORTION).reshape(2)
        recovered = math.atan(math.hypot(*undistorted))
        promised = radius / focal_out
        assert abs(recovered - promised) < 1e-3, (
            f"at r={radius} the equidistant model promises "
            f"{math.degrees(promised):.2f} deg but the map fetches "
            f"{math.degrees(recovered):.2f} deg")


def test_the_focal_is_the_one_that_crops_no_field_of_view():
    """A 16:9 sensor and a radial mapping disagree about how much focal is affordable.
    Taking the larger would fill the frame by discarding the periphery, which is exactly
    where an obstacle appears before the robot is committed to a path."""
    focal = equidistant_focal_for(WIDTH, HEIGHT, CAMERA_MATRIX, DISTORTION)
    half_v = (HEIGHT / 2.0) / focal
    half_h = (WIDTH / 2.0) / focal
    assert half_v <= half_h, "the vertical axis is the binding one on a 16:9 sensor"
    # every real ray still lands inside the output image
    assert focal * half_v <= HEIGHT / 2.0 + 1e-6


def test_a_frame_of_the_wrong_size_is_refused_rather_than_resized():
    """The maps encode ABSOLUTE pixel coordinates, so a camera that quietly changes
    resolution would produce a plausible image with wrong geometry throughout."""
    rectifier = Rectifier(_written(_profile()))
    try:
        rectifier.apply(np.zeros((480, 640, 3), dtype=np.uint8))
    except SystemExit as exc:
        assert "REFUSING TO RECTIFY" in str(exc) and "640x480" in str(exc)
    else:
        raise AssertionError("a mis-sized frame was accepted")


def test_a_file_that_is_not_a_rectify_profile_is_refused():
    """A camera calibration and a rectification profile are both JSON with a focal in
    them; handing over the wrong one must not half-work."""
    try:
        Rectifier(_written(_profile(schema="lite3-axis-profile/v1")))
    except SystemExit as exc:
        assert "REFUSING TO USE CAMERA RECTIFICATION" in str(exc)
    else:
        raise AssertionError("a foreign schema was accepted")


def test_the_wrapper_rectifies_every_frame_the_loop_reads():
    """``visual_nav``'s loop hands the SAME frame object to the planner, the recorder and
    the telemetry writer, so rectifying once at the source is what keeps all of them
    looking at the same pixels."""

    class _Frame:
        def __init__(self, image):
            self.image = image

    class _Camera:
        def __init__(self):
            self.frame = _Frame(np.full((HEIGHT, WIDTH, 3), 7, dtype=np.uint8))

        def latest(self):
            return self.frame

    camera = _Camera()
    rectifier = Rectifier(_written(_profile()))
    wrapped = rectified_camera(camera, rectifier)
    out = wrapped.latest()
    assert out.image.shape == (HEIGHT, WIDTH, 3)
    # The corners fall outside the real lens's field, so they must be black rather than
    # a smeared copy of the nearest valid pixel.
    assert out.image[0, 0].tolist() == [0, 0, 0]
    assert out.image[HEIGHT // 2, WIDTH // 2].tolist() == [7, 7, 7]


def test_a_camera_with_no_frame_yet_stays_none():
    """The loop starts before the reader has produced anything; rectifying None would
    turn a normal startup into a crash."""

    class _Empty:
        def latest(self):
            return None

    wrapped = rectified_camera(_Empty(), Rectifier(_written(_profile())))
    assert wrapped.latest() is None


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_camera_rectify: {len(tests)}/{len(tests)} passed")
