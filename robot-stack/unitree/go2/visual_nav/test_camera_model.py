#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the fisheye camera model — projection, ranging, calibration search.

No robot, no images: the model is pure geometry, so every property that the navigator
depends on can be pinned here. The load-bearing one is that a synthetic object of known
size at a known distance ranges back to that distance, since the whole pipeline's
metric scale rests on it.

Run: ``python3 test_camera_model.py``
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_model import GO2_CAMERA_HEIGHT_M, FisheyeCamera, solve_focal_px

WIDTH, HEIGHT = 1920, 1080
#: Deliberately built WITHOUT a lens height, so every test below that does not name one is
#: also a statement that the model does not supply a platform's number on its own.
MODEL = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0)

#: The same optics with the Go2's measured lens height named at the call site. This is the
#: only shape in which the floor is locatable.
ON_A_GO2 = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0, height_m=GO2_CAMERA_HEIGHT_M)


def test_hfov_round_trips():
    for hfov in (90.0, 120.0, 150.0):
        model = FisheyeCamera.from_hfov(WIDTH, HEIGHT, hfov)
        assert abs(model.hfov_deg - hfov) < 1e-9, model.hfov_deg


def test_centre_pixel_is_straight_ahead():
    azimuth, elevation = MODEL.bearing_elevation(WIDTH / 2.0, HEIGHT / 2.0)
    assert abs(float(azimuth)) < 1e-9
    assert abs(float(elevation)) < 1e-9
    direction = MODEL.unit_vector(WIDTH / 2.0, HEIGHT / 2.0)
    assert np.allclose(direction, [1.0, 0.0, 0.0])


def test_left_of_frame_is_positive_bearing():
    # +y is the robot's LEFT, and a smaller u is further left in the image.
    left, _ = MODEL.bearing_elevation(WIDTH / 4.0, HEIGHT / 2.0)
    right, _ = MODEL.bearing_elevation(3.0 * WIDTH / 4.0, HEIGHT / 2.0)
    assert float(left) > 0.0, left
    assert float(right) < 0.0, right
    assert abs(float(left) + float(right)) < 1e-9, "should be symmetric about centre"


def test_frame_edge_is_half_the_hfov():
    azimuth, _ = MODEL.bearing_elevation(0.0, HEIGHT / 2.0)
    assert abs(math.degrees(float(azimuth)) - 60.0) < 1e-6, math.degrees(float(azimuth))


def test_above_centre_is_positive_elevation():
    _, elevation = MODEL.bearing_elevation(WIDTH / 2.0, HEIGHT / 4.0)
    assert float(elevation) > 0.0


def test_unit_vectors_are_unit_and_vectorised():
    us = np.array([0.0, WIDTH / 2.0, WIDTH - 1.0])
    vs = np.array([0.0, HEIGHT / 2.0, HEIGHT - 1.0])
    directions = MODEL.unit_vector(us, vs)
    assert directions.shape == (3, 3)
    assert np.allclose(np.linalg.norm(directions, axis=-1), 1.0)


def test_angle_between_matches_bearing_difference_on_the_centre_line():
    p1 = (600.0, HEIGHT / 2.0)
    p2 = (1300.0, HEIGHT / 2.0)
    b1, _ = MODEL.bearing_elevation(*p1)
    b2, _ = MODEL.bearing_elevation(*p2)
    assert abs(float(MODEL.angle_between(p1, p2)) - abs(float(b1) - float(b2))) < 1e-9


def _project_vertical_segment(model: FisheyeCamera, length_m: float, distance_m: float):
    """Pixels of a vertical segment of ``length_m`` centred on the optical axis."""
    half_angle = math.atan2(length_m / 2.0, distance_m)
    offset = model.focal_px * half_angle          # equidistant: radius = f * theta
    return ((model.cx, model.cy - offset), (model.cx, model.cy + offset))


def test_range_from_span_recovers_a_known_distance():
    for distance in (1.5, 3.0, 6.0):
        top, bottom = _project_vertical_segment(MODEL, 1.70, distance)
        estimate = MODEL.range_from_span(top, bottom, 1.70)
        assert abs(estimate - distance) < 1e-6, (distance, estimate)


def test_range_from_span_is_degenerate_safe():
    """A zero-length span has no range — ANYWHERE in the frame, not just on axis.

    The principal point alone is not a test: ``unit_vector`` special-cases radius 0
    and returns the optical axis exactly, so ``arccos`` there is exactly 0 and any
    threshold at all would pass. Off axis, two identical rays measure ~2e-8 rad
    instead, which a float-epsilon gate waves through as a finite 9,000 km.
    """
    for point in ((MODEL.cx, MODEL.cy), (500.0, 500.0), (1600.0, 250.0), (80.0, 1000.0)):
        assert math.isinf(MODEL.range_from_span(point, point, 1.7)), point


def test_range_from_span_rejects_a_sub_pixel_span():
    """The floor is the sensor's resolution, not a float epsilon."""
    assert MODEL.pixel_angle_rad > 1e-6, "one pixel should be a real angle"
    centre = np.array([3.0, 0.0, 0.0])
    # A span the model resolves at half a pixel: unmeasurable, whatever it claims.
    half_pixel_m = 3.0 * MODEL.pixel_angle_rad / 2.0
    top = MODEL.project(centre + np.array([0.0, 0.0, half_pixel_m / 2.0]))
    bottom = MODEL.project(centre + np.array([0.0, 0.0, -half_pixel_m / 2.0]))
    assert math.isinf(MODEL.range_from_span(top, bottom, half_pixel_m))


def test_projection_round_trips():
    for u, v in ((MODEL.cx, MODEL.cy), (300.0, 200.0), (1700.0, 950.0), (0.0, 0.0)):
        direction = MODEL.unit_vector(u, v)
        back_u, back_v = MODEL.project(direction)
        assert abs(back_u - u) < 1e-6 and abs(back_v - v) < 1e-6, (u, v, back_u, back_v)


def test_projection_round_trips_with_pitch():
    tilted = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0, pitch_rad=math.radians(12.0))
    for u, v in ((500.0, 300.0), (1400.0, 800.0)):
        back_u, back_v = tilted.project(tilted.unit_vector(u, v))
        assert abs(back_u - u) < 1e-6 and abs(back_v - v) < 1e-6


def test_range_holds_up_off_axis():
    """A naive pixels/focal ranger reads low at the frame edge; the model must not.

    Built by projecting a REAL 1.70 m vertical segment placed 45 deg off the nose,
    rather than by assuming the segment maps to +-f*theta in v — under a fisheye it
    does not, because the projection is radial about the optical axis.
    """
    distance, height = 3.0, 1.70
    bearing = math.radians(45.0)
    base = np.array([distance * math.cos(bearing), distance * math.sin(bearing), 0.0])
    top = MODEL.project(base + np.array([0.0, 0.0, height / 2.0]))
    bottom = MODEL.project(base + np.array([0.0, 0.0, -height / 2.0]))
    estimate = MODEL.range_from_span(top, bottom, height)
    assert abs(estimate - distance) / distance < 0.01, estimate


def test_naive_pixel_ranger_would_be_wrong_here():
    """Pins WHY angle_between exists rather than a pixel-height shortcut.

    ``height * focal / pixel_span`` is the obvious ranger and is exact on the optical
    axis. Off-axis the fisheye stretches the radial span, so it reads NEARER than the
    truth — measured here at ~10% low by 50 deg out. Wrong in the safe direction, but
    wrong, and it would make the same person's range jump as they cross the frame.
    """
    distance, height = 3.0, 1.70
    base_angle = math.radians(50.0)
    base = np.array([distance * math.cos(base_angle), distance * math.sin(base_angle), 0.0])
    top = MODEL.project(base + np.array([0.0, 0.0, height / 2.0]))
    bottom = MODEL.project(base + np.array([0.0, 0.0, -height / 2.0]))
    pixel_span = math.hypot(top[0] - bottom[0], top[1] - bottom[1])
    naive = height * MODEL.focal_px / pixel_span
    error = (naive - distance) / distance
    assert error < -0.05, f"expected the shortcut to read low; got {naive:.2f} m"
    # ...while the model's own ranger stays exact at the same point.
    assert abs(MODEL.range_from_span(top, bottom, height) - distance) < 1e-6


def test_scaled_preserves_bearings():
    small = MODEL.scaled(WIDTH // 2, HEIGHT // 2)
    full, _ = MODEL.bearing_elevation(480.0, 270.0)
    half, _ = small.bearing_elevation(240.0, 135.0)
    assert abs(float(full) - float(half)) < 1e-9
    assert abs(small.hfov_deg - MODEL.hfov_deg) < 1e-9


def test_scaled_rejects_aspect_change():
    try:
        MODEL.scaled(WIDTH // 2, HEIGHT)
    except ValueError:
        return
    raise AssertionError("non-uniform rescale should raise")


def test_save_load_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.json")
        MODEL.save(path, method="unit-test")
        restored = FisheyeCamera.load(path)
        assert abs(restored.focal_px - MODEL.focal_px) < 1e-9
        assert restored.width == MODEL.width
        # Provenance is written for humans but must not break the loader.
        assert json.loads(Path(path).read_text())["method"] == "unit-test"


def test_ground_point_and_horizon():
    assert ON_A_GO2.ground_point(WIDTH / 2.0, HEIGHT / 2.0) is None, \
        "horizon never meets the floor"
    forward, lateral = ON_A_GO2.ground_point(WIDTH / 2.0, HEIGHT - 1.0)
    assert forward > 0.0 and abs(lateral) < 1e-6


# ── the lens height ─────────────────────────────────────────────────────────
def test_a_model_states_no_lens_height_until_one_is_given():
    """The regression pin. `height_m` defaulted to 0.32 — the GO2's measured standing
    lens height — on a model this repository also builds for a Lite3, whose lens height
    has never been measured (issue #13). Every Lite3 calibration that did not pass one
    inherited the Go2's, in a field that bounds a width-derived range in
    `person_detector.object_fit_range`. There is no default that is right for two robots,
    so there is no default."""
    assert FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0).height_m is None
    assert FisheyeCamera(width=WIDTH, height=HEIGHT, focal_px=1290.2,
                         cx=960.0, cy=540.0).height_m is None
    assert GO2_CAMERA_HEIGHT_M == 0.32, "the Go2's measurement, unchanged — just named"
    assert ON_A_GO2.height_m == GO2_CAMERA_HEIGHT_M


def test_a_model_with_no_lens_height_refuses_to_find_the_floor():
    """`ground_point` already returns None for "above the horizon", so it cannot use None
    for "I do not know where the floor is" as well. It refuses, and the refusal has to
    carry both halves an operator needs: what to pass, and whose 0.32 it is."""
    try:
        MODEL.ground_point(WIDTH / 2.0, HEIGHT - 1.0)
    except ValueError as refusal:
        message = str(refusal)
    else:
        raise AssertionError("a model with no lens height must not locate the floor")
    assert "no lens height" in message, message
    assert "height_m=" in message, "the refusal has to say what to pass"
    assert str(GO2_CAMERA_HEIGHT_M) in message and "Go2" in message, message
    assert "Lite3" in message, "and why the Go2's number is not a default"


def test_the_lens_height_survives_a_rescale_and_a_save_load_round_trip():
    """A calibration file must carry whatever height it was given and NEVER acquire one.

    The Lite3 commissioning wrapper reads this field back out of the file the shared
    fitter wrote (`stamp_lens_height`), so what `save` puts there is what a Lite3 is
    judged on. An unmeasured model has to round-trip as unmeasured."""
    assert ON_A_GO2.scaled(WIDTH // 2, HEIGHT // 2).height_m == GO2_CAMERA_HEIGHT_M
    assert MODEL.scaled(WIDTH // 2, HEIGHT // 2).height_m is None
    with tempfile.TemporaryDirectory() as directory:
        measured = os.path.join(directory, "measured.json")
        ON_A_GO2.save(measured, method="unit-test")
        assert json.loads(Path(measured).read_text())["height_m"] == GO2_CAMERA_HEIGHT_M
        assert FisheyeCamera.load(measured).height_m == GO2_CAMERA_HEIGHT_M

        unmeasured = os.path.join(directory, "unmeasured.json")
        MODEL.save(unmeasured, method="unit-test")
        assert json.loads(Path(unmeasured).read_text())["height_m"] is None, \
            "a fitted calibration must record that nobody measured the lens height"
        assert FisheyeCamera.load(unmeasured).height_m is None

        absent = os.path.join(directory, "absent.json")
        data = json.loads(Path(measured).read_text())
        del data["height_m"]
        Path(absent).write_text(json.dumps(data))
        assert FisheyeCamera.load(absent).height_m is None, \
            "an omitted key must not be filled in with a number from another robot"


def test_solve_focal_px_recovers_a_known_focal():
    truth = MODEL.focal_px
    recovered = solve_focal_px(lambda f: (f - truth) ** 2, WIDTH)
    assert abs(recovered - truth) / truth < 1e-4, recovered


def test_pitch_tilts_the_optical_axis_down():
    tilted = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0,
                                     pitch_rad=math.radians(10.0))
    _, elevation = tilted.bearing_elevation(WIDTH / 2.0, HEIGHT / 2.0)
    assert abs(math.degrees(float(elevation)) + 10.0) < 1e-6, math.degrees(float(elevation))


def test_pitch_barely_moves_azimuth():
    """Pins the claim that lets the navigator ignore trunk pitch (see the docstring).

    Tilting shifts azimuth through the cosine of the tilt, so the error is largest at
    the frame edge and still small. If this bound ever breaks, per-frame IMU pitch
    compensation becomes necessary.
    """
    tilted = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0,
                                     pitch_rad=math.radians(10.0))
    worst = 0.0
    for u in (0.0, WIDTH / 4.0, WIDTH / 2.0, 3.0 * WIDTH / 4.0, WIDTH - 1.0):
        azimuth, _ = tilted.bearing_elevation(u, HEIGHT / 2.0)
        reference, _ = MODEL.bearing_elevation(u, HEIGHT / 2.0)
        worst = max(worst, abs(math.degrees(float(azimuth) - float(reference))))
    assert worst < 0.5, f"10 deg of pitch moved a bearing by {worst:.3f} deg"


def test_rejects_nonsense_hfov():
    for bad in (0.0, -30.0, 400.0):
        try:
            FisheyeCamera.from_hfov(WIDTH, HEIGHT, bad)
        except ValueError:
            continue
        raise AssertionError(f"hfov {bad} should raise")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"camera_model: {len(tests)}/{len(tests)} passed")
