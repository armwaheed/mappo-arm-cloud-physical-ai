#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the calibration fits — synthetic sweeps, no robot.

Every one of these builds a camera with a KNOWN focal length, projects a target
through it, and checks the fit recovers the number it started from. That round trip
is the only thing standing between a confident printout and a wrong calibration that
silently rescales every range the navigator produces.

The quality split gets its own tests because it is what tells an operator whether to
take more samples or throw the model away, and those are opposite actions.

Run: ``python3 test_calibrate_camera.py``
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_camera import (
    MIN_SPIN_YAW_DEG,
    SPIN_DEADBAND_RAD_S,
    SPIN_RATE_RAD_S,
    build_parser,
    calibrate_from_spans,
    calibrate_from_spin,
    spin_fit_quality,
    yaw_span_deg,
)
from camera_model import FisheyeCamera

WIDTH, HEIGHT = 1920, 1080
TRUTH = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 85.0)


def _sweep(model: FisheyeCamera, start_bearing: float = 0.35, span: float = 1.2,
           count: int = 60, jitter_deg: float = 0.0, seed: int = 0) -> list:
    """A synthetic spin: the robot yaws, a fixed target's bearing follows it."""
    rng = np.random.RandomState(seed)
    yaws = np.linspace(0.0, span, count)
    bearings = start_bearing - yaws
    if jitter_deg:
        bearings = bearings + rng.normal(0.0, math.radians(jitter_deg), count)
    centres = [np.array(model.project([math.cos(b), math.sin(b), 0.0]))
               for b in bearings]
    return list(zip(centres, yaws))


def test_spin_recovers_a_known_focal_length():
    fitted = calibrate_from_spin(_sweep(TRUTH), WIDTH, HEIGHT)
    assert abs(fitted.focal_px - TRUTH.focal_px) / TRUTH.focal_px < 0.01, fitted.focal_px


def test_spin_recovers_focal_across_wildly_different_lenses():
    for hfov in (70.0, 100.0, 140.0):
        truth = FisheyeCamera.from_hfov(WIDTH, HEIGHT, hfov)
        fitted = calibrate_from_spin(_sweep(truth), WIDTH, HEIGHT)
        assert abs(fitted.hfov_deg - hfov) < 1.0, (hfov, fitted.hfov_deg)


def test_spin_survives_a_noisy_target():
    """A person is not a rigid fiducial; the fit must still land close."""
    fitted = calibrate_from_spin(_sweep(TRUTH, jitter_deg=3.0, seed=1), WIDTH, HEIGHT)
    assert abs(fitted.hfov_deg - 85.0) < 3.0, fitted.hfov_deg


def test_quality_split_attributes_noise_to_jitter():
    samples = _sweep(TRUTH, jitter_deg=2.0, seed=0)
    quality = spin_fit_quality(TRUTH, samples)
    assert abs(quality["jitter"] - 2.0) < 0.5, quality
    # Systematic should be near zero — but variance subtraction on a finite sample
    # leaves a floor, which is why the tool warns above 1 deg rather than above 0.
    assert quality["systematic"] < 1.0, quality


def test_quality_split_detects_real_systematic_error():
    """Fit a sweep with the WRONG model: the error must not be blamed on jitter."""
    wrong = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 120.0)
    quality = spin_fit_quality(wrong, _sweep(TRUTH))
    assert quality["systematic"] > 2.0, quality
    assert quality["systematic"] > quality["jitter"], quality


def test_quality_standard_error_shrinks_with_samples():
    few = spin_fit_quality(TRUTH, _sweep(TRUTH, jitter_deg=3.0, count=16, seed=2))
    many = spin_fit_quality(TRUTH, _sweep(TRUTH, jitter_deg=3.0, count=160, seed=2))
    assert many["standard_error"] < few["standard_error"] / 2.0, (few, many)


def test_spans_recover_a_known_focal_length():
    """Static span fit: a 1.70 m object at 3.0 m must fit back to the true focal."""
    distance, height_m = 3.0, 1.70
    base = np.array([distance, 0.0, 0.0])
    top = TRUTH.project(base + np.array([0.0, 0.0, height_m / 2.0]))
    bottom = TRUTH.project(base + np.array([0.0, 0.0, -height_m / 2.0]))
    fitted = calibrate_from_spans([(top, bottom)], WIDTH, HEIGHT, height_m, distance)
    assert abs(fitted.focal_px - TRUTH.focal_px) / TRUTH.focal_px < 0.01, fitted.focal_px


def test_spans_scale_with_a_wrong_distance():
    """The static method inherits its distance error — pins WHY spin is preferred."""
    distance, height_m = 3.0, 1.70
    base = np.array([distance, 0.0, 0.0])
    top = TRUTH.project(base + np.array([0.0, 0.0, height_m / 2.0]))
    bottom = TRUTH.project(base + np.array([0.0, 0.0, -height_m / 2.0]))
    # Claim the object is 10% further away than it is.
    fitted = calibrate_from_spans([(top, bottom)], WIDTH, HEIGHT, height_m,
                                  distance * 1.10)
    assert fitted.focal_px > TRUTH.focal_px * 1.05, fitted.focal_px


def test_yaw_span_measures_an_ordinary_sweep():
    samples = _sweep(TRUTH, span=math.radians(70.0), count=30)
    assert abs(yaw_span_deg(samples) - 70.0) < 0.1, yaw_span_deg(samples)


def test_yaw_span_survives_the_pi_branch_cut():
    """A 10 deg sweep straddling +-pi must not read as a full turn.

    This is the guard that stops an unconstrained fit being written out as a
    calibration, and raw ``max - min`` over wrapped yaw defeats it completely: every
    sample is real, the sweep is genuinely 10 deg, and the gate sees 359 deg.
    """
    sweep = np.linspace(math.pi - math.radians(5.0), math.pi + math.radians(5.0), 12)
    wrapped = np.arctan2(np.sin(sweep), np.cos(sweep))
    samples = [(np.array([0.0, 0.0]), float(yaw)) for yaw in wrapped]

    naive = math.degrees(max(y for _, y in samples) - min(y for _, y in samples))
    assert naive > 300.0, f"the wrapped sweep should look huge to max-min: {naive}"
    assert abs(yaw_span_deg(samples) - 10.0) < 0.5, yaw_span_deg(samples)
    assert yaw_span_deg(samples) < MIN_SPIN_YAW_DEG, "the guard must reject this sweep"


def test_spin_rate_default_clears_the_measured_deadband():
    """The default sweep rate must be one this robot can actually turn at.

    0.30 rad/s was the shipped default; it is also the value that produced 6.7 deg of
    yaw and a refused calibration on the first live attempt, because below ~0.4 rad/s
    this robot does not reliably initiate a turn at all.
    """
    assert SPIN_RATE_RAD_S > SPIN_DEADBAND_RAD_S
    assert build_parser().parse_args(["--spin"]).spin_rate > SPIN_DEADBAND_RAD_S
    # format_help() expands '%' in help strings, so this also catches an unescaped one.
    assert "SKILL.md" in build_parser().format_help()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"calibrate_camera: {len(tests)}/{len(tests)} passed")
