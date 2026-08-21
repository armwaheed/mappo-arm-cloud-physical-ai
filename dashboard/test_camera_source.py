#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the camera source: the ceilings, the rate cap, and who is watching.

The interesting behaviour here is all refusal and lifecycle. There is no test that a frame
"looks right" — that is what the viewport is for, and a synthetic frame that renders is not
evidence about a robot's camera anyway.

``python3 test_camera_source.py``.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_source import (
    DEFAULT_FPS,
    MAX_FPS,
    MAX_FRAME_BYTES,
    CameraUnavailable,
    Viewers,
    clamp_fps,
    encode,
    open_source,
)


def test_the_frame_rate_is_capped_at_the_sensor_rate():
    """Above ~15 Hz the Go2 returns duplicate frames and you pay full bandwidth for them."""
    assert clamp_fps(60) == MAX_FPS
    assert clamp_fps(12) == 12.0
    assert clamp_fps(6) == 6.0


def test_a_nonsense_rate_falls_back_rather_than_dividing_by_zero():
    """The fps comes off a number input on a web page, so it arrives as anything."""
    for bad in (0, -4, None, "fast", float("nan")):
        result = clamp_fps(bad)
        assert result == DEFAULT_FPS or 0 < result <= MAX_FPS, (bad, result)


def test_an_implausibly_large_frame_is_refused():
    """A camera misconfigured to 4K would otherwise put megabytes on the mesh at 6 Hz."""
    try:
        encode(b"x" * (MAX_FRAME_BYTES + 1))
    except CameraUnavailable as exc:
        assert "ceiling" in str(exc)
        return
    raise AssertionError("an oversized frame was encoded")


def test_an_empty_frame_encodes_to_nothing_rather_than_to_an_empty_image():
    """A camera that has not produced a frame yet must not publish a zero-byte JPEG, which a
    browser renders as a broken image and an operator reads as a broken camera."""
    assert encode(None) is None
    assert encode(b"") is None


def test_interest_expires_so_a_closed_tab_stops_the_stream():
    """Nobody streams to an empty room. Without expiry a closed browser tab leaves a robot
    emitting for the rest of the session."""
    now = [100.0]
    viewers = Viewers(timeout_s=10.0, clock=lambda: now[0])

    assert viewers.watching() is False, "a fresh viewer set claims someone is watching"
    viewers.note_interest()
    assert viewers.watching() is True

    now[0] += 5.0
    assert viewers.watching() is True, "interest expired inside the timeout"
    now[0] += 6.0
    assert viewers.watching() is False, "interest outlived the timeout"


def test_stopping_explicitly_beats_waiting_for_the_timeout():
    now = [0.0]
    viewers = Viewers(timeout_s=10.0, clock=lambda: now[0])
    viewers.note_interest()
    viewers.clear()
    assert viewers.watching() is False


def test_an_unknown_platform_is_refused_by_name():
    try:
        open_source("submarine")
    except CameraUnavailable as exc:
        assert "submarine" in str(exc)
        return
    raise AssertionError("a camera was opened for an unknown platform")


def test_the_sim_source_produces_a_real_jpeg_that_changes():
    """The synthetic frame has to actually change, or a frozen stream is indistinguishable
    from a working one — which is the failure this viewport exists to make visible."""
    try:
        source = open_source("sim")
    except CameraUnavailable as exc:
        print(f"  skip  Pillow not installed ({exc})")
        return
    first, second = source.read(), source.read()
    source.close()
    assert first[:2] == b"\xff\xd8", "not a JPEG"
    assert first != second, "the synthetic feed is a still image"
    assert len(first) < MAX_FRAME_BYTES


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"camera_source: {len(tests)}/{len(tests)} passed")
