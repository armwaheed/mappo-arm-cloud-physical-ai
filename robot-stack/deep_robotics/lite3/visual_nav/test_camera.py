#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 OpenCV camera adapter."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.visual_nav.camera import (
    Lite3Camera,
    camera_source_kind,
    parse_camera_source,
)


class _Capture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.counter = 0
        self.allow = threading.Event()
        self.allow.set()

    def isOpened(self):
        return self.opened

    def set(self, _prop, _value):
        return True

    def read(self):
        self.allow.wait(0.1)
        if self.released:
            return False, None
        self.counter += 1
        return True, np.full((2, 3, 3), self.counter % 255, dtype=np.uint8)

    def release(self):
        self.released = True
        self.allow.set()


def test_numeric_sources_are_v4l2_indices_and_uris_remain_strings():
    assert parse_camera_source("0") == 0
    assert parse_camera_source(" 12 ") == 12
    assert parse_camera_source("rtsp://camera/live") == "rtsp://camera/live"


def test_start_refuses_a_source_that_did_not_open():
    capture = _Capture(opened=False)
    camera = Lite3Camera(0, capture_factory=lambda _source: capture)
    try:
        camera.start(wait_s=0.01)
    except RuntimeError as exc:
        assert "cannot open" in str(exc)
        assert capture.released
        return
    raise AssertionError("an unopened camera source was accepted")


def test_camera_errors_and_telemetry_labels_cannot_leak_source_credentials():
    source = "rtsp://operator:secret@camera.example/live?token=private"
    capture = _Capture(opened=False)
    camera = Lite3Camera(source, capture_factory=lambda _source: capture)
    try:
        camera.start(wait_s=0.01)
    except RuntimeError as exc:
        assert str(exc) == "cannot open Lite3 rtsp camera source"
        assert "secret" not in str(exc) and "private" not in str(exc)
    else:
        raise AssertionError("an unopened credentialed camera source was accepted")
    assert camera_source_kind(source) == "rtsp"
    assert camera_source_kind("credentials-in-a-pipeline", gstreamer=True) == "gstreamer"


def test_frames_are_bgr_sequenced_and_pose_stamped():
    capture = _Capture()
    camera = Lite3Camera("test", stamp_fn=lambda: (1.0, 2.0, 0.3),
                         capture_factory=lambda _source: capture)
    camera.start()
    first = camera.latest()
    newer = camera.wait_for_new(first.seq, timeout=0.5)
    camera.stop()
    assert first.image.shape == (2, 3, 3)
    assert newer is not None and newer.seq > first.seq
    assert newer.stamp == (1.0, 2.0, 0.3)
    assert newer.capture_time <= time.monotonic()


def test_a_stamp_failure_does_not_take_down_the_camera():
    capture = _Capture()

    def bad_stamp():
        raise RuntimeError("odom unavailable")

    camera = Lite3Camera(0, stamp_fn=bad_stamp, capture_factory=lambda _source: capture)
    camera.start()
    frame = camera.latest()
    camera.stop()
    assert frame is not None and frame.stamp is None


def test_stop_releases_the_capture_and_is_idempotent():
    capture = _Capture()
    camera = Lite3Camera(0, capture_factory=lambda _source: capture)
    camera.start()
    camera.stop()
    camera.stop()
    assert capture.released


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_camera: {len(tests)}/{len(tests)} passed")
