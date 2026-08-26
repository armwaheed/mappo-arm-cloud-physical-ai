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

from deep_robotics.lite3.visual_nav import camera as camera_module
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


class _StalledCapture(_Capture):
    """A source that serves ``served_frames`` frames and then blocks inside ``read()``.

    This is a dead RTSP stream with no FFmpeg ``rw_timeout``: the reader is parked in a
    call that will not return, so ``stop()``'s join is guaranteed to time out. Parking
    the reader is also what makes "did this thread publish?" a decidable question — a
    parked reader publishes nothing, so ``latest()`` stops moving. Each instance uses a
    distinct frame shape so a published frame names the reader that published it.
    """

    def __init__(self, served_frames=0, shape=(4, 5, 3)):
        super().__init__()
        self.unblock = threading.Event()
        self.shape = shape
        self._remaining = served_frames

    def read(self):
        if self._remaining <= 0:
            self.unblock.wait(10.0)
        else:
            self._remaining -= 1
        self.counter += 1
        return True, np.full(self.shape, self.counter % 255, dtype=np.uint8)

    def release(self):
        self.released = True
        self.unblock.set()


class _ShortStopTimeout:
    """Shrink the reader join so the stall tests cost 50 ms rather than 2 s each."""

    def __init__(self, seconds=0.05):
        self._seconds = seconds
        self._saved = camera_module.STOP_JOIN_TIMEOUT_S

    def __enter__(self):
        camera_module.STOP_JOIN_TIMEOUT_S = self._seconds
        return self

    def __exit__(self, *_exc_info):
        camera_module.STOP_JOIN_TIMEOUT_S = self._saved


class _ConcurrentReleaseCapture(_Capture):
    """Capture that records whether release overlaps a native-style blocking read."""

    def __init__(self):
        super().__init__()
        self.reading = threading.Event()
        self.concurrent_release = False
        self.reader_thread = None
        self.release_thread = None

    def read(self):
        self.reader_thread = threading.get_ident()
        self.reading.set()
        time.sleep(0.05)
        self.reading.clear()
        return super().read()

    def release(self):
        self.release_thread = threading.get_ident()
        self.concurrent_release = self.reading.is_set()
        super().release()


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


def test_stop_never_releases_while_the_reader_is_inside_read():
    capture = _ConcurrentReleaseCapture()
    camera = Lite3Camera(0, capture_factory=lambda _source: capture)
    camera.start()
    assert capture.reading.wait(0.5)
    camera.stop()
    assert not capture.concurrent_release
    assert capture.release_thread == capture.reader_thread


def test_start_reports_the_missing_frame_rather_than_its_own_cleanup_timeout():
    """``start()`` calls ``stop()`` before raising, so ``stop()`` must not raise first.

    A stalled RTSP source is exactly the case ``start()``'s message exists to report,
    and it is exactly the case that makes the join time out.
    """
    stalled = _StalledCapture()
    camera = Lite3Camera("rtsp://camera/live", capture_factory=lambda _source: stalled)
    try:
        with _ShortStopTimeout():
            try:
                camera.start(wait_s=0.05)
            except RuntimeError as error:
                assert "no frame from Lite3 rtsp camera source" in str(error)
            else:
                raise AssertionError("a source that never produced a frame was accepted")
            assert camera.stop_timed_out
    finally:
        stalled.unblock.set()
        camera.stop()


def test_a_stalled_reader_leaves_a_reusable_camera_and_cannot_publish_into_its_replacement():
    """The wedged object used to answer "camera is already running" on every retry.

    The replacement source serves one frame and then parks its reader, so ``latest()``
    is fixed at that frame. Anything that moves it afterwards came from the abandoned
    reader, which by then owns nothing this camera should still be listening to.
    """
    stalled = _StalledCapture()
    replacement = _StalledCapture(served_frames=1, shape=(2, 3, 3))
    captures = [stalled, replacement]
    camera = Lite3Camera("rtsp://camera/live",
                         capture_factory=lambda _source: captures.pop(0))
    try:
        with _ShortStopTimeout():
            try:
                camera.start(wait_s=0.05)
            except RuntimeError:
                pass
            else:
                raise AssertionError("a source that never produced a frame was accepted")
            assert camera.stop_timed_out
            assert camera._thread is None and camera._capture is None

        camera.start(wait_s=2.0)
        assert not camera.stop_timed_out
        assert camera.latest().image.shape == (2, 3, 3)

        # The abandoned reader now leaves read(). It must release its own capture and
        # publish nothing: its frames are (4, 5, 3), the live one's are (2, 3, 3).
        stalled.unblock.set()
        deadline = time.monotonic() + 2.0
        while not stalled.released and time.monotonic() < deadline:
            time.sleep(0.005)
        assert stalled.released
        assert camera.latest().image.shape == (2, 3, 3)
    finally:
        stalled.unblock.set()
        replacement.unblock.set()
        camera.stop()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_camera: {len(tests)}/{len(tests)} passed")
