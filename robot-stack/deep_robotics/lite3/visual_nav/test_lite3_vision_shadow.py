#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 camera-only shadow recorder."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[2] / "unitree" / "go2" / "visual_nav"))

from deep_robotics.lite3.visual_nav import camera as camera_module
from deep_robotics.lite3.visual_nav.camera import Frame, Lite3Camera
from deep_robotics.lite3.visual_nav.lite3_vision_shadow import (
    SCHEMA,
    parse_voc_classes,
    run,
)
from person_detector import Detection


class _Camera:
    def __init__(self, frames: list[Frame]):
        self._frames = frames
        self._index = 0
        self.started = False
        self.stopped = False
        self.error_count = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def wait_for_new(self, after_sequence, _timeout):
        while self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            if frame.seq > after_sequence:
                return frame
        return None


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _DelayedStartCamera(_Camera):
    def __init__(self, frames, clock):
        super().__init__(frames)
        self._clock = clock

    def start(self):
        super().start()
        self._clock.now += 5.0


class _Detector:
    def __init__(self, *_args, **_kwargs):
        self.calls = 0

    def detect(self, _image):
        self.calls += 1
        if self.calls == 1:
            return [Detection(1.0, 2.0, 3.0, 4.0, 0.9, "chair")]
        return []


class _StalledCapture:
    """A source that serves one frame and then blocks inside ``read()``.

    This is the case the camera's own cleanup cannot rush: a dead RTSP stream with no
    FFmpeg ``rw_timeout``. It is also the case in which the run is most likely to be
    failing for a reason worth reporting.
    """

    def __init__(self):
        self.unblock = threading.Event()
        self.released = False
        self._served = False

    def isOpened(self):
        return True

    def set(self, _prop, _value):
        return True

    def read(self):
        if self._served:
            self.unblock.wait(10.0)
        self._served = True
        return True, np.zeros((2, 3, 3), dtype=np.uint8)

    def release(self):
        self.released = True
        self.unblock.set()


class _FailingDetector:
    def __init__(self, *_args, **_kwargs):
        pass

    def detect(self, _image):
        raise RuntimeError("detector failed")


def _frames() -> list[Frame]:
    now = time.monotonic()
    return [
        Frame(np.zeros((2, 3, 3), dtype=np.uint8), now, 1),
        Frame(np.zeros((2, 3, 3), dtype=np.uint8), now, 2),
    ]


def test_parse_voc_classes_refuses_empty_unknown_and_deduplicates():
    assert parse_voc_classes("person, chair,person") == ("person", "chair")
    for value in ("", "person,unicorn"):
        try:
            parse_voc_classes(value)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"accepted invalid classes: {value!r}")


def test_shadow_run_writes_credential_free_bounded_detection_evidence():
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "lite3-shadow-test.jsonl"
        camera = _Camera(_frames())
        summary = run(
            camera_source="rtsp://user:secret@example.test/live",
            camera_gstreamer=False,
            model_dir=Path("models"),
            classes=("person", "chair"),
            confidence=0.4,
            input_size=300,
            seconds=1.0,
            max_frames=2,
            output=output,
            camera_factory=lambda *_args, **_kwargs: camera,
            detector_factory=_Detector,
        )
        evidence = output.read_text()
        records = [json.loads(line) for line in evidence.splitlines()]
        assert camera.started and camera.stopped
        assert summary["frames"] == 2
        assert summary["detections"] == {"chair": 1}
        assert records[0]["schema"] == SCHEMA
        assert records[0]["camera_kind"] == "rtsp"
        assert records[1]["detections"][0]["label"] == "chair"
        assert records[-1]["kind"] == "outcome"
        assert "secret" not in evidence


def test_shadow_run_refuses_to_overwrite_before_starting_camera():
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "lite3-shadow-existing.jsonl"
        output.write_text("{}\n")
        camera = _Camera(_frames())
        try:
            run(
                camera_source=0,
                camera_gstreamer=False,
                model_dir=Path("models"),
                classes=("person",),
                confidence=0.4,
                input_size=300,
                seconds=1.0,
                max_frames=1,
                output=output,
                camera_factory=lambda *_args, **_kwargs: camera,
                detector_factory=_Detector,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("overwrote existing evidence")
        assert not camera.started and not camera.stopped


def test_recording_window_starts_after_camera_startup():
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "lite3-shadow-delayed-start.jsonl"
        clock = _Clock()
        camera = _DelayedStartCamera(_frames(), clock)
        summary = run(
            camera_source=0,
            camera_gstreamer=False,
            model_dir=Path("models"),
            classes=("person",),
            confidence=0.4,
            input_size=300,
            seconds=1.0,
            max_frames=1,
            output=output,
            camera_factory=lambda *_args, **_kwargs: camera,
            detector_factory=_Detector,
            clock=clock,
        )
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert summary["frames"] == 1
        assert records[0]["started_monotonic_s"] == 5.0
        assert camera.stopped


def test_a_stalled_camera_stop_does_not_replace_the_exception_that_ended_the_run():
    """``run`` stops the camera in a ``finally``; a stalled reader must not become the story.

    Mirrors ``unitree/go2/visual_nav/test_lifecycle.py``'s
    ``test_a_cleanup_failure_does_not_replace_the_exception_that_ended_the_run``. This
    is the real camera, not a fake one, because the invariant being pinned belongs to
    ``Lite3Camera.stop`` — a stalled reader is exactly when the run has something to say.
    """
    capture = _StalledCapture()
    camera = Lite3Camera(0, capture_factory=lambda _source: capture)
    saved_timeout = camera_module.STOP_JOIN_TIMEOUT_S
    camera_module.STOP_JOIN_TIMEOUT_S = 0.05
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "lite3-shadow-stalled.jsonl"
            try:
                run(
                    camera_source=0,
                    camera_gstreamer=False,
                    model_dir=Path("models"),
                    classes=("person",),
                    confidence=0.4,
                    input_size=300,
                    seconds=1.0,
                    max_frames=1,
                    output=output,
                    camera_factory=lambda *_args, **_kwargs: camera,
                    detector_factory=_FailingDetector,
                )
            except RuntimeError as error:
                assert "detector failed" in str(error)
            else:
                raise AssertionError("cleanup replaced or swallowed the run failure")
        # The stall was real: stop() gave up waiting for the reader and recorded it.
        assert camera.stop_timed_out
    finally:
        camera_module.STOP_JOIN_TIMEOUT_S = saved_timeout
        capture.unblock.set()
        camera.stop()


def test_shadow_recorder_does_not_import_or_call_a_motion_transport():
    source = (_HERE / "lite3_vision_shadow.py").read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imported = imported_modules.union(imported_names)
    assert not any("locomotion" in name or "udp" in name or name == "socket" for name in imported)
    assert "--live" not in source
    assert "set_velocity" not in source


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_vision_shadow: {len(tests)}/{len(tests)} passed")
