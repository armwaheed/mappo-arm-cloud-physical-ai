# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Latest-frame RGB capture for a Lite3 camera source supported by OpenCV.

The Venture camera endpoint is deployment-specific, so the source is explicit: a V4L2
index, an RTSP URI, or a GStreamer pipeline.  Frames use the same small API as the Go2
camera, which lets the detector and control loop remain unchanged.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    """One BGR frame, its local arrival time, sequence, and optional pose stamp."""

    image: np.ndarray
    capture_time: float
    seq: int
    stamp: Any = None

    @property
    def age(self) -> float:
        return time.monotonic() - self.capture_time


def parse_camera_source(value: str) -> int | str:
    """Turn a decimal device index into ``int``; leave URIs/pipelines untouched."""
    stripped = value.strip()
    try:
        return int(stripped)
    except ValueError:
        return stripped


def camera_source_kind(source: int | str, *, gstreamer: bool = False) -> str:
    """Return a credential-free description suitable for errors and telemetry."""
    if gstreamer:
        return "gstreamer"
    if isinstance(source, int):
        return "v4l2"
    scheme, separator, _remainder = source.partition("://")
    if separator and scheme:
        return scheme.lower()
    return "opencv"


class Lite3Camera:
    """Background reader for a V4L2, RTSP, file, or GStreamer RGB source."""

    def __init__(self, source: int | str, *, gstreamer: bool = False,
                 stamp_fn: Callable[[], Any] | None = None,
                 capture_factory: Callable = cv2.VideoCapture) -> None:
        self._source = source
        self._gstreamer = gstreamer
        self._source_kind = camera_source_kind(source, gstreamer=gstreamer)
        self._stamp_fn = stamp_fn
        self._capture_factory = capture_factory
        self._capture = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._new_frame = threading.Event()
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._seq = 0
        self._errors = 0

    def start(self, wait_s: float = 5.0) -> None:
        """Open the source and wait for its first decoded frame."""
        if self._thread is not None:
            raise RuntimeError("camera is already running")
        if self._gstreamer:
            self._capture = self._capture_factory(self._source, cv2.CAP_GSTREAMER)
        else:
            self._capture = self._capture_factory(self._source)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"cannot open Lite3 {self._source_kind} camera source")

        # Ask the backend for the newest frame rather than a deep decode queue. Not all
        # backends honour this, but unsupported properties fail harmlessly.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._stop.clear()
        self._new_frame.clear()
        self._thread = threading.Thread(target=self._run, name="lite3-camera", daemon=True)
        self._thread.start()
        if not self._new_frame.wait(wait_s):
            self.stop()
            raise RuntimeError(
                f"no frame from Lite3 {self._source_kind} camera source within {wait_s:.1f}s "
                f"({self._errors} read errors)"
            )

    def stop(self) -> None:
        """Stop capture. Safe after partial startup and safe to call twice."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("Lite3 camera reader did not stop within 2.0s")
            self._thread = None

        # VideoCapture backends are not required to make read() and release() safe from
        # different threads. The reader releases its own capture in _run(); this branch
        # only covers a partially-started camera that never acquired a reader thread.
        capture, self._capture = self._capture, None
        if thread is None and capture is not None:
            capture.release()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def _run(self) -> None:
        capture = self._capture
        if capture is None:
            return
        try:
            while not self._stop.is_set():
                ok, image = capture.read()
                arrival = time.monotonic()
                if not ok or image is None:
                    if not self._stop.is_set():
                        self._errors += 1
                        self._stop.wait(0.02)
                    continue

                stamp = None
                if self._stamp_fn is not None:
                    try:
                        stamp = self._stamp_fn()
                    except Exception:
                        stamp = None
                self._seq += 1
                frame = Frame(image=image, capture_time=arrival, seq=self._seq, stamp=stamp)
                with self._lock:
                    self._frame = frame
                self._new_frame.set()
        finally:
            capture.release()

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    def wait_for_new(self, after_seq: int, timeout: float) -> Frame | None:
        deadline = time.monotonic() + timeout
        while True:
            self._new_frame.clear()
            frame = self.latest()
            if frame is not None and frame.seq > after_seq:
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            self._new_frame.wait(remaining)

    @property
    def error_count(self) -> int:
        return self._errors
