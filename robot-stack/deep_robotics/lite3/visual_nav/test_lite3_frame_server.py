#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the RTSP-to-JPEG frame server.

Nothing here opens a camera, binds a port or reaches the network. The handler is driven
directly over a ``BytesIO`` and the reader over a capture double, because the two failures
this file exists to catch are both invisible to a test that only asks "did the server
start": a reply that is stale, and a reader that starves the HTTP thread while the port is
still listening.
"""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.visual_nav.lite3_frame_server import (
    Handler,
    LatestFrame,
    read_forever,
)

_URL = "rtsp://127.0.0.1:8554/nothing"


class _ClosedCapture:
    """A capture that never opens: the RTSP source is not publishing yet."""

    def __init__(self):
        self.released = False

    def isOpened(self):
        return False

    def read(self):
        raise AssertionError("read() on a capture that never opened")

    def release(self):
        self.released = True


class _FrameCapture:
    """A capture that serves ``frames`` in order and then fails its read, like a dropped
    stream. It reports itself closed once released, so the reader's reopen path is real."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def isOpened(self):
        return not self.released

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


class _Reply(Handler):
    """The real handler, wired to a buffer instead of a socket.

    ``BaseHTTPRequestHandler.__init__`` reads a request off a socket and dispatches it, so it
    is deliberately not called: binding a port to test a reply makes the test depend on a
    free port and on the OS, and measures neither of the things below.
    """

    def __init__(self, frames):
        self.frames = frames
        self.wfile = io.BytesIO()
        self.rfile = io.BytesIO(b"")
        self.requestline = "GET / HTTP/1.1"
        self.request_version = "HTTP/1.1"
        self.command = "GET"
        self.client_address = ("127.0.0.1", 0)


def _get(frames):
    """Serve one GET from ``frames`` and return its ``(status line, headers, body)``."""
    reply = _Reply(frames)
    reply.do_GET()
    head, _, body = reply.wfile.getvalue().partition(b"\r\n\r\n")
    status, _, headers = head.partition(b"\r\n")
    return status, headers, body


def _image(value):
    """A 4x4 BGR frame of one colour, so two of them encode to different bytes."""
    return np.full((4, 4, 3), value, dtype=np.uint8)


def _encode(image, quality=70):
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


def test_a_get_before_any_frame_is_503_and_not_an_empty_200():
    status, headers, _ = _get(LatestFrame())
    assert b"503" in status, status
    # An empty 200 would render as a broken image too, which is why the code cannot be that:
    # it is the one reply that makes "not started yet" indistinguishable from "dark camera".
    assert b"image/jpeg" not in headers, headers


def test_a_stored_frame_is_served_as_a_jpeg_of_its_own_length():
    frames = LatestFrame()
    jpeg = b"\xff\xd8" + b"x" * 4093 + b"\xff\xd9"
    frames.publish(jpeg)
    status, headers, body = _get(frames)
    assert b"200" in status, status
    assert b"Content-Type: image/jpeg" in headers, headers
    # The length is checked against the frame, not against a constant: a Content-Length that
    # disagrees with the body truncates the picture in the browser and logs nothing here.
    assert b"Content-Length: %d" % len(jpeg) in headers, headers
    assert body == jpeg


def test_the_reply_forbids_caching_the_frame():
    frames = LatestFrame()
    frames.publish(b"\xff\xd8jpeg\xff\xd9")
    _, headers, _ = _get(frames)
    # Every GET is a different picture at the same URL. A cached one is frozen and looks live.
    assert b"Cache-Control: no-store" in headers, headers


def test_only_the_latest_frame_is_served():
    frames = LatestFrame()
    old = b"\xff\xd8" + b"o" * 100 + b"\xff\xd9"
    new = b"\xff\xd8" + b"n" * 900 + b"\xff\xd9"
    frames.publish(old)
    frames.publish(new)
    _, headers, body = _get(frames)
    assert body == new
    # Deliberately different lengths: a stale body would also carry a stale header, and the
    # header is the half a browser acts on before it draws anything.
    assert b"Content-Length: %d" % len(new) in headers, headers


def test_the_reader_sleeps_before_retrying_a_source_it_cannot_open():
    """The regression this exists for cost a working port and gave no log line.

    Reopening RTSP in a tight loop holds the GIL through each blocking ``VideoCapture``
    attempt and starves the HTTP thread: the port listens and every GET times out. The fix
    is one sleep, so the measurement is that a failed open is always followed by a wait, and
    that the wait is not zero.
    """
    stop = threading.Event()
    attempts, naps = [], []

    def capture(url):
        attempts.append(url)
        if len(attempts) == 3:
            stop.set()
        return _ClosedCapture()

    read_forever(_URL, 70, LatestFrame(), capture=capture, sleep=naps.append, stop=stop)

    assert attempts == [_URL] * 3, attempts
    assert len(naps) == len(attempts), naps
    assert all(nap > 0 for nap in naps), naps


def test_the_reader_keeps_only_the_newest_frame():
    stop = threading.Event()
    first, second = _image(10), _image(200)
    frames = LatestFrame()

    read_forever(_URL, 70, frames,
                 capture=lambda url: _FrameCapture([first, second]),
                 sleep=lambda seconds: stop.set(), stop=stop)

    assert frames.latest() == _encode(second)
    assert frames.latest() != _encode(first)


def test_the_reader_releases_a_capture_that_stops_reading_before_opening_another():
    """A dropped stream must not leak the dead capture. FFmpeg holds a socket per capture,
    so a reader that reopens without releasing accumulates them for as long as it runs."""
    stop = threading.Event()
    opened = []

    def capture(url):
        cap = _FrameCapture([_image(30)] if not opened else [])
        opened.append(cap)
        if len(opened) == 2:
            stop.set()
        return cap

    read_forever(_URL, 70, LatestFrame(), capture=capture,
                 sleep=lambda seconds: None, stop=stop)

    assert len(opened) == 2, opened
    assert opened[0].released


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_frame_server: {len(tests)}/{len(tests)} passed")
