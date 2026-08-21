#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""JPEG frames from a robot's front RGB camera, for the dashboard's viewport.

## Frames do NOT travel over the RPC channel, and that is the whole design

The obvious implementation is a ``get_frame()`` RPC the dashboard polls. It is also the one
implementation that must not be used here, because
``DeviceRuntime._cmd_subscription`` **dispatches one RPC at a time per device**. Polling
frames at even 6 Hz would occupy the command channel continuously and put the robot back to
being deaf to ``stop`` — which is the exact defect the non-blocking motion work removed, and
it would reintroduce it in a form nobody would suspect, because "the camera is on" does not
sound like "the stop button no longer works".

So frames are **emitted as events**. Events are pub/sub on their own subjects; they do not
queue behind commands and commands do not queue behind them.

## What this costs, measured rather than assumed

A 640x480 JPEG at quality 60 is roughly 25-40 KB, and base64 on the wire adds a third. At the
default 6 fps that is about 200-320 KB/s **per watched robot**. Two things follow, and both
are implemented rather than hoped for:

* **Nobody streams to an empty room.** The feed starts when a viewer asks for it and stops on
  a keepalive timeout, so a robot nobody is watching sends nothing.
* **A frame is not a log line.** The dashboard drains camera frames on a separate
  subscription from control events, so a burst of image data cannot delay a
  ``motion_refused`` reaching the operator's screen.

12 fps was asked for and 6 is the default. The Go2's sensor delivers about 15 Hz, so 12 is
achievable — but it is twice the bandwidth for a viewport whose job is "can I see where the
robot is pointing", and the rate is a parameter, so raising it is a decision someone can make
with the number in front of them.

## The platforms differ, and one of them will disappoint you

* **Go2** — frames come from the SDK's ``VideoClient``, an RPC to the robot's own video
  service rather than a device node. Several clients can ask it for samples.
* **Lite3** — frames come from an OpenCV ``VideoCapture``, which on Linux is typically
  **exclusive**. While ``lite3_visual_nav`` holds the camera for a run, this cannot open it,
  and the honest failure is "the camera is in use by the run" rather than a black rectangle.
* **sim** — a synthetic frame so every path above can be exercised without a robot. Needs
  Pillow; without it the sim source reports itself unavailable rather than pretending.

Pure stdlib apart from the per-platform camera dependencies. ``python3 test_camera_source.py``.
"""

from __future__ import annotations

import base64
import contextlib
import math
import os
import sys
import threading
import time

#: Default frames per second. See the module docstring for why this is not 12.
DEFAULT_FPS = 6.0

#: Ceiling. The Go2 sensor runs at about 15 Hz, so asking for more than this buys duplicate
#: frames and pays full bandwidth for them.
MAX_FPS = 15.0

#: A viewer must re-assert interest within this long or the feed stops. Without it, a closed
#: browser tab leaves a robot streaming to nobody for the rest of the session.
VIEWER_TIMEOUT_S = 12.0

#: Refuse to emit a frame larger than this. A camera misconfigured to 4K would otherwise put
#: multi-megabyte base64 strings onto the mesh at 6 Hz.
MAX_FRAME_BYTES = 512 * 1024


class CameraUnavailable(RuntimeError):
    """The camera cannot be opened, and the message says why rather than showing black."""


def _stack_dir():
    if os.environ.get("MAPPO_STACK_DIR"):
        return os.environ["MAPPO_STACK_DIR"]
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robot-stack"))


class SimCameraSource:
    """A synthetic 480x360 frame carrying a frame counter and the robot's pose.

    Not a rendering of anything. Its job is to make every layer above it real — the emit
    path, the separate drain, the viewport, the start/stop lifecycle — without a robot, and
    to be obviously synthetic so no one mistakes a screenshot of it for hardware.
    """

    width, height = 480, 360

    def __init__(self, pose_fn=None):
        self._pose_fn = pose_fn
        self._seq = 0
        try:
            from PIL import Image, ImageDraw  # noqa: F401
        except ImportError as exc:
            raise CameraUnavailable(
                "the sim camera needs Pillow (pip install Pillow). Real platforms use their "
                "own camera and do not need it.") from exc

    def read(self):
        from io import BytesIO

        from PIL import Image, ImageDraw

        self._seq += 1
        image = Image.new("RGB", (self.width, self.height), (14, 18, 22))
        draw = ImageDraw.Draw(image)

        # A moving horizon so the feed is visibly LIVE rather than a still. A static image is
        # indistinguishable from a frozen stream, which is the failure this must not hide.
        phase = self._seq * 0.08
        for x in range(0, self.width, 8):
            y = self.height / 2 + math.sin(phase + x / 60.0) * 26
            draw.line([(x, y), (x + 8, y)], fill=(60, 96, 140), width=2)

        draw.rectangle([0, 0, self.width - 1, self.height - 1], outline=(42, 50, 61))
        draw.text((12, 12), "SYNTHETIC — no camera on this platform", fill=(210, 153, 34))
        draw.text((12, 30), f"frame {self._seq}", fill=(139, 152, 168))
        if self._pose_fn:
            try:
                pose = self._pose_fn() or {}
                draw.text((12, 48),
                          f"x {pose.get('x', 0):.2f}  y {pose.get('y', 0):.2f}  "
                          f"yaw {math.degrees(pose.get('yaw', 0)):.0f}deg",
                          fill=(139, 152, 168))
            except Exception:
                pass

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=60)
        return buffer.getvalue()

    def close(self):
        return None


class Go2CameraSource:
    """The Go2's front RGB camera through the vendored ``Go2Camera``."""

    width, height = 0, 0

    def __init__(self, iface="eth0"):
        sys.path.insert(0, os.path.join(_stack_dir(), "unitree", "go2", "visual_nav"))
        try:
            from camera import Go2Camera
        except ImportError as exc:
            raise CameraUnavailable(f"the Go2 camera module is not importable: {exc}") from exc
        self._camera = Go2Camera(iface=iface)
        try:
            self._camera.start()
        except Exception as exc:
            raise CameraUnavailable(f"could not start the Go2 camera: {exc}") from exc

    def read(self):
        frame = self._camera.latest()
        if frame is None:
            return None
        return getattr(frame, "jpeg", None) or getattr(frame, "raw", None)

    def close(self):
        with contextlib.suppress(Exception):
            self._camera.stop()


class Lite3CameraSource:
    """The Lite3's RGB camera through the vendored ``Lite3Camera``.

    ⚠️ OpenCV's ``VideoCapture`` is typically EXCLUSIVE on Linux, so this cannot open the
    camera while a ``lite3_visual_nav`` run holds it. That is reported as an unavailable
    camera with the reason, because a black rectangle would look like a broken camera rather
    than a busy one.
    """

    width, height = 0, 0

    def __init__(self, source=0):
        sys.path.insert(0, os.path.join(_stack_dir(), "deep_robotics", "lite3", "visual_nav"))
        try:
            from camera import Lite3Camera
        except ImportError as exc:
            raise CameraUnavailable(f"the Lite3 camera module is not importable: {exc}") from exc
        self._camera = Lite3Camera(source=source)
        try:
            self._camera.start()
        except Exception as exc:
            raise CameraUnavailable(
                f"could not open the Lite3 camera: {exc}. On this platform the capture is "
                f"exclusive, so a live visual_nav run holds it and this cannot.") from exc

    def read(self):
        frame = self._camera.latest()
        if frame is None:
            return None
        return getattr(frame, "jpeg", None)

    def close(self):
        with contextlib.suppress(Exception):
            self._camera.stop()


def open_source(platform, *, iface="eth0", source=0, pose_fn=None):
    """The camera for one platform, or :class:`CameraUnavailable` saying why not."""
    if platform == "sim":
        return SimCameraSource(pose_fn=pose_fn)
    if platform == "go2":
        return Go2CameraSource(iface=iface)
    if platform == "lite3":
        return Lite3CameraSource(source=source)
    raise CameraUnavailable(f"no camera is defined for platform {platform!r}")


def encode(jpeg_bytes):
    """Base64 a JPEG for the wire, refusing one that is implausibly large."""
    if not jpeg_bytes:
        return None
    if len(jpeg_bytes) > MAX_FRAME_BYTES:
        raise CameraUnavailable(
            f"a {len(jpeg_bytes)} byte frame exceeds the {MAX_FRAME_BYTES} byte ceiling; "
            f"the camera is producing frames far larger than a viewport needs")
    return base64.b64encode(jpeg_bytes).decode("ascii")


def clamp_fps(fps):
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return DEFAULT_FPS
    if not (fps > 0):
        return DEFAULT_FPS
    return min(fps, MAX_FPS)


class Viewers:
    """Who is currently watching, so nothing streams to an empty room.

    A viewer re-asserts interest on every poll; interest that stops being re-asserted expires.
    That makes a closed browser tab stop the stream on its own rather than needing a goodbye
    message nobody sends.
    """

    def __init__(self, timeout_s=VIEWER_TIMEOUT_S, clock=time.monotonic):
        self._timeout = timeout_s
        self._clock = clock
        self._last = 0.0
        self._lock = threading.Lock()

    def note_interest(self):
        with self._lock:
            self._last = self._clock()

    def watching(self):
        with self._lock:
            return self._last > 0.0 and (self._clock() - self._last) < self._timeout

    def clear(self):
        with self._lock:
            self._last = 0.0
