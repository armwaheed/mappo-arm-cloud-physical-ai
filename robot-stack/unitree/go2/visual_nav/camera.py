# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Front RGB camera capture for the Go2 — a background grabber with a latest-frame API.

The Go2 exposes its forward fisheye through the ``VideoClient`` RPC, which hands back
a JPEG of the most recent sample. Measured on this unit: **1920x1080, ~14.7 genuinely
new frames per second** (67 ms apart, very steady), while the RPC itself will answer
~4x faster than that and simply repeat the last JPEG.

That mismatch is why this module exists rather than callers poking ``VideoClient``
directly:

  * **Repeats are dropped before the decode**, by comparing the compressed bytes. A
    1080p JPEG decode costs ~25 ms on this Jetson; spending that on a frame we have
    already seen would burn a quarter of a core for nothing, and those cores are
    contended by the person detector.
  * **Capture is decoupled from consumption.** The detector runs at ~7 Hz and the
    control loop at 10 Hz; neither should be pinned to the camera's cadence, and
    neither should block on an RPC. The grabber runs in its own thread and
    :meth:`Go2Camera.latest` is a non-blocking read of the newest frame.
  * **Frames carry a capture timestamp and a caller-supplied stamp.** Perception is
    a few hundred milliseconds behind reality once decode and inference are paid for —
    measured over a live walking run on this unit, median **285 ms** and p90 388 ms,
    which is well above the ~150 ms a bench measurement suggests. Transforming a
    detection with the pose the robot has *now* smears every bearing while it is
    turning, so the pose must be sampled when the shutter did — hence ``stamp_fn``.

The ``VideoClient`` RPC segfaults unless ``LD_LIBRARY_PATH`` points at the CycloneDDS
build its Python binding was compiled against; source
``../install/setup_env.sh`` first. That is a property of every RPC client on this
robot, not of this module — the full diagnosis is in that script's header.
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
    """One decoded camera frame and everything needed to interpret it later."""

    image: np.ndarray          # BGR, full sensor resolution
    capture_time: float        # time.monotonic() immediately after the RPC returned
    seq: int                   # monotonically increasing count of NEW frames
    stamp: Any = None          # whatever stamp_fn returned at capture (e.g. a Pose)

    @property
    def age(self) -> float:
        """Seconds since this frame was captured."""
        return time.monotonic() - self.capture_time


class Go2Camera:
    """Threaded reader for the Go2's front RGB camera.

    Args:
        stamp_fn: called on the capture thread the instant a new frame arrives, with
            no arguments; the result rides along on :attr:`Frame.stamp`. Use it to
            sample robot pose at shutter time. Must be cheap and non-blocking — it is
            on the capture path. Exceptions from it are swallowed (a stamp failure
            must not kill the camera) and leave ``stamp`` as ``None``.
        poll_hz: how often to ask the RPC for a sample. Must exceed the sensor's
            ~15 Hz or new frames are missed; the default oversamples ~2x so a late
            poll still lands within one frame period.
        timeout_s: RPC timeout handed to ``VideoClient``.
    """

    def __init__(self, iface: str = "eth0", init_dds: bool = True,
                 stamp_fn: Callable[[], Any] | None = None,
                 poll_hz: float = 30.0, timeout_s: float = 3.0) -> None:
        self._iface = iface
        self._init_dds = init_dds
        self._stamp_fn = stamp_fn
        self._poll_interval = 1.0 / poll_hz
        self._timeout_s = timeout_s

        self._client = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._last_raw: bytes | None = None
        self._seq = 0
        self._errors = 0
        self._new_frame = threading.Event()

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def start(self, wait_s: float = 5.0) -> None:
        """Connect and begin grabbing; block until the first frame or raise."""
        from unitree_sdk2py.go2.video.video_client import VideoClient

        if self._init_dds:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(0, self._iface)

        self._client = VideoClient()
        self._client.SetTimeout(self._timeout_s)
        self._client.Init()

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="go2-camera", daemon=True)
        self._thread.start()

        if not self._new_frame.wait(wait_s):
            self.stop()
            raise RuntimeError(
                f"no camera frame within {wait_s:.1f}s ({self._errors} RPC errors). "
                f"Did you source install/setup_env.sh? An unset LD_LIBRARY_PATH makes "
                f"the VideoClient RPC segfault or time out.")

    def stop(self) -> None:
        """Stop the grab thread. Safe to call twice, and from an error path."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> Go2Camera:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ── Capture thread ──────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            deadline = time.monotonic() + self._poll_interval
            try:
                code, data = self._client.GetImageSample()
            except Exception:
                self._errors += 1
                code, data = -1, None

            if code == 0 and data:
                raw = bytes(data)
                # Byte-compare against the previous JPEG: the RPC re-serves the same
                # sample between sensor frames, and an exact compare is both cheaper
                # than a hash and free of collision risk.
                if raw != self._last_raw:
                    self._last_raw = raw
                    self._publish(raw)
            elif code != 0:
                self._errors += 1

            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._stop.wait(remaining)

    def _publish(self, raw: bytes) -> None:
        stamp = None
        if self._stamp_fn is not None:
            try:
                stamp = self._stamp_fn()
            except Exception:
                stamp = None  # a bad stamp must never take the camera down
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self._errors += 1
            return
        self._seq += 1
        frame = Frame(image=image, capture_time=time.monotonic(), seq=self._seq,
                      stamp=stamp)
        with self._lock:
            self._frame = frame
        self._new_frame.set()

    # ── Consumer API ────────────────────────────────────────────────────────
    def latest(self) -> Frame | None:
        """The newest frame, or ``None`` before the first one. Never blocks."""
        with self._lock:
            return self._frame

    def wait_for_new(self, after_seq: int, timeout: float) -> Frame | None:
        """Block until a frame newer than ``after_seq`` arrives, or ``timeout``.

        Lets the detector thread sleep instead of spinning, and guarantees it never
        re-runs inference on a frame it has already processed.
        """
        deadline = time.monotonic() + timeout
        while True:
            # Clear BEFORE reading, not after. The other order loses a wakeup: a frame
            # that lands between the read and the clear sets the event, the clear wipes
            # it, and this waits out the full timeout for a frame already published.
            # Clearing first means any such frame either shows up in the read below or
            # re-sets the event after the clear — never both missed.
            self._new_frame.clear()
            frame = self.latest()
            if frame is not None and frame.seq > after_seq:
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._new_frame.wait(remaining)

    @property
    def error_count(self) -> int:
        """RPC/decode failures since :meth:`start` — a health signal for the caller."""
        return self._errors


def main() -> None:
    """Record a few seconds of camera to disk — the read-only camera smoke test."""
    import argparse

    ap = argparse.ArgumentParser(description="Go2 front-camera capture check (no motion).")
    ap.add_argument("--iface", default="eth0", help="DDS network interface")
    ap.add_argument("--seconds", type=float, default=5.0, help="how long to record")
    ap.add_argument("--out", default="go2_camera_check.mp4", help="output video path")
    args = ap.parse_args()

    with Go2Camera(iface=args.iface) as cam:
        first = cam.latest()
        height, width = first.image.shape[:2]
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15.0,
                                 (width, height))
        if not writer.isOpened():
            raise SystemExit(f"cannot open {args.out} for writing (mp4v)")
        try:
            seq, end = 0, time.monotonic() + args.seconds
            while time.monotonic() < end:
                frame = cam.wait_for_new(seq, timeout=1.0)
                if frame is None:
                    print("  (no new frame in 1.0s)")
                    continue
                seq = frame.seq
                writer.write(frame.image)
        finally:
            writer.release()
        print(f"{seq} frames -> {args.out}  ({width}x{height}, "
              f"{cam.error_count} RPC errors)")


if __name__ == "__main__":
    main()
