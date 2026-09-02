#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve the Lite3's RTSP camera as one JPEG per GET, for the dashboard camera viewport.

The dashboard's ``--camera-url`` wants an endpoint that answers a GET with a single JPEG.
``dashboard/go2_frame_server.py`` cannot be that endpoint on a Lite3: it imports
``unitree_sdk2py`` and pulls frames through the Go2 SDK's ``VideoClient``, an RPC to a video
service this robot does not run. The Lite3 publishes **RTSP** instead, so this bridges the
two and does nothing else -- it opens the stream, encodes, and serves. No lease, no motion,
no writes.

Measured on both robots (192.168.1.120 and 192.168.1.2), reading
``rtsp://127.0.0.1:8554/test``: one GET returns HTTP 200 with a **127-135 KB** JPEG at the
default quality, and the dashboard viewport streams from it. Both run it as an enabled
``lite3-frame-server`` systemd unit.

A reader thread keeps ONLY the newest frame. OpenCV's RTSP capture buffers internally, so
serving straight out of ``read()`` inside the request handler hands the viewer a frame that
is however many seconds behind the robot -- which on the page an operator watches a run on
is worse than no picture, because it looks current.
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

#: How long the reader waits before reopening a source it could not open. Load-bearing; see
#: :func:`read_forever`. Any positive value fixes the starvation -- 1 s is the one measured.
RETRY_INTERVAL_S = 1.0

#: How long it waits after a read that failed on a capture that *was* open. Shorter, because
#: this is a stream that dropped rather than one that was never there.
REOPEN_INTERVAL_S = 0.2


class LatestFrame:
    """The one encoded frame a GET is answered from, and the lock over it.

    Only the newest is kept, deliberately: a queue here would serve an operator a backlog,
    and a backlog of camera frames is indistinguishable on screen from a live feed.

    The lock is not decoration. Under ``ThreadingHTTPServer`` the reader thread and one
    handler thread per client touch this object at the same time.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None

    def publish(self, jpeg: bytes) -> None:
        """Replace the stored frame. The previous one is dropped, not queued."""
        with self._lock:
            self._jpeg = jpeg

    def latest(self) -> bytes | None:
        """The newest frame, or ``None`` when the reader has not published one yet."""
        with self._lock:
            return self._jpeg


#: The store ``main`` runs against. :class:`Handler` is instantiated per request by
#: ``HTTPServer``, so the store cannot be passed to its constructor; it is a class attribute
#: instead, which is also the seam the offline tests point at their own store.
FRAMES = LatestFrame()


def open_rtsp(url: str):
    """Open ``url`` on the FFmpeg backend, which is the one that speaks RTSP.

    Named and separate so the reader takes a one-argument factory: an offline test can hand
    it a double without also standing up a capture backend.
    """
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def read_forever(url, quality, frames, *, capture=None, sleep=time.sleep, stop=None) -> None:
    """Publish the newest JPEG from ``url`` into ``frames`` until ``stop`` is set.

    ``capture``, ``sleep`` and ``stop`` exist so this loop can be measured without a camera,
    a network or a wall clock. Left alone it opens RTSP, never sleeps a real thread more
    than a second, and runs until the process ends.
    """
    if capture is None:
        capture = open_rtsp
    if stop is None:
        stop = threading.Event()
    cap = None
    try:
        while not stop.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = capture(url)
                if not cap.isOpened():
                    # SLEEP, DO NOT SPIN. Reopening RTSP in a tight loop holds the GIL for
                    # the whole of each blocking attempt and starves the HTTP thread: the
                    # port listens, and every GET times out with nothing in the log to say
                    # why. Measured on robot 2 -- this one line is the difference between a
                    # dead viewport and a working one.
                    sleep(RETRY_INTERVAL_S)
                    continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                sleep(REOPEN_INTERVAL_S)
                continue
            encoded, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if encoded:
                frames.publish(buf.tobytes())
    finally:
        if cap is not None:
            cap.release()


class Handler(BaseHTTPRequestHandler):
    """One JPEG per GET, and 503 while there is not one."""

    #: The store this handler answers from. ``main`` leaves it as the module singleton.
    frames = FRAMES

    def do_GET(self):
        jpeg = self.frames.latest()
        if jpeg is None:
            # 503, not an empty 200: "the stream has not come up" and "the camera is dark"
            # are different faults, and a zero-length 200 renders as a broken image for both.
            self.send_error(503, "no frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        # Every GET is a different frame at the same URL, so a cached reply is a frozen
        # picture that still looks live -- the same failure the latest-only store avoids.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpeg)

    def log_message(self, *args):
        """Silence per-request logging: at 6 fps it is six lines a second carrying nothing."""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rtsp", default="rtsp://127.0.0.1:8554/test",
                        help="the RTSP URI to read. Deployment-specific; no default is right.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8801,
                        help="the port --camera-url is pointed at.")
    parser.add_argument("--quality", type=int, default=70,
                        help="JPEG quality. 70 measured 127-135 KB a frame on both robots.")
    args = parser.parse_args(argv)
    threading.Thread(target=read_forever, args=(args.rtsp, args.quality, FRAMES),
                     name="rtsp-reader", daemon=True).start()
    print(f"[frame-server] {args.rtsp} -> http://{args.host}:{args.port}/", flush=True)
    # THREADING, not HTTPServer. The dashboard polls this several times a second, and a
    # single-threaded server serialises every other client behind that poll: the port
    # accepts the connection and the request then waits, which reads as an unreachable
    # camera from anywhere except the machine already being served. This is correctness for
    # concurrent clients -- a second viewer, an evidence capture -- and not a fix for an
    # outage; the one outage it was suspected of was a broken shell harness, not the server.
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
