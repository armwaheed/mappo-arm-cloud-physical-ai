#!/usr/bin/env python3
"""Serve the Go2's front camera as JPEG over HTTP, from the robot itself.

The dashboard's driver needs Python >= 3.11 and this robot has 3.8/3.9, so the driver
runs on a workstation and cannot reach `unitree_sdk2py`. This is the smallest thing that
closes that gap: the SDK call stays on the robot, and the frame crosses as HTTP.

Read-only. Opens the video client and nothing else. No motion, no lease, no writes.
"""
import http.server, socketserver, threading, time, sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

PORT = 8801
_latest = {"jpeg": None, "seq": 0, "t": 0.0}
_lock = threading.Lock()


def pump():
    ChannelFactoryInitialize(0, "eth0")
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    seq = 0
    while True:
        try:
            code, data = client.GetImageSample()
            if code == 0 and data:
                seq += 1
                with _lock:
                    _latest.update(jpeg=bytes(data), seq=seq, t=time.time())
        except Exception as exc:                      # a dropped frame is not fatal
            sys.stderr.write("frame error: %r\n" % (exc,))
        time.sleep(0.08)                              # ~12 Hz ceiling; the SDK sets the real rate


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):                        # keep the robot's console quiet
        pass

    def do_GET(self):
        with _lock:
            jpeg, seq, t = _latest["jpeg"], _latest["seq"], _latest["t"]
        if self.path.startswith("/status"):
            body = ('{"seq": %d, "age_s": %.2f, "have_frame": %s}'
                    % (seq, time.time() - t if t else -1.0,
                       "true" if jpeg else "false")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not jpeg:
            self.send_error(503, "no frame yet")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(jpeg)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    threading.Thread(target=pump, daemon=True).start()
    time.sleep(2.0)
    print("go2 frame server on :%d  (/ = latest jpeg, /status = json)" % PORT, flush=True)
    Server(("0.0.0.0", PORT), Handler).serve_forever()
