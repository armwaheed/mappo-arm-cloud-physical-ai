#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A self-hosted MAPPO checkpoint server — the alternative to a cloud bucket.

Serves a JSON index and the ``.npz`` files beside it, which is the whole contract the
dashboard's "Browse server" needs. That is the point worth noticing: **pulling checkpoints
from your own hardware required no new capability in the dashboard.** ``cloud_models`` has
taken an ``http(s)://`` source since the first commit; what was missing was only that a
server could be *browsed* the way a bucket could, so the UI implied a bucket was the real
answer. It no longer does.

⚠️ **``--label`` is a claim about where this is running and nothing enforces it.** Point it
at a machine in Tokyo and it is true; run it on a laptop and it is a caption. The index says
so in ``location_claimed`` rather than ``location``, and the dashboard shows what the server
says about itself — so the honest reading is always available to anyone who looks.

    python3 model_server.py --dir ./checkpoints --port 9000 \\
        --label "Arm AGI CPU server" --location "Tokyo, Japan"
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.server
import json
import os
import socketserver
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

SUFFIX = ".npz"


def build_index(directory: Path, label: str, location: str, simulated: bool) -> dict:
    models = []
    for path in sorted(directory.glob(f"*{SUFFIX}")):
        stat = path.stat()
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        models.append({
            "name": path.name,
            "size_bytes": stat.st_size,
            "sha256": digest.hexdigest(),
            "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                                .isoformat().replace("+00:00", "Z"),
        })
    return {
        "server": label,
        # NOT "location". Nothing here verifies where this process is running, and a field
        # called "location" would read as a fact. This one reads as what it is.
        "location_claimed": location,
        "simulated": simulated,
        "arch": os.uname().machine,
        "models": models,
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    index_payload: ClassVar[dict] = {}

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()}  {fmt % args}", flush=True)

    def end_headers(self):
        # The dashboard is served from a different origin, so the browser needs this for the
        # index fetch. Only the index and the checkpoints live here; there is nothing to
        # protect from a cross-origin read that is not already being handed out.
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] in ("/index.json", "/"):
            body = json.dumps(self.index_payload, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True, help="Directory of .npz checkpoints.")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--label", default="model server")
    parser.add_argument("--location", default="unstated")
    parser.add_argument("--real", action="store_true",
                        help="Assert this is genuinely running where --location says. "
                             "Without it the index reports itself as simulated.")
    args = parser.parse_args()

    directory = Path(args.dir).resolve()
    if not directory.is_dir():
        parser.error(f"{directory} is not a directory")

    Handler.index_payload = build_index(directory, args.label, args.location, not args.real)
    os.chdir(directory)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), Handler) as server:
        count = len(Handler.index_payload["models"])
        print(f"{args.label} ({args.location}) serving {count} checkpoint(s) from "
              f"{directory} on {args.host}:{args.port}", flush=True)
        if not args.real:
            print("  NOTE: index reports simulated=true; --location is a claim, not a fact.",
                  flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
