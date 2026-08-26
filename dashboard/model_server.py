#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve a directory of MAPPO checkpoints as a Cloud AI source, over plain HTTP.

The other end of :mod:`cloud_models` — that module is the robot pulling a checkpoint down,
and this is something to pull it from. Until the training bucket exists there is nowhere to
point ``download_model`` at except a presigned URL nobody has, so the Cloud AI panel could
only ever be exercised against a fixture inside a test. This makes it a thing you can start.

**This is a stand-in for the model server, not a smaller version of it.** The delivered
article is an S3 bucket, and the reason a stand-in is worth writing is that the seam between
"where checkpoints live" and "the robot that fetches one" gets exercised now rather than on
the day the bucket's URL arrives.

## The seam, and how a bucket drops into it

Nothing in the dashboard knows this server exists. A checkpoint source is a **base address**
carried in the driver's ``--model-sources`` file and advertised by the robot, and
``cloud_models.parse_source`` dispatches on its scheme. So the two deployments differ by one
JSON object and no code:

```jsonc
{"label": "workstation", "location": "the demo LAN",           // this server
 "index_url": "http://192.168.123.50:8800/index.json"}

{"label": "Cloud AI", "location": "eu-west-1",                 // the bucket, later
 "bucket": "mappo-checkpoints", "prefix": "go2/"}
```

``list_cloud_models`` already branches on which key is present, ``download_model`` already
routes ``s3://`` through :func:`cloud_models.fetch_s3` and ``http(s)://`` through
:func:`cloud_models.fetch_http`, and ``ModelStore.install`` inspects the bytes either way.
Swapping this server for the bucket is an edit to a config file on the robot.

**Per-model URLs are relative by default, and that is the load-bearing choice.**
:func:`cloud_models.list_http_index` resolves each entry against the URL the index was
actually fetched from, so this server never has to know its own public address — it works
unchanged on loopback, on the demo LAN, and behind a port forward. ``--base-url`` overrides
that for the one case relative addresses cannot survive: when the robot must reach the files
at a *different* address from the one the index came from, which is what a reverse proxy or
a NAT hop does. Hard-coding an address that happened to work on the author's laptop is the
failure this pair of behaviours exists to prevent.

**``--emit-sources`` writes the driver's source file from the address the server actually
bound**, so the URL is produced once instead of typed into two files that then disagree.
``robot_driver._load_sources`` reads what it writes, and ``test_model_server.py`` pins that.

## What it refuses

The names come from a directory listing, so this is a far weaker threat model than a URL
field on a web page — but the checkpoint names it advertises become the last path segment
``cloud_models.name_from_source`` turns into a filename on the robot. Serving a name the
robot must then reject is a defect that surfaces two machines away, so the gate is applied
**here**, by importing ``model_store.safe_model_name`` rather than restating it: a file
whose name this store could not install is not advertised and cannot be fetched. Requests
are resolved against the served directory and refused if they land outside it, which is what
stops ``/models/..%2f..%2fetc%2fpasswd`` being a file read.

Read-only: there is no upload path, and nothing here writes into the served directory.

Pure stdlib. ``python3 test_model_server.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import os
import sys
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# This directory goes on sys.path before any sibling import, in ONE block. `ruff --fix`
# will hoist an import above the line that makes it importable if they are interleaved.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_store import MODEL_SUFFIX, ModelStoreError, sha256_file
from model_store import safe_model_name as _safe_model_name

log = logging.getLogger("model_server")

#: Where the index lives. A fixed path rather than serving it from ``/`` so that ``/`` can
#: stay a human-readable page — an operator who pastes the base address into a browser to
#: check the server is up should see what is on it, not a JSON download.
INDEX_PATH = "/index.json"

#: The path prefix the checkpoints themselves are served under. Kept distinct from the index
#: so a future addition (a log, a metrics endpoint) cannot collide with a checkpoint name.
MODELS_PREFIX = "/models/"

DEFAULT_PORT = 8800

#: Block size for the file send. The delivered checkpoint is 262 KiB, so this is one read in
#: practice; it is a loop because a 24-ray retrain is the kind of thing that grows.
_BLOCK = 1 << 16


class ModelServerError(RuntimeError):
    """The server cannot be started as asked."""


def _iso(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()


class ModelCatalogue:
    """The checkpoints in one directory, and their digests.

    Re-read on every request rather than cached, so dropping a new checkpoint into the
    directory publishes it without a restart — which is how a training run will actually
    hand one over. Only the SHA-256 is memoised, keyed on the file's size and mtime, because
    hashing is the one part that is not free and the key changes whenever the bytes could
    have.
    """

    #: Ceiling on the digest memo. The key includes mtime, so every REWRITE of a checkpoint
    #: adds an entry that can never be hit again — a server left up across a training run
    #: that keeps replacing one file would otherwise grow without bound. Far above any
    #: plausible directory, so in normal use nothing is ever evicted.
    MAX_MEMO = 512

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise ModelServerError(f"{self.directory} is not a directory")
        self._digests: dict = {}
        self._guard = threading.Lock()

    def _digest(self, path: Path, size: int, mtime_ns: int) -> str:
        key = (path.name, size, mtime_ns)
        with self._guard:
            cached = self._digests.get(key)
        if cached:
            return cached
        digest = sha256_file(path)
        with self._guard:
            if len(self._digests) >= self.MAX_MEMO:
                # Oldest insertion first — dicts preserve insertion order, and the entry
                # least recently ADDED is the best available proxy for the one least likely
                # to be asked for again when every key is a distinct file version.
                del self._digests[next(iter(self._digests))]
            self._digests[key] = digest
        return digest

    def entries(self) -> list:
        """Every servable checkpoint, newest first.

        A file whose name ``model_store`` would refuse is skipped and logged once per scan
        rather than advertised: the robot would reject it on install, and a listing that
        offers a checkpoint which cannot be installed is worse than one that does not.
        """
        found = []
        for path in sorted(self.directory.glob(f"*{MODEL_SUFFIX}")):
            if not path.is_file():
                continue
            try:
                name = _safe_model_name(path.name)
            except ModelStoreError as exc:
                log.warning("not advertising %s: %s", path.name, exc)
                continue
            stat = path.stat()
            found.append({
                "name": name,
                "url": f"models/{urllib.parse.quote(name)}",
                "size_bytes": stat.st_size,
                "sha256": self._digest(path, stat.st_size, stat.st_mtime_ns),
                "modified": _iso(stat.st_mtime),
            })
        found.sort(key=lambda entry: (entry["modified"], entry["name"]), reverse=True)
        return found

    def resolve(self, name: str) -> Path | None:
        """The path for a requested checkpoint name, or None if it is not servable.

        Two independent checks, because they fail on different inputs: the name gate rejects
        a traversal spelled in the name, and the containment check catches anything that
        reaches a path outside the directory by another route — a symlink pointing out of
        it, most plausibly, which no amount of string inspection would see.
        """
        try:
            safe = _safe_model_name(name)
        except ModelStoreError:
            return None
        candidate = (self.directory / safe).resolve()
        if candidate.parent != self.directory or not candidate.is_file():
            return None
        return candidate


def build_index(catalogue: ModelCatalogue, base_url: str = "") -> dict:
    """The JSON document ``cloud_models.list_http_index`` reads.

    ``base_url``, when given, makes every per-model address absolute. Empty — the default —
    leaves them relative, which the client resolves against wherever it fetched the index
    from. See the module docstring for why that is the default and not the special case.
    """
    entries = catalogue.entries()
    if base_url:
        prefix = base_url if base_url.endswith("/") else base_url + "/"
        for entry in entries:
            entry["url"] = urllib.parse.urljoin(prefix, entry["url"])
    return {
        "server": "mappo-model-server",
        "generated": _iso(dt.datetime.now(tz=dt.timezone.utc).timestamp()),
        "directory": str(catalogue.directory),
        "count": len(entries),
        "models": entries,
    }


def sources_document(index_url: str, *, label: str, location: str,
                     default_model: str = "") -> dict:
    """A ``--model-sources`` file naming this server, in the driver's own schema.

    Written by ``--emit-sources`` so the address exists in one place. ``default_model`` is
    the URL the dashboard prefills its download field with, which is what stops a demo
    opening on an empty text box.
    """
    source = {"label": label, "location": location, "index_url": index_url}
    if default_model:
        source["default_model"] = default_model
    return {"sources": [source]}


class _Handler(BaseHTTPRequestHandler):
    """Three routes and nothing else: a human page, the index, and the files."""

    protocol_version = "HTTP/1.1"
    server_version = "mappo-model-server/1.0"
    catalogue: ModelCatalogue
    base_url: str = ""

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    # ── responses ────────────────────────────────────────────────────────────
    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, json.dumps(payload, indent=2).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _not_found(self, detail: str) -> None:
        # JSON rather than the stdlib's HTML error page: every consumer of this server is a
        # program, and `list_http_index` reports the status code it got either way.
        self._send_json(HTTPStatus.NOT_FOUND, {"error": detail})

    # ── routes ───────────────────────────────────────────────────────────────
    def _landing(self) -> None:
        """A plain-text page, so pasting the address into a browser answers 'is it up'."""
        entries = self.catalogue.entries()
        lines = [f"mappo-model-server  —  {self.catalogue.directory}",
                 f"{len(entries)} checkpoint(s); index at {INDEX_PATH}", ""]
        lines += [f"{e['size_bytes']:>10}  {e['sha256'][:12]}  {e['name']}" for e in entries]
        lines.append("")
        self._send(HTTPStatus.OK, "\n".join(lines).encode("utf-8"),
                   "text/plain; charset=utf-8")

    def _serve_model(self, raw_name: str) -> None:
        path = self.catalogue.resolve(urllib.parse.unquote(raw_name))
        if path is None:
            self._not_found(f"no checkpoint named {raw_name!r} is served here")
            return
        # Content-Length is read before the body is streamed. A checkpoint rewritten in
        # place between the two would send a truncated or short response — acceptable here
        # because this serves a directory that a human drops files into, and the client
        # verifies a SHA-256 it got from the index either way.
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(_BLOCK), b""):
                self.wfile.write(block)

    def do_GET(self):                        # the stdlib's spelling, not ours
        route = urllib.parse.urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._landing()
        elif route == INDEX_PATH:
            self._send_json(HTTPStatus.OK, build_index(self.catalogue, self.base_url))
        elif route == "/healthz":
            self._send_json(HTTPStatus.OK, {"ok": True})
        elif route.startswith(MODELS_PREFIX):
            self._serve_model(route[len(MODELS_PREFIX):])
        else:
            self._not_found(f"{route!r} is not served; try {INDEX_PATH}")

    def do_HEAD(self):                       # the stdlib's spelling, not ours
        self.do_GET()


class _ReusableServer(ThreadingHTTPServer):
    """Threaded, and able to rebind a fixed port that is still in TIME_WAIT."""

    allow_reuse_address = True
    daemon_threads = True


class ModelServer:
    """A running server, so a test can start one and know the port it got."""

    def __init__(self, directory: str | Path, *, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, base_url: str = "") -> None:
        self.catalogue = ModelCatalogue(directory)
        handler = type("_BoundHandler", (_Handler,),
                       {"catalogue": self.catalogue, "base_url": base_url})
        # Port 0 asks the OS for a free one, which is what the tests use. The reuse flag is
        # set on a SUBCLASS rather than on ThreadingHTTPServer itself: mutating the stdlib
        # class would change the behaviour of every other server in the process, including
        # the ones the dashboard's own tests stand up.
        self._httpd = _ReusableServer((host, port), handler)
        self.host, self.port = self._httpd.server_address[:2]
        self._thread: threading.Thread | None = None

    @property
    def index_url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        return f"http://{host}:{self.port}{INDEX_PATH}"

    def start(self) -> ModelServer:
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="model-server", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def __enter__(self) -> ModelServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a directory of MAPPO checkpoints as a Cloud AI source.")
    parser.add_argument("--models-dir",
                        default=str(Path(__file__).resolve().parent.parent / "policy" / "models"),
                        help="Directory of .npz checkpoints to serve. Read-only.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. Loopback by default; pass an address on the "
                             "demo LAN for a robot to reach it. THE ROBOT does the "
                             "fetching, so 127.0.0.1 is unreachable from one.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-url", default="",
                        help="Advertise absolute per-model URLs under this address. Only "
                             "needed when the files are reachable at a different address "
                             "from the index — a reverse proxy or a NAT hop. Left unset, "
                             "addresses are relative and the client resolves them.")
    parser.add_argument("--emit-sources", default="",
                        help="Write a robot_driver --model-sources file naming this server, "
                             "so the address is written once rather than typed twice.")
    parser.add_argument("--label", default="workstation",
                        help="How this source is named in the dashboard's picker.")
    parser.add_argument("--location", default="local model server",
                        help="Shown beside the label: WHERE this is, in an operator's terms.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(levelname)-7s  %(message)s")
    try:
        server = ModelServer(args.models_dir, host=args.host, port=args.port,
                             base_url=args.base_url)
    except (ModelServerError, OSError) as exc:
        log.error("cannot serve %s: %s", args.models_dir, exc)
        return 1

    entries = server.catalogue.entries()
    log.info("serving %d checkpoint(s) from %s", len(entries), server.catalogue.directory)
    for entry in entries:
        log.info("    %s  %d bytes  sha256:%s", entry["name"], entry["size_bytes"],
                 entry["sha256"][:12])
    log.info("index:  %s", server.index_url)
    if args.host == "127.0.0.1":
        log.warning("bound to loopback: a ROBOT cannot fetch from this address. The "
                    "download runs on the robot. Pass --host with a LAN address for that.")

    if args.emit_sources:
        default_model = ""
        if entries:
            default_model = urllib.parse.urljoin(server.index_url, entries[0]["url"])
        document = sources_document(server.index_url, label=args.label,
                                    location=args.location, default_model=default_model)
        try:
            Path(args.emit_sources).write_text(json.dumps(document, indent=2) + "\n")
        except OSError as exc:
            # A convenience that cannot be delivered must not take the server down with it:
            # the address is in the log above, and someone can still type it.
            log.error("could not write %s: %s — the index URL is above", args.emit_sources, exc)
        else:
            log.info("wrote %s — pass it as robot_driver.py --model-sources", args.emit_sources)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("interrupted; shutting down")
    finally:
        # serve_forever() has already returned by here, so shutdown() cannot deadlock on
        # its own thread. OSError covers a socket the interpreter is already tearing down.
        with contextlib.suppress(OSError):
            server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
