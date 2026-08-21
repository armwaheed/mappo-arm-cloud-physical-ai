#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Cloud AI fetch: what it will not download, and what it will not call it.

The HTTP tests run against a real ``http.server`` on loopback rather than a mocked
``urlopen``. A mock cannot fail the way a server fails — it cannot 404, it cannot redirect,
and it cannot send more bytes than it promised, which is the case the size ceiling exists
for. The S3 tests use a fake client object, because the alternative is a network and an
account.

Pure stdlib. ``python3 test_cloud_models.py``.
"""
from __future__ import annotations

import contextlib
import http.server
import os
import socket
import sys
import tempfile
import threading
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_models import (
    CloudFetchError,
    fetch,
    fetch_http,
    fetch_s3,
    list_s3,
    name_from_source,
    parse_source,
)
from model_store import ModelStoreError


# ── a real server on loopback ────────────────────────────────────────────────
class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves three paths: a small file, an unbounded body, and a 404."""

    protocol_version = "HTTP/1.0"   # so a response with no Content-Length is close-delimited

    payloads: ClassVar[dict] = {"/small.npz": b"x" * 4096}

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/liar.npz":
            # Sends 5 MiB with NO Content-Length, body delimited by close. This is the shape
            # that can actually overrun a client: urllib truncates a response at a declared
            # Content-Length, so a server that UNDER-states its size cannot overflow anyone.
            # A server that declares nothing can, and that is what the streamed ceiling is
            # for. The first version of this fixture under-stated and the test failed by
            # never reaching the ceiling at all — which is the fixture being wrong, not the
            # guard.
            self.send_response(200)
            self.end_headers()
            try:
                for _ in range(80):
                    self.wfile.write(b"y" * (64 * 1024))
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        body = self.payloads.get(self.path)
        if body is None:
            self.send_error(404, "no such object")
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


# ── scheme allow-list ────────────────────────────────────────────────────────
def test_a_file_url_is_refused():
    """``urlopen`` will happily read ``file:///etc/shadow`` and call it a download.

    The allow-list is the only thing between a text field on a web page and local file
    disclosure, so this is the single most important test in the file.
    """
    for hostile in ("file:///etc/passwd", "file://./secret.npz",
                    "ftp://host/x.npz", "gopher://host/x", "/etc/passwd"):
        try:
            parse_source(hostile)
        except CloudFetchError:
            continue
        raise AssertionError(f"{hostile!r} was accepted as a cloud source")


def test_http_can_be_required_to_be_https():
    parse_source("http://192.168.1.20:8000/a.npz", allow_http=True)
    try:
        parse_source("http://192.168.1.20:8000/a.npz", allow_http=False)
    except CloudFetchError as exc:
        assert "http" in str(exc)
        return
    raise AssertionError("plain http was allowed with allow_http=False")


def test_an_s3_uri_splits_into_bucket_and_key():
    assert parse_source("s3://bucket/a/b/c.npz") == ("s3", "bucket", "a/b/c.npz")
    for incomplete in ("s3://bucket", "s3:///key.npz", "s3://"):
        try:
            parse_source(incomplete)
        except CloudFetchError:
            continue
        raise AssertionError(f"{incomplete!r} was accepted")


# ── naming ───────────────────────────────────────────────────────────────────
def test_the_filename_comes_from_the_path_and_never_from_the_query():
    """A presigned URL carries its signature in the query string.

    A name built from one is neither a filename nor a secret that belongs in a listing.
    """
    name = name_from_source(
        "https://bucket.s3.amazonaws.com/ckpt/actor.npz?X-Amz-Signature=deadbeef&e=1")
    assert name == "actor.npz", name


def test_a_traversing_key_cannot_produce_a_name_that_escapes():
    """The property is that the result is a BARE filename, not that the call raises.

    Taking the last path segment is the defence, so ``https://h/../../etc/passwd.npz``
    correctly yields ``passwd.npz`` — a legal name that lands in the models directory like
    any other. An earlier version of this test asserted a refusal and failed, which was the
    test being wrong about what safety means here: a download named after someone else's
    file is harmless, and a download that WRITES to someone else's file is not.
    """
    for hostile in ("https://h/../../etc/passwd.npz", "s3://b/../../../root/x.npz",
                    "https://h/%2e%2e%2f%2e%2e%2fshadow.npz"):
        try:
            name = name_from_source(hostile)
        except (ModelStoreError, CloudFetchError):
            continue                              # a refusal is also a safe outcome
        assert "/" not in name and "\\" not in name, name
        assert not name.startswith("."), name
        assert Path(name).name == name, name


def test_an_explicit_name_is_still_checked():
    """An override is a filename too, and the field it comes from is on a web page."""
    try:
        name_from_source("https://h/a.npz", "../../evil.npz")
    except ModelStoreError:
        return
    raise AssertionError("an overriding name skipped the gate")


# ── downloads ────────────────────────────────────────────────────────────────
def test_a_download_that_lies_about_its_size_is_stopped_mid_stream():
    server, base = _serve()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "out.npz"
            try:
                fetch_http(f"{base}/liar.npz", destination, max_bytes=64 * 1024)
            except CloudFetchError as exc:
                assert "ceiling" in str(exc), str(exc)
                # It stopped early rather than writing 5 MiB and complaining afterwards.
                assert destination.stat().st_size <= 64 * 1024 + 65536
                return
            raise AssertionError("a 5 MiB body was accepted under a 64 KiB ceiling")
    finally:
        server.shutdown()


def test_a_404_is_reported_as_a_404():
    server, base = _serve()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                fetch_http(f"{base}/missing.npz", Path(tmp) / "out.npz")
            except CloudFetchError as exc:
                assert "404" in str(exc), str(exc)
                return
            raise AssertionError("a 404 was not reported")
    finally:
        server.shutdown()


def test_a_failed_fetch_leaves_no_temporary_file_behind():
    """A truncated file in /tmp is something a later retry could mistake for a cache."""
    server, base = _serve()
    before = set(Path(tempfile.gettempdir()).glob("mappo-model-*"))
    try:
        with contextlib.suppress(CloudFetchError):
            fetch(f"{base}/missing.npz")
        after = set(Path(tempfile.gettempdir()).glob("mappo-model-*"))
        assert after == before, sorted(p.name for p in after - before)
    finally:
        server.shutdown()


def test_a_successful_fetch_returns_the_bytes_and_the_name():
    server, base = _serve()
    try:
        path, name, written = fetch(f"{base}/small.npz")
        try:
            assert name == "small.npz", name
            assert written == 4096, written
            assert path.stat().st_size == 4096
        finally:
            path.unlink(missing_ok=True)
    finally:
        server.shutdown()


# ── S3, against a fake client ────────────────────────────────────────────────
class _FakeS3:
    def __init__(self, objects):
        self.objects = objects

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, Bucket, Key):
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, _name):
        objects = self.objects
        class _Paginator:
            def paginate(self, Bucket, Prefix=""):
                yield {"Contents": [
                    {"Key": k, "Size": len(v), "LastModified": None}
                    for k, v in objects.items() if k.startswith(Prefix)]}
        return _Paginator()


def test_an_oversized_object_is_refused_before_a_byte_moves():
    client = _FakeS3({"big.npz": b"z" * 5000})
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "out.npz"
        try:
            fetch_s3("bucket", "big.npz", destination, max_bytes=1000, client=client)
        except CloudFetchError as exc:
            assert "ceiling" in str(exc)
            assert not destination.exists() or destination.stat().st_size == 0
            return
        raise AssertionError("an oversized object was downloaded")


def test_a_bucket_listing_shows_only_checkpoints():
    """A training run's bucket is full of tensorboard logs nobody wants to scroll past."""
    client = _FakeS3({"a/actor.npz": b"x", "a/events.tfevents": b"y", "a/log.txt": b"z",
                      "b/other.npz": b"w"})
    listing = list_s3("bucket", "a/", client=client)
    assert [o["key"] for o in listing] == ["a/actor.npz"], listing
    assert listing[0]["uri"] == "s3://bucket/a/actor.npz"


def test_an_s3_failure_is_reported_rather_than_raised_raw():
    """botocore's exception tree is wide; a web request must not surface it as a 500."""
    class _Broken:
        def head_object(self, **kwargs):
            raise RuntimeError("NoCredentialsError: unable to locate credentials")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fetch_s3("b", "k.npz", Path(tmp) / "o.npz", client=_Broken())
        except CloudFetchError as exc:
            assert "credentials" in str(exc)
            return
        raise AssertionError("a raw botocore error escaped")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"cloud_models: {len(tests)}/{len(tests)} passed")
