#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the local checkpoint server.

**These drive the real client, not a parallel reimplementation of it.** The server exists to
be read by ``cloud_models.list_http_index`` and fetched by ``cloud_models.fetch``, so the
tests start a real server on loopback and call those two functions — the same relationship
``test_peer_link.py`` has with ``integration/peer_source.py``, and for the same reason: a
writer and a reader that are only ever tested against their own idea of the format agree
right up until they are used together.

The end-to-end test goes one step further and hands the fetched bytes to
``model_store.inspect_model``, because "the transfer completed" and "what arrived is a
checkpoint this robot can run" are different claims and only the second one is worth making.

Pure stdlib + numpy (through ``model_store``). ``python3 test_model_server.py``.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_store
from cloud_models import fetch, list_http_index
from model_server import ModelCatalogue, ModelServer, ModelServerError, sources_document

HERE = Path(__file__).resolve().parent
DELIVERED = HERE.parent / "policy" / "models"


@contextlib.contextmanager
def _serving(directory, **kwargs):
    """A server on an OS-chosen port, always shut down."""
    server = ModelServer(directory, port=0, **kwargs)
    try:
        yield server.start()
    finally:
        server.stop()


def _get(url: str) -> tuple:
    """``(status, body)`` for a URL, without raising on a 404."""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# ── the end-to-end claim ─────────────────────────────────────────────────────
def test_the_delivered_checkpoint_survives_the_round_trip_and_is_still_drivable():
    """Index -> the real client -> the real fetch -> the store's own inspection.

    Fails if the index advertises an address the fetch cannot reach, if any byte is lost in
    transfer (the digest), or if what lands is not a MAPPO actor this adapter could drive.
    """
    with _serving(DELIVERED) as server:
        objects = list_http_index(server.index_url)
        assert len(objects) == 1, objects
        entry = objects[0]

        temp_path, name, written = fetch(entry["uri"])
        try:
            assert name == entry["key"], (name, entry["key"])
            assert written == entry["size_bytes"], (written, entry["size_bytes"])
            assert model_store.sha256_file(temp_path) == entry["sha256"]
            report = model_store.inspect_model(temp_path).as_dict()
            assert report["loadable"] is True, report["problems"]
            assert report["rays"] == model_store.N_RAYS, report
        finally:
            temp_path.unlink(missing_ok=True)


def test_the_index_carries_the_digest_the_store_computes():
    """A digest that is not the store's own is a number nobody can check anything against."""
    with _serving(DELIVERED) as server:
        entry = list_http_index(server.index_url)[0]
    on_disk = model_store.sha256_file(DELIVERED / entry["key"])
    assert entry["sha256"] == on_disk, (entry["sha256"], on_disk)


# ── the address seam ─────────────────────────────────────────────────────────
def test_addresses_are_relative_so_the_server_need_not_know_its_own_name():
    """Fetch the index as 'localhost'; the model URL must come back as 'localhost' too.

    This is the falsifier for a hard-coded base address. A server that wrote its own bind
    address into every entry would answer '127.0.0.1' here and the test would fail — which
    is exactly the failure a robot on the demo LAN would hit against a laptop-shaped URL.
    """
    with _serving(DELIVERED) as server:
        via_localhost = f"http://localhost:{server.port}/index.json"
        entry = list_http_index(via_localhost)[0]
    assert entry["uri"].startswith(f"http://localhost:{server.port}/models/"), entry["uri"]


def test_a_base_url_overrides_that_for_a_proxy_or_a_nat_hop():
    with _serving(DELIVERED, base_url="https://models.example.invalid/mappo") as server:
        entry = list_http_index(server.index_url)[0]
    assert entry["uri"] == (
        "https://models.example.invalid/mappo/models/mappo_actor_3agent_1910000.npz"
    ), entry["uri"]


def test_the_emitted_source_is_the_shape_the_driver_advertises():
    """``label`` and ``index_url`` are what ``_load_sources`` and the page's picker read.

    The round trip through ``robot_driver._load_sources`` itself lives in
    ``test_robot_driver.py``, because importing the driver needs the Device Connect edge
    package and this suite must run without it.
    """
    document = sources_document("http://10.0.0.5:8800/index.json", label="workstation",
                                location="the demo LAN", default_model="http://x/a.npz")
    assert list(document) == ["sources"], document
    source = document["sources"][0]
    assert source["label"] == "workstation"
    assert source["index_url"] == "http://10.0.0.5:8800/index.json"
    assert source["default_model"] == "http://x/a.npz"
    # An omitted default must be ABSENT, not empty: the page tests `source.default_model`
    # for truthiness, but a key that is present and empty reads as configured-to-nothing to
    # anyone looking at the file.
    assert "default_model" not in sources_document("http://x/i.json", label="l",
                                                   location="")["sources"][0]


# ── what is not advertised, and what is not served ───────────────────────────
def test_a_traversal_is_refused_however_it_is_spelled():
    with tempfile.TemporaryDirectory() as tmp:
        secret = Path(tmp) / "secret.txt"
        secret.write_text("not a checkpoint")
        served = Path(tmp) / "models"
        served.mkdir()
        (served / "real.npz").write_bytes(b"z" * 32)
        with _serving(served) as server:
            base = f"http://127.0.0.1:{server.port}"
            for suffix in ("../secret.txt", "..%2Fsecret.txt", "%2e%2e%2fsecret.txt",
                           "/etc/hosts", "sub/real.npz"):
                status, body = _get(f"{base}/models/{suffix}")
                assert status == 404, (suffix, status, body[:200])
            assert _get(f"{base}/models/real.npz")[0] == 200


def test_a_symlink_pointing_out_of_the_directory_is_not_served():
    """The name gate cannot see this one; the containment check is what catches it."""
    with tempfile.TemporaryDirectory() as tmp:
        outside = Path(tmp) / "outside.npz"
        outside.write_bytes(b"y" * 16)
        served = Path(tmp) / "models"
        served.mkdir()
        (served / "escape.npz").symlink_to(outside)
        with _serving(served) as server:
            status, _ = _get(f"http://127.0.0.1:{server.port}/models/escape.npz")
    assert status == 404, status


def test_a_name_the_store_would_refuse_is_never_advertised():
    """Advertising a name the robot must reject moves the failure two machines away."""
    with tempfile.TemporaryDirectory() as tmp:
        served = Path(tmp)
        (served / "good.npz").write_bytes(b"a" * 8)
        (served / ".hidden.npz").write_bytes(b"b" * 8)       # leading dot: refused
        (served / "notes.txt").write_bytes(b"c" * 8)         # not a checkpoint
        # The skip is logged, and the log line is the point of the design — but a suite that
        # prints a WARNING while passing reads as a suite that nearly failed.
        logging.getLogger("model_server").setLevel(logging.CRITICAL)
        try:
            names = [entry["name"] for entry in ModelCatalogue(served).entries()]
        finally:
            logging.getLogger("model_server").setLevel(logging.NOTSET)
    assert names == ["good.npz"], names


def test_a_badly_named_file_is_not_fetchable_even_though_it_is_right_there():
    """Not advertising it is not the same as not serving it.

    The listing and the fetch are separate code paths and an operator can type an address
    the listing never offered. A dotfile inside the served directory reaches the fetch
    handler with the containment check satisfied — it really is in the directory — so the
    name gate is the only thing that refuses it, and this is the case that proves the gate
    is not redundant with containment. Written after a mutation run removed that gate and
    every other test still passed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        served = Path(tmp)
        (served / ".hidden.npz").write_bytes(b"b" * 8)
        (served / "good.npz").write_bytes(b"a" * 8)
        with _serving(served) as server:
            base = f"http://127.0.0.1:{server.port}/models"
            assert _get(f"{base}/good.npz")[0] == 200
            status, _ = _get(f"{base}/.hidden.npz")
    assert status == 404, status


def test_an_unserved_path_is_a_json_404_that_names_the_index():
    with _serving(DELIVERED) as server:
        status, body = _get(f"http://127.0.0.1:{server.port}/nope")
    assert status == 404, status
    assert "index.json" in json.loads(body)["error"], body


def test_the_landing_page_names_every_checkpoint_it_serves():
    """Pasting the base address into a browser has to answer 'is it up, and with what'."""
    with _serving(DELIVERED) as server:
        status, body = _get(f"http://127.0.0.1:{server.port}/")
    assert status == 200, status
    assert b"mappo_actor_3agent_1910000.npz" in body, body


# ── the catalogue ────────────────────────────────────────────────────────────
def test_a_checkpoint_dropped_in_appears_without_a_restart():
    """A training run hands over a file, not a redeploy."""
    with tempfile.TemporaryDirectory() as tmp:
        served = Path(tmp)
        (served / "first.npz").write_bytes(b"a" * 8)
        catalogue = ModelCatalogue(served)
        assert [e["name"] for e in catalogue.entries()] == ["first.npz"]
        (served / "second.npz").write_bytes(b"b" * 8)
        assert sorted(e["name"] for e in catalogue.entries()) == ["first.npz", "second.npz"]


def test_the_digest_follows_the_bytes_and_not_the_filename():
    """The memo is keyed on size and mtime, so a rewrite in place must re-hash.

    Same name, same LENGTH, different contents — which is the case a cache keyed on the
    name alone, or on the name and size, gets wrong. A stale digest here would be reported
    to an operator as the identity of a file that is not the one on disk.
    """
    with tempfile.TemporaryDirectory() as tmp:
        served = Path(tmp)
        target = served / "actor.npz"
        target.write_bytes(b"A" * 64)
        catalogue = ModelCatalogue(served)
        before = catalogue.entries()[0]["sha256"]

        target.write_bytes(b"B" * 64)
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
        after = catalogue.entries()[0]["sha256"]
    assert before != after, "the digest did not change when the bytes did"
    assert after == model_store.hashlib.sha256(b"B" * 64).hexdigest()


def test_serving_a_directory_that_is_not_there_fails_at_startup():
    """Not at the first request, by which time a demo is already running."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ModelCatalogue(Path(tmp) / "absent")
        except ModelServerError as exc:
            assert "not a directory" in str(exc), exc
            return
    raise AssertionError("a missing directory was accepted")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"model_server: {len(tests)}/{len(tests)} passed")
