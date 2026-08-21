#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pull a MAPPO checkpoint onto the robot from an S3 bucket or a server on the LAN.

Two sources, because the demo has two situations. A bucket is where a training run puts its
checkpoints and it can be listed, so the dashboard can show what is available rather than
asking someone to type a key. A direct address is what a workstation in the same room can
serve in one line of ``python3 -m http.server``, and on a demo floor with no AWS credentials
that is the difference between a working handover and a shrug.

## What this module refuses, and why each refusal is not paranoia

**A name it did not choose.** The filename comes from an S3 key or a URL path — input from
outside the robot — and it is used to build a path in a directory that sits next to a config
file. ``model_store.safe_model_name`` is the gate and it rejects rather than sanitises: a key
of ``../../../home/unitree/.ssh/authorized_keys`` is not a checkpoint with an unfortunate
name, and quietly rewriting it to ``authorized_keys`` would hide that someone tried.

**A file that is too big.** The download is bounded and enforced *while streaming*, not
checked afterwards from ``Content-Length``. A server that lies in that header, or omits it,
would otherwise fill a Jetson's SD card — and the delivered checkpoint is 262 KiB, so the
default 64 MiB ceiling is 250x headroom and still nowhere near a disk.

**A scheme that is not http, https or s3.** ``urllib`` will happily open ``file:///etc/shadow``
and hand it back as a "download". The allow-list is the only thing between a URL field on a
web page and local file disclosure.

**Anything that is not a MAPPO actor.** Enforced one layer up, in ``ModelStore.install``,
which inspects before the file lands. This module's job ends at bytes on local disk in a
temporary file; nothing here decides what is drivable.

## http, not just https

A "direct server IP address" on a demo LAN does not have a certificate, and requiring one
would mean the feature does not work in the room it was asked for. The transport is
therefore unauthenticated and this module does not pretend otherwise — what makes a
checkpoint trustworthy here is that its arrays are inspected before it can be armed and its
SHA-256 is reported, not that it arrived over TLS. Pass ``--allow-http false`` on the driver
to require https.

boto3 is an optional dependency. Its absence is reported as the one-line install it is, not
as an ImportError traceback from inside a web request.

Pure stdlib + optional boto3. ``python3 test_cloud_models.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from model_store import MODEL_SUFFIX, ModelStoreError, safe_model_name

#: Ceiling on a single download. The delivered checkpoint is 268 063 bytes; a 24-ray retrain
#: (issue #29) grows the first layer only, so nothing plausible approaches this.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024

#: Wall-clock ceiling on a single HTTP request. Long enough for a checkpoint over a slow
#: demo-floor link, short enough that a dead server does not wedge an RPC until it times out
#: somewhere else.
DEFAULT_TIMEOUT_S = 60.0

_HTTP_SCHEMES = ("http", "https")


class CloudFetchError(RuntimeError):
    """The checkpoint could not be brought down, and the message says at which step."""


def parse_source(uri: str, *, allow_http: bool = True) -> tuple:
    """Split a source into ``(kind, a, b)`` and reject anything unsupported.

    Returns ``("s3", bucket, key)`` or ``("http", url, "")``. Raising here rather than at
    the point of use keeps the scheme allow-list in exactly one place — the property that
    makes it worth having.
    """
    if not isinstance(uri, str) or not uri.strip():
        raise CloudFetchError("no source given")
    uri = uri.strip()
    parsed = urllib.parse.urlparse(uri)

    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise CloudFetchError(
                f"{uri!r} is not a complete S3 source; it must be s3://bucket/key")
        return ("s3", bucket, key)

    if parsed.scheme in _HTTP_SCHEMES:
        if parsed.scheme == "http" and not allow_http:
            raise CloudFetchError(
                f"{uri!r} is plain http and this driver was started with http disabled")
        if not parsed.netloc:
            raise CloudFetchError(f"{uri!r} has no host")
        return ("http", uri, "")

    raise CloudFetchError(
        f"{uri!r} uses scheme {parsed.scheme!r}; only s3://, https:// and http:// are "
        f"supported. A local path is not a cloud source — copy the file to the models "
        f"directory instead.")


def name_from_source(uri: str, override: str | None = None) -> str:
    """The checkpoint filename this source will install as.

    ``override`` wins when given, so an operator can rename a generically-named bucket
    object into something they will recognise in a list six months later. Either way the
    result goes through the same gate — an override is a filename too, and the field it
    comes from is on a web page.
    """
    if override:
        return safe_model_name(override)
    parsed = urllib.parse.urlparse(uri.strip())
    # Take the last path segment and nothing else. Query strings are how presigned S3 URLs
    # carry their signature, and a name built from one is neither a filename nor a secret
    # that belongs in a directory listing.
    candidate = parsed.path.rsplit("/", 1)[-1]
    if not candidate:
        raise ModelStoreError(
            f"cannot work out a filename from {uri!r}; pass an explicit name")
    return safe_model_name(candidate)


def _stream_to_file(response, destination: Path, max_bytes: int) -> int:
    """Copy a stream to disk, stopping the moment it exceeds ``max_bytes``.

    Counted as it arrives. ``Content-Length`` is a claim by the server and the whole point
    of the ceiling is to survive a server that is wrong about it, deliberately or not.
    """
    written = 0
    with open(destination, "wb") as handle:
        while True:
            block = response.read(1 << 16)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                raise CloudFetchError(
                    f"download exceeded the {max_bytes} byte ceiling; aborted mid-stream")
            handle.write(block)
    return written


def fetch_http(url: str, destination: Path, *, max_bytes: int = DEFAULT_MAX_BYTES,
               timeout: float = DEFAULT_TIMEOUT_S) -> int:
    """GET ``url`` into ``destination``. Returns the byte count."""
    request = urllib.request.Request(url, headers={"User-Agent": "mappo-dashboard"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # urlopen follows redirects, and a redirect can change the scheme. Re-check the
            # URL we actually landed on, or the allow-list above is advisory.
            final = getattr(response, "url", url)
            if urllib.parse.urlparse(final).scheme not in _HTTP_SCHEMES:
                raise CloudFetchError(
                    f"{url!r} redirected to {final!r}, which is not http or https")
            return _stream_to_file(response, destination, max_bytes)
    except urllib.error.HTTPError as exc:
        raise CloudFetchError(f"{url} returned HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise CloudFetchError(f"{url} could not be reached: {exc.reason}") from exc
    except TimeoutError as exc:
        raise CloudFetchError(f"{url} timed out after {timeout}s") from exc


def _s3_client():
    """A boto3 S3 client, or a CloudFetchError that says how to get one."""
    try:
        import boto3
    except ImportError as exc:
        raise CloudFetchError(
            "S3 sources need boto3, which is not installed in this environment: "
            "pip install boto3. A direct https:// or http:// address needs nothing extra."
        ) from exc
    endpoint = os.environ.get("MAPPO_S3_ENDPOINT_URL") or None
    return boto3.client("s3", endpoint_url=endpoint)


def fetch_s3(bucket: str, key: str, destination: Path, *,
             max_bytes: int = DEFAULT_MAX_BYTES, client=None) -> int:
    """Download ``s3://bucket/key`` into ``destination``. Returns the byte count.

    The size is checked with a ``head_object`` before any bytes move, because S3 will tell
    us for free and refusing a 4 GB object before downloading it is better than refusing it
    after. The streamed ceiling in :func:`_stream_to_file` still applies — head is an
    optimisation, not the guard.
    """
    client = client or _s3_client()
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        size = int(head.get("ContentLength", 0))
        if size > max_bytes:
            raise CloudFetchError(
                f"s3://{bucket}/{key} is {size} bytes, over the {max_bytes} byte ceiling")
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        return _stream_to_file(body, destination, max_bytes)
    except CloudFetchError:
        raise
    except Exception as exc:
        raise CloudFetchError(f"s3://{bucket}/{key} could not be fetched: {exc}") from exc


def list_s3(bucket: str, prefix: str = "", *, limit: int = 100, client=None) -> list:
    """The ``.npz`` objects under ``prefix``, newest first.

    This is what makes a bucket browsable instead of something you have to already know the
    contents of. Filtered to ``.npz`` on the client rather than asking the operator to read
    past a training run's tensorboard logs, and capped so a bucket with ten thousand
    checkpoints returns a page rather than a timeout.
    """
    client = client or _s3_client()
    try:
        paginator = client.get_paginator("list_objects_v2")
        found = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(MODEL_SUFFIX):
                    continue
                modified = obj.get("LastModified")
                found.append({
                    "key": obj["Key"],
                    "uri": f"s3://{bucket}/{obj['Key']}",
                    "size_bytes": int(obj.get("Size", 0)),
                    "last_modified": modified.isoformat() if modified else None,
                })
            if len(found) >= limit * 4:            # enough to sort meaningfully, then stop
                break
    except Exception as exc:
        raise CloudFetchError(f"could not list s3://{bucket}/{prefix}: {exc}") from exc

    found.sort(key=lambda o: (o["last_modified"] or "", o["key"]), reverse=True)
    return found[:limit]


def list_http_index(index_url: str, *, limit: int = 100,
                    timeout: float = DEFAULT_TIMEOUT_S, allow_http: bool = True) -> list:
    """The checkpoints a plain HTTP model server advertises, from a JSON index.

    This exists so a self-hosted model server is a FIRST-CLASS source rather than a URL you
    have to already know. Only S3 being browsable would say, in the shape of the UI, that a
    bucket is the real answer and anything else is a fallback — which is precisely backwards
    for a demo whose point is that the checkpoints can live on your own hardware.

    The index is a JSON object with a ``models`` array; each entry needs a ``name`` and may
    carry ``size_bytes``, ``sha256`` and ``modified``. A relative ``name`` resolves against
    the index URL, so a server does not have to know its own public address.
    """
    parse_source(index_url, allow_http=allow_http)      # same scheme allow-list, one place
    request = urllib.request.Request(index_url, headers={"User-Agent": "mappo-dashboard"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(1 << 20)                # an index is small; cap it anyway
            final = getattr(response, "url", index_url)
    except urllib.error.HTTPError as exc:
        raise CloudFetchError(f"{index_url} returned HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise CloudFetchError(f"{index_url} could not be reached: {exc.reason}") from exc
    except TimeoutError as exc:
        raise CloudFetchError(f"{index_url} timed out after {timeout}s") from exc

    try:
        index = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise CloudFetchError(
            f"{index_url} did not return a JSON index. A model server should serve "
            f'{{"models": [{{"name": "actor.npz", ...}}]}} at this address.') from exc
    entries = index.get("models") if isinstance(index, dict) else None
    if not isinstance(entries, list):
        raise CloudFetchError(f"{index_url}: the index has no 'models' array")

    found = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("key")
        if not isinstance(name, str) or not name.endswith(MODEL_SUFFIX):
            continue
        found.append({
            "key": name,
            # Resolved against the INDEX's final URL, so a server behind a redirect still
            # hands out addresses that work, and so it need not know its own hostname.
            "uri": urllib.parse.urljoin(final, entry.get("url") or name),
            "size_bytes": int(entry.get("size_bytes") or 0),
            "last_modified": entry.get("modified") or entry.get("last_modified"),
            "sha256": entry.get("sha256"),
            "served_by": index.get("server") if isinstance(index, dict) else None,
        })
    found.sort(key=lambda o: (o["last_modified"] or "", o["key"]), reverse=True)
    return found[:limit]


def fetch(uri: str, *, name: str | None = None, allow_http: bool = True,
          max_bytes: int = DEFAULT_MAX_BYTES, timeout: float = DEFAULT_TIMEOUT_S,
          s3_client=None) -> tuple:
    """Bring a checkpoint down to a temporary file. Returns ``(temp_path, name, bytes)``.

    The caller owns the temporary file and must move or delete it —
    ``ModelStore.install`` moves it, which is why this does not write into the models
    directory itself. A download that is never validated must never land somewhere a
    listing would show it.
    """
    kind, a, b = parse_source(uri, allow_http=allow_http)
    filename = name_from_source(uri if kind == "http" else f"s3://{a}/{b}", name)

    handle, temp_name = tempfile.mkstemp(prefix="mappo-model-", suffix=MODEL_SUFFIX)
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        if kind == "s3":
            written = fetch_s3(a, b, temp_path, max_bytes=max_bytes, client=s3_client)
        else:
            written = fetch_http(a, temp_path, max_bytes=max_bytes, timeout=timeout)
    except BaseException:
        # Includes CloudFetchError and anything the transport raises. A failed download must
        # not leave a truncated file in /tmp that a later retry could mistake for a cache.
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise
    return temp_path, filename, written


def free_bytes(path: str | Path) -> int:
    """Bytes free on the filesystem holding ``path``.

    Reported next to a download so the operator sees a Jetson filling up before the write
    fails, rather than after.
    """
    return shutil.disk_usage(Path(path)).free
