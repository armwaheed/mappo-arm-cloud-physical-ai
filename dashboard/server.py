#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The dashboard: discover the robots on the mesh, watch them, and drive them.

Runs on a workstation, not on the robot. It holds no robot code and no SDK — everything it
knows about a robot it learned by discovering it on the Device Connect mesh, and everything
it does it does through ``invoke_device``. Point it at a different quadruped and the buttons
still work, because the buttons are generated from the RPC schemas the device advertises.

## Why a separate page rather than the Device Connect portal

``device-connect-server`` ships a full multi-tenant portal with a devices page, a generic
invoke form and an event stream, and for fleet administration it is the right tool. It is
also generic by design: it renders a JSON form per function, which is the correct answer for
forty device types and the wrong one for a motion pad where the useful property is that
"strafe left" is one key press away from "stop". This page is demo-specific UI and lives in
the demo repository; the portal stays generic and stays where it is.

The two are not exclusive. This connects through ``device_connect_agent_tools.connect()``,
which speaks D2D or a router transparently, so a robot registered with a full server
appears here as readily as one announcing itself by multicast on a demo LAN.

## The threading, which is the only subtle part

``device-connect-agent-tools`` is a synchronous API and aiohttp is asynchronous, so every
call into the mesh goes through an executor. That executor has **exactly one worker** on
purpose. The connection holds subscription buffers that are appended to by a messaging
callback, and serialising every call through one thread removes the question of whether two
concurrent requests can interleave inside it. A dashboard makes a handful of calls a second;
there is no throughput to trade away.

## Events

One long-lived subscription to ``event(*)`` is drained on a timer into a ring buffer and
fanned out to every connected browser over Server-Sent Events. A page that opens late gets
the ring buffer first, so an operator who opens the tab after a run started still sees what
happened rather than an empty box.

    python3 server.py --port 8080

``python3 test_server.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

from aiohttp import web

log = logging.getLogger("mappo-dashboard")

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

#: How often the mesh subscription is drained. The subscription buffers between reads, so
#: this sets latency, not whether an event is seen.
EVENT_POLL_S = 0.4

#: Events kept for a page that connects late. Twenty minutes of periodic state at the
#: driver's 5 s interval is ~240 records, so this holds a session rather than a moment.
EVENT_RING = 500

#: Ceiling on how long one RPC may block the single mesh worker. Longer than the driver's own
#: worker timeout so that a slow robot surfaces as the driver's error rather than as this
#: one's, which is more specific.
INVOKE_TIMEOUT_S = 45.0

#: Functions this dashboard will invoke. An allow-list rather than a pass-through: this
#: endpoint takes a function name from a browser and calls it on a robot, and "whatever the
#: device advertises" is a wider door than a demo dashboard needs. Adding a capability to
#: the driver is a deliberate act; adding it here should be too.
ALLOWED_FUNCTIONS = frozenset({
    "get_status", "get_capabilities",
    "walk_forward", "walk_back", "strafe_left", "strafe_right",
    "turn_left", "turn_right", "lie_down", "stand", "stop",
    "list_models", "select_model", "download_model", "delete_model",
    "list_cloud_models",
})


class Mesh:
    """The dashboard's one connection to the Device Connect mesh.

    Every method is a coroutine that hands the real, synchronous call to a single worker
    thread. Nothing else in this file touches ``device_connect_agent_tools``.
    """

    def __init__(self, allow_insecure: bool = True) -> None:
        if allow_insecure:
            # Set before the first import-time connection so the SDK sees it. A demo LAN has
            # no PKI; a router deployment passes its own credentials and this is ignored.
            os.environ.setdefault("DEVICE_CONNECT_ALLOW_INSECURE", "true")
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mesh")
        self._subscription = None
        self.events: deque = deque(maxlen=EVENT_RING)
        self._listeners: set = set()
        self._seq = 0

    async def _call(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(self._pool, lambda: fn(*args, **kwargs)),
            timeout=INVOKE_TIMEOUT_S)

    async def start(self) -> None:
        from device_connect_agent_tools import connect
        from device_connect_agent_tools.tools import subscribe
        await self._call(connect)
        # One subscription for every event from every device. Filtering is the page's job:
        # an operator watching two robots wants both, and a dashboard that subscribed per
        # device would have to re-subscribe every time one joined.
        self._subscription = await self._call(subscribe, "event(*)")
        log.info("connected to the mesh; streaming events")

    async def close(self) -> None:
        if self._subscription is not None:
            try:
                await self._call(self._subscription.close)
            except Exception as exc:
                log.warning("closing the subscription failed: %r", exc)
        try:
            from device_connect_agent_tools import disconnect
            await self._call(disconnect)
        except Exception as exc:
            log.warning("disconnect failed: %r", exc)
        self._pool.shutdown(wait=False)

    async def devices(self) -> list:
        from device_connect_agent_tools.tools import discover
        result = await self._call(discover, "device(*)")
        return result.get("results", [])

    async def invoke(self, device_id: str, function: str, params: dict) -> dict:
        from device_connect_agent_tools.tools import invoke_device
        return await self._call(invoke_device, device_id, function, params)

    # ── event fan-out ────────────────────────────────────────────────────────
    def listen(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._listeners.add(queue)
        return queue

    def unlisten(self, queue: asyncio.Queue) -> None:
        self._listeners.discard(queue)

    async def pump(self) -> None:
        """Drain the subscription into the ring buffer and out to every open page."""
        while True:
            try:
                messages = await self._call(self._subscription.read)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("event read failed: %r", exc)
                await asyncio.sleep(2.0)
                continue
            for message in _in_time_order(messages):
                self._seq += 1
                record = _shape_event(message, self._seq)
                self.events.append(record)
                for queue in list(self._listeners):
                    # A page that has stopped reading has gone away or is in a background
                    # tab. Dropping is correct: the alternative is this pump blocking and
                    # every OTHER page stalling with it.
                    with contextlib.suppress(asyncio.QueueFull):
                        queue.put_nowait(record)
            await asyncio.sleep(EVENT_POLL_S)


def _in_time_order(messages: list) -> list:
    """Sort one drained batch by the emitting device's timestamp.

    A ``Subscription`` keeps one inbox per subject and drains them one after another, and
    each event NAME is its own subject — so a batch arrives grouped by event type, not by
    time. Left alone, the operator's log shows a run's three ``motion_completed`` lines
    above the three ``motion_started`` lines that caused them, which reads as though the
    robot finished before it began. Measured, not theorised: it is visible in
    ``evidence/2026-08-21-device-connect-dashboard/``.

    ``ts`` is stamped by the device at emit and has ONE-SECOND resolution, so this fixes
    ordering across seconds and cannot fix it within one. The sort is stable, so events
    sharing a timestamp keep their arrival order rather than being shuffled. Two events a
    few hundred milliseconds apart may still appear in either order; two events a second
    apart will not.
    """
    def key(message):
        params = message.get("params") or {}
        # A missing ts sorts first rather than crashing the pump. An event with no timestamp
        # is a malformed one, and losing the whole batch over it would be worse.
        return str(params.get("ts") or "")
    return sorted(messages, key=key)


def _shape_event(message: dict, seq: int) -> dict:
    """Turn one mesh message into the record the page renders.

    The device id comes from the subject rather than the payload. The payload is whatever
    the emitting device chose to send and does not have to identify it; the subject is
    assigned by the transport and always does.
    """
    subject = message.get("_subject", "")
    parts = subject.replace("/", ".").split(".")
    device_id = parts[2] if len(parts) > 4 else ""
    params = message.get("params") or {}
    return {
        "seq": seq,
        "received": time.time(),
        "device_id": device_id,
        "event": message.get("method") or (parts[-1] if parts else "event"),
        "ts": params.get("ts"),
        "payload": {k: v for k, v in params.items()
                    if k not in ("event_id", "ts", "_trace_id")},
    }


# ── routes ────────────────────────────────────────────────────────────────────────────────
async def index(request: web.Request) -> web.Response:
    return web.FileResponse(TEMPLATE_DIR / "index.html")


async def api_devices(request: web.Request) -> web.Response:
    mesh: Mesh = request.app["mesh"]
    try:
        devices = await mesh.devices()
    except asyncio.TimeoutError:
        return web.json_response({"ok": False, "error": "the mesh did not answer in time"},
                                 status=504)
    return web.json_response({"ok": True, "devices": devices})


async def api_invoke(request: web.Request) -> web.Response:
    mesh: Mesh = request.app["mesh"]
    try:
        body = await request.json()
    except ValueError:
        return web.json_response({"ok": False, "error": "body must be JSON"}, status=400)

    device_id = body.get("device_id")
    function = body.get("function")
    params = body.get("params") or {}
    if not device_id or not function:
        return web.json_response(
            {"ok": False, "error": "device_id and function are both required"}, status=400)
    if function not in ALLOWED_FUNCTIONS:
        return web.json_response(
            {"ok": False, "error": f"{function!r} is not one of this dashboard's functions"},
            status=403)
    if not isinstance(params, dict):
        return web.json_response({"ok": False, "error": "params must be an object"},
                                 status=400)

    try:
        result = await mesh.invoke(device_id, function, params)
    except asyncio.TimeoutError:
        return web.json_response(
            {"ok": False, "error": f"{function} did not return within {INVOKE_TIMEOUT_S}s. "
                                   f"If it was a motion command, check the robot before "
                                   f"commanding it again."}, status=504)
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                                 status=502)
    return web.json_response({"ok": True, "result": result})


async def api_events(request: web.Request) -> web.StreamResponse:
    """Server-Sent Events: the backlog, then everything new."""
    mesh: Mesh = request.app["mesh"]
    response = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        # This dashboard is commonly reached through an SSH tunnel or a reverse proxy, and a
        # proxy that buffers turns a live stream into a page that updates every 4 KB.
        "X-Accel-Buffering": "no",
    })
    await response.prepare(request)

    queue = mesh.listen()
    try:
        for record in list(mesh.events):
            await _send_event(response, record)
        while True:
            try:
                record = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # A comment frame keeps the connection open through proxies that time out an
                # idle stream, and lets the browser notice a dead server.
                await response.write(b": keep-alive\n\n")
                continue
            await _send_event(response, record)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        mesh.unlisten(queue)
    return response


async def _send_event(response: web.StreamResponse, record: dict) -> None:
    await response.write(f"data: {json.dumps(record)}\n\n".encode())


def create_app(allow_insecure: bool = True) -> web.Application:
    app = web.Application()
    app["mesh"] = Mesh(allow_insecure=allow_insecure)
    app.router.add_get("/", index)
    app.router.add_get("/api/devices", api_devices)
    app.router.add_post("/api/invoke", api_invoke)
    app.router.add_get("/api/events", api_events)
    app.router.add_static("/static", STATIC_DIR, name="static")
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


async def _on_startup(app: web.Application) -> None:
    mesh: Mesh = app["mesh"]
    await mesh.start()
    app["pump"] = asyncio.create_task(mesh.pump())


async def _on_cleanup(app: web.Application) -> None:
    task = app.get("pump")
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await app["mesh"].close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MAPPO Device Connect dashboard.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. The default is loopback only; pass 0.0.0.0 to "
                             "reach it from another machine on the demo LAN.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--require-tls", action="store_true",
                        help="Do not set DEVICE_CONNECT_ALLOW_INSECURE for the mesh.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(levelname)-7s  %(message)s")
    web.run_app(create_app(allow_insecure=not args.require_tls),
                host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
