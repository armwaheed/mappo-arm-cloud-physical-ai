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

## The threading — three lanes, and the reason is a safety measurement

``device-connect-agent-tools`` is a synchronous API and aiohttp is asynchronous, so every
call into the mesh goes through an executor.

**This was one executor with one worker, and that was a safety defect.** The reasoning
written here was "a dashboard makes a handful of calls a second; there is no throughput to
trade away". That is true for one robot and false for two. Measured, with robot A one second
into a five-second walk:

| call | returned after |
| --- | --- |
| STOP to a **different** robot | **4.23 s** |
| the 5 s walk itself | 5.17 s |

4.23 s is the walk's remaining time. The stop was not slow — it was never *sent* until the
walk finished, because it was queued behind it in a single-worker pool. The driver
deliberately puts ``stop()`` outside its own motion lock so a stop never waits; queueing it
one layer up defeated exactly that.

The premise was also wrong. ``DeviceConnection`` owns a dedicated event loop on its own
thread and bridges with ``asyncio.run_coroutine_threadsafe``; each request carries its own
id and the loop multiplexes them. Concurrent callers are what it is built for.

So: three lanes, sized by what must not be blocked rather than by throughput.

* ``_pool`` (8) — everything ordinary: discovery, invokes, checkpoint operations.
* ``_stop_pool`` (2) — **stop and stop-all only.** Even if every general worker is wedged on
  a hung call, a stop still has a thread. ``DeviceConnection._run`` calls ``future.result()``
  with no timeout, so a wedged mesh call holds its worker forever; a dedicated lane is the
  only way to actually guarantee the stop path rather than hope for it.
* ``_event_pool`` (1) — the subscription drain. Serialised because one reader is correct for
  it, and separate so the operator's live view cannot go dark at the moment the fleet is
  busiest.

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

#: Ceiling on how long one RPC may block. Longer than the driver's own worker timeout so that
#: a slow robot surfaces as the driver's error rather than as this one's, which is more
#: specific.
INVOKE_TIMEOUT_S = 45.0

#: Ceiling on a stop. Deliberately short: a stop that has not landed in this long is a stop
#: the operator needs to know has NOT landed, so they can reach for the physical abort
#: instead of watching a spinner.
STOP_TIMEOUT_S = 12.0

#: General-purpose mesh workers. Enough that several robots can be commanded at once without
#: queueing; the cap exists because each wedged call holds its thread forever.
MESH_WORKERS = 8

#: Camera frames are drained on their own subscription and their own thread. A burst of
#: image data must never delay a motion_refused reaching the operator's screen — the same
#: lane argument as the stop pool, applied to the event side.
CAMERA_POLL_S = 0.05

#: How long a device stays listed after it stops announcing itself. D2D presence is
#: ephemeral, so a robot that dies simply vanishes from discovery — and a robot dropping off
#: the mesh mid-walk is the single event an operator must not have hidden from them. It is
#: shown as GONE with its last-seen age until this expires.
TOMBSTONE_S = 120.0

#: Functions this dashboard will invoke. An allow-list rather than a pass-through: this
#: endpoint takes a function name from a browser and calls it on a robot, and "whatever the
#: device advertises" is a wider door than a demo dashboard needs. Adding a capability to
#: the driver is a deliberate act; adding it here should be too.
ALLOWED_FUNCTIONS = frozenset({
    "get_status", "get_capabilities",
    "walk_forward", "walk_back", "strafe_left", "strafe_right",
    "turn_left", "turn_right", "lie_down", "stand", "stop",
    "list_models", "select_model", "download_model", "delete_model",
    "list_cloud_models", "watch_camera", "stop_camera",
})


class Mesh:
    """The dashboard's one connection to the Device Connect mesh.

    Every method is a coroutine that hands the real, synchronous call to a worker thread.
    Nothing else in this file touches ``device_connect_agent_tools``. See the module
    docstring for why there are three pools and not one.
    """

    def __init__(self, allow_insecure: bool = True) -> None:
        if allow_insecure:
            # Set before the first import-time connection so the SDK sees it. A demo LAN has
            # no PKI; a router deployment passes its own credentials and this is ignored.
            os.environ.setdefault("DEVICE_CONNECT_ALLOW_INSECURE", "true")
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=MESH_WORKERS, thread_name_prefix="mesh")
        self._stop_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="mesh-stop")
        self._event_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mesh-events")
        self._subscription = None
        self.events: deque = deque(maxlen=EVENT_RING)
        self._listeners: set = set()
        self._seq = 0
        #: Everything the dashboard knows about every robot it has seen, keyed by device id.
        #: Populated by discovery AND by the event stream, so a fleet of N robots costs N
        #: capability lookups once rather than N status polls per refresh.
        self._fleet: dict = {}
        self._camera_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mesh-camera")
        self._camera_sub = None
        #: Latest JPEG per device. One frame deep on purpose: a viewport wants the newest
        #: frame, and a queue of stale ones is latency with extra steps.
        self._frames: dict = {}

    async def _call(self, fn, *args, _pool=None, _timeout=INVOKE_TIMEOUT_S, **kwargs):
        """Hand one synchronous mesh call to a worker thread.

        The wrapper's own knobs are underscore-prefixed so they cannot collide with a
        callee's keyword arguments. Without that, ``_call(invoke_many, ..., timeout=12)``
        silently applies the 12 s to the WRAPPER and leaves ``invoke_many`` on its own
        30 s default — a wrapper that quietly eats a parameter meant for the thing it wraps.
        """
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(_pool or self._pool, lambda: fn(*args, **kwargs)),
            timeout=_timeout)

    async def start(self) -> None:
        from device_connect_agent_tools import connect
        from device_connect_agent_tools.tools import subscribe
        await self._call(connect)
        # One subscription for every event from every device. Filtering is the page's job:
        # an operator watching two robots wants both, and a dashboard that subscribed per
        # device would have to re-subscribe every time one joined.
        self._subscription = await self._call(subscribe, "event(*)", _pool=self._event_pool)
        self._camera_sub = await self._call(subscribe, "event(camera_frame)",
                                            _pool=self._camera_pool)
        log.info("connected to the mesh; streaming events")

    async def close(self) -> None:
        if self._subscription is not None:
            try:
                await self._call(self._subscription.close, _pool=self._event_pool)
            except Exception as exc:
                log.warning("closing the subscription failed: %r", exc)
        try:
            from device_connect_agent_tools import disconnect
            await self._call(disconnect)
        except Exception as exc:
            log.warning("disconnect failed: %r", exc)
        for pool in (self._pool, self._stop_pool, self._event_pool, self._camera_pool):
            pool.shutdown(wait=False)

    async def devices(self) -> list:
        from device_connect_agent_tools.tools import discover
        result = await self._call(discover, "device(*)")
        return result.get("results", [])

    async def invoke(self, device_id: str, function: str, params: dict) -> dict:
        from device_connect_agent_tools.tools import invoke_device
        # A stop rides the dedicated lane wherever it is issued from, including the generic
        # invoke endpoint the motion pad uses. Routing by function name rather than by
        # endpoint means there is one rule and no way to reach stop by a path that queues.
        if function == "stop":
            return await self._call(invoke_device, device_id, function, params,
                                    _pool=self._stop_pool, _timeout=STOP_TIMEOUT_S)
        return await self._call(invoke_device, device_id, function, params)

    async def stop_all(self, device_ids=None) -> dict:
        """Stop robots and report what each one actually said.

        ``device_ids`` narrows it to a named set — a platform group, say. ``None`` means the
        whole mesh. One implementation for both so a group stop and a fleet stop can never
        report in different shapes.

        ``invoke_many`` rather than ``broadcast``. Broadcast returns immediately with a
        correlation id and the replies arrive separately, which is the right shape for a
        fan-out whose outcome is advisory. A stop's outcome is not advisory: an operator
        pressing this needs to know which robots confirmed and which did not, because the
        ones that did not are the ones they must now deal with physically. ``invoke_many``
        blocks on the slowest device and hands back per-device results, with its own
        concurrency internally so the robots are not stopped one after another.
        """
        from device_connect_agent_tools.tools import invoke_device, invoke_many

        if device_ids:
            # A named set. Fired concurrently on the stop lane rather than through a
            # selector, because the selector language matches on device id patterns and a
            # platform group is not a naming convention — inferring one would break the
            # first time somebody named a robot after the room it is in.
            results = await asyncio.gather(*(
                self._call(invoke_device, device_id, "stop", {},
                           _pool=self._stop_pool, _timeout=STOP_TIMEOUT_S)
                for device_id in device_ids), return_exceptions=True)
            rows = []
            for device_id, outcome in zip(device_ids, results):
                if isinstance(outcome, Exception):
                    rows.append({"device_id": device_id,
                                 "result": {"ok": False, "error": repr(outcome)}})
                else:
                    rows.append({"device_id": device_id,
                                 "result": (outcome or {}).get("result", outcome)})
            return {"matched": len(device_ids), "results": rows}

        return await self._call(
            invoke_many, "device(*).function(stop)", {},
            timeout=STOP_TIMEOUT_S,          # invoke_many's own per-device ceiling
            _pool=self._stop_pool,
            # The wrapper's ceiling sits ABOVE invoke_many's so that a partial result — some
            # robots confirmed, some not — comes back and is shown, instead of this layer
            # timing out first and reporting nothing about any of them.
            _timeout=STOP_TIMEOUT_S + 3.0)

    # ── camera ───────────────────────────────────────────────────────────────
    async def camera_pump(self) -> None:
        """Drain camera frames into a one-deep buffer per device.

        Its own subscription and its own thread. Sharing the control drain would mean a
        burst of image data delaying the event an operator is actually waiting for.
        """
        import base64
        while True:
            try:
                messages = await self._call(self._camera_sub.read, _pool=self._camera_pool,
                                            _timeout=10.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("camera read failed: %r", exc)
                await asyncio.sleep(2.0)
                continue
            for message in messages:
                params = message.get("params") or {}
                encoded = params.get("jpeg_b64")
                if not encoded:
                    continue
                subject = message.get("_subject", "")
                parts = subject.replace("/", ".").split(".")
                device_id = parts[2] if len(parts) > 4 else ""
                try:
                    self._frames[device_id] = base64.b64decode(encoded)
                except (ValueError, TypeError):
                    continue                     # a malformed frame costs that frame only
            await asyncio.sleep(CAMERA_POLL_S)

    def latest_frame(self, device_id: str):
        return self._frames.get(device_id)

    async def keep_camera_alive(self, device_id: str, fps: float) -> dict:
        """Re-assert a viewer's interest. The driver stops streaming when it lapses."""
        return await self.invoke(device_id, "watch_camera", {"fps": fps})

    # ── the fleet ────────────────────────────────────────────────────────────
    async def fleet(self) -> list:
        """Every robot the dashboard has seen, present or departed.

        Discovery gives the roster. Capabilities are fetched once per device and cached —
        with N robots on the page, re-asking each of them what it can do on every 10 s poll
        would be N calls a poll for an answer that does not change while a driver is up.
        Pose, mode and armed checkpoint come from the ``robot_state`` events already
        streaming through the pump, so the table stays live at **zero** extra RPCs.
        """
        now = time.time()
        try:
            present = await self.devices()
        except Exception as exc:
            log.warning("fleet discovery failed: %r", exc)
            present = []

        seen = set()
        for device in present:
            device_id = device.get("device_id")
            if not device_id:
                continue
            seen.add(device_id)
            row = self._fleet.setdefault(device_id, {"device_id": device_id})
            row["device_type"] = device.get("device_type")
            row["present"] = True
            row["last_seen"] = now

        for device_id, row in self._fleet.items():
            if device_id not in seen:
                # Departed. Kept, not deleted — see TOMBSTONE_S.
                row["present"] = False

        await self._fill_capabilities(seen)

        rows = [row for row in self._fleet.values()
                if row.get("present") or now - row.get("last_seen", 0) < TOMBSTONE_S]
        # Drop anything past the tombstone window so a long session does not accumulate
        # every robot that has ever appeared.
        self._fleet = {row["device_id"]: row for row in rows}
        for row in rows:
            row["age_s"] = round(now - row.get("last_seen", now), 1)
        return sorted(rows, key=lambda r: (not r.get("present"), r["device_id"]))

    async def _fill_capabilities(self, device_ids) -> None:
        """Ask each robot what it can do, once, and cache the answer.

        Concurrently — this is the case the general pool was widened for. Cached because
        with N robots on the page, re-asking every one of them on every 10 s poll is N calls
        a poll for an answer that cannot change while a driver is up: it is fixed at startup
        by ``--allow-motion`` and the platform.

        A failure is skipped rather than cached, so a robot that was mid-restart is asked
        again on the next poll instead of being stuck without capabilities forever.
        """
        missing = [d for d in device_ids if not self._fleet[d].get("capabilities")]
        if not missing:
            return
        results = await asyncio.gather(
            *(self.invoke(d, "get_capabilities", {}) for d in missing),
            return_exceptions=True)
        for device_id, result in zip(missing, results):
            if isinstance(result, Exception):
                continue
            caps = (result or {}).get("result")
            if isinstance(caps, dict) and caps.get("platform"):
                self._fleet[device_id]["capabilities"] = caps

    def note_state_event(self, record: dict) -> None:
        """Fold a ``robot_state`` event into the fleet table.

        This is what makes the fleet view free. Every driver already emits pose, mode and
        armed checkpoint on a timer; reading them out of the stream the page is already
        subscribed to means adding a robot costs no additional polling.
        """
        if record.get("event") != "robot_state":
            return
        device_id = record.get("device_id")
        if not device_id:
            return
        payload = record.get("payload") or {}
        row = self._fleet.setdefault(device_id, {"device_id": device_id})
        row["pose"] = payload.get("pose")
        row["mode"] = payload.get("mode")
        row["active_model"] = payload.get("active_model")
        row["present"] = True
        row["last_seen"] = time.time()

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
                messages = await self._call(self._subscription.read, _pool=self._event_pool)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("event read failed: %r", exc)
                await asyncio.sleep(2.0)
                continue
            for message in _in_time_order(messages):
                if message.get("method") == "camera_frame":
                    continue        # drained separately; a frame is not a log line
                self._seq += 1
                record = _shape_event(message, self._seq)
                self.note_state_event(record)
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


async def api_fleet(request: web.Request) -> web.Response:
    """Every robot the dashboard has seen — the view the controls are built on."""
    mesh: Mesh = request.app["mesh"]
    try:
        rows = await mesh.fleet()
    except asyncio.TimeoutError:
        return web.json_response({"ok": False, "error": "the mesh did not answer in time"},
                                 status=504)
    return web.json_response({"ok": True, "fleet": rows, "tombstone_s": TOMBSTONE_S})


async def api_stop_all(request: web.Request) -> web.Response:
    """Stop every robot, and report per robot whether it confirmed.

    Never returns a bare success. An operator pressing this has to know WHICH robots
    acknowledged, because the ones that did not are the ones they now have to walk over to.
    """
    mesh: Mesh = request.app["mesh"]
    device_ids = None
    if request.can_read_body:
        try:
            body = await request.json()
            requested = (body or {}).get("device_ids")
            if isinstance(requested, list) and requested:
                device_ids = [str(d) for d in requested]
        except ValueError:
            pass                            # no body, or not JSON: stop everything
    try:
        result = await mesh.stop_all(device_ids)
    except asyncio.TimeoutError:
        return web.json_response(
            {"ok": False, "error": "STOP ALL did not complete in time. Assume one or more "
                                   "robots are STILL MOVING and use the physical abort."},
            status=504)
    except Exception as exc:
        return web.json_response(
            {"ok": False, "error": f"STOP ALL failed: {type(exc).__name__}: {exc}. Assume "
                                   f"robots are STILL MOVING and use the physical abort."},
            status=502)

    stopped, failed = [], []
    for row in (result.get("results") or []):
        device_id = row.get("device_id")
        inner = row.get("result")
        if isinstance(inner, dict) and inner.get("ok"):
            stopped.append(device_id)
        else:
            reason = (inner or {}).get("error") if isinstance(inner, dict) else str(inner)
            failed.append({"device_id": device_id, "error": reason or "no acknowledgement"})
    for row in (result.get("errors") or []):
        failed.append({"device_id": row.get("device_id"), "error": str(row)})

    return web.json_response({"ok": True, "stopped": stopped, "failed": failed,
                              "matched": result.get("matched", 0)})


async def api_camera(request: web.Request) -> web.StreamResponse:
    """An MJPEG stream of one robot's front camera.

    ``multipart/x-mixed-replace`` because a browser renders it natively in a plain ``<img>``:
    no canvas, no per-frame JavaScript, and the decode happens off the main thread. It also
    means the viewport degrades to a broken-image icon rather than to a silently frozen last
    frame, which is the failure mode that matters for something an operator uses to decide
    where a robot is pointing.

    Requesting the stream IS the statement of interest — the driver stops emitting when it
    lapses — so simply closing the tab stops the robot streaming.
    """
    mesh: Mesh = request.app["mesh"]
    device_id = request.match_info["device_id"]
    fps = _clamp_query_fps(request.query.get("fps"))

    boundary = "mappoframe"
    response = web.StreamResponse(headers={
        "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "X-Accel-Buffering": "no",
    })
    await response.prepare(request)

    last_keepalive = 0.0
    last_sent = None
    try:
        while True:
            now = time.monotonic()
            # Re-assert interest well inside the driver's expiry rather than at the edge of
            # it, or the feed stutters every time a keepalive lands late.
            if now - last_keepalive > 4.0:
                last_keepalive = now
                try:
                    await mesh.keep_camera_alive(device_id, fps)
                except Exception as exc:
                    log.warning("camera keepalive for %s failed: %r", device_id, exc)

            frame = mesh.latest_frame(device_id)
            if frame is not None and frame is not last_sent:
                last_sent = frame
                await response.write(
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n".encode() + frame + b"\r\n")
            await asyncio.sleep(1.0 / max(fps, 1.0))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as exc:
        log.warning("camera stream for %s ended: %r", device_id, exc)
    return response


def _clamp_query_fps(value) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return 6.0
    return max(1.0, min(fps, 15.0))


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
    app.router.add_get("/api/fleet", api_fleet)
    app.router.add_post("/api/stop-all", api_stop_all)
    app.router.add_get("/api/camera/{device_id}", api_camera)
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
    app["camera_pump"] = asyncio.create_task(mesh.camera_pump())


async def _on_cleanup(app: web.Application) -> None:
    for name in ("pump", "camera_pump"):
        task = app.get(name)
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
