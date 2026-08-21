#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dashboard's HTTP surface, with a fake mesh in place of a real one.

The mesh is faked here on purpose. What these tests are about is the layer between a browser
and ``invoke_device`` — the allow-list, the error shapes, the event fan-out — and a real
Zenoh session would only add latency and a reason for this file to be flaky. The real mesh is
exercised end to end by the run recorded in the pull request, not here.

The event-shaping test is the one carrying a real decision: the device id comes from the
subject and never from the payload, because a device chooses its payload and the transport
assigns the subject.

Needs aiohttp. ``python3 test_server.py``.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import deque

from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server as dashboard
from server import ALLOWED_FUNCTIONS, Mesh, _in_time_order, _shape_event


class _FakeMesh(Mesh):
    """A Mesh that answers from memory. Never touches Zenoh, never starts a thread pool."""

    def __init__(self, devices=None, answer=None):
        self._devices = devices if devices is not None else [
            {"device_id": "mappo-go2", "device_type": "mappo_quadruped"}]
        self._answer = answer or {"success": True, "result": {"ok": True}}
        self.calls = []
        self.events = deque(maxlen=500)
        self._listeners = set()
        self._seq = 0
        self._subscription = None

    async def start(self):
        return None

    async def close(self):
        return None

    async def devices(self):
        return list(self._devices)

    async def invoke(self, device_id, function, params):
        self.calls.append((device_id, function, params))
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer

    async def pump(self):
        await asyncio.sleep(3600)


def _app(mesh):
    app = dashboard.create_app()
    app["mesh"] = mesh
    return app


def _serve(coro_fn, mesh=None):
    """Run one coroutine against a live test server holding ``mesh``."""
    mesh = mesh or _FakeMesh()

    async def main():
        async with TestClient(TestServer(_app(mesh))) as client:
            return await coro_fn(client, mesh)

    return asyncio.run(main())


# ── the allow-list ───────────────────────────────────────────────────────────
def test_a_function_outside_the_allow_list_is_refused_and_never_reaches_the_mesh():
    """This endpoint takes a function NAME from a browser and calls it on a robot.

    "Whatever the device advertises" is a wider door than a demo dashboard needs, and the
    second assertion is the one that matters: refused means not forwarded, not merely
    reported.
    """
    async def check(client, mesh):
        for hostile in ("__init__", "connect", "disconnect", "shutdown", "eval"):
            response = await client.post("/api/invoke", json={
                "device_id": "mappo-go2", "function": hostile, "params": {}})
            assert response.status == 403, (hostile, response.status)
            body = await response.json()
            assert body["ok"] is False and hostile in body["error"]
        assert mesh.calls == [], f"a blocked function still reached the mesh: {mesh.calls}"
    _serve(check)


def test_every_allowed_function_is_one_the_driver_actually_has():
    """A stale allow-list entry is a button that 403s or a capability nobody can reach.

    Read from the driver class rather than from a second hand-written list, so adding an
    ``@rpc`` and forgetting the dashboard shows up here.
    """
    try:
        from robot_driver import MappoRobotDriver
    except ImportError as exc:
        print(f"  skip  driver not importable ({exc})")
        return
    unknown = sorted(ALLOWED_FUNCTIONS - set(vars(MappoRobotDriver)))
    assert not unknown, f"the dashboard allows {unknown}, which the driver does not define"


# ── request shapes ───────────────────────────────────────────────────────────
def test_a_malformed_request_is_a_400_not_a_500():
    async def check(client, _mesh):
        response = await client.post("/api/invoke", data=b"not json",
                                     headers={"Content-Type": "application/json"})
        assert response.status == 400, response.status

        response = await client.post("/api/invoke", json={"device_id": "d"})
        assert response.status == 400
        assert "function" in (await response.json())["error"]

        response = await client.post("/api/invoke", json={
            "device_id": "d", "function": "get_status", "params": "not-an-object"})
        assert response.status == 400
        assert "object" in (await response.json())["error"]
    _serve(check)


def test_a_slow_robot_is_a_504_that_says_to_check_it():
    """A motion command that timed out may have left a robot moving.

    The message says so, because the operator's next action should be to look at the robot
    rather than to press the button again.
    """
    async def check(client, _mesh):
        response = await client.post("/api/invoke", json={
            "device_id": "d", "function": "walk_forward", "params": {}})
        assert response.status == 504, response.status
        body = await response.json()
        assert "check the robot" in body["error"], body["error"]
    _serve(check, mesh=_FakeMesh(answer=asyncio.TimeoutError()))


def test_a_mesh_failure_is_a_502_carrying_the_reason():
    async def check(client, _mesh):
        response = await client.post("/api/invoke", json={
            "device_id": "d", "function": "get_status", "params": {}})
        assert response.status == 502, response.status
        assert "the router went away" in (await response.json())["error"]
    _serve(check, mesh=_FakeMesh(answer=RuntimeError("the router went away")))


def test_a_good_request_reaches_the_mesh_verbatim():
    async def check(client, mesh):
        response = await client.post("/api/invoke", json={
            "device_id": "mappo-go2", "function": "walk_forward",
            "params": {"seconds": 1.5, "speed_mps": 0.35}})
        assert response.status == 200
        assert mesh.calls == [("mappo-go2", "walk_forward",
                               {"seconds": 1.5, "speed_mps": 0.35})], mesh.calls
    _serve(check)


def test_the_device_list_is_served_as_json():
    async def check(client, _mesh):
        body = await (await client.get("/api/devices")).json()
        assert body["ok"] is True
        assert [d["device_id"] for d in body["devices"]] == ["mappo-go2"]
    _serve(check)


def test_the_page_is_served():
    async def check(client, _mesh):
        response = await client.get("/")
        assert response.status == 200
        text = await response.text()
        assert "dashboard.js" in text and "Device Connect" in text
    _serve(check)


# ── events ───────────────────────────────────────────────────────────────────
def test_the_device_id_comes_from_the_subject_not_the_payload():
    """A device chooses its payload; the transport assigns the subject.

    A payload that happened to carry a ``device_id`` field — or that deliberately carried
    someone else's — must not be able to relabel a line in the operator's event log.
    """
    record = _shape_event({
        "_subject": "device-connect/default/mappo-go2/event/motion_completed",
        "method": "motion_completed",
        "params": {"travelled_m": 0.36, "device_id": "somebody-else",
                   "event_id": "abc", "ts": "2026-08-21T18:00:00Z",
                   "_trace_id": "deadbeef"},
    }, seq=7)
    assert record["device_id"] == "mappo-go2", record
    assert record["event"] == "motion_completed"
    assert record["seq"] == 7
    # Transport bookkeeping is stripped; the operator sees the payload, not the envelope.
    assert set(record["payload"]) == {"travelled_m", "device_id"}, record["payload"]


def test_a_batch_is_reordered_by_the_devices_timestamp():
    """A completion must not appear above the start that caused it.

    The subscription keeps one inbox per subject and each event NAME is a subject, so a
    drained batch arrives grouped by type. Without this the operator's log reads as though
    the robot finished before it began — which is what the first live capture showed.
    """
    batch = [
        {"method": "motion_completed", "params": {"ts": "2026-08-21T18:00:05Z"}},
        {"method": "motion_completed", "params": {"ts": "2026-08-21T18:00:15Z"}},
        {"method": "motion_started", "params": {"ts": "2026-08-21T18:00:01Z"}},
        {"method": "motion_started", "params": {"ts": "2026-08-21T18:00:11Z"}},
    ]
    order = [(m["method"], m["params"]["ts"][-3:-1]) for m in _in_time_order(batch)]
    assert order == [("motion_started", "01"), ("motion_completed", "05"),
                     ("motion_started", "11"), ("motion_completed", "15")], order


def test_an_event_with_no_timestamp_does_not_kill_the_batch():
    """A malformed event must cost that event's position, not every other event with it."""
    batch = [
        {"method": "b", "params": {"ts": "2026-08-21T18:00:05Z"}},
        {"method": "a", "params": {}},
        {"method": "c", "params": {"ts": "2026-08-21T18:00:09Z"}},
    ]
    assert [m["method"] for m in _in_time_order(batch)] == ["a", "b", "c"]


def test_an_event_with_no_subject_still_shapes_without_raising():
    """The pump must survive a malformed message; one bad line cannot kill the stream."""
    record = _shape_event({"method": "robot_state", "params": {}}, seq=1)
    assert record["device_id"] == ""
    assert record["event"] == "robot_state"


def test_a_page_that_stopped_reading_is_dropped_rather_than_stalling_the_others():
    """One background tab must not be able to stall every other open dashboard.

    The queue is bounded and a full one is skipped, so the pump never blocks on a listener.
    """
    async def check():
        mesh = _FakeMesh()
        queue = mesh.listen()
        for i in range(queue.maxsize + 50):
            try:
                queue.put_nowait({"seq": i})
            except asyncio.QueueFull:
                assert i == queue.maxsize, i
                return
        raise AssertionError("the queue is unbounded; a dead page would grow it forever")
    asyncio.run(check())


def test_the_backlog_is_replayed_to_a_page_that_connects_late():
    """An operator who opens the tab after a run started should see what happened."""
    async def check(client, mesh):
        for i in range(3):
            mesh.events.append({"seq": i, "event": "motion_completed",
                                "device_id": "mappo-go2", "payload": {}, "received": 0})
        # Read only the backlog, then walk away — the handler blocks forever otherwise.
        response = await client.get("/api/events")
        chunks = b""
        for _ in range(3):
            chunks += await response.content.readuntil(b"\n\n")
        response.close()
        seqs = [json.loads(line.split(b"data: ", 1)[1])["seq"]
                for line in chunks.strip().split(b"\n\n")]
        assert seqs == [0, 1, 2], seqs
    _serve(check)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"server: {len(tests)}/{len(tests)} passed")
