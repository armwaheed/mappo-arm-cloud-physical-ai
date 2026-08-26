#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Device Connect driver: the gate, the interlock, and the name collision.

The first test in this file exists because of a real hour lost. ``DeviceDriver.capabilities``
is a property on the base class that the runtime reads to build a device's presence
announcement. An ``@rpc()`` named ``capabilities`` overrides it, the announcement is never
published, and the device simply never appears on the mesh — no exception, no log line, and
the driver otherwise runs perfectly, answering nothing because nothing can find it. It looks
exactly like a network problem. ``test_no_rpc_shadows_a_base_class_member`` turns that into
a test failure, and it generalises: ``status``, ``identity``, ``invoke``, ``events``,
``functions``, ``connect`` and a dozen others are all live mines.

Needs device-connect-edge (Python >= 3.11). ``python3 test_robot_driver.py``.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device_connect_edge.drivers import DeviceDriver

import run_control
from robot_driver import PEER_FOOTPRINT_M, MappoRobotDriver

#: The checkpoint this repository actually ships, served as-is by the model-server tests.
DELIVERED_MODELS = Path(__file__).resolve().parent.parent / "policy" / "models"


def _package(tmp):
    """A minimal policy package the driver's ModelStore will accept."""
    package = Path(tmp)
    (package / "models").mkdir(parents=True, exist_ok=True)
    np.savez(
        package / "models" / "armed.npz",
        W1=np.zeros((256, 18), dtype=np.float32), b1=np.zeros((256,), dtype=np.float32),
        W2=np.zeros((256, 256), dtype=np.float32), b2=np.zeros((256,), dtype=np.float32),
        W3=np.zeros((4, 256), dtype=np.float32), b3=np.zeros((4,), dtype=np.float32),
        metadata_json=np.array(json.dumps({
            "actor_input_dim": 18,
            "observation_layout": ["x", "y", "vx", "vy", "dx", "dy"] +
                                  [f"lidar_{i}" for i in range(12)],
            "training_lidar_range_vmas": 0.35})))
    (package / "config.json").write_text(json.dumps(
        {"model_path": "models/armed.npz", "lidar_range_vmas": 0.35}, indent=2))
    return package


def _driver(tmp, events=None, **kwargs):
    """A driver wired to an event sink instead of a DeviceRuntime.

    ``@emit`` raises if the driver is not mounted, so an unmounted driver cannot be tested
    at all without this. ``set_event_callback`` is the SDK's own seam for it.
    """
    driver = MappoRobotDriver(platform=kwargs.pop("platform", "sim"),
                              package_dir=str(_package(tmp)), **kwargs)
    sink = events if events is not None else []
    driver.set_event_callback(lambda name, payload: sink.append((name, payload)))
    return driver


def _run(coro):
    return asyncio.run(coro)


async def _settle(driver):
    """Await the background nudge a motion RPC started.

    Motion RPCs return as soon as the nudge is accepted, so a test that asserts on what the
    worker was handed must wait for it. Without this the assertions depend on whether the
    task happened to be scheduled before ``asyncio.run`` tore the loop down — green by luck.
    """
    task = getattr(driver, "_motion_task", None)
    if task is not None:
        await task


# ── the collision that made the device invisible ─────────────────────────────
def test_no_rpc_shadows_a_base_class_member():
    """An @rpc named after a base-class member breaks presence with no error anywhere.

    Made to fail by renaming ``get_capabilities`` back to ``capabilities``: the device then
    starts, logs "presence announcer started", publishes nothing, and is undiscoverable.
    """
    base_members = {name for name in dir(DeviceDriver) if not name.startswith("__")}
    shadowed = []
    for name, member in vars(MappoRobotDriver).items():
        if name.startswith("_"):
            continue
        if not callable(member) and not isinstance(member, property):
            continue
        if name in base_members and name not in _DELIBERATE_OVERRIDES:
            shadowed.append(name)
    assert not shadowed, (
        f"{shadowed} shadow members of DeviceDriver. If one of these is an @rpc, the device "
        f"will never appear on the mesh and nothing will say why. Rename it, or add it to "
        f"_DELIBERATE_OVERRIDES if it is a lifecycle hook the runtime means you to override.")


#: The base-class members this driver is SUPPOSED to override — the lifecycle hooks and the
#: two descriptive properties the runtime asks a driver to provide. Everything else is a
#: collision.
_DELIBERATE_OVERRIDES = {"connect", "disconnect", "identity", "status", "device_type"}


def test_the_deliberate_overrides_really_are_base_class_members():
    """Keeps the allow-list above honest.

    If the SDK renames a lifecycle hook, an entry here becomes a stale exemption that would
    silently re-permit a real collision under the old name.
    """
    base_members = {name for name in dir(DeviceDriver) if not name.startswith("__")}
    stale = sorted(_DELIBERATE_OVERRIDES - base_members)
    assert not stale, f"{stale} are exempted but no longer exist on DeviceDriver"


def test_every_rpc_and_event_is_discoverable():
    """The dashboard renders from the advertised schema, so an RPC that is not there is a
    button that does nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp)
        # These are the objects the runtime puts on the presence announcement, so reading
        # them here is reading exactly what a discovering client will see.
        functions = {f.name for f in driver.capabilities.functions}
        events = {e.name for e in driver.capabilities.events}

    expected_functions = {
        "walk_forward", "walk_back", "strafe_left", "strafe_right",
        "turn_left", "turn_right", "lie_down", "stand", "stop",
        "get_status", "get_capabilities",
        "list_models", "select_model", "download_model", "delete_model",
        "list_cloud_models"}
    assert expected_functions <= functions, sorted(expected_functions - functions)

    expected_events = {"motion_started", "motion_completed", "motion_refused",
                       "motion_interrupted", "model_armed", "model_downloaded",
                       "model_deleted", "robot_state"}
    assert expected_events <= events, sorted(expected_events - events)


# ── the motion gate ──────────────────────────────────────────────────────────
def test_motion_is_refused_and_the_worker_is_never_started():
    """Two properties, and the second is the one worth having.

    Refusing is easy. Refusing WITHOUT spawning the worker is what makes the gate a gate
    rather than a message printed after the fact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=False)
        started = []
        driver._bridge_blocking = lambda *a, **k: started.append(a) or {"ok": True}

        for action in ("walk_forward", "walk_back", "strafe_left", "strafe_right",
                       "turn_left", "turn_right", "lie_down", "stand"):
            result = _run(getattr(driver, action)())
            assert result["ok"] is False and result["refused"] is True, (action, result)
            assert "--allow-motion" in result["error"], result["error"]
        assert not started, f"the worker was started {len(started)} times despite the gate"


def test_stop_is_allowed_even_with_motion_disabled():
    """A stop that is gated is the wrong affordance on a robot console."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=False)
        driver._bridge_blocking = lambda *a, **k: {"ok": True, "stopped": True}
        result = _run(driver.stop())
        assert result["ok"] is True and result["stopped"] is True, result


def test_checkpoint_operations_work_with_motion_disabled():
    """Managing what a robot carries is not moving it, and should not need the motion flag."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=False)
        listing = _run(driver.list_models())
        assert listing["ok"] is True
        assert listing["active"] == "armed.npz"


# ── the interlock ────────────────────────────────────────────────────────────
def test_a_second_motion_command_is_refused_rather_than_queued():
    """Queued would mean a double-press walks twice, the second time unattended."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)
        released = asyncio.Event()

        async def main():
            # A worker that blocks until we let it go, so the lock is genuinely held.
            def slow(*_a, **_k):
                asyncio.run_coroutine_threadsafe(_noop(), loop)
                import time
                while not released.is_set():
                    time.sleep(0.01)
                return {"ok": True, "travelled_m": 0.3}

            loop = asyncio.get_running_loop()
            driver._bridge_blocking = slow
            first = asyncio.create_task(driver.walk_forward(seconds=1.0))
            await asyncio.sleep(0.3)              # let the first take the lock
            second = await driver.strafe_left(seconds=1.0)
            released.set()
            await first
            return second

        async def _noop():
            return None

        second = _run(main())
        assert second["ok"] is False and second.get("busy") is True, second
        assert "already executing" in second["error"], second["error"]


def test_a_refusal_is_emitted_as_an_event():
    """A refusal is the interesting event; a stream that shows only successes teaches an
    operator that nothing happened."""
    with tempfile.TemporaryDirectory() as tmp:
        emitted = []
        driver = _driver(tmp, events=emitted, allow_motion=False)
        _run(driver.walk_forward())
        names = [name for name, _ in emitted]
        assert "motion_refused" in names, names
        payload = dict(emitted[names.index("motion_refused")][1])
        assert payload["action"] == "walk_forward", payload
        assert "--allow-motion" in payload["reason"], payload


def test_a_failure_to_emit_never_stops_the_robot_being_commanded():
    """``_move`` emits BEFORE it commands, so a broken event bus must not block a walk —
    and must not block a stop.

    Made to fail by calling the @emit methods directly instead of through ``_announce``:
    the exception propagates and the worker is never reached.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)

        def explode(_name, _payload):
            raise RuntimeError("the event bus went away")

        driver.set_event_callback(explode)
        commanded = []
        driver._bridge_blocking = lambda *a, **k: (
            commanded.append(a) or {"ok": True, "travelled_m": 0.3})

        async def main():
            result = await driver.walk_forward(seconds=1.0)
            await _settle(driver)
            return result
        result = _run(main())
        assert result["ok"] is True, result
        assert commanded, "the walk never reached the worker because an event failed"


# ── the stop path, which is why motion is non-blocking ───────────────────────
def test_a_motion_rpc_returns_before_the_nudge_finishes():
    """THE reason this design exists. The edge runtime dispatches one RPC at a time per
    device, so a motion handler that blocks makes the robot deaf to stop for its duration.

    Measured against a live driver before the change: a stop issued 1 s into a 5 s walk was
    delivered 4.17 s later. Made to fail by awaiting ``_run_motion`` inside ``_move``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)
        released = threading.Event()
        entered = threading.Event()

        def slow(*_a, **_k):
            entered.set()
            released.wait(timeout=5)
            return {"ok": True, "travelled_m": 0.3}

        driver._bridge_blocking = slow

        async def main():
            result = await driver.walk_forward(seconds=5.0)   # must NOT wait for `slow`
            assert result["accepted"] is True, result
            assert "travelled_m" not in result, (
                "the RPC returned a measurement, so it waited for the nudge")
            # The worker really is running: the lock is held and the task is alive.
            assert driver._motion_lock.locked()
            released.set()
            await _settle(driver)
            assert not driver._motion_lock.locked(), "the lock outlived the nudge"

        _run(main())


def test_stop_terminates_the_in_flight_worker_before_commanding_zero():
    """A stop that only commands zero is overwritten within 100 ms.

    The worker refreshes the velocity at 10 Hz for the whole nudge, so the running worker
    has to be killed as well. Made to fail by deleting the ``_terminate_worker()`` call in
    ``stop()``: ``interrupted_motion`` comes back False and the walk runs to completion.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)

        class _Proc:
            def __init__(self): self.terminated = False
            def poll(self): return None
            def terminate(self): self.terminated = True

        proc = _Proc()
        driver._worker = proc
        driver._bridge_blocking = lambda *a, **k: {"ok": True, "stopped": True}

        result = _run(driver.stop())
        assert proc.terminated is True, "the in-flight worker was left running"
        assert result["interrupted_motion"] is True, result
        assert driver._worker is None, "the worker handle was not cleared"


def test_stopping_an_idle_robot_reports_that_nothing_was_interrupted():
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)
        driver._bridge_blocking = lambda *a, **k: {"ok": True, "stopped": True}
        result = _run(driver.stop())
        assert result["interrupted_motion"] is False, result


def test_an_interrupted_nudge_is_not_reported_as_a_refusal():
    """An operator must not see their own stop in the colour of a fault.

    A refusal means the command never ran; an interruption means it ran and was ended.
    """
    with tempfile.TemporaryDirectory() as tmp:
        emitted = []
        driver = _driver(tmp, events=emitted, allow_motion=True)
        driver._bridge_blocking = lambda *a, **k: {
            "ok": False, "interrupted": True, "error": "the walk worker was stopped"}

        async def main():
            await driver.walk_forward(seconds=2.0)
            await _settle(driver)
        _run(main())

        names = [name for name, _ in emitted]
        assert "motion_interrupted" in names, names
        assert "motion_refused" not in names, (
            "an interruption was reported as a refusal: " + str(names))


def test_a_worker_that_dies_of_a_signal_is_interrupted_not_failed():
    """The bridge distinguishes 'killed' from 'broken' by the negative return code.

    Without it, every stop would post a scary failure into the operator's event log.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)

        class _Killed:
            returncode = -15
            def poll(self): return -15
            def communicate(self, timeout=None): return ("", "")
            def terminate(self): pass

        import subprocess as sp
        original = sp.Popen
        sp.Popen = lambda *a, **k: _Killed()
        try:
            result = driver._bridge_blocking("walk", ["--vx", "0.35"])
        finally:
            sp.Popen = original
        assert result["interrupted"] is True, result
        assert result["ok"] is False


def test_a_sub_gait_floor_speed_is_refused_in_the_REPLY_not_only_as_an_event():
    """Regression. Making motion non-blocking moved the worker's verdict off the reply path.

    The RPC then answered "accepted" for a speed measured not to walk, emitted a
    motion_started for a motion that never started, and the refusal turned up in the alert
    list milliseconds later. Caught by pressing 0.21 m/s at a simulated Go2 on the demo host.

    Made to fail by deleting the ``_precheck`` call in ``_move``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        emitted = []
        driver = _driver(tmp, events=emitted, allow_motion=True)
        driver.platform = "go2"
        started = []
        driver._bridge_blocking = lambda *a, **k: started.append(a) or {"ok": True}

        result = _run(driver.walk_forward(seconds=1.0, speed_mps=0.21))
        assert result["ok"] is False and result["refused"] is True, result
        assert "0.350" in result["error"], result["error"]
        assert not started, "the worker was started for a speed that cannot walk"

        names = [n for n, _ in emitted]
        assert "motion_started" not in names, (
            "a motion_started was announced for a motion that never started: " + str(names))
        assert "motion_refused" in names, names


def test_force_still_gets_past_the_synchronous_check():
    """The floor is overridable — that is what --force is for — and the override must not be
    swallowed by the new early check."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)
        driver.platform = "go2"
        started = []
        driver._bridge_blocking = lambda *a, **k: started.append(a) or {"ok": True}

        async def main():
            result = await driver.walk_forward(seconds=1.0, speed_mps=0.21, force=True)
            await _settle(driver)
            return result
        result = _run(main())
        assert result.get("accepted") is True, result
        assert started, "force did not reach the worker"


def test_reverse_is_not_subjected_to_the_forward_floor():
    """The floor measures the FORWARD gait; applying it to reverse is the axis conflation
    issue #42 is about."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)
        driver.platform = "go2"
        driver._bridge_blocking = lambda *a, **k: {"ok": True}

        async def main():
            result = await driver.walk_back(seconds=1.0, speed_mps=0.21)
            await _settle(driver)
            return result
        assert _run(main()).get("accepted") is True


# ── advertised checkpoint sources ────────────────────────────────────────────
def test_the_robot_advertises_where_its_checkpoints_come_from():
    """The dashboard should name a source, not ask an operator to remember a URL.

    Advertised by the ROBOT because it is a property of the deployment the robot sits in:
    two robots on one mesh can legitimately pull from different places, which a
    dashboard-level setting cannot express.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sources = [
            {"label": "Arm AGI CPU server", "location": "Tokyo, Japan",
             "kind": "server", "index_url": "http://models:9000/index.json",
             "default_model": "http://models:9000/actor.npz", "simulated": True},
            {"label": "AWS S3", "location": "cn-north-1, Beijing", "kind": "s3",
             "index_url": "http://s3-standin:9001/index.json", "simulated": True},
        ]
        driver = _driver(tmp, model_sources=sources)
        caps = _run(driver.get_capabilities())
        advertised = caps["cloud"]["sources"]
        assert [s["location"] for s in advertised] == ["Tokyo, Japan", "cn-north-1, Beijing"]
        # The demo's stand-ins must carry their own disclaimer into the UI.
        assert all(s["simulated"] for s in advertised)
        assert "robot" in caps["cloud"]["resolved_by"]


def test_an_advertised_source_may_name_a_default_checkpoint():
    """So the dashboard's Source field opens with a loadable address in it.

    Carried through verbatim rather than derived: the page prefills from this with NO
    network call, which is what makes the panel populated the instant a robot is focused
    even when the source itself is unreachable.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, model_sources=[
            {"label": "Arm AGI CPU server", "location": "Tokyo, Japan",
             "index_url": "http://models:9000/index.json",
             "default_model": "http://models:9000/actor_2450000.npz"}])
        source = _run(driver.get_capabilities())["cloud"]["sources"][0]
        assert source["default_model"] == "http://models:9000/actor_2450000.npz"


def test_a_robot_with_no_configured_sources_advertises_an_empty_list():
    """Not a missing key: the page distinguishes 'none configured' from 'older driver'."""
    with tempfile.TemporaryDirectory() as tmp:
        caps = _run(_driver(tmp).get_capabilities())
        assert caps["cloud"]["sources"] == []


def test_a_malformed_sources_file_fails_at_startup_rather_than_advertising_nothing():
    """A robot that quietly advertises nowhere looks, on the dashboard, exactly like one
    configured with no sources on purpose."""
    from robot_driver import _load_sources
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sources.json"
        path.write_text(json.dumps({"sources": [{"location": "Tokyo"}]}))   # no label
        try:
            _load_sources(str(path))
        except ValueError as exc:
            assert "label" in str(exc)
        else:
            raise AssertionError("a source with no label was accepted")

        path.write_text("not json at all")
        try:
            _load_sources(str(path))
        except ValueError:
            return
        raise AssertionError("an unparseable sources file was accepted")


def test_the_emitted_sources_file_is_one_the_driver_loads_and_advertises():
    """``model_server --emit-sources`` writes the address; ``_load_sources`` reads it.

    The two halves are in different files and the address is the thing they must agree
    about, so the writer is checked against the real reader rather than against a copy of
    its schema. A key rename on either side fails here.
    """
    from model_server import ModelServer, sources_document
    from robot_driver import _load_sources
    with tempfile.TemporaryDirectory() as tmp:
        server = ModelServer(DELIVERED_MODELS, port=0).start()
        try:
            path = Path(tmp) / "sources.json"
            path.write_text(json.dumps(sources_document(
                server.index_url, label="workstation", location="this laptop")))
            driver = _driver(tmp, model_sources=_load_sources(str(path)))
            advertised = _run(driver.get_capabilities())["cloud"]["sources"]
        finally:
            server.stop()
    assert len(advertised) == 1, advertised
    assert advertised[0]["label"] == "workstation"
    assert advertised[0]["index_url"] == server.index_url


def test_a_checkpoint_is_browsed_downloaded_and_armed_from_a_real_model_server():
    """The whole Cloud AI path against a real HTTP server, with nothing stubbed.

    ``test_cloud_models.py`` proves the fetch refuses what it should and
    ``test_model_store.py`` proves the store arms what it should, but neither runs the two
    through the driver's own RPCs against a server that is really listening. The three
    steps here are the three the operator takes, in order, and each asserts on the reply the
    dashboard would render.

    It is deliberately the DELIVERED 268 063-byte checkpoint rather than a synthetic one:
    a fixture built by the test cannot catch the delivered file drifting out of the shapes
    this adapter can drive.
    """
    from model_server import ModelServer
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp)
        server = ModelServer(DELIVERED_MODELS, port=0).start()
        try:
            listing = _run(driver.list_cloud_models(index_url=server.index_url))
            assert listing["ok"] is True, listing
            assert listing["kind"] == "server", listing
            entry = listing["objects"][0]
            assert entry["key"] == "mappo_actor_3agent_1910000.npz", entry

            downloaded = _run(driver.download_model(source=entry["uri"]))
            assert downloaded["ok"] is True, downloaded
            assert downloaded["downloaded_bytes"] == entry["size_bytes"], downloaded
            assert downloaded["model"]["sha256"] == entry["sha256"], downloaded
            # Downloading must NOT arm it; arming is a separate, deliberate click.
            assert driver.store.active_model() == "armed.npz"
        finally:
            server.stop()

        names = [m["name"] for m in _run(driver.list_models())["models"]]
        assert entry["key"] in names, names

        armed = _run(driver.select_model(name=entry["key"]))
        assert armed["ok"] is True, armed
        assert driver.store.active_model() == entry["key"]
        assert armed["previous"] == "armed.npz", armed


# ── bounds and plumbing ──────────────────────────────────────────────────────
def test_seconds_is_clamped_before_it_reaches_the_worker():
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)
        seen = {}

        def capture(command, extra=None):
            seen["extra"] = extra
            return {"ok": True}

        driver._bridge_blocking = capture

        async def main():
            await driver.walk_forward(seconds=900)
            await _settle(driver)
        _run(main())
        assert "--seconds" in seen["extra"]
        value = float(seen["extra"][seen["extra"].index("--seconds") + 1])
        assert value <= 5.0, value


def test_the_reverse_cap_is_applied_by_the_driver_too_not_only_by_the_worker():
    """Otherwise the event stream announces a 5 s reverse the worker runs for 2 s.

    The worker caps reverse independently, so the robot is safe either way — what this
    protects is the operator's log agreeing with what the robot did.
    """
    with tempfile.TemporaryDirectory() as tmp:
        announced = []
        driver = _driver(tmp, events=announced, allow_motion=True)
        driver._bridge_blocking = lambda *a, **k: {"ok": True}

        _run(driver.walk_back(seconds=5.0))
        started = next(p for name, p in announced if name == "motion_started")
        assert started["seconds"] <= 2.0, started

        announced.clear()
        _run(driver.walk_forward(seconds=5.0))
        started = next(p for name, p in announced if name == "motion_started")
        assert started["seconds"] == 5.0, started


def test_direction_is_set_by_the_method_not_by_the_caller():
    """``strafe_right(speed_mps=0.2)`` must go right even though the number is positive.

    A caller supplying a sign would make ``strafe_right(-0.2)`` go left, which is a control
    that does the opposite of its label.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)
        seen = []
        driver._bridge_blocking = lambda command, extra=None: (
            seen.append((command, extra)) or {"ok": True})

        for action, flag, expect_negative in (
                ("walk_back", "--vx", True), ("strafe_left", "--vy", False),
                ("strafe_right", "--vy", True), ("turn_left", "--wz", False),
                ("turn_right", "--wz", True)):
            seen.clear()
            # Deliberately pass a POSITIVE magnitude; the method owns the sign.
            kwargs = {"rate_rad_s": 0.7} if "turn" in action else {"speed_mps": 0.3}

            async def main(action=action, kwargs=kwargs):
                await getattr(driver, action)(**kwargs)
                await _settle(driver)
            _run(main())
            _, extra = seen[0]
            value = float(extra[extra.index(flag) + 1])
            assert (value < 0) == expect_negative, (action, value)


def test_the_worker_reads_the_last_json_line_not_the_first():
    """The SDK prints a banner on import and CycloneDDS writes during discovery.

    Reading forwards would parse the robot's startup noise as a result the first time a
    vendor library printed valid JSON.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)
        assert "reversed" in inspect.getsource(driver._bridge_blocking), (
            "the worker's stdout is no longer read backwards")


def test_a_download_that_fails_inspection_leaves_nothing_behind():
    """The temp file must be dropped, or a later retry could mistake it for a cache."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp)
        bad = Path(tmp) / "incoming.npz"
        np.savez(bad, W1=np.zeros((3, 3), dtype=np.float32))

        import cloud_models
        original = cloud_models.fetch
        cloud_models.fetch = lambda *a, **k: (bad, "incoming.npz", 100)
        try:
            result = _run(driver.download_model(source="https://example/incoming.npz"))
        finally:
            cloud_models.fetch = original

        assert result["ok"] is False, result
        assert not bad.exists(), "the rejected download was left on disk"
        assert not (driver.store.models_dir / "incoming.npz").exists()


def test_the_periodic_state_poll_is_skipped_while_the_robot_is_moving():
    """Each poll is a subprocess that connects to DDS. Starting one while the robot walks
    puts a second client on the bus for no benefit."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True)
        calls = []
        driver._bridge_blocking = lambda *a, **k: calls.append(a) or {"ok": True}

        async def main():
            async with driver._motion_lock:
                await driver.publish_state()
        _run(main())
        assert not calls, "a state poll ran while a motion command held the lock"


# ── the peer pose channel ────────────────────────────────────────────────────
class _FakeStdout:
    """The lines a pose worker would print, then EOF."""

    def __init__(self, lines):
        self.lines = list(lines)

    async def readline(self):
        return self.lines.pop(0) if self.lines else b""


class _FakeWorker:
    def __init__(self, lines):
        self.stdout = _FakeStdout(lines)
        self.terminated = False

    def terminate(self):
        self.terminated = True

    async def wait(self):
        return 0


def _pose_lines(*records):
    return [(json.dumps(r) + "\n").encode() for r in records]


def test_publishing_a_pose_is_off_unless_it_is_asked_for():
    """It holds a persistent SDK worker on the DDS bus, so it is not something a robot
    starts because somebody ran the driver. It needs no ``--allow-motion`` though: telling
    other robots where you are must not require permission to walk."""
    with tempfile.TemporaryDirectory() as tmp:
        assert _driver(tmp)._pose_hz == 0.0
        assert _driver(tmp, publish_pose_hz=10.0, allow_motion=False)._pose_hz == 10.0


def test_the_published_footprint_is_the_half_diagonal_of_the_platform():
    """The half-LENGTH of a 0.70 x 0.31 m Go2 is 0.35 and its half-DIAGONAL is 0.383, and
    a peer's heading is not something the robot avoiding it gets to choose.
    ``avoidance.NavConfig.robot_radius_m`` already rounded that to 0.40 for this robot's
    own body; a peer gets the same number. The colour-prop path's 0.15 m is a bin."""
    import math
    assert PEER_FOOTPRINT_M["go2"] > math.hypot(0.35, 0.155)
    assert min(PEER_FOOTPRINT_M.values()) > 2.0 * 0.15


def test_each_worker_line_becomes_one_peer_pose_carrying_an_age_not_a_timestamp():
    """``mono_s`` is the worker's own clock and never leaves this host. What goes on the
    mesh is ``sample_age_s``, a duration — two robots on a demo LAN with no NTP share no
    clock, so an interval crosses it and an instant does not.

    Made to fail by emitting ``mono_s`` itself: the consumer would then subtract two
    unrelated clocks and compute an age of about -1.8e9 s, which is under every threshold
    and so never fires. That is the same fail-OPEN bug ``mappo_bridge.robot_input``
    documents, one machine further out.
    """
    import time as _time
    with tempfile.TemporaryDirectory() as tmp:
        events = []
        driver = _driver(tmp, events=events, publish_pose_hz=10.0)
        now = _time.monotonic()
        worker = _FakeWorker(_pose_lines(
            {"ok": True, "mono_s": now - 0.03, "pose": {"x": 1.0, "y": 2.0, "yaw": 0.0},
             "velocity": [0.1, 0.0, 0.0]},
            {"ok": True, "mono_s": now - 0.01, "pose": {"x": 1.1, "y": 2.0, "yaw": 0.0},
             "velocity": [0.1, 0.0, 0.0]}))
        async def _fake_exec(*_a, **_k):
            return worker

        # Swapped on the module rather than on the driver: `_pump_pose_worker` is the code
        # under test and stubbing it would leave nothing being tested.
        real_exec = asyncio.create_subprocess_exec
        asyncio.create_subprocess_exec = _fake_exec
        try:
            seq = _run(driver._pump_pose_worker(["cmd"], PEER_FOOTPRINT_M["sim"], 0))
        finally:
            asyncio.create_subprocess_exec = real_exec

        assert seq == 2
        names = [name for name, _payload in events]
        assert names == ["peer_pose", "peer_pose"], names
        payload = events[0][1]
        assert "mono_s" not in payload, "a raw clock reading must not reach the mesh"
        assert 0.0 <= payload["sample_age_s"] < 1.0, payload["sample_age_s"]
        assert payload["radius_m"] == PEER_FOOTPRINT_M["sim"]
        assert payload["pose"] == {"x": 1.0, "y": 2.0, "yaw": 0.0}
        assert worker.terminated is True, "the worker was left running"


def test_a_line_that_cannot_be_dated_is_dropped_rather_than_given_an_age_of_zero():
    """An undatable safety input is not a safety input. Emitting it with an invented age
    of zero would make the one sample nobody can date look like the freshest there is;
    dropping it ages the consumer out, which stops the other robot."""
    assert MappoRobotDriver._pose_line(b'{"ok": true, "pose": {"x": 1.0}}') is None
    assert MappoRobotDriver._pose_line(b'{"ok": true, "mono_s": "soon"}') is None


def test_sdk_chatter_on_stdout_is_skipped_rather_than_fatal():
    """The SDK prints on import, which is the same reason ``_bridge_blocking`` reads the
    result backwards from the end of stdout instead of parsing all of it."""
    assert MappoRobotDriver._pose_line(b"[CycloneDDS] discovery started\n") is None
    assert MappoRobotDriver._pose_line(b'"a bare string"') is None


def test_a_failed_estimator_read_is_published_rather_than_swallowed():
    """It reaches the consumer as a pose it cannot use, which reports the estimator. A
    dropped sample would reach it as silence, and silence is reported as a dead link —
    true, and pointing at the wrong machine."""
    record = MappoRobotDriver._pose_line(
        json.dumps({"ok": False, "mono_s": 1.0, "pose": None, "velocity": None,
                    "error": "the estimator went away"}).encode())
    assert record is not None
    assert record["ok"] is False
    assert record["pose"] == {} and record["velocity"] == []


# ── starting a run, and who has the legs while it goes ───────────────────────
class _RunPipe:
    """A pipe that yields queued lines and then blocks until the process exits.

    ⚠️ NOT named ``_FakeStdout``. This file already has one of those, further up, for the
    pose worker — and it does the opposite thing at EOF: it returns ``b""`` immediately,
    because a pose worker that has printed its lines has exited. A second class of the same
    name down here silently replaced the first for every caller in the file, and
    ``test_each_worker_line_becomes_one_peer_pose_carrying_an_age_not_a_timestamp`` hung
    forever on a pipe that was waiting for a process that does not exist. Nothing failed;
    the suite simply stopped, thirteen tests in.
    """

    def __init__(self, lines):
        self._lines = [bytes(line) for line in lines]
        self.closed = asyncio.Event()

    async def readline(self):
        if self._lines:
            return self._lines.pop(0)
        await self.closed.wait()
        return b""


class _RunProcess:
    """A subprocess that never existed, with the surface the driver actually uses."""

    def __init__(self, lines=()):
        self.pid = 4242
        self.stdout = _RunPipe(lines)
        self.returncode = None
        self.terminated = 0
        self._exited = asyncio.Event()

    def terminate(self):
        self.terminated += 1
        if self.returncode is None:
            self.exit(-15)

    def exit(self, code=0):
        self.returncode = code
        self._exited.set()
        self.stdout.closed.set()

    async def wait(self):
        await self._exited.wait()
        return self.returncode

    async def communicate(self):
        await self._exited.wait()
        return (b"", b"")


class _RunSpawner:
    """Stands in for ``asyncio.create_subprocess_exec`` and records every command.

    The FIRST spawn is the run. Every later one is a helper — the remote stop — and exits
    at once, which is what a working ``kill -TERM`` over SSH looks like from here.
    ``confirm`` off models the other case: the signal went out and the run did not die.
    """

    def __init__(self, lines=(), confirm=True):
        self.commands = []
        self.processes = []
        self._lines = list(lines)
        self.confirm = confirm

    async def __call__(self, *command, **_kwargs):
        self.commands.append(list(command))
        first = not self.processes
        process = _RunProcess(self._lines if first else ())
        self.processes.append(process)
        if not first:
            process.exit(0)                       # the helper returns
            if self.confirm:
                self.processes[0].exit(-15)       # ...and the run really did stop
        return process

    @property
    def run(self):
        return self.processes[0]

    def drive(self, main):
        """Run one coroutine in a fresh loop and leave no fake process pending.

        ⚠️ Without the ``finally``, every test that starts a run costs four real seconds.
        ``asyncio.run`` cancels the output pump on the way out, the pump's own ``finally``
        reports the run, and reporting it waits ``RUN_STOP_TIMEOUT_S`` for an exit status
        that a process which never existed will never produce. A dozen of those is a suite
        that takes a minute to say the same thing.
        """
        async def wrapper():
            try:
                return await main()
            finally:
                for process in self.processes:
                    if process.returncode is None:
                        process.exit(0)
                await asyncio.sleep(0)
        return _run(wrapper())


class _PatchSpawn:
    """Swap ``asyncio.create_subprocess_exec`` for the duration of one test."""

    def __init__(self, spawner):
        self.spawner = spawner

    def __enter__(self):
        self._real = asyncio.create_subprocess_exec
        asyncio.create_subprocess_exec = self.spawner
        return self.spawner

    def __exit__(self, *_exc):
        asyncio.create_subprocess_exec = self._real
        return False


#: The deployment shape this feature exists for: the driver on a workstation, the run on the
#: robot, ssh in between. See ``run_control``'s module docstring for why it cannot be local.
_REMOTE = run_control.RunProfile(
    label="lab go2", workdir="/home/unitree/mappo-run/integration",
    python="/home/unitree/robotics-connect-envs/armwaheed/bin/python3",
    script="mappo_drive.py", package="/home/unitree/mappo-run/policy",
    launch_prefix=("ssh", "unitree@192.168.123.18"),
    extra_args=("--robot-radius", "0.25"), output_dir="/home/unitree/dashboard-runs")


def _first(emitted, name):
    """The FIRST payload emitted under ``name``.

    ``dict(emitted)`` would keep the LAST, and a run that starts and then ends emits
    ``control_changed`` twice — so the collapsed dict reports the hand-back and the test
    asserting the hand-over reads it as a failure to hand over.
    """
    for emitted_name, payload in emitted:
        if emitted_name == name:
            return dict(payload)
    raise AssertionError(f"{name} was never emitted; saw {[n for n, _ in emitted]}")


def _sport(*_a, **_k):
    """A bridge whose robot is in a sport mode, so the preflight passes."""
    return {"ok": True, "mode": "normal", "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "velocity": [0.0, 0.0, 0.0]}


def _runner(tmp, events=None, **kwargs):
    kwargs.setdefault("run_profile", _REMOTE)
    kwargs.setdefault("allow_motion", True)
    driver = _driver(tmp, events=events, **kwargs)
    driver._bridge_blocking = _sport
    return driver


async def _start(driver, **kwargs):
    result = await driver.start_run(**kwargs)
    # Let the output pump reach its first await, so a test that then stops the run is
    # stopping a run that is genuinely being read rather than one that never started.
    await asyncio.sleep(0)
    return result


#: ⚠️ Anything about a RUNNING run has to be asserted INSIDE the loop.
#:
#: ``asyncio.run`` cancels whatever is still pending when its main coroutine returns, and
#: ``_pump_run``'s ``finally`` then reports the run and hands control back. So a test that
#: starts a run, lets ``_run`` return and then checks ``_control_owner`` reads ``operator``
#: every time — including when the driver never took control at all, which is the state
#: those tests are trying to distinguish. It is the same shape as ``_settle``: a background
#: task the assertions depend on, and a green that would otherwise be luck.


def test_a_run_cannot_be_started_without_motion_enabled_and_nothing_is_spawned():
    """⛔ THE gate, at the driver. Same shape as ``walk_forward``'s and same two properties:
    it refuses, and it refuses WITHOUT spawning anything.

    Made to fail by passing ``allow_motion=True`` into ``build_run_argv`` from ``start_run``
    instead of ``self.allow_motion``: the run is then launched from a device that was
    started status-and-checkpoints only. (``run_control`` holds the second copy of this
    gate, and ``test_run_control`` covers that one — two processes with one shared
    assumption is one process with an unwritten contract.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, allow_motion=False)
        with _PatchSpawn(_RunSpawner()) as spawner:
            result = spawner.drive(lambda: _start(driver, arm_motion=True))
        assert result["ok"] is False and result["refused"] is True, result
        assert "--allow-motion" in result["error"], result["error"]
        assert not spawner.commands, "a run was spawned despite the gate"
        assert driver._control_owner == "operator"


def test_arming_motion_on_an_ungated_driver_refuses_instead_of_downgrading():
    """A silent downgrade would start a run the operator believes is driving.

    They asked for motion, they watched a run start, and they are now watching a robot that
    was never going to move — which is the same thing they would see from ``mode='mcf'``,
    from a flat battery and from a sub-gait-floor command, and they would spend the demo
    telling those apart. Made to fail by dropping ``arm_motion`` to False and starting a dry
    run when the gate is shut.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, allow_motion=False, platform="go2")
        with _PatchSpawn(_RunSpawner()) as spawner:
            result = spawner.drive(lambda: _start(driver, arm_motion=True))
        assert result["ok"] is False and result["refused"] is True, result
        assert "--allow-motion" in result["error"], result["error"]
        assert not spawner.commands, "a downgraded run was started anyway"


def test_a_run_is_refused_on_a_robot_that_would_accept_it_and_never_step():
    """The Go2 answered ``mode='mcf'`` on 2026-08-26. ``Move`` is ignored there, so the run
    would command velocities for its whole length against a robot that never moves.

    Made to fail by dropping the ``check_control_mode`` call from ``start_run``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        driver._bridge_blocking = lambda *a, **k: {"ok": True, "mode": "mcf"}
        with _PatchSpawn(_RunSpawner()) as spawner:
            result = spawner.drive(lambda: _start(driver, arm_motion=True))
        assert result["refused"] is True and "mcf" in result["error"], result
        assert not spawner.commands, "a run was spawned at a robot that would ignore it"


def test_a_started_run_reports_the_command_and_claims_no_commit():
    """The deployed tree is not a checkout, so the run can be named by its COMMAND and by
    nothing else. Made to fail by dropping ``argv`` from the reply or the event."""
    with tempfile.TemporaryDirectory() as tmp:
        emitted = []
        driver = _runner(tmp, events=emitted, platform="go2")
        with _PatchSpawn(_RunSpawner()) as spawner:
            result = spawner.drive(lambda: _start(driver, seconds=20,
                                                  arm_motion=True))
        assert result["ok"] is True and result["started"] is True, result
        assert result["named_commit"] is None
        assert "--live" in result["argv"] and "--max-seconds" in result["argv"]
        assert spawner.commands[0][0] == "ssh", spawner.commands[0]
        payload = _first(emitted, "run_started")
        assert payload["argv"] == result["argv"]
        assert payload["command"][0] == "ssh"
        assert "not a git checkout" in payload["note"]


def test_a_live_run_takes_the_legs_and_says_so():
    with tempfile.TemporaryDirectory() as tmp:
        emitted = []
        driver = _runner(tmp, events=emitted, platform="go2")
        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                return driver._control_owner
            owner = spawner.drive(main)
        assert owner == "policy", owner
        announced = _first(emitted, "control_changed")
        assert announced["owner"] == "policy", announced


def test_the_default_run_takes_no_flags_no_gate_and_cannot_move_the_robot():
    """``start_run()`` with nothing at all is the scene check, and it is the first thing
    anybody presses.

    Three properties in one: it works on a driver started WITHOUT ``--allow-motion``, the
    command line it builds carries no ``--live`` — so it has no path to a leg at all — and
    it does not take the motion pad, because a run that cannot command a leg has no claim on
    it. Made to fail by defaulting ``arm_motion`` to True, or by handing control over
    unconditionally in ``_launch``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, allow_motion=False, platform="go2")
        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                started = await _start(driver)          # no arguments: the scene check
                return started, driver._control_owner
            result, owner = spawner.drive(main)
        assert result["ok"] is True, result
        assert "--live" not in result["argv"]
        assert owner == "operator", "a dry run took the motion pad away from the operator"
        assert spawner.commands, "a dry run did not start"


def test_a_manual_command_is_refused_while_the_policy_is_driving():
    """One authority at a time. A manual nudge and a policy tick would both write a velocity
    on the same bus at 10 Hz and the last writer would win — an arbitration with no owner
    and no record.

    Made to fail by deleting the ``_control_owner`` check in ``_move``: the walk is then
    accepted and the worker is started underneath a running policy.
    """
    with tempfile.TemporaryDirectory() as tmp:
        emitted = []
        driver = _runner(tmp, events=emitted, platform="go2")
        started = []
        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                driver._bridge_blocking = lambda *a, **k: (
                    started.append(a) or {"ok": True, "travelled_m": 0.3})
                return await driver.walk_forward(seconds=1.0)
            refusal = spawner.drive(main)
        assert refusal["ok"] is False and refusal["policy_driving"] is True, refusal
        assert "stop_run" in refusal["error"], refusal["error"]
        assert not started, "a manual nudge reached the worker while the policy was driving"
        assert "motion_refused" in [name for name, _ in emitted]


def test_stop_is_never_gated_by_the_policy_holding_the_legs():
    """The one command that must work in every state. Made to fail by moving the ownership
    check above ``stop``'s own path."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                return await driver.stop()
            result = spawner.drive(main)
        assert result["ok"] is True, result
        assert result["ended_run"] is True


def test_stop_ends_a_running_policy_before_it_commands_zero():
    """⛔ The mutation this test exists for.

    ``stop`` used to address the motion pad and nothing else, and a ``mappo_drive`` run is
    the second thing that can be commanding this robot — the more urgent of the two, because
    it holds the legs indefinitely and refreshes its velocity every tick. A stop that only
    commanded zero would be overwritten inside one control period and the robot would
    visibly pause and carry on: the exact failure ``_terminate_worker`` exists for, in a form
    no test covered until the run existed.

    ``STOP ALL`` and every per-row stop invoke this same function, so they inherit it.

    Made to fail by deleting the ``await self._end_run("stop")`` line from ``stop()``:
    ``ended_run`` comes back False, the run process is never signalled, and the policy is
    still driving after the operator pressed stop.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        order = []
        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                driver._bridge_blocking = lambda *a, **k: (
                    order.append("zeroed") or {"ok": True, "stopped": True})
                return await driver.stop()
            result = spawner.drive(main)

        assert result["ended_run"] is True, result
        assert result["run_stop_confirmed"] is True, result
        # The remote stop went out, and it went out BEFORE the velocity was zeroed.
        stop_commands = [c for c in spawner.commands[1:] if "kill -TERM" in " ".join(c)]
        assert stop_commands, f"the run was never signalled: {spawner.commands}"
        assert order == ["zeroed"], order
        assert driver._control_owner == "operator"
        assert driver._run is None


def test_an_unconfirmed_run_stop_makes_the_reply_a_failure():
    """The dashboard classifies a stop by ``ok``. A policy that may still be driving must
    not come back as a tick — the operator has to be sent to the physical abort.

    Made to fail by leaving ``result["ok"]`` alone when ``confirmed`` is False.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        original = run_control.RUN_STOP_TIMEOUT_S
        run_control.RUN_STOP_TIMEOUT_S = 0.05     # do not sit out the real budget
        try:
            with _PatchSpawn(_RunSpawner(confirm=False)) as spawner:
                async def main():
                    await _start(driver, arm_motion=True)
                    driver._bridge_blocking = lambda *a, **k: {"ok": True, "stopped": True}
                    return await driver.stop()
                result = spawner.drive(main)
        finally:
            run_control.RUN_STOP_TIMEOUT_S = original
        assert result["ok"] is False, result
        assert result["run_stop_confirmed"] is False
        assert "STILL DRIVING" in result["error"], result["error"]


def test_stopping_the_run_hands_the_pad_back_and_the_next_press_is_accepted():
    """Taking control is one deliberate call, and afterwards the keys work.

    Made to fail by not handing control back in ``_end_run``: the pad stays dead after the
    run has been stopped, which is the state an operator cannot get out of.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                stopped = await driver.stop_run(reason="taking manual control")
                driver._bridge_blocking = lambda *a, **k: {"ok": True, "travelled_m": 0.3}
                walk = await driver.walk_forward(seconds=1.0)
                await _settle(driver)
                return stopped, walk
            stopped, walk = spawner.drive(main)
        assert stopped["was_running"] is True and stopped["ok"] is True, stopped
        assert driver._control_owner == "operator"
        assert walk["ok"] is True and walk["accepted"] is True, walk


def test_the_pad_comes_back_even_when_the_stop_did_not_confirm():
    """The state an operator cannot get out of, and the reason ``_end_run`` hands control
    back BEFORE it waits.

    A remote stop is a second SSH round trip and it can fail to confirm — the policy may
    still be driving. Waiting for that before releasing the pad means an operator whose stop
    did not land also cannot press a key, which is precisely backwards: they need the keys
    more, not less. (``_finish_run`` hands the pad back too, but only once the run's pipe
    closes, and an unconfirmed run's pipe may never close. That is why the sibling test with
    a confirming stop cannot catch this and this one can.)

    Made to fail by deleting the ``_hand_control`` call from ``_end_run``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        original = run_control.RUN_STOP_TIMEOUT_S
        run_control.RUN_STOP_TIMEOUT_S = 0.05
        try:
            with _PatchSpawn(_RunSpawner(confirm=False)) as spawner:
                async def main():
                    await _start(driver, arm_motion=True)
                    stopped = await driver.stop_run()
                    owner = driver._control_owner
                    driver._bridge_blocking = lambda *a, **k: {"ok": True, "travelled_m": 0.1}
                    walk = await driver.walk_forward(seconds=1.0)
                    await _settle(driver)
                    return stopped, owner, walk
                stopped, owner, walk = spawner.drive(main)
        finally:
            run_control.RUN_STOP_TIMEOUT_S = original
        assert stopped["confirmed"] is False, stopped
        assert owner == "operator", (
            "the stop did not confirm and the motion pad was left locked; the operator can "
            "neither drive nor stop driving")
        assert walk["ok"] is True and walk["accepted"] is True, walk


def test_stopping_a_run_that_is_not_running_is_not_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp)
        result = _run(driver.stop_run())
        assert result["ok"] is True and result["was_running"] is False, result


def test_a_second_run_is_refused_rather_than_started_on_top_of_the_first():
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                return await driver.start_run()
            second = spawner.drive(main)
        assert second["refused"] is True and "already in flight" in second["error"], second
        assert len(spawner.commands) == 1, spawner.commands


def test_a_run_that_ends_by_itself_is_reported_once_and_gives_the_pad_back():
    with tempfile.TemporaryDirectory() as tmp:
        emitted = []
        driver = _runner(tmp, events=emitted, platform="go2")
        with _PatchSpawn(_RunSpawner(lines=[b"[mappo_drive] policy supervised\n"])) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                spawner.run.exit(0)
                await driver._run_task
            spawner.drive(main)
        names = [name for name, _ in emitted]
        assert names.count("run_finished") == 1, names
        finished = _first(emitted, "run_finished")
        assert finished["reason"] == "the run completed", finished
        assert driver._control_owner == "operator" and driver._run is None
        assert "run_output" in names, names


def test_the_watchdog_ends_a_run_that_outlives_its_bound():
    """A nudge is capped at 5 s by the worker. A run's cap is its own ``--max-seconds``, and
    this is what happens when the run does not honour it — a wedged camera open, a stalled
    detector, or a tree 43 commits behind whose flag means something else.

    Driven directly rather than by waiting: the budget is the run's length plus 30 s of
    startup, and a test that slept through it would be a minute long.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                driver._run_deadline_task.cancel()
                await driver._run_deadline(driver._run, 0.0)
            spawner.drive(main)
        assert driver._run is None or driver._run.finished_reason
        assert any("kill -TERM" in " ".join(c) for c in spawner.commands[1:]), (
            "the watchdog did not signal the run")


def test_the_periodic_state_poll_is_skipped_for_the_whole_of_a_run():
    """One ``status`` is 1.94 s of cold SDK import and DDS discovery on the Go2's Jetson —
    39% of the 5 s poll period — and a run holds the bus at 10 Hz for its whole length.
    Putting a second client on it buys a pose the fleet row could do without.

    Made to fail by dropping ``or self._run is not None`` from ``publish_state``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                polled = []
                driver._bridge_blocking = lambda *a, **k: (
                    polled.append(a) or {"ok": True, "mode": "normal"})
                await driver.publish_state()
                return polled
            polled = spawner.drive(main)
        assert not polled, "the state poll ran a DDS client against a live control loop"


def test_get_status_says_who_has_the_legs_and_what_is_running():
    """The page cannot render an unambiguous "who is in charge" from an absence."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        idle = _run(driver.get_status())
        assert idle["control"]["owner"] == "operator"
        assert idle["control"]["manual_motion_allowed"] is True
        assert idle["run"]["active"] is False and idle["run"]["supported"] is True

        with _PatchSpawn(_RunSpawner()) as spawner:
            async def main():
                await _start(driver, arm_motion=True)
                return await driver.get_status()
            busy = spawner.drive(main)
        assert busy["control"]["owner"] == "policy", busy["control"]
        assert busy["control"]["manual_motion_allowed"] is False
        assert busy["run"]["active"] is True and busy["run"]["live"] is True
        assert busy["run"]["named_commit"] is None
        assert "elapsed_s" in busy["run"] and "started_at" not in busy["run"]


def test_get_status_reports_a_mode_that_would_ignore_every_command():
    """``mode='mcf'`` is the answer to "why is nothing happening", and a page that cannot
    say it sends the operator to look for a tether."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        driver._bridge_blocking = lambda *a, **k: {"ok": True, "mode": "mcf"}
        result = _run(driver.get_status())
        assert result["mode_accepts_motion"] is False
        assert "mcf" in result["mode_note"]

        driver._bridge_blocking = _sport
        assert _run(driver.get_status())["mode_accepts_motion"] is True


def test_a_driver_with_no_run_profile_refuses_and_advertises_that_it_cannot():
    with tempfile.TemporaryDirectory() as tmp:
        driver = _driver(tmp, allow_motion=True, run_profile=None)
        driver._bridge_blocking = _sport
        result = _run(driver.start_run())
        assert result["refused"] is True and "--run-profile" in result["error"], result
        assert _run(driver.get_capabilities())["run"]["supported"] is False


def test_the_capabilities_show_both_commands_a_press_could_run():
    """The page can show the operator the command BEFORE the press rather than in the reply
    after it, and it can show which of the two an arm would change it into."""
    with tempfile.TemporaryDirectory() as tmp:
        driver = _runner(tmp, platform="go2")
        caps = _run(driver.get_capabilities())["run"]
        assert caps["supported"] is True and caps["remote"] is True
        assert "--live" not in caps["command_preview"], caps["command_preview"]
        assert "--live" in caps["armed_command_preview"]
        assert caps["named_commit"] is None

        ungated = _driver(tmp, allow_motion=False, run_profile=_REMOTE, platform="go2")
        gated = _run(ungated.get_capabilities())["run"]
        assert gated["armed_command_preview"] is None, (
            "a device that will refuse an armed run advertised one anyway")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"robot_driver: {len(tests)}/{len(tests)} passed")
