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

from robot_driver import PEER_FOOTPRINT_M, MappoRobotDriver


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
    driver = MappoRobotDriver(platform="sim", package_dir=str(_package(tmp)), **kwargs)
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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"robot_driver: {len(tests)}/{len(tests)} passed")
