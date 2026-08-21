#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The quadruped as a Device Connect device: motion, checkpoints, and an event stream.

⛔ **``robot-stack/SAFETY.md`` governs the motion RPCs.** They command legs.

Runs ON the robot, in a Python >= 3.11 environment that holds Device Connect and nothing
else. The robot's own SDK — ``unitree_sdk2py`` on Python 3.8, or ROS 2 — stays in its own
environment and is reached through ``drive_bridge.py`` as a subprocess. See that file for
why the two cannot share one interpreter, and why the split turns out to be a safety
property rather than only a packaging one.

## The surface, and why it is shaped this way

**Six named motion RPCs, not one parameterised ``move(direction, amount)``.** ``@rpc``
generates a JSON Schema from the type hints and docstring of each method, and that schema is
what the dashboard renders its controls from and what an agent reads to decide what this
robot can do. Six methods produce six self-describing capabilities; one method with a string
parameter produces one capability and a free-text field, and neither a person nor a model
can tell from it that reverse is special. The six share a single guarded implementation, so
the duplication is in the signatures only.

**Motion is off unless it was enabled at startup.** ``--allow-motion`` is this file's
``--live``. A dashboard that can walk a robot the moment someone opens a browser tab is a
worse safety posture than the command line it replaces, so the default device is
status-and-checkpoints only. The gate is checked here AND again in the worker, because two
processes with one shared assumption is one process with an unwritten contract.

**One motion at a time, and a second request is refused rather than queued.** Queuing is
the wrong answer for a robot: a button pressed twice would walk twice, the second time
several seconds after anyone was looking at it. :attr:`_motion_lock` is held for the whole
subprocess, and a concurrent request comes back immediately as ``busy``.

**Checkpoint operations are not motion and are not gated by it.** Swapping a checkpoint
rewrites one field in ``config.json`` and takes effect on the NEXT run — a live
``mappo_drive`` holds its weights in memory and cannot be affected. See ``model_store.py``,
which is where that argument is made properly and where the compatibility check that makes
a swap safe actually lives.

## What the event stream carries

``motion_started`` / ``motion_completed`` / ``motion_refused`` — every command, including the
ones that were turned down, because a refusal is the interesting event and a dashboard that
only shows successes teaches an operator that nothing happened.
``model_armed`` / ``model_downloaded`` / ``model_deleted`` — every change to what the robot
is carrying. ``robot_state`` — pose, velocity and controller mode on a timer, which is what
makes the page live rather than a form.

Run on the robot:
    python3 robot_driver.py --platform go2 --package ../policy --allow-motion
Off-robot, against the bench double:
    python3 robot_driver.py --platform sim --package ../policy --allow-motion

``python3 test_robot_driver.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from device_connect_edge import DeviceRuntime
from device_connect_edge.drivers import DeviceDriver, emit, periodic, rpc
from device_connect_edge.types import DeviceIdentity, DeviceStatus

import cloud_models
from cloud_models import CloudFetchError
from drive_bridge import (
    DEFAULT_VX,
    DEFAULT_VY,
    DEFAULT_WZ,
    MAX_REVERSE_SECONDS,
    MAX_SECONDS,
)
from model_store import ModelStore, ModelStoreError

log = logging.getLogger("mappo-dashboard-driver")

DEFAULT_BRIDGE = str(Path(__file__).resolve().parent / "drive_bridge.py")

#: Ceiling on how long the driver waits for one worker process. A nudge is capped at
#: ``MAX_SECONDS``; the rest of this budget is process start, SDK import and DDS discovery,
#: which on a cold Jetson is the slow part.
BRIDGE_TIMEOUT_S = MAX_SECONDS + 25.0

#: How often ``robot_state`` goes out. Fast enough that the page feels live, slow enough
#: that a status poll is not competing with a control loop for the DDS channel — every one
#: of these is a subprocess that connects, reads and exits.
STATE_INTERVAL_S = 5.0


class MappoRobotDriver(DeviceDriver):
    """A Go2 or Lite3 Venture carrying a MAPPO checkpoint, on the Device Connect mesh."""

    device_type = "mappo_quadruped"

    def __init__(self, *, platform: str = "sim", package_dir: str = "../policy",
                 bridge_script: str = DEFAULT_BRIDGE, bridge_python: str | None = None,
                 iface: str = "eth0", allow_motion: bool = False,
                 operator_ready: bool = False, allow_http: bool = True) -> None:
        super().__init__()
        self.platform = platform
        self.bridge_script = bridge_script
        # The interpreter that can import the robot's SDK. Defaults to this one, which is
        # correct for the sim and wrong for a real robot — the runbook says to pass it, and
        # `capabilities()` reports it so a misconfiguration is visible before a run.
        self.bridge_python = bridge_python or sys.executable
        self.iface = iface
        self.allow_motion = allow_motion
        self.operator_ready = operator_ready
        self.allow_http = allow_http
        self.store = ModelStore(package_dir)
        self._motion_lock = asyncio.Lock()

    # ── identity ─────────────────────────────────────────────────────────────
    @property
    def identity(self) -> DeviceIdentity:
        vendor = {"go2": "Unitree", "lite3": "Deep Robotics", "sim": "none"}[self.platform]
        model = {"go2": "Go2 EDU", "lite3": "Lite3 Venture",
                 "sim": "bench double"}[self.platform]
        return DeviceIdentity(
            device_type=self.device_type,
            manufacturer=vendor,
            model=model,
            description=(
                f"{model} running a MAPPO policy. Bounded motion nudges, checkpoint "
                f"management, and a live event stream. Motion is "
                f"{'ENABLED' if self.allow_motion else 'DISABLED (status and checkpoints only)'}."
            ),
        )

    @property
    def status(self) -> DeviceStatus:
        return DeviceStatus(availability="busy" if self._motion_lock.locked() else "available")

    async def connect(self) -> None:
        log.info("driver up: platform=%s motion=%s package=%s", self.platform,
                 "ENABLED" if self.allow_motion else "disabled", self.store.package_dir)

    async def disconnect(self) -> None:
        """Stop the robot on the way out.

        The runtime calls this on a clean shutdown. It is best-effort by design — if the
        process is being torn down because the worker environment is broken, a failure here
        must not mask that.
        """
        if self.allow_motion:
            try:
                await self._bridge("stop")
            except Exception as exc:
                log.warning("stop on shutdown failed: %r", exc)

    # ── events ───────────────────────────────────────────────────────────────
    async def _announce(self, event, **payload) -> None:
        """Emit an event, and never let a failure to emit affect the robot.

        This is the same rule ``robot-stack``'s telemetry writer states for itself: a
        failure to record must not take down the thing being recorded. It matters more here
        than it looks, because ``_move`` emits ``motion_started`` BEFORE it commands
        anything — an unmounted driver, or a transport hiccup, would otherwise mean a
        button press that never reaches the legs and a stop that never gets sent.

        The run is the product; the event stream is a record of it.
        """
        try:
            await event(**payload)
        except Exception as exc:
            log.warning("could not emit %s: %r", getattr(event, "__name__", event), exc)

    @emit()
    async def motion_started(self, action: str, commanded: dict, seconds: float):
        """A bounded motion command was accepted and the robot is moving."""

    @emit()
    async def motion_completed(self, action: str, travelled_m: float, turned_deg: float,
                               delivered_fraction: float, warning: str):
        """A motion command finished. ``travelled_m`` is measured, not commanded."""

    @emit()
    async def motion_refused(self, action: str, reason: str):
        """A motion command was turned down — by the gate, a gait floor, or a busy robot."""

    @emit()
    async def model_armed(self, name: str, previous: str):
        """A different checkpoint will be used by the next run."""

    @emit()
    async def model_downloaded(self, name: str, source: str, size_bytes: int, sha256: str):
        """A checkpoint arrived from Cloud AI and passed inspection."""

    @emit()
    async def model_deleted(self, name: str, freed_bytes: int):
        """A checkpoint was removed from the robot."""

    @emit()
    async def robot_state(self, pose: dict, velocity: list, mode: str, active_model: str):
        """Periodic liveness: where the robot is and what it is carrying."""

    # ── the worker bridge ────────────────────────────────────────────────────
    def _bridge_blocking(self, command: str, extra=None) -> dict:
        """Run one ``drive_bridge.py`` command and parse its last stdout line.

        The timeout path sends SIGTERM before SIGKILL. That ordering is load-bearing: the
        worker installs a damp on SIGTERM, and ``SportClient.Move`` has no dead-man timeout,
        so a bare kill of a walking robot leaves the last velocity latched on the bus.
        """
        cmd = [self.bridge_python, self.bridge_script, command,
               "--platform", self.platform, "--iface", self.iface]
        cmd += [str(a) for a in (extra or [])]
        if self.allow_motion:
            cmd.append("--allow-motion")
        if self.operator_ready:
            cmd.append("--operator-ready")

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
        except OSError as exc:
            return {"ok": False, "error": f"could not start the worker: {exc}. "
                                          f"--bridge-python must be an interpreter that can "
                                          f"import this robot's SDK."}
        try:
            stdout, stderr = proc.communicate(timeout=BRIDGE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                return {"ok": False, "error": f"{command} did not respond to SIGTERM and was "
                                              f"killed; assume the robot was NOT stopped "
                                              f"cleanly and check it before commanding again"}
            return {"ok": False, "error": f"{command} timed out after {BRIDGE_TIMEOUT_S}s; "
                                          f"the worker was signalled and stopped the robot"}

        # The result is the LAST JSON line: the SDK prints a banner on import, and CycloneDDS
        # writes to stdout during discovery. Reading backwards is what makes that survivable.
        for line in reversed((stdout or "").strip().splitlines()):
            try:
                return json.loads(line)
            except ValueError:
                continue
        return {"ok": False, "error": f"worker exited {proc.returncode} with no JSON result; "
                                      f"stderr tail: {(stderr or '')[-400:]!r}"}

    async def _bridge(self, command: str, extra=None) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._bridge_blocking, command, extra)

    async def _move(self, action: str, command: str, extra: list, seconds: float,
                    commanded: dict) -> dict:
        """The one guarded path every motion RPC funnels through."""
        if not self.allow_motion:
            reason = ("this device was started without --allow-motion, so it is "
                      "status-and-checkpoints only")
            await self._announce(self.motion_refused, action=action, reason=reason)
            return {"ok": False, "refused": True, "error": reason}

        if self._motion_lock.locked():
            reason = "the robot is already executing a motion command"
            await self._announce(self.motion_refused, action=action, reason=reason)
            return {"ok": False, "refused": True, "busy": True, "error": reason}

        async with self._motion_lock:
            await self._announce(self.motion_started, action=action,
                                 commanded=commanded, seconds=seconds)
            result = await self._bridge(command, extra)
            result["action"] = action
            if result.get("ok"):
                await self._announce(
                    self.motion_completed,
                    action=action,
                    travelled_m=result.get("travelled_m") or 0.0,
                    turned_deg=result.get("turned_deg") or 0.0,
                    delivered_fraction=result.get("delivered_fraction") or 0.0,
                    warning=result.get("warning") or "",
                )
            else:
                await self._announce(self.motion_refused, action=action,
                                     reason=result.get("error", "unknown failure"))
            return result

    @staticmethod
    def _clamp_seconds(seconds: float, reverse: bool = False) -> float:
        """Clamp to the same ceiling the worker will apply.

        The worker caps reverse at ``MAX_REVERSE_SECONDS`` independently, so without the
        ``reverse`` arm here the ``motion_started`` event announces 5 s for a command the
        worker is about to run for 2 — an event stream that disagrees with the robot.
        """
        ceiling = MAX_REVERSE_SECONDS if reverse else MAX_SECONDS
        return max(0.1, min(float(seconds), ceiling))

    # ── motion RPCs ──────────────────────────────────────────────────────────
    @rpc()
    async def walk_forward(self, seconds: float = 1.5, speed_mps: float = DEFAULT_VX,
                           force: bool = False) -> dict:
        """Walk forward for a bounded time, then stop.

        Args:
            seconds: how long to hold the command, capped at 5 s.
            speed_mps: commanded forward speed. Below 0.35 m/s neither platform walks.
            force: command a speed below the measured gait floor anyway.

        Returns:
            What the robot actually did: distance travelled, and the fraction of the
            commanded speed that was delivered.
        """
        seconds = self._clamp_seconds(seconds)
        extra = ["--vx", abs(speed_mps), "--seconds", seconds] + (["--force"] if force else [])
        return await self._move("walk_forward", "walk", extra, seconds,
                                {"vx": abs(speed_mps)})

    @rpc()
    async def walk_back(self, seconds: float = 1.0, speed_mps: float = DEFAULT_VX) -> dict:
        """Walk backwards for a bounded time, then stop. NOTHING IS WATCHING BEHIND.

        Neither platform has rear sensing and the planner never samples this direction, so
        this is open-loop into unobserved space. Capped harder than forward, at 2 s.

        Args:
            seconds: how long to hold the command, capped at 2 s for reverse.
            speed_mps: commanded speed; the sign is supplied by this method.
        """
        seconds = self._clamp_seconds(seconds, reverse=True)
        return await self._move("walk_back", "walk",
                                ["--vx", -abs(speed_mps), "--seconds", seconds],
                                seconds, {"vx": -abs(speed_mps)})

    @rpc()
    async def strafe_left(self, seconds: float = 1.5, speed_mps: float = DEFAULT_VY,
                          force: bool = False) -> dict:
        """Crab sideways to the left for a bounded time, then stop.

        Args:
            seconds: how long to hold the command, capped at 5 s.
            speed_mps: commanded lateral speed. The Lite3's lateral floor is 0.20 m/s; the
                Go2's has never been measured, so a Go2 strafe may produce no gait.
            force: command a speed below the measured gait floor anyway.
        """
        seconds = self._clamp_seconds(seconds)
        extra = ["--vy", abs(speed_mps), "--seconds", seconds] + (["--force"] if force else [])
        return await self._move("strafe_left", "strafe", extra, seconds,
                                {"vy": abs(speed_mps)})

    @rpc()
    async def strafe_right(self, seconds: float = 1.5, speed_mps: float = DEFAULT_VY,
                           force: bool = False) -> dict:
        """Crab sideways to the right for a bounded time, then stop.

        Args:
            seconds: how long to hold the command, capped at 5 s.
            speed_mps: commanded lateral speed; the sign is supplied by this method.
            force: command a speed below the measured gait floor anyway.
        """
        seconds = self._clamp_seconds(seconds)
        extra = ["--vy", -abs(speed_mps), "--seconds", seconds] + (["--force"] if force else [])
        return await self._move("strafe_right", "strafe", extra, seconds,
                                {"vy": -abs(speed_mps)})

    @rpc()
    async def turn_left(self, seconds: float = 1.0, rate_rad_s: float = DEFAULT_WZ) -> dict:
        """Turn counter-clockwise in place for a bounded time, then stop.

        Args:
            seconds: how long to hold the command, capped at 5 s.
            rate_rad_s: commanded yaw rate. 0.70 rad/s is the planner's envelope cap.
        """
        seconds = self._clamp_seconds(seconds)
        return await self._move("turn_left", "turn",
                                ["--wz", abs(rate_rad_s), "--seconds", seconds],
                                seconds, {"wz": abs(rate_rad_s)})

    @rpc()
    async def turn_right(self, seconds: float = 1.0, rate_rad_s: float = DEFAULT_WZ) -> dict:
        """Turn clockwise in place for a bounded time, then stop.

        Args:
            seconds: how long to hold the command, capped at 5 s.
            rate_rad_s: commanded yaw rate; the sign is supplied by this method.
        """
        seconds = self._clamp_seconds(seconds)
        return await self._move("turn_right", "turn",
                                ["--wz", -abs(rate_rad_s), "--seconds", seconds],
                                seconds, {"wz": -abs(rate_rad_s)})

    @rpc()
    async def lie_down(self) -> dict:
        """Lie the robot down.

        On a Go2 this issues StandDown. On a Lite3 it only STOPS the robot: posture there is
        operator-controlled through the vendor app and the ROS bridge exposes no lie-down
        command. The result says which of the two happened rather than reporting success for
        both.
        """
        return await self._move("lie_down", "stand-down", [], 0.0, {})

    @rpc()
    async def stand(self) -> dict:
        """Enter a balanced stand, ready to walk.

        On a Lite3 this validates that the operator confirmed standing and high-level
        navigation mode; it does not change posture.
        """
        return await self._move("stand", "stand", [], 0.0, {})

    @rpc()
    async def stop(self) -> dict:
        """Stop the robot immediately. Never gated — a stop is always allowed.

        Deliberately outside the motion lock: if a nudge is in flight, this must not wait
        for it. The worker is a separate process and a zero velocity from a second one is
        still a zero velocity on the bus.
        """
        result = await self._bridge("stop")
        result["action"] = "stop"
        return result

    # ── status ───────────────────────────────────────────────────────────────
    @rpc()
    async def get_status(self) -> dict:
        """Pose, measured velocity, controller mode, and the armed checkpoint."""
        result = await self._bridge("status")
        try:
            result["active_model"] = self.store.active_model()
        except ModelStoreError as exc:
            result["active_model"] = None
            result["model_error"] = str(exc)
        result["motion_enabled"] = self.allow_motion
        result["busy"] = self._motion_lock.locked()
        return result

    @rpc()
    async def get_capabilities(self) -> dict:
        """What this platform can actually do, including where it differs from the other.

        Exists because the two robots are not interchangeable and a dashboard that renders
        one set of buttons for both would be lying about one of them. Read this to decide
        what to grey out.

        ⚠️ NOT named ``capabilities``. ``DeviceDriver.capabilities`` is a property on the
        base class and the runtime reads it to build the presence announcement. Defining an
        ``@rpc()`` with that name overrides it, the announcement is never published, and the
        device simply never appears on the mesh — no exception, no log line, and the driver
        otherwise runs perfectly. ``test_robot_driver.py`` asserts that no RPC on this class
        shadows a base-class member, so the next one is caught at test time instead of by
        an hour of staring at a working robot nothing can see.
        """
        from drive_bridge import GAIT_FLOORS
        floors = GAIT_FLOORS.get(self.platform, {})
        return {
            "platform": self.platform,
            "motion_enabled": self.allow_motion,
            "gait_floors_m_s": floors,
            "unmeasured_axes": [axis for axis, value in floors.items() if value is None],
            "lie_down_changes_posture": self.platform != "lite3",
            "posture_note": {
                "lite3": "Lite3 posture is operator-controlled through the vendor app; "
                         "lie_down stops the robot and does not lay it down.",
                "go2": "Go2 lie_down issues StandDown.",
                "sim": "The bench double records the posture change and moves nothing.",
            }[self.platform],
            "reverse_supported": True,
            "reverse_note": ("no rear sensing on either platform; reverse is open-loop into "
                             "unobserved space and is capped at 2 s"),
            "max_seconds": MAX_SECONDS,
            "bridge_python": self.bridge_python,
            "package_dir": str(self.store.package_dir),
        }

    # ── checkpoint RPCs ──────────────────────────────────────────────────────
    @rpc()
    async def list_models(self) -> dict:
        """Every checkpoint on this robot, with the armed one flagged.

        ``compatible_with_config`` is the field that matters: a checkpoint can be a perfectly
        valid network and still be un-runnable under the current config, because the
        training lidar range and the configured one must agree.
        """
        return {"ok": True, "models": self.store.list_models(),
                "active": self.store.active_model(),
                "config_path": str(self.store.config_path),
                "free_bytes": cloud_models.free_bytes(self.store.models_dir)}

    @rpc()
    async def select_model(self, name: str) -> dict:
        """Arm a checkpoint that is already on the robot, for the NEXT run.

        A live run holds its weights in memory and is unaffected — this rewrites
        ``model_path`` in config.json and nothing else. Refuses a checkpoint whose training
        constants disagree with the config, which would otherwise fail at the start of the
        next live run rather than here.

        Args:
            name: the checkpoint filename, as reported by list_models.
        """
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, self.store.select, name)
        except ModelStoreError as exc:
            return {"ok": False, "error": str(exc)}
        await self._announce(self.model_armed, name=result["active"],
                             previous=result["previous"] or "")
        result["ok"] = True
        return result

    @rpc()
    async def download_model(self, source: str, name: str = "") -> dict:
        """Pull a checkpoint from Cloud AI onto the robot.

        Args:
            source: an ``s3://bucket/key`` URI, or an ``http(s)://`` address — a presigned
                bucket URL or a server on the LAN both work.
            name: filename to install as. Defaults to the last path segment of the source.

        Returns:
            The installed checkpoint's inspection report, including its SHA-256 and whether
            it can be run under the current config. It is NOT armed by this call; arming is
            a separate, deliberate step.
        """
        loop = asyncio.get_running_loop()
        try:
            temp_path, filename, written = await loop.run_in_executor(
                None, lambda: cloud_models.fetch(source, name=name or None,
                                                 allow_http=self.allow_http))
        except (CloudFetchError, ModelStoreError) as exc:
            return {"ok": False, "error": str(exc)}

        try:
            installed = await loop.run_in_executor(
                None, lambda: self.store.install(filename, temp_path))
        except ModelStoreError as exc:
            # install() only moves a file it has already validated, so a failure here means
            # the download is not a usable checkpoint. Drop it rather than leaving it in /tmp
            # for a later run to trip over.
            await loop.run_in_executor(None, lambda: temp_path.unlink(missing_ok=True))
            return {"ok": False, "error": str(exc)}

        await self._announce(self.model_downloaded, name=installed["name"], source=source,
                             size_bytes=written, sha256=installed["sha256"])
        return {"ok": True, "model": installed, "source": source,
                "downloaded_bytes": written}

    @rpc()
    async def delete_model(self, name: str) -> dict:
        """Remove a checkpoint from the robot.

        Refuses the armed one — deleting it would leave config.json pointing at a file that
        is not there, and the failure would surface at the start of the next live run.

        Args:
            name: the checkpoint filename, as reported by list_models.
        """
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, self.store.remove, name)
        except ModelStoreError as exc:
            return {"ok": False, "error": str(exc)}
        await self._announce(self.model_deleted, name=result["removed"],
                             freed_bytes=result["freed_bytes"])
        result["ok"] = True
        return result

    @rpc()
    async def list_cloud_models(self, bucket: str, prefix: str = "") -> dict:
        """List the ``.npz`` checkpoints in an S3 bucket, newest first.

        Needs boto3 and AWS credentials on the robot. A direct http(s) source needs neither
        and can be pasted straight into download_model.

        Args:
            bucket: the S3 bucket name.
            prefix: an optional key prefix to narrow the listing.
        """
        try:
            objects = await asyncio.get_running_loop().run_in_executor(
                None, lambda: cloud_models.list_s3(bucket, prefix))
        except CloudFetchError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "objects": objects, "bucket": bucket, "prefix": prefix}

    # ── liveness ─────────────────────────────────────────────────────────────
    @periodic(interval=STATE_INTERVAL_S)
    async def publish_state(self) -> None:
        """Emit ``robot_state`` on a timer so the dashboard is live rather than a form.

        Skipped while a motion command holds the lock. Each of these is a subprocess that
        connects to DDS, reads and exits, and starting a second one while the robot is
        walking would put two clients on the bus for no benefit.
        """
        if self._motion_lock.locked():
            return
        try:
            state = await self._bridge("status")
        except Exception as exc:
            log.warning("state poll failed: %r", exc)
            return
        if not state.get("ok"):
            return
        try:
            active = self.store.active_model() or ""
        except ModelStoreError:
            active = ""
        await self._announce(self.robot_state, pose=state.get("pose") or {},
                             velocity=state.get("velocity") or [],
                             mode=state.get("mode") or "",
                             active_model=active)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a MAPPO quadruped on the Device Connect mesh.")
    parser.add_argument("--platform", default="sim", choices=("go2", "lite3", "sim"))
    parser.add_argument("--package", default=str(Path(__file__).resolve().parent.parent / "policy"),
                        help="The policy package: the directory holding config.json and models/.")
    parser.add_argument("--device-id", default=None,
                        help="Device id on the mesh. Defaults to mappo-<platform>.")
    parser.add_argument("--bridge-python", default=None,
                        help="Interpreter that can import this robot's SDK. On a Go2 that is "
                             "the SDK env's python, NOT this one.")
    parser.add_argument("--bridge-script", default=DEFAULT_BRIDGE)
    parser.add_argument("--iface", default=os.environ.get("GO2_DDS_IFACE", "eth0"),
                        help="DDS interface (Go2).")
    parser.add_argument("--messaging-url", action="append", default=None,
                        help="Router URL. Omit for D2D mode, which needs no infrastructure.")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--allow-motion", action="store_true",
                        help="Enable the motion RPCs. This is this file's --live: it needs a "
                             "clear area, an operator on the abort, and adequate battery.")
    parser.add_argument("--operator-ready", action="store_true",
                        help="Lite3: the operator has confirmed STANDING + high-level "
                             "navigation mode on the vendor interface.")
    parser.add_argument("--no-http-sources", action="store_true",
                        help="Refuse plain-http checkpoint sources; require https or s3.")
    # NOT "--allow-insecure": a store_true with default=True can never be switched off, so
    # the flag would read as a choice while being a constant. The negative flag is the one
    # that can actually change anything.
    parser.add_argument("--require-tls", action="store_true",
                        help="Require TLS on the mesh. The default is insecure, which is "
                             "what a demo LAN with no PKI needs.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(levelname)-7s  %(message)s")

    if args.platform != "sim" and args.bridge_python is None:
        log.warning("--bridge-python was not given, so the worker will run under %s. "
                    "On a real robot that interpreter almost certainly cannot import the "
                    "SDK; every command will fail with an import error.", sys.executable)

    driver = MappoRobotDriver(
        platform=args.platform, package_dir=args.package, bridge_script=args.bridge_script,
        bridge_python=args.bridge_python, iface=args.iface, allow_motion=args.allow_motion,
        operator_ready=args.operator_ready, allow_http=not args.no_http_sources)

    if args.allow_motion:
        log.warning("MOTION IS ENABLED. robot-stack/SAFETY.md applies: clear area, operator "
                    "on the abort, tether slack checked.")

    runtime = DeviceRuntime(
        driver=driver,
        device_id=args.device_id or f"mappo-{args.platform}",
        messaging_urls=args.messaging_url,
        tenant=args.tenant,
        allow_insecure=not args.require_tls,
    )
    try:
        asyncio.run(runtime.run())
    except KeyboardInterrupt:
        log.info("interrupted; shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
