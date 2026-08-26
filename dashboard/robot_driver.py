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

**A motion RPC returns as soon as the nudge STARTS, not when it finishes**, and the outcome
arrives on the event stream as ``motion_completed``. That is not a style preference — it is
what makes the stop button work:

    the edge runtime dispatches ONE RPC AT A TIME PER DEVICE.

``DeviceRuntime._cmd_subscription`` awaits the driver call inline in its message handler, so
a handler that blocks for five seconds makes the robot deaf to every other command for five
seconds — including ``stop``. Measured against a live driver with no dashboard in the path:
a stop issued one second into a five-second walk was delivered **4.17 s** later, i.e. only
once the walk had finished on its own. Putting ``stop()`` outside :attr:`_motion_lock` did
nothing about that, because the stop never reached the lock.

**And a delivered stop would not have been enough on its own.** The worker refreshes the
velocity at 10 Hz for the whole nudge, so a stop landing mid-walk is overwritten within
100 ms. :meth:`stop` therefore terminates the in-flight worker *first* — SIGTERM, so the
worker's ``SafeStop`` damps on the way out — and only then commands zero.

**Checkpoint operations are not motion and are not gated by it.** Swapping a checkpoint
rewrites one field in ``config.json`` and takes effect on the NEXT run — a live
``mappo_drive`` holds its weights in memory and cannot be affected. See ``model_store.py``,
which is where that argument is made properly and where the compatibility check that makes
a swap safe actually lives.

## Starting a run, and who is in charge while it is going

``select_model`` arms a checkpoint *for the next run*, and until ``start_run`` there was no
way to begin one: the run was a person typing ``mappo_drive.py`` over SSH. ``start_run`` /
``stop_run`` are that pair. The command line they build, where it runs, and why a remote
subprocess is the honest shape are all in ``run_control.py``; what lives here is the process,
the events, and the arbitration.

**One authority at a time, and it is named.** :attr:`_control_owner` is ``operator`` or
``policy``, never both and never neither. ``start_run(live=True)`` moves it to ``policy``;
``stop_run`` and ``stop`` move it back. While it is ``policy``, **the motion RPCs are
refused** — the refusal names the run and names ``stop_run``, and it is emitted as
``motion_refused`` like every other turn-down.

Refusing is the choice, and the alternative is what makes it one. If a manual nudge and a
policy tick both ran, both would write ``set_velocity`` at 10 Hz on the same bus and the last
writer would win — an arbitration with no owner, no record, and nothing an operator could
watch. Silently killing the run on a key press is the other alternative and it is worse: the
operator's press would end a run they may not have known was happening. So the pad goes
inert, loudly, and taking it back is one deliberate call.

**Nothing in that gates a stop.** ``stop`` and ``stop_run`` are ungated in every state, and
``stop`` is the one an operator reaches for: they take the legs away from the policy first,
the in-flight nudge second, and command zero last.

**A run does not transfer control when it is not driving.** ``start_run(live=False)`` is the
shadow rung: perception, policy and telemetry with no ``--live``, so nothing commands a leg
and the pad stays with the operator. Claiming otherwise would put a lock on the page for a
process that could not have moved anything.

## What the event stream carries

``motion_started`` / ``motion_completed`` / ``motion_refused`` / ``motion_interrupted`` —
every command, including the ones that were turned down, because a refusal is the
interesting event and a dashboard that only shows successes teaches an operator that nothing
happened. ``interrupted`` is separate from ``refused`` on purpose: one ran and was stopped,
the other never ran, and an operator should not see their own stop in the colour of a fault.
``model_armed`` / ``model_downloaded`` / ``model_deleted`` — every change to what the robot
is carrying. ``robot_state`` — pose, velocity and controller mode on a timer, which is what
makes the page live rather than a form.

``peer_pose`` — this robot's pose at 10 Hz, for ANOTHER robot to avoid, and off unless
``--publish-pose`` asks for it. It is a separate event from ``robot_state`` rather than a
faster one because ``robot_state`` is skipped while a motion command holds the lock and
costs a subprocess per sample: both are disqualifying for a channel that matters most
while this robot is walking and is needed ten times a second. It is the input that lets a
peer be avoided with no detector, no marker and no training — see
``integration/peer_source.py``, and ``peer_link.py`` for the other end.

Run on the robot:
    python3 robot_driver.py --platform go2 --package ../policy --allow-motion
Off-robot, against the bench double:
    python3 robot_driver.py --platform sim --package ../policy --allow-motion

``python3 test_robot_driver.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from device_connect_edge import DeviceRuntime
from device_connect_edge.drivers import DeviceDriver, emit, periodic, rpc
from device_connect_edge.types import DeviceIdentity, DeviceStatus

import camera_source
import cloud_models
import run_control
from camera_source import CameraUnavailable
from cloud_models import CloudFetchError
from drive_bridge import (
    DEFAULT_VX,
    DEFAULT_VY,
    DEFAULT_WZ,
    MAX_POSE_STREAM_HZ,
    MAX_REVERSE_SECONDS,
    MAX_SECONDS,
    POSE_STREAM_HZ,
    BridgeError,
    check_gait_floor,
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

#: The disc another robot should model this one as, metres, per platform.
#:
#: **Half-diagonal, not half-length**, which is the whole reason this is a table and not a
#: parameter with a friendly default. ``avoidance.NavConfig.robot_radius_m`` already made
#: this call for the Go2 and its comment states it: "Go2 is ~0.70 x 0.31 m; half-diagonal,
#: rounded up" — ``hypot(0.35, 0.155) = 0.383``, so 0.40. The half-LENGTH, 0.35, leaves the
#: corners outside the disc, and a peer's heading is not something the robot avoiding it
#: gets to choose.
#:
#: Published by the robot rather than assumed by whoever is avoiding it, for the same
#: reason ``model_sources`` is: it is a property of this machine, and two platforms on one
#: mesh have different ones. The colour-prop path's 0.15 m default is a bin and would
#: under-model either of these by more than half.
#:
#: ⚠️ The Lite3 figure is derived the same way from the vendor's 610 x 370 mm and has never
#: been measured on the robot. ⚠️ A Go2 **Wheel** carries wheel modules the base Go2's
#: 0.70 x 0.31 m does not describe; ``mappo_drive.py --peer-radius`` overrides this when the
#: peer is one and somebody has a tape measure.
PEER_FOOTPRINT_M = {"go2": 0.40, "lite3": 0.36, "sim": 0.40}

#: How long to wait for the pose worker's first line before calling it dead. It pays the
#: same cold-Jetson SDK import and DDS discovery every other worker does, so it is the
#: same budget — measured in tens of seconds, not in control periods.
POSE_FIRST_LINE_S = BRIDGE_TIMEOUT_S

#: Lines of a run's output kept for :meth:`get_status`. A ring, not a log: the run writes
#: its own telemetry to a file named in ``run_started``, and this is only what an operator
#: sees on the page without going and fetching it.
RUN_LOG_LINES = 40

#: Cap on ``run_output`` events per run, after which the ring is the only copy. ``mappo_drive``
#: prints a startup banner, its warnings and a per-run report, but the vendored stack under it
#: can be chattier than that — and a driver that turned a talkative run into thousands of mesh
#: events would be competing for the channel a ``motion_refused`` has to arrive on. Same
#: argument as the camera being events on their own subject rather than RPC replies.
RUN_OUTPUT_EVENTS = 120

#: How long to wait before restarting a dead pose worker. Short, because every second it
#: is down is a second the robot avoiding this one is stopped — but not zero, or a worker
#: that fails on startup becomes a spawn loop nobody can read the log of.
POSE_RESTART_S = 1.0


class MappoRobotDriver(DeviceDriver):
    """A Go2 or Lite3 Venture carrying a MAPPO checkpoint, on the Device Connect mesh."""

    device_type = "mappo_quadruped"

    def __init__(self, *, platform: str = "sim", package_dir: str = "../policy",
                 bridge_script: str = DEFAULT_BRIDGE, bridge_python: str | None = None,
                 iface: str = "eth0", allow_motion: bool = False,
                 operator_ready: bool = False, allow_http: bool = True,
                 simulate: bool = False, camera_replay_dir: str = "", camera_url: str = "",
                 model_sources: list | None = None, publish_pose_hz: float = 0.0,
                 run_profile=None) -> None:
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
        #: Present as ``platform`` — its gait floors, its posture rules, its identity — while
        #: driving the bench double instead of a robot. For a demo host with no robots
        #: attached. It is advertised in ``get_capabilities`` and badged in the dashboard,
        #: because a demo fleet that is indistinguishable from a real one is a hazard rather
        #: than a convincing demo: somebody will eventually press a key believing a robot is
        #: on the other end of it.
        self.simulate = simulate
        self.camera_replay_dir = camera_replay_dir
        #: A live JPEG endpoint, used when the driver cannot host the platform camera
        #: itself -- see HttpCameraSource. Takes precedence over the replay directory.
        self.camera_url = camera_url
        #: Where THIS robot's checkpoints can come from — a list, because a deployment
        #: legitimately has more than one and the interesting question is which. Advertised
        #: by the ROBOT rather than configured in the dashboard: it is a property of the
        #: deployment the robot sits in, and two robots on one mesh can pull from different
        #: places, which a dashboard-level setting cannot express.
        #:
        #: ⚠️ Every address here must be routable FROM THE ROBOT. The download runs on the
        #: robot, not in the operator's browser, so an address only a laptop can reach gives
        #: a field that looks right and fails on fetch.
        self.model_sources = list(model_sources or [])
        self.store = ModelStore(package_dir)
        self._motion_lock = asyncio.Lock()
        #: The in-flight worker process, so a stop can terminate it. Guarded by a threading
        #: lock because it is assigned from an executor thread and read from the event loop.
        self._worker = None
        self._worker_guard = threading.Lock()
        self._motion_task = None
        self._camera = None
        self._last_pose: dict = {}
        self._camera_task = None
        self._camera_fps = camera_source.DEFAULT_FPS
        self._camera_error = ""
        self._viewers = camera_source.Viewers()
        #: Rate for ``peer_pose``, or 0.0 for off. NOT interest-gated the way the camera
        #: is, and that is the deliberate difference between the two: a camera feed costs
        #: 200-320 KB/s and nobody is harmed by it starting late, while a pose is ~200
        #: bytes a sample and is what stops another robot's legs. Making it wait for a
        #: consumer to ask adds "the request did not arrive" as a way for a peer to become
        #: invisible, to a channel whose entire job is to not be silently absent.
        self._pose_hz = min(float(publish_pose_hz or 0.0), MAX_POSE_STREAM_HZ)
        self._pose_task = None
        #: Where an autonomous run happens and what is constant about it, or ``None`` for a
        #: driver that cannot start one. See ``run_control.RunProfile``: this is a property
        #: of the DEPLOYMENT and never of a request, which is what keeps a browser out of
        #: the robot's command line.
        self.run_profile = run_profile
        #: The run in flight, as ``run_control.RunRecord``, or ``None``.
        self._run = None
        self._run_proc = None
        self._run_task = None
        self._run_deadline_task = None
        self._run_lines: deque = deque(maxlen=RUN_LOG_LINES)
        self._run_emitted = 0
        #: Who is allowed to command a leg right now — ``operator`` or ``policy``, never
        #: both and never neither. It is a single value rather than two flags on purpose: a
        #: pair of booleans has a state meaning "both" and a state meaning "neither", and
        #: neither of those is a thing this robot can be in.
        self._control_owner = "operator"
        self._control_reason = "no run has been started"
        self._control_since = time.monotonic()

    # ── identity ─────────────────────────────────────────────────────────────
    @property
    def identity(self) -> DeviceIdentity:
        vendor = {"go2": "Unitree", "lite3": "Deep Robotics", "sim": "none"}[self.platform]
        model = {"go2": "Go2 EDU", "lite3": "Lite3 Venture",
                 "sim": "bench double"}[self.platform]
        return DeviceIdentity(
            device_type=self.device_type,
            manufacturer=vendor,
            model=f"{model} (SIMULATED)" if self.simulate else model,
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
        if self._pose_hz > 0.0:
            # Started here rather than on request: see `_pose_hz`. "This robot is on the
            # mesh" and "this robot's pose is on the mesh" should be the same fact.
            self._pose_task = asyncio.create_task(self._pose_stream_loop())

    async def disconnect(self) -> None:
        """Stop the robot on the way out.

        The runtime calls this on a clean shutdown. It is best-effort by design — if the
        process is being torn down because the worker environment is broken, a failure here
        must not mask that.
        """
        if self._pose_task is not None:
            # Cancelled, not awaited. The task's own `finally` SIGTERMs the worker, and if
            # the loop tears down before that runs, the worker's stdout pipe closes with
            # this process and it exits on the BrokenPipeError it already handles. There
            # is no velocity to leave latched either way — see `command_pose_stream`.
            self._pose_task.cancel()
            self._pose_task = None
        # BEFORE the bridge stop, and for the same reason ``stop`` does it in that order:
        # the policy refreshes the velocity at its own rate, so a zero commanded while it is
        # still running is overwritten before anybody sees it.
        #
        # ⚠️ This runs on a CLEAN shutdown. A driver that is hard-killed never reaches it,
        # and a run launched over SSH is a process on another machine that does not die with
        # its launcher — see ``run_control``'s module docstring. What bounds it then is the
        # run's own ``--max-seconds`` and ``visual_nav``'s teardown, not this line.
        if self._run is not None:
            try:
                await self._end_run("the driver is shutting down")
            except Exception as exc:
                log.warning("ending the run on shutdown failed: %r", exc)
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
    async def motion_interrupted(self, action: str, reason: str):
        """A running nudge was cut short — almost always because someone pressed stop.

        Its own event rather than a ``motion_refused``. A refusal means the command never
        ran; this one ran and was ended deliberately, and an operator scanning the stream
        should not see their own stop reported to them in the same colour as a fault.
        """

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
    async def run_started(self, run_id: str, live: bool, policy_mode: str,
                          heading_servo: str, seconds: float, remote: bool,
                          argv: list, command: list, outputs: dict, note: str):
        """An autonomous MAPPO run was started, with the command line it was started with.

        ``argv`` and ``command`` are both here and they are not the same thing. ``argv`` is
        what a person reads; ``command`` is what was handed to the operating system, SSH
        prefix and shell line included, and on a remote run reporting only the first would
        describe a local run that never happened.

        There is no commit in this payload and there cannot be: the deployed tree is not a
        checkout. ``note`` says so. What this event can name is the command.
        """

    @emit()
    async def run_output(self, run_id: str, line: str, truncated: bool):
        """One line the run printed. Capped at ``RUN_OUTPUT_EVENTS`` per run."""

    @emit()
    async def run_finished(self, run_id: str, reason: str, exit_code: int,
                           elapsed_s: float, tail: list):
        """A run ended, whether by itself, by an operator, or by the watchdog.

        ``reason`` distinguishes those, in the same spirit ``motion_interrupted`` is separate
        from ``motion_refused``: an operator scanning the stream should not see their own
        stop in the colour of a crash.
        """

    @emit()
    async def run_refused(self, reason: str):
        """A run was turned down — by the motion gate, by a mode the robot would ignore, or
        because something was already in flight."""

    @emit()
    async def control_changed(self, owner: str, reason: str):
        """Who may command a leg: ``operator`` or ``policy``.

        Its own event and not a field on ``robot_state``, because ``robot_state`` is a
        periodic poll that is SKIPPED while a run is driving — which is precisely the
        interval during which this is the thing an operator most needs to be told. A state
        that only appears on a timer that stops is a state nobody is told about.
        """

    @emit()
    async def camera_frame(self, jpeg_b64: str, seq: int, fps: float):
        """One frame from the front RGB camera.

        An EVENT and not an RPC reply, deliberately. The runtime dispatches one RPC at a time
        per device, so polling frames over that channel would occupy it continuously and make
        the robot deaf to ``stop``. Events are pub/sub and do not queue behind commands.
        """

    @emit()
    async def robot_state(self, pose: dict, velocity: list, mode: str, active_model: str):
        """Periodic liveness: where the robot is and what it is carrying."""

    @emit()
    async def peer_pose(self, pose: dict, velocity: list, radius_m: float, seq: int,
                        sample_age_s: float, platform: str, ok: bool):
        """Where this robot is, at 10 Hz, so another robot on the mesh can avoid it.

        NOT ``robot_state`` at a faster interval, and the difference is not the rate.
        ``robot_state`` is a dashboard liveness poll: it is SKIPPED while a motion command
        holds the lock, and each one is a subprocess that connects to DDS, reads and exits.
        Both of those are disqualifying here. A peer pose is needed most precisely while
        the peer is walking, which is exactly when that lock is held, and 10 Hz of process
        starts on a Jetson is not a rate this robot can produce.

        ``pose`` is in THIS robot's odom frame, which begins where it was switched on and
        means nothing to anybody else. ``velocity`` is (vx, vy, wz) in this robot's BODY
        frame — the estimator's own frame, per ``mappo_bridge.VELOCITY_FRAME``. Turning
        either into something the consumer can plan against is the consumer's job and it
        needs a measured transform between the two robots; see ``integration/peer_source``.

        ``sample_age_s`` is how long ago the estimator was read, measured here, on this
        host, as a DURATION. It is not a timestamp and must not become one: two robots on
        a demo LAN with no NTP share no clock, so a duration crosses the mesh and an
        instant does not.
        """

    # ── the worker bridge ────────────────────────────────────────────────────
    def _bridge_blocking(self, command: str, extra=None) -> dict:
        """Run one ``drive_bridge.py`` command and parse its last stdout line.

        The timeout path sends SIGTERM before SIGKILL. That ordering is load-bearing: the
        worker installs a damp on SIGTERM, and ``SportClient.Move`` has no dead-man timeout,
        so a bare kill of a walking robot leaves the last velocity latched on the bus.
        """
        # --platform carries the RULES and --backend what is driven. When simulating they
        # differ, which is what makes a simulated Go2 refuse a sub-gait-floor speed exactly
        # as a real one does. Passing "sim" as the platform instead would hand the demo the
        # bench double's floors of zero and quietly delete the refusal.
        cmd = [self.bridge_python, self.bridge_script, command,
               "--platform", self.platform, "--iface", self.iface]
        if self.simulate:
            cmd += ["--backend", "sim"]
        cmd += [str(a) for a in (extra or [])]
        if self.allow_motion:
            cmd.append("--allow-motion")
        if self.operator_ready:
            cmd.append("--operator-ready")

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            # Published so stop() can terminate it. Only motion workers are worth
            # interrupting: a status read holds no velocity, so killing it gains nothing.
            # ⚠️ It is NOT milliseconds, which this comment claimed until it was measured.
            # On the Go2's Jetson `drive_bridge.py status` costs 1.93-1.98 s over three
            # runs (2026-08-26), nearly all of it the cold SDK import and DDS discovery
            # that every invocation pays. Against STATE_INTERVAL_S of 5.0 s that is ~39%
            # of the poll period with a DDS client on the bus.
            if command not in ("status", "stop"):
                with self._worker_guard:
                    self._worker = proc
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
        with self._worker_guard:
            if self._worker is proc:
                self._worker = None

        for line in reversed((stdout or "").strip().splitlines()):
            try:
                return json.loads(line)
            except ValueError:
                continue
        if proc.returncode is not None and proc.returncode < 0:
            # Killed by a signal, which for a motion worker means stop() interrupted it.
            # That is a normal outcome, not a fault, and it must not be reported as one.
            return {"ok": False, "interrupted": True,
                    "error": f"the {command} worker was stopped before it finished"}
        return {"ok": False, "error": f"worker exited {proc.returncode} with no JSON result; "
                                      f"stderr tail: {(stderr or '')[-400:]!r}"}

    async def _bridge(self, command: str, extra=None) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._bridge_blocking, command, extra)

    def _precheck(self, action: str, commanded: dict, force: bool) -> str:
        """Refuse a speed measured not to walk, synchronously, before anything is accepted.

        The worker checks this too, and that check is still the backstop. But since motion
        RPCs became non-blocking the worker's verdict arrives as an EVENT some milliseconds
        later, so the reply said "accepted" and the refusal turned up in the alert list —
        and a ``motion_started`` was emitted for a motion that never started. Found by
        pressing 0.21 m/s at a simulated Go2 on the demo host and watching it be accepted.

        The check is a pure table lookup (``check_gait_floor`` takes no robot), so doing it
        here costs nothing and restores the immediate answer.
        """
        axis, speed = None, 0.0
        if action == "walk_forward":
            axis, speed = "forward", commanded.get("vx", 0.0)
        elif action in ("strafe_left", "strafe_right"):
            axis, speed = "lateral", commanded.get("vy", 0.0)
        elif action in ("turn_left", "turn_right"):
            axis, speed = "yaw", commanded.get("wz", 0.0)
        # walk_back is deliberately absent: the floor is a measurement of the FORWARD gait
        # and applying it to reverse is the axis conflation issue #42 is about.
        if axis is None:
            return ""
        try:
            check_gait_floor(self.platform, axis, speed, force)
        except BridgeError as exc:
            return str(exc)
        return ""

    async def _move(self, action: str, command: str, extra: list, seconds: float,
                    commanded: dict, force: bool = False) -> dict:
        """Accept a motion command and return — the nudge runs in the background.

        Returning here rather than at the end of the nudge is what keeps the robot able to
        hear ``stop``; see the module docstring for the measurement. The caller gets
        ``accepted``, and the outcome — including the distance actually travelled — arrives
        as a ``motion_completed`` event.
        """
        if not self.allow_motion:
            reason = ("this device was started without --allow-motion, so it is "
                      "status-and-checkpoints only")
            await self._announce(self.motion_refused, action=action, reason=reason)
            return {"ok": False, "refused": True, "error": reason}

        if self._control_owner != "operator":
            reason = self._policy_has_the_legs(action)
            await self._announce(self.motion_refused, action=action, reason=reason)
            return {"ok": False, "refused": True, "policy_driving": True,
                    "control_owner": self._control_owner, "error": reason}

        if self._motion_lock.locked():
            reason = "the robot is already executing a motion command"
            await self._announce(self.motion_refused, action=action, reason=reason)
            return {"ok": False, "refused": True, "busy": True, "error": reason}

        # Before the lock and before motion_started: a command refused here never started,
        # and announcing that it did would put a lie in the operator's event log.
        floor_refusal = self._precheck(action, commanded, force)
        if floor_refusal:
            await self._announce(self.motion_refused, action=action, reason=floor_refusal)
            return {"ok": False, "refused": True, "error": floor_refusal}

        # Taken here, not inside the task: between this line and the check above there is no
        # suspension point, so two rapid calls cannot both get past the guard. Acquiring it
        # inside the task instead would leave exactly that gap.
        await self._motion_lock.acquire()
        await self._announce(self.motion_started, action=action,
                             commanded=commanded, seconds=seconds)
        self._motion_task = asyncio.create_task(self._run_motion(action, command, extra))
        return {"ok": True, "accepted": True, "action": action, "seconds": seconds,
                "note": "the nudge is running; its measured outcome arrives as a "
                        "motion_completed event"}

    async def _run_motion(self, action: str, command: str, extra: list) -> None:
        """Run one accepted nudge to completion and report it on the event stream."""
        try:
            result = await self._bridge(command, extra)
            if result.get("ok"):
                await self._announce(
                    self.motion_completed,
                    action=action,
                    travelled_m=result.get("travelled_m") or 0.0,
                    turned_deg=result.get("turned_deg") or 0.0,
                    delivered_fraction=result.get("delivered_fraction") or 0.0,
                    warning=result.get("warning") or "",
                )
            elif result.get("interrupted"):
                await self._announce(self.motion_interrupted, action=action,
                                     reason=result.get("error", "stopped"))
            else:
                await self._announce(self.motion_refused, action=action,
                                     reason=result.get("error", "unknown failure"))
        except Exception as exc:
            log.warning("%s failed: %r", action, exc)
            await self._announce(self.motion_refused, action=action, reason=repr(exc))
        finally:
            self._motion_lock.release()

    def _policy_has_the_legs(self, action: str) -> str:
        """Why a manual command is refused while the policy is driving.

        Long, and deliberately so. The one thing an operator must not have to work out
        mid-run is who is in charge, and a two-word refusal ("busy") is how they would end
        up pressing the key again to find out.
        """
        run_id = self._run.run_id if self._run is not None else "?"
        return (f"the MAPPO policy is driving this robot (run {run_id}), so {action} is "
                f"refused. It is not queued and it was not sent: a manual command and a "
                f"policy tick would both write a velocity on the same bus at 10 Hz and the "
                f"last writer would win, which is not an arbitration anybody can watch. "
                f"To drive by hand, call stop_run — it ends the run and hands the pad back. "
                f"To stop the robot now, press stop; neither of those is gated.")

    def _terminate_worker(self) -> bool:
        """SIGTERM the in-flight motion worker, if there is one. Returns whether there was.

        SIGTERM and not SIGKILL: the worker installs ``SafeStop``, which turns the signal
        into a damp, and ``SportClient.Move`` has no dead-man timeout — a hard kill would
        leave the last velocity latched on the bus.
        """
        with self._worker_guard:
            proc = self._worker
            self._worker = None
        if proc is None or proc.poll() is not None:
            return False
        try:
            proc.terminate()
        except OSError:
            return False
        return True

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
                                {"vx": abs(speed_mps)}, force=force)

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
            speed_mps: commanded lateral speed. ⚠️ 0.20 m/s is the **Go2's** measured
                lateral floor (issue #42, 2026-08-19: vy 0.15 moved 0.010 m with no gait,
                vy 0.20 walked 3/3). The **Lite3's** has never been measured, so a Lite3
                strafe may produce no gait. This docstring said the opposite until
                2026-08-26, and Device Connect publishes it.
            force: command a speed below the measured gait floor anyway.
        """
        seconds = self._clamp_seconds(seconds)
        extra = ["--vy", abs(speed_mps), "--seconds", seconds] + (["--force"] if force else [])
        return await self._move("strafe_left", "strafe", extra, seconds,
                                {"vy": abs(speed_mps)}, force=force)

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
                                {"vy": -abs(speed_mps)}, force=force)

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
        """Stop the robot immediately: the policy, the nudge, and the velocity. Never gated.

        Deliberately outside the motion lock: if a nudge is in flight, this must not wait
        for it. The worker is a separate process and a zero velocity from a second one is
        still a zero velocity on the bus.

        **There are two things that can be commanding this robot and this ends both.** Until
        ``start_run`` existed there was only the nudge worker, and this method terminated it.
        A ``mappo_drive`` run is the second, and it is the more urgent of the two: it holds
        the legs indefinitely and refreshes its velocity every tick, so a stop that only
        commanded zero would be overwritten inside one control period and the robot would
        visibly pause and carry on — the exact failure this method was written for, in a form
        that no test covered until the run existed. ``STOP ALL`` and every per-row stop on
        the dashboard invoke THIS function, so they inherit it rather than growing a second
        implementation that could disagree.

        The order is: policy, then nudge, then zero. Commanding zero first would be
        commanding it into a bus that two other processes are still writing to.

        ⚠️ **A run stop that does not confirm makes this reply a failure.** A remote run is
        signalled over a second SSH round trip, and if that does not come back the policy may
        still be driving. The dashboard classifies a stop by ``ok``, so an unconfirmed one has
        to be ``ok: false`` — the operator needs to be sent to the physical abort, not shown a
        tick.
        """
        ended = await self._end_run("stop")
        # Then the in-flight nudge. Its worker refreshes the velocity at 10 Hz too, so a stop
        # issued while it is still running is overwritten within 100 ms.
        interrupted = self._terminate_worker()
        result = await self._bridge("stop")
        result["action"] = "stop"
        result["interrupted_motion"] = interrupted
        result["ended_run"] = bool(ended.get("was_running"))
        result["run_stop_confirmed"] = ended.get("confirmed")
        if ended.get("was_running") and not ended.get("confirmed"):
            result["ok"] = False
            result["error"] = (
                f"the velocity was zeroed, but run {ended.get('run_id')} did NOT confirm it "
                f"stopped within {run_control.RUN_STOP_TIMEOUT_S:.0f}s. Assume the policy is "
                f"STILL DRIVING and use the physical abort. " + str(ended.get("error") or ""))
        return result

    # ── run control ──────────────────────────────────────────────────────────
    @rpc()
    async def start_run(self, seconds: float = run_control.DEFAULT_RUN_SECONDS,
                        policy_mode: str = "supervised", heading_servo: str = "off",
                        arm_motion: bool = False) -> dict:
        """Start a MAPPO run. **By default it cannot move the robot**, and that is the point.

        ⛔ ``robot-stack/SAFETY.md`` governs this.

        **With no arguments this is the scene check**, the same command line
        ``run-smoke.sh scene`` builds: the camera, the detector, the policy, the planner's
        veto and the telemetry, with no ``--live``. ``--live`` is the only flag in
        ``mappo_drive.py`` that commands a leg, so a run without it has no path to one —
        an absent capability rather than a checked permission. It needs no ``--allow-motion``
        and it is the right thing to press first, every time.

        **Motion is opted into twice.** The driver must have been started with
        ``--allow-motion`` (an operator's decision, at a command line, with somebody on the
        abort), AND this call must pass ``arm_motion``. Missing either is a **refusal with
        the reason in it, never a quiet dry run** — a run that starts and cannot move looks
        exactly like a run that starts and will not, and the difference is a demo spent
        diagnosing a robot.

        It also carries one bound a nudge does not need. A nudge is capped at 5 s by the
        worker; an autonomous run has no ceiling of its own, so ``seconds`` is clamped, sent
        as ``--max-seconds``, and backstopped by a watchdog here.

        The reply carries the whole command line. It cannot carry a commit — the deployed
        tree is not a git checkout, and on 2026-08-26 the lab Go2's ``~/mappo-main`` matched
        no single commit on ``main``: 67 commits behind on ``mappo_drive.py``, 95 on
        ``config.json``. Read ``argv``.

        Args:
            seconds: how long the run may last, clamped to 120 s. Passed as
                ``--max-seconds`` and backstopped by a watchdog here.
            policy_mode: ``supervised`` keeps the planner's feasibility veto. ``raw`` removes
                it, and in the closed-loop simulation the raw policy collided in every
                configuration tested while the supervised one did not.
            heading_servo: ``off`` (the default, and the only setting that has not driven a
                robot into something), ``goal``, or ``travel`` — issue #16's control law.
                Always sent explicitly, because the deployed tree's default is whatever it
                was on the day it was copied.
            arm_motion: ⛔ the second of two gates. ``true`` adds ``--live`` and hands the
                legs to the policy, and needs the driver to have been started with
                ``--allow-motion``. Default ``false``: a scene check that cannot move.
        """
        live = bool(arm_motion)
        if self.run_profile is None:
            return await self._refuse_run(run_control.unsupported()["reason"])
        if self._run is not None:
            return await self._refuse_run(
                f"run {self._run.run_id} is already in flight; stop it before starting "
                f"another. Two policies on one robot is not a thing this driver will build.")
        if self._motion_lock.locked():
            return await self._refuse_run(
                "the robot is executing a manual motion command; a run must not start on top "
                "of one. Wait for it, or press stop.")

        # Cheap refusals first: the gate, the parameter check and the argv are all decided
        # from a table, and connecting to DDS to find out otherwise would cost seconds on a
        # Jetson and put a client on the bus for a run that was never going to start. Same
        # ordering, and the same reason, as `drive_bridge.dispatch` planning before it loads.
        run_id = run_control.new_run_id()
        try:
            argv = run_control.build_run_argv(
                self.run_profile, seconds=seconds, policy_mode=policy_mode,
                heading_servo=heading_servo, live=live, allow_motion=self.allow_motion,
                run_id=run_id)
        except run_control.RunRefused as exc:
            return await self._refuse_run(str(exc))

        # The preflight is a READ, and it is the only place a robot that would accept this
        # run and never act on it gets to say so. One bridge round trip — 1.94 s on the lab
        # Go2, measured, nearly all of it cold SDK import and DDS discovery.
        #
        # ⚠️ It BLOCKS the command channel for that time: the edge runtime dispatches one
        # RPC at a time per device, which is the measurement the whole non-blocking motion
        # design comes from. It is acceptable here and only here, because of what is true
        # during the window: no run is in flight (refused above), no nudge holds the lock
        # (refused above), and nothing has been spawned yet. The robot is stationary, so a
        # `stop` arriving 1.9 s late has nothing it arrived late for. The moment anything
        # IS moving — after `_launch` — this method has already returned.
        state = await self._bridge("status")
        if not state.get("ok"):
            return await self._refuse_run(
                "the pre-run status read failed, so nothing is known about the state of this "
                "robot: " + str(state.get("error") or "the worker returned no result"))
        try:
            mode_note = run_control.check_control_mode(
                self.platform, state.get("mode"), simulated=self.simulate)
        except run_control.RunRefused as exc:
            return await self._refuse_run(str(exc))

        return await self._launch(run_id, argv, live=live, policy_mode=policy_mode,
                                  heading_servo=heading_servo,
                                  seconds=run_control.clamp_seconds(seconds),
                                  mode_note=mode_note or "")

    @rpc()
    async def stop_run(self, reason: str = "the operator took control") -> dict:
        """End the autonomous run and give the motion pad back. Never gated.

        **This is how a person takes control mid-run**, and it is one call rather than two
        because there is exactly one authority and moving it is one act. A separate
        "take control" that did not end the run would leave two things able to command a leg
        and call that an arbitration.

        It stops the robot as well as the policy: SIGTERM to the run first, so its own
        teardown damps, then a zero velocity as the backstop. It never sends SIGKILL —
        ``SAFETY.md`` §0 — because a hard kill is the opposite of a stop.

        ``stop`` does everything this does and also terminates an in-flight manual nudge. Use
        that one in an emergency; this is the one for taking the robot back.

        Args:
            reason: recorded in ``run_finished`` and in the event stream, so a stopped run
                says who stopped it and why.
        """
        ended = await self._end_run(str(reason or "the operator took control"))
        if not ended.get("was_running"):
            return {"ok": True, "was_running": False,
                    "note": "no run was in flight; the motion pad was already the operator's",
                    "control_owner": self._control_owner}
        # A zero after the policy is gone, for the same reason `stop` commands one: the run
        # is another process and this driver cannot see what it left on the bus.
        zeroed = await self._bridge("stop")
        ended["ok"] = bool(ended.get("confirmed")) and bool(zeroed.get("ok"))
        ended["velocity_zeroed"] = bool(zeroed.get("ok"))
        ended["control_owner"] = self._control_owner
        if not ended.get("confirmed"):
            ended["error"] = (
                f"run {ended.get('run_id')} did not confirm it stopped within "
                f"{run_control.RUN_STOP_TIMEOUT_S:.0f}s. Assume the policy is STILL DRIVING "
                f"and use the physical abort. " + str(ended.get("error") or ""))
        return ended

    # ── run control internals ────────────────────────────────────────────────
    async def _refuse_run(self, reason: str) -> dict:
        """Turn a run down, and put the turn-down on the event stream.

        Same rule as ``motion_refused``: a refusal is the interesting event, and a stream
        that carries only the runs that started teaches an operator that nothing happened.
        """
        await self._announce(self.run_refused, reason=reason)
        return {"ok": False, "refused": True, "error": reason}

    async def _hand_control(self, owner: str, reason: str) -> bool:
        """Move the single authority, and announce it only when it actually moved.

        Re-asserting the current owner updates the explanation and emits nothing. An event
        per assertion would put a ``control_changed`` on the stream every time a run ended
        that had never taken control, which trains an operator to ignore the one event on
        this device that says who is driving.
        """
        if owner == self._control_owner:
            self._control_reason = reason
            return False
        self._control_owner = owner
        self._control_reason = reason
        self._control_since = time.monotonic()
        log.warning("control -> %s (%s)", owner, reason)
        await self._announce(self.control_changed, owner=owner, reason=reason)
        return True

    def _control_snapshot(self) -> dict:
        """Who has the legs, as ``get_status`` reports it.

        ``held_s`` is a DURATION measured between two readings of this host's monotonic
        clock, and it must not become a timestamp — the same rule ``peer_pose`` follows, for
        the same measured reason: the lab Go2 has no working RTC and its wall clock was 56
        years behind on 2026-08-26.
        """
        return {
            "owner": self._control_owner,
            "reason": self._control_reason,
            "held_s": round(max(0.0, time.monotonic() - self._control_since), 1),
            "manual_motion_allowed": self._control_owner == "operator",
        }

    def _run_snapshot(self) -> dict:
        if self._run is None:
            return {"active": False, "supported": self.run_profile is not None}
        snapshot = self._run.snapshot(time.monotonic(), tail=list(self._run_lines)[-10:])
        snapshot["active"] = True
        snapshot["supported"] = True
        return snapshot

    async def _launch(self, run_id: str, argv: list, *, live: bool, policy_mode: str,
                      heading_servo: str, seconds: float, mode_note: str) -> dict:
        """Spawn one run and start reporting it. Everything that could refuse already has."""
        profile = self.run_profile
        pidfile = run_control.pidfile_for(profile, run_id) if profile.launch_prefix else ""
        command = run_control.launch_command(profile, argv, pidfile)
        # ``cwd`` applies to the process THIS machine starts. On a remote run that process
        # is ``ssh`` and the directory that matters is the ``cd`` inside the shell line;
        # setting cwd here would apply this machine's path to the other machine's tree.
        cwd = None if profile.launch_prefix else (profile.workdir or None)
        # A remote run gets its variables from `export` lines inside the shell command, a
        # local one from the spawn — same source, two renderings, because there is
        # deliberately no shell in the local path to sit between a SIGTERM and the run.
        try:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=cwd, env=run_control.local_env(profile),
                stdout=asyncio.subprocess.PIPE,
                # Merged, not separate: ``visual_nav`` warns on stderr and ``mappo_drive``
                # reports on stdout, and an operator reading two interleaved streams out of
                # order is how a warning ends up under the line it was about.
                stderr=asyncio.subprocess.STDOUT)
        except OSError as exc:
            return await self._refuse_run(
                f"could not start the run: {exc}. The command was {command[0]!r}; on a "
                f"remote profile that is the ssh client, and on a local one it is the "
                f"interpreter that has to be able to import this robot's stack.")

        record = run_control.RunRecord(
            run_id=run_id, argv=list(argv), command=list(command), live=bool(live),
            policy_mode=policy_mode, heading_servo=heading_servo, seconds=seconds,
            remote=bool(profile.launch_prefix), pidfile=pidfile,
            outputs=run_control.output_paths(profile, run_id),
            started_at=time.monotonic())
        record.pid = process.pid
        self._run = record
        self._run_proc = process
        self._run_lines = deque(maxlen=RUN_LOG_LINES)
        self._run_emitted = 0
        # Only a LIVE run takes the legs. A dry run commands nothing, so locking the motion
        # pad against it would be a claim about who is driving that is not true.
        if live:
            await self._hand_control(
                "policy", f"run {run_id} is driving ({policy_mode}, servo {heading_servo})")
        note = " ".join(part for part in (mode_note, run_control.TREE_NOTE) if part)
        await self._announce(
            self.run_started, run_id=run_id, live=bool(live), policy_mode=policy_mode,
            heading_servo=heading_servo, seconds=seconds, remote=record.remote,
            argv=record.argv, command=record.command, outputs=record.outputs, note=note)

        self._run_task = asyncio.create_task(self._pump_run(record, process))
        budget = seconds + run_control.RUN_STARTUP_GRACE_S
        self._run_deadline_task = asyncio.create_task(self._run_deadline(record, budget))
        return {
            "ok": True, "started": True, "run_id": run_id, "live": bool(live),
            "arm_motion": bool(live),
            "can_move": bool(live),
            "policy_mode": policy_mode, "heading_servo": heading_servo, "seconds": seconds,
            "remote": record.remote, "pid": process.pid, "argv": record.argv,
            "command": record.command, "outputs": record.outputs,
            "control_owner": self._control_owner, "watchdog_s": round(budget, 1),
            "named_commit": None, "note": note,
            "outcome": "the run's output arrives as run_output events and its end as "
                       "run_finished; this reply only says it was launched",
        }

    async def _pump_run(self, record, process) -> None:
        """Turn the run's output into events, and its exit into exactly one run_finished.

        The only place a run is finalised, whoever ended it. ``_end_run`` signals the
        process and returns; the pipe then closes, this loop falls out, and the report
        happens here. Two reporters would be two ``run_finished`` events for one run, and
        the second would be the one an operator saw.
        """
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip()
                if not text:
                    continue
                self._run_lines.append(text)
                await self._emit_run_line(record, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("the run's output pump failed: %r", exc)
        finally:
            with contextlib.suppress(Exception):
                await self._finish_run(record, process)

    async def _emit_run_line(self, record, text: str) -> None:
        """One output line as an event, until the cap, then one line saying so."""
        if self._run_emitted < RUN_OUTPUT_EVENTS:
            self._run_emitted += 1
            await self._announce(self.run_output, run_id=record.run_id, line=text,
                                 truncated=False)
            return
        if self._run_emitted == RUN_OUTPUT_EVENTS:
            self._run_emitted += 1
            await self._announce(
                self.run_output, run_id=record.run_id, truncated=True,
                line=(f"[dashboard] this run has printed more than {RUN_OUTPUT_EVENTS} "
                      f"lines; the rest are in get_status's tail and in the run's own "
                      f"telemetry, not on the event stream"))

    async def _finish_run(self, record, process) -> None:
        """Report one ended run and give the pad back."""
        try:
            exit_code = await asyncio.wait_for(process.wait(), run_control.RUN_STOP_TIMEOUT_S)
        except asyncio.TimeoutError:
            exit_code = process.returncode
        except Exception as exc:
            log.warning("could not read the run's exit status: %r", exc)
            exit_code = process.returncode
        record.exit_code = exit_code
        reason = record.finished_reason or _natural_end(exit_code)
        elapsed = round(max(0.0, time.monotonic() - record.started_at), 2)
        if self._run is record:
            self._run = None
            self._run_proc = None
            if self._run_deadline_task is not None:
                self._run_deadline_task.cancel()
                self._run_deadline_task = None
        await self._hand_control("operator", f"run {record.run_id} ended: {reason}")
        log.info("run %s ended after %.1f s: %s (exit %s)",
                 record.run_id, elapsed, reason, exit_code)
        await self._announce(
            self.run_finished, run_id=record.run_id, reason=reason,
            # -1 rather than None: the field is an integer in the schema, and a run whose
            # exit status could not be read is not a run that exited 0.
            exit_code=-1 if exit_code is None else int(exit_code),
            elapsed_s=elapsed, tail=list(self._run_lines)[-10:])

    async def _end_run(self, reason: str) -> dict:
        """Take the legs back from the policy. Never raises, never sends SIGKILL.

        Returns ``{"was_running": False}`` when there is nothing to end, so both callers can
        treat "no run" as an ordinary outcome rather than an error — pressing stop on an idle
        robot is not a mistake.

        ⛔ **There is no escalation to SIGKILL here or anywhere in this path.** ``SAFETY.md``
        §0 is the cardinal rule and it was written from an incident: a hard kill leaves the
        last command latched with nothing left to update or damp it. A run that ignores
        SIGTERM is reported as unstopped, which sends the operator to the physical abort —
        the right place — instead of this driver improvising a kill.
        """
        record, process = self._run, self._run_proc
        if record is None or process is None:
            return {"was_running": False}
        if not record.finished_reason:
            record.finished_reason = reason

        # The pad comes back BEFORE the waiting. The policy has been signalled; making an
        # operator sit out an SSH round trip before their keys work again would be a lock
        # held for the convenience of the reporting.
        await self._hand_control("operator", f"run {record.run_id} stopped: {reason}")

        error = ""
        stop_cmd = (run_control.stop_command(self.run_profile, record.pidfile)
                    if self.run_profile is not None else None)
        if stop_cmd is not None:
            # A REMOTE run. Signalling the local ssh client closes a socket; the process on
            # the far end never hears about it. So the stop is a second connection that
            # signals the pid the launch recorded.
            error = await self._signal_remote(stop_cmd)
        else:
            with contextlib.suppress(ProcessLookupError, OSError):
                process.terminate()

        confirmed = True
        try:
            await asyncio.wait_for(process.wait(), run_control.RUN_STOP_TIMEOUT_S)
        except asyncio.TimeoutError:
            confirmed = False
            error = error or (
                f"the run had not exited {run_control.RUN_STOP_TIMEOUT_S:.0f}s after SIGTERM")
            if record.remote:
                # Close this side's connection so the driver stops holding a dead one. It
                # does NOT stop the run: that is the point of reporting it unconfirmed.
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.terminate()
        return {"was_running": True, "run_id": record.run_id, "confirmed": confirmed,
                "reason": record.finished_reason, "error": error, "remote": record.remote}

    async def _signal_remote(self, command: list) -> str:
        """Run one remote stop command. Returns "" on success or why not."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE)
        except OSError as exc:
            return f"could not run the remote stop ({command[0]!r}): {exc}"
        try:
            _out, err = await asyncio.wait_for(proc.communicate(),
                                               run_control.RUN_STOP_TIMEOUT_S)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()
            return (f"the remote stop did not answer within "
                    f"{run_control.RUN_STOP_TIMEOUT_S:.0f}s")
        if proc.returncode:
            tail = (err or b"").decode("utf-8", "replace")[-200:]
            return f"the remote stop exited {proc.returncode}: {tail!r}"
        return ""

    async def _run_deadline(self, record, budget: float) -> None:
        """End a run that outlived its bound. The nudge's 5 s cap, at a run's scale.

        A backstop and not the primary bound: ``--max-seconds`` is passed to the run itself
        and the run is expected to honour it. This fires when it does not — a stalled
        detector, a wedged camera open, a stale tree whose ``--max-seconds`` means something
        else — and those are exactly the cases where nothing else would end it.
        """
        try:
            await asyncio.sleep(budget)
        except asyncio.CancelledError:
            return
        if self._run is not record:
            return
        log.warning("run %s outlived its %.1f s budget; ending it", record.run_id, budget)
        await self._end_run(
            f"the watchdog ended this run: it outlived its {budget:.0f}s budget "
            f"(--max-seconds {record.seconds:g} plus {run_control.RUN_STARTUP_GRACE_S:.0f}s "
            f"of startup). It did not stop itself.")
        with contextlib.suppress(Exception):
            await self._bridge("stop")

    # ── camera ───────────────────────────────────────────────────────────────
    @rpc()
    async def watch_camera(self, fps: float = camera_source.DEFAULT_FPS) -> dict:
        """Start (or keep alive) the front-camera feed, emitted as ``camera_frame`` events.

        Call it repeatedly while watching. Interest expires, so a closed browser tab stops
        the stream by itself rather than leaving a robot emitting to nobody.

        Args:
            fps: frames per second, capped at 15 — roughly what the Go2 sensor delivers.
        """
        self._camera_fps = camera_source.clamp_fps(fps)
        self._viewers.note_interest()
        if self._camera_task is None or self._camera_task.done():
            self._camera_error = ""
            self._camera_task = asyncio.create_task(self._camera_loop())
        return {"ok": True, "fps": self._camera_fps, "watching": True,
                "error": self._camera_error}

    @rpc()
    async def stop_camera(self) -> dict:
        """Stop the camera feed now rather than waiting for interest to expire."""
        self._viewers.clear()
        return {"ok": True, "watching": False}

    async def _camera_loop(self) -> None:
        """Emit frames while someone is watching, then release the camera.

        The camera is opened on a worker thread: both real backends block on a device or an
        SDK call, and doing that on the event loop would stall every other RPC on this
        driver — including the stop this whole design exists to keep fast.
        """
        loop = asyncio.get_running_loop()
        seq = 0
        try:
            self._camera = await loop.run_in_executor(
                None, lambda: camera_source.open_source(
                    self.platform, iface=self.iface,
                    camera_url=self.camera_url or None,
                    replay_dir=self.camera_replay_dir or None,
                    replay_label=f"REPLAY · {self.platform}",
                    pose_fn=lambda: (self._last_pose or {})))
        except CameraUnavailable as exc:
            self._camera_error = str(exc)
            log.warning("camera unavailable: %s", exc)
            return

        try:
            while self._viewers.watching():
                started = loop.time()
                try:
                    raw = await loop.run_in_executor(None, self._camera.read)
                    encoded = camera_source.encode(raw)
                except CameraUnavailable as exc:
                    self._camera_error = str(exc)
                    break
                except Exception as exc:
                    log.warning("frame read failed: %r", exc)
                    encoded = None
                if encoded:
                    seq += 1
                    await self._announce(self.camera_frame, jpeg_b64=encoded, seq=seq,
                                         fps=self._camera_fps)
                # Pace from the START of the cycle so encoding time comes out of the interval
                # rather than being added to it; otherwise the real rate is always under the
                # requested one by however long a frame took.
                elapsed = loop.time() - started
                await asyncio.sleep(max(0.0, (1.0 / self._camera_fps) - elapsed))
        finally:
            await loop.run_in_executor(None, self._camera.close)
            self._camera = None
            log.info("camera released after %d frames", seq)

    # ── peer pose ────────────────────────────────────────────────────────────
    async def _pose_stream_loop(self) -> None:
        """Run the pose worker and turn each of its lines into a ``peer_pose`` event.

        One long-lived subprocess, restarted if it dies. The alternative — a subprocess per
        sample, the way ``publish_state`` does it — cannot reach 10 Hz on a Jetson, where
        SDK import and DDS discovery are most of :data:`BRIDGE_TIMEOUT_S`.

        A worker that exits is restarted after :data:`POSE_RESTART_S` and the gap is logged
        at warning level. It is worth being loud about: while this is down, every robot
        avoiding this one is stopped, because their side treats a pose that stopped
        arriving as a peer whose position is unknown rather than as a peer standing still.
        """
        cmd = [self.bridge_python, self.bridge_script, "pose-stream",
               "--platform", self.platform, "--iface", self.iface,
               "--hz", str(self._pose_hz)]
        if self.simulate:
            cmd += ["--backend", "sim"]
        radius = PEER_FOOTPRINT_M.get(self.platform, PEER_FOOTPRINT_M["go2"])
        log.info("publishing peer_pose at %.1f Hz, modelled as a %.2f m disc",
                 self._pose_hz, radius)
        seq = 0
        while True:
            try:
                seq = await self._pump_pose_worker(cmd, radius, seq)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("pose worker failed: %r", exc)
            log.warning("pose worker stopped after %d samples; restarting in %.1f s. "
                        "Any robot avoiding this one is holding until it does.",
                        seq, POSE_RESTART_S)
            await asyncio.sleep(POSE_RESTART_S)

    async def _pump_pose_worker(self, cmd: list, radius: float, seq: int) -> int:
        """One worker's lifetime. Returns the sample counter it reached."""
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            first = True
            while True:
                # Only the FIRST line gets a deadline, and it is the full worker budget:
                # everything before it is SDK import and DDS discovery. After that the
                # worker is pacing itself and a wait is just the next sample.
                line = await asyncio.wait_for(
                    process.stdout.readline(), POSE_FIRST_LINE_S if first else None)
                if not line:
                    return seq                       # the worker exited
                first = False
                record = self._pose_line(line)
                if record is None:
                    continue
                seq += 1
                await self._announce(
                    self.peer_pose, pose=record["pose"], velocity=record["velocity"],
                    radius_m=radius, seq=seq, sample_age_s=record["sample_age_s"],
                    platform=self.platform, ok=record["ok"])
        finally:
            # SIGTERM first, as everywhere else in this file. The pose worker latches no
            # velocity so it needs no damp, but a bare kill would still leave a DDS client
            # to time out on the bus, and the next worker would be the second one on it.
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), 5.0)

    @staticmethod
    def _pose_line(line: bytes):
        """One worker stdout line as an emittable record, or ``None`` to skip it.

        Non-JSON lines are skipped rather than fatal: the SDK prints on import, which is
        the same reason ``_bridge_blocking`` reads the result backwards from the end of
        stdout rather than parsing all of it.

        A line with no usable ``mono_s`` is DROPPED, and that is the one judgement call
        here. Its age cannot be computed, and a safety input that cannot be dated is not a
        safety input — emitting it with an invented age of zero would make an undatable
        sample look like the freshest one there is. Dropping it ages the consumer out
        instead, which stops the other robot, which is the outcome an unreadable clock
        should have.
        """
        try:
            record = json.loads(line.decode("utf-8", "replace"))
        except ValueError:
            return None
        if not isinstance(record, dict):
            return None
        mono = record.get("mono_s")
        if not isinstance(mono, (int, float)) or isinstance(mono, bool):
            log.warning("pose worker line carried no readable clock; dropping it")
            return None
        return {
            # Same host, same CLOCK_MONOTONIC, two processes — so this subtraction is
            # meaningful, and it is the only one in the whole chain that is. What crosses
            # the mesh is the result, a duration.
            "sample_age_s": max(0.0, time.monotonic() - float(mono)),
            "pose": record.get("pose") or {},
            "velocity": record.get("velocity") or [],
            "ok": bool(record.get("ok")),
        }

    # ── status ───────────────────────────────────────────────────────────────
    @rpc()
    async def get_status(self) -> dict:
        """Pose, velocity, controller mode, the armed checkpoint — and who is driving.

        ``control`` is the field to read before deciding what a button should do. It says
        whether the legs belong to the ``operator`` or to the ``policy``, why, and for how
        long; ``run`` says what the policy is running if it is. Both are here rather than on
        ``robot_state`` because ``robot_state`` is a timer that is SKIPPED while a run is in
        flight, which is the exact interval this matters in.

        ``mode_accepts_motion`` is the Go2 finding turned into a field. On 2026-08-26 this
        robot answered ``mode='mcf'`` — not a sport mode — in which ``SportClient.Move`` is
        ignored, so a robot in that state accepts every command and never steps. It is
        ``false`` there, with the whole sentence in ``mode_note``, so a page can say so
        before an operator spends a demo diagnosing a tether.
        """
        result = await self._bridge("status")
        try:
            result["active_model"] = self.store.active_model()
        except ModelStoreError as exc:
            result["active_model"] = None
            result["model_error"] = str(exc)
        result["motion_enabled"] = self.allow_motion
        result["busy"] = self._motion_lock.locked()
        result["control"] = self._control_snapshot()
        result["run"] = self._run_snapshot()
        try:
            note = run_control.check_control_mode(self.platform, result.get("mode"),
                                                  simulated=self.simulate)
            result["mode_accepts_motion"] = True
            result["mode_note"] = note or ""
        except run_control.RunRefused as exc:
            # Reported, not raised. ``get_status`` is the call a page makes on a robot that
            # is not moving, and the answer to "why is nothing happening" belongs in it.
            result["mode_accepts_motion"] = False
            result["mode_note"] = str(exc)
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
            "simulated": self.simulate,
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
            "camera": {
                "available": self.platform in ("sim", "go2", "lite3"),
                "synthetic": (self.platform == "sim" and not self.camera_replay_dir
                              and not self.camera_url),
                "replay": bool(self.camera_replay_dir) and not self.camera_url,
                "remote_url": self.camera_url or None,
                "default_fps": camera_source.DEFAULT_FPS,
                "max_fps": camera_source.MAX_FPS,
                "error": self._camera_error,
                "note": ("the Lite3 capture is exclusive, so a live visual_nav run holds the "
                         "camera and this cannot" if self.platform == "lite3" else
                         "synthetic frames; there is no camera on this platform"
                         if self.platform == "sim" else
                         "the Go2 video service is shared, so a run and this can both read"),
            },
            "cloud": {
                "sources": self.model_sources,
                # Stated so the dashboard can say whose reachability matters. Nothing here
                # verifies the robot can reach any of them; that is what pressing Browse
                # finds out, and it finds out from the robot's side of the network.
                "resolved_by": "the robot, not the browser",
            },
            "max_seconds": MAX_SECONDS,
            "bridge_python": self.bridge_python,
            "package_dir": str(self.store.package_dir),
            # How, and whether, this robot can be handed to the policy. Advertised by the
            # ROBOT for the same reason ``cloud.sources`` is: the machine, the interpreter
            # and the tree are properties of the deployment, and two robots on one mesh do
            # not share them. ``command_preview`` is what start_run would run right now,
            # gate included, so the page can show the command BEFORE the press.
            "run": (run_control.describe(self.run_profile, allow_motion=self.allow_motion)
                    if self.run_profile is not None else run_control.unsupported()),
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
    async def list_cloud_models(self, bucket: str = "", prefix: str = "",
                                index_url: str = "") -> dict:
        """List the checkpoints a Cloud AI source advertises, newest first.

        Two kinds of source, and neither is privileged over the other. ``index_url`` reads a
        JSON index from a self-hosted model server; ``bucket`` lists an S3 bucket and needs
        boto3 plus AWS credentials on the robot. Browsing a bucket but not a server would
        say, in the shape of this API, that S3 is the real answer — which is backwards for a
        deployment whose checkpoints live on its own hardware.

        Args:
            bucket: an S3 bucket name.
            prefix: an optional key prefix, for the S3 form.
            index_url: a model server's JSON index — takes precedence when both are given.
        """
        loop = asyncio.get_running_loop()
        try:
            if index_url:
                objects = await loop.run_in_executor(
                    None, lambda: cloud_models.list_http_index(
                        index_url, allow_http=self.allow_http))
                return {"ok": True, "objects": objects, "source": index_url,
                        "kind": "server"}
            if not bucket:
                return {"ok": False,
                        "error": "give either an S3 bucket or a model server index URL"}
            objects = await loop.run_in_executor(
                None, lambda: cloud_models.list_s3(bucket, prefix))
        except CloudFetchError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "objects": objects, "bucket": bucket, "prefix": prefix,
                "kind": "s3"}

    # ── liveness ─────────────────────────────────────────────────────────────
    @periodic(interval=STATE_INTERVAL_S)
    async def publish_state(self) -> None:
        """Emit ``robot_state`` on a timer so the dashboard is live rather than a form.

        Skipped while a motion command holds the lock, and skipped for the whole of an
        autonomous run. Each of these is a subprocess that connects to DDS, reads and exits,
        and starting a second one while the robot is walking would put two clients on the
        bus for no benefit. A run makes that argument stronger, not weaker: it holds the
        camera, the estimator and the bus for its whole length at 10 Hz, and on the Go2's
        Jetson one ``status`` costs 1.94 s of cold SDK import and DDS discovery — 39% of
        this interval, spent competing with a control loop.

        The cost is that pose and mode go stale on the fleet row for the length of a run.
        That is the right trade and it is not a blackout: ``run_output``, ``run_finished``
        and ``control_changed`` carry the run, and ``--publish-pose`` is the channel built
        for a pose that is needed while the robot is moving.
        """
        if self._motion_lock.locked() or self._run is not None:
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
        self._last_pose = state.get("pose") or {}
        await self._announce(self.robot_state, pose=state.get("pose") or {},
                             velocity=state.get("velocity") or [],
                             mode=state.get("mode") or "",
                             active_model=active)


def _natural_end(exit_code) -> str:
    """How a run that nobody stopped ended, in words an operator can act on."""
    if exit_code is None:
        return "the run ended and its exit status could not be read"
    if exit_code == 0:
        return "the run completed"
    if exit_code < 0:
        return f"the run was ended by signal {-exit_code}"
    return f"the run exited {exit_code}; read its output for why"


def _load_sources(path: str) -> list:
    """Read the advertised checkpoint sources, or none at all.

    A malformed file is a startup error rather than a silent empty list: a robot that
    quietly advertises nowhere to get checkpoints from looks, on the dashboard, exactly
    like a robot that was configured with no sources on purpose.
    """
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    sources = data.get("sources") if isinstance(data, dict) else data
    if not isinstance(sources, list):
        raise ValueError(f"{path}: expected a list of sources, or an object with 'sources'")
    for source in sources:
        if not isinstance(source, dict) or not source.get("label"):
            raise ValueError(f"{path}: every source needs at least a 'label'")
    return sources


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
    parser.add_argument("--simulate", action="store_true",
                        help="Present as --platform but drive the bench double. For a demo "
                             "host with no robots. Advertised, and badged in the dashboard.")
    parser.add_argument("--model-sources", default="",
                        help="JSON file listing where this robot's checkpoints can come "
                             "from; advertised to the dashboard so an operator picks a named "
                             "source instead of remembering a URL. Every address in it MUST "
                             "be routable from the robot — the download runs here, not in "
                             "the operator's browser.")
    parser.add_argument("--publish-pose", nargs="?", type=float, const=POSE_STREAM_HZ,
                        default=0.0, metavar="HZ",
                        help="Publish this robot's pose as peer_pose events so another "
                             f"robot on the mesh can avoid it. Default {POSE_STREAM_HZ} Hz, "
                             f"capped at {MAX_POSE_STREAM_HZ}. Off unless asked for: it "
                             "holds a persistent SDK worker on the DDS bus. It needs no "
                             "--allow-motion, because it only reads.")
    parser.add_argument("--run-profile", default="",
                        help="JSON file describing where an autonomous MAPPO run happens: "
                             "the machine, the interpreter, the working directory and the "
                             "deployment's own flags. Without it start_run refuses, because "
                             "there is nothing to start. ⚠️ On a Go2 the run CANNOT be a "
                             "child of this driver -- mappo_drive.py needs the robot's "
                             "Python 3.8 SDK and this driver needs 3.11 -- so its "
                             "launch_prefix is an ssh command and the run is a subprocess "
                             "on the robot. See dashboard/run_control.py.")
    parser.add_argument("--camera-url", default="",
                        help="pull frames from an HTTP endpoint that returns one JPEG "
                             "per GET, instead of opening a camera locally. For a robot "
                             "whose interpreter cannot host this driver: the SDK call "
                             "stays on the robot, only the frame crosses. Live, and "
                             "deliberately unlabelled -- unlike --camera-replay-dir.")
    parser.add_argument("--camera-replay-dir", default="",
                        help="Serve JPEGs from this directory as the camera feed. Each frame "
                             "is labelled REPLAY in the pixels.")
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

    if args.simulate:
        log.warning("SIMULATED %s: presenting this platform's rules against the bench "
                    "double. No robot is attached.", args.platform)
    if args.platform != "sim" and not args.simulate and args.bridge_python is None:
        log.warning("--bridge-python was not given, so the worker will run under %s. "
                    "On a real robot that interpreter almost certainly cannot import the "
                    "SDK; every command will fail with an import error.", sys.executable)

    driver = MappoRobotDriver(
        platform=args.platform, package_dir=args.package, bridge_script=args.bridge_script,
        bridge_python=args.bridge_python, iface=args.iface, allow_motion=args.allow_motion,
        operator_ready=args.operator_ready, allow_http=not args.no_http_sources,
        simulate=args.simulate, camera_replay_dir=args.camera_replay_dir,
        camera_url=args.camera_url,
        model_sources=_load_sources(args.model_sources),
        publish_pose_hz=args.publish_pose,
        run_profile=run_control.load_profile(args.run_profile) if args.run_profile else None)

    if args.allow_motion:
        log.warning("MOTION IS ENABLED. robot-stack/SAFETY.md applies: clear area, operator "
                    "on the abort, tether slack checked.")
    if driver.run_profile is not None:
        # Printed at startup rather than only in an RPC reply, because the person reading
        # this line is the one who can still fix a wrong path before a demo, and because a
        # remote launch is a command on somebody else's machine that nobody else will see.
        preview = run_control.describe(driver.run_profile, allow_motion=args.allow_motion)
        where = ("the robot, over " + " ".join(driver.run_profile.launch_prefix)
                 if driver.run_profile.launch_prefix else "this machine")
        log.info("start_run() runs, on %s:", where)
        log.info("    %s", " ".join(preview["command_preview"]))
        if preview["armed_command_preview"]:
            log.warning("start_run(arm_motion=true) runs, and it MOVES THE ROBOT:")
            log.warning("    %s", " ".join(preview["armed_command_preview"]))
        else:
            log.info("start_run(arm_motion=true) is REFUSED: no --allow-motion on this "
                     "driver. The scene check above still runs.")
        log.warning("  %s", preview["tree_note"])

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
