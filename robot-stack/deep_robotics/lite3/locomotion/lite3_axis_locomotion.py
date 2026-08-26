#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Profile-gated Lite3 simple-axis locomotion over a verified high-level state link.

This transport preserves the existing navigator interface while replacing the legacy
``320/325/321`` complex-velocity sender with the vendor's moving-mode axis protocol. It does
not select control or moving mode: the operator must explicitly establish the vendor-approved
state before a live run. No nonzero primitive is built in; a validated local profile is required.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_robotics.lite3.locomotion.lite3_axis_udp import (
    AXIS_LIMIT,
    FORWARD_AXIS_CODE,
    HEARTBEAT_CODE,
    LATERAL_AXIS_CODE,
    YAW_AXIS_CODE,
    axis_packet,
)
from deep_robotics.lite3.locomotion.lite3_udp_locomotion import (
    DEFAULT_COMMAND_PORT,
    DEFAULT_MOTION_HOST,
    DEFAULT_STATE_PORT,
    Lite3LinkLost,
    Lite3UdpLocomotion,
)

AXIS_PROFILE_SCHEMA = "lite3-axis-profile/v1"
AXIS_RATE_HZ = 20.0
HEARTBEAT_HZ = 4.0
COMMAND_TTL_S = 0.15
STOP_SECONDS = 2.0
DEFAULT_LOCAL_PORT = 20001

FORWARD_DEAD_ZONE = 6553
LATERAL_DEAD_ZONE = 12553
YAW_DEAD_ZONE = 9553
ALLOWED_MOVING_GAIT_STATES = frozenset((0, 2, 4, 5, 6, 13))


class AxisProfileError(ValueError):
    """An axis profile cannot safely convert a requested navigation component."""


@dataclass(frozen=True)
class AxisValues:
    """One three-axis vendor joystick command."""

    forward: int = 0
    lateral: int = 0
    yaw: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("forward", self.forward),
            ("lateral", self.lateral),
            ("yaw", self.yaw),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise AxisProfileError(f"{name} axis value must be an integer")
            if not -AXIS_LIMIT <= value <= AXIS_LIMIT:
                raise AxisProfileError(
                    f"{name} axis value must be within {-AXIS_LIMIT}..{AXIS_LIMIT}"
                )

    @property
    def is_zero(self) -> bool:
        return self == AxisValues()


@dataclass(frozen=True)
class AxisProfile:
    """A locally evidenced mapping from navigation intent signs to vendor axis primitives."""

    forward_positive: int | None
    forward_negative: int | None
    lateral_positive: int | None
    lateral_negative: int | None
    yaw_positive: int | None
    yaw_negative: int | None
    linear_deadband_m_s: float
    yaw_deadband_rad_s: float
    allowed_gait_states: tuple[int, ...]
    evidence: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Reject unsafe direct construction as well as malformed JSON profiles."""
        try:
            evidence = dict(self.evidence)
        except (TypeError, ValueError) as error:
            raise AxisProfileError(
                "axis profile evidence must contain name/reference pairs"
            ) from error
        for name, value, dead_zone in (
            ("forward_positive", self.forward_positive, FORWARD_DEAD_ZONE),
            ("forward_negative", self.forward_negative, FORWARD_DEAD_ZONE),
            ("lateral_positive", self.lateral_positive, LATERAL_DEAD_ZONE),
            ("lateral_negative", self.lateral_negative, LATERAL_DEAD_ZONE),
            ("yaw_positive", self.yaw_positive, YAW_DEAD_ZONE),
            ("yaw_negative", self.yaw_negative, YAW_DEAD_ZONE),
        ):
            self._validate_primitive(name, value, dead_zone, evidence)
        self._validate_deadband("linear_m_s", self.linear_deadband_m_s)
        self._validate_deadband("yaw_rad_s", self.yaw_deadband_rad_s)
        if not self.allowed_gait_states:
            raise AxisProfileError("axis profile requires at least one allowed moving gait state")
        for gait_state in self.allowed_gait_states:
            if isinstance(gait_state, bool) or not isinstance(gait_state, int):
                raise AxisProfileError("axis profile gait states must be integers")
            if gait_state not in ALLOWED_MOVING_GAIT_STATES:
                raise AxisProfileError(
                    f"axis profile gait state {gait_state} is not a documented moving gait"
                )

    @classmethod
    def load(cls, path: Path) -> AxisProfile:
        """Load a versioned local profile; no physical command values are shipped by default."""
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise AxisProfileError(f"cannot read axis profile {path}: {error}") from None
        if not isinstance(data, dict):
            raise AxisProfileError("axis profile must be a JSON object")
        if data.get("schema") != AXIS_PROFILE_SCHEMA:
            raise AxisProfileError(
                f"axis profile schema must be {AXIS_PROFILE_SCHEMA!r}, got {data.get('schema')!r}"
            )
        primitives = data.get("primitives")
        if not isinstance(primitives, dict):
            raise AxisProfileError("axis profile requires an object field 'primitives'")
        deadband = data.get("input_deadband")
        if not isinstance(deadband, dict):
            raise AxisProfileError("axis profile requires an object field 'input_deadband'")
        evidence = data.get("evidence", {})
        if not isinstance(evidence, dict):
            raise AxisProfileError("axis profile field 'evidence' must be an object")
        gait_states = data.get("allowed_gait_states")
        if not isinstance(gait_states, list):
            raise AxisProfileError("axis profile field 'allowed_gait_states' must be a list")

        return cls(
            forward_positive=primitives.get("forward_positive"),
            forward_negative=primitives.get("forward_negative"),
            lateral_positive=primitives.get("lateral_positive"),
            lateral_negative=primitives.get("lateral_negative"),
            yaw_positive=primitives.get("yaw_positive"),
            yaw_negative=primitives.get("yaw_negative"),
            linear_deadband_m_s=deadband.get("linear_m_s"),
            yaw_deadband_rad_s=deadband.get("yaw_rad_s"),
            allowed_gait_states=tuple(gait_states),
            evidence=tuple(evidence.items()),
        )

    @staticmethod
    def _validate_primitive(name: str, value: Any, dead_zone: int,
                            evidence: dict[str, str]) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise AxisProfileError(f"primitive {name} must be an integer or null")
        if not -AXIS_LIMIT <= value <= AXIS_LIMIT:
            raise AxisProfileError(
                f"primitive {name} must be within {-AXIS_LIMIT}..{AXIS_LIMIT}"
            )
        if abs(value) <= dead_zone:
            raise AxisProfileError(
                f"primitive {name}={value} is inside the documented dead zone ±{dead_zone}"
            )
        expected_positive = name.endswith("_positive")
        if (value > 0) != expected_positive:
            direction = "positive" if expected_positive else "negative"
            raise AxisProfileError(f"primitive {name} must be {direction}")
        provenance = evidence.get(name)
        if not isinstance(provenance, str) or not provenance.strip():
            raise AxisProfileError(
                f"primitive {name} requires a non-empty evidence reference"
            )

    @staticmethod
    def _validate_deadband(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AxisProfileError(f"input_deadband.{name} must be a finite positive number")
        if not math.isfinite(float(value)) or value <= 0.0:
            raise AxisProfileError(f"input_deadband.{name} must be a finite positive number")

    def map_velocity(self, vx: float, vy: float, yaw: float) -> AxisValues:
        """Map shared body-frame intent to explicitly evidenced vendor primitives.

        The shared navigator uses positive lateral velocity and positive yaw for left. The
        vendor moving-mode axes use positive raw values for right, so those two axes invert.
        """
        return AxisValues(
            forward=self._map_component(
                vx,
                self.linear_deadband_m_s,
                self.forward_positive,
                self.forward_negative,
                "forward",
            ),
            lateral=self._map_component(
                vy,
                self.linear_deadband_m_s,
                self.lateral_negative,
                self.lateral_positive,
                "lateral",
            ),
            yaw=self._map_component(
                yaw,
                self.yaw_deadband_rad_s,
                self.yaw_negative,
                self.yaw_positive,
                "yaw",
            ),
        )

    @staticmethod
    def _map_component(value: float, deadband: float, positive: int | None,
                       negative: int | None, name: str) -> int:
        if not math.isfinite(value):
            raise AxisProfileError(f"{name} navigation component must be finite")
        if abs(value) < deadband:
            return 0
        primitive = positive if value > 0.0 else negative
        if primitive is None:
            direction = "positive" if value > 0.0 else "negative"
            raise AxisProfileError(
                f"axis profile has no physically evidenced {direction} {name} primitive"
            )
        return primitive


@dataclass(frozen=True)
class _AxisSetpoint:
    values: AxisValues
    updated_at: float


class AxisStreamSender:
    """Continuously send the freshest valid three-axis command with TTL-based zeroing."""

    def __init__(self, *, host: str = DEFAULT_MOTION_HOST,
                 port: int = DEFAULT_COMMAND_PORT, source_address: str | None = None,
                 local_port: int = DEFAULT_LOCAL_PORT, axis_rate_hz: float = AXIS_RATE_HZ,
                 heartbeat_hz: float = HEARTBEAT_HZ, command_ttl_s: float = COMMAND_TTL_S,
                 stop_seconds: float = STOP_SECONDS, socket_factory=socket.socket,
                 clock=time.monotonic, sleep=time.sleep) -> None:
        self._validate_rate(axis_rate_hz, AXIS_RATE_HZ, "axis rate")
        self._validate_rate(heartbeat_hz, 2.0, "heartbeat rate")
        if not math.isfinite(command_ttl_s) or not 0.0 < command_ttl_s < 0.25:
            raise ValueError("command TTL must be finite, positive, and below the 250 ms watchdog")
        if not math.isfinite(stop_seconds) or stop_seconds <= 0.0:
            raise ValueError("stop seconds must be finite and positive")
        if not 0 <= local_port <= 65535:
            raise ValueError("local port must be within 0..65535")

        self._host = host
        self._port = port
        self._source_address = source_address
        self._local_port = local_port
        self._axis_interval = 1.0 / axis_rate_hz
        self._heartbeat_interval = 1.0 / heartbeat_hz
        self._command_ttl_s = command_ttl_s
        self._stop_seconds = stop_seconds
        self._socket_factory = socket_factory
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._setpoint = _AxisSetpoint(AxisValues(), float("-inf"))
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._failure: OSError | None = None

    @staticmethod
    def _validate_rate(value: float, minimum: float, name: str) -> None:
        if not math.isfinite(value) or value < minimum:
            raise ValueError(f"{name} must be finite and at least {minimum:g} Hz")

    @property
    def local_port(self) -> int:
        """The actual bound source port."""
        return self._local_port

    def start(self) -> None:
        """Bind once and start the independent 20 Hz axis/heartbeat stream."""
        if self._socket is not None:
            raise RuntimeError("axis stream is already running")
        sock = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self._source_address or "0.0.0.0", self._local_port))
        except OSError:
            sock.close()
            raise
        self._socket = sock
        self._local_port = sock.getsockname()[1]
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="lite3-axis-stream", daemon=True)
        self._thread.start()

    def set_axes(self, values: AxisValues) -> None:
        """Publish the newest intended axes; the worker zeroes them after the command TTL."""
        if self._socket is None:
            raise RuntimeError("start() first")
        self._raise_if_failed()
        with self._lock:
            self._setpoint = _AxisSetpoint(values, self._clock())

    def stop(self) -> None:
        """Replace the current setpoint with zero axes without stopping the safety streamer."""
        if self._socket is None:
            return
        with self._lock:
            self._setpoint = _AxisSetpoint(AxisValues(), self._clock())

    def shutdown(self) -> None:
        """Stop the worker, stream zeros for the cleanup interval, and close the socket."""
        sock = self._socket
        if sock is None:
            return
        self.stop()
        self._running.clear()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._socket = None

        cleanup_error = None
        try:
            self._send_zeros_for_duration(sock)
        except OSError as error:
            cleanup_error = error
        finally:
            sock.close()
        self._raise_if_failed()
        if cleanup_error is not None:
            raise cleanup_error

    def effective_axes(self, now: float | None = None) -> AxisValues:
        """Return the latest command only while it is younger than the application TTL."""
        now = self._clock() if now is None else now
        with self._lock:
            setpoint = self._setpoint
        return setpoint.values if now - setpoint.updated_at <= self._command_ttl_s else AxisValues()

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(f"axis stream failed: {self._failure}") from self._failure

    def _run(self) -> None:
        sock = self._socket
        if sock is None:
            return
        next_axis = self._clock()
        next_heartbeat = next_axis
        while self._running.is_set():
            now = self._clock()
            try:
                self._send_axes(sock, self.effective_axes(now))
                if now >= next_heartbeat:
                    self._send_heartbeat(sock)
                    while next_heartbeat <= now:
                        next_heartbeat += self._heartbeat_interval
            except OSError as error:
                self._failure = error
                self._running.clear()
                return

            next_axis += self._axis_interval
            now = self._clock()
            while next_axis <= now:
                next_axis += self._axis_interval
            self._sleep(max(0.0, next_axis - now))

    def _send_zeros_for_duration(self, sock: socket.socket) -> None:
        deadline = self._clock() + self._stop_seconds
        next_axis = self._clock()
        next_heartbeat = next_axis
        errors: list[OSError] = []
        while self._clock() < deadline:
            now = self._clock()
            try:
                self._send_axes(sock, AxisValues())
                if now >= next_heartbeat:
                    self._send_heartbeat(sock)
                    while next_heartbeat <= now:
                        next_heartbeat += self._heartbeat_interval
            except OSError as error:
                errors.append(error)
            next_axis += self._axis_interval
            now = self._clock()
            while next_axis <= now:
                next_axis += self._axis_interval
            self._sleep(min(max(0.0, next_axis - now), max(0.0, deadline - now)))
        if errors:
            raise OSError(f"{len(errors)} zero-axis cleanup frame(s) failed") from errors[-1]

    def _send_axes(self, sock: socket.socket, values: AxisValues) -> None:
        sock.sendto(axis_packet(FORWARD_AXIS_CODE, values.forward), (self._host, self._port))
        sock.sendto(axis_packet(LATERAL_AXIS_CODE, values.lateral), (self._host, self._port))
        sock.sendto(axis_packet(YAW_AXIS_CODE, values.yaw), (self._host, self._port))

    def _send_heartbeat(self, sock: socket.socket) -> None:
        packet = struct.Struct("<3I").pack(HEARTBEAT_CODE, 0, 0)
        sock.sendto(packet, (self._host, self._port))


class Lite3AxisLocomotion(Lite3UdpLocomotion):
    """Navigator locomotion interface with inherited state decoding and profile-gated axes."""

    def __init__(self, *, axis_profile: AxisProfile | None,
                 axis_source_address: str | None = None,
                 axis_local_port: int = DEFAULT_LOCAL_PORT,
                 axis_rate_hz: float = AXIS_RATE_HZ,
                 heartbeat_hz: float = HEARTBEAT_HZ,
                 command_ttl_s: float = COMMAND_TTL_S,
                 streamer_factory=AxisStreamSender, **kwargs) -> None:
        super().__init__(**kwargs)
        self._axis_profile = axis_profile
        self._axis_source_address = axis_source_address
        self._axis_local_port = axis_local_port
        self._axis_rate_hz = axis_rate_hz
        self._heartbeat_hz = heartbeat_hz
        self._command_ttl_s = command_ttl_s
        self._streamer_factory = streamer_factory
        self._streamer: AxisStreamSender | None = None

    def connect(self) -> None:
        """Connect the inherited state reader without retaining a legacy command socket."""
        super().connect()
        legacy_socket, self._command_socket = self._command_socket, None
        if legacy_socket is not None:
            legacy_socket.close()

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Map velocity intent to profile primitives and stream only while state is fresh."""
        profile = self._axis_profile
        if profile is None:
            raise AxisProfileError("a local axis profile is required before axis commands")
        axes = profile.map_velocity(vx, vy, vyaw)
        if axes.is_zero and self._streamer is None:
            return
        if not axes.is_zero:
            age = self.state_age()
            if age is None or age > self._state_timeout_s:
                raise Lite3LinkLost(
                    f"the Lite3 state stream has been silent for "
                    f"{'ever' if age is None else f'{age:.2f}s'}; refusing to command "
                    "axis motion blind"
                )
            self.assert_axis_state_ready()
        if self._streamer is None:
            self._streamer = self._streamer_factory(
                host=self._motion_host,
                port=self._command_port,
                source_address=self._axis_source_address,
                local_port=self._axis_local_port,
                axis_rate_hz=self._axis_rate_hz,
                heartbeat_hz=self._heartbeat_hz,
                command_ttl_s=self._command_ttl_s,
            )
            self._streamer.start()
        self._streamer.set_axes(axes)

    def assert_axis_state_ready(self) -> None:
        """Require the documented manual/moving state before nonzero axis motion."""
        profile = self._axis_profile
        if profile is None:
            raise AxisProfileError("a local axis profile is required before axis commands")
        state = self._require_state()
        basic, gait, policy, motion = state.mode
        if state.error_state != 0:
            raise Lite3LinkLost(f"Lite3 error_state={state.error_state}; refusing axis motion")
        if basic != 6:
            raise Lite3LinkLost(
                f"Lite3 basic_state={basic}; axis motion requires documented force-control state 6"
            )
        if policy != 0:
            raise Lite3LinkLost(
                f"Lite3 policy_state={policy}; profile-gated manual moving mode requires policy 0"
            )
        if gait not in profile.allowed_gait_states:
            raise Lite3LinkLost(
                f"Lite3 gait_state={gait}; axis profile allows {profile.allowed_gait_states}"
            )
        if motion not in (0, 1):
            raise Lite3LinkLost(
                f"Lite3 motion_state={motion}; refusing axis motion outside stationary/stepping"
            )

    def stop(self) -> None:
        """Set zero axes immediately; cleanup streaming occurs in :meth:`shutdown`."""
        if self._streamer is not None:
            self._streamer.stop()

    def shutdown(self) -> None:
        """Zero/disarm the axis stream before releasing inherited state resources."""
        streamer, self._streamer = self._streamer, None
        try:
            if streamer is not None:
                streamer.shutdown()
        finally:
            super().shutdown()


def axis_locomotion_factory(*, axis_profile: AxisProfile | None,
                            axis_source_address: str | None = None,
                            axis_local_port: int = DEFAULT_LOCAL_PORT,
                            axis_rate_hz: float = AXIS_RATE_HZ,
                            heartbeat_hz: float = HEARTBEAT_HZ,
                            command_ttl_s: float = COMMAND_TTL_S,
                            motion_host: str = DEFAULT_MOTION_HOST,
                            command_port: int = DEFAULT_COMMAND_PORT,
                            state_port: int = DEFAULT_STATE_PORT,
                            bind: str = "0.0.0.0"):
    """Build the shared navigator factory for the profile-gated simple-axis transport."""

    def factory(*, cmd_vel_topic=None, odom_topic=None, stamped=None, node_name=None):
        return Lite3AxisLocomotion(
            axis_profile=axis_profile,
            axis_source_address=axis_source_address,
            axis_local_port=axis_local_port,
            axis_rate_hz=axis_rate_hz,
            heartbeat_hz=heartbeat_hz,
            command_ttl_s=command_ttl_s,
            motion_host=motion_host,
            command_port=command_port,
            state_port=state_port,
            bind=bind,
        )

    return factory
