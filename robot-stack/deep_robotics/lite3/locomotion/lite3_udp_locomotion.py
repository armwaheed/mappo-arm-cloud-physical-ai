#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lite3 high-level locomotion over the vendor's own UDP interface, without ROS 2.

This speaks the same protocol ``Lite3_ROS``'s ``jetson2motion`` speaks, to the same motion
host, at the same level of abstraction. It is **not** ``Lite3_MotionSDK``: it commands a
body velocity and the manufacturer's controller keeps the robot balanced and gaited, which
is the property that made the ROS bridge acceptable in the first place. Nothing here
touches a joint, a gain, or a balance controller.

What it removes is the ROS 2 runtime, and only that. Reading ``Jetson2Motion.cpp``, the
bridge's entire command path is:

* on ``/cmd_vel``, send three 20-byte frames to the motion host's command port —
  ``{int32 cmd_code, int32 size, int32 type, double data}`` with code ``320`` carrying
  ``linear.x``, ``325`` carrying ``linear.y``, and ``321`` carrying ``-angular.z``;
* nothing else. There is no timer, no keepalive and no periodic transmission: the
  executable sends only in its subscription callbacks.

So the ROS 2 Foxy install, the perception-host build and the C++ toolchain are packaging
around three ``sendto`` calls. On a Venture with no provisioned perception host that
packaging costs more than it carries.

The receive half reuses :mod:`deep_robotics.lite3.commissioning.lite3_state_probe` rather
than restating the wire format. That module's frame offsets came from a compiler, and one
decoder means one place to be wrong.

**Two units deliberately do not follow the vendor bridge.**

``rpy`` is degrees, and the bridge copies ``rpy_vel`` into a ROS field documented as rad/s
without converting it. Rather than propagate a rate whose unit is unconfirmed on the
installed firmware, :meth:`velocity` differentiates pose yaw, which is degrees by
documentation and by the bridge's own ``/180*PI``. ``calibrate_camera.py --spin`` already
made this choice for the same reason; this keeps the two consistent. The raw field stays
available as :meth:`reported_yaw_rate` so commissioning can still compare them.

Yaw sign follows the bridge exactly: it transmits ``-1 * angular.z``, so a positive
(left) yaw command goes on the wire negated. Getting that backwards is a heading servo
that runs away from its target rather than toward it.

See ``../../../SAFETY.md``. ``--live`` is still the only flag that moves the robot, the
operator still holds the emergency stop, and the approved AUTO-mode transition remains an
external vendor interface that this module does not attempt.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from deep_robotics.lite3.commissioning.lite3_state_probe import DecodeError, decode_frame

#: ``MotionComplexCMD`` under ``#pragma pack(push, 4)``: three int32 then a double, 20
#: bytes with no padding.
_COMMAND = struct.Struct("<3id")

#: Command codes, from ``Jetson2Motion.cpp``'s ``CmdVelCallback``.
FORWARD_VELOCITY_CODE = 320
LATERAL_VELOCITY_CODE = 325
YAW_VELOCITY_CODE = 321
COMPLEX_CMD_TYPE = 1
COMPLEX_CMD_SIZE = 8
VELOCITY_COMMAND_CODES = frozenset((
    FORWARD_VELOCITY_CODE,
    LATERAL_VELOCITY_CODE,
    YAW_VELOCITY_CODE,
))

DEFAULT_MOTION_HOST = "192.168.1.120"
DEFAULT_COMMAND_PORT = 43893
DEFAULT_STATE_PORT = 43897

#: How long the state stream may be silent before a non-zero command is refused.
#: At the ~35 Hz this stream runs, half a second is about seventeen lost frames — an
#: outage, not jitter.
STATE_TIMEOUT_S = 0.5

#: How long :meth:`connect` waits for the first state frame before giving up.
CONNECT_TIMEOUT_S = 5.0


class Lite3LinkLost(RuntimeError):
    """The motion host stopped reporting state, so commanding it would be blind."""


def velocity_packet(code: int, value: float) -> bytes:
    """Encode one vendor high-level velocity command without native padding."""
    if code not in VELOCITY_COMMAND_CODES:
        raise ValueError(f"unsupported Lite3 velocity command code: {code}")
    if not math.isfinite(value):
        raise ValueError(f"Lite3 velocity command must be finite: {value}")
    return _COMMAND.pack(code, COMPLEX_CMD_SIZE, COMPLEX_CMD_TYPE, float(value))


@dataclass(frozen=True)
class Lite3Pose:
    """Plan-view pose in the motion host's world frame. ``yaw`` is radians."""

    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class _StateSnapshot:
    """One decoded ``RobotState``, published to readers by whole-object replacement.

    Readers take a single attribute reference, so no lock is needed: they either see the
    previous complete snapshot or the next one, never a half-updated mixture of the two.
    """

    received_at: float
    x: float
    y: float
    yaw_rad: float
    vx: float
    vy: float
    reported_yaw_rate: float
    battery_level: float
    error_state: int
    mode: tuple


def _unwrap_radians(previous: float, current: float) -> float:
    """Return ``current - previous`` folded into (-pi, pi]."""
    delta = (current - previous) % (2.0 * math.pi)
    if delta > math.pi:
        delta -= 2.0 * math.pi
    return delta


class Lite3UdpLocomotion:
    """The ``Lite3Locomotion`` implementation contract, over UDP instead of ROS 2.

    Provides ``connect``, ``set_velocity``, ``stop``, ``pose``, ``velocity`` and
    ``shutdown`` — the six methods :class:`~deep_robotics.lite3.locomotion.
    lite3_locomotion.Lite3Locomotion` composes.
    """

    def __init__(self, *, motion_host: str = DEFAULT_MOTION_HOST,
                 command_port: int = DEFAULT_COMMAND_PORT,
                 state_port: int = DEFAULT_STATE_PORT,
                 bind: str = "0.0.0.0",
                 state_timeout_s: float = STATE_TIMEOUT_S,
                 connect_timeout_s: float = CONNECT_TIMEOUT_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._motion_host = motion_host
        self._command_port = command_port
        self._state_port = state_port
        self._bind = bind
        self._state_timeout_s = state_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._clock = clock

        self._command_socket: socket.socket | None = None
        self._state_socket: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._running = False
        self._state: _StateSnapshot | None = None
        self._previous_yaw: tuple[float, float] | None = None
        self._yaw_rate = 0.0

    # ----- lifecycle ------------------------------------------------------------------

    def connect(self) -> None:
        """Start listening, then wait for proof the robot is actually reporting.

        Failing here is the point. A Lite3 streams state to the single address in
        ``~/jy_exe/conf/network.toml``; if this host does not hold that address the robot
        is silent, commands vanish into the network, and the symptom is a robot that never
        moves and never explains why. That is the failure the vendor bridge produces too,
        and it is indistinguishable from a software bug until someone reads the config.
        """
        if self._running:
            raise RuntimeError("locomotion is already connected")

        self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._state_socket.bind((self._bind, self._state_port))
        # Honour a 0 request by reporting the port the OS actually chose.
        self._state_port = self._state_socket.getsockname()[1]
        self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._running = True
        self._reader = threading.Thread(target=self._read_state, name="lite3-udp-state",
                                        daemon=True)
        self._reader.start()

        deadline = self._clock() + self._connect_timeout_s
        while self._clock() < deadline:
            if self._state is not None:
                return
            time.sleep(0.02)

        self.shutdown()
        raise Lite3LinkLost(
            f"no Lite3 state frame arrived on {self._bind}:{self._state_port} within "
            f"{self._connect_timeout_s:.0f}s. The motion host sends to exactly one "
            f"address: check 'ip' in ~/jy_exe/conf/network.toml against this host's "
            f"address, and confirm {self._motion_host} is reachable."
        )

    def shutdown(self) -> None:
        """Stop the reader and release both sockets. Safe after a partial connect."""
        self._running = False
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        for name in ("_state_socket", "_command_socket"):
            sock = getattr(self, name)
            setattr(self, name, None)
            if sock is not None:
                sock.close()
        self._state = None
        self._previous_yaw = None
        self._yaw_rate = 0.0

    @property
    def state_port(self) -> int:
        """The bound state port, resolved if the caller asked for an ephemeral one."""
        return self._state_port

    # ----- commands -------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Send one body-velocity command, as three frames, exactly as the bridge does."""
        if self._command_socket is None:
            raise RuntimeError("connect() first")
        commanding_motion = not (vx == 0.0 and vy == 0.0 and vyaw == 0.0)
        if commanding_motion:
            age = self.state_age()
            if age is None or age > self._state_timeout_s:
                raise Lite3LinkLost(
                    f"the Lite3 state stream has been silent for "
                    f"{'ever' if age is None else f'{age:.2f}s'}; refusing to command "
                    f"motion blind. stop() and shutdown() still transmit."
                )
        self._send(FORWARD_VELOCITY_CODE, vx)
        self._send(LATERAL_VELOCITY_CODE, vy)
        # The bridge transmits -angular.z. Matching it keeps +yaw meaning left here and
        # on every vendor tool, so the heading servo turns toward its target.
        self._send(YAW_VELOCITY_CODE, -vyaw)

    def stop(self) -> None:
        """Command zero velocity. Never refuses: a stop must survive a lost link."""
        if self._command_socket is None:
            return
        self._send(FORWARD_VELOCITY_CODE, 0.0)
        self._send(LATERAL_VELOCITY_CODE, 0.0)
        self._send(YAW_VELOCITY_CODE, 0.0)

    def _send(self, code: int, value: float) -> None:
        frame = velocity_packet(code, value)
        self._command_socket.sendto(frame, (self._motion_host, self._command_port))

    # ----- state ----------------------------------------------------------------------

    def pose(self) -> Lite3Pose:
        # One read: three would straddle two frames and mix an old x with a new yaw.
        state = self._require_state()
        return Lite3Pose(state.x, state.y, state.yaw_rad)

    def velocity(self) -> tuple[float, float, float]:
        """Measured body velocity, with yaw rate differentiated from pose yaw.

        The vendor's ``rpy_vel`` field is not used: the bridge forwards it into a rad/s
        ROS field without converting, and the low-level SDK documents the corresponding
        quantity in degrees/s, so its unit is a property of the installed firmware rather
        than of the protocol. Pose yaw has a documented unit, so differentiating it cannot
        be wrong by a factor of 57.
        """
        state = self._require_state()
        return (state.vx, state.vy, self._yaw_rate)

    def reported_yaw_rate(self) -> float:
        """The raw ``rpy_vel`` z field, in whatever unit this firmware uses.

        Exposed for commissioning to compare against :meth:`velocity`, not for control.
        """
        return self._require_state().reported_yaw_rate

    def battery_level(self) -> float:
        """Battery percentage, which the vendor ROS bridge drops and this stream carries."""
        return self._require_state().battery_level

    def mode(self) -> tuple:
        """Documented vendor state tuple: basic state, gait, policy, and motion."""
        return self._require_state().mode

    def error_state(self) -> int:
        """The latest documented high-level error state."""
        return self._require_state().error_state

    def state_age(self) -> float | None:
        """Seconds since the last state frame, or ``None`` if none has ever arrived."""
        state = self._state
        if state is None:
            return None
        return self._clock() - state.received_at

    def _require_state(self) -> _StateSnapshot:
        state = self._state
        if state is None:
            raise Lite3LinkLost("no Lite3 state frame has arrived yet")
        return state

    def _read_state(self) -> None:
        sock = self._state_socket
        if sock is None:
            return
        sock.settimeout(0.2)
        while self._running:
            try:
                payload, _address = sock.recvfrom(2048)
            except (socket.timeout, OSError):
                continue
            try:
                frame = decode_frame(payload)
            except DecodeError:
                continue
            if frame["kind"] != "robot_state":
                continue
            self._publish(frame)

    def _publish(self, frame: dict) -> None:
        now = self._clock()
        yaw_rad = math.radians(frame["rpy_deg"][2])
        if self._previous_yaw is not None:
            previous_time, previous_yaw = self._previous_yaw
            interval = now - previous_time
            # Below a few milliseconds the quotient is arrival jitter, not rotation.
            if interval >= 0.005:
                self._yaw_rate = _unwrap_radians(previous_yaw, yaw_rad) / interval
        self._previous_yaw = (now, yaw_rad)
        self._state = _StateSnapshot(
            received_at=now,
            x=frame["pos_world"][0], y=frame["pos_world"][1], yaw_rad=yaw_rad,
            vx=frame["vel_body"][0], vy=frame["vel_body"][1],
            reported_yaw_rate=frame["rpy_vel"][2],
            battery_level=frame["battery_level"],
            error_state=frame["error_state"],
            mode=(frame["robot_basic_state"], frame["robot_gait_state"],
                  frame["robot_policy_state"], frame["robot_motion_state"]),
        )


def udp_locomotion_factory(*, motion_host: str = DEFAULT_MOTION_HOST,
                           command_port: int = DEFAULT_COMMAND_PORT,
                           state_port: int = DEFAULT_STATE_PORT,
                           bind: str = "0.0.0.0") -> Callable[..., Lite3UdpLocomotion]:
    """Build an ``implementation_factory`` for ``Lite3Locomotion``.

    ``Lite3Locomotion`` hands its factory the ROS topic names and node name it would use.
    A UDP transport has no topics and no node, so they are named and discarded here rather
    than threaded into a transport that cannot honour them — a silently ignored
    ``--cmd-vel-topic`` would otherwise read as configuration that does something.
    """
    def factory(*, cmd_vel_topic=None, odom_topic=None, stamped=None,
                node_name=None) -> Lite3UdpLocomotion:
        return Lite3UdpLocomotion(motion_host=motion_host, command_port=command_port,
                                  state_port=state_port, bind=bind)
    return factory
