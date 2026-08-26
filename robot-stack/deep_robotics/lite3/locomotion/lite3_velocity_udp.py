#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded, vendor-wire-compatible Lite3 high-level velocity test sender.

This is intentionally not a navigation or gait interface. It reproduces only the three
velocity frames emitted by the vendor's ROS bridge and does not decode state or select the
robot's external-control mode. A non-zero command therefore needs a separately confirmed
operator-ready state and explicit ``--live``.
"""

from __future__ import annotations

import argparse
import math
import signal
import socket
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass

from deep_robotics.lite3.locomotion.lite3_udp_locomotion import (
    DEFAULT_COMMAND_PORT,
    DEFAULT_MOTION_HOST,
    FORWARD_VELOCITY_CODE,
    LATERAL_VELOCITY_CODE,
    YAW_VELOCITY_CODE,
    velocity_packet,
)

_SIMPLE_COMMAND = struct.Struct("<3I")

HEARTBEAT_CODE = 0x21040001

MAX_FORWARD_M_S = 0.10
MAX_DURATION_S = 1.0
MAX_RATE_HZ = 10.0
STOP_SECONDS = 2.0


def heartbeat_packet() -> bytes:
    """Encode the vendor's documented type-0 heartbeat command."""
    return _SIMPLE_COMMAND.pack(HEARTBEAT_CODE, 0, 0)


@dataclass(frozen=True)
class VelocityCommand:
    """One bounded high-level velocity request in the vendor body frame."""

    vx: float
    vy: float
    wz: float
    duration_s: float
    rate_hz: float

    def validate(self) -> None:
        values = {
            "--vx": self.vx,
            "--vy": self.vy,
            "--wz": self.wz,
            "--duration": self.duration_s,
            "--rate": self.rate_hz,
        }
        invalid = [name for name, value in values.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"values must be finite: {', '.join(invalid)}")
        if not 0.0 <= self.vx <= MAX_FORWARD_M_S:
            raise ValueError(f"--vx must be within 0.0..{MAX_FORWARD_M_S:.2f} m/s")
        if self.vy != 0.0:
            raise ValueError("--vy must be 0.0 for the first bounded test")
        if self.wz != 0.0:
            raise ValueError("--wz must be 0.0 for the first bounded test")
        if not 0.0 <= self.duration_s <= MAX_DURATION_S:
            raise ValueError(f"--duration must be within 0.0..{MAX_DURATION_S:.1f} s")
        if self.vx != 0.0 and self.duration_s == 0.0:
            raise ValueError("non-zero --vx requires a positive --duration")
        if not 0.0 < self.rate_hz <= MAX_RATE_HZ:
            raise ValueError(f"--rate must be within 0.0..{MAX_RATE_HZ:.0f} Hz")

def velocity_triplet(command: VelocityCommand) -> tuple[bytes, bytes, bytes]:
    """Return the exact three vendor datagrams for one velocity update."""
    return (
        velocity_packet(FORWARD_VELOCITY_CODE, command.vx),
        velocity_packet(LATERAL_VELOCITY_CODE, command.vy),
        velocity_packet(YAW_VELOCITY_CODE, -command.wz),
    )


def send_triplet(sock: socket.socket, host: str, port: int, command: VelocityCommand) -> None:
    """Transmit exactly one forward/lateral/yaw update."""
    for packet in velocity_triplet(command):
        sock.sendto(packet, (host, port))


def send_for_duration(sock: socket.socket, host: str, port: int, command: VelocityCommand,
                      *, clock: Callable[[], float] = time.monotonic,
                      sleep: Callable[[float], None] = time.sleep,
                      heartbeat_hz: float = 0.0) -> None:
    """Send a bounded update series and optional vendor heartbeat."""
    if not math.isfinite(heartbeat_hz) or heartbeat_hz < 0.0:
        raise ValueError("--heartbeat-hz must be finite and non-negative")
    deadline = clock() + command.duration_s
    next_heartbeat = clock()
    first_update = True
    while first_update or clock() < deadline:
        now = clock()
        if heartbeat_hz and now >= next_heartbeat:
            sock.sendto(heartbeat_packet(), (host, port))
            next_heartbeat = now + 1.0 / heartbeat_hz
        send_triplet(sock, host, port, command)
        first_update = False
        remaining = deadline - clock()
        if remaining <= 0.0:
            return
        sleep(min(1.0 / command.rate_hz, remaining))


def run_live(command: VelocityCommand, *, host: str, port: int,
             socket_factory: Callable[..., socket.socket] = socket.socket,
             clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep,
             stop_seconds: float = STOP_SECONDS,
             heartbeat_hz: float = 0.0,
             source_address: str | None = None) -> None:
    """Send the bounded command, then repeatedly send zero velocity on every exit path."""
    sock = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
    if source_address is not None:
        sock.bind((source_address, 0))
    stop = VelocityCommand(0.0, 0.0, 0.0, stop_seconds, command.rate_hz)
    try:
        send_for_duration(
            sock,
            host,
            port,
            command,
            clock=clock,
            sleep=sleep,
            heartbeat_hz=heartbeat_hz,
        )
    finally:
        send_for_duration(
            sock,
            host,
            port,
            stop,
            clock=clock,
            sleep=sleep,
            heartbeat_hz=heartbeat_hz,
        )
        sock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one bounded vendor high-level Lite3 velocity test.")
    parser.add_argument("--host", default=DEFAULT_MOTION_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_COMMAND_PORT)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=MAX_DURATION_S)
    parser.add_argument("--rate", type=float, default=MAX_RATE_HZ)
    parser.add_argument(
        "--source-address",
        help="optional local IPv4 address to bind before sending",
    )
    parser.add_argument(
        "--heartbeat-hz",
        type=float,
        default=2.0,
        help="vendor heartbeat frequency during live command and zero cleanup (minimum 2)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument(
        "--operator-ready",
        action="store_true",
        help="confirm high-level external-control mode, clear lane, and emergency stop in hand",
    )
    return parser


def _interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = VelocityCommand(args.vx, args.vy, args.wz, args.duration, args.rate)
    try:
        command.validate()
        if not math.isfinite(args.heartbeat_hz) or args.heartbeat_hz < 2.0:
            raise ValueError("--heartbeat-hz must be finite and at least 2 Hz")
        if args.live and not args.operator_ready:
            raise ValueError("--live commands require --operator-ready")
        if not args.live:
            print(f"dry run: {command}")
            return 0
        signal.signal(signal.SIGINT, _interrupt)
        signal.signal(signal.SIGTERM, _interrupt)
        run_live(
            command,
            host=args.host,
            port=args.port,
            heartbeat_hz=args.heartbeat_hz,
            source_address=args.source_address,
        )
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as error:
        print(f"REFUSING TO SEND: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
