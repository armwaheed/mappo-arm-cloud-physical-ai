#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded vendor axis-command sender for Lite3 moving or AI state.

The vendor-provided ``Lite3_New_All_Control_v2_0_158.py`` sends a heartbeat every 500 ms and
the forward axis code ``0x21010130`` with value ``+32767`` every 50 ms while the forward key is
held. This tool implements only that documented forward/zero subset. ``--zero-only`` provides a
non-actuating packet-arrival health check before a separately authorized forward pulse. It does
not select a mode; the operator must first use the separate documented mode command and verify
stable posture.
"""

from __future__ import annotations

import argparse
import math
import signal
import socket
import struct
import time
from collections.abc import Callable

from deep_robotics.lite3.locomotion.lite3_udp_locomotion import (
    DEFAULT_COMMAND_PORT,
    DEFAULT_MOTION_HOST,
)

_SIMPLE_COMMAND = struct.Struct("<3I")

HEARTBEAT_CODE = 0x21040001
FORWARD_AXIS_CODE = 0x21010130
LATERAL_AXIS_CODE = 0x21010131
YAW_AXIS_CODE = 0x21010135
FORWARD_AXIS_VALUE = 32767
AXIS_CODES = frozenset((FORWARD_AXIS_CODE, LATERAL_AXIS_CODE, YAW_AXIS_CODE))
AXIS_LIMIT = 32767
AXIS_RATE_HZ = 20.0
HEARTBEAT_HZ = 2.0
MAX_DURATION_S = 1.0
STOP_SECONDS = 2.0
DEFAULT_LOCAL_PORT = 20001


def axis_packet(code: int, value: int) -> bytes:
    """Encode one documented signed joystick-axis command."""
    if code not in AXIS_CODES:
        raise ValueError(f"unsupported Lite3 axis command code: {code:#x}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Lite3 axis value must be an integer")
    if not -AXIS_LIMIT <= value <= AXIS_LIMIT:
        raise ValueError(f"Lite3 axis value must be within {-AXIS_LIMIT}..{AXIS_LIMIT}")
    return _SIMPLE_COMMAND.pack(code, value & 0xFFFFFFFF, 0)


def simple_packet(code: int, value: int = 0) -> bytes:
    """Encode the bounded forward/zero subset used by this standalone tool."""
    if code not in (HEARTBEAT_CODE, FORWARD_AXIS_CODE):
        raise ValueError(f"unsupported Lite3 axis command code: {code:#x}")
    if code == HEARTBEAT_CODE and value != 0:
        raise ValueError("heartbeat requires a zero command value")
    if code == FORWARD_AXIS_CODE and value not in (0, FORWARD_AXIS_VALUE):
        raise ValueError("only documented forward full-scale or zero axis values are allowed")
    if code == HEARTBEAT_CODE:
        return _SIMPLE_COMMAND.pack(code, 0, 0)
    return axis_packet(code, value)


def send_for_duration(sock: socket.socket, host: str, port: int, axis_value: int,
                      duration_s: float, *, clock: Callable[[], float] = time.monotonic,
                      sleep: Callable[[float], None] = time.sleep,
                      continue_after_send_failure: bool = False) -> None:
    """Send 20 Hz axis commands and 2 Hz heartbeat for a bounded duration."""
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("axis duration must be finite and positive")
    start = clock()
    deadline = start + duration_s
    next_axis = start
    next_heartbeat = start
    first_update = True
    axis_interval = 1.0 / AXIS_RATE_HZ
    heartbeat_interval = 1.0 / HEARTBEAT_HZ
    send_errors: list[OSError] = []

    def send(packet: bytes) -> None:
        try:
            sock.sendto(packet, (host, port))
        except OSError as error:
            if not continue_after_send_failure:
                raise
            send_errors.append(error)

    while first_update or clock() < deadline:
        send(simple_packet(FORWARD_AXIS_CODE, axis_value))
        first_update = False
        next_axis += axis_interval
        now = clock()
        if now >= next_heartbeat:
            send(simple_packet(HEARTBEAT_CODE))
            while next_heartbeat <= now:
                next_heartbeat += heartbeat_interval
        remaining = deadline - clock()
        if remaining <= 0.0:
            break
        while next_axis <= clock():
            next_axis += axis_interval
        sleep(min(next_axis - clock(), remaining))
    if send_errors:
        raise OSError(f"{len(send_errors)} zero-axis cleanup frame(s) failed") from send_errors[-1]


def run_live(*, host: str, port: int, source_address: str | None,
             local_port: int, duration_s: float, axis_value: int = FORWARD_AXIS_VALUE,
             socket_factory: Callable[..., socket.socket] = socket.socket,
             clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep) -> None:
    """Send one documented axis interval, then retry zero axis updates on every exit."""
    if axis_value not in (0, FORWARD_AXIS_VALUE):
        raise ValueError("only documented forward full-scale or zero axis values are allowed")
    if not math.isfinite(duration_s) or not 0.0 < duration_s <= MAX_DURATION_S:
        raise ValueError(f"axis duration must be within 0.0..{MAX_DURATION_S:.1f} s")
    sock = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
    bind_address = source_address or "0.0.0.0"
    bound = False
    try:
        sock.bind((bind_address, local_port))
        bound = True
        send_for_duration(
            sock,
            host,
            port,
            axis_value,
            duration_s,
            clock=clock,
            sleep=sleep,
        )
    finally:
        try:
            if bound:
                send_for_duration(
                    sock,
                    host,
                    port,
                    0,
                    STOP_SECONDS,
                    clock=clock,
                    sleep=sleep,
                    continue_after_send_failure=True,
                )
        finally:
            sock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one bounded vendor Lite3 forward axis pulse in moving or AI state.")
    parser.add_argument("--host", default=DEFAULT_MOTION_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_COMMAND_PORT)
    parser.add_argument("--source-address")
    parser.add_argument("--local-port", type=int, default=DEFAULT_LOCAL_PORT)
    parser.add_argument("--duration", type=float, default=MAX_DURATION_S)
    parser.add_argument(
        "--zero-only",
        action="store_true",
        help="send only the documented zero forward-axis value for packet-arrival verification",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument(
        "--operator-ready",
        action="store_true",
        help="confirm vendor moving/AI mode, clear area, and emergency stop in hand",
    )
    return parser


def _interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not 0.0 < args.duration <= MAX_DURATION_S:
            raise ValueError(f"--duration must be within 0.0..{MAX_DURATION_S:.1f} s")
        if not 0 <= args.local_port <= 65535:
            raise ValueError("--local-port must be within 0..65535")
        if args.live and not args.operator_ready:
            raise ValueError("--live axis commands require --operator-ready")
        if not args.live:
            axis_value = 0 if args.zero_only else FORWARD_AXIS_VALUE
            print(
                f"dry run: forward-axis={axis_value} rate={AXIS_RATE_HZ:.0f}Hz "
                f"heartbeat={HEARTBEAT_HZ:.0f}Hz duration={args.duration:.1f}s"
            )
            return 0
        signal.signal(signal.SIGINT, _interrupt)
        signal.signal(signal.SIGTERM, _interrupt)
        run_live(
            host=args.host,
            port=args.port,
            source_address=args.source_address,
            local_port=args.local_port,
            duration_s=args.duration,
            axis_value=0 if args.zero_only else FORWARD_AXIS_VALUE,
        )
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as error:
        print(f"REFUSING TO SEND: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
