#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Vendor-documented Lite3 control and motion-mode switch.

The vendor motion-host communication guide documents the 12-byte, little-endian
``CommandHead`` used here. This tool intentionally permits only the documented manual,
autonomous, stationary, and moving mode commands; it does not send heartbeat, axis, velocity,
gait, or low-level joint commands.
"""

from __future__ import annotations

import argparse
import socket
import struct
from collections.abc import Callable

from deep_robotics.lite3.locomotion.lite3_udp_locomotion import (
    DEFAULT_COMMAND_PORT,
    DEFAULT_MOTION_HOST,
)

_SIMPLE_COMMAND = struct.Struct("<3I")

AUTONOMOUS_MODE_CODE = 0x21010C03
MANUAL_MODE_CODE = 0x21010C02
MOVING_MODE_CODE = 0x21010D06
STATIONARY_MODE_CODE = 0x21010D05

MODE_CODES = {
    "autonomous": AUTONOMOUS_MODE_CODE,
    "manual": MANUAL_MODE_CODE,
    "moving": MOVING_MODE_CODE,
    "stationary": STATIONARY_MODE_CODE,
}


def simple_packet(code: int, value: int = 0) -> bytes:
    """Encode one documented type-0 command without native alignment."""
    if code not in MODE_CODES.values():
        raise ValueError(f"unsupported Lite3 simple command code: {code:#x}")
    if value != 0:
        raise ValueError("Lite3 mode-switch commands require a zero command value")
    return _SIMPLE_COMMAND.pack(code, value, 0)


def send_mode(mode: str, *, host: str, port: int,
              socket_factory: Callable[..., socket.socket] = socket.socket) -> None:
    """Send exactly one mode transition command."""
    try:
        code = MODE_CODES[mode]
    except KeyError as error:
        raise ValueError(f"unsupported Lite3 control mode: {mode}") from error
    sock = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(simple_packet(code), (host, port))
    finally:
        sock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one vendor-documented Lite3 manual/autonomous mode command.")
    parser.add_argument("--mode", choices=tuple(MODE_CODES), required=True)
    parser.add_argument("--host", default=DEFAULT_MOTION_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_COMMAND_PORT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument(
        "--operator-ready",
        action="store_true",
        help="confirm stable posture, clear area, and emergency stop in hand",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.live and not args.operator_ready:
            raise ValueError("--live mode changes require --operator-ready")
        if not args.live:
            print(f"dry run: mode={args.mode} code={MODE_CODES[args.mode]:#x}")
            return 0
        send_mode(args.mode, host=args.host, port=args.port)
    except (OSError, ValueError) as error:
        print(f"REFUSING TO SEND: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
