#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vendor-documented Lite3 control-mode command sender."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.locomotion.lite3_control_mode_udp import (
    AUTONOMOUS_MODE_CODE,
    MANUAL_MODE_CODE,
    MOVING_MODE_CODE,
    STATIONARY_MODE_CODE,
    main,
    send_mode,
    simple_packet,
)

_SIMPLE_COMMAND = struct.Struct("<3I")


class _Socket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def sendto(self, packet, address):
        self.sent.append((packet, address))

    def close(self):
        self.closed = True


def test_mode_packets_are_the_vendor_twelve_byte_command_head():
    for code in (
        AUTONOMOUS_MODE_CODE,
        MANUAL_MODE_CODE,
        MOVING_MODE_CODE,
        STATIONARY_MODE_CODE,
    ):
        packet = simple_packet(code)
        assert len(packet) == 12
        assert _SIMPLE_COMMAND.unpack(packet) == (code, 0, 0)


def test_unknown_code_and_nonzero_value_are_refused():
    for code, value in ((0x21010130, 0), (AUTONOMOUS_MODE_CODE, 1)):
        try:
            simple_packet(code, value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsupported simple command {(code, value)}")


def test_mode_sender_emits_only_one_documented_command():
    sock = _Socket()
    send_mode(
        "autonomous",
        host="127.0.0.1",
        port=43893,
        socket_factory=lambda *_: sock,
    )
    assert sock.closed
    assert len(sock.sent) == 1
    assert _SIMPLE_COMMAND.unpack(sock.sent[0][0]) == (AUTONOMOUS_MODE_CODE, 0, 0)
    assert sock.sent[0][1] == ("127.0.0.1", 43893)


def test_dry_run_never_constructs_a_socket():
    assert main(["--mode", "autonomous"]) == 0


def test_live_mode_change_requires_operator_confirmation():
    assert main(["--mode", "autonomous", "--live"]) == 2


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_control_mode_udp: {len(tests)}/{len(tests)} passed")
