#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bounded high-level Lite3 velocity test sender."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.locomotion.lite3_udp_locomotion import (
    FORWARD_VELOCITY_CODE,
    LATERAL_VELOCITY_CODE,
    YAW_VELOCITY_CODE,
    velocity_packet,
)
from deep_robotics.lite3.locomotion.lite3_velocity_udp import (
    HEARTBEAT_CODE,
    VelocityCommand,
    heartbeat_packet,
    main,
    run_live,
    velocity_triplet,
)

_COMMAND = struct.Struct("<3id")
_SIMPLE_COMMAND = struct.Struct("<3I")


class _Socket:
    def __init__(self, fail_first_nonzero=False):
        self.frames = []
        self.closed = False
        self.bound = None
        self._fail_first_nonzero = fail_first_nonzero

    def bind(self, address):
        self.bound = address

    def sendto(self, payload, address):
        frame = _COMMAND.unpack(payload) if len(payload) == 20 else _SIMPLE_COMMAND.unpack(payload)
        if self._fail_first_nonzero and len(payload) == 20 and frame[3] != 0.0:
            self._fail_first_nonzero = False
            raise OSError("simulated send failure")
        self.frames.append((frame, address))

    def close(self):
        self.closed = True


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_velocity_packet_matches_the_vendor_wire_structure():
    for code, value in (
        (FORWARD_VELOCITY_CODE, 0.1),
        (LATERAL_VELOCITY_CODE, 0.0),
        (YAW_VELOCITY_CODE, -0.2),
    ):
        packet = velocity_packet(code, value)
        assert len(packet) == 20
        assert _COMMAND.unpack(packet) == (code, 8, 1, value)


def test_velocity_packet_refuses_unknown_codes_and_nonfinite_values():
    for code, value in ((999, 0.0), (FORWARD_VELOCITY_CODE, float("nan"))):
        try:
            velocity_packet(code, value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe packet ({code}, {value})")


def test_heartbeat_packet_matches_the_vendor_simple_command_head():
    packet = heartbeat_packet()
    assert len(packet) == 12
    assert _SIMPLE_COMMAND.unpack(packet) == (HEARTBEAT_CODE, 0, 0)


def test_velocity_triplet_matches_the_bridges_yaw_sign():
    command = VelocityCommand(0.1, 0.0, 0.0, 1.0, 10.0)
    assert [_COMMAND.unpack(packet) for packet in velocity_triplet(command)] == [
        (FORWARD_VELOCITY_CODE, 8, 1, 0.1),
        (LATERAL_VELOCITY_CODE, 8, 1, 0.0),
        (YAW_VELOCITY_CODE, 8, 1, -0.0),
    ]


def test_first_sender_rejects_lateral_reverse_yaw_and_long_commands():
    invalid = (
        VelocityCommand(-0.01, 0.0, 0.0, 1.0, 10.0),
        VelocityCommand(0.11, 0.0, 0.0, 1.0, 10.0),
        VelocityCommand(0.1, 0.01, 0.0, 1.0, 10.0),
        VelocityCommand(0.1, 0.0, 0.01, 1.0, 10.0),
        VelocityCommand(0.1, 0.0, 0.0, 0.0, 10.0),
        VelocityCommand(0.1, 0.0, 0.0, 1.1, 10.0),
    )
    for command in invalid:
        try:
            command.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe command {command}")


def test_live_sender_repeats_zero_triplets_after_normal_completion():
    sock = _Socket()
    clock = _Clock()
    command = VelocityCommand(0.1, 0.0, 0.0, 0.1, 10.0)
    run_live(command, host="127.0.0.1", port=43893, socket_factory=lambda *_: sock,
             clock=clock, sleep=clock.sleep, stop_seconds=0.2)

    frames = [frame for frame, _address in sock.frames]
    assert frames[:3] == [
        (FORWARD_VELOCITY_CODE, 8, 1, 0.1),
        (LATERAL_VELOCITY_CODE, 8, 1, 0.0),
        (YAW_VELOCITY_CODE, 8, 1, -0.0),
    ]
    assert len(frames) >= 9
    assert all(frame[3] == 0.0 for frame in frames[3:])
    assert sock.closed


def test_live_sender_repeats_zero_triplets_after_a_send_failure():
    sock = _Socket(fail_first_nonzero=True)
    clock = _Clock()
    command = VelocityCommand(0.1, 0.0, 0.0, 0.1, 10.0)
    try:
        run_live(command, host="127.0.0.1", port=43893, socket_factory=lambda *_: sock,
                 clock=clock, sleep=clock.sleep, stop_seconds=0.2)
    except OSError:
        pass
    else:
        raise AssertionError("expected the simulated command send failure")

    assert len(sock.frames) >= 6
    assert all(frame[3] == 0.0 for frame, _address in sock.frames)
    assert sock.closed


def test_live_sender_sends_heartbeat_at_the_requested_rate():
    sock = _Socket()
    clock = _Clock()
    command = VelocityCommand(0.1, 0.0, 0.0, 1.0, 10.0)
    run_live(
        command,
        host="127.0.0.1",
        port=43893,
        socket_factory=lambda *_: sock,
        clock=clock,
        sleep=clock.sleep,
        stop_seconds=0.0,
        heartbeat_hz=2.0,
    )

    frames = [frame for frame, _address in sock.frames]
    # Two heartbeats cover the 1-second command and one begins zero-velocity cleanup.
    assert sum(frame[0] == HEARTBEAT_CODE for frame in frames) == 3
    assert all(len(frame) in (3, 4) for frame in frames)


def test_live_sender_binds_an_explicit_source_address():
    sock = _Socket()
    clock = _Clock()
    run_live(
        VelocityCommand(0.0, 0.0, 0.0, 0.0, 10.0),
        host="127.0.0.1",
        port=43893,
        socket_factory=lambda *_: sock,
        clock=clock,
        sleep=clock.sleep,
        stop_seconds=0.0,
        source_address="192.168.1.103",
    )
    assert sock.bound == ("192.168.1.103", 0)


def test_dry_run_never_constructs_a_socket():
    assert main(["--vx", "0.1", "--duration", "1.0"]) == 0


def test_live_commands_require_operator_confirmation_even_at_zero_velocity():
    assert main(["--live"]) == 2


def test_heartbeat_below_vendor_minimum_is_refused():
    assert main(["--heartbeat-hz", "1"]) == 2


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_velocity_udp: {len(tests)}/{len(tests)} passed")
