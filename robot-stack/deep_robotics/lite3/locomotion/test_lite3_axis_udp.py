#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the restricted vendor Lite3 axis-command sender."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.locomotion.lite3_axis_udp import (
    FORWARD_AXIS_CODE,
    FORWARD_AXIS_VALUE,
    HEARTBEAT_CODE,
    main,
    run_live,
    send_for_duration,
    simple_packet,
)

_SIMPLE_COMMAND = struct.Struct("<3I")


class _Socket:
    def __init__(self, fail_forward=False, fail_zero_attempts=0):
        self.bound = None
        self.closed = False
        self.frames = []
        self.attempts = []
        self.fail_forward = fail_forward
        self.fail_zero_attempts = fail_zero_attempts

    def bind(self, address):
        self.bound = address

    def sendto(self, packet, address):
        frame = _SIMPLE_COMMAND.unpack(packet)
        self.attempts.append((frame, address))
        if self.fail_forward and frame == (FORWARD_AXIS_CODE, FORWARD_AXIS_VALUE, 0):
            self.fail_forward = False
            raise OSError("simulated forward-axis send failure")
        if self.fail_zero_attempts and frame == (FORWARD_AXIS_CODE, 0, 0):
            self.fail_zero_attempts -= 1
            raise OSError("simulated zero-axis send failure")
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


class _DelayedSocket(_Socket):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.sent_at = []

    def sendto(self, packet, address):
        self.sent_at.append((_SIMPLE_COMMAND.unpack(packet), self.clock.now))
        super().sendto(packet, address)
        self.clock.now += 0.005


class _BindFailSocket(_Socket):
    def bind(self, address):
        self.bound = address
        raise OSError("simulated source-port bind failure")


def test_simple_packets_match_vendor_command_head():
    assert _SIMPLE_COMMAND.unpack(simple_packet(HEARTBEAT_CODE)) == (HEARTBEAT_CODE, 0, 0)
    assert _SIMPLE_COMMAND.unpack(simple_packet(FORWARD_AXIS_CODE, FORWARD_AXIS_VALUE)) == (
        FORWARD_AXIS_CODE,
        FORWARD_AXIS_VALUE,
        0,
    )
    assert _SIMPLE_COMMAND.unpack(simple_packet(FORWARD_AXIS_CODE, 0)) == (
        FORWARD_AXIS_CODE,
        0,
        0,
    )


def test_axis_sender_refuses_undocumented_values():
    for code, value in ((0x21010131, 0), (FORWARD_AXIS_CODE, -FORWARD_AXIS_VALUE)):
        try:
            simple_packet(code, value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsupported command {(code, value)}")


def test_live_axis_sender_binds_vendor_default_port_and_zeros_after_completion():
    sock = _Socket()
    clock = _Clock()
    run_live(
        host="127.0.0.1",
        port=43893,
        source_address="192.168.1.103",
        local_port=20001,
        duration_s=0.1,
        socket_factory=lambda *_: sock,
        clock=clock,
        sleep=clock.sleep,
    )
    assert sock.bound == ("192.168.1.103", 20001)
    assert sock.closed
    assert any(frame == (FORWARD_AXIS_CODE, FORWARD_AXIS_VALUE, 0)
               for frame, _address in sock.frames)
    assert any(frame == (HEARTBEAT_CODE, 0, 0) for frame, _address in sock.frames)
    forward_indices = [
        index for index, (frame, _address) in enumerate(sock.frames)
        if frame == (FORWARD_AXIS_CODE, FORWARD_AXIS_VALUE, 0)
    ]
    assert forward_indices
    assert all(
        frame == (FORWARD_AXIS_CODE, 0, 0) or frame == (HEARTBEAT_CODE, 0, 0)
        for frame, _address in sock.frames[forward_indices[-1] + 1:]
    )


def test_live_axis_sender_zeros_after_forward_send_failure():
    sock = _Socket(fail_forward=True)
    clock = _Clock()
    try:
        run_live(
            host="127.0.0.1",
            port=43893,
            source_address=None,
            local_port=20001,
            duration_s=0.1,
            socket_factory=lambda *_: sock,
            clock=clock,
            sleep=clock.sleep,
        )
    except OSError:
        pass
    else:
        raise AssertionError("expected simulated forward-axis send failure")
    assert sock.closed
    assert any(frame == (FORWARD_AXIS_CODE, 0, 0) for frame, _address in sock.frames)


def test_live_axis_sender_retries_zero_after_cleanup_send_failure_and_closes():
    sock = _Socket(fail_zero_attempts=1)
    clock = _Clock()
    try:
        run_live(
            host="127.0.0.1",
            port=43893,
            source_address=None,
            local_port=20001,
            duration_s=0.1,
            socket_factory=lambda *_: sock,
            clock=clock,
            sleep=clock.sleep,
        )
    except OSError as error:
        assert "zero-axis cleanup" in str(error)
    else:
        raise AssertionError("expected simulated zero-axis cleanup failure")
    zero_attempts = [
        frame for frame, _address in sock.attempts if frame == (FORWARD_AXIS_CODE, 0, 0)
    ]
    assert len(zero_attempts) > 1
    assert any(frame == (FORWARD_AXIS_CODE, 0, 0) for frame, _address in sock.frames)
    assert sock.closed


def test_live_axis_sender_closes_without_sending_when_source_port_bind_fails():
    sock = _BindFailSocket()
    try:
        run_live(
            host="127.0.0.1",
            port=43893,
            source_address="192.168.1.103",
            local_port=20001,
            duration_s=0.1,
            socket_factory=lambda *_: sock,
        )
    except OSError:
        pass
    else:
        raise AssertionError("expected simulated source-port bind failure")
    assert sock.bound == ("192.168.1.103", 20001)
    assert sock.attempts == []
    assert sock.closed


def test_zero_only_live_sender_never_emits_a_nonzero_axis_value():
    sock = _Socket()
    clock = _Clock()
    run_live(
        host="127.0.0.1",
        port=43893,
        source_address="192.168.1.103",
        local_port=20001,
        duration_s=0.1,
        axis_value=0,
        socket_factory=lambda *_: sock,
        clock=clock,
        sleep=clock.sleep,
    )
    assert sock.bound == ("192.168.1.103", 20001)
    assert sock.closed
    assert any(frame == (HEARTBEAT_CODE, 0, 0) for frame, _address in sock.frames)
    assert all(
        frame != (FORWARD_AXIS_CODE, FORWARD_AXIS_VALUE, 0)
        for frame, _address in sock.frames
    )
    assert all(
        frame in ((HEARTBEAT_CODE, 0, 0), (FORWARD_AXIS_CODE, 0, 0))
        for frame, _address in sock.frames
    )


def test_live_axis_sender_refuses_non_documented_axis_value():
    try:
        run_live(
            host="127.0.0.1",
            port=43893,
            source_address=None,
            local_port=20001,
            duration_s=0.1,
            axis_value=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("accepted unsupported axis value")


def test_axis_sender_refuses_nonpositive_or_unbounded_duration_before_socket_creation():
    constructed = []

    def socket_factory(*_args):
        constructed.append(True)
        return _Socket()

    for duration in (0.0, -0.1, float("nan"), 1.1):
        try:
            run_live(
                host="127.0.0.1",
                port=43893,
                source_address=None,
                local_port=20001,
                duration_s=duration,
                socket_factory=socket_factory,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe duration {duration!r}")
    assert constructed == []

    sock = _Socket()
    for duration in (0.0, -0.1, float("nan")):
        try:
            send_for_duration(sock, "127.0.0.1", 43893, 0, duration)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sent a frame for unsafe duration {duration!r}")
    assert sock.frames == []


def test_axis_schedule_does_not_accumulate_send_overhead():
    clock = _Clock()
    sock = _DelayedSocket(clock)
    send_for_duration(
        sock,
        "127.0.0.1",
        43893,
        FORWARD_AXIS_VALUE,
        0.7,
        clock=clock,
        sleep=clock.sleep,
    )
    axis_times = [
        timestamp
        for frame, timestamp in sock.sent_at
        if frame == (FORWARD_AXIS_CODE, FORWARD_AXIS_VALUE, 0)
    ]
    heartbeat_times = [
        timestamp
        for frame, timestamp in sock.sent_at
        if frame == (HEARTBEAT_CODE, 0, 0)
    ]
    assert len(axis_times) == 14
    assert all(
        later - earlier <= 1.0 / 20.0 + 1e-9
        for earlier, later in zip(axis_times, axis_times[1:])
    )
    assert all(
        later - earlier <= 1.0 / 2.0 + 1e-9
        for earlier, later in zip(heartbeat_times, heartbeat_times[1:])
    )


def test_dry_run_never_constructs_a_socket():
    assert main([]) == 0
    assert main(["--zero-only"]) == 0


def test_live_axis_sender_requires_operator_confirmation():
    assert main(["--live"]) == 2


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_axis_udp: {len(tests)}/{len(tests)} passed")
