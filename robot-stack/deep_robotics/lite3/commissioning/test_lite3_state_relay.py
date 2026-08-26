#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline loopback tests for the Lite3 telemetry relay."""

from __future__ import annotations

import ast
import socket
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning.lite3_state_probe import ROBOT_STATE_CODE
from deep_robotics.lite3.commissioning.lite3_state_relay import (
    MOTION_COMMAND_PORT,
    Lite3StateRelay,
)


def _robot_state_frame() -> bytes:
    frame = bytearray(220)
    struct.pack_into("<3i", frame, 0, ROBOT_STATE_CODE, 208, 0)
    struct.pack_into("<3i", frame, 12, 3, 2, 1)
    struct.pack_into("<3d", frame, 28, 0.0, 0.0, 30.0)
    struct.pack_into("<3d", frame, 100, 1.0, 2.0, 0.32)
    struct.pack_into("<3d", frame, 148, 0.2, -0.1, 0.0)
    struct.pack_into("<d", frame, 188, 80.0)
    return bytes(frame)


def _handle_state_frame() -> bytes:
    frame = bytearray(60)
    struct.pack_into("<3i", frame, 0, 2309, 48, 0)
    return bytes(frame)


def _receiver():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    return receiver


def test_relay_forwards_only_a_valid_raw_state_frame():
    target = _receiver()
    relay = Lite3StateRelay(
        listen_host="127.0.0.1",
        listen_port=0,
        target_host="127.0.0.1",
        target_port=target.getsockname()[1],
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        relay.start()
        payload = _robot_state_frame()
        sender.sendto(payload, ("127.0.0.1", relay.listen_port))
        assert relay.forward_once(1.0)
        received, _source = target.recvfrom(2048)
        assert received == payload
    finally:
        sender.close()
        relay.shutdown()
        target.close()


def test_relay_rejects_unknown_frames_without_forwarding_them():
    target = _receiver()
    relay = Lite3StateRelay(
        listen_host="127.0.0.1",
        listen_port=0,
        target_host="127.0.0.1",
        target_port=target.getsockname()[1],
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        relay.start()
        sender.sendto(b"not a Lite3 state frame", ("127.0.0.1", relay.listen_port))
        assert not relay.forward_once(1.0)
        try:
            target.recvfrom(2048)
        except socket.timeout:
            pass
        else:
            raise AssertionError("sent an undecodable state frame")
    finally:
        sender.close()
        relay.shutdown()
        target.close()


def test_relay_ignores_valid_non_robot_state_frames():
    target = _receiver()
    relay = Lite3StateRelay(
        listen_host="127.0.0.1",
        listen_port=0,
        target_host="127.0.0.1",
        target_port=target.getsockname()[1],
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        relay.start()
        sender.sendto(_handle_state_frame(), ("127.0.0.1", relay.listen_port))
        assert not relay.forward_once(1.0)
        assert relay._ignored == 1
        try:
            target.recvfrom(2048)
        except socket.timeout:
            pass
        else:
            raise AssertionError("sent a non-RobotState frame")
    finally:
        sender.close()
        relay.shutdown()
        target.close()


def test_relay_binds_configured_output_source_address():
    target = _receiver()
    relay = Lite3StateRelay(
        listen_host="127.0.0.1",
        listen_port=0,
        target_host="127.0.0.1",
        target_port=target.getsockname()[1],
        source_address="127.0.0.1",
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        relay.start()
        sender.sendto(_robot_state_frame(), ("127.0.0.1", relay.listen_port))
        assert relay.forward_once(1.0)
        _payload, source = target.recvfrom(2048)
        assert source[0] == "127.0.0.1"
    finally:
        sender.close()
        relay.shutdown()
        target.close()


def test_failed_source_bind_leaves_relay_unstarted():
    relay = Lite3StateRelay(
        listen_host="127.0.0.1",
        listen_port=0,
        target_host="127.0.0.1",
        target_port=43898,
        source_address="not-an-ip-address",
    )
    try:
        relay.start()
    except OSError:
        pass
    else:
        raise AssertionError("accepted an invalid relay source address")
    assert relay._receiver is None
    assert relay._sender is None


def test_relay_refuses_command_and_loop_ports():
    for listen_port, target_port in ((43897, MOTION_COMMAND_PORT), (43898, 43898)):
        try:
            Lite3StateRelay(
                listen_port=listen_port,
                target_host="127.0.0.1",
                target_port=target_port,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe port pair {listen_port}, {target_port}")


def test_relay_has_no_locomotion_or_vendor_control_import():
    source = (_HERE / "lite3_state_relay.py").read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imported = imported_modules.union(imported_names)
    assert "lite3_axis_udp" not in imported
    assert "lite3_control_mode_udp" not in imported
    assert "Lite3Locomotion" not in imported


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_state_relay: {len(tests)}/{len(tests)} passed")
