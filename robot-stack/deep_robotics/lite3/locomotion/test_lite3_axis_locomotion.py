#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the profile-gated Lite3 simple-axis locomotion transport."""

from __future__ import annotations

import json
import socket
import struct
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning.lite3_state_probe import ROBOT_STATE_CODE
from deep_robotics.lite3.locomotion.lite3_axis_locomotion import (
    AXIS_PROFILE_SCHEMA,
    AxisProfile,
    AxisProfileError,
    AxisStreamSender,
    AxisValues,
    Lite3AxisLocomotion,
)
from deep_robotics.lite3.locomotion.lite3_axis_udp import (
    FORWARD_AXIS_CODE,
    LATERAL_AXIS_CODE,
    YAW_AXIS_CODE,
    axis_packet,
)
from deep_robotics.lite3.locomotion.lite3_udp_locomotion import Lite3LinkLost

_SIMPLE_COMMAND = struct.Struct("<3I")


def _profile_data(**primitives):
    return {
        "schema": AXIS_PROFILE_SCHEMA,
        "input_deadband": {
            "linear_m_s": 0.05,
            "yaw_rad_s": 0.10,
        },
        "allowed_gait_states": [0],
        "evidence": {
            "forward_positive": "test-forward-positive",
            "forward_negative": "test-forward-negative",
            "lateral_positive": "test-lateral-positive",
            "lateral_negative": "test-lateral-negative",
            "yaw_positive": "test-yaw-positive",
            "yaw_negative": "test-yaw-negative",
        },
        "primitives": {
            "forward_positive": 7000,
            "forward_negative": -7000,
            "lateral_positive": 13000,
            "lateral_negative": -13000,
            "yaw_positive": 10000,
            "yaw_negative": -10000,
            **primitives,
        },
    }


def _load_profile(data):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "axis-profile.json"
        path.write_text(json.dumps(data))
        return AxisProfile.load(path)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _Socket:
    def __init__(self):
        self.bound = None
        self.closed = False
        self.frames = []

    def bind(self, address):
        self.bound = address

    def getsockname(self):
        return self.bound

    def sendto(self, payload, address):
        self.frames.append((_SIMPLE_COMMAND.unpack(payload), address))

    def close(self):
        self.closed = True


class _TimedSocket(_Socket):
    def __init__(self, clock):
        super().__init__()
        self._clock = clock
        self.sent_at = []

    def sendto(self, payload, address):
        self.sent_at.append((_SIMPLE_COMMAND.unpack(payload), self._clock()))
        super().sendto(payload, address)


class _Streamer:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.shutdown_called = False
        self.axes = []

    def start(self):
        self.started = True

    def set_axes(self, axes):
        self.axes.append(axes)

    def stop(self):
        self.stopped = True

    def shutdown(self):
        self.shutdown_called = True


class _CommandHost:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.settimeout(0.2)
        self.port = self.socket.getsockname()[1]

    def send_state(self, port):
        frame = bytearray(220)
        struct.pack_into("<3i", frame, 0, ROBOT_STATE_CODE, 208, 0)
        struct.pack_into("<3i", frame, 12, 3, 2, 1)
        struct.pack_into("<3d", frame, 28, 0.0, 0.0, 30.0)
        struct.pack_into("<3d", frame, 100, 1.0, 2.0, 0.32)
        struct.pack_into("<3d", frame, 148, 0.2, 0.0, 0.0)
        struct.pack_into("<i", frame, 184, 0)
        struct.pack_into("<d", frame, 188, 80.0)
        self.socket.sendto(frame, ("127.0.0.1", port))

    def close(self):
        self.socket.close()


def _signed(value):
    return value - (1 << 32) if value >= 1 << 31 else value


def test_axis_packet_encodes_all_documented_moving_axes_with_signed_values():
    values = (
        (FORWARD_AXIS_CODE, 7000),
        (LATERAL_AXIS_CODE, -13000),
        (YAW_AXIS_CODE, 10000),
    )
    for code, value in values:
        packed = _SIMPLE_COMMAND.unpack(axis_packet(code, value))
        assert packed[0] == code
        assert _signed(packed[1]) == value
        assert packed[2] == 0


def test_profile_maps_configured_primitives_and_preserves_input_deadbands():
    profile = _load_profile(_profile_data())
    assert profile.map_velocity(0.2, -0.2, 0.3) == AxisValues(7000, 13000, -10000)
    assert profile.map_velocity(0.01, 0.01, 0.01) == AxisValues()


def test_profile_refuses_inside_deadzone_and_unavailable_primitives():
    data = _profile_data(forward_positive=6553, lateral_positive=None)
    try:
        _load_profile(data)
    except AxisProfileError as error:
        assert "dead zone" in str(error)
    else:
        raise AssertionError("accepted a forward primitive inside the vendor dead zone")

    profile = _load_profile(_profile_data(lateral_negative=None))
    try:
        profile.map_velocity(0.0, 0.2, 0.0)
    except AxisProfileError as error:
        assert "no physically evidenced positive lateral" in str(error)
    else:
        raise AssertionError("silently mapped an unavailable lateral primitive")

    data = _profile_data()
    del data["evidence"]["yaw_positive"]
    try:
        _load_profile(data)
    except AxisProfileError as error:
        assert "evidence reference" in str(error)
    else:
        raise AssertionError("accepted a nonzero primitive without evidence")

    try:
        AxisProfile(
            forward_positive=1,
            forward_negative=None,
            lateral_positive=None,
            lateral_negative=None,
            yaw_positive=None,
            yaw_negative=None,
            linear_deadband_m_s=0.05,
            yaw_deadband_rad_s=0.1,
            allowed_gait_states=(0,),
            evidence=(("forward_positive", "attempted bypass"),),
        )
    except AxisProfileError as error:
        assert "dead zone" in str(error)
    else:
        raise AssertionError("direct construction bypassed primitive dead-zone validation")


def test_streamer_stales_a_setpoint_to_zero_and_encodes_all_axes():
    clock = _Clock()
    sender = AxisStreamSender(clock=clock, sleep=lambda seconds: None)
    fake_socket = _Socket()
    sender._socket = fake_socket
    sender.set_axes(AxisValues(7000, -13000, 10000))
    assert sender.effective_axes() == AxisValues(7000, -13000, 10000)
    sender._send_axes(fake_socket, sender.effective_axes())
    assert [frame for frame, _address in fake_socket.frames] == [
        (FORWARD_AXIS_CODE, 7000, 0),
        (LATERAL_AXIS_CODE, (1 << 32) - 13000, 0),
        (YAW_AXIS_CODE, 10000, 0),
    ]
    clock.now += 0.151
    assert sender.effective_axes() == AxisValues()


def test_streamer_refuses_rates_below_vendor_minima():
    for kwargs, message in (
        ({"axis_rate_hz": 19.9}, "axis rate"),
        ({"heartbeat_hz": 1.9}, "heartbeat rate"),
    ):
        try:
            AxisStreamSender(**kwargs)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"accepted unsafe streamer rate: {kwargs}")


def test_streamer_emits_three_axes_at_20hz_heartbeat_at_4hz_and_ttl_zeros():
    clock = _Clock()
    sender = AxisStreamSender(clock=clock, sleep=lambda _seconds: None)
    fake_socket = _TimedSocket(clock)
    sender._socket = fake_socket
    sender.set_axes(AxisValues(7000, -13000, 10000))
    sender._running.set()

    def advance(seconds):
        clock.now += seconds
        if clock.now >= 1000.3:
            sender._running.clear()

    sender._sleep = advance
    sender._run()

    axis_frames = [
        (frame, timestamp)
        for frame, timestamp in fake_socket.sent_at
        if frame[0] in (FORWARD_AXIS_CODE, LATERAL_AXIS_CODE, YAW_AXIS_CODE)
    ]
    heartbeat_frames = [
        (frame, timestamp)
        for frame, timestamp in fake_socket.sent_at
        if frame[0] not in (FORWARD_AXIS_CODE, LATERAL_AXIS_CODE, YAW_AXIS_CODE)
    ]
    assert len(axis_frames) % 3 == 0
    assert len(axis_frames) >= 18
    assert len(heartbeat_frames) == 2
    axis_times = [timestamp for _frame, timestamp in axis_frames[::3]]
    assert all(
        later - earlier <= 1.0 / 20.0 + 1e-9
        for earlier, later in zip(axis_times, axis_times[1:])
    )
    first_zero = next(
        timestamp
        for frame, timestamp in axis_frames
        if frame[0] == FORWARD_AXIS_CODE and frame[1] == 0
    )
    assert first_zero <= 1000.2 + 1e-9
    assert all(frame[1] == 0 for frame, timestamp in axis_frames if timestamp >= first_zero)


def test_shutdown_sets_zero_before_stopping_the_streamer():
    sender = AxisStreamSender()
    fake_socket = _Socket()
    sender._socket = fake_socket
    sender._running.set()
    sender._setpoint = SimpleNamespace(values=AxisValues(7000, 0, 0), updated_at=0.0)
    sender._send_zeros_for_duration = lambda _sock: None
    sender.shutdown()
    assert sender._setpoint.values == AxisValues()
    assert fake_socket.closed


def test_axis_locomotion_requires_profile_and_fresh_state_before_starting_stream():
    clock = _Clock()
    no_profile = Lite3AxisLocomotion(
        axis_profile=None,
        motion_host="127.0.0.1",
        command_port=43893,
        state_port=0,
        bind="127.0.0.1",
        clock=clock,
    )
    try:
        no_profile.set_velocity(0.2, 0.0, 0.0)
    except AxisProfileError:
        pass
    else:
        raise AssertionError("started an axis stream without a profile")

    created = []

    def streamer_factory(**_kwargs):
        streamer = _Streamer()
        created.append(streamer)
        return streamer

    loco = Lite3AxisLocomotion(
        axis_profile=_load_profile(_profile_data()),
        motion_host="127.0.0.1",
        command_port=43893,
        state_port=0,
        bind="127.0.0.1",
        clock=clock,
        streamer_factory=streamer_factory,
    )
    loco._state = SimpleNamespace(
        received_at=clock.now, error_state=0, mode=(6, 0, 0, 0),
    )
    loco.set_velocity(0.2, 0.0, -0.3)
    assert len(created) == 1
    assert created[0].started
    assert created[0].axes == [AxisValues(7000, 0, 10000)]
    loco.stop()
    loco.shutdown()
    assert created[0].stopped
    assert created[0].shutdown_called

    stale = Lite3AxisLocomotion(
        axis_profile=_load_profile(_profile_data()),
        motion_host="127.0.0.1",
        command_port=43893,
        state_port=0,
        bind="127.0.0.1",
        clock=clock,
        state_timeout_s=0.1,
        streamer_factory=streamer_factory,
    )
    stale._state = SimpleNamespace(
        received_at=clock.now - 0.11, error_state=0, mode=(6, 0, 0, 0),
    )
    try:
        stale.set_velocity(0.2, 0.0, 0.0)
    except Lite3LinkLost:
        pass
    else:
        raise AssertionError("started an axis stream with stale state")
    assert len(created) == 1


def test_axis_locomotion_refuses_undocumented_or_unhealthy_vendor_state():
    profile = _load_profile(_profile_data())
    cases = (
        (98, 0, 0, 0, 0, "basic_state=98"),
        (6, 0, 16, 0, 0, "policy_state=16"),
        (6, 2, 0, 0, 0, "gait_state=2"),
        (6, 0, 0, 4, 0, "motion_state=4"),
        (6, 0, 0, 0, 7, "error_state=7"),
    )
    for basic, gait, policy, motion, error_state, expected in cases:
        loco = Lite3AxisLocomotion(
            axis_profile=profile,
            motion_host="127.0.0.1",
            command_port=43893,
            state_port=0,
            bind="127.0.0.1",
        )
        loco._state = SimpleNamespace(
            received_at=loco._clock(),
            error_state=error_state,
            mode=(basic, gait, policy, motion),
        )
        try:
            loco.assert_axis_state_ready()
        except Lite3LinkLost as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"accepted unsafe vendor state {(basic, gait, policy, motion)}")


def test_axis_connect_discards_legacy_command_socket_without_sending():
    host = _CommandHost()
    scout = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    scout.bind(("127.0.0.1", 0))
    state_port = scout.getsockname()[1]
    scout.close()
    stop = threading.Event()

    def feed():
        while not stop.is_set():
            host.send_state(state_port)
            stop.wait(0.02)

    feeder = threading.Thread(target=feed, daemon=True)
    loco = Lite3AxisLocomotion(
        axis_profile=_load_profile(_profile_data()),
        motion_host="127.0.0.1",
        command_port=host.port,
        state_port=state_port,
        bind="127.0.0.1",
        connect_timeout_s=3.0,
    )
    try:
        feeder.start()
        loco.connect()
        assert loco._command_socket is None
        try:
            host.socket.recvfrom(2048)
        except socket.timeout:
            pass
        else:
            raise AssertionError("axis connect sent a legacy complex-velocity packet")
    finally:
        stop.set()
        feeder.join(timeout=1.0)
        loco.shutdown()
        host.close()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_axis_locomotion: {len(tests)}/{len(tests)} passed")
