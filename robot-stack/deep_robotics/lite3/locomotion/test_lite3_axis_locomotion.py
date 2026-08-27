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


def _profile_data(*, measured_m_s=None, measured_rad_s=None, **primitives):
    return {
        "schema": AXIS_PROFILE_SCHEMA,
        "input_deadband": {
            "linear_m_s": 0.05,
            "yaw_rad_s": 0.10,
        },
        "allowed_gait_states": [0],
        "measured_m_s": {} if measured_m_s is None else measured_m_s,
        "measured_rad_s": {} if measured_rad_s is None else measured_rad_s,
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


def test_the_linear_deadband_gates_the_vector_rather_than_each_axis():
    """A per-axis gate drops the smaller component, which rotates the command."""
    profile = _load_profile(_profile_data())  # linear deadband 0.05 m/s

    # 0.071 m/s at 46 degrees. Both components straddle the gate; a per-axis gate passed
    # only the lateral one and turned a near-diagonal into a full-scale 90 degree strafe.
    axes = profile.map_velocity(0.049, 0.051, 0.0)
    assert axes.forward == 7000 and axes.lateral == -13000

    # 0.057 m/s at 45 degrees: neither component clears 0.05 on its own, the vector does.
    assert profile.map_velocity(0.04, 0.04, 0.0) == AxisValues(7000, -13000, 0)

    # 0.306 m/s at 11 degrees: nearly straight ahead, so it goes straight ahead.
    assert profile.map_velocity(0.30, 0.06, 0.0) == AxisValues(7000, 0, 0)

    # 0.346 m/s at 30 degrees: past halfway to the diagonal, so it takes the diagonal.
    # The bearing is snapped to the NEAREST expressible direction, not down to the last
    # one passed, and an exact 45 degrees stays on the diagonal rather than tipping to 90.
    assert profile.map_velocity(0.30, 0.1732, 0.0) == AxisValues(7000, -13000, 0)
    assert profile.map_velocity(0.20, 0.20, 0.0) == AxisValues(7000, -13000, 0)

    # 0.042 m/s: below the gate as a vector, so neither axis moves.
    assert profile.map_velocity(0.03, 0.03, 0.0) == AxisValues()

    # Yaw is a third axis with its own vendor dead zone, not part of the linear vector.
    assert profile.map_velocity(0.30, 0.0, 0.05) == AxisValues(7000, 0, 0)
    assert profile.map_velocity(0.0, 0.0, 0.20) == AxisValues(0, 0, -10000)


def test_the_mapping_is_sign_only_so_commanded_magnitude_never_reaches_the_wire():
    """Pinned deliberately: this is the property that makes the preflight gate necessary.

    ``Lite3Bindings._validate_axis_profile_speeds`` enforces ``--derate`` because
    ``map_velocity`` cannot. If this ever starts scaling, that gate needs revisiting
    rather than silently double-derating.
    """
    profile = _load_profile(_profile_data())
    emitted = {profile.map_velocity(0.30 * derate, 0.0, 0.0).forward
               for derate in (1.0, 0.6, 0.3, 0.2)}
    assert emitted == {7000}


def test_measured_primitive_speeds_must_name_a_real_primitive_and_be_positive():
    """The measurement is what preflight compares against the derated envelope."""
    profile = _load_profile(_profile_data(measured_m_s={"forward_positive": 0.729},
                                          measured_rad_s={"yaw_positive": 0.55}))
    assert profile.measured_speeds == {"forward_positive": 0.729, "yaw_positive": 0.55}
    assert _load_profile(_profile_data()).measured_speeds == {}

    cases = (
        (_profile_data(forward_positive=None, measured_m_s={"forward_positive": 0.7}),
         "has no primitive value"),
        (_profile_data(measured_m_s={"yaw_positive": 0.5}), "expected one of"),
        (_profile_data(measured_rad_s={"forward_positive": 0.5}), "expected one of"),
        (_profile_data(measured_m_s={"forward_positive": 0.0}), "finite and positive"),
        (_profile_data(measured_m_s={"forward_positive": float("inf")}),
         "finite and positive"),
        (_profile_data(measured_m_s=[["forward_positive", 0.7]]), "must be an object"),
    )
    for data, expected in cases:
        try:
            _load_profile(data)
        except AxisProfileError as error:
            assert expected in str(error), f"{error} does not mention {expected!r}"
        else:
            raise AssertionError(
                f"accepted an unusable speed measurement: {data['measured_m_s']} "
                f"{data['measured_rad_s']}"
            )


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


def test_the_vendor_state_gate_refuses_a_snapshot_that_stopped_arriving():
    """``prepare_motion`` calls this gate at pre-flight with no age check ahead of it.

    ``set_velocity`` checks ``state_age()`` and then calls the gate, so a stale snapshot
    never reaches it that way. ``Lite3Bindings.prepare_motion`` calls it directly, and a
    link that died between ``connect()`` and pre-flight would otherwise authorise motion
    from a frozen ``basic=6`` recorded seconds earlier.
    """
    clock = _Clock()
    loco = Lite3AxisLocomotion(
        axis_profile=_load_profile(_profile_data()),
        motion_host="127.0.0.1",
        command_port=43893,
        state_port=0,
        bind="127.0.0.1",
        clock=clock,
        state_timeout_s=0.1,
    )
    # Everything the gate inspects is healthy and permitted; only the age is wrong.
    loco._state = SimpleNamespace(
        received_at=clock.now, error_state=0, mode=(6, 0, 0, 0))
    loco.assert_axis_state_ready()  # fresh: must not raise

    clock.now += 0.11
    try:
        loco.assert_axis_state_ready()
    except Lite3LinkLost as error:
        assert "silent" in str(error)
    else:
        raise AssertionError("authorised axis motion from a snapshot that stopped arriving")


def test_set_velocity_checks_the_vendor_state_before_it_creates_a_streamer():
    """``assert_axis_state_ready`` is well covered; that it is *called* was not.

    Deleting the call in ``set_velocity`` left this suite at 10/10. The distinction
    matters: everything else here is fresh and healthy, and the only thing wrong is a
    gait the profile does not allow.
    """
    created = []

    def streamer_factory(**_kwargs):
        streamer = _Streamer()
        created.append(streamer)
        return streamer

    clock = _Clock()
    loco = Lite3AxisLocomotion(
        axis_profile=_load_profile(_profile_data()),
        motion_host="127.0.0.1",
        command_port=43893,
        state_port=0,
        bind="127.0.0.1",
        clock=clock,
        streamer_factory=streamer_factory,
    )
    loco._state = SimpleNamespace(received_at=clock.now, error_state=0, mode=(6, 2, 0, 0))
    try:
        loco.set_velocity(0.2, 0.0, 0.0)
    except Lite3LinkLost as error:
        assert "gait_state=2" in str(error)
    else:
        raise AssertionError("commanded axis motion in a gait the profile does not allow")
    assert created == []
    assert loco._streamer is None


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


def test_a_sub_floor_command_fires_the_full_speed_primitive_rather_than_crawling():
    """⚠️ THE GO2'S GAIT-FLOOR BUG INVERTS ON THIS TRANSPORT, AND NOBODY HAD SAID SO.

    Issue #26 is about a planner that slows down near an obstacle and commands a speed
    the robot cannot walk at. On a Go2 that is a freeze: 0.05 m/s produces no gait, the
    robot stands still, and nothing faults. THIS transport discards the magnitude, so the
    same 0.05 m/s command is above ``input_deadband.linear_m_s`` and fires the forward
    primitive at whatever speed the profile evidenced it at. The careful crawl leaves as
    a full-speed walk at the obstacle the planner was creeping past.

    So a fix for #26 phrased as clamping or scaling the MAGNITUDE is a no-op here — there
    is no magnitude to clamp. The only thing a sign-only transport can be told is GO or
    STOP, which is why the guard in ``avoidance.DynamicWindowPlanner._gait_floor_stop``
    emits a stop rather than a slower command: ``map_velocity(0, 0, 0)`` is zero raw on
    every axis, and that sentence is expressible on every platform in this repository.

    Measured against the shipped example profile's 0.05 m/s deadband. The executable
    forward set is two values — ``{0, the evidenced primitive speed}`` — and this pins
    both the gap and its edges.
    """
    profile = _load_profile(_profile_data(measured_m_s={"forward_positive": 0.30}))
    assert profile.linear_deadband_m_s == 0.05, profile.linear_deadband_m_s
    assert profile.measured_speeds["forward_positive"] == 0.30

    # Under the deadband: nothing moves. This is the ONLY way this transport goes slow.
    for crawl in (0.0, 0.02, 0.049):
        assert profile.map_velocity(crawl, 0.0, 0.0).forward == 0, crawl

    # At and above it: one primitive, one speed, whatever was asked for. 0.05 m/s of
    # intent and 0.55 m/s of intent are the same bytes on the wire.
    executed = {profile.map_velocity(vx, 0.0, 0.0).forward
                for vx in (0.05, 0.10, 0.20, 0.34, 0.35, 0.55)}
    assert executed == {7000}, executed
    assert profile.map_velocity(0.05, 0.0, 0.0) == profile.map_velocity(0.55, 0.0, 0.0), (
        "if these ever differ this transport has grown a magnitude and the reasoning in "
        "this test — and in `_gait_floor_stop` — has to be redone")


def test_the_only_two_forward_speeds_this_transport_has_are_zero_and_the_primitive():
    """The set an ``avoidance.Limits.gait_floor`` for this robot has to describe.

    Stated as a set rather than as prose because ``--gait-floor`` is a single number and
    a reader will assume it names the bottom of a range. It does not: there is no range.
    Issue #42 is the other half of this — one field cannot hold a forward floor and a
    lateral one that differ by 2x — and neither can it hold a floor that is also the
    ceiling.
    """
    profile = _load_profile(_profile_data(
        measured_m_s={"forward_positive": 0.30, "lateral_positive": 0.12}))
    speeds = {0.0}
    for vx in (0.0, 0.03, 0.05, 0.12, 0.30, 0.9):
        raw = profile.map_velocity(vx, 0.0, 0.0).forward
        speeds.add(0.0 if raw == 0 else profile.measured_speeds["forward_positive"])
    assert speeds == {0.0, 0.30}, speeds


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_axis_locomotion: {len(tests)}/{len(tests)} passed")
