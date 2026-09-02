#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for Lite3 high-level locomotion over the vendor's UDP interface.

These run over real loopback sockets on ephemeral ports rather than against a mocked
transport. The thing most worth getting wrong here is the twenty bytes on the wire, and a
mock would happily accept a frame the motion host would reject or, worse, misread.
"""

from __future__ import annotations

import socket
import struct
import sys
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning.lite3_state_probe import ROBOT_STATE_CODE
from deep_robotics.lite3.locomotion.lite3_locomotion import Lite3Locomotion
from deep_robotics.lite3.locomotion.lite3_udp_locomotion import (
    FORWARD_VELOCITY_CODE,
    LATERAL_VELOCITY_CODE,
    STATE_TIMEOUT_S,
    YAW_VELOCITY_CODE,
    Lite3LinkLost,
    Lite3UdpLocomotion,
    udp_locomotion_factory,
)

_COMMAND = struct.Struct("<3id")

# offsetof(RobotState, field), from compiling the vendor protocol.hpp on aarch64.
_OFFSETS = {"rpy": 16, "rpy_vel": 40, "pos": 88, "velw": 112, "velb": 136,
            "motion": 172, "batt": 176}


def _state_frame(*, x=1.0, y=2.0, yaw_deg=30.0, vx=0.4, vy=-0.05,
                 reported_yaw_rate=0.3, battery=88.0) -> bytes:
    buffer = bytearray(220)
    struct.pack_into("<3i", buffer, 0, ROBOT_STATE_CODE, 208, 0)
    struct.pack_into("<3i", buffer, 12, 3, 2, 1)
    struct.pack_into("<3d", buffer, 12 + _OFFSETS["rpy"], 0.0, 0.0, yaw_deg)
    struct.pack_into("<3d", buffer, 12 + _OFFSETS["rpy_vel"], 0.0, 0.0, reported_yaw_rate)
    struct.pack_into("<3d", buffer, 12 + _OFFSETS["pos"], x, y, 0.32)
    struct.pack_into("<3d", buffer, 12 + _OFFSETS["velw"], vx, vy, 0.0)
    struct.pack_into("<3d", buffer, 12 + _OFFSETS["velb"], vx, vy, 0.0)
    struct.pack_into("<i", buffer, 12 + _OFFSETS["motion"], 4)
    struct.pack_into("<d", buffer, 12 + _OFFSETS["batt"], battery)
    return bytes(buffer)


class _MotionHost:
    """Stands in for the robot: receives commands, sends state."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(2.0)
        self.port = self.sock.getsockname()[1]

    def received(self, count):
        frames = []
        for _ in range(count):
            payload, _address = self.sock.recvfrom(2048)
            frames.append(_COMMAND.unpack(payload))
        return frames

    def send_state(self, port, **kwargs):
        self.sock.sendto(_state_frame(**kwargs), ("127.0.0.1", port))

    def close(self):
        self.sock.close()


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _connected(host, clock=None, **kwargs):
    """A locomotion object bound to an ephemeral state port, with one state frame in."""
    loco = Lite3UdpLocomotion(motion_host="127.0.0.1", command_port=host.port,
                              state_port=0, bind="127.0.0.1",
                              connect_timeout_s=3.0,
                              **({"clock": clock} if clock else {}), **kwargs)
    loco._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    loco._state_socket.bind(("127.0.0.1", 0))
    loco._state_port = loco._state_socket.getsockname()[1]
    loco._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return loco


def test_a_velocity_command_is_the_three_frames_the_vendor_bridge_sends():
    host = _MotionHost()
    loco = _connected(host)
    try:
        loco._state = _Snapshot(loco)
        loco.set_velocity(0.35, -0.10, 0.40)
        frames = host.received(3)
    finally:
        loco.shutdown()
        host.close()

    assert [frame[0] for frame in frames] == [
        FORWARD_VELOCITY_CODE, LATERAL_VELOCITY_CODE, YAW_VELOCITY_CODE]
    assert all(frame[1] == 8 and frame[2] == 1 for frame in frames)
    assert abs(frames[0][3] - 0.35) < 1e-12
    assert abs(frames[1][3] - (-0.10)) < 1e-12
    # The bridge transmits -angular.z; a positive (left) command goes out negated.
    assert abs(frames[2][3] - (-0.40)) < 1e-12


def test_the_command_frame_is_twenty_bytes():
    host = _MotionHost()
    loco = _connected(host)
    try:
        loco._state = _Snapshot(loco)
        loco.set_velocity(0.1, 0.0, 0.0)
        payload, _address = host.sock.recvfrom(2048)
    finally:
        loco.shutdown()
        host.close()
    assert len(payload) == 20


def test_a_blind_command_is_refused_but_stop_still_transmits():
    """A stop must survive the failure that makes commanding unsafe.

    If the state stream drops, commanding motion is driving blind and is refused. Refusing
    the *stop* as well would leave a walking robot with no software brake, so stop() is
    deliberately exempt.
    """
    host = _MotionHost()
    clock = _Clock()
    loco = _connected(host, clock=clock)
    try:
        loco._state = _Snapshot(loco)
        clock.now += 10.0  # the robot has gone quiet
        try:
            loco.set_velocity(0.3, 0.0, 0.0)
        except Lite3LinkLost as error:
            assert "silent" in str(error)
        else:
            raise AssertionError("commanded motion with no state for ten seconds")
        loco.stop()
        frames = host.received(3)
    finally:
        loco.shutdown()
        host.close()
    assert [frame[3] for frame in frames] == [0.0, 0.0, 0.0]


def test_a_zero_velocity_is_not_treated_as_commanding_motion():
    host = _MotionHost()
    clock = _Clock()
    loco = _connected(host, clock=clock)
    try:
        loco._state = _Snapshot(loco)
        clock.now += 10.0
        loco.set_velocity(0.0, 0.0, 0.0)  # must not raise
        assert len(host.received(3)) == 3
    finally:
        loco.shutdown()
        host.close()


def test_connect_fails_loudly_when_the_robot_streams_somewhere_else():
    """The wrong destination address must not present as a robot that simply never moves."""
    loco = Lite3UdpLocomotion(motion_host="127.0.0.1", command_port=59999,
                              state_port=0, bind="127.0.0.1", connect_timeout_s=0.3)
    try:
        loco.connect()
    except Lite3LinkLost as error:
        assert "network.toml" in str(error)
    else:
        loco.shutdown()
        raise AssertionError("connect() succeeded with no state stream")


def test_connect_reports_arriving_undecodable_frames_instead_of_blaming_the_network():
    """A firmware whose frame length this build cannot decode is not a network fault.

    The original message named ``network.toml`` unconditionally. On the second robot,
    which streams a 212-byte state frame, that sent an hour of debugging into an innocent
    file while datagrams arrived at 34 Hz and were dropped by length. See whitepaper A21.
    """
    scout = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    scout.bind(("127.0.0.1", 0))
    state_port = scout.getsockname()[1]
    scout.close()

    stop_feeding = threading.Event()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def feed():
        while not stop_feeding.is_set():
            try:
                # A length no Lite3 frame has: arriving, and undecodable.
                sender.sendto(b"\x00" * 999, ("127.0.0.1", state_port))
            except OSError:
                return
            stop_feeding.wait(0.02)

    feeder = threading.Thread(target=feed, daemon=True)
    loco = Lite3UdpLocomotion(motion_host="127.0.0.1", command_port=59999,
                              state_port=state_port, bind="127.0.0.1",
                              connect_timeout_s=0.5)
    try:
        feeder.start()
        loco.connect()
    except Lite3LinkLost as error:
        message = str(error)
        assert "999 B" in message, message
        assert "DID arrive" in message, message
        # The whole point: it must not send the reader to the address config.
        assert "network.toml" not in message, message
    else:
        loco.shutdown()
        raise AssertionError("connect() succeeded on undecodable frames")
    finally:
        stop_feeding.set()
        feeder.join(timeout=2.0)
        sender.close()


def test_connect_then_decode_pose_velocity_and_battery():
    host = _MotionHost()
    # Claim an ephemeral port, release it, and hand the number to connect(). Datagrams
    # sent before the bind are simply dropped, and the feeder keeps sending, so connect()
    # still sees one without the test having to synchronise on the bind.
    scout = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    scout.bind(("127.0.0.1", 0))
    state_port = scout.getsockname()[1]
    scout.close()

    stop_feeding = threading.Event()

    def feed():
        while not stop_feeding.is_set():
            try:
                host.send_state(state_port, x=1.5, y=-0.5, yaw_deg=90.0,
                                vx=0.31, vy=0.02, battery=76.5)
            except OSError:
                return
            stop_feeding.wait(0.02)

    feeder = threading.Thread(target=feed, daemon=True)
    loco = Lite3UdpLocomotion(motion_host="127.0.0.1", command_port=host.port,
                              state_port=state_port, bind="127.0.0.1",
                              connect_timeout_s=3.0)
    try:
        feeder.start()
        loco.connect()

        pose = loco.pose()
        assert abs(pose.x - 1.5) < 1e-9
        assert abs(pose.y - (-0.5)) < 1e-9
        assert abs(pose.yaw - 1.5707963267948966) < 1e-9  # 90 deg in radians
        vx, vy, _vyaw = loco.velocity()
        assert abs(vx - 0.31) < 1e-9
        assert abs(vy - 0.02) < 1e-9
        assert abs(loco.battery_level() - 76.5) < 1e-9
        assert loco.state_age() is not None
    finally:
        stop_feeding.set()
        feeder.join(timeout=2.0)
        loco.shutdown()
        host.close()


def test_a_frozen_snapshot_is_not_reported_as_a_current_battery_reading():
    """A dead link must not read as a healthy battery.

    ``_require_state`` raises only when *no* frame has ever arrived, so one frame used to
    be enough for ``battery_level()`` to return that value forever.
    ``Lite3HealthMonitor._poll`` re-stamps whatever it gets with its own clock at 10 Hz,
    which makes ``HEALTH_STALE_S`` measure the age of the stamp rather than the age of
    the frame. Measured before the fix: 5.01 s of silence -- ten times
    ``STATE_TIMEOUT_S`` -- with ``abort_reason()`` still ``None`` on a frozen 50%.
    """
    host = _MotionHost()
    clock = _Clock()
    loco = _connected(host, clock=clock)
    try:
        loco._state = _Snapshot(loco)
        assert abs(loco.battery_level() - 90.0) < 1e-9  # fresh, so it is reported

        clock.now += STATE_TIMEOUT_S + 0.01  # the stream stopped; the snapshot did not
        try:
            loco.battery_level()
        except Lite3LinkLost as error:
            assert "silent for 0.51s" in str(error)
        else:
            raise AssertionError("a frozen snapshot was reported as a current battery")

        # Deliberately unchanged. The navigator reads pose() on every tick with no
        # handler, and the health gate above is what turns a dead link into a diagnosed
        # abort rather than a traceback out of the control loop.
        assert loco.pose().x == 0.0
        assert loco.mode() == (3, 2, 1, 4)
    finally:
        loco.shutdown()
        host.close()


def test_yaw_rate_is_differentiated_from_pose_not_taken_from_the_ambiguous_field():
    """The reported field is off by 57x if its unit is guessed; pose yaw cannot be.

    ``rpy_vel`` here reports 30, which is right if the firmware means degrees/s and wrong
    by a factor of 57.3 if it means radians/s. The pose moves 30 deg in 1 s either way, so
    a correct implementation returns 0.5236 rad/s and never inspects the field.
    """
    loco = Lite3UdpLocomotion(motion_host="127.0.0.1", state_port=0)
    clock = _Clock()
    loco._clock = clock
    loco._publish(_decoded(yaw_deg=0.0, reported_yaw_rate=30.0))
    clock.now += 1.0
    loco._publish(_decoded(yaw_deg=30.0, reported_yaw_rate=30.0))
    _vx, _vy, vyaw = loco.velocity()
    assert abs(vyaw - 0.5235987755982988) < 1e-9
    assert abs(loco.reported_yaw_rate() - 30.0) < 1e-9


def test_yaw_rate_folds_a_wrap_instead_of_spiking():
    loco = Lite3UdpLocomotion(motion_host="127.0.0.1", state_port=0)
    clock = _Clock()
    loco._clock = clock
    loco._publish(_decoded(yaw_deg=179.0))
    clock.now += 1.0
    loco._publish(_decoded(yaw_deg=-179.0))
    _vx, _vy, vyaw = loco.velocity()
    assert abs(vyaw - 0.03490658503988659) < 1e-9  # +2 deg/s, not -358


def test_the_factory_names_the_ros_arguments_it_cannot_honour():
    factory = udp_locomotion_factory(motion_host="10.0.0.5", command_port=1, state_port=2)
    implementation = factory(cmd_vel_topic="/cmd_vel", odom_topic="/leg_odom2",
                             stamped=False, node_name="anything")
    assert isinstance(implementation, Lite3UdpLocomotion)
    assert implementation._motion_host == "10.0.0.5"


def test_it_satisfies_the_lite3_locomotion_contract():
    """Lite3Locomotion must drive it with no change to its own code."""
    host = _MotionHost()
    loco = Lite3Locomotion(operator_ready=True,
                           implementation_factory=udp_locomotion_factory(
                               motion_host="127.0.0.1", command_port=host.port,
                               state_port=0, bind="127.0.0.1"))
    implementation = None
    try:
        # Bypass connect()'s link wait; this test is about the interface, not the link.
        implementation = loco._implementation_factory(
            cmd_vel_topic=None, odom_topic=None, stamped=False, node_name="x")
        implementation._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        loco._impl = implementation
        loco.prepare_motion()
        loco.stop()
        assert len(host.received(9)) == 9  # stop() repeats three times, three frames each
    finally:
        if implementation is not None:
            implementation.shutdown()
        host.close()


def test_shutdown_is_idempotent_and_releases_sockets():
    host = _MotionHost()
    loco = _connected(host)
    loco.shutdown()
    loco.shutdown()
    assert loco._command_socket is None
    assert loco._state_socket is None
    host.close()


def _Snapshot(loco):
    """A fresh state snapshot stamped at the object's own clock."""
    from deep_robotics.lite3.locomotion.lite3_udp_locomotion import _StateSnapshot
    return _StateSnapshot(received_at=loco._clock(), x=0.0, y=0.0, yaw_rad=0.0,
                          vx=0.0, vy=0.0, reported_yaw_rate=0.0, battery_level=90.0,
                          error_state=0, mode=(3, 2, 1, 4))


def _decoded(*, yaw_deg=0.0, reported_yaw_rate=0.0):
    return {
        "rpy_deg": [0.0, 0.0, yaw_deg], "rpy_vel": [0.0, 0.0, reported_yaw_rate],
        "pos_world": [0.0, 0.0, 0.32], "vel_body": [0.0, 0.0, 0.0],
        "battery_level": 90.0, "robot_basic_state": 3, "robot_gait_state": 2,
        "robot_policy_state": 1, "robot_motion_state": 4, "error_state": 0,
    }


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_udp_locomotion: {len(tests)}/{len(tests)} passed")
