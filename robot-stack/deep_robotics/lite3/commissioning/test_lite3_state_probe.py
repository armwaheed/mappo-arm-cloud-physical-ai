#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the receive-only Lite3 high-level state probe.

The frame sizes and field offsets asserted here are not read off the vendor header. They
were produced by compiling ``Lite3_ROS`` ``src/transfer/include/protocol.hpp`` (branch
``ros2-foxy``) and printing ``sizeof``/``offsetof`` on aarch64:

    RobotStateReceived  220    RobotState 208 (align 8)   data at offset 12
    JointStateReceived  108    HandleStateReceived 60     ImuDataReceived 52
    basic 0  gait 4  policy 8  rpy 16  rpy_vel 40  xyz_acc 64  pos_world 88
    vel_world 112  vel_body 136  touch_down 160  is_charging 164  error_state 168
    robot_motion_state 172  battery_level 176  task_state 184  is_robot_need_move 188
    zero_position_flag 189  ultrasound 192

``RobotState`` sits outside the header's ``pack(4)`` region and so keeps natural alignment.
That is the one thing a hand-written format string gets wrong, and it fails silently: every
double after ``rpy`` shears by four bytes and still decodes to a plausible float. The
round-trip test below builds its buffer from those offsets directly, so a format string
that drifts from the C layout fails rather than producing believable numbers.
"""

from __future__ import annotations

import ast
import socket
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning.lite3_state_probe import (
    FRAME_KINDS,
    HANDLE_STATE_CODE,
    IMU_DATA_CODE,
    JOINT_STATE_CODE,
    ROBOT_STATE_CODE,
    DecodeError,
    ProbeStatistics,
    _format_report,
    decode_frame,
    run_probe,
    unwrap_degrees,
)

_PROBE_SOURCE = _HERE / "lite3_state_probe.py"

# offsetof(RobotState, field), from the compiler run quoted above.
_ROBOT_STATE_OFFSETS = {
    "robot_basic_state": 0, "robot_gait_state": 4, "robot_policy_state": 8,
    "rpy": 16, "rpy_vel": 40, "xyz_acc": 64, "pos_world": 88,
    "vel_world": 112, "vel_body": 136, "touch_down_and_stair_trot": 160,
    "is_charging": 164, "error_state": 168, "robot_motion_state": 172,
    "battery_level": 176, "task_state": 184, "is_robot_need_move": 188,
    "zero_position_flag": 189, "ultrasound": 192,
}
_DATA_OFFSET = 12


def _robot_state_frame(**overrides) -> bytes:
    """Build a 220-byte RobotState frame field by field, at the compiler's offsets."""
    fields = {
        "robot_basic_state": 3, "robot_gait_state": 2, "robot_policy_state": 1,
        "rpy": (1.5, -2.5, 30.0), "rpy_vel": (0.1, 0.2, 0.3),
        "xyz_acc": (0.0, 0.0, 9.81), "pos_world": (1.0, 2.0, 0.35),
        "vel_world": (0.4, 0.05, 0.0), "vel_body": (0.42, -0.01, 0.0),
        "touch_down_and_stair_trot": 15, "is_charging": False, "error_state": 0,
        "robot_motion_state": 4, "battery_level": 87.5, "task_state": 0,
        "is_robot_need_move": False, "zero_position_flag": True,
        "ultrasound": (0.0, 1.25),
    }
    fields.update(overrides)

    buffer = bytearray(220)
    struct.pack_into("<3i", buffer, 0, ROBOT_STATE_CODE, 208, 0)
    for name, offset in _ROBOT_STATE_OFFSETS.items():
        at = _DATA_OFFSET + offset
        value = fields[name]
        if name in ("rpy", "rpy_vel", "xyz_acc", "pos_world", "vel_world", "vel_body"):
            struct.pack_into("<3d", buffer, at, *value)
        elif name == "ultrasound":
            struct.pack_into("<2d", buffer, at, *value)
        elif name == "battery_level":
            struct.pack_into("<d", buffer, at, value)
        elif name in ("touch_down_and_stair_trot", "error_state"):
            struct.pack_into("<I", buffer, at, value)
        elif name in ("is_charging", "is_robot_need_move", "zero_position_flag"):
            struct.pack_into("<?", buffer, at, value)
        else:
            struct.pack_into("<i", buffer, at, value)
    return bytes(buffer)


def _handle_state_frame(forward=0.3, side=0.0, yaw=0.0) -> bytes:
    return struct.pack("<3i6d", HANDLE_STATE_CODE, 48, 0,
                       0.5, 0.0, 0.0, forward, side, yaw)


def _imu_frame(angle=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0)) -> bytes:
    return struct.pack("<3iI9f", IMU_DATA_CODE, 40, 0, 0, *angle, *angular_velocity,
                       0.0, 0.0, 9.81)


def test_frame_lengths_match_the_compiled_vendor_header():
    assert sorted(FRAME_KINDS) == [52, 60, 108, 220]
    assert FRAME_KINDS[220] == ("robot_state", ROBOT_STATE_CODE)
    assert FRAME_KINDS[108] == ("joint_state", JOINT_STATE_CODE)
    assert FRAME_KINDS[60] == ("handle_state", HANDLE_STATE_CODE)
    assert FRAME_KINDS[52] == ("imu", IMU_DATA_CODE)


def test_robot_state_decodes_at_the_compilers_offsets_not_a_packed_guess():
    frame = decode_frame(_robot_state_frame())
    assert frame["kind"] == "robot_state"
    # Everything after rpy is where the four bytes of alignment padding land it.
    assert frame["rpy_deg"] == [1.5, -2.5, 30.0]
    assert frame["vel_body"] == [0.42, -0.01, 0.0]
    assert frame["pos_world"] == [1.0, 2.0, 0.35]
    assert frame["battery_level"] == 87.5
    assert frame["robot_motion_state"] == 4
    assert frame["ultrasound"] == [0.0, 1.25]
    assert frame["is_charging"] is False
    assert frame["zero_position_flag"] is True


def test_handle_state_carries_the_firmwares_own_derived_velocity():
    frame = decode_frame(_handle_state_frame(forward=0.28, yaw=-0.4))
    assert frame["kind"] == "handle_state"
    assert frame["goal_vel_forward"] == 0.28
    assert frame["goal_vel_yaw"] == -0.4


def test_a_right_sized_frame_with_the_wrong_code_is_refused():
    payload = bytearray(_robot_state_frame())
    struct.pack_into("<i", payload, 0, 9999)
    try:
        decode_frame(bytes(payload))
    except DecodeError as error:
        assert "9999" in str(error)
    else:
        raise AssertionError("a foreign 220-byte datagram was decoded as robot state")


def test_an_unknown_length_is_refused_rather_than_partially_decoded():
    try:
        decode_frame(b"\x00" * 64)
    except DecodeError as error:
        assert "64" in str(error)
    else:
        raise AssertionError("a 64-byte datagram was decoded")


def test_unwrap_degrees_folds_a_yaw_wrap_instead_of_spiking():
    assert unwrap_degrees(179.0, -179.0) == 2.0
    assert unwrap_degrees(-179.0, 179.0) == -2.0
    assert unwrap_degrees(10.0, 20.0) == 10.0


def test_yaw_unit_check_reads_degrees_per_second_as_degrees():
    statistics = ProbeStatistics()
    for index in range(20):
        # 30 deg/s of real motion, reported as 30 in rpy_vel -> the field is deg/s.
        frame = decode_frame(_robot_state_frame(
            rpy=(0.0, 0.0, index * 3.0), rpy_vel=(0.0, 0.0, 30.0)))
        statistics.observe(frame, index * 0.1)
    unit = statistics.yaw_rate_unit()
    assert unit is not None
    assert abs(unit["median_ratio"] - 1.0) < 0.01
    assert "degrees/s" in unit["verdict"]


def test_yaw_unit_check_reads_radians_per_second_as_radians():
    statistics = ProbeStatistics()
    for index in range(20):
        # The same 30 deg/s of real motion reported as 0.5236 -> the field is rad/s.
        frame = decode_frame(_robot_state_frame(
            rpy=(0.0, 0.0, index * 3.0), rpy_vel=(0.0, 0.0, 0.5235987755982988)))
        statistics.observe(frame, index * 0.1)
    unit = statistics.yaw_rate_unit()
    assert unit is not None
    assert "radians/s" in unit["verdict"]


def test_yaw_unit_check_falls_back_to_imu_when_robot_state_is_absent():
    statistics = ProbeStatistics()
    for index in range(20):
        frame = decode_frame(_imu_frame(
            angle=(0.0, 0.0, index * 3.0),
            angular_velocity=(0.0, 0.0, 0.5235987755982988)))
        statistics.observe(frame, index * 0.1)

    unit = statistics.yaw_rate_unit()

    assert unit is not None
    assert unit["source"] == "imu"
    assert "radians/s" in unit["verdict"]


def test_a_stationary_robot_cannot_decide_the_yaw_unit():
    statistics = ProbeStatistics()
    for index in range(40):
        frame = decode_frame(_robot_state_frame(
            rpy=(0.0, 0.0, 30.0), rpy_vel=(0.0, 0.0, 0.0)))
        statistics.observe(frame, index * 0.1)
    assert statistics.yaw_rate_unit() is None


def test_command_response_pairs_the_remotes_request_with_measured_speed():
    statistics = ProbeStatistics()
    for index in range(5):
        # The remote keeps transmitting, as it does while the operator holds the stick.
        statistics.observe(decode_frame(_handle_state_frame(forward=0.32)), 0.1 * index)
        statistics.observe(
            decode_frame(_robot_state_frame(vel_body=(0.24, 0.0, 0.0))), 0.1 * index)
    rows = statistics.command_response()
    assert len(rows) == 1
    assert rows[0]["samples"] == 5
    assert abs(rows[0]["measured_m_s"] - 0.24) < 1e-9
    # The bin is 0.30-0.35, but the gain divides by what was actually commanded (0.32),
    # not by the bin centre (0.325). At demo speeds that difference is a ~2% error in
    # --actuator-gain, and it would ride into every --max-seconds budget derived from it.
    assert abs(rows[0]["commanded_m_s"] - 0.32) < 1e-9
    assert abs(rows[0]["gain"] - 0.24 / 0.32) < 1e-9


def test_gain_divides_by_the_mean_command_not_the_bin_centre():
    statistics = ProbeStatistics()
    # Two commands in the same 0.30-0.35 bin, both well off its 0.325 centre.
    for commanded in (0.305, 0.345):
        statistics.observe(decode_frame(_handle_state_frame(forward=commanded)), 0.0)
        statistics.observe(
            decode_frame(_robot_state_frame(vel_body=(0.25, 0.0, 0.0))), 0.1)
    rows = statistics.command_response()
    assert len(rows) == 1
    assert abs(rows[0]["commanded_m_s"] - 0.325) < 1e-9  # mean happens to equal the centre
    for commanded in (0.31, 0.315, 0.32):
        statistics.observe(decode_frame(_handle_state_frame(forward=commanded)), 0.0)
        statistics.observe(
            decode_frame(_robot_state_frame(vel_body=(0.25, 0.0, 0.0))), 0.1)
    rows = statistics.command_response()
    assert abs(rows[0]["commanded_m_s"] - 0.319) < 1e-9  # now it does not


def test_a_command_that_stopped_arriving_is_not_paired_with_new_measurements():
    """A latched command would bias the gait floor down, which is the unsafe direction.

    The operator releases the stick and the remote stops transmitting, but the robot
    coasts to a stop over the next second. Pairing that decaying speed against the last
    command seen would report a robot that walks at a fraction of what it was asked for.
    """
    statistics = ProbeStatistics()
    statistics.observe(decode_frame(_handle_state_frame(forward=0.32)), 0.0)
    statistics.observe(decode_frame(_robot_state_frame(vel_body=(0.24, 0.0, 0.0))), 0.1)
    for index in range(10):
        statistics.observe(
            decode_frame(_robot_state_frame(vel_body=(0.02, 0.0, 0.0))), 0.3 + 0.1 * index)
    rows = statistics.command_response()
    assert len(rows) == 1
    assert rows[0]["samples"] == 1
    assert abs(rows[0]["measured_m_s"] - 0.24) < 1e-9


def test_a_zero_command_is_excluded_so_a_parked_robot_cannot_set_the_gait_floor():
    statistics = ProbeStatistics()
    statistics.observe(decode_frame(_handle_state_frame(forward=0.0)), 0.0)
    for index in range(5):
        statistics.observe(decode_frame(_robot_state_frame()), 0.1 * index)
    assert statistics.command_response() == []


def test_report_distinguishes_a_missing_measurement_from_a_missing_command():
    statistics = ProbeStatistics()
    statistics.observe(decode_frame(_handle_state_frame(forward=0.32)), 0.0)

    report = _format_report(statistics)

    assert "1 forward-command frame arrived (0.320-0.320 m/s)" in report
    assert "no robot_state frame arrived" in report
    assert "no forward command seen" not in report


def test_mode_transitions_record_changes_only():
    statistics = ProbeStatistics()
    statistics.observe(decode_frame(_robot_state_frame(robot_motion_state=1)), 0.0)
    statistics.observe(decode_frame(_robot_state_frame(robot_motion_state=1)), 0.1)
    statistics.observe(decode_frame(_robot_state_frame(robot_motion_state=4)), 0.2)
    assert len(statistics.mode_transitions) == 2
    assert statistics.mode_transitions[0][1]["robot_motion_state"] == 1
    assert statistics.mode_transitions[1][1]["robot_motion_state"] == 4


class _FakeSocket:
    """Records every call so the test can prove nothing was transmitted."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def settimeout(self, value):
        self.calls.append(("settimeout", value))

    def recvfrom(self, size):
        self.calls.append(("recvfrom", size))
        if not self._payloads:
            raise socket.timeout
        return self._payloads.pop(0), ("192.168.1.120", 43893)

    def __getattr__(self, name):
        raise AssertionError(f"the probe called socket.{name}")


def test_run_probe_decodes_and_records_without_transmitting():
    sock = _FakeSocket([_robot_state_frame(), _handle_state_frame(), b"\x00" * 7])
    recorded = []
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 5.0, 5.0])
    statistics = run_probe(sock, 1.0, ProbeStatistics(), record=recorded.append,
                           clock=lambda: next(ticks))
    assert statistics.counts == {"robot_state": 1, "handle_state": 1}
    assert statistics.undecoded == 1
    assert len(recorded) == 2
    assert all(entry["timestamp_s"] is not None for entry in recorded)
    assert {name for name, _ in sock.calls} == {"settimeout", "recvfrom"}


def test_the_probe_module_contains_no_send_path():
    """Structural, not aspirational: adding a sender to this module fails the suite.

    ``jetson2motion`` is unusable for discovery precisely because one executable owns both
    a receiver and a sender aimed at the motion host's command port. This asserts the
    property that makes this module a safe substitute.
    """
    forbidden = {"send", "sendto", "sendall", "sendmsg", "connect", "connect_ex"}
    tree = ast.parse(_PROBE_SOURCE.read_text(encoding="utf-8"))
    offenders = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden
    ]
    assert offenders == [], f"the probe gained a transmit path: {offenders}"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_state_probe: {len(tests)}/{len(tests)} passed")
