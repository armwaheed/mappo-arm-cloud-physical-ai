#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 high-level ROS locomotion adapter."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.locomotion.lite3_locomotion import Lite3Locomotion


class _Implementation:
    def __init__(self):
        self.calls = []

    def connect(self):
        self.calls.append("connect")

    def set_velocity(self, vx, vy, wz):
        self.calls.append((vx, vy, wz))

    def stop(self):
        self.calls.append("stop")

    def pose(self):
        return "pose"

    def velocity(self):
        return (0.2, -0.1, 0.3)

    def shutdown(self):
        self.calls.append("shutdown")


class _BatteryImplementation(_Implementation):
    def battery_level(self):
        return 76.5


def _loco(operator_ready=True, implementation=None):
    if implementation is None:
        implementation = _Implementation()
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return implementation

    loco = Lite3Locomotion(operator_ready=operator_ready,
                           implementation_factory=factory, sleep=lambda _seconds: None)
    return loco, implementation, captured


def test_the_documented_lite3_topics_are_the_defaults():
    loco, _implementation, captured = _loco()
    loco.connect()
    assert captured["cmd_vel_topic"] == "/cmd_vel"
    assert captured["odom_topic"] == "/leg_odom2"
    assert captured["stamped"] is False


def test_connect_is_single_owner_and_cannot_leak_a_second_ros_node():
    loco, _implementation, _captured = _loco()
    loco.connect()
    try:
        loco.connect()
    except RuntimeError as exc:
        assert "already connected" in str(exc)
        return
    raise AssertionError("a second ROS locomotion node was created")


def test_motion_is_refused_until_the_operator_confirms_high_level_navigation_mode():
    loco, _implementation, _captured = _loco(operator_ready=False)
    loco.connect()
    try:
        loco.prepare_motion()
    except RuntimeError as exc:
        assert "STANDING + high-level navigation mode" in str(exc)
        return
    raise AssertionError("motion was accepted without the operator readiness gate")


def test_velocity_and_measured_state_delegate_to_the_shared_ros_binding():
    loco, implementation, _captured = _loco()
    loco.connect()
    loco.set_velocity(0.3, -0.1, 0.2)
    assert implementation.calls[-1] == (0.3, -0.1, 0.2)
    assert loco.pose() == "pose"
    assert loco.velocity() == (0.2, -0.1, 0.3)


def test_battery_level_delegates_to_the_selected_transport():
    loco, _implementation, _captured = _loco(implementation=_BatteryImplementation())
    loco.connect()
    assert loco.battery_level() == 76.5


def test_battery_level_is_refused_clearly_when_the_transport_cannot_report_it():
    loco, _implementation, _captured = _loco()
    loco.connect()
    try:
        loco.battery_level()
    except RuntimeError as exc:
        assert "does not report battery level" in str(exc)
        return
    raise AssertionError("battery level was fabricated for a transport without one")


def test_stop_repeats_zero_so_one_best_effort_sample_cannot_lose_the_stop():
    loco, implementation, _captured = _loco()
    loco.connect()
    loco.stop()
    assert implementation.calls[-3:] == ["stop", "stop", "stop"]


def test_shutdown_stops_before_releasing_ros():
    loco, implementation, _captured = _loco()
    loco.connect()
    loco.shutdown()
    assert implementation.calls[-4:] == ["stop", "stop", "stop", "shutdown"]
    loco.shutdown()  # idempotent


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_locomotion: {len(tests)}/{len(tests)} passed")
