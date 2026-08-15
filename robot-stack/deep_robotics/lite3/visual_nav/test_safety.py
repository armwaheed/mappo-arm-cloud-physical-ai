#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 battery/temperature safety gate."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.visual_nav.safety import Lite3HealthMonitor


class _Clock:
    now = 100.0

    def __call__(self):
        return self.now


def _monitor(required=True):
    clock = _Clock()
    monitor = Lite3HealthMonitor(required=required, clock=clock)
    return monitor, clock


def test_live_health_fails_closed_when_either_feed_is_missing():
    monitor, _clock = _monitor()
    assert "battery" in monitor.abort_reason()
    monitor.update_battery(0.8)
    assert "motor temperatures" in monitor.abort_reason()
    monitor.update_temperatures([31.0] * 12)
    assert monitor.abort_reason() is None


def test_dry_runs_can_operate_without_the_unpublished_vendor_health_topics():
    monitor, _clock = _monitor(required=False)
    assert monitor.latest() is None
    assert monitor.abort_reason() is None


def test_battery_fraction_is_converted_to_percent_and_low_battery_aborts():
    monitor, _clock = _monitor()
    monitor.update_battery(0.19)
    monitor.update_temperatures([30.0] * 12)
    assert monitor.latest().battery_soc_pct == 19.0
    assert "battery 19%" in monitor.abort_reason()


def test_the_hottest_motor_is_reported_and_the_limit_aborts():
    monitor, _clock = _monitor()
    temperatures = [30.0] * 12
    temperatures[7] = 71.0
    monitor.update_battery(75.0)
    monitor.update_temperatures(temperatures)
    health = monitor.latest()
    assert health.hottest_motor == 7 and health.max_motor_temp_c == 71.0
    assert "motor 7" in monitor.abort_reason()


def test_a_stale_half_makes_the_combined_sample_stale():
    monitor, clock = _monitor()
    monitor.update_battery(80.0)
    clock.now += 1.5
    monitor.update_temperatures([32.0] * 12)
    clock.now += 0.6
    assert "stale" in monitor.abort_reason()


def test_warning_precedes_abort_and_bad_samples_are_ignored():
    monitor, _clock = _monitor()
    monitor.update_battery(float("nan"))
    monitor.update_temperatures([])
    assert monitor.latest() is None
    monitor.update_battery(80.0)
    monitor.update_temperatures([56.0, *([33.0] * 11)])
    assert "warning mark" in monitor.warning_reason()
    assert monitor.abort_reason() is None


def test_invalid_health_limits_are_rejected_at_configuration_time():
    for kwargs in ({"stale_s": 0.0}, {"battery_abort_pct": 101.0},
                   {"motor_temp_warn_c": 70.0, "motor_temp_abort_c": 70.0},
                   {"motor_temp_abort_c": float("nan")}, {"motor_count": 0}):
        try:
            Lite3HealthMonitor(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid health limits were accepted: {kwargs}")


def test_an_incomplete_motor_array_is_not_treated_as_complete_health():
    monitor, _clock = _monitor()
    monitor.update_battery(80.0)
    monitor.update_temperatures([32.0] * 11)
    assert "motor temperatures" in monitor.abort_reason()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_safety: {len(tests)}/{len(tests)} passed")
