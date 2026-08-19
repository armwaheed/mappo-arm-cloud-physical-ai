#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 battery/temperature safety gate."""

from __future__ import annotations

import math
import sys
import time
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


def _override_monitor(required=True, battery_source=None):
    clock = _Clock()
    monitor = Lite3HealthMonitor(required=required, accept_missing_temperatures=True,
                                 battery_source=battery_source, clock=clock)
    return monitor, clock


def test_a_percent_scaled_one_percent_battery_is_not_read_as_a_full_one():
    """The auto guess fails open at exactly the reading that matters most.

    ROS publishes 0..1 or 0..100 depending on the driver, so "auto" has to guess, and a
    genuine 1% becomes 100%. The UDP stream's unit is known, so it says so.
    """
    monitor, _clock = _monitor()
    monitor.update_temperatures([30.0] * 12)
    monitor.update_battery(1.0)  # auto: indistinguishable from a full battery
    assert monitor.latest().battery_soc_pct == 100.0
    assert monitor.abort_reason() is None
    monitor.update_battery(1.0, scale="percent")
    assert monitor.latest().battery_soc_pct == 1.0
    assert "battery 1%" in monitor.abort_reason()


def test_an_unknown_battery_scale_is_refused_rather_than_guessed():
    monitor, _clock = _monitor()
    try:
        monitor.update_battery(50.0, scale="per-cent")
    except ValueError as error:
        assert "per-cent" in str(error)
    else:
        raise AssertionError("a misspelled scale silently fell back to guessing")


def test_accepting_missing_temperatures_still_enforces_battery_and_staleness():
    """The override removes one check, not the gate."""
    monitor, clock = _override_monitor()
    monitor.update_battery(80.0, scale="percent")
    assert monitor.abort_reason() is None
    health = monitor.latest()
    assert not math.isfinite(health.max_motor_temp_c)
    assert health.hottest_motor == -1

    monitor.update_battery(15.0, scale="percent")
    assert "battery 15%" in monitor.abort_reason()

    monitor.update_battery(80.0, scale="percent")
    clock.now += 10.0
    assert "stale" in monitor.abort_reason()


def test_an_unmonitored_run_says_so_on_every_tick():
    """An override that stops being visible stops being a decision."""
    monitor, _clock = _override_monitor()
    monitor.update_battery(80.0, scale="percent")
    for _ in range(3):
        warning = monitor.warning_reason()
        assert warning is not None
        assert "NOT monitored" in warning


def test_a_hot_motor_still_aborts_when_temperatures_are_present_under_the_override():
    monitor, _clock = _override_monitor()
    monitor.update_battery(80.0, scale="percent")
    monitor.update_temperatures([30.0] * 11 + [95.0])
    assert "95C" in monitor.abort_reason()


def test_without_the_override_missing_temperatures_still_refuse_a_live_run():
    monitor, _clock = _monitor()
    monitor.update_battery(80.0, scale="percent")
    assert "motor temperatures" in monitor.abort_reason()


def test_the_battery_poller_reads_the_udp_link_and_survives_it_being_down():
    """A link that is not up yet must read as no sample, not as a fabricated one."""
    state = {"up": False}

    def source():
        if not state["up"]:
            raise RuntimeError("no Lite3 state frame has arrived yet")
        return 73.0

    monitor = Lite3HealthMonitor(required=False, accept_missing_temperatures=True,
                                 battery_source=source)
    monitor.start(wait_s=0.0)
    try:
        time.sleep(0.25)
        assert monitor.latest() is None  # the link was down; nothing was invented
        state["up"] = True
        deadline = time.monotonic() + 2.0
        while monitor.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        health = monitor.latest()
        assert health is not None
        assert abs(health.battery_soc_pct - 73.0) < 1e-9
    finally:
        monitor.stop()


def test_the_poller_path_imports_no_ros():
    """The whole point of the UDP source is that a live Lite3 run needs no ROS runtime."""
    import sys as _sys

    assert "rclpy" not in _sys.modules
    monitor = Lite3HealthMonitor(required=False, accept_missing_temperatures=True,
                                 battery_source=lambda: 55.0)
    monitor.start(wait_s=0.0)
    try:
        deadline = time.monotonic() + 2.0
        while monitor.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert monitor.latest() is not None
    finally:
        monitor.stop()
    assert "rclpy" not in _sys.modules


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_safety: {len(tests)}/{len(tests)} passed")
