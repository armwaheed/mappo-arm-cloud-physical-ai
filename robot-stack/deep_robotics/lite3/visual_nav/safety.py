# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed battery and motor-temperature monitoring for the Lite3 Venture.

The public ``Lite3_ROS`` bridge publishes odometry, IMU, and joint positions, but drops
the battery field present in its UDP ``RobotState`` and exposes no motor temperatures.
Rather than silently deleting the Go2 safety gate, this monitor requires two companion
ROS topics for a live run:

* ``sensor_msgs/BatteryState`` (``percentage``), and
* ``std_msgs/Float64MultiArray`` (one Celsius value per motor).

Topic names are configurable.  Until the two Venture installations publish them, live
navigation refuses to start; dry camera/perception work remains available.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

MOTOR_TEMP_WARN_C = 55.0
MOTOR_TEMP_ABORT_C = 70.0
BATTERY_SOC_ABORT_PCT = 20.0
HEALTH_STALE_S = 2.0
LITE3_MOTOR_COUNT = 12


@dataclass(frozen=True)
class Health:
    """One complete health sample in the shape consumed by visual navigation."""

    max_motor_temp_c: float
    hottest_motor: int
    battery_soc_pct: float
    sample_time: float

    @property
    def age(self) -> float:
        return time.monotonic() - self.sample_time


class Lite3HealthMonitor:
    """Combine standard ROS battery and motor-temperature topics into one guard."""

    def __init__(self, *, battery_topic: str | None = "/battery_state",
                 temperature_topic: str | None = "/motor_temperatures",
                 required: bool = True, motor_temp_warn_c: float = MOTOR_TEMP_WARN_C,
                 motor_temp_abort_c: float = MOTOR_TEMP_ABORT_C,
                 battery_abort_pct: float = BATTERY_SOC_ABORT_PCT,
                 stale_s: float = HEALTH_STALE_S,
                 motor_count: int = LITE3_MOTOR_COUNT,
                 clock: Callable[[], float] = time.monotonic) -> None:
        limits = {
            "motor_temp_warn_c": motor_temp_warn_c,
            "motor_temp_abort_c": motor_temp_abort_c,
            "battery_abort_pct": battery_abort_pct,
            "stale_s": stale_s,
        }
        invalid = [name for name, value in limits.items()
                   if not math.isfinite(value) or value < 0.0]
        if invalid:
            raise ValueError(f"health limits must be finite and non-negative: {invalid}")
        if motor_temp_warn_c >= motor_temp_abort_c:
            raise ValueError("motor temperature warning must be below the abort limit")
        if battery_abort_pct > 100.0:
            raise ValueError("battery abort percentage cannot exceed 100")
        if stale_s == 0.0:
            raise ValueError("health stale timeout must be positive")
        if not isinstance(motor_count, int) or isinstance(motor_count, bool) \
                or motor_count <= 0:
            raise ValueError("motor count must be a positive integer")
        self._battery_topic = battery_topic
        self._temperature_topic = temperature_topic
        self._required = required
        self._motor_temp_warn_c = motor_temp_warn_c
        self._motor_temp_abort_c = motor_temp_abort_c
        self._battery_abort_pct = battery_abort_pct
        self._stale_s = stale_s
        self._motor_count = motor_count
        self._clock = clock

        self._lock = threading.Lock()
        self._battery: tuple[float, float] | None = None
        self._temperatures: tuple[tuple[float, ...], float] | None = None
        self._node = None
        self._spin_thread: threading.Thread | None = None
        self._spinning = False
        self._we_inited_ros = False

    def start(self, wait_s: float = 3.0) -> None:
        """Subscribe and wait briefly for a complete sample when health is required."""
        if self._node is not None:
            raise RuntimeError("Lite3 health monitor is already running")
        if self._battery_topic is None and self._temperature_topic is None:
            return
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import BatteryState
        from std_msgs.msg import Float64MultiArray

        if not rclpy.ok():
            rclpy.init()
            self._we_inited_ros = True
        self._node = Node("mappo_lite3_health")
        if self._battery_topic is not None:
            self._node.create_subscription(
                BatteryState, self._battery_topic, self._on_battery, 10)
        if self._temperature_topic is not None:
            self._node.create_subscription(
                Float64MultiArray, self._temperature_topic, self._on_temperatures, 10)
        self._spinning = True
        self._spin_thread = threading.Thread(
            target=self._spin, name="lite3-health-spin", daemon=True)
        self._spin_thread.start()

        if self._required:
            deadline = self._clock() + wait_s
            while self.latest() is None and self._clock() < deadline:
                time.sleep(0.02)

    def _spin(self) -> None:
        import rclpy

        while self._spinning and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def stop(self) -> None:
        self._spinning = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            self._spin_thread = None
        if self._node is not None:
            with contextlib.suppress(Exception):
                self._node.destroy_node()
            self._node = None
        if self._we_inited_ros:
            import rclpy

            with contextlib.suppress(Exception):
                if rclpy.ok():
                    rclpy.shutdown()
            self._we_inited_ros = False

    def _on_battery(self, msg) -> None:
        self.update_battery(float(msg.percentage))

    def _on_temperatures(self, msg) -> None:
        self.update_temperatures(msg.data)

    def update_battery(self, percentage: float) -> None:
        """Accept ROS's usual 0..1 fraction, or an explicit 0..100 percentage."""
        if not math.isfinite(percentage) or percentage < 0.0:
            return
        percent = percentage * 100.0 if percentage <= 1.0 else percentage
        if percent > 100.0:
            return
        with self._lock:
            self._battery = (percent, self._clock())

    def update_temperatures(self, temperatures_c: Sequence[float]) -> None:
        values = tuple(float(value) for value in temperatures_c)
        if len(values) != self._motor_count \
                or any(not math.isfinite(value) for value in values):
            return
        with self._lock:
            self._temperatures = (values, self._clock())

    def latest(self) -> Health | None:
        with self._lock:
            battery = self._battery
            temperatures = self._temperatures
        if battery is None or temperatures is None:
            return None
        values, temperature_time = temperatures
        hottest = max(range(len(values)), key=values.__getitem__)
        return Health(
            max_motor_temp_c=values[hottest],
            hottest_motor=hottest,
            battery_soc_pct=battery[0],
            # A combined sample is only as fresh as its older half.
            sample_time=min(battery[1], temperature_time),
        )

    def missing_reason(self) -> str | None:
        with self._lock:
            missing = []
            if self._battery is None:
                missing.append(f"battery on {self._battery_topic!r}")
            if self._temperatures is None:
                missing.append(f"motor temperatures on {self._temperature_topic!r}")
        return None if not missing else "no " + " or ".join(missing)

    def abort_reason(self) -> str | None:
        health = self.latest()
        if health is None:
            return self.missing_reason() if self._required else None
        age = self._clock() - health.sample_time
        if age > self._stale_s:
            return f"Lite3 health stale by {age:.1f}s"
        if health.max_motor_temp_c >= self._motor_temp_abort_c:
            return (f"motor {health.hottest_motor} at {health.max_motor_temp_c:.0f}C "
                    f"(limit {self._motor_temp_abort_c:.0f}C)")
        if health.battery_soc_pct <= self._battery_abort_pct:
            return f"battery {health.battery_soc_pct:.0f}%"
        return None

    def warning_reason(self) -> str | None:
        health = self.latest()
        if health is None or health.max_motor_temp_c < self._motor_temp_warn_c:
            return None
        return (f"motor {health.hottest_motor} at {health.max_motor_temp_c:.0f}C is "
                f"past the {self._motor_temp_warn_c:.0f}C warning mark "
                f"(abort at {self._motor_temp_abort_c:.0f}C)")
