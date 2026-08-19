# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed battery and motor-temperature monitoring for the Lite3 Venture.

Battery and motor temperatures reach this monitor from one of two places.

**From the vendor UDP state stream** (``battery_source``), which is the default now that
locomotion speaks UDP directly. ``RobotState`` carries ``battery_level``; the public
``Lite3_ROS`` bridge drops that field, which is why the ROS path needed a companion
publisher for something the robot was already reporting. This path imports no ROS.

**From two companion ROS topics** — ``sensor_msgs/BatteryState`` and
``std_msgs/Float64MultiArray`` — for a unit running the vendor bridge.

Neither path supplies motor temperatures: the high-level interface does not carry them in
any form, and taking low-level ``Lite3_MotionSDK`` control merely to read them would
remove the vendor controller that keeps the robot upright. ``accept_missing_temperatures``
therefore exists as an explicit, recorded operator decision rather than a silent default.
It is not a way to make the gate pass: battery and staleness stay enforced, the run stays
time-bounded by its caller, and :meth:`warning_reason` reports the unmonitored state on
every tick for as long as it is in force.
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
                 battery_source: Callable[[], float] | None = None,
                 accept_missing_temperatures: bool = False,
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
        self._battery_source = battery_source
        self._accept_missing_temperatures = accept_missing_temperatures
        self._clock = clock

        self._lock = threading.Lock()
        self._battery: tuple[float, float] | None = None
        self._temperatures: tuple[tuple[float, ...], float] | None = None
        self._node = None
        self._spin_thread: threading.Thread | None = None
        self._spinning = False
        self._we_inited_ros = False
        self._polling = False
        self._poll_thread: threading.Thread | None = None

    def start(self, wait_s: float = 3.0) -> None:
        """Subscribe and wait briefly for a complete sample when health is required."""
        if self._node is not None or self._poll_thread is not None:
            raise RuntimeError("Lite3 health monitor is already running")
        if self._battery_source is not None:
            self._start_polling(wait_s)
            return
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

    def _start_polling(self, wait_s: float) -> None:
        """Read battery from the locomotion link's own state stream. Imports no ROS."""
        self._polling = True
        self._poll_thread = threading.Thread(target=self._poll, name="lite3-health-poll",
                                             daemon=True)
        self._poll_thread.start()
        if self._required:
            deadline = self._clock() + wait_s
            while self.latest() is None and self._clock() < deadline:
                time.sleep(0.02)

    def _poll(self) -> None:
        while self._polling:
            try:
                value = self._battery_source()
            except Exception:
                # The link is not up yet, or has dropped. Reporting nothing is correct:
                # the sample then goes stale and the staleness gate does its job.
                value = None
            if value is not None:
                # The stream reports a percentage, so say so. Left on "auto", a genuine
                # 1% reading would be read as a full battery and fail open.
                self.update_battery(float(value), scale="percent")
            time.sleep(0.1)

    def _spin(self) -> None:
        import rclpy

        while self._spinning and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)

    def stop(self) -> None:
        self._polling = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
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

    def update_battery(self, percentage: float, *, scale: str = "auto") -> None:
        """Record a battery reading.

        ``scale`` is ``"auto"`` for ROS, whose ``BatteryState.percentage`` is documented
        as a 0..1 fraction but is published as 0..100 by some drivers. Guessing is only
        safe where the caller genuinely does not know: a real 1% reading guessed as a
        fraction becomes 100%, which fails open at exactly the moment it matters. Callers
        that know the unit should say so.
        """
        if scale not in ("auto", "fraction", "percent"):
            raise ValueError(f"unknown battery scale {scale!r}")
        if not math.isfinite(percentage) or percentage < 0.0:
            return
        if scale == "percent":
            percent = percentage
        elif scale == "fraction":
            percent = percentage * 100.0
        else:
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
        if battery is None:
            return None
        if temperatures is None:
            if not self._accept_missing_temperatures:
                return None
            # NaN cannot exceed the abort limit, so the temperature comparison is skipped
            # rather than passed with a fabricated value.
            return Health(max_motor_temp_c=math.nan, hottest_motor=-1,
                          battery_soc_pct=battery[0], sample_time=battery[1])
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
            if self._temperatures is None and not self._accept_missing_temperatures:
                missing.append(f"motor temperatures on {self._temperature_topic!r}")
        return None if not missing else "no " + " or ".join(missing)

    def abort_reason(self) -> str | None:
        health = self.latest()
        if health is None:
            return self.missing_reason() if self._required else None
        age = self._clock() - health.sample_time
        if age > self._stale_s:
            return f"Lite3 health stale by {age:.1f}s"
        if math.isfinite(health.max_motor_temp_c) \
                and health.max_motor_temp_c >= self._motor_temp_abort_c:
            return (f"motor {health.hottest_motor} at {health.max_motor_temp_c:.0f}C "
                    f"(limit {self._motor_temp_abort_c:.0f}C)")
        if health.battery_soc_pct <= self._battery_abort_pct:
            return f"battery {health.battery_soc_pct:.0f}%"
        return None

    def warning_reason(self) -> str | None:
        health = self.latest()
        if health is None:
            return None
        if not math.isfinite(health.max_motor_temp_c):
            # Reported every tick, deliberately. An override that stops being visible
            # stops being a decision and becomes the new default.
            return ("motor temperatures are NOT monitored on this run "
                    "(--accept-no-motor-temperatures); keep runs short and the "
                    "emergency stop in hand")
        if health.max_motor_temp_c < self._motor_temp_warn_c:
            return None
        return (f"motor {health.hottest_motor} at {health.max_motor_temp_c:.0f}C is "
                f"past the {self._motor_temp_warn_c:.0f}C warning mark "
                f"(abort at {self._motor_temp_abort_c:.0f}C)")
