# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lite3-specific bindings for the shared visual-navigation and calibration loops."""

from __future__ import annotations

import json
import math
from pathlib import Path

from deep_robotics.lite3.locomotion.lite3_locomotion import Lite3Locomotion
from deep_robotics.lite3.locomotion.lite3_udp_locomotion import (
    DEFAULT_COMMAND_PORT,
    DEFAULT_MOTION_HOST,
    DEFAULT_STATE_PORT,
    udp_locomotion_factory,
)
from deep_robotics.lite3.visual_nav.camera import (
    Lite3Camera,
    camera_source_kind,
    parse_camera_source,
)
from deep_robotics.lite3.visual_nav.safety import Lite3HealthMonitor

#: Ceiling on a single run whose motor temperatures are unmonitored. A 3x3 m booth run is
#: about twenty seconds; two minutes is generous for one. This bounds one run and nothing
#: more -- the real thermal hazard is repeated runs, which no flag here can see, so the
#: pre-flight says so out loud rather than implying the ceiling is protection.
MAX_UNMONITORED_RUN_S = 120.0


class Lite3Bindings:
    """Construct the Lite3 seams without copying the navigator's safety/run loop."""

    platform_name = "deep-robotics-lite3-venture"
    rest_when_blocked = False
    initially_standing = True
    default_calibration_output = "lite3_front_camera.json"
    spin_rate_help = ("commanded yaw rate in rad/s; required for a Lite3 spin because "
                      "the Go2 default is not a measurement of this platform")

    def __init__(self) -> None:
        # create_health_monitor reads battery through whatever create_locomotion built,
        # because only one process can hold the robot's single state port.
        self._locomotion = None

    def add_navigation_arguments(self, parser, envelope) -> None:
        parser.add_argument(
            "--camera-source", required=True,
            help="V4L2 index, RTSP URI, or GStreamer pipeline for the forward RGB camera",
        )
        parser.add_argument("--camera-gstreamer", action="store_true",
                            help="open --camera-source with OpenCV's GStreamer backend")
        self._add_ros_arguments(parser)
        calibration = parser.add_argument_group("Lite3 measured calibration")
        calibration.add_argument(
            "--gait-floor", type=float, default=None, metavar="M_S",
            help="lowest command measured to sustain this Lite3's gait; required live",
        )
        calibration.add_argument(
            "--actuator-gain", type=float, default=None, metavar="RATIO",
            help="measured achieved/commanded speed at this envelope; required live",
        )
        calibration.add_argument(
            "--operator-ready", action="store_true",
            help="confirm the Lite3 is standing in vendor high-level navigation mode "
                 "with emergency stop held",
        )
        health = parser.add_argument_group("Lite3 health feeds")
        health.add_argument("--battery-topic", default="/battery_state")
        health.add_argument("--temperature-topic", default="/motor_temperatures")
        health.add_argument("--motor-temp-warn", type=float, default=55.0)
        health.add_argument("--motor-temp-abort", type=float, default=70.0)
        health.add_argument("--battery-abort", type=float, default=20.0)
        health.add_argument(
            "--accept-no-motor-temperatures", action="store_true",
            help="run without motor-temperature monitoring. The high-level interface does "
                 "not carry temperatures, so this is an explicit operator decision: it is "
                 "recorded in telemetry, warned about every tick, and caps --max-seconds "
                 f"at {MAX_UNMONITORED_RUN_S:.0f}s. Battery and staleness stay enforced.",
        )
        # No Lite3 radius is defensible before measuring the loaded body. ``None`` lets
        # the live pre-flight distinguish an omitted value from a deliberate 0.40 m.
        parser.set_defaults(robot_radius=None)

    def add_calibration_arguments(self, parser, _spin) -> None:
        parser.add_argument(
            "--camera-source", required=True,
            help="V4L2 index, RTSP URI, or GStreamer pipeline for the forward RGB camera",
        )
        parser.add_argument("--camera-gstreamer", action="store_true")
        self._add_ros_arguments(parser)
        parser.add_argument("--operator-ready", action="store_true")
        parser.add_argument("--battery-topic", default="/battery_state")
        parser.add_argument("--temperature-topic", default="/motor_temperatures")
        parser.add_argument("--motor-temp-warn", type=float, default=55.0)
        parser.add_argument("--motor-temp-abort", type=float, default=70.0)
        parser.add_argument("--battery-abort", type=float, default=20.0)
        parser.add_argument("--accept-no-motor-temperatures", action="store_true")
        # The Go2's measured 0.8 rad/s default is not a Lite3 calibration. Make the
        # operator supply a rate already shown to clear this robot's yaw deadband.
        parser.set_defaults(spin_rate=None)

    @staticmethod
    def _add_ros_arguments(parser) -> None:
        """Locomotion transport selection.

        ``udp`` is the default because it is the shorter path to the same vendor
        interface: it sends the frames ``Lite3_ROS``'s ``jetson2motion`` sends, to the same
        motion host, and needs no ROS 2 runtime on a Venture that may not have a
        provisioned perception host to put one on. ``ros2`` remains available for a unit
        that does ship the vendor bridge.
        """
        transport = parser.add_argument_group("Lite3 locomotion transport")
        transport.add_argument(
            "--locomotion-transport", choices=("udp", "ros2"), default="udp",
            help="udp: the vendor high-level UDP interface directly, no ROS 2 (default); "
                 "ros2: the Lite3_ROS bridge topics",
        )
        transport.add_argument("--motion-host", default=DEFAULT_MOTION_HOST,
                               help="Lite3 motion host address, for --locomotion-transport udp")
        transport.add_argument("--command-port", type=int, default=DEFAULT_COMMAND_PORT,
                               help="motion host command port")
        transport.add_argument("--state-port", type=int, default=DEFAULT_STATE_PORT,
                               help="local port the motion host streams state to; it must "
                                    "match 'ip'/'target_port' in ~/jy_exe/conf/network.toml")
        transport.add_argument("--cmd-vel-topic", default="/cmd_vel",
                               help="Lite3_ROS geometry_msgs/Twist command topic (ros2 only)")
        transport.add_argument("--odom-topic", default="/leg_odom2",
                               help="Lite3_ROS nav_msgs/Odometry topic (ros2 only)")

    def create_locomotion(self, args):
        arguments = {
            "cmd_vel_topic": args.cmd_vel_topic,
            "odom_topic": args.odom_topic,
            "operator_ready": args.operator_ready,
        }
        if getattr(args, "locomotion_transport", "udp") == "udp":
            arguments["implementation_factory"] = udp_locomotion_factory(
                motion_host=args.motion_host,
                command_port=args.command_port,
                state_port=args.state_port,
            )
        self._locomotion = Lite3Locomotion(**arguments)
        return self._locomotion

    def create_health_monitor(self, args, *, live: bool):
        battery_source = None
        if getattr(args, "locomotion_transport", "udp") == "udp":
            # Late-bound: the navigator may build the monitor before the locomotion, and
            # the link is not up until connect(). The poller treats a raise as "no
            # sample", so the staleness gate covers the gap rather than a fabricated one.
            def battery_source():
                if self._locomotion is None:
                    return None
                return self._locomotion.battery_level()

        return Lite3HealthMonitor(
            battery_topic=args.battery_topic,
            temperature_topic=args.temperature_topic,
            required=live,
            motor_temp_warn_c=args.motor_temp_warn,
            motor_temp_abort_c=args.motor_temp_abort,
            battery_abort_pct=args.battery_abort,
            battery_source=battery_source,
            accept_missing_temperatures=args.accept_no_motor_temperatures,
        )

    def start(self, _args) -> None:
        return

    def create_camera(self, args, stamp_fn):
        return Lite3Camera(
            parse_camera_source(args.camera_source),
            gstreamer=args.camera_gstreamer,
            stamp_fn=stamp_fn,
        )

    def preflight_navigation(self, args, _config, health) -> None:
        measurements = {
            "--gait-floor": args.gait_floor,
            "--actuator-gain": args.actuator_gain,
            "--robot-radius": args.robot_radius,
        }
        invalid = [name for name, value in measurements.items()
                   if value is not None and not self._positive_finite(value)]
        if invalid:
            raise SystemExit(
                "[lite3] REFUSING TO RUN: measurements must be finite and positive: "
                + ", ".join(invalid)
            )
        if args.live:
            missing = []
            if args.calibration is None:
                missing.append("--calibration from this Lite3 camera")
            if args.gait_floor is None:
                missing.append("--gait-floor measured on this Lite3")
            if args.actuator_gain is None:
                missing.append("--actuator-gain measured at this envelope")
            if args.robot_radius is None:
                missing.append("--robot-radius measured for the loaded Lite3")
            if not args.operator_ready:
                missing.append("--operator-ready after STANDING + navigation mode")
            if missing:
                raise SystemExit("[lite3] REFUSING TO WALK: missing " + ", ".join(missing))
            if args.accept_no_motor_temperatures:
                if not self._positive_finite(args.max_seconds) \
                        or args.max_seconds > MAX_UNMONITORED_RUN_S:
                    raise SystemExit(
                        "[lite3] REFUSING TO WALK: --accept-no-motor-temperatures needs "
                        f"--max-seconds set to {MAX_UNMONITORED_RUN_S:.0f}s or less; an "
                        "unbounded run with no thermal feed is not an informed decision"
                    )
                print("[lite3] MOTOR TEMPERATURES ARE NOT MONITORED on this run.")
                print(f"[lite3]   this run is bounded to {args.max_seconds:.0f}s, but "
                      "nothing bounds the NEXT one: heat builds across back-to-back")
                print("[lite3]   runs and no software here can see it. Let the robot "
                      "cool between runs, keep the")
                print("[lite3]   emergency stop in hand, and stop if a motor smells hot "
                      "or the gait changes.")
        self._report_health(health, live=args.live, prefix="visual_nav")

    def preflight_calibration(self, args, health) -> None:
        if not args.spin:
            return
        if not args.operator_ready:
            raise SystemExit(
                "[calibrate] REFUSING TO MOVE: put the Lite3 in STANDING + vendor "
                "high-level navigation mode, keep the emergency stop in hand, then "
                "pass --operator-ready"
            )
        if not self._positive_finite(args.spin_rate):
            raise SystemExit(
                "[calibrate] REFUSING TO MOVE: pass --spin-rate measured above this "
                "Lite3's yaw deadband"
            )
        self._report_health(health, live=True, prefix="calibrate")

    @staticmethod
    def _positive_finite(value) -> bool:
        return value is not None and math.isfinite(value) and value > 0.0

    @staticmethod
    def _report_health(health, *, live: bool, prefix: str) -> None:
        unhealthy = health.abort_reason()
        if unhealthy is not None:
            raise SystemExit(f"[{prefix}] REFUSING TO MOVE: {unhealthy}")
        sample = health.latest()
        if sample is None:
            if live:
                raise SystemExit(f"[{prefix}] REFUSING TO MOVE: Lite3 health unavailable")
            print(f"[{prefix}] Lite3 health feeds unavailable (dry run only)")
            return
        print(f"[{prefix}] motors {sample.max_motor_temp_c:.0f}C, "
              f"battery {sample.battery_soc_pct:.0f}%")
        warning = health.warning_reason()
        if warning is not None:
            print(f"[{prefix}] WARNING: {warning}")

    def validate_camera_calibration(self, args) -> None:
        if args.calibration is None:
            return
        action = "WALK" if args.live else "USE CAMERA CALIBRATION"
        path = Path(args.calibration)
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"[lite3] REFUSING TO {action}: cannot read calibration {path}: {exc}"
            ) from None
        if not isinstance(data, dict):
            raise SystemExit(
                f"[lite3] REFUSING TO {action}: {path} must contain a JSON object"
            )
        if data.get("platform") != self.platform_name:
            raise SystemExit(
                f"[lite3] REFUSING TO {action}: {path} was not produced by the "
                f"Lite3 calibration runner (platform={data.get('platform')!r})"
            )

    def prepare_motion(self, _args, loco) -> None:
        loco.prepare_motion()

    def stand_up(self, loco) -> None:
        loco.recover()
        loco.stand()

    def lie_down(self, loco) -> None:
        # No public ROS stand-down verb exists. Stop now; the operator returns to manual
        # and lowers the robot through the approved vendor interface after the run.
        loco.stop()

    def warn_if_below_gait_floor(self, max_vx: float, args) -> bool:
        if args.gait_floor is None or max_vx >= args.gait_floor:
            return False
        print("!" * 78)
        print(f"[lite3] TOP SPEED {max_vx:.3f} m/s IS BELOW THIS ROBOT'S MEASURED "
              f"GAIT FLOOR {args.gait_floor:.3f} m/s")
        print("    Raise the envelope/command scale or do not run; a sub-floor command")
        print("    can look like a tether or transport failure while reporting no fault.")
        print("!" * 78)
        return True

    @staticmethod
    def gait_floor(args) -> float | None:
        return args.gait_floor

    def robot_radius(self, args, default: float) -> float:
        return default if args.robot_radius is None else args.robot_radius

    def telemetry_config(self, args) -> dict:
        return {
            "platform": {
                "name": self.platform_name,
                "transport": getattr(args, "locomotion_transport", "udp"),
                "motion_host": getattr(args, "motion_host", None),
                "cmd_vel_topic": args.cmd_vel_topic,
                "odom_topic": args.odom_topic,
                "motor_temperatures_monitored": not args.accept_no_motor_temperatures,
                "camera_source_kind": camera_source_kind(
                    parse_camera_source(args.camera_source),
                    gstreamer=args.camera_gstreamer,
                ),
                "gait_floor_m_s": args.gait_floor,
                "actuator_gain": args.actuator_gain,
            }
        }

    def calibration_provenance(self, _args) -> dict:
        return {"platform": self.platform_name}

    def actuation_summary(self, top_speed: float, args) -> str:
        achieved = (None if args.actuator_gain is None
                    else top_speed * args.actuator_gain)
        if achieved is None:
            return (f"top commanded speed {top_speed:.3f} m/s; Lite3 actuator gain "
                    f"not supplied (dry/offline only)")
        return (f"top commanded speed {top_speed:.3f} m/s; measured gain "
                f"{args.actuator_gain:.3f} predicts {achieved:.3f} m/s achieved")

    def shutdown(self) -> None:
        return
