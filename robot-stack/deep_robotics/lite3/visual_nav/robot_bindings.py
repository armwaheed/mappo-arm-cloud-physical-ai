# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lite3-specific bindings for the shared visual-navigation and calibration loops."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from deep_robotics.lite3.locomotion.lite3_axis_locomotion import (
    AXIS_PROFILE_SCHEMA,
    AXIS_RATE_HZ,
    COMMAND_TTL_S,
    HEARTBEAT_HZ,
    AxisProfile,
    AxisProfileError,
    axis_locomotion_factory,
)
from deep_robotics.lite3.locomotion.lite3_axis_udp import DEFAULT_LOCAL_PORT
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

#: The velocity-envelope flags the shared navigator offers, mapped to their ``args``
#: attribute. Their parser defaults come from ``avoidance.Limits``, whose three velocity
#: fields are the UNITREE GO2's arm-fitted profile -- 0.35 m/s forward, 0.20 m/s strafe,
#: 0.70 rad/s yaw, two of them measured gait floors on that robot. Nothing about them was
#: measured on a Lite3, and issue #13 still owns that measurement, so this binding blanks
#: them the way it blanks ``--robot-radius``: a live run refuses, a dry run says whose
#: numbers it is standing on. They are captured from the parser rather than imported,
#: because a copy here would pin whatever the Go2's numbers were the day it was written.
ENVELOPE_ARGUMENTS = {
    "--max-vx": ("max_vx", "m/s"),
    "--max-vy": ("max_vy", "m/s"),
    "--max-wz": ("max_wz", "rad/s"),
}

#: Just the ``args`` attributes, in flag order.
ENVELOPE_NAMES = tuple(name for name, _unit in ENVELOPE_ARGUMENTS.values())


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
        self._axis_profile = None
        #: What ``--max-vx`` / ``--max-vy`` / ``--max-wz`` defaulted to before this
        #: binding blanked them, i.e. the Go2's numbers, kept so a dry run can fall back
        #: to exactly today's behaviour while naming where the values came from.
        self._inherited_envelope: dict[str, float] = {}
        #: Which of those a run actually fell back on. Empty until ``preflight_navigation``
        #: has resolved them, and empty forever on a run that stated all three.
        self._envelope_inherited: frozenset[str] = frozenset()

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
        # Same argument, one flag along, and the reason this binding is where it happens:
        # ``Limits``' velocity defaults are correct FOR THE GO2 and are the Go2 navigator's
        # to keep. Reading them here before blanking them is what turns a silent
        # inheritance into a stated one. See ENVELOPE_ARGUMENTS.
        self._inherited_envelope = {name: parser.get_default(name)
                                    for name in ENVELOPE_NAMES}
        parser.set_defaults(**dict.fromkeys(ENVELOPE_NAMES))

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
            "--locomotion-transport", choices=("udp", "axis", "ros2"), default="udp",
            help="udp: the vendor high-level UDP interface directly, no ROS 2 (default); "
                 "axis: profile-gated moving-mode simple axes; ros2: the Lite3_ROS bridge topics",
        )
        transport.add_argument("--motion-host", default=DEFAULT_MOTION_HOST,
                               help="Lite3 motion host address, for --locomotion-transport udp")
        transport.add_argument("--command-port", type=int, default=DEFAULT_COMMAND_PORT,
                               help="motion host command port")
        transport.add_argument("--state-port", type=int, default=DEFAULT_STATE_PORT,
                               help="local port the motion host streams state to; it must "
                                    "match 'ip'/'target_port' in ~/jy_exe/conf/network.toml")
        transport.add_argument("--state-bind", default="0.0.0.0",
                               help="local address for state telemetry; use 127.0.0.1 only "
                                       "after verified host-local telemetry loopback")
        axis = transport.add_argument_group("Lite3 simple-axis transport")
        axis.add_argument(
            "--axis-profile", type=Path,
            help=f"versioned local profile ({AXIS_PROFILE_SCHEMA}) containing only "
                 "physically evidenced axis primitives; required for live axis runs",
        )
        axis.add_argument(
            "--axis-source-address",
            help="source address for simple-axis UDP; omit for the kernel-selected local address",
        )
        axis.add_argument("--axis-local-port", type=int, default=DEFAULT_LOCAL_PORT,
                          help="simple-axis sender source port (vendor reference: 20001)")
        axis.add_argument("--axis-rate-hz", type=float, default=AXIS_RATE_HZ,
                          help="simple-axis stream rate; vendor minimum is 20 Hz")
        axis.add_argument("--axis-heartbeat-hz", type=float, default=HEARTBEAT_HZ,
                          help="simple-axis heartbeat rate; vendor minimum is 2 Hz")
        axis.add_argument("--axis-command-ttl", type=float, default=COMMAND_TTL_S,
                          help="maximum age of a policy command before all axes zero")
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
        transport = getattr(args, "locomotion_transport", "udp")
        self._axis_profile = None
        if transport == "udp":
            arguments["implementation_factory"] = udp_locomotion_factory(
                motion_host=args.motion_host,
                command_port=args.command_port,
                state_port=args.state_port,
                bind=args.state_bind,
            )
        elif transport == "axis":
            profile = self._load_axis_profile(args)
            self._axis_profile = profile
            arguments["implementation_factory"] = axis_locomotion_factory(
                axis_profile=profile,
                axis_source_address=args.axis_source_address,
                axis_local_port=args.axis_local_port,
                axis_rate_hz=args.axis_rate_hz,
                heartbeat_hz=args.axis_heartbeat_hz,
                command_ttl_s=args.axis_command_ttl,
                motion_host=args.motion_host,
                command_port=args.command_port,
                state_port=args.state_port,
                bind=args.state_bind,
            )
        self._locomotion = Lite3Locomotion(**arguments)
        return self._locomotion

    def create_health_monitor(self, args, *, live: bool):
        battery_source = None
        if getattr(args, "locomotion_transport", "udp") in ("udp", "axis"):
            # Late-bound: the navigator may build the monitor before the locomotion, and
            # the link is not up until connect(). The poller treats a raise as "no
            # sample", and battery_level() raises on a link that has GONE silent as well
            # as on one that never started -- see Lite3UdpLocomotion._require_fresh_state.
            # Without that second case the staleness gate covers nothing: the poller
            # re-stamps a frozen snapshot at 10 Hz, so HEALTH_STALE_S measures the age of
            # the stamp rather than the age of the frame and can never elapse.
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
        # The envelope is checked separately because ZERO is a legal value for it and is
        # not a legal measurement: ``--max-vy 0`` is how the deployment SOP disables the
        # strafe axis, and ``_validate_axis_profile_for_envelope`` reads it that way.
        invalid += [flag for flag, (name, _unit) in ENVELOPE_ARGUMENTS.items()
                    if getattr(args, name) is not None
                    and not self._non_negative_finite(getattr(args, name))]
        if invalid:
            raise SystemExit(
                "[lite3] REFUSING TO RUN: measurements must be finite and positive, and "
                "envelope ceilings finite and not negative: " + ", ".join(invalid)
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
            missing += self._missing_envelope(args)
            if not args.operator_ready:
                missing.append("--operator-ready after STANDING + navigation mode")
            if getattr(args, "locomotion_transport", "udp") == "axis" \
                    and args.axis_profile is None:
                missing.append("--axis-profile with physically evidenced primitives")
            # Raised BEFORE the axis checks below, which is a change of order and a
            # deliberate one. ``_validate_axis_profile_for_envelope`` compares a measured
            # primitive against ``--max-vx x --derate``; with the envelope unstated that
            # comparison has no right-hand side, and running it anyway would either
            # trip over ``None`` or -- worse -- answer using the Go2's number and print
            # a refusal that reads like a Lite3 verdict. Nothing is lost: every path
            # below this point ends in a refusal too.
            if missing:
                raise SystemExit("[lite3] REFUSING TO WALK: missing " + ", ".join(missing))
            if getattr(args, "locomotion_transport", "udp") == "axis":
                if self._axis_profile is None:
                    self._axis_profile = self._load_axis_profile(args)
                self._validate_axis_transport(args)
                self._validate_axis_profile_for_envelope(args)
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
        # After the live gate and never before it. Reaching this line means either the
        # operator stated the envelope or the run cannot move a leg, so on a live run
        # this resolves nothing and records that nothing was inherited.
        self._resolve_envelope(args)
        self._report_health(health, live=args.live, prefix="visual_nav")

    def _go2_envelope(self) -> dict:
        """The Go2 numbers this binding is refusing to let a Lite3 inherit silently.

        Primary source is what ``add_navigation_arguments`` read off the parser, because
        that is literally the default that was in force. The fallback covers a caller
        that builds the parser with one ``Lite3Bindings`` and pre-flights with another --
        the offline suite does exactly that -- where the capture is empty and the
        alternative is printing ``the Go2's None`` at an operator.

        Imported inside the method rather than at module scope. Every caller today puts
        the shared navigator on ``sys.path`` before importing this file, so a top-level
        import would work; it would also be a hard import-order dependency that
        ``ruff --fix`` is documented in ``AGENTS.md`` to break by hoisting, in a file that
        currently has none.
        """
        if self._inherited_envelope:
            return self._inherited_envelope
        from avoidance import GO2_MAX_VX_M_S, GO2_MAX_VY_M_S, GO2_MAX_WZ_RAD_S

        return {"max_vx": GO2_MAX_VX_M_S, "max_vy": GO2_MAX_VY_M_S,
                "max_wz": GO2_MAX_WZ_RAD_S}

    def _missing_envelope(self, args) -> list:
        """Envelope flags a live Lite3 run has to state, and why it cannot be defaulted.

        A ceiling is not a measurement, so this is not the same claim ``--gait-floor``
        makes. It is the claim that *whoever set it looked at this robot* -- because this
        number is the right-hand side of a safety gate. ``_validate_axis_profile_speeds``
        refuses a primitive whose ``measured_m_s`` exceeds ``--max-vx x --derate``, and
        the ``axis_primitive_probe`` measures that left-hand side carefully. Against a
        borrowed right-hand side that is not a gate, it is arithmetic.
        """
        inherited = self._go2_envelope()
        return [f"{flag} stated for this Lite3 (unset it inherits the Go2's "
                f"{inherited[name]:g} {unit})"
                for flag, (name, unit) in ENVELOPE_ARGUMENTS.items()
                if getattr(args, name) is None]

    def _resolve_envelope(self, args) -> None:
        """Put the Go2's numbers back for a run that cannot move, and say so.

        A dry run has to keep working: perception, planning, the shadow ladder and every
        offline suite go through here, and a navigator that refuses to plan is a navigator
        nobody runs before a live one. So the fallback is the SAME behaviour as before
        this gate existed -- and it is announced, which is the whole difference. The
        values are the ones this parser inherited, not a copy, so they cannot drift from
        the Go2's.
        """
        unset = tuple(name for name in ENVELOPE_NAMES if getattr(args, name) is None)
        self._envelope_inherited = frozenset(unset)
        if not unset:
            return
        inherited = self._go2_envelope()
        for name in unset:
            setattr(args, name, inherited[name])
        flags = ", ".join(flag for flag, (name, _unit) in ENVELOPE_ARGUMENTS.items()
                          if name in self._envelope_inherited)
        print(f"[lite3] ENVELOPE NOT STATED: {flags} fall back to the shared navigator's")
        print("[lite3]   defaults, which are the UNITREE GO2's arm-fitted profile "
              f"(vx<={inherited['max_vx']:g} m/s vy<={inherited['max_vy']:g} m/s "
              f"wz<={inherited['max_wz']:g} rad/s).")
        print("[lite3]   No Lite3 produced them; issue #13 still owns that measurement. "
              "A --live run")
        print("[lite3]   refuses without them. This one cannot move, so it continues.")

    @staticmethod
    def _non_negative_finite(value) -> bool:
        """Zero is a legal envelope entry -- it disables an axis -- and NaN is not."""
        return value is not None and math.isfinite(value) and value >= 0.0

    @staticmethod
    def _validate_axis_transport(args) -> None:
        if not math.isfinite(args.axis_rate_hz) or args.axis_rate_hz < AXIS_RATE_HZ:
            raise SystemExit(
                f"[lite3] REFUSING TO WALK: --axis-rate-hz must be at least "
                f"{AXIS_RATE_HZ:.0f}"
            )
        if not math.isfinite(args.axis_heartbeat_hz) or args.axis_heartbeat_hz < 2.0:
            raise SystemExit(
                "[lite3] REFUSING TO WALK: --axis-heartbeat-hz must be at least 2"
            )
        if not math.isfinite(args.axis_command_ttl) \
                or not 0.0 < args.axis_command_ttl < 0.25:
            raise SystemExit(
                "[lite3] REFUSING TO WALK: --axis-command-ttl must be within 0..0.25 s"
            )
        if not 0 <= args.axis_local_port <= 65535:
            raise SystemExit(
                "[lite3] REFUSING TO WALK: --axis-local-port must be within 0..65535"
            )

    @staticmethod
    def _load_axis_profile(args) -> AxisProfile | None:
        if args.axis_profile is None:
            return None
        try:
            return AxisProfile.load(args.axis_profile)
        except AxisProfileError as error:
            raise SystemExit(f"[lite3] REFUSING TO USE AXIS PROFILE: {error}") from None

    def _validate_axis_profile_for_envelope(self, args) -> None:
        profile = self._axis_profile
        if profile is None:
            return
        missing = []
        if args.max_vx > 0.0 and args.derate > 0.0 and profile.forward_positive is None:
            missing.append("forward_positive")
        if args.max_vy > 0.0 and args.derate > 0.0:
            if profile.lateral_positive is None:
                missing.append("lateral_positive")
            if profile.lateral_negative is None:
                missing.append("lateral_negative")
        if args.max_wz > 0.0 and args.derate > 0.0:
            if profile.yaw_positive is None:
                missing.append("yaw_positive")
            if profile.yaw_negative is None:
                missing.append("yaw_negative")
        if missing:
            raise SystemExit(
                "[lite3] REFUSING TO WALK: axis profile lacks evidenced primitives for "
                + ", ".join(missing)
            )
        self._validate_axis_profile_speeds(args, profile)

    @staticmethod
    def _validate_axis_profile_speeds(args, profile) -> None:
        """Enforce the derated envelope on a mapping that cannot scale to it.

        ``AxisProfile.map_velocity`` is sign-only: a primitive leaves at the speed it was
        measured at whatever the planner asked for, so ``--derate`` and ``--max-vx`` never
        reach the wire. This is where they are honoured instead. A profile that declares
        ``measured_m_s``/``measured_rad_s`` above the derated ceiling is refused, because
        the alternative is a safety veto reasoning about a command the robot will not
        execute. A profile that declares nothing says so on stdout: it is not a claim
        that the envelope holds, only that nobody checked.
        """
        measured = profile.measured_speeds
        enabled = []
        if args.derate > 0.0:
            if args.max_vx > 0.0:
                enabled += [(name, args.max_vx * args.derate, "m/s")
                            for name in ("forward_positive", "forward_negative")]
            if args.max_vy > 0.0:
                enabled += [(name, args.max_vy * args.derate, "m/s")
                            for name in ("lateral_positive", "lateral_negative")]
            if args.max_wz > 0.0:
                enabled += [(name, args.max_wz * args.derate, "rad/s")
                            for name in ("yaw_positive", "yaw_negative")]

        over = []
        unmeasured = []
        for name, ceiling, unit in enabled:
            if getattr(profile, name) is None:
                continue
            speed = measured.get(name)
            if speed is None:
                unmeasured.append(name)
            elif speed > ceiling:
                over.append(f"{name} measured {speed:.3f} {unit} against a "
                            f"{ceiling:.3f} {unit} ceiling")
        if over:
            raise SystemExit(
                "[lite3] REFUSING TO WALK: the axis mapping is sign-only, so --derate "
                "cannot scale these primitives down to the envelope: " + "; ".join(over)
            )
        if unmeasured:
            print("[lite3] AXIS SPEEDS ARE NOT VERIFIED AGAINST THE ENVELOPE: "
                  + ", ".join(unmeasured))
            print("[lite3]   the axis mapping is sign-only, so --derate and --max-vx do "
                  "not reach the wire.")
            print("[lite3]   Add measured_m_s/measured_rad_s to the profile once these "
                  "primitives are timed.")

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
        if args.live and data.get("calibration_status") != "validated":
            raise SystemExit(
                f"[lite3] REFUSING TO WALK: {path} is not a validated Lite3 calibration"
            )

    def prepare_motion(self, _args, loco) -> None:
        loco.prepare_motion()
        if getattr(_args, "locomotion_transport", "udp") == "axis":
            loco.assert_axis_state_ready()

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
        axis_profile = self._axis_profile_telemetry(args)
        return {
            "platform": {
                "name": self.platform_name,
                "transport": getattr(args, "locomotion_transport", "udp"),
                "motion_host": getattr(args, "motion_host", None),
                "state_bind": getattr(args, "state_bind", None),
                "cmd_vel_topic": args.cmd_vel_topic,
                "odom_topic": args.odom_topic,
                "motor_temperatures_monitored": not args.accept_no_motor_temperatures,
                "camera_source_kind": camera_source_kind(
                    parse_camera_source(args.camera_source),
                    gstreamer=args.camera_gstreamer,
                ),
                "gait_floor_m_s": args.gait_floor,
                "actuator_gain": args.actuator_gain,
                "envelope_provenance": self._envelope_provenance(args),
                "axis_profile_schema": (
                    AXIS_PROFILE_SCHEMA
                    if getattr(args, "locomotion_transport", "udp") == "axis"
                    else None
                ),
                "axis_profile": axis_profile,
            }
        }

    def _envelope_provenance(self, args) -> dict:
        """Whose envelope this recording was made under.

        The navigator already writes the VALUES into the telemetry header. What it cannot
        write is whether a Lite3 operator chose them or the shared Go2 default supplied
        them, and a run stamped ``platform.name: deep-robotics-lite3-venture`` reads as
        the former either way. That is the half of the defect a reviewer opening a JSONL
        six weeks later has no other way to recover.
        """
        return {
            "stated": sorted(flag for flag, (name, _unit) in ENVELOPE_ARGUMENTS.items()
                             if getattr(args, name, None) is not None
                             and name not in self._envelope_inherited),
            "inherited_from_unitree_go2": sorted(
                flag for flag, (name, _unit) in ENVELOPE_ARGUMENTS.items()
                if name in self._envelope_inherited),
        }

    def _axis_profile_telemetry(self, args) -> dict | None:
        if getattr(args, "locomotion_transport", "udp") != "axis":
            return None
        profile = self._axis_profile
        if profile is None and args.axis_profile is not None:
            profile = self._load_axis_profile(args)
        if profile is None:
            return None
        try:
            digest = hashlib.sha256(args.axis_profile.read_bytes()).hexdigest()
        except OSError as error:
            raise SystemExit(
                f"[lite3] REFUSING TO RECORD AXIS PROFILE: cannot read {args.axis_profile}: {error}"
            ) from None
        return {
            "sha256": digest,
            "allowed_gait_states": list(profile.allowed_gait_states),
            "primitives": {
                "forward_positive": profile.forward_positive,
                "forward_negative": profile.forward_negative,
                "lateral_positive": profile.lateral_positive,
                "lateral_negative": profile.lateral_negative,
                "yaw_positive": profile.yaw_positive,
                "yaw_negative": profile.yaw_negative,
            },
            "measured_speeds": profile.measured_speeds,
            "evidence": dict(profile.evidence),
        }

    def calibration_provenance(self, _args) -> dict:
        return {
            "platform": self.platform_name,
            "calibration_status": "provisional",
        }

    def actuation_summary(self, top_speed: float, args) -> str:
        achieved = (None if args.actuator_gain is None
                    else top_speed * args.actuator_gain)
        if achieved is None:
            summary = (f"top commanded speed {top_speed:.3f} m/s; Lite3 actuator gain "
                       f"not supplied (dry/offline only)")
        else:
            summary = (f"top commanded speed {top_speed:.3f} m/s; measured gain "
                       f"{args.actuator_gain:.3f} predicts {achieved:.3f} m/s achieved")
        return summary + self._policy_envelope_note(args)

    @staticmethod
    def _policy_envelope_note(args) -> str:
        """The SECOND route the Go2's envelope reaches this robot by, and it is worse.

        ``--max-vx`` is a CLAMP. The number that decides what is actually commanded on
        the policy drive path is ``max_vx_mps x command_scale``, and ``max_vx_mps`` /
        ``max_vy_mps`` live in the shipped ``policy/config.json`` -- where they are the
        same Go2 pair, carried in without a provenance comment while every neighbouring
        field has one. Stating ``--max-vx`` explicitly, as the deployment SOP does, does
        not touch them: a clamp above the commanded speed changes nothing.

        This warns rather than refusing, on purpose. There is no measured Lite3 value to
        put in a replacement config and no per-field override to put one in with, so a
        refusal here would force somebody to type a plausible number -- which is the
        defect, not the fix. ``--policy-config`` (a whole package config) and
        ``--policy-command-scale`` (both axes at once) are the only knobs that exist.

        Empty on the plain navigator, which has no policy and no such flag.
        """
        if not hasattr(args, "policy_config") or args.policy_config is not None:
            return ""
        return (
            "\n[mappo_drive] ⚠️  that speed is max_vx_mps x command_scale from the "
            "shipped policy/config.json,\n"
            "[mappo_drive]   whose max_vx_mps/max_vy_mps are the UNITREE GO2's envelope. "
            "--max-vx clamps\n"
            "[mappo_drive]   ABOVE them and so does not replace them; no Lite3 produced "
            "either number\n"
            "[mappo_drive]   (issue #13). --policy-config is the only way to state a "
            "different pair."
        )

    def shutdown(self) -> None:
        return
