# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Go2-specific bindings injected into the shared visual-navigation run loop."""

from __future__ import annotations

from avoidance import MIN_GAIT_COMMAND_M_S
from camera import Go2Camera
from safety import (
    LATCH_DRIFT_TOLERANCE_DEG,
    STOWED_YAW_DEG,
    ArmStowMonitor,
    HealthMonitor,
    latch_arm,
    lie_down,
    stand_up,
)


def warn_if_below_go2_gait_floor(max_vx: float) -> bool:
    """Report the measured Go2 gait floor. Return whether a warning was emitted."""
    if max_vx >= MIN_GAIT_COMMAND_M_S:
        return False
    print("!" * 78)
    print(f"[visual_nav] ⚠️  TOP SPEED {max_vx:.2f} m/s IS BELOW THE GAIT FLOOR OF "
          f"{MIN_GAIT_COMMAND_M_S:.2f} m/s")
    print("    The Go2 does not walk slowly — below this it stands up, shuffles a few")
    print("    steps and then STANDS STILL while still being commanded forward. No")
    print("    fault is reported. The stall gate will then say 'something is holding")
    print("    the robot — check the tether'. It is not the tether. It is this number.")
    print(f"    Measured: 0.21 m/s stalled 5 of 5 runs across two controllers; "
          f"{MIN_GAIT_COMMAND_M_S:.2f} m/s walked 2.07 m in 9 s and arrived.")
    print("!" * 78)
    return True


class Go2Bindings:
    """Construct and guard the Go2 pieces around the robot-agnostic navigator."""

    platform_name = "unitree-go2"
    rest_when_blocked = True
    initially_standing = False
    default_calibration_output = "go2_front_camera.json"
    spin_rate_help = ("rad/s. Measured on this robot: 0.30 commanded achieves "
                      "0.02-0.04 (7-14%%) and below ~0.4 it does not reliably "
                      "initiate a turn at all; 0.80 achieves 0.45-0.49. See SKILL.md")

    def __init__(self) -> None:
        self._arm = ArmStowMonitor()

    def add_navigation_arguments(self, parser, envelope) -> None:
        parser.add_argument("--iface", default="eth0", help="DDS network interface")
        envelope.add_argument("--no-require-arm", action="store_true",
                              help="proceed without D1 feedback (arm physically removed)")
        envelope.add_argument(
            "--no-latch-arm", action="store_true",
            help="do NOT lock the D1 before walking. The arm is latched by default "
                 "because an unpowered one back-drives and 3.15 kg off the dorsal "
                 "centreline unbalances the vendor gait controller. Use only if it is "
                 "already locked by other means",
        )
        envelope.add_argument("--motion-mode", default="normal", choices=("normal", "ai"),
                              help="Go2 sport mode to select before walking")

    def add_calibration_arguments(self, parser, spin) -> None:
        parser.add_argument("--iface", default="eth0", help="DDS network interface")
        spin.add_argument(
            "--latch-arm", action="store_true",
            help="freeze the D1 at its current hand-posed angles before standing",
        )

    def create_locomotion(self, args):
        from locomotion.go2_locomotion import Go2Locomotion

        return Go2Locomotion(iface=args.iface)

    def create_health_monitor(self, _args, *, live: bool):
        return HealthMonitor()

    def start(self, _args) -> None:
        self._arm.start()

    def create_camera(self, args, stamp_fn):
        return Go2Camera(iface=args.iface, init_dds=False, stamp_fn=stamp_fn)

    def preflight_navigation(self, args, config, health) -> None:
        blocking = self._arm.blocking_reason(required=config.require_arm)
        if blocking is not None:
            raise SystemExit(f"[visual_nav] REFUSING TO WALK: {blocking}")
        reach = self._arm.reach_m()
        if reach is not None:
            jaw = self._arm.jaw_xyz()
            sway = self._arm.sway_deg()
            print(f"[visual_nav] D1 arm stowed (jaw {reach:.3f} m from base, sway "
                  f"{sway:.1f} deg of {STOWED_YAW_DEG:.1f} deg allowed, "
                  f"{abs(jaw[1]) * 1000:.0f} mm off the dorsal centreline)")
            if config.latch_arm:
                result = latch_arm(self._arm, iface=args.iface)
                print(f"[visual_nav] {result}")
                if not result.held:
                    raise SystemExit(
                        f"[visual_nav] REFUSING TO WALK: the D1 latch did not take "
                        f"(joints drifted {result.drift_deg:.2f} deg after enable, "
                        f"tolerance {LATCH_DRIFT_TOLERANCE_DEG:.1f} deg). Hand-pose "
                        f"the arm flat along the spine and retry. Pass --no-latch-arm "
                        f"only if it is already locked by other means."
                    )
        self._report_health(health)

    def preflight_calibration(self, args, health) -> None:
        if not args.spin:
            return
        for reason in (self._arm.blocking_reason(required=True), health.abort_reason()):
            if reason is not None:
                raise SystemExit(f"[calibrate] REFUSING TO MOVE: {reason}")
        if args.latch_arm:
            result = latch_arm(self._arm, iface=args.iface)
            print(f"[calibrate] {result}")
            if not result.held:
                raise SystemExit(
                    f"[calibrate] REFUSING TO MOVE: the D1 latch did not take "
                    f"(joints drifted {result.drift_deg:.2f} deg after enable). "
                    f"Hand-pose the arm flat along the spine and retry."
                )

    @staticmethod
    def _report_health(health) -> None:
        unhealthy = health.abort_reason()
        if unhealthy is not None:
            raise SystemExit(f"[visual_nav] REFUSING TO WALK: {unhealthy}")
        sample = health.latest()
        print(f"[visual_nav] motors {sample.max_motor_temp_c:.0f}C, "
              f"battery {sample.battery_soc_pct:.0f}%")
        warning = health.warning_reason()
        if warning is not None:
            print(f"[visual_nav] WARNING: {warning}")

    def validate_camera_calibration(self, _args) -> None:
        return

    def prepare_motion(self, args, loco) -> None:
        loco.ensure_sport_mode(getattr(args, "motion_mode", "normal"))

    def stand_up(self, loco) -> None:
        stand_up(loco)

    def lie_down(self, loco) -> None:
        lie_down(loco)

    def warn_if_below_gait_floor(self, max_vx: float, _args) -> bool:
        return warn_if_below_go2_gait_floor(max_vx)

    def gait_floor(self, _args) -> float:
        return MIN_GAIT_COMMAND_M_S

    def robot_radius(self, args, default: float) -> float:
        return default if args.robot_radius is None else args.robot_radius

    def telemetry_config(self, _args) -> dict:
        return {"platform": {"name": self.platform_name, "transport": "SportClient"}}

    def calibration_provenance(self, _args) -> dict:
        return {"platform": self.platform_name}

    def actuation_summary(self, top_speed: float, _args) -> str:
        return (f"top commanded speed {top_speed:.3f} m/s; this robot measured about "
                f"0.70 of that at full command, 0.45 derated")

    def shutdown(self) -> None:
        self._arm.stop()
