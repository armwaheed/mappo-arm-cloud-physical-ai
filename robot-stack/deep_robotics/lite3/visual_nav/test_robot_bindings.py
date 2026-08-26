#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 injection seam and its live calibration gates."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_ROBOT_STACK = _HERE.parents[2]
_COMMON = _ROBOT_STACK / "unitree" / "go2" / "visual_nav"
sys.path.insert(0, str(_ROBOT_STACK))
sys.path.insert(0, str(_COMMON))

import visual_nav
from deep_robotics.lite3.visual_nav.robot_bindings import Lite3Bindings


class _Health:
    def abort_reason(self):
        return None

    def latest(self):
        return SimpleNamespace(max_motor_temp_c=32.0, battery_soc_pct=80.0)

    def warning_reason(self):
        return None


def _args(*extra):
    parser = visual_nav.build_parser(Lite3Bindings())
    return parser.parse_args(["--camera-source", "0", *extra])


def test_the_lite3_cli_has_no_go2_arm_bypass_or_motion_mode_flags():
    args = _args()
    assert not hasattr(args, "no_require_arm")
    assert not hasattr(args, "no_latch_arm")
    assert not hasattr(args, "motion_mode")


def test_the_public_lite3_ros_topics_are_defaults_and_radius_is_not_guessed():
    args = _args()
    assert args.cmd_vel_topic == "/cmd_vel"
    assert args.odom_topic == "/leg_odom2"
    assert args.robot_radius is None


def test_the_default_transport_is_udp_and_it_reaches_the_motion_host():
    """The demo path must not require a ROS 2 runtime the Venture may not have."""
    from deep_robotics.lite3.locomotion.lite3_udp_locomotion import Lite3UdpLocomotion

    args = _args()
    assert args.locomotion_transport == "udp"
    assert args.motion_host == "192.168.1.120"
    assert args.command_port == 43893
    assert args.state_port == 43897

    loco = Lite3Bindings().create_locomotion(args)
    implementation = loco._implementation_factory(
        cmd_vel_topic=args.cmd_vel_topic, odom_topic=args.odom_topic,
        stamped=False, node_name="test")
    assert isinstance(implementation, Lite3UdpLocomotion)
    assert implementation._motion_host == "192.168.1.120"


def test_selecting_ros2_keeps_the_bridge_factory_rather_than_the_udp_one():
    args = _args("--locomotion-transport", "ros2")
    loco = Lite3Bindings().create_locomotion(args)
    # The ROS factory is the module default; selecting ros2 must not inject a UDP one.
    from deep_robotics.lite3.locomotion.lite3_locomotion import _ros2_locomotion
    assert loco._implementation_factory is _ros2_locomotion


def test_the_udp_transport_can_be_pointed_at_a_second_robot():
    args = _args("--motion-host", "192.168.2.1", "--state-port", "43898")
    loco = Lite3Bindings().create_locomotion(args)
    implementation = loco._implementation_factory(
        cmd_vel_topic=None, odom_topic=None, stamped=False, node_name="test")
    assert implementation._motion_host == "192.168.2.1"
    assert implementation._state_port == 43898


def _axis_profile(path: Path):
    path.write_text(json.dumps({
        "schema": "lite3-axis-profile/v1",
        "input_deadband": {"linear_m_s": 0.05, "yaw_rad_s": 0.1},
        "allowed_gait_states": [0],
        "evidence": {
            "forward_positive": "test-forward-positive",
        },
        "primitives": {
            "forward_positive": 7000,
            "forward_negative": None,
            "lateral_positive": None,
            "lateral_negative": None,
            "yaw_positive": None,
            "yaw_negative": None,
        },
    }))


def test_axis_transport_uses_explicit_profile_and_state_bind():
    from deep_robotics.lite3.locomotion.lite3_axis_locomotion import Lite3AxisLocomotion

    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "axis-profile.json"
        _axis_profile(profile)
        args = _args(
            "--locomotion-transport", "axis",
            "--axis-profile", str(profile),
            "--state-bind", "127.0.0.1",
            "--axis-source-address", "127.0.0.1",
            "--axis-local-port", "20001",
        )
        loco = Lite3Bindings().create_locomotion(args)
        implementation = loco._implementation_factory(
            cmd_vel_topic=None, odom_topic=None, stamped=False, node_name="test",
        )
    assert isinstance(implementation, Lite3AxisLocomotion)
    assert implementation._bind == "127.0.0.1"
    assert implementation._axis_source_address == "127.0.0.1"
    assert implementation._axis_local_port == 20001
    assert implementation._axis_profile.forward_positive == 7000


def test_live_axis_transport_requires_an_evidenced_profile():
    binding = Lite3Bindings()
    args = _live_args("--locomotion-transport", "axis")
    try:
        binding.preflight_navigation(args, None, _Health())
    except SystemExit as error:
        assert "--axis-profile" in str(error)
    else:
        raise AssertionError("accepted a live axis run with no primitive profile")


def test_live_axis_transport_requires_primitives_for_enabled_axes():
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "axis-profile.json"
        _axis_profile(profile)
        args = _live_args("--locomotion-transport", "axis", "--axis-profile", str(profile))
        try:
            binding.preflight_navigation(args, None, _Health())
        except SystemExit as error:
            message = str(error)
            assert "lateral_positive" in message
            assert "yaw_positive" in message
        else:
            raise AssertionError("accepted a live axis run with unsupported directions")

        args = _live_args(
            "--locomotion-transport", "axis",
            "--axis-profile", str(profile),
            "--max-vy", "0",
            "--max-wz", "0",
        )
        binding.preflight_navigation(args, None, _Health())


def test_live_axis_transport_refuses_protocol_rate_and_ttl_violations():
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "axis-profile.json"
        _axis_profile(profile)
        invalid_cases = (
            ("--axis-rate-hz", "19.9", "--axis-rate-hz"),
            ("--axis-heartbeat-hz", "1.9", "--axis-heartbeat-hz"),
            ("--axis-command-ttl", "0.25", "--axis-command-ttl"),
        )
        for flag, value, expected in invalid_cases:
            args = _live_args(
                "--locomotion-transport", "axis",
                "--axis-profile", str(profile),
                flag, value,
            )
            try:
                binding.preflight_navigation(args, None, _Health())
            except SystemExit as error:
                assert expected in str(error)
            else:
                raise AssertionError(f"accepted invalid axis transport option {flag}={value}")


def test_telemetry_does_not_persist_camera_source_credentials():
    binding = Lite3Bindings()
    args = _args("--camera-source", "rtsp://operator:secret@camera/live?token=private")
    serialized = json.dumps(binding.telemetry_config(args))
    assert "secret" not in serialized and "private" not in serialized
    assert binding.telemetry_config(args)["platform"]["camera_source_kind"] == "rtsp"


def test_axis_telemetry_records_profile_hash_and_provenance_without_path():
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "axis-profile.json"
        _axis_profile(profile)
        args = _args("--locomotion-transport", "axis", "--axis-profile", str(profile))
        platform = binding.telemetry_config(args)["platform"]
        assert platform["axis_profile_schema"] == "lite3-axis-profile/v1"
        expected_sha256 = hashlib.sha256(profile.read_bytes()).hexdigest()
        assert platform["axis_profile"]["sha256"] == expected_sha256
        assert platform["axis_profile"]["allowed_gait_states"] == [0]
        assert platform["axis_profile"]["primitives"]["forward_positive"] == 7000
        assert platform["axis_profile"]["evidence"]["forward_positive"] == "test-forward-positive"
        assert str(profile) not in json.dumps(platform)


def test_a_live_run_requires_every_robot_specific_measurement_and_operator_gate():
    binding = Lite3Bindings()
    args = _args("--live")
    try:
        binding.preflight_navigation(args, None, _Health())
    except SystemExit as exc:
        message = str(exc)
        for required in ("--calibration", "--gait-floor", "--actuator-gain",
                         "--robot-radius", "--operator-ready"):
            assert required in message
        return
    raise AssertionError("an uncalibrated live Lite3 run was accepted")


def _live_args(*extra):
    return _args("--live", "--calibration", "x.json", "--gait-floor", "0.3",
                 "--actuator-gain", "0.7", "--robot-radius", "0.25",
                 "--operator-ready", *extra)


def test_running_without_temperature_monitoring_must_be_time_bounded():
    binding = Lite3Bindings()
    args = _live_args("--accept-no-motor-temperatures", "--max-seconds", "600")
    try:
        binding.preflight_navigation(args, None, _Health())
    except SystemExit as exc:
        assert "--accept-no-motor-temperatures" in str(exc)
        assert "120s or less" in str(exc)
    else:
        raise AssertionError("a ten-minute run with no thermal feed was accepted")


def test_a_bounded_unmonitored_run_is_allowed():
    binding = Lite3Bindings()
    args = _live_args("--accept-no-motor-temperatures", "--max-seconds", "60")
    binding.preflight_navigation(args, None, _Health())  # must not raise


def test_the_override_is_not_needed_when_temperatures_are_monitored():
    binding = Lite3Bindings()
    args = _live_args("--max-seconds", "600")
    binding.preflight_navigation(args, None, _Health())  # must not raise


def test_telemetry_records_the_real_transport_and_whether_motors_were_watched():
    """A recording that does not say the temperatures were off cannot be reviewed later."""
    binding = Lite3Bindings()
    platform = binding.telemetry_config(_args())["platform"]
    assert platform["transport"] == "udp"
    assert platform["motion_host"] == "192.168.1.120"
    assert platform["motor_temperatures_monitored"] is True

    platform = binding.telemetry_config(
        _args("--accept-no-motor-temperatures", "--locomotion-transport", "ros2")
    )["platform"]
    assert platform["transport"] == "ros2"
    assert platform["motor_temperatures_monitored"] is False


def test_non_finite_measurements_cannot_even_poison_a_dry_run():
    binding = Lite3Bindings()
    args = _args(
        "--gait-floor", "nan", "--actuator-gain", "nan", "--robot-radius", "nan",
    )
    try:
        binding.preflight_navigation(args, None, _Health())
    except SystemExit as exc:
        message = str(exc)
        assert "--gait-floor" in message
        assert "--actuator-gain" in message
        assert "--robot-radius" in message
        return
    raise AssertionError("NaN Lite3 measurements passed the dry-run gate")


def test_live_rejects_a_calibration_file_from_another_camera_platform():
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "camera.json"
        path.write_text(json.dumps({"platform": "unitree-go2"}))
        args = _args("--live", "--calibration", str(path))
        try:
            binding.validate_camera_calibration(args)
        except SystemExit as exc:
            assert "not produced by the Lite3" in str(exc)
            return
    raise AssertionError("a Go2 camera calibration was accepted for the Lite3")


def test_lite3_calibration_provenance_round_trips_through_the_gate():
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "camera.json"
        path.write_text(json.dumps(binding.calibration_provenance(None)))
        args = _args("--calibration", str(path))
        binding.validate_camera_calibration(args)
        args = _args("--live", "--calibration", str(path))
        try:
            binding.validate_camera_calibration(args)
        except SystemExit as error:
            assert "not a validated Lite3 calibration" in str(error)
        else:
            raise AssertionError("generated provisional calibration was accepted for live movement")


def test_live_accepts_a_separately_validated_lite3_calibration():
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "camera.json"
        path.write_text(json.dumps({
            "platform": binding.platform_name,
            "calibration_status": "validated",
        }))
        args = _args("--live", "--calibration", str(path))
        binding.validate_camera_calibration(args)


def test_live_rejects_provisional_lite3_calibration():
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "camera.json"
        path.write_text(json.dumps({
            "platform": binding.platform_name,
            "calibration_status": "provisional",
        }))
        args = _args("--live", "--calibration", str(path))
        try:
            binding.validate_camera_calibration(args)
        except SystemExit as error:
            assert "not a validated Lite3 calibration" in str(error)
        else:
            raise AssertionError("a provisional calibration was accepted for live movement")


def test_live_rejects_a_missing_or_malformed_calibration_without_a_traceback():
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "camera.json"
        args = _args("--live", "--calibration", str(path))
        for contents in (None, "not JSON"):
            if contents is not None:
                path.write_text(contents)
            try:
                binding.validate_camera_calibration(args)
            except SystemExit as exc:
                assert "REFUSING TO WALK: cannot read calibration" in str(exc)
            else:
                raise AssertionError("an unreadable Lite3 calibration was accepted")


def test_gait_floor_warning_uses_the_measured_lite3_value():
    binding = Lite3Bindings()
    args = _args("--gait-floor", "0.28")
    assert binding.warn_if_below_gait_floor(0.20, args)
    assert not binding.warn_if_below_gait_floor(0.30, args)


def test_blocked_rest_does_not_claim_an_undocumented_ros_lie_down_verb():
    assert Lite3Bindings.rest_when_blocked is False


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_bindings: {len(tests)}/{len(tests)} passed")
