#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 injection seam and its live calibration gates."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import time
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


def _axis_profile(path: Path, **overrides):
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
        **overrides,
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

        # `measured_m_s` declared for the second half, which is a live run that must be
        # ACCEPTED. Since issue #145 a fired primitive with no measured speed is a
        # refusal in its own right — see
        # `test_an_unmeasured_linear_primitive_refuses_the_run_rather_than_warning` —
        # so without it this half would pass for the wrong reason.
        _axis_profile(profile, measured_m_s={"forward_positive": 0.30})
        args = _live_args(
            "--locomotion-transport", "axis",
            "--axis-profile", str(profile),
            "--max-vy", "0",
            "--max-wz", "0",
        )
        Lite3Bindings().preflight_navigation(args, None, _Health())


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


def test_a_primitive_measured_above_the_derated_envelope_is_refused_at_preflight():
    """``--derate`` cannot reach a sign-only mapping, so preflight is where it is honoured.

    The one primitive with physical evidence behind it, ``+32767``, measured 0.729 m/s
    peak. A run derated to 0.20 m/s plans a sweep for a robot that will travel 3.6x
    further than the safety veto assumed.
    """
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "axis-profile.json"
        _axis_profile(profile, measured_m_s={"forward_positive": 0.729})
        common = ("--locomotion-transport", "axis", "--axis-profile", str(profile),
                  "--max-vy", "0", "--max-wz", "0")

        args = _live_args(*common, "--max-vx", "0.30", "--derate", "0.2")
        try:
            binding.preflight_navigation(args, None, _Health())
        except SystemExit as error:
            message = str(error)
            assert "forward_positive measured 0.729 m/s" in message
            assert "0.060 m/s ceiling" in message
        else:
            raise AssertionError("accepted a primitive that outruns the derated envelope")

        # Same profile, an envelope it actually fits inside.
        Lite3Bindings().preflight_navigation(
            _live_args(*common, "--max-vx", "0.80", "--derate", "1.0"), None, _Health())


def test_an_unmeasured_linear_primitive_refuses_the_run_rather_than_warning():
    """⚠️ WAS A WARNING UNTIL ISSUE #145, AND THE WARNING WAS THE WHOLE GAP.

    An undeclared ``measured_m_s`` used to mean only that ``--max-vx`` had nothing to
    compare against — a check nobody may have wanted. It means more than that now. The
    mapping is sign-only, so the speed the legs produce IS the primitive's measured
    speed; with the field absent, ``avoidance.DynamicWindowPlanner`` cannot roll out
    what the robot will do, marks every candidate unnameable, and holds for the whole
    run. A refusal in the operator's face before a leg moves is the same outcome,
    findable.

    Made to fail by putting the number back: the second half of this test is the same
    profile with ``measured_m_s`` declared, and it walks through.
    """
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "axis-profile.json"
        _axis_profile(profile)
        args = _live_args("--locomotion-transport", "axis", "--axis-profile", str(profile),
                          "--max-vy", "0", "--max-wz", "0")
        try:
            binding.preflight_navigation(args, None, _Health())
        except SystemExit as error:
            message = str(error)
            assert "forward_positive" in message, message
            assert "sign-only" in message, message
            assert "axis_primitive_probe" in message, message
        else:
            raise AssertionError(
                "accepted a linear primitive whose delivered speed nobody has timed")

        _axis_profile(profile, measured_m_s={"forward_positive": 0.30})
        Lite3Bindings().preflight_navigation(
            _live_args("--locomotion-transport", "axis", "--axis-profile", str(profile),
                       "--max-vy", "0", "--max-wz", "0"), None, _Health())


def test_an_unmeasured_yaw_rate_warns_and_names_what_the_planner_will_not_do():
    """The SAME defect one axis along, and deliberately not a refusal. Issue #145.

    ``input_deadband.yaw_rad_s`` is the rate at which the yaw primitive fires at full
    scale, exactly as ``linear_m_s`` is for forward, so ``--max-wz`` never reaches the
    wire either. It cannot be fixed the same way because nothing has measured it:
    ``commissioning/axis_primitive_probe.py`` refuses to time yaw while
    ``Segment.yaw_change_deg`` can report a turn through pi backwards, so
    ``measured_rad_s`` is empty on every profile in this repository and the deployment
    SOP's live command runs at ``--max-wz 0.90``. Refusing would leave a robot that
    cannot turn at all, which is a worse failure than the one being fixed.

    What is enforced instead is that the cost is SAID: the planner will not combine a
    turn with a step, because a rollout that translates on an unknown arc ends somewhere
    nobody can name.
    """
    binding = Lite3Bindings()
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "axis-profile.json"
        _axis_profile(
            profile,
            evidence={"forward_positive": "e", "yaw_positive": "e", "yaw_negative": "e"},
            primitives={"forward_positive": 7000, "forward_negative": None,
                        "lateral_positive": None, "lateral_negative": None,
                        "yaw_positive": 10000, "yaw_negative": -10000},
            measured_m_s={"forward_positive": 0.30},
        )
        args = _live_args("--locomotion-transport", "axis", "--axis-profile", str(profile),
                          "--max-vy", "0", "--max-wz", "0.90")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            binding.preflight_navigation(args, None, _Health())
        printed = stdout.getvalue()
        assert "THE EXECUTED YAW RATE IS NOT MEASURED" in printed, printed
        assert "yaw_positive" in printed and "yaw_negative" in printed, printed
        assert "WILL NOT TURN AND STEP AT THE SAME TIME" in printed, printed

        # And it goes quiet once the rate is declared — the complement, without which
        # this test would pass against a binding that printed the banner unconditionally.
        _axis_profile(
            profile,
            evidence={"forward_positive": "e", "yaw_positive": "e", "yaw_negative": "e"},
            primitives={"forward_positive": 7000, "forward_negative": None,
                        "lateral_positive": None, "lateral_negative": None,
                        "yaw_positive": 10000, "yaw_negative": -10000},
            measured_m_s={"forward_positive": 0.30},
            measured_rad_s={"yaw_positive": 0.60, "yaw_negative": 0.60},
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            Lite3Bindings().preflight_navigation(args, None, _Health())
        assert "THE EXECUTED YAW RATE IS NOT MEASURED" not in stdout.getvalue()


def test_the_bindings_hand_the_planner_a_sign_only_transport_only_on_the_axis_path():
    """The seam issue #145's fix travels through, and the one place it is chosen.

    ``visual_nav.main`` asks the bindings for a transport model and puts it on
    ``Limits``. Getting this wrong in either direction is silent: a Lite3 answering
    ``PROPORTIONAL`` on the axis transport is the bug, and a UDP run answering
    ``SignOnlyAxisTransport`` would refuse commands the vendor's complex-velocity sender
    honours perfectly well.
    """
    from avoidance import PROPORTIONAL
    from deep_robotics.lite3.locomotion.lite3_axis_locomotion import SignOnlyAxisTransport

    binding = Lite3Bindings()
    assert binding.transport_model(_args()) is PROPORTIONAL, "the UDP transport scales"
    assert binding.transport_model(_args("--locomotion-transport", "ros2")) is PROPORTIONAL

    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "axis-profile.json"
        _axis_profile(profile, measured_m_s={"forward_positive": 0.30})
        args = _args("--locomotion-transport", "axis", "--axis-profile", str(profile))
        model = Lite3Bindings().transport_model(args)
        assert isinstance(model, SignOnlyAxisTransport), model
        assert not model.is_proportional
        rows, known = model.executed([(0.05, 0.0, 0.0), (0.55, 0.0, 0.0)])
        assert rows[0] == rows[1] == (0.30, 0.0, 0.0), rows
        assert known == [True, True]


def test_prepare_motion_checks_the_vendor_state_before_the_axis_transport_moves():
    """Deleting the ``assert_axis_state_ready()`` call here left this suite at 24/24."""
    from deep_robotics.lite3.locomotion.lite3_udp_locomotion import Lite3LinkLost

    class _Loco:
        def __init__(self, ready=True):
            self.calls = []
            self._ready = ready

        def prepare_motion(self):
            self.calls.append("prepare_motion")

        def assert_axis_state_ready(self):
            self.calls.append("assert_axis_state_ready")
            if not self._ready:
                raise Lite3LinkLost("Lite3 gait_state=2; axis profile allows (0,)")

    binding = Lite3Bindings()
    blocked = _Loco(ready=False)
    try:
        binding.prepare_motion(_args("--locomotion-transport", "axis"), blocked)
    except Lite3LinkLost as error:
        assert "gait_state=2" in str(error)
    else:
        raise AssertionError("prepared axis motion without checking the vendor state")
    assert blocked.calls == ["prepare_motion", "assert_axis_state_ready"]

    # The legacy transport has no such state contract and must not be asked for one.
    udp = _Loco()
    binding.prepare_motion(_args(), udp)
    assert udp.calls == ["prepare_motion"]


class _BatteryLoco:
    """A locomotion stub that reports a battery, or refuses the way a dead link does."""

    def __init__(self, battery):
        self._battery = battery

    def battery_level(self):
        from deep_robotics.lite3.locomotion.lite3_udp_locomotion import Lite3LinkLost

        if self._battery is None:
            raise Lite3LinkLost("the Lite3 state stream has been silent for 3.20s")
        return self._battery


def test_the_bindings_carry_the_battery_from_the_locomotion_to_the_health_monitor():
    """The seam, not the combinator. This is the hop the live run actually broke.

    ``test_lite3_locomotion.py`` covers the combinator's ``battery_level`` well, but
    nothing called ``Lite3Bindings.create_health_monitor`` at all. Renaming the call it
    makes -- ``battery_level()`` to ``battery()``, the exact production bug -- left the
    three Lite3 suites at 308/308.
    """
    binding = Lite3Bindings()
    monitor = binding.create_health_monitor(
        _args("--accept-no-motor-temperatures"), live=True)
    monitor.start(wait_s=0.0)
    try:
        # The source returns None while the binding's locomotion is unset. Both shipped
        # entry points (visual_nav.py:1239-1240, calibrate_camera.py:601-602) build the
        # locomotion first, so that guard never fires there -- but it is what decides
        # whether "no reading" is reported as a broken reader, and it must not be.
        time.sleep(0.3)
        assert monitor.latest() is None
        assert monitor._battery_source_raises == 0, \
            "a not-yet-connected locomotion was reported as a broken source"

        binding._locomotion = _BatteryLoco(76.5)
        deadline = time.monotonic() + 3.0
        while monitor.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        health = monitor.latest()
        assert health is not None, "the battery never reached the monitor"
        assert abs(health.battery_soc_pct - 76.5) < 1e-9
        assert monitor.abort_reason() is None
    finally:
        monitor.stop()


def test_a_live_run_refuses_when_the_locomotion_reports_no_battery():
    """``required=live`` is the master switch on the live gate, and nothing pinned it.

    Turning it into ``required=False`` also left the three suites at 308/308: a live run
    would then proceed with no battery reading at all. The refusal must also name the
    feed that is actually in play -- a ROS topic name here sent the live investigation
    to a subscription this path never makes.
    """
    binding = Lite3Bindings()
    binding._locomotion = _BatteryLoco(None)  # the link is silent
    args = _args("--accept-no-motor-temperatures")

    monitor = binding.create_health_monitor(args, live=True)
    monitor.start(wait_s=0.3)
    try:
        reason = monitor.abort_reason()
        assert reason is not None, "a live run was cleared with no battery reading"
        assert "battery" in reason
        assert "locomotion state stream" in reason, reason
        assert args.battery_topic not in reason, \
            f"the refusal names a ROS topic this path never subscribes to: {reason}"
    finally:
        monitor.stop()

    # A dry run has no legs to protect and must not be blocked by the same absence.
    monitor = binding.create_health_monitor(args, live=False)
    monitor.start(wait_s=0.0)
    try:
        assert monitor.abort_reason() is None
    finally:
        monitor.stop()


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
                         "--robot-radius", "--max-vx", "--max-vy", "--max-wz",
                         "--operator-ready"):
            assert required in message
        return
    raise AssertionError("an uncalibrated live Lite3 run was accepted")


def test_the_velocity_envelope_is_blanked_the_way_the_radius_is():
    """``Limits``' three velocity defaults are the Go2's arm-fitted profile, and this
    parser used to hand them to a Lite3 as though somebody had chosen them.

    They are checked as ``None`` rather than "not 0.35": the point is that an omitted
    value is DISTINGUISHABLE from a stated one, which is what lets the live gate below
    refuse. A default of any other number would be the same defect with a different
    robot's numbers in it.
    """
    args = _args()
    assert args.max_vx is None
    assert args.max_vy is None
    assert args.max_wz is None
    # ...and the neighbouring knob that is NOT platform-specific is untouched.
    assert args.derate == 1.0


def test_a_live_run_refuses_the_go2s_envelope_and_names_whose_it_is():
    """The mutation this exists to catch is ``set_defaults(max_vx=0.35, ...)`` coming
    back -- i.e. the state of ``main`` before this change."""
    binding = Lite3Bindings()
    parser = visual_nav.build_parser(binding)
    args = parser.parse_args([
        "--camera-source", "0", "--live", "--calibration", "x.json",
        "--gait-floor", "0.3", "--actuator-gain", "0.7", "--robot-radius", "0.25",
        "--operator-ready",
    ])
    try:
        binding.preflight_navigation(args, None, _Health())
    except SystemExit as exc:
        message = str(exc)
        assert "--max-vx" in message and "--max-vy" in message and "--max-wz" in message
        assert "Go2" in message, message
        # The Go2's actual numbers are quoted WITH THEIR UNITS, so the operator can see
        # what they would have been running under and cannot read the yaw rate as a
        # third speed. Matched as whole fragments: a bare "0.2" also matches "0.25".
        assert "the Go2's 0.35 m/s" in message, message
        assert "the Go2's 0.2 m/s" in message, message
        assert "the Go2's 0.7 rad/s" in message, message
        return
    raise AssertionError("a live Lite3 run was accepted on the Go2's velocity envelope")


def test_a_dry_run_keeps_working_and_says_whose_envelope_it_is_standing_on():
    """A gate that stops the offline path is a gate nobody runs before a live one.

    So the dry run behaves exactly as it did before -- and announces the inheritance,
    which is the entire difference.
    """
    binding = Lite3Bindings()
    parser = visual_nav.build_parser(binding)
    args = parser.parse_args(["--camera-source", "0"])
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        binding.preflight_navigation(args, None, _Health())
    printed = stdout.getvalue()
    assert "ENVELOPE NOT STATED" in printed
    assert "GO2" in printed
    assert "#13" in printed
    # The planner still gets a usable envelope, and it is the one it got before.
    assert (args.max_vx, args.max_vy, args.max_wz) == (0.35, 0.20, 0.70)


def test_a_dry_run_that_states_the_envelope_is_not_warned_at():
    binding = Lite3Bindings()
    parser = visual_nav.build_parser(binding)
    args = parser.parse_args(["--camera-source", "0", *_TEST_ENVELOPE])
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        binding.preflight_navigation(args, None, _Health())
    assert "ENVELOPE NOT STATED" not in stdout.getvalue()
    assert (args.max_vx, args.max_vy, args.max_wz) == (0.31, 0.13, 0.61)


def test_the_captured_go2_default_is_the_named_constant_and_not_a_copy():
    """Two ways to name the same three numbers, and they have to agree.

    ``add_navigation_arguments`` reads what the shared parser is actually defaulting to;
    ``_go2_envelope``'s fallback imports the named constants. If a future change sets
    ``Limits``' defaults from something other than ``GO2_MAX_*`` the two diverge, and an
    operator gets told the wrong numbers were the ones they would have inherited.
    """
    from avoidance import GO2_MAX_VX_M_S, GO2_MAX_VY_M_S, GO2_MAX_WZ_RAD_S, Limits

    binding = Lite3Bindings()
    visual_nav.build_parser(binding)
    captured = binding._inherited_envelope
    assert captured == {"max_vx": GO2_MAX_VX_M_S, "max_vy": GO2_MAX_VY_M_S,
                        "max_wz": GO2_MAX_WZ_RAD_S}
    assert captured == {"max_vx": Limits().max_vx, "max_vy": Limits().max_vy,
                        "max_wz": Limits().max_wz}
    # And the fallback used when the parser was built by a different instance agrees.
    assert Lite3Bindings()._go2_envelope() == captured


def test_zero_disables_an_axis_and_is_not_treated_as_an_omitted_measurement():
    """``--max-vy 0`` is how DEPLOYMENT-SOP.md turns the strafe axis off. Reusing
    ``_positive_finite`` here would have refused the documented live command."""
    binding = Lite3Bindings()
    args = _live_args("--max-vy", "0")
    binding.preflight_navigation(args, None, _Health())
    assert args.max_vy == 0.0


def test_a_negative_or_nan_envelope_cannot_even_poison_a_dry_run():
    for flag, value in (("--max-vx", "nan"), ("--max-vy", "-0.1"),
                        ("--max-wz", "inf")):
        binding = Lite3Bindings()
        try:
            binding.preflight_navigation(_args(flag, value), None, _Health())
        except SystemExit as exc:
            assert flag in str(exc)
        else:
            raise AssertionError(f"a Lite3 dry run accepted {flag}={value}")


def test_the_axis_speed_gate_is_never_answered_by_the_go2s_ceiling():
    """``_validate_axis_profile_speeds`` is the gate PR #100's probe supplies the left
    side of. Its right side is ``--max-vx x --derate``.

    A profile measured at 0.729 m/s is over the Go2's 0.35 and under a stated 0.80. With
    the envelope unstated the run must be refused for the MISSING ENVELOPE -- not
    "refused" by a comparison against a ceiling no Lite3 produced, which would read like
    a verdict on this robot and would be one on a different one.
    """
    binding = Lite3Bindings()
    parser = visual_nav.build_parser(binding)
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "axis-profile.json"
        _axis_profile(profile, measured_m_s={"forward_positive": 0.729})
        args = parser.parse_args([
            "--camera-source", "0", "--live", "--calibration", "x.json",
            "--gait-floor", "0.3", "--actuator-gain", "0.7", "--robot-radius", "0.25",
            "--operator-ready", "--locomotion-transport", "axis",
            "--axis-profile", str(profile),
        ])
        try:
            binding.preflight_navigation(args, None, _Health())
        except SystemExit as exc:
            message = str(exc)
            assert "--max-vx" in message, message
            assert "0.729" not in message, (
                "the sign-only speed gate was evaluated against the Go2's ceiling: "
                + message)
            return
    raise AssertionError("a live axis run was accepted with no envelope stated")


def test_the_recording_says_the_envelope_was_inherited_rather_than_chosen():
    """A JSONL stamped ``deep-robotics-lite3-venture`` reads as this robot's numbers
    whether or not anybody chose them. This is the only field that can tell them apart."""
    binding = Lite3Bindings()
    parser = visual_nav.build_parser(binding)
    args = parser.parse_args(["--camera-source", "0"])
    with contextlib.redirect_stdout(io.StringIO()):
        binding.preflight_navigation(args, None, _Health())
    provenance = binding.telemetry_config(args)["platform"]["envelope_provenance"]
    assert provenance["inherited_from_unitree_go2"] == ["--max-vx", "--max-vy", "--max-wz"]
    assert provenance["stated"] == []


def test_a_stated_envelope_is_recorded_as_stated():
    binding = Lite3Bindings()
    parser = visual_nav.build_parser(binding)
    args = parser.parse_args(["--camera-source", "0", *_TEST_ENVELOPE])
    with contextlib.redirect_stdout(io.StringIO()):
        binding.preflight_navigation(args, None, _Health())
    provenance = binding.telemetry_config(args)["platform"]["envelope_provenance"]
    assert provenance["inherited_from_unitree_go2"] == []
    assert provenance["stated"] == ["--max-vx", "--max-vy", "--max-wz"]


def test_the_drive_path_names_the_second_route_the_go2s_numbers_arrive_by():
    """``--max-vx`` is a clamp. What is COMMANDED on the policy path is
    ``max_vx_mps x command_scale`` out of ``policy/config.json``, which carries the same
    Go2 pair and has no per-field override. Stating ``--max-vx`` does not touch it.
    """
    binding = Lite3Bindings()
    drive = SimpleNamespace(actuator_gain=1.07, policy_config=None)
    summary = binding.actuation_summary(0.35, drive)
    assert "policy/config.json" in summary
    assert "GO2" in summary

    # An operator who supplied their own package config has answered for it.
    stated = SimpleNamespace(actuator_gain=1.07, policy_config=Path("lite3.json"))
    assert "policy/config.json" not in binding.actuation_summary(0.35, stated)

    # And the plain navigator, which has no policy at all, says nothing about one.
    plain = SimpleNamespace(actuator_gain=1.07)
    assert "policy/config.json" not in binding.actuation_summary(0.35, plain)


#: Envelope for the live fixture. Deliberately NOT the Go2's 0.35/0.20/0.70: if a change
#: ever puts those back as the Lite3 default, a test that happened to state the same
#: numbers would keep passing while the defect was live again.
_TEST_ENVELOPE = ("--max-vx", "0.31", "--max-vy", "0.13", "--max-wz", "0.61")


def _live_args(*extra):
    # ``*extra`` LAST so a caller can override any of these; argparse takes the final
    # occurrence.
    return _args("--live", "--calibration", "x.json", "--gait-floor", "0.3",
                 "--actuator-gain", "0.7", "--robot-radius", "0.25",
                 *_TEST_ENVELOPE, "--operator-ready", *extra)


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


# --------------------------------------------------------- the virtualenv guard is wired
#
# What the guard DECIDES is tested in robot-stack/preflight/test_venv_guard.py against an
# injected environment and an injected filesystem. What these two test is that this binding
# actually calls it on the live path, and calls it first -- the wiring is the half that a
# refactor silently drops, and a guard nobody calls passes every test it has.

def _patched_guard(monkeypatched, raises=None):
    import deep_robotics.lite3.visual_nav.robot_bindings as bindings_module

    calls = []

    def fake(component, reaching_hardware, **kwargs):
        calls.append((component, reaching_hardware))
        if raises is not None:
            raise SystemExit(raises)
        return monkeypatched

    original = bindings_module.require_virtualenv
    bindings_module.require_virtualenv = fake
    return bindings_module, original, calls


def test_a_live_run_asks_the_virtualenv_guard_and_says_what_it_decided():
    from preflight.venv_guard import Decision

    decision = Decision(False, None, "no robot-host evidence on this machine")
    module, original, calls = _patched_guard(decision)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            try:
                Lite3Bindings().preflight_navigation(_live_args(), None, _Health())
            except SystemExit as exc:
                raise AssertionError(
                    f"a complete live namespace was refused: {exc}") from exc
    finally:
        module.require_virtualenv = original
    assert calls == [("lite3 visual_nav --live", True)], calls
    assert "venv-guard: not enforced" in buffer.getvalue(), (
        "a guard that decided NOT to enforce has to say so; this binding's robot has no "
        "measured host marker, and an invisible gate is one nobody can audit")


def test_the_virtualenv_refusal_precedes_the_missing_measurement_list():
    """Answering '--gait-floor is missing' to a system-Python problem sends an operator
    to measure a robot when what they needed was `source bin/activate`."""
    module, original, calls = _patched_guard(None, raises="[venv-guard] REFUSING TO RUN")
    try:
        # _args("--live") alone is missing every measurement, so without the guard this
        # raises "REFUSING TO WALK: missing ...". The guard has to win.
        try:
            Lite3Bindings().preflight_navigation(_args("--live"), None, _Health())
        except SystemExit as exc:
            assert "[venv-guard] REFUSING TO RUN" in str(exc), str(exc)
        else:
            raise AssertionError("preflight_navigation accepted a refused interpreter")
    finally:
        module.require_virtualenv = original
    assert calls == [("lite3 visual_nav --live", True)], calls


def test_an_offline_run_never_reaches_the_virtualenv_guard():
    """Parsing telemetry, printing --help and a shadow pass touch no leg and no SDK."""
    module, original, calls = _patched_guard(None, raises="should not have been called")
    try:
        Lite3Bindings().preflight_navigation(_args(), None, _Health())
    finally:
        module.require_virtualenv = original
    assert calls == []


def test_a_lite3_crawl_past_a_bin_is_refused_rather_than_walked_at_full_speed():
    """⚠️ ISSUE #145 END TO END: the real profile, the real planner, the real geometry.

    The two halves are pinned separately — ``AxisProfile.executed_velocity`` in
    ``locomotion/test_lite3_axis_locomotion.py``, the planner's contract in
    ``unitree/go2/visual_nav/test_avoidance.py`` — and each could pass while the wire
    between them was missing. ``visual_nav.main`` is the wire: it asks the bindings for a
    transport model and puts it on ``Limits``. Nothing else does.

    The scene is the one the issue replays. On 2026-08-27 a Go2 stood 0.72 m from a bin
    while its planner commanded 0.052, 0.103, 0.050, 0.101 m/s, and froze. The same
    commands on this transport are all above ``input_deadband.linear_m_s``, so every one
    of them fires the forward primitive: the crawl leaves as a 0.30 m/s walk at the bin,
    and it used to be validated as a 0.05 m/s one.
    """
    from avoidance import (
        PROPORTIONAL,
        STATIC_HARD_GAP_M,
        STATIC_SOFT_GAP_M,
        DynamicWindowPlanner,
        Limits,
        Obstacle,
        PlannerConfig,
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "axis-profile.json"
        _axis_profile(path, measured_m_s={"forward_positive": 0.30})
        args = _args("--locomotion-transport", "axis", "--axis-profile", str(path))
        transport = Lite3Bindings().transport_model(args)

    bin_072 = Obstacle(x=0.72, y=0.0, vx=0.0, vy=0.0, radius_m=0.20, label="bin",
                       person_shaped=False, kind="static",
                       soft_gap_m=STATIC_SOFT_GAP_M, hard_gap_m=STATIC_HARD_GAP_M)
    config = PlannerConfig(robot_radius_m=0.25)

    def planner(model):
        # The deployment SOP's live envelope, and the transport is the ONLY difference.
        return DynamicWindowPlanner(
            limits=Limits(max_vx=0.55, max_vy=0.0, max_wz=0.90, gait_floor=0.30,
                          transport=model),
            config=config)

    # The veto, on the four speeds that run actually commanded. NONE of them is safe as
    # executed, and on this transport all four ARE the same command.
    for asked in (0.050, 0.052, 0.101, 0.103):
        assert not planner(transport).is_feasible(
            (0.0, 0.0, 0.0), (asked, 0.0, 0.0), [bin_072]), (
                f"a {asked:.3f} m/s crawl was validated on a robot that will walk it at "
                f"0.30 m/s")
    # The control, and it is not all four: at 0.72 m a proportional robot's own veto
    # already refuses 0.101 and 0.103, because 0.25 m of travel leaves 0.018 m of a
    # 0.12 m hard gap. The two the crawl was actually made of clear it comfortably —
    # which is the whole point. The transport is what turns those two into a refusal,
    # not the geometry, and a test that asserted all four would have proved nothing
    # about the two that matter.
    for safe in (0.050, 0.052):
        assert planner(PROPORTIONAL).is_feasible(
            (0.0, 0.0, 0.0), (safe, 0.0, 0.0), [bin_072]), safe
    for refused_anyway in (0.101, 0.103):
        assert not planner(PROPORTIONAL).is_feasible(
            (0.0, 0.0, 0.0), (refused_anyway, 0.0, 0.0), [bin_072]), refused_anyway

    # And the planner's own command, from the state that run was in: a hold that says
    # which of the two kinds of hold it is.
    command = planner(transport).plan((0.0, 0.0, 0.0), (4.0, 0.0), (0.10, 0.0, 0.0),
                                      [bin_072], control_dt=0.1, now=0.0)
    assert command.is_stop and command.reason == "hold", command
    assert command.transport_refusal is not None, command
    assert "0.300 m/s" in command.transport_refusal, command.transport_refusal
    assert "{0, 0.300} m/s" in command.transport_refusal, command.transport_refusal

    # The control: the same tick on a transport that honours the magnitude walks.
    proportional = planner(PROPORTIONAL).plan((0.0, 0.0, 0.0), (4.0, 0.0),
                                              (0.10, 0.0, 0.0), [bin_072],
                                              control_dt=0.1, now=0.0)
    assert proportional.vx > 0.0 and proportional.transport_refusal is None, proportional


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lite3_bindings: {len(tests)}/{len(tests)} passed")
