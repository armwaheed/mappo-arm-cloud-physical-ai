#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the arrival spin, the surrender shake, and the two vendor actions.

Both gestures are state machines over measured heading, so all of this runs with no robot.
What earns a test is what hardware would otherwise have taught: that a full revolution
crosses the +/-pi discontinuity, that the two gestures are distinguishable rather than one
being a shorter version of the other, and that neither one travels.

⛔ THE VENDOR CANNED ACTIONS ARE TESTED FOR THE OPPOSITE PROPERTY. `backflip` and
`twist-jump` are single opcodes whose effect NOBODY IN THIS REPOSITORY HAS OBSERVED, so
there is no state machine to test and no measured number to check one against. What is left
is the only thing that can be checked without a robot, and it happens to be the thing that
matters: that the right four bytes go out in the right order, and that every path which
could put them on the wire without a human deciding to is closed. Those tests are the only
safety layer these two kinds have, because there is no second one downstream.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import flourish
from flourish import (
    ARRIVAL_KINDS,
    BACKFLIP_CODE,
    BACKFLIP_TRAVEL_M,
    CARPET_BACKFLIP_CODE,
    END_ACTION_CODE,
    LEG_TOLERANCE_RAD,
    OPERATOR_ONLY_KINDS,
    SHAKE_COUNT,
    SHAKE_SWEEP_RAD,
    STALL_WINDOW_S,
    TWIST_JUMP_CODE,
    UNKNOWNS,
    VENDOR_ACTIONS,
    Flourish,
    Refusal,
    action_brief,
    action_packets,
    describe,
    fire,
    vendor_packet,
)
from reverse_along_path import wrap_pi

YAW_RAD_S = 0.8563


def _turn(gesture, *, hz=10.0, delivered_rad_s=YAW_RAD_S, start_yaw=0.0, max_ticks=4000):
    """Run the gesture against a robot that delivers exactly what it is commanded.

    Integrates the command back into a heading, which is the only honest way to test a
    controller that closes its own loop: feeding it the heading it wants to see would pass
    a controller that ignored heading entirely.
    """
    yaw, now, dt = start_yaw, 0.0, 1.0 / hz
    signs, swept = [], 0.0
    while max_ticks > 0:
        max_ticks -= 1
        wz = gesture.step(yaw, now)
        if wz is None:
            return yaw, now, signs, swept
        signs.append(1 if wz > 0 else -1)
        step = math.copysign(delivered_rad_s, wz) * dt
        swept += abs(step)
        yaw = wrap_pi(yaw + step)
        now += dt
    raise AssertionError("the gesture never finished")


# ── the spin ────────────────────────────────────────────────────────────────
def test_the_spin_turns_a_full_circle():
    gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    _, _, signs, swept = _turn(gesture)
    assert abs(swept - 2 * math.pi) < LEG_TOLERANCE_RAD + 0.1, swept
    assert set(signs) == {1}, "a spin is one direction throughout"


def test_the_spin_survives_the_pi_discontinuity():
    """A full revolution passes through the wrap TWICE from some start headings. Comparing
    against the start reads as no rotation there, and the robot spins to its timeout."""
    for start in (0.0, math.pi / 2, math.pi - 0.05, -math.pi + 0.05, 2.5, -2.5):
        gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
        _, _, _, swept = _turn(gesture, start_yaw=start)
        assert abs(swept - 2 * math.pi) < LEG_TOLERANCE_RAD + 0.1, (start, swept)


def test_the_spin_ends_where_it_started():
    """It is fired at the end of a run nobody is steering any more, so it must not leave
    the robot on a new heading for whoever picks it up next."""
    gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    yaw, _, _, _ = _turn(gesture, start_yaw=0.7)
    assert abs(wrap_pi(yaw - 0.7)) < 0.25, yaw


def test_either_direction_spins_the_same_amount():
    """Which way it turns is a CLEARANCE choice, not a distance one."""
    swept = []
    for sign in (+1, -1):
        gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S, turn_sign=sign)
        _, _, signs, s = _turn(gesture)
        swept.append(s)
        assert set(signs) == {sign}
    assert abs(swept[0] - swept[1]) < 0.05, swept


# ── the shake ───────────────────────────────────────────────────────────────
def test_the_shake_alternates_and_is_not_just_a_short_spin():
    """The two gestures have to be told apart at a glance, and from behind. A shorter spin
    would not be: it is the same motion, and an operator cannot judge 'less than a full
    turn' on a robot they are looking at from one side."""
    gesture = Flourish(Flourish.SHAKE, yaw_speed_rad_s=YAW_RAD_S)
    _, _, signs, _ = _turn(gesture)
    assert set(signs) == {1, -1}, "a shake must go both ways"
    changes = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    assert changes == SHAKE_COUNT * 2 - 1, changes


def test_the_shake_ends_on_the_heading_it_began_on():
    """The first and last half-sweeps are half width for exactly this reason; without them
    the gesture drifts a full sweep to one side and leaves the robot facing wrong."""
    gesture = Flourish(Flourish.SHAKE, yaw_speed_rad_s=YAW_RAD_S)
    yaw, _, _, _ = _turn(gesture, start_yaw=-1.2)
    assert abs(wrap_pi(yaw + 1.2)) < SHAKE_SWEEP_RAD, yaw


def test_the_shake_is_much_shorter_than_the_spin():
    """It fires on surrender, when an operator wants the robot to stop being busy."""
    spin = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    shake = Flourish(Flourish.SHAKE, yaw_speed_rad_s=YAW_RAD_S)
    _, spin_s, _, _ = _turn(spin)
    _, shake_s, _, _ = _turn(shake)
    assert shake_s < spin_s, (shake_s, spin_s)


# ── neither gesture travels ─────────────────────────────────────────────────
def test_neither_gesture_ever_commands_a_linear_velocity():
    """`step` returns a YAW command and nothing else, which is the property that makes
    these safe to fire automatically at the end of a run nobody is steering."""
    for kind in (Flourish.SPIN, Flourish.SHAKE):
        gesture = Flourish(kind, yaw_speed_rad_s=YAW_RAD_S)
        yaw, now = 0.0, 0.0
        while True:
            wz = gesture.step(yaw, now)
            if wz is None:
                break
            assert isinstance(wz, float), wz   # one scalar: yaw. No vx, no vy, ever.
            yaw = wrap_pi(yaw + math.copysign(YAW_RAD_S, wz) * 0.1)
            now += 0.1


# ── the abandonments ────────────────────────────────────────────────────────
def test_a_robot_that_is_not_turning_is_abandoned_not_commanded():
    gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    now = 0.0
    try:
        while now < STALL_WINDOW_S * 3:
            gesture.step(0.0, now)      # heading never changes
            now += 0.1
    except Refusal as refusal:
        assert "stopped moving" in str(refusal), refusal
    else:
        raise AssertionError("a stalled gesture must be abandoned")


def test_a_gesture_that_outruns_its_measured_budget_is_abandoned():
    gesture = Flourish(Flourish.SPIN, yaw_speed_rad_s=YAW_RAD_S)
    try:
        _turn(gesture, delivered_rad_s=YAW_RAD_S * 0.1, max_ticks=8000)
    except Refusal as refusal:
        assert "outran its budget" in str(refusal), refusal
    else:
        raise AssertionError("a gesture that cannot finish must be abandoned")


def test_it_refuses_to_time_itself_against_an_unmeasured_yaw_speed():
    """An unmeasured speed is an absence, not a zero -- the gait floor's own rule."""
    try:
        Flourish(Flourish.SPIN, yaw_speed_rad_s=0.0)
    except Refusal as refusal:
        assert "axis_primitive_probe" in str(refusal), refusal
    else:
        raise AssertionError("an unmeasured yaw speed must be refused")


def test_an_unknown_gesture_is_refused_rather_than_guessed():
    try:
        Flourish("dance", yaw_speed_rad_s=YAW_RAD_S)
    except Refusal as refusal:
        assert "unknown gesture" in str(refusal), refusal
    else:
        raise AssertionError("an unknown gesture must be refused")


def test_the_plan_says_the_robot_does_not_travel():
    """The property that licenses firing this automatically. If the sentence goes, the
    reason it was safe went with it."""
    for kind in (Flourish.SPIN, Flourish.SHAKE):
        text = describe(kind, YAW_RAD_S)
        assert "travel" in text and "does not move" in text, text
        assert "0.8563" in text, "the plan must quote the MEASURED yaw speed"


def test_the_pose_fields_these_runners_read_actually_exist():
    """⚠️ THE BUG THIS EXISTS FOR. `flourish.perform` and `reverse_along_path.walk` read
    `loco.pose()`, and both were written against the SIBLING UPSTREAM repository, whose
    `Lite3Pose` names the heading `yaw_rad`. This repository names it `yaw`. Nothing
    offline touched a real pose, so it passed every test here and raised
    `AttributeError: 'Lite3Pose' object has no attribute 'yaw_rad'` on a robot that had
    already completed its run.

    The state machines are pure and cannot catch this; only the seam can. So the seam is
    asserted directly: whatever `Lite3Pose` carries, these two files must read that name.
    """
    import ast
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from deep_robotics.lite3.locomotion.lite3_udp_locomotion import Lite3Pose

    fields = set(getattr(Lite3Pose, "__dataclass_fields__", {}))
    assert fields, "Lite3Pose is expected to be a dataclass"

    here = _Path(__file__).resolve().parent
    for name in ("flourish.py", "reverse_along_path.py"):
        tree = ast.parse((here / name).read_text(), filename=name)
        read = {node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id == "pose"}
        assert read, f"{name} no longer reads a pose; delete this test or fix the scan"
        unknown = read - fields
        assert not unknown, (
            f"{name} reads {sorted(unknown)} off a pose, and Lite3Pose has "
            f"{sorted(fields)}. This is the upstream/downstream naming split.")


# ── look: the one gesture that does NOT come home ───────────────────────────
def test_a_look_turns_the_asked_for_angle_and_stays_there():
    """The whole difference from `sweep`. This runs between attempts of a run that ended
    "goal never sighted", so returning to the start heading would leave the next attempt
    facing exactly the direction that saw nothing."""
    for degrees in (90.0, -90.0, 180.0):
        gesture = Flourish(Flourish.LOOK, yaw_speed_rad_s=YAW_RAD_S,
                           look_rad=math.radians(degrees))
        yaw, _, signs, swept = _turn(gesture)
        assert abs(swept - abs(math.radians(degrees))) < LEG_TOLERANCE_RAD + 0.1, swept
        assert len(set(signs)) == 1, "a look is one continuous turn"
        assert abs(wrap_pi(yaw - wrap_pi(math.radians(degrees)))) < 0.25, (degrees, yaw)


def test_the_sign_of_the_angle_picks_the_direction():
    left = Flourish(Flourish.LOOK, yaw_speed_rad_s=YAW_RAD_S, look_rad=math.radians(90))
    right = Flourish(Flourish.LOOK, yaw_speed_rad_s=YAW_RAD_S, look_rad=math.radians(-90))
    assert left.step(0.0, 0.0) > 0
    assert right.step(0.0, 0.0) < 0


def test_a_look_of_zero_is_refused_rather_than_being_a_silent_no_op():
    """It would turn nowhere, see nothing new, and cost an attempt to discover that."""
    try:
        Flourish(Flourish.LOOK, yaw_speed_rad_s=YAW_RAD_S, look_rad=0.0)
    except Refusal as refusal:
        assert "sees nothing new" in str(refusal), refusal
    else:
        raise AssertionError("a zero-degree look must be refused")


# ── the vendor canned actions: the opcodes ──────────────────────────────────
class _FakeSocket:
    """Records datagrams instead of sending them. ``sendto`` is all ``fire`` uses."""

    def __init__(self) -> None:
        self.sent: list = []
        self.bound = None
        self.closed = False

    def bind(self, address) -> None:
        self.bound = address

    def sendto(self, packet: bytes, address) -> None:
        self.sent.append((packet, address))

    def close(self) -> None:
        self.closed = True


def _run_cli(argv: list[str]):
    """``flourish.main(argv)`` with its printing captured. Returns (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = flourish.main(argv)
    return code, out.getvalue(), err.getvalue()


def _argv(kind: str, *, operator_triggered: bool = True, live: bool = False, **changes):
    """A COMPLETE, valid operator invocation for a vendor canned action.

    Keyword names are flags with their dashes as underscores, and passing ``None`` DROPS
    the flag -- which is how "the operator did not state it" is exercised. Complete by
    default, so every refusal test differs from a working invocation in exactly one way and
    a test cannot pass because of some other missing flag.
    """
    values = {"kind": kind, "robot-id": "LITE3-A", "firmware": "V1.0.8", "payload": "none",
              "locomotion-transport": "axis", "axis-profile": "lite3-axis-LITE3-A.json",
              "lane-width-metres": "2.0", "rear-clearance-metres": "2.5",
              "acrobatic-battery-floor-pct": "60", "action-hold-seconds": "4"}
    values.update({name.replace("_", "-"): value for name, value in changes.items()})
    argv = [item for name, value in values.items() if value is not None
            for item in (f"--{name}", str(value))]
    if operator_triggered:
        argv.append("--operator-triggered")
    if live:
        argv += ["--live", "--operator-ready"]
    return argv


def test_the_new_kinds_emit_the_opcodes_the_vendor_gui_sends():
    """The four bytes, computed here independently of the module that builds them.

    Reference GUI `Lite3_All_control_v20153.py:310-320`, `send_simple(code)`. A typo in one
    of these is not a test failure on hardware, it is an unknown opcode executed by a robot
    standing in the room, so the numbers are restated rather than imported.
    """
    assert VENDOR_ACTIONS["backflip"].code == 0x21010502, "后空翻 backflip"
    assert VENDOR_ACTIONS["twist-jump"].code == 0x2101020D, "扭身跳 twist jump"
    assert END_ACTION_CODE == 0x21010C0B, "结束动作 END ACTION"
    for code in (0x21010502, 0x2101020D, 0x21010C0B):
        assert vendor_packet(code) == struct.pack("<3I", code, 0, 0), f"{code:#010x}"
        assert len(vendor_packet(code)) == 12


def test_no_end_action_is_sent_because_the_controller_does_not_send_one():
    """MEASURED, not assumed. The hand controller was packet-captured performing all three
    of these actions and sent NO end-action for any of them, while the state stream showed
    the robot returning to force-control 6 unaided every time. Sending a stop the vendor
    never sends would be inventing behaviour on a robot mid-acrobatic."""
    for kind, action in VENDOR_ACTIONS.items():
        packets = action_packets(action)
        assert [label for label, _, _ in packets] == [kind], packets
        assert [code for _, code, _ in packets] == [action.code]
        assert END_ACTION_CODE not in [code for _, code, _ in packets]


def test_fire_sends_one_datagram_then_holds_for_the_manoeuvre():
    """One opcode, then the operator's `--action-hold-seconds`. The hold outlives the
    manoeuvre rather than separating two packets: a backflip was measured taking 5.4-5.9 s
    to return the robot to force-control 6, and returning sooner closes the socket while
    the robot is still inverted."""
    for action in VENDOR_ACTIONS.values():
        sock, slept = _FakeSocket(), []
        fire(sock, ("10.0.0.1", 43893), action, hold_s=3.5,
             sleep=slept.append, printer=lambda _line: None)
        assert [packet for packet, _ in sock.sent] == [
            struct.pack("<3I", action.code, 0, 0)], sock.sent
        assert {address for _, address in sock.sent} == {("10.0.0.1", 43893)}
        assert slept == [3.5], "exactly one wait, and it is the operator's number"


def test_an_opcode_this_file_does_not_name_is_refused_rather_than_sent():
    """Same rule as `lite3_control_mode_udp` and `lite3_axis_udp`: on this channel an
    unrecognised code is not an error that comes back, it is whatever that code means on
    this firmware, executed by a robot in the room."""
    for code in (0x21010202, 0x21010C05, 0x21010D06, 0x21010C0E, 0x00000000):
        try:
            vendor_packet(code)
        except Refusal as refusal:
            assert "unsupported" in str(refusal), refusal
        else:
            raise AssertionError(f"{code:#010x} was packed without being named")


# ── the vendor canned actions: the room ─────────────────────────────────────
def test_a_missing_rear_clearance_is_refused_and_says_why():
    """⛔ THIS ROBOT HAS ZERO REAR SENSING. There is no rear camera, no ultrasonic and no
    bumper, so nothing in software will ever notice what a backflip is about to travel
    into. The clearance is the operator's tape measure and there is no default."""
    for kind in OPERATOR_ONLY_KINDS:
        code, _, err = _run_cli(_argv(kind, rear_clearance_metres=None))
        assert code == 1, kind
        assert "--rear-clearance-metres" in err, err
        assert "NO REAR SENSING" in err, "the refusal has to say why, not just which flag"


def test_a_rear_clearance_below_the_backflip_travel_is_refused():
    """1.5 m is the vendor's own backward-travel figure. It is a floor to refuse below,
    never a promise about where this robot will stop."""
    for stated in ("0.0", "0.5", "1.49"):
        code, _, err = _run_cli(_argv("backflip", rear_clearance_metres=stated))
        assert code == 1, stated
        assert "--rear-clearance-metres" in err and f"{BACKFLIP_TRAVEL_M:.2f}" in err, err
    code, _, _ = _run_cli(_argv("backflip", rear_clearance_metres="1.5", live=False))
    assert code == 0, "exactly the vendor figure is the floor, not one above it"


def test_the_twist_jump_clearance_floor_is_the_platform_and_not_an_invented_number():
    """Nobody knows how far a twist jump goes, so the floor is the one number that is
    already evidenced here: the robot's own footprint. Inventing a travel figure for it
    would be the exact failure this file exists to avoid."""
    from reverse_along_path import PLATFORM_HALF_DIAGONAL_M
    assert VENDOR_ACTIONS["twist-jump"].rear_clearance_m == 2 * PLATFORM_HALF_DIAGONAL_M
    assert "UNKNOWN" in VENDOR_ACTIONS["twist-jump"].travel
    code, _, err = _run_cli(_argv("twist-jump", rear_clearance_metres="0.4"))
    assert code == 1 and "--rear-clearance-metres" in err, err


def test_every_unknown_precondition_is_refused_rather_than_defaulted():
    """The house rule, applied to the three things nobody here has measured. A default for
    any of them would be this file inventing a precondition and then checking it."""
    for flag in ("acrobatic_battery_floor_pct", "action_hold_seconds", "lane_width_metres"):
        code, _, err = _run_cli(_argv("backflip", **{flag: None}))
        assert code == 1, flag
        assert flag.replace("_", "-") in err, (flag, err)


def test_the_battery_floor_is_the_operators_and_is_fed_to_robot_links_own_gate():
    """⛔ ONE BATTERY GATE, AND IT IS robot_link's. An acrobatic manoeuvre is the highest
    current this robot draws and a brownout mid-flip is a fall -- but the number is the
    operator's, and it is folded into `--battery-abort` rather than becoming a second check
    that somebody can forget to apply."""
    parser = flourish.build_parser()
    args = parser.parse_args(_argv("backflip", acrobatic_battery_floor_pct="65"))
    assert args.battery_abort < 65, "the default abort is the one every probe shares"
    flourish._validate_action(args, VENDOR_ACTIONS["backflip"])
    assert args.battery_abort == 65, "the operator's floor must reach robot_link.preflight"

    lower = parser.parse_args(_argv("backflip", acrobatic_battery_floor_pct="1"))
    default_abort = lower.battery_abort
    flourish._validate_action(lower, VENDOR_ACTIONS["backflip"])
    assert lower.battery_abort == default_abort, "a lower floor must not WEAKEN the gate"


def _stub_robot_link(implementation, order):
    """Swap robot_link's two live entry points for recorders. Returns the restorer.

    Stubbed rather than skipped because the ORDER of the gates is the property under test,
    and the order is the whole of what protects a robot here: there is no state machine
    downstream of these opcodes that could refuse anything later.
    """
    sys.path.insert(0, str(_HERE.parents[2]))
    from deep_robotics.lite3.commissioning import robot_link

    class _Locomotion:
        def shutdown(self):
            order.append("shutdown")

    saved = robot_link.connect, robot_link.preflight
    robot_link.connect = lambda args: robot_link.Link(
        locomotion=_Locomotion(), implementation=implementation)
    robot_link.preflight = lambda link, args, printer=print: order.append("preflight")

    def restore():
        robot_link.connect, robot_link.preflight = saved

    return restore


def test_no_opcode_reaches_a_socket_until_both_gates_have_passed():
    """⛔ THE ORDER IS THE SAFETY PROPERTY. robot_link's preflight -- battery, state age,
    error_state -- and the transport's own vendor state gate both have to have passed
    before the first byte goes out. A backflip fired at a robot that is lying down is a
    backflip fired at a robot that is lying down, whatever the flags said."""
    order, sockets = [], []

    class _Implementation:
        def assert_axis_state_ready(self):
            order.append("gate")

    def _socket_factory(*_args):
        order.append("socket")
        sockets.append(_FakeSocket())
        return sockets[-1]

    restore = _stub_robot_link(_Implementation(), order)
    try:
        args = flourish.build_parser().parse_args(_argv("backflip", live=True))
        flourish._validate_action(args, VENDOR_ACTIONS["backflip"])
        code = flourish.perform_action(args, VENDOR_ACTIONS["backflip"],
                                       printer=lambda _line: None,
                                       socket_factory=_socket_factory, sleep=lambda _s: None)
    finally:
        restore()
    assert code == 0
    assert order == ["preflight", "gate", "socket", "shutdown"], order
    assert [packet for packet, _ in sockets[0].sent] == [
        struct.pack("<3I", BACKFLIP_CODE, 0, 0)], sockets[0].sent
    assert sockets[0].bound == ("0.0.0.0", args.axis_local_port), (
        "the opcode leaves from the same source port the accepted axis commands do, and "
        "the bind doubles as a lock against a run already in progress elsewhere")
    assert sockets[0].closed


def test_a_transport_with_no_state_gate_is_refused_before_a_socket_exists():
    """The gate lives on the axis transport only. Firing an unvalidated acrobatic opcode
    with no state gate at all is worse than not firing it."""
    order = []

    def _socket_factory(*_args):
        raise AssertionError("a socket was opened without a vendor state gate")

    restore = _stub_robot_link(object(), order)   # no assert_axis_state_ready on it
    try:
        args = flourish.build_parser().parse_args(_argv("backflip", live=True))
        flourish._validate_action(args, VENDOR_ACTIONS["backflip"])
        flourish.perform_action(args, VENDOR_ACTIONS["backflip"],
                                printer=lambda _line: None, socket_factory=_socket_factory)
    except Refusal as refusal:
        assert "--locomotion-transport axis" in str(refusal), refusal
    else:
        raise AssertionError("an opcode was sent through a transport with no state gate")
    finally:
        restore()
    assert order == ["preflight", "shutdown"], order


# ── not wired to arrival ────────────────────────────────────────────────────
def test_no_arrival_path_can_name_a_travelling_kind():
    """⛔ THE SAFETY PROPERTY, PINNED FROM THE MISSION END.

    `mission.play_flourish` fires a gesture when a run arrives, gives up, or loses sight of
    its goal -- by which point nobody is necessarily watching the robot. Every kind it can
    reach must be one that keeps the robot's centre where it is. This asserts the two halves
    separately, because they fail differently: a new `play_flourish` call site with a
    travelling kind, and a travelling kind's name appearing anywhere on that path at all.

    ⚠️ READ THIS BEFORE TRUSTING IT. On 2026-09-07 the operator had the rear look disabled
    and `mission.py` gained a DIRECT arrival path to a travelling kind. This test went on
    passing, unchanged, because that path names the kind through `args.flip_kind` rather
    than as a literal -- so both halves above stayed true while the property in the title
    stopped holding. It is still worth keeping: a hardcoded travelling kind is a real
    regression and this catches it. But it is no longer the whole boundary, and
    `test_the_direct_arrival_flip_is_gated_and_carries_the_only_remaining_check` is the
    half that covers what replaced it. Neither is sufficient alone.
    """
    import ast

    visual_nav = _HERE.parents[0] / "visual_nav"
    for name in ("mission.py", "venue_run.py"):
        source = (visual_nav / name).read_text()
        for kind in OPERATOR_ONLY_KINDS:
            assert kind not in source, (
                f"{name} names {kind!r}, which travels and must never be fired by an "
                f"end-of-run path that nobody is watching")

    tree = ast.parse((visual_nav / "mission.py").read_text(), filename="mission.py")
    fired = [node.args[1].value for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "play_flourish"
             and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)]
    assert fired, "mission.py no longer calls play_flourish; fix this scan, don't delete it"
    assert set(fired) <= set(ARRIVAL_KINDS), (
        f"mission.py fires {sorted(set(fired))} automatically and only "
        f"{sorted(ARRIVAL_KINDS)} keep the robot's centre where it is")


def test_the_direct_arrival_flip_is_gated_and_carries_the_only_remaining_check():
    """⛔⛔ WHAT REPLACED THE REAR LOOK, pinned at the source.

    Until 2026-09-07 the route from an arrival to a kind that TRAVELS ran through
    `look_behind`, which refused until a completed outward turn, a completed return turn
    and unanimous fresh frames from the run's own detector said the space behind was
    clear. The operator had that disabled -- twice asked for, and granted -- so the route
    is now direct.

    That leaves exactly ONE check on the space a ~1.5 m backward manoeuvre travels into,
    on a platform with no rear camera, no ultrasonic and no bumper: the operator's
    `--flip-rear-clearance-metres` tape measurement, enforced by `check_rear`. If that
    argument ever stops being passed on this path, nothing anywhere refuses and nothing
    else in this suite notices -- the literal-name test above kept passing right through
    the change that removed the look.

    So this asserts three things about the direct path: it is gated on `flip_on_arrival`,
    it passes the rear clearance, and the parser refuses when the clearance is absent.
    """
    import ast

    source = (_HERE.parents[0] / "visual_nav" / "mission.py").read_text()
    tree = ast.parse(source, filename="mission.py")

    # The dynamic gesture calls -- the ones a literal-name scan cannot see.
    dynamic = [node for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               and getattr(node.func, "id", None) == "run_gesture"
               and len(node.args) > 2
               and not isinstance(node.args[2], ast.Constant)]
    assert dynamic, (
        "no dynamic run_gesture call in mission.py. If the direct arrival flip was "
        "removed and the rear look restored, delete this test and say so; do not leave "
        "it passing vacuously")

    for call in dynamic:
        rendered = ast.dump(call)
        assert "rear-clearance-metres" in rendered, (
            "a dynamically-named gesture is fired without --rear-clearance-metres. With "
            "the rear look gone that argument is the ONLY thing standing between this "
            "robot and whatever is behind it")
        assert "acrobatic-battery-floor-pct" in rendered, "no battery floor on the flip path"
        assert "action-hold-seconds" in rendered, (
            "no hold seconds: returning before the firmware finishes hands control back "
            "mid-manoeuvre")

    # Gated, not unconditional. `flip_on_arrival` has to appear in a test guarding it.
    guards = [node for node in ast.walk(tree)
              if isinstance(node, ast.If) and "flip_on_arrival" in ast.dump(node.test)
              and any(call in ast.walk(node) for call in dynamic)]
    assert guards, "the direct flip is not gated on flip_on_arrival"

    # And the parser refuses a flip armed without the clearance, rather than formatting
    # None with :.4f at the end of a run on a robot standing at its goal.
    assert "--flip-rear-clearance-metres" in source
    assert "if value is None" in source, (
        "the parser no longer checks the flip values for None; they were validated inside "
        "look_behind, which this path bypasses")


def test_a_mission_shaped_invocation_of_a_travelling_kind_is_refused():
    """⛔ THE SAME PROPERTY, PINNED FROM THE FLOURISH END, and it is the half that holds if
    somebody later adds the call site. The command `mission.flourish_command` builds is the
    real one, so this breaks if that command ever grows the flags these kinds need."""
    sys.path.insert(0, str(_HERE.parents[0] / "visual_nav"))
    import mission

    args = argparse.Namespace(flourish=True, robot_id="LITE3-A", firmware="V1.0.8",
                              payload="none", flourish_lane_width=3.0)
    for kind in OPERATOR_ONLY_KINDS:
        for drive in ([], ["--live"]):
            argv = mission.flourish_command(drive, kind, args)
            assert argv is not None and "--operator-triggered" not in argv, argv
            code, _, err = _run_cli(argv[2:])   # drop the interpreter and the script path
            assert code == 1, (kind, drive, err)
            assert "--operator-triggered" in err, err


def test_the_dry_run_of_a_travelling_kind_opens_nothing_and_sends_nothing():
    """A plan is text. Nothing without `--live` may reach a socket, and a `--kind backflip`
    dry run is the one an operator will run first and read."""
    reached = []
    original = flourish.perform_action
    flourish.perform_action = lambda *args, **kwargs: reached.append(args) or 0
    try:
        for kind in OPERATOR_ONLY_KINDS:
            code, out, err = _run_cli(_argv(kind))
            assert code == 0, (kind, err)
            assert not reached, "a run without --live reached the send path"
            assert "nothing was sent" in out, out

        # And the complement, so the assertion above cannot pass by never running at all.
        code, _, err = _run_cli(_argv("backflip", live=True))
        assert code == 0 and len(reached) == 1, (code, err, reached)
    finally:
        flourish.perform_action = original


def test_live_without_operator_ready_is_refused():
    code, _, err = _run_cli([*_argv("backflip"), "--live"])
    assert code == 1 and "--operator-ready" in err, err


# ── the existing kinds are untouched ────────────────────────────────────────
def test_an_arrival_kind_is_never_asked_for_any_of_the_new_flags():
    """`--kind spin` must behave exactly as it did. It runs here against a profile with no
    measured yaw speed, so reaching THAT refusal is the proof it went down the turn path
    and asked for no clearance, no battery floor and no hold on the way."""
    code, out, err = _run_cli([
        "--robot-id", "LITE3-A", "--firmware", "V1.0.8", "--payload", "none",
        "--locomotion-transport", "axis",
        "--axis-profile", str(_HERE / "lite3_axis_profile.example.json"),
        "--kind", "spin", "--lane-width-metres", "2.0"])
    assert code == 1, (code, out, err)
    assert "no measured yaw speed" in err, err
    assert "VENDOR CANNED ACTION" not in out, "a spin must not print the acrobatic brief"
    for flag in ("--rear-clearance-metres", "--operator-triggered", "--action-hold-seconds"):
        assert flag not in err, f"a spin was asked for {flag}"


def test_the_two_sets_of_kinds_are_disjoint_and_cover_every_choice():
    assert not set(ARRIVAL_KINDS) & set(OPERATOR_ONLY_KINDS)
    assert set(OPERATOR_ONLY_KINDS) == set(VENDOR_ACTIONS)
    choices = flourish.build_parser()._option_string_actions["--kind"].choices
    assert set(choices) == set(ARRIVAL_KINDS) | set(OPERATOR_ONLY_KINDS), choices


def test_describe_refuses_a_canned_action_rather_than_printing_the_shake_plan():
    """⚠️ THE ONE SENTENCE THAT MUST NEVER BE PRINTED ABOUT A BACKFLIP is `describe`'s
    "travel none -- the centre does not move". Falling through its final `else` would print
    exactly that over an opcode that travels 1.5 m backward."""
    for kind in OPERATOR_ONLY_KINDS:
        try:
            describe(kind, YAW_RAD_S)
        except Refusal as refusal:
            assert "vendor canned action" in str(refusal), refusal
        else:
            raise AssertionError(f"describe({kind!r}) printed a turn's plan")


# ── the brief says what is not known ────────────────────────────────────────
def test_the_brief_says_the_opcodes_have_never_been_sent_and_lists_every_unknown():
    """Constraint: where a precondition is unknown, refuse or say so LOUDLY. The operator
    reading this cannot read the source, and this is the only place they are told."""
    parser = flourish.build_parser()
    for kind, action in VENDOR_ACTIONS.items():
        args = parser.parse_args(_argv(kind))
        text = action_brief(action, args)
        assert "NEITHER OPCODE HAS EVER BEEN SENT BY THIS REPOSITORY" in text, text
        assert "NO REAR SENSING" in text and f"{action.code:#010x}" in text
        assert f"{END_ACTION_CODE:#010x}" not in text, (
            "the stop opcode is no longer sent, so it must not appear in the plan")
        assert action.travel.split(" --")[0] in text
        squashed = " ".join(text.split())
        for unknown in UNKNOWNS:
            head = " ".join(unknown.split()[:6])
            assert head in squashed, f"the brief dropped an unknown: {head}"


def test_the_docstring_says_which_kinds_travel_and_which_do_not():
    """⛔ THE REASONING THAT LICENSED FIRING A GESTURE UNATTENDED IS IN THE DOCSTRING, and
    it is only true of the turns. If the travel table or the unattended warning is deleted,
    the next reader inherits "these are safe to fire automatically" applied to a backflip."""
    doc = flourish.__doc__
    assert "keep the robot's centre where it" in doc, "the original argument must survive"
    assert "spin, shake, sweep, look" in doc, "the in-place kinds must still be named"
    assert f"TRAVELS ~{BACKFLIP_TRAVEL_M:.1f} m BACKWARD" in doc
    assert "NOT SAFE TO FIRE UNATTENDED" in doc
    assert "ZERO REAR SENSING" in doc
    assert "travel UNKNOWN" in doc, "the twist jump's travel must not be claimed"


def test_the_opcodes_are_not_reachable_without_the_travelling_kinds():
    """A last sweep of the source: the three opcodes may appear only in the constants that
    name them, so no other path in this file can put one on the wire."""
    import ast

    tree = ast.parse((_HERE / "flourish.py").read_text(), filename="flourish.py")
    codes = [BACKFLIP_CODE, CARPET_BACKFLIP_CODE, TWIST_JUMP_CODE, END_ACTION_CODE]
    literals = [node for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and node.value in codes]
    assigned = [node for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and node.value.value in codes]
    assert len(literals) == len(assigned) == 4, (
        "an opcode literal appears somewhere other than its own constant; every send has "
        "to go through vendor_packet, which refuses a code it does not name")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"flourish: {len(tests)}/{len(tests)} passed")
