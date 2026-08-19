#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Receive-only decoder for the Lite3 motion host's high-level UDP state stream.

The motion host at ``192.168.1.120`` streams its high-level state to one configured
destination address — ``ip`` in ``~/jy_exe/conf/network.toml`` on the motion host, factory
default ``192.168.1.102``, target port ``43897``.  This module binds that port and decodes
what arrives.  It is the whole read-only half of the vendor interface, without ROS 2.

**This module cannot move the robot.** It has no destination address, no command encoder,
and no send path; ``test_lite3_state_probe.py`` asserts that structurally, so adding one
fails the suite rather than passing review.  ``Lite3_ROS``'s ``jetson2motion`` is *not* a
read-only agent — its single executable owns a receiver *and* a sender aimed at the motion
host's command port — which is why commissioning uses this instead.

The four frame layouts come from ``Lite3_ROS`` ``src/transfer/include/protocol.hpp``
(branch ``ros2-foxy``).  Frames are dispatched by length, exactly as the vendor bridge
does, then confirmed by their command code.  Sizes and field offsets here were taken from
a compiler, not read off the header: see ``test_lite3_state_probe.py``.

What a passive capture settles, with the operator driving on the vendor remote and this
process only listening:

* **Whether the link is alive at all**, and at what rate each frame type arrives.
* **The mode state machine.** ``robot_basic_state``/``robot_gait_state``/
  ``robot_policy_state``/``robot_motion_state`` change as the operator moves the robot
  between manual, standing and high-level navigation. Watching those numbers *is* the
  empirical answer to "what is the AUTO transition", and it costs no transmitted byte.
* **Gait floor and actuator gain.** ``HandleState`` carries the velocity the firmware
  itself derived from the remote's sticks (``goal_vel_forward``) alongside ``RobotState``'s
  measured ``vel_body``. Pairing them measures both numbers while the *vendor's own*
  controller is the only thing commanding the legs.
* **The angular-velocity unit.** ``Lite3_ROS`` copies ``rpy_vel`` into a ROS field
  documented as rad/s without converting, while ``rpy`` is degrees. the report divides measured yaw
  change by the reported rate, so the unit is decided by the installed firmware rather
  than assumed.
* **Battery.** ``RobotState.battery_level`` is in this stream. Half of the health gate in
  issue #13 needs no vendor bridge at all; the 12 motor temperatures genuinely are absent.

Nothing here establishes safe *motion* semantics — heartbeat, the clean stop path, or the
approved return to manual. Those remain vendor questions. See ``../../../SAFETY.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import struct
import sys
import time
from collections.abc import Callable

# ``code`` values the motion host stamps on each frame, from the vendor bridge's parsers.
ROBOT_STATE_CODE = 2305
JOINT_STATE_CODE = 2306
HANDLE_STATE_CODE = 2309
IMU_DATA_CODE = 0x010901

# ``int code, int size, int cons_code`` under ``#pragma pack(push, 4)``.
_HEADER = struct.Struct("<3i")

# ``RobotState`` is declared *outside* the header's pack(4) region, so it keeps natural
# alignment: the three leading ints are followed by 4 bytes of padding before ``rpy``, and
# ``is_charging``/``zero_position_flag`` are each followed by padding. Explicit ``x`` bytes
# reproduce that; a naive field-by-field format would silently shear every double.
_ROBOT_STATE = struct.Struct("<3i4x18dI?3xIidi??2x2d")
_JOINT_STATE = struct.Struct("<12d")
_HANDLE_STATE = struct.Struct("<6d")
_IMU_DATA = struct.Struct("<I9f")

JOINT_NAMES = (
    "LF_Joint", "LF_Joint_1", "LF_Joint_2",
    "RF_Joint", "RF_Joint_1", "RF_Joint_2",
    "LB_Joint", "LB_Joint_1", "LB_Joint_2",
    "RB_Joint", "RB_Joint_1", "RB_Joint_2",
)

#: Frame length -> (kind, expected ``code``). The vendor bridge dispatches on length alone;
#: this also checks the code, so a same-length frame from another sender is dropped rather
#: than decoded into plausible-looking nonsense.
FRAME_KINDS = {
    _HEADER.size + _ROBOT_STATE.size: ("robot_state", ROBOT_STATE_CODE),
    _HEADER.size + _JOINT_STATE.size: ("joint_state", JOINT_STATE_CODE),
    _HEADER.size + _HANDLE_STATE.size: ("handle_state", HANDLE_STATE_CODE),
    _HEADER.size + _IMU_DATA.size: ("imu", IMU_DATA_CODE),
}

#: Fields whose transitions are worth printing: the control-mode state machine.
MODE_FIELDS = ("robot_basic_state", "robot_gait_state", "robot_policy_state",
               "robot_motion_state")

DEFAULT_PORT = 43897
#: A remote command older than this is not paired with a measurement.
HANDLE_STALE_S = 0.2
DEGREES_PER_RADIAN = 57.29577951308232


class DecodeError(ValueError):
    """A datagram did not match any known Lite3 high-level frame."""


def decode_frame(payload: bytes) -> dict:
    """Decode one datagram into a plain dict, or raise :class:`DecodeError`.

    The returned dict always carries ``kind`` and the raw ``code``/``cons_code`` so an
    unexpected variant is visible in a recording rather than smoothed away.
    """
    kind_and_code = FRAME_KINDS.get(len(payload))
    if kind_and_code is None:
        raise DecodeError(f"no Lite3 frame is {len(payload)} bytes long")
    kind, expected_code = kind_and_code

    code, size, cons_code = _HEADER.unpack_from(payload, 0)
    if code != expected_code:
        raise DecodeError(
            f"{len(payload)}-byte frame carries code {code}, not the {kind} code "
            f"{expected_code}"
        )

    frame = {"kind": kind, "code": code, "size": size, "cons_code": cons_code}
    frame.update(_DECODERS[kind](payload, _HEADER.size))
    return frame


def _decode_robot_state(payload: bytes, offset: int) -> dict:
    values = _ROBOT_STATE.unpack_from(payload, offset)
    return {
        "robot_basic_state": values[0],
        "robot_gait_state": values[1],
        "robot_policy_state": values[2],
        # ``rpy`` is degrees: the vendor bridge divides by 180/pi on the way to ROS.
        "rpy_deg": list(values[3:6]),
        # Units unconfirmed by the vendor; the report measures which one it is.
        "rpy_vel": list(values[6:9]),
        "xyz_acc": list(values[9:12]),
        "pos_world": list(values[12:15]),
        "vel_world": list(values[15:18]),
        "vel_body": list(values[18:21]),
        "touch_down_and_stair_trot": values[21],
        "is_charging": values[22],
        "error_state": values[23],
        "robot_motion_state": values[24],
        "battery_level": values[25],
        "task_state": values[26],
        "is_robot_need_move": values[27],
        "zero_position_flag": values[28],
        "ultrasound": list(values[29:31]),
    }


def _decode_joint_state(payload: bytes, offset: int) -> dict:
    values = _JOINT_STATE.unpack_from(payload, offset)
    # The vendor bridge negates every joint on the way to ROS; positions are reported here
    # in the wire's own sign so a capture stays comparable with the vendor's own tools.
    return {"joint_positions": dict(zip(JOINT_NAMES, values))}


def _decode_handle_state(payload: bytes, offset: int) -> dict:
    values = _HANDLE_STATE.unpack_from(payload, offset)
    return {
        "left_axis_forward": values[0],
        "left_axis_side": values[1],
        "right_axis_yaw": values[2],
        "goal_vel_forward": values[3],
        "goal_vel_side": values[4],
        "goal_vel_yaw": values[5],
    }


def _decode_imu(payload: bytes, offset: int) -> dict:
    values = _IMU_DATA.unpack_from(payload, offset)
    return {
        "imu_timestamp": values[0],
        "angle_deg": list(values[1:4]),
        "angular_velocity": list(values[4:7]),
        "acc": list(values[7:10]),
    }


_DECODERS: dict[str, Callable[[bytes, int], dict]] = {
    "robot_state": _decode_robot_state,
    "joint_state": _decode_joint_state,
    "handle_state": _decode_handle_state,
    "imu": _decode_imu,
}


def unwrap_degrees(previous: float, current: float) -> float:
    """Return ``current - previous`` folded into (-180, 180].

    Yaw is reported in degrees and wraps. Differencing it raw turns one wrap into a
    ~360 deg/sample spike, which would dominate any rate estimate built from it.
    """
    delta = (current - previous) % 360.0
    if delta > 180.0:
        delta -= 360.0
    return delta


class ProbeStatistics:
    """Accumulates what a passive capture can settle. Holds no socket."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.undecoded = 0
        self.first_seen: float | None = None
        self.last_seen: float | None = None
        self.mode_transitions: list[tuple[float, dict[str, int]]] = []
        self.battery: float | None = None
        self.is_charging: bool | None = None
        self.error_states: dict[int, int] = {}
        self._modes: dict[str, int] | None = None
        self._yaw: tuple[float, float] | None = None
        self._yaw_rate_pairs: list[tuple[float, float]] = []
        self._handle: tuple[float, dict] | None = None
        self._command_pairs: list[tuple[float, float, float, float]] = []

    def observe(self, frame: dict, timestamp: float) -> None:
        kind = frame["kind"]
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self.first_seen is None:
            self.first_seen = timestamp
        self.last_seen = timestamp

        if kind == "handle_state":
            self._handle = (timestamp, frame)
        elif kind == "robot_state":
            self._observe_robot_state(frame, timestamp)

    def observe_undecoded(self) -> None:
        self.undecoded += 1

    def _observe_robot_state(self, frame: dict, timestamp: float) -> None:
        self.battery = frame["battery_level"]
        self.is_charging = frame["is_charging"]
        error = frame["error_state"]
        self.error_states[error] = self.error_states.get(error, 0) + 1

        modes = {name: frame[name] for name in MODE_FIELDS}
        if modes != self._modes:
            self.mode_transitions.append((timestamp, modes))
            self._modes = modes

        yaw = frame["rpy_deg"][2]
        if self._yaw is not None:
            previous_time, previous_yaw = self._yaw
            interval = timestamp - previous_time
            # One sample interval is the resolution floor of a finite difference; below a
            # few milliseconds the quotient is dominated by arrival jitter, not by motion.
            if interval >= 0.005:
                measured = unwrap_degrees(previous_yaw, yaw) / interval
                self._yaw_rate_pairs.append((measured, frame["rpy_vel"][2]))
        self._yaw = (timestamp, yaw)

        # Pair only against a command still being transmitted. Latching the last
        # handle frame forever would keep a stale command paired with fresh measurements
        # after the operator released the remote, which biases the gait floor downward --
        # the direction that would make the robot look like it walks slower than it can.
        if self._handle is not None:
            handle_time, handle = self._handle
            if timestamp - handle_time <= HANDLE_STALE_S:
                self._command_pairs.append((
                    handle["goal_vel_forward"], frame["vel_body"][0],
                    handle["goal_vel_yaw"], frame["rpy_deg"][2],
                ))

    @property
    def duration(self) -> float:
        if self.first_seen is None or self.last_seen is None:
            return 0.0
        return self.last_seen - self.first_seen

    def rates_hz(self) -> dict[str, float]:
        if self.duration <= 0.0:
            return {}
        return {kind: count / self.duration for kind, count in sorted(self.counts.items())}

    def yaw_rate_unit(self) -> dict | None:
        """Compare measured yaw change against the reported ``rpy_vel`` z field.

        Samples below the threshold are dropped, not clamped: near standstill both terms
        are noise and their ratio is unbounded, so including them would let a stationary
        robot decide the unit.
        """
        moving = [(measured, reported) for measured, reported in self._yaw_rate_pairs
                  if abs(measured) >= 5.0 and abs(reported) >= 1e-3]
        if not moving:
            return None
        ratios = sorted(measured / reported for measured, reported in moving)
        median = ratios[len(ratios) // 2]
        if abs(median - 1.0) < 0.25:
            verdict = "rpy_vel is degrees/s (matches rpy, which is degrees)"
        elif abs(median - DEGREES_PER_RADIAN) < 12.0:
            verdict = "rpy_vel is radians/s"
        else:
            verdict = "inconclusive — neither degrees/s nor radians/s; do not use it"
        return {"samples": len(moving), "median_ratio": median, "verdict": verdict}

    def command_response(self, bin_width: float = 0.05) -> list[dict]:
        """Bin measured forward speed by the speed the *remote* asked the firmware for.

        The gait floor is the lowest commanded bin whose measured speed is a walk rather
        than a shuffle; the actuator gain is measured/commanded at the demo envelope. Both
        come out of a capture in which this process transmitted nothing.
        """
        bins: dict[int, list[tuple[float, float]]] = {}
        for commanded, measured, _yaw_cmd, _yaw in self._command_pairs:
            if commanded <= 0.0:
                continue
            bins.setdefault(int(commanded / bin_width), []).append((commanded, measured))
        rows = []
        for index in sorted(bins):
            pairs = bins[index]
            mean_commanded = sum(pair[0] for pair in pairs) / len(pairs)
            mean_measured = sum(pair[1] for pair in pairs) / len(pairs)
            rows.append({
                "commanded_m_s": mean_commanded,
                "measured_m_s": mean_measured,
                "gain": mean_measured / mean_commanded if mean_commanded else 0.0,
                "samples": len(pairs),
            })
        return rows


def _format_report(statistics: ProbeStatistics) -> str:
    lines = ["", "=" * 74, "Lite3 passive state capture", "=" * 74]

    total = sum(statistics.counts.values())
    if total == 0:
        lines += [
            "",
            "NO FRAMES RECEIVED.",
            "",
            "The motion host streams to exactly one address. Check, in this order:",
            "  1. This host's address is the one in ~/jy_exe/conf/network.toml on the",
            "     motion host (factory default 192.168.1.102, target_port 43897).",
            "  2. This host is on 192.168.1.0/24 and can ping 192.168.1.120.",
            "  3. No local firewall is dropping inbound UDP on this port.",
            "",
        ]
        return "\n".join(lines)

    lines.append("")
    lines.append(f"{total} frames over {statistics.duration:.1f} s")
    for kind, rate in statistics.rates_hz().items():
        lines.append(f"  {kind:<14} {statistics.counts[kind]:>7} frames  {rate:7.1f} Hz")
    if statistics.undecoded:
        lines.append(f"  {'undecoded':<14} {statistics.undecoded:>7} frames")

    lines += ["", "Health"]
    if statistics.battery is None:
        lines.append("  no robot_state frame arrived, so battery is unknown")
    else:
        lines.append(f"  battery_level  {statistics.battery:.1f}    charging: "
                     f"{statistics.is_charging}")
        lines.append("  motor temperatures are ABSENT from this stream — vendor question")
    if statistics.error_states:
        observed = ", ".join(f"{state} x{count}"
                             for state, count in sorted(statistics.error_states.items()))
        lines.append(f"  error_state    {observed}")

    lines += ["", "Control-mode transitions"]
    if not statistics.mode_transitions:
        lines.append("  none observed")
    else:
        base = statistics.first_seen or 0.0
        for timestamp, modes in statistics.mode_transitions:
            fields = "  ".join(f"{name.replace('robot_', '')}={value}"
                               for name, value in modes.items())
            lines.append(f"  t+{timestamp - base:6.2f}s  {fields}")
        lines.append("  Record which operator action produced each transition; that is the")
        lines.append("  AUTO/manual answer for THIS firmware.")

    unit = statistics.yaw_rate_unit()
    lines += ["", "Angular-velocity unit (rpy_vel z vs measured yaw change)"]
    if unit is None:
        lines.append("  not enough yaw motion to decide — turn the robot on the remote")
    else:
        lines.append(f"  median ratio {unit['median_ratio']:.2f} over {unit['samples']} "
                     f"moving samples")
        lines.append(f"  -> {unit['verdict']}")

    rows = statistics.command_response()
    lines += ["", "Remote-commanded vs measured forward speed"]
    if not rows:
        lines.append("  no forward command seen — drive the robot on the vendor remote")
    else:
        lines.append("   commanded   measured    gain   samples")
        for row in rows:
            lines.append(f"    {row['commanded_m_s']:7.3f}   {row['measured_m_s']:8.3f}  "
                         f"{row['gain']:6.2f}   {row['samples']:>7}")
        lines.append("  Lowest bin that sustains a gait -> --gait-floor;")
        lines.append("  gain at the demo envelope -> --actuator-gain. Record the robot ID.")

    lines += ["", "Not settled by a passive capture: the heartbeat, the approved AUTO",
              "transition, the clean stop path, and motor temperatures.", ""]
    return "\n".join(lines)


def run_probe(sock: socket.socket, seconds: float, statistics: ProbeStatistics,
              record: Callable[[dict], None] | None = None,
              clock: Callable[[], float] = time.monotonic) -> ProbeStatistics:
    """Read datagrams until ``seconds`` elapse. Never transmits.

    ``clock`` is monotonic because every timestamp this repository compares is monotonic;
    a wall clock here would make a downstream age computation negative and fail open.
    """
    deadline = clock() + seconds
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sock.settimeout(min(remaining, 0.5))
        try:
            payload, _address = sock.recvfrom(2048)
        except socket.timeout:
            continue
        timestamp = clock()
        try:
            frame = decode_frame(payload)
        except DecodeError:
            statistics.observe_undecoded()
            continue
        statistics.observe(frame, timestamp)
        if record is not None:
            record({"timestamp_s": timestamp, **frame})
    return statistics


def open_listener(bind: str, port: int) -> socket.socket:
    """Bind a UDP socket for receiving only.

    No ``connect``, no ``sendto``, and no destination address is ever supplied, so this
    socket has nowhere to transmit even if a caller tried.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind, port))
    return sock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Passively decode the Lite3 motion host's high-level UDP state stream. "
                    "Cannot move the robot.",
    )
    parser.add_argument("--bind", default="0.0.0.0",
                        help="local address to listen on (default: all interfaces)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"target_port from network.toml (default: {DEFAULT_PORT})")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="how long to listen (default: 30)")
    parser.add_argument("--record", metavar="PATH",
                        help="append every decoded frame to this JSONL file")
    parser.add_argument("--robot-id", default=None,
                        help="stamp the recording with which Venture this is")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    statistics = ProbeStatistics()

    print(f"Listening on {args.bind}:{args.port} for {args.seconds:.0f}s. "
          f"This process cannot transmit.", file=sys.stderr)

    with contextlib.ExitStack() as stack:
        record = None
        if args.record:
            handle = stack.enter_context(open(args.record, "a", encoding="utf-8"))
            handle.write(json.dumps({"kind": "header", "robot_id": args.robot_id,
                                     "port": args.port, "clock": "monotonic"}) + "\n")

            def record(frame: dict) -> None:
                handle.write(json.dumps(frame) + "\n")

        sock = stack.enter_context(contextlib.closing(open_listener(args.bind, args.port)))
        try:
            run_probe(sock, args.seconds, statistics, record=record)
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)

    print(_format_report(statistics))
    if args.record:
        print(f"recording: {args.record}")
    return 0 if statistics.counts else 1


if __name__ == "__main__":
    sys.exit(main())
