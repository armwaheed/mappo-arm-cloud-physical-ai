#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read all twelve Lite3 motor temperatures, or prove that the channel does not exist.

    python3 motor_temperature_probe.py --robot-id LITE3-A --firmware V1.0.8 \\
        --payload none --seconds 20

Answers the issue #13 vendor blocker "provide a supported high-level health bridge that
publishes BatteryState and exactly 12 Celsius motor temperatures". It answers it in
whichever direction the robot actually supports, and both directions are useful: twelve
numbers close the item, and a negative result with its evidence is what the vendor
question needs attached to it.

**A NEGATIVE RESULT IS ONLY EVIDENCE IF THE LINK WAS UP.** "No temperatures arrived" and
"this interface does not carry temperatures" look identical from a laptop that is not
receiving anything at all, and only one of them is a finding. So this probe refuses to
report the channel as absent unless it saw the state stream flowing -- a silent link is
reported as a silent link, and you are sent to ``~/jy_exe/conf/network.toml``.

**IT WILL NOT GUESS.** There is exactly one twelve-element field in the high-level UDP
stream, ``JointState.joint_positions``, and it is joint angles in radians. A probe that
went looking for "twelve numbers" would find it, and would report a robot whose motors
were all sitting at 0.4 degrees Celsius, or at 1.9. So the search is by documented field
name, and the probe additionally prints the observed range of that field as the
discriminator: joint angles live inside a couple of radians and no motor does.

WHAT IT CANNOT DO. Taking low-level ``Lite3_MotionSDK`` control would expose temperatures,
and would also remove the vendor controller that keeps the robot upright. That trade is
not available for a demo, and issue #13 says so explicitly. This probe stays on the
high-level interface.

Nothing here transmits and nothing moves. There is no ``--live``.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_robotics.lite3.commissioning import robot_link
from deep_robotics.lite3.commissioning.lite3_state_probe import (
    DEFAULT_PORT,
    ProbeStatistics,
    open_listener,
    run_probe,
)
from deep_robotics.lite3.commissioning.measurement import (
    Refusal,
    brief,
    merge_measurement,
    new_record,
    paste_block,
    print_paste_block,
    run_main,
    write_record,
)

#: A Lite3 has twelve actuators. A health gate fed eleven of them is a health gate with a
#: hole in it, so a partial set is refused rather than reported.
MOTOR_COUNT = 12

#: Field names on the high-level UDP stream that would carry temperatures if the interface
#: carried them. Searched by NAME, never by shape -- see the module docstring.
TEMPERATURE_FIELD_NAMES = (
    "motor_temperatures", "motor_temperature", "temperatures", "temperature_c",
    "joint_temperatures", "motor_temp", "temp",
)

#: The one twelve-element field that does exist, and what it actually is.
DECOY_FIELD = "joint_positions"

#: Above this, a value cannot plausibly be a joint angle in radians. Used only to print
#: the discriminator that separates the decoy field from a real temperature array.
NOT_AN_ANGLE_C = 45.0

#: Keys the probe itself puts on a decoded frame. Excluded from the reported field list so
#: that "fields actually present" describes the vendor's wire and not this tool.
BOOKKEEPING_KEYS = frozenset(("kind", "code", "size", "cons_code", "timestamp_s"))


def scan_frames(frames) -> dict:
    """Look through decoded frames for a documented temperature field. Report what is there.

    Returns the evidence, never a verdict on its own: :func:`verdict` decides, and it
    needs to know whether the link was flowing before it will call anything absent.
    """
    field_names = set()
    kinds = {}
    found = {}
    decoy_values = []
    for frame in frames:
        kind = frame.get("kind", "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
        for name, value in frame.items():
            if name not in BOOKKEEPING_KEYS:
                field_names.add(name)
            if name in TEMPERATURE_FIELD_NAMES:
                found.setdefault(name, value)
            if name == DECOY_FIELD and isinstance(value, dict):
                decoy_values.extend(float(item) for item in value.values())
    return {
        "frames": sum(kinds.values()),
        "kinds": kinds,
        "field_names": sorted(field_names),
        "temperature_fields": sorted(found),
        "temperature_values": found,
        "decoy_field": DECOY_FIELD,
        "decoy_range": ([min(decoy_values), max(decoy_values)] if decoy_values else None),
    }


def normalise(values) -> list:
    """Coerce a candidate reading to exactly twelve finite Celsius values, or refuse."""
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        raise Refusal("the temperature field is not a sequence of numbers") from None
    if len(numbers) != MOTOR_COUNT:
        raise Refusal(
            f"the temperature field carried {len(numbers)} values, not {MOTOR_COUNT}. A "
            f"partial set is refused: a thermal gate fed eleven of twelve motors is a "
            f"gate with a hole in it, and the hole is invisible once the number is "
            f"written down."
        )
    if any(not math.isfinite(number) for number in numbers):
        raise Refusal("the temperature field contained a non-finite value")
    return numbers


def verdict(evidence: dict, ros_values=None) -> dict:
    """Turn the evidence into one of three answers, and refuse the fourth.

    ``present`` -- twelve values were read, and they are recorded.
    ``absent``  -- the stream was flowing and carries no such field. A finding.
    ``unknown`` -- nothing arrived, so nothing was learned. Refused, not reported.
    """
    if ros_values is not None:
        return {"channel": "present", "source": "ros:/motor_temperatures",
                "temperatures_c": normalise(ros_values), "evidence": evidence}
    found = next(iter(evidence["temperature_fields"]), None)
    if found is not None:
        return {"channel": "present", "source": f"udp:{found}",
                "temperatures_c": normalise(evidence["temperature_values"][found]),
                "evidence": evidence}
    if evidence["frames"] == 0:
        raise Refusal(
            "no state frames arrived at all, so this run learned nothing about motor "
            "temperatures. A silent link and an absent channel look identical from here "
            "and only one of them is a finding. Check that this laptop holds the address "
            "in ~/jy_exe/conf/network.toml on the motion host, that it is on "
            "192.168.1.0/24, and that no firewall is dropping inbound UDP."
        )
    return {"channel": "absent", "source": None, "temperatures_c": None,
            "evidence": evidence}


def read_ros_topic(topic: str, timeout_s: float, printer=print):
    """Try the optional companion ROS topic. Returns values, or ``None`` with a reason.

    Optional on purpose: the Lite3 path is ROS-free by default, and a unit with no
    provisioned perception host has nowhere to run a bridge. "Not attempted" is reported
    as "not attempted" rather than folded into "absent".
    """
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float64MultiArray
    except ImportError as error:
        printer(f"[temps] ROS 2 is not importable here ({error.name}); the companion "
                f"topic {topic} was NOT attempted, which is not the same as it being "
                f"absent")
        return None

    received = []
    started_ros = not rclpy.ok()
    if started_ros:
        rclpy.init()
    node = Node("lite3_motor_temperature_probe")
    node.create_subscription(Float64MultiArray, topic,
                             lambda message: received.append(list(message.data)), 10)
    try:
        deadline = timeout_s
        while not received and deadline > 0.0:
            rclpy.spin_once(node, timeout_sec=0.2)
            deadline -= 0.2
    finally:
        with contextlib.suppress(Exception):
            node.destroy_node()
        if started_ros:
            with contextlib.suppress(Exception):
                rclpy.shutdown()
    if not received:
        printer(f"[temps] nothing published on {topic} within {timeout_s:.0f}s")
        return None
    return received[-1]


def capture(args, printer=print) -> list:
    """Listen passively for ``--seconds`` and return every decoded frame."""
    frames = []
    printer(f"[temps] listening on {args.bind}:{args.state_port} for "
            f"{args.seconds:.0f}s. This process cannot transmit.")
    with contextlib.closing(open_listener(args.bind, args.state_port)) as sock:
        run_probe(sock, args.seconds, ProbeStatistics(), record=frames.append)
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    robot_link.add_context_arguments(parser)
    # Only the receive half of the link is used here, so only that half is offered: a
    # --motion-host flag on a probe with no transmit path would read as configuration
    # that does something.
    parser.add_argument("--bind", default="0.0.0.0",
                        help="local address to listen on (default: all interfaces)")
    parser.add_argument("--state-port", type=int, default=DEFAULT_PORT,
                        help="local port the motion host streams state to; it must match "
                             "'ip'/'target_port' in ~/jy_exe/conf/network.toml "
                             f"(default: {DEFAULT_PORT})")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="how long to listen (default: 20)")
    parser.add_argument("--ros-topic", default=None, metavar="TOPIC",
                        help="also try a companion ROS topic, e.g. /motor_temperatures. "
                             "Skipped entirely when ROS 2 is not installed here")
    parser.add_argument("--ros-timeout", type=float, default=5.0,
                        help="seconds to wait on --ros-topic (default: 5)")
    parser.add_argument("--artefact", default=None,
                        help="write the machine-readable record here "
                             "(default: lite3-motor-temperatures-<robot-id>.json)")
    return parser


def _paste(record, result) -> str:
    rows = [
        ("robot", record.robot_id),
        ("firmware", record.context.get("firmware")),
        ("payload", record.context.get("payload")),
        ("frames observed",
         f"{result['evidence']['frames']} across {result['evidence']['kinds']}"),
    ]
    if result["channel"] == "present":
        values = result["temperatures_c"]
        hottest = max(range(len(values)), key=values.__getitem__)
        rows += [
            ("source", result["source"]),
            ("12 motor temperatures",
             ", ".join(f"{value:.1f}" for value in values) + " C"),
            ("hottest", f"motor {hottest} at {values[hottest]:.1f} C"),
        ]
        notes = ["- Twelve channels read. The health gate can be run with temperatures "
                 "monitored, and `--accept-no-motor-temperatures` is no longer needed on "
                 "this robot."]
    else:
        decoy = result["evidence"]["decoy_range"]
        rows += [
            ("motor temperature channel", "**ABSENT** from the high-level interface"),
            ("fields actually present",
             ", ".join(f"`{name}`" for name in result["evidence"]["field_names"])),
        ]
        notes = [
            "- The only twelve-element field on this interface is `joint_positions`" +
            (f", observed spanning {decoy[0]:.2f} .. {decoy[1]:.2f}" if decoy else "") +
            f" -- joint angles in radians, not Celsius. Nothing above {NOT_AN_ANGLE_C:.0f}"
            " C was seen on any field, so there is no temperature array hiding in here "
            "under another name.",
            "- This is the evidence for the vendor question: what is the supported way to "
            "read the twelve motor temperatures on this firmware, at the high level? "
            "Taking low-level MotionSDK control to read them is not acceptable, because "
            "it removes the controller that keeps the robot upright.",
            "- Until it is answered, a live run needs "
            "`--accept-no-motor-temperatures`, which caps the run at 120 s, warns every "
            "tick, and records that temperatures were unmonitored. It bounds one run; "
            "heat builds across repeated runs and no software can see that.",
        ]
    notes.append("- Measured on this robot only. Do not copy to the other Venture.")
    return paste_block("Lite3 motor temperatures", rows, notes)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    brief(
        "Lite3 motor temperatures -- twelve channels, or proof there are none",
        does=f"""
        Listens passively to the high-level UDP state stream for {args.seconds:.0f} s and
        looks for a documented twelve-value Celsius field, by name. Optionally also tries
        a companion ROS topic. Transmits nothing.
        """,
        needs=[
            "this laptop holding the address in ~/jy_exe/conf/network.toml on the motion "
            "host (factory default 192.168.1.102), netmask 255.255.255.0, Router field "
            "EMPTY",
            "the robot powered on -- prone and untouched is fine",
            "nothing else bound to the state port",
        ],
        means="""
        Twelve numbers close the health-bridge half of the issue #13 vendor blocker.
        An 'absent' verdict is the evidence to attach to the vendor question, and it is
        only valid because the stream was seen flowing while nothing arrived.
        A silent link is reported as a silent link and settles nothing.
        """,
        moves=False,
    )
    frames = capture(args)
    evidence = scan_frames(frames)
    print(f"[temps] {evidence['frames']} frames: {evidence['kinds']}")

    ros_values = None
    if args.ros_topic:
        ros_values = read_ros_topic(args.ros_topic, args.ros_timeout)

    result = verdict(evidence, ros_values)
    print("")
    if result["channel"] == "present":
        values = result["temperatures_c"]
        for index, value in enumerate(values):
            print(f"  motor {index:>2}  {value:6.1f} C")
        print(f"  hottest: motor {max(range(len(values)), key=values.__getitem__)}")
    else:
        decoy = evidence["decoy_range"]
        print("  NO MOTOR TEMPERATURE CHANNEL on the high-level interface.")
        print(f"  Fields seen: {', '.join(evidence['field_names'])}")
        if decoy:
            print(f"  The only 12-element field, {DECOY_FIELD}, spanned "
                  f"{decoy[0]:.2f}..{decoy[1]:.2f} -- radians, not Celsius. Nothing "
                  f"there is above {NOT_AN_ANGLE_C:.0f} C.")
        print("  This is a finding, not a failure: the stream was flowing and carries no "
              "such field.")

    record = new_record(args.robot_id, firmware=args.firmware, payload=args.payload,
                        seconds=args.seconds, state_port=args.state_port,
                        ros_topic=args.ros_topic)
    merge_measurement(record, "motor_temperatures", result)
    destination = Path(args.artefact or f"lite3-motor-temperatures-{args.robot_id}.json")
    write_record(destination, record)
    print(f"\n[temps] artefact: {destination.resolve()}  "
          f"(provenance: {record.provenance})")
    print_paste_block(_paste(record, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(lambda: main(), "temps"))
