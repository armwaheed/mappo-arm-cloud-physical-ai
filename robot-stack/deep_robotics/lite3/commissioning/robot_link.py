# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Opening, checking and closing the Lite3 link for a commissioning measurement.

Two of the five commissioning probes have to move the robot, and both of them have the
same pre-flight to do first. This module is that pre-flight, so it is written once and
tested once rather than drifting into two private copies -- which is exactly how the Go2
grew three private stand-up sequences, one of which never stood the robot at all.

**What this module is not.** It contains no gait, no balance controller and no mode
transition. It commands body velocities through the vendor's own high-level interface and
the manufacturer's controller keeps the robot upright, which is the property that made
that interface acceptable in the first place. The approved transition into the vendor's
high-level navigation mode is still an external operator action on the vendor remote --
see ``../README.md`` -- and ``--operator-ready`` records that it succeeded.

See ``../../../SAFETY.md``. ``--live`` is the only flag here that moves a leg.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deep_robotics.lite3.commissioning.measurement import Refusal, require_positive_finite
from deep_robotics.lite3.locomotion.lite3_locomotion import Lite3Locomotion
from deep_robotics.lite3.locomotion.lite3_udp_locomotion import (
    DEFAULT_COMMAND_PORT,
    DEFAULT_MOTION_HOST,
    DEFAULT_STATE_PORT,
    Lite3LinkLost,
    udp_locomotion_factory,
)
from deep_robotics.lite3.visual_nav.safety import BATTERY_SOC_ABORT_PCT

#: How stale the state stream may be when a moving probe starts. Deliberately tighter
#: than the transport's own 0.5 s command gate: that one decides whether a single command
#: may go out, this one decides whether a multi-minute measurement is worth beginning.
PREFLIGHT_STATE_MAX_AGE_S = 0.3


def add_context_arguments(parser) -> None:
    """The four things issue #13 requires beside every number.

    "Record robot ID, firmware, payload, and command envelope beside every result." A
    number without them is not transferable between the two Ventures, and the whole point
    of #13 is that nothing transfers between the two Ventures.
    """
    context = parser.add_argument_group("issue #13 context (required beside every number)")
    context.add_argument("--robot-id", required=True,
                         help="which Venture this is, e.g. LITE3-A. Never reuse a number "
                              "measured on the other robot")
    context.add_argument("--firmware", required=True,
                         help="firmware version string read from the vendor app/remote")
    context.add_argument("--payload", required=True,
                         help="what is mounted for the event, e.g. "
                              "'stock + 0.6 kg camera mast'. 'none' is a valid answer, "
                              "but it has to be an answer")


def add_link_arguments(parser, *, moving: bool) -> None:
    """Transport selection and, for a probe that moves, the authority to move."""
    link = parser.add_argument_group("Lite3 high-level UDP link")
    link.add_argument("--motion-host", default=DEFAULT_MOTION_HOST,
                      help=f"motion host address (default: {DEFAULT_MOTION_HOST})")
    link.add_argument("--command-port", type=int, default=DEFAULT_COMMAND_PORT,
                      help=f"motion host command port (default: {DEFAULT_COMMAND_PORT})")
    link.add_argument("--state-port", type=int, default=DEFAULT_STATE_PORT,
                      help="local port the motion host streams state to; it must match "
                           "'ip'/'target_port' in ~/jy_exe/conf/network.toml")
    if not moving:
        return
    authority = parser.add_argument_group("authority to move (SAFETY.md)")
    authority.add_argument("--live", action="store_true",
                           help="DANGER: actually walk the robot. Without this the probe "
                                "prints its plan and exits without opening a socket")
    authority.add_argument("--operator-ready", action="store_true",
                           help="confirm the Lite3 is STANDING in vendor high-level "
                                "navigation mode with the emergency stop in your hand")
    authority.add_argument("--battery-abort", type=float, default=BATTERY_SOC_ABORT_PCT,
                           help="refuse to start below this battery percentage "
                                f"(default: {BATTERY_SOC_ABORT_PCT:.0f})")


@dataclass(frozen=True)
class Link:
    """The navigator-facing locomotion object, and the transport underneath it.

    Both, because they answer different questions. ``locomotion`` is what a probe drives;
    ``implementation`` is what knows the battery, the state age and the vendor mode
    fields. Reaching into ``Lite3Locomotion`` for a private attribute to get the second
    would make this module depend on the shape of another module's internals.
    """

    locomotion: object
    implementation: object


def connect(args, *, factory=None) -> Link:
    """Open the vendor high-level UDP link and wait for proof the robot is reporting.

    Failure here is informative rather than annoying: a Lite3 streams state to the single
    address configured in ``~/jy_exe/conf/network.toml``, and if this laptop does not hold
    that address the robot is silent, every command vanishes into the network, and the
    symptom is a robot that never moves and never explains why.
    """
    build = factory or udp_locomotion_factory(
        motion_host=args.motion_host, command_port=args.command_port,
        state_port=args.state_port)
    created = []

    def capturing(**kwargs):
        implementation = build(**kwargs)
        created.append(implementation)
        return implementation

    loco = Lite3Locomotion(operator_ready=args.operator_ready,
                           implementation_factory=capturing)
    try:
        loco.connect()
    except Lite3LinkLost as lost:
        raise Refusal(str(lost)) from None
    return Link(locomotion=loco, implementation=created[0] if created else None)


def preflight(link: Link, args, *, printer=print) -> dict:
    """Refuse before the first command if the robot is not fit to be measured.

    Returns the health snapshot that was checked, so it can be recorded next to the
    numbers the run goes on to produce.
    """
    if not args.operator_ready:
        raise Refusal(
            "the operator has not confirmed STANDING + vendor high-level navigation "
            "mode. Stand the robot on the vendor remote, keep the emergency stop in your "
            "hand, then pass --operator-ready. Commanding a robot in manual mode does "
            "nothing at all, and 'nothing moved' is indistinguishable from 'the floor is "
            "below every setting we tried'."
        )
    require_positive_finite(**{"--battery-abort": args.battery_abort})

    implementation = link.implementation
    age = _state_age(implementation)
    if age is None or age > PREFLIGHT_STATE_MAX_AGE_S:
        raise Refusal(
            f"the Lite3 state stream is "
            f"{'silent' if age is None else f'{age:.2f}s stale'}; a measurement taken "
            f"through a link this cold would be timestamped nonsense. Check 'ip' in "
            f"~/jy_exe/conf/network.toml against this laptop's address."
        )

    battery = _read(implementation, "battery_level")
    if battery is None:
        raise Refusal(
            "no battery reading arrived on the state stream. RobotState.battery_level is "
            "present on this interface, so a missing one means the link is wrong rather "
            "than the field being absent."
        )
    if battery <= args.battery_abort:
        raise Refusal(
            f"battery is {battery:.0f}%, at or below the {args.battery_abort:.0f}% abort "
            f"limit. A gait floor measured on a flat battery is a measurement of the "
            f"battery."
        )

    error_state = _read(implementation, "error_state")
    if error_state not in (None, 0):
        raise Refusal(f"the Lite3 reports error_state={error_state}; refusing to command "
                      f"a robot that is already reporting a fault")

    mode = _read(implementation, "mode")
    printer(f"[link] battery {battery:.0f}%, state {age * 1000:.0f} ms old, "
            f"mode {mode if mode is not None else 'not reported by this build'}")
    if error_state is None:
        printer("[link] this build does not expose error_state on the state snapshot "
                "(it arrives with PR #74); the fault gate above could not be applied")
    return {"battery_pct": battery, "state_age_s": age, "mode": list(mode) if mode else None,
            "error_state": error_state}


def _state_age(implementation):
    getter = getattr(implementation, "state_age", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


def _read(implementation, name):
    """Read an optional accessor, returning ``None`` when this build does not have it.

    ``error_state()`` and ``mode()`` are added to the UDP transport by PR #74. Treating
    their absence as ``None`` keeps this harness runnable on ``main`` today, and
    :func:`preflight` says out loud which gate it could not apply rather than reporting a
    pass it never performed.
    """
    getter = getattr(implementation, name, None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None
