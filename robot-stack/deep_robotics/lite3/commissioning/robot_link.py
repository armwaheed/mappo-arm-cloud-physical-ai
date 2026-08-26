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
from deep_robotics.lite3.locomotion.lite3_axis_locomotion import (
    DEFAULT_LOCAL_PORT,
    AxisProfile,
    AxisProfileError,
    axis_locomotion_factory,
)
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


@dataclass(frozen=True)
class Transport:
    """One selectable way to reach the motors, and the two facts a probe has to know.

    ``preserves_magnitude`` decides whether a *measurement* is possible at all. A probe
    that varies a commanded speed and reads back what the robot did is measuring the
    robot only if the commanded number reaches the wire. On a transport that discards it,
    the same probe runs to completion, refuses nothing, and reports a confident number
    that describes the ladder rather than the robot.

    ``walked`` is the hardware evidence that this transport moves these two Ventures at
    all, or ``None`` when there is none. It is separate from the first because the two
    fail in opposite directions: a magnitude-discarding transport produces a wrong
    number, and an unwalked one produces a table of zeros that reads exactly like a floor
    that is real and total.
    """

    preserves_magnitude: bool
    walked: str | None
    summary: str


#: The transports the moving probes can select, and what is actually known about each.
#:
#: These two rows are the whole reason this module grew a transport argument. The Lite3
#: navigator has offered ``--locomotion-transport {udp,axis,ros2}`` since PR #74; this
#: harness offered nothing and hard-wired ``udp``, so every walking measurement in it was
#: aimed at the one interface no Venture has been seen to move on.
TRANSPORTS = {
    "udp": Transport(
        preserves_magnitude=True,
        walked=None,
        summary=(
            "legacy complex-velocity UDP, codes 320/325/321. The commanded float reaches "
            "the wire, so a ladder measured here would mean something -- but on "
            "2026-08-24 a vx=0.10 m/s, 10 Hz, 1 s pulse was captured arriving correctly "
            "at the motion host and the robot did not move: zero world-pose delta over "
            "six seconds, error_state 0. The vendor guide gives the reason -- these "
            "commands require autonomous mode, and a robot in AI state cannot enter it."
        ),
    ),
    "axis": Transport(
        preserves_magnitude=False,
        walked=(
            "2026-08-24, one bounded forward-axis pulse moved 0.401 m and stopped "
            "cleanly; 2026-08-26, four --live navigation runs, one of which walked "
            "0.54 m under the policy and one of which arrived"
        ),
        summary=(
            "profile-gated moving-mode simple axes. This is the transport both Ventures "
            "have actually walked on, and its mapping is SIGN-ONLY: every command past "
            "the profile's linear deadband emits the same full-scale primitive, so a "
            "commanded speed is not a speed here -- it is a direction."
        ),
    ),
}


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
    link.add_argument("--locomotion-transport", choices=sorted(TRANSPORTS), default="udp",
                      help="which vendor interface to command through. 'udp' is the "
                           "legacy complex-velocity interface and matches the "
                           "navigator's own default; 'axis' is the profile-gated "
                           "moving-mode simple axes, which is the only one either "
                           "Venture has been seen to walk on (default: udp)")
    link.add_argument("--axis-profile", default=None, metavar="PATH",
                      help="evidenced lite3-axis-profile/v1 JSON; required by "
                           "--locomotion-transport axis and refused by the others")
    link.add_argument("--axis-local-port", type=int, default=DEFAULT_LOCAL_PORT,
                      help=f"local port the axis stream sends from "
                           f"(default: {DEFAULT_LOCAL_PORT})")
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
    authority.add_argument("--accept-unwalked-transport", action="store_true",
                           help="proceed on a transport no Venture has been seen to walk "
                                "on. Without this, such a run is refused: a transport "
                                "that does not actuate reports 0.000 m/s on every "
                                "segment, which reads exactly like a floor")


def selected_transport(args) -> Transport:
    """The :class:`Transport` row for ``--locomotion-transport``, defaulting to udp."""
    name = getattr(args, "locomotion_transport", "udp")
    try:
        return TRANSPORTS[name]
    except KeyError:
        raise Refusal(f"unknown --locomotion-transport {name!r}; expected one of "
                      + ", ".join(sorted(TRANSPORTS))) from None


def require_magnitude_transport(args, *, measures: str, instead: str) -> None:
    """Refuse a magnitude-dependent measurement on a transport that discards magnitude.

    Called at argument-validation time, so it fires during the dry run the RUNBOOK asks
    for first and costs no robot time at all.

    This is the failure that is worse than a crash. Point a descending ladder at the axis
    transport and nothing refuses: every rung above the profile's linear deadband emits
    the *same* full-scale primitive, so every rung walks, at the same speed, and the
    anchors and the drift controls all pass. The probe then reports the lowest rung it
    tried as the gait floor -- a plausible, safe-looking number that is a property of the
    ladder and not of the robot, and which goes straight into ``--gait-floor`` on a live
    run.
    """
    transport = selected_transport(args)
    if transport.preserves_magnitude:
        return
    raise Refusal(
        f"--locomotion-transport {args.locomotion_transport} discards the commanded "
        f"magnitude, and {measures} is a measurement OF that magnitude.\n"
        f"{transport.summary}\n"
        f"Nothing here would have refused: every rung above the profile's linear "
        f"deadband fires the same primitive, so every rung walks at the same speed and "
        f"every check in this probe passes. The number would describe the ladder.\n"
        f"On this transport the measurement that exists is {instead}."
    )


def require_sign_only_transport(args, *, measures: str, instead: str) -> None:
    """Refuse a per-primitive measurement on a transport that has no primitives.

    The mirror of :func:`require_magnitude_transport`, and it exists for the same reason:
    a probe that commands one direction and reads back one speed produces a number on
    either transport, and only on the axis transport is that number a primitive's speed.
    On the velocity transport it is the speed of whatever this probe happened to command.
    """
    transport = selected_transport(args)
    if not transport.preserves_magnitude:
        return
    raise Refusal(
        f"--locomotion-transport {args.locomotion_transport} carries the commanded "
        f"magnitude to the wire, so it has no fixed primitives and {measures} is not "
        f"defined on it.\n"
        f"{transport.summary}\n"
        f"Select --locomotion-transport axis, or measure {instead} instead."
    )


def require_walked_transport(args) -> None:
    """Refuse to spend robot time on a transport no Venture has been seen to walk on.

    Called at argument-validation time, so it fires in the dry run before any socket
    exists. That placement is the whole point: this check needs no robot, and finding out
    at the dry run costs nothing while finding out live costs the session.

    The failure it prevents is not a crash. A transport that does not actuate reports
    0.000 m/s on every segment, which is exactly what a floor above the entire ladder
    looks like, and the anchor refusal that eventually fires names the stand sequence and
    the control mode -- so the time goes on re-standing a robot that was standing fine.
    """
    transport = selected_transport(args)
    if transport.walked is not None or getattr(args, "accept_unwalked_transport", False):
        return
    name = args.locomotion_transport
    raise Refusal(
        f"no Lite3 Venture has been seen to walk on --locomotion-transport {name}, and "
        f"this probe is about to spend minutes of robot time asking one to.\n"
        f"{transport.summary}\n"
        f"A transport that does not actuate reports 0.000 m/s on every segment. That is "
        f"indistinguishable from a floor above the whole ladder, and the refusal it "
        f"eventually triggers blames the stand sequence and the control mode -- so the "
        f"session gets spent re-standing a robot that was standing fine.\n"
        f"Use --locomotion-transport axis, which both Ventures have walked on, or pass "
        f"--accept-unwalked-transport if establishing whether this interface actuates at "
        f"all is the point of the run."
    )


def load_axis_profile(args) -> AxisProfile | None:
    """Load and validate the axis profile, refusing the combinations that cannot work."""
    transport = getattr(args, "locomotion_transport", "udp")
    path = getattr(args, "axis_profile", None)
    if transport != "axis":
        if path is not None:
            raise Refusal(
                f"--axis-profile was given but --locomotion-transport is {transport!r}, "
                f"which does not read one. Passing a profile to a transport that ignores "
                f"it looks exactly like a profile that took effect."
            )
        return None
    if path is None:
        raise Refusal(
            "--locomotion-transport axis requires --axis-profile. The transport ships no "
            "raw axis value for any direction and will not invent one: a profile "
            "supplies each primitive together with the evidence reference behind it."
        )
    try:
        return AxisProfile.load(Path(path))
    except AxisProfileError as error:
        raise Refusal(f"cannot use axis profile {path}: {error}") from None


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


def _transport_factory(args):
    """Build the implementation factory the selected transport needs.

    Both factories take the same navigator-facing keyword arguments and ignore them, so
    the choice is entirely contained here and :class:`Lite3Locomotion` never learns which
    one it composed.
    """
    if getattr(args, "locomotion_transport", "udp") == "axis":
        return axis_locomotion_factory(
            axis_profile=load_axis_profile(args),
            axis_local_port=getattr(args, "axis_local_port", DEFAULT_LOCAL_PORT),
            motion_host=args.motion_host, command_port=args.command_port,
            state_port=args.state_port)
    load_axis_profile(args)  # refuses a profile handed to a transport that ignores it
    return udp_locomotion_factory(
        motion_host=args.motion_host, command_port=args.command_port,
        state_port=args.state_port)


def connect(args, *, factory=None) -> Link:
    """Open the vendor high-level UDP link and wait for proof the robot is reporting.

    Failure here is informative rather than annoying: a Lite3 streams state to the single
    address configured in ``~/jy_exe/conf/network.toml``, and if this laptop does not hold
    that address the robot is silent, every command vanishes into the network, and the
    symptom is a robot that never moves and never explains why.
    """
    build = factory or _transport_factory(args)
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

    transport = selected_transport(args)
    name = getattr(args, "locomotion_transport", "udp")
    printer(f"[link] transport {name}: "
            + (f"has walked a Venture -- {transport.walked}" if transport.walked
               else "NO Venture has been seen to walk on this transport"))

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
            "error_state": error_state, "locomotion_transport": name,
            "transport_walked_evidence": transport.walked}


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
