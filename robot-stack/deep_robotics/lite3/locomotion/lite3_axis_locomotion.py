#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Profile-gated Lite3 simple-axis locomotion over a verified high-level state link.

This transport preserves the existing navigator interface while replacing the legacy
``320/325/321`` complex-velocity sender with the vendor's moving-mode axis protocol. It does
not select control or moving mode: the operator must explicitly establish the vendor-approved
state before a live run. No nonzero primitive is built in; a validated local profile is required.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from deep_robotics.lite3.locomotion.lite3_axis_udp import (
    AXIS_LIMIT,
    FORWARD_AXIS_CODE,
    HEARTBEAT_CODE,
    LATERAL_AXIS_CODE,
    YAW_AXIS_CODE,
    axis_packet,
)
from deep_robotics.lite3.locomotion.lite3_udp_locomotion import (
    DEFAULT_COMMAND_PORT,
    DEFAULT_MOTION_HOST,
    DEFAULT_STATE_PORT,
    Lite3LinkLost,
    Lite3UdpLocomotion,
)

AXIS_PROFILE_SCHEMA = "lite3-axis-profile/v1"
AXIS_RATE_HZ = 20.0
HEARTBEAT_HZ = 4.0
COMMAND_TTL_S = 0.15
STOP_SECONDS = 2.0
DEFAULT_LOCAL_PORT = 20001

FORWARD_DEAD_ZONE = 6553
LATERAL_DEAD_ZONE = 12553
YAW_DEAD_ZONE = 9553
ALLOWED_MOVING_GAIT_STATES = frozenset((0, 2, 4, 5, 6, 13))

#: The four primitives that move the robot's POSITION, and the two that move only its
#: heading. Public because the split is a safety distinction rather than a detail: an
#: undeclared LINEAR speed leaves the robot travelling at a rate nobody has timed, and
#: an undeclared YAW rate leaves a pure turn — whose rollout is a single point — merely
#: unpredictable in heading. ``Lite3Bindings`` refuses a live run for the first and
#: warns for the second, and it has to be able to tell them apart by name.
LINEAR_PRIMITIVES = ("forward_positive", "forward_negative",
                     "lateral_positive", "lateral_negative")
YAW_PRIMITIVES = ("yaw_positive", "yaw_negative")

#: The eight (forward, lateral) sign pairs this mapping can express, indexed by the
#: commanded bearing in units of 45 degrees counter-clockwise from straight ahead.
_LINEAR_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1),
                      (-1, 0), (-1, -1), (0, -1), (1, -1))

#: Which evidenced primitive delivers each ``(axis, navigator sign)``, and therefore
#: which ``measured_m_s``/``measured_rad_s`` entry names the speed the legs will produce.
#:
#: ⚠️ **MAPPED BY ROLE, NOT BY NAME.** The shared navigator's ``+y`` and ``+yaw`` are
#: LEFT and the vendor's positive raw value is RIGHT, so two of these six rows cross:
#: a navigator command to go left is delivered by the primitive called
#: ``lateral_negative``, whose ``measured_m_s`` entry is therefore the speed of a LEFT
#: step. Reading the table off the names instead would put the robot's measured left
#: speed on its right-hand strafe and vice versa, and no test that only ever measures
#: one side would notice. The one thing that keeps this honest is that
#: :meth:`AxisProfile.map_velocity` and :meth:`AxisProfile.executed_velocity` derive
#: their signs from the same :meth:`AxisProfile._signs`, so a raw value and a speed
#: cannot come from different rows.
_DELIVERING_PRIMITIVE = {
    ("forward", 1): "forward_positive",
    ("forward", -1): "forward_negative",
    ("lateral", 1): "lateral_negative",
    ("lateral", -1): "lateral_positive",
    ("yaw", 1): "yaw_negative",
    ("yaw", -1): "yaw_positive",
}


class AxisProfileError(ValueError):
    """An axis profile cannot safely convert a requested navigation component."""


@dataclass(frozen=True)
class ExecutedVelocity:
    """The velocity the LEGS will receive for one commanded velocity.

    Same frame and same units as the command it answers about — the shared navigator's
    body frame, ``+x`` forward, ``+y`` left, ``+yaw`` left, m/s and rad/s — because the
    whole point is that a caller asked for a velocity and this says which velocity it
    is going to get instead. An axis whose primitive does not fire is exactly ``0.0``.

    ⚠️ **THE COMMANDED MAGNITUDE IS NOT IN HERE, AND THAT IS THE FINDING.**
    :meth:`AxisProfile.map_velocity` is sign-only, so this transport's executable set on
    each axis is TWO VALUES — ``{0, the primitive's evidenced speed}``. Asking for
    0.05 m/s and asking for 0.55 m/s produce the same bytes on the wire and the same
    object here. Anything upstream that reasons about a commanded speed — a
    stopping-distance cap, a feasibility rollout, a gait floor — is reasoning about a
    number that never reaches the legs unless it reads this first.

    ``unmeasured`` names every primitive that WILL fire and whose delivered speed the
    profile does not declare in ``measured_m_s``/``measured_rad_s``. That axis's number
    is ``nan``, not ``0.0``: an undeclared speed is an absence, and ``0.0`` would read
    as "this axis stays still", which is the one thing it is guaranteed not to do. A
    consumer must branch on ``unmeasured`` rather than on the numbers.
    """

    vx: float
    vy: float
    yaw: float
    unmeasured: tuple[str, ...] = ()

    @property
    def is_known(self) -> bool:
        """Whether every primitive that will fire has a declared speed."""
        return not self.unmeasured

    @property
    def translates(self) -> bool:
        """Whether the legs will move the robot's POSITION, as opposed to its heading.

        ``False`` for a pure turn and for a stop. The distinction is what lets a
        consumer tolerate an unmeasured YAW rate — a rollout of a pure turn is a point,
        so its positions do not depend on the rate — while refusing an unmeasured
        forward or lateral one, which moves the robot somewhere nobody can name.
        Answers ``False`` on a ``nan`` axis only when that axis is not in
        ``unmeasured``, which cannot happen; ``nan != 0.0`` is True.
        """
        return self.vx != 0.0 or self.vy != 0.0


@dataclass(frozen=True)
class AxisValues:
    """One three-axis vendor joystick command."""

    forward: int = 0
    lateral: int = 0
    yaw: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("forward", self.forward),
            ("lateral", self.lateral),
            ("yaw", self.yaw),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise AxisProfileError(f"{name} axis value must be an integer")
            if not -AXIS_LIMIT <= value <= AXIS_LIMIT:
                raise AxisProfileError(
                    f"{name} axis value must be within {-AXIS_LIMIT}..{AXIS_LIMIT}"
                )

    @property
    def is_zero(self) -> bool:
        return self == AxisValues()


@dataclass(frozen=True)
class AxisProfile:
    """A locally evidenced mapping from navigation intent signs to vendor axis primitives."""

    forward_positive: int | None
    forward_negative: int | None
    lateral_positive: int | None
    lateral_negative: int | None
    yaw_positive: int | None
    yaw_negative: int | None
    linear_deadband_m_s: float
    yaw_deadband_rad_s: float
    allowed_gait_states: tuple[int, ...]
    evidence: tuple[tuple[str, str], ...] = ()
    measured_m_s: tuple[tuple[str, float], ...] = ()
    measured_rad_s: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Reject unsafe direct construction as well as malformed JSON profiles."""
        try:
            evidence = dict(self.evidence)
        except (TypeError, ValueError) as error:
            raise AxisProfileError(
                "axis profile evidence must contain name/reference pairs"
            ) from error
        for name, value, dead_zone in (
            ("forward_positive", self.forward_positive, FORWARD_DEAD_ZONE),
            ("forward_negative", self.forward_negative, FORWARD_DEAD_ZONE),
            ("lateral_positive", self.lateral_positive, LATERAL_DEAD_ZONE),
            ("lateral_negative", self.lateral_negative, LATERAL_DEAD_ZONE),
            ("yaw_positive", self.yaw_positive, YAW_DEAD_ZONE),
            ("yaw_negative", self.yaw_negative, YAW_DEAD_ZONE),
        ):
            self._validate_primitive(name, value, dead_zone, evidence)
        self._validate_deadband("linear_m_s", self.linear_deadband_m_s)
        self._validate_deadband("yaw_rad_s", self.yaw_deadband_rad_s)
        self._validate_measured("measured_m_s", self.measured_m_s, LINEAR_PRIMITIVES)
        self._validate_measured("measured_rad_s", self.measured_rad_s, YAW_PRIMITIVES)
        if not self.allowed_gait_states:
            raise AxisProfileError("axis profile requires at least one allowed moving gait state")
        for gait_state in self.allowed_gait_states:
            if isinstance(gait_state, bool) or not isinstance(gait_state, int):
                raise AxisProfileError("axis profile gait states must be integers")
            if gait_state not in ALLOWED_MOVING_GAIT_STATES:
                raise AxisProfileError(
                    f"axis profile gait state {gait_state} is not a documented moving gait"
                )

    @classmethod
    def load(cls, path: Path) -> AxisProfile:
        """Load a versioned local profile; no physical command values are shipped by default."""
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise AxisProfileError(f"cannot read axis profile {path}: {error}") from None
        if not isinstance(data, dict):
            raise AxisProfileError("axis profile must be a JSON object")
        if data.get("schema") != AXIS_PROFILE_SCHEMA:
            raise AxisProfileError(
                f"axis profile schema must be {AXIS_PROFILE_SCHEMA!r}, got {data.get('schema')!r}"
            )
        primitives = data.get("primitives")
        if not isinstance(primitives, dict):
            raise AxisProfileError("axis profile requires an object field 'primitives'")
        deadband = data.get("input_deadband")
        if not isinstance(deadband, dict):
            raise AxisProfileError("axis profile requires an object field 'input_deadband'")
        evidence = data.get("evidence", {})
        if not isinstance(evidence, dict):
            raise AxisProfileError("axis profile field 'evidence' must be an object")
        gait_states = data.get("allowed_gait_states")
        if not isinstance(gait_states, list):
            raise AxisProfileError("axis profile field 'allowed_gait_states' must be a list")
        measured = {}
        for field in ("measured_m_s", "measured_rad_s"):
            measured[field] = data.get(field, {})
            if not isinstance(measured[field], dict):
                raise AxisProfileError(f"axis profile field {field!r} must be an object")

        return cls(
            forward_positive=primitives.get("forward_positive"),
            forward_negative=primitives.get("forward_negative"),
            lateral_positive=primitives.get("lateral_positive"),
            lateral_negative=primitives.get("lateral_negative"),
            yaw_positive=primitives.get("yaw_positive"),
            yaw_negative=primitives.get("yaw_negative"),
            linear_deadband_m_s=deadband.get("linear_m_s"),
            yaw_deadband_rad_s=deadband.get("yaw_rad_s"),
            allowed_gait_states=tuple(gait_states),
            evidence=tuple(evidence.items()),
            measured_m_s=tuple(measured["measured_m_s"].items()),
            measured_rad_s=tuple(measured["measured_rad_s"].items()),
        )

    @staticmethod
    def _validate_primitive(name: str, value: Any, dead_zone: int,
                            evidence: dict[str, str]) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise AxisProfileError(f"primitive {name} must be an integer or null")
        if not -AXIS_LIMIT <= value <= AXIS_LIMIT:
            raise AxisProfileError(
                f"primitive {name} must be within {-AXIS_LIMIT}..{AXIS_LIMIT}"
            )
        if abs(value) <= dead_zone:
            raise AxisProfileError(
                f"primitive {name}={value} is inside the documented dead zone ±{dead_zone}"
            )
        expected_positive = name.endswith("_positive")
        if (value > 0) != expected_positive:
            direction = "positive" if expected_positive else "negative"
            raise AxisProfileError(f"primitive {name} must be {direction}")
        provenance = evidence.get(name)
        if not isinstance(provenance, str) or not provenance.strip():
            raise AxisProfileError(
                f"primitive {name} requires a non-empty evidence reference"
            )

    def _validate_measured(self, field: str, entries: Any, allowed: tuple[str, ...]) -> None:
        """Reject a speed measurement that names nothing, or measures an absent primitive."""
        try:
            items = dict(entries)
        except (TypeError, ValueError) as error:
            raise AxisProfileError(
                f"axis profile {field} must contain primitive/speed pairs"
            ) from error
        for name, value in items.items():
            if name not in allowed:
                raise AxisProfileError(
                    f"axis profile {field} names {name!r}; expected one of "
                    + ", ".join(allowed)
                )
            if getattr(self, name) is None:
                raise AxisProfileError(
                    f"axis profile {field} measures {name}, which has no primitive value"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or value <= 0.0:
                raise AxisProfileError(
                    f"axis profile {field}.{name} must be finite and positive"
                )

    @property
    def measured_speeds(self) -> dict[str, float]:
        """Every measured primitive speed by name — m/s for linear, rad/s for yaw."""
        return {**dict(self.measured_m_s), **dict(self.measured_rad_s)}

    @staticmethod
    def _validate_deadband(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AxisProfileError(f"input_deadband.{name} must be a finite positive number")
        if not math.isfinite(float(value)) or value <= 0.0:
            raise AxisProfileError(f"input_deadband.{name} must be a finite positive number")

    def map_velocity(self, vx: float, vy: float, yaw: float) -> AxisValues:
        """Map shared body-frame intent to explicitly evidenced vendor primitives.

        **This mapping is sign-only: the commanded magnitude is discarded.** A profile
        holds one evidenced raw value per direction, so every command past the deadband
        leaves at whatever speed that one primitive was measured to produce. Nothing
        here scales with ``vx``, which means nothing here honours ``--derate`` either:
        the envelope is enforced at preflight instead, by refusing a profile whose
        declared ``measured_m_s`` exceeds the derated ceiling. That is deliberate — a
        raw axis value with no physical evidence behind it is not something this
        transport will invent — but it has to be said out loud, because a filled-in
        profile executes it silently.

        Because only the sign survives, the linear pair is one vector rather than two
        independent axes. Its magnitude gates both components together, and its bearing
        is then snapped to the nearest of the eight directions in
        :data:`_LINEAR_DIRECTIONS`. A per-axis deadband instead drops the smaller
        component and *rotates the command*: at the shipped 0.05 m/s deadband,
        ``(0.049, 0.051)`` m/s is a 46 degree command that clears only the lateral gate
        and leaves as a 90 degree strafe. Snapping does not make a diagonal accurate —
        the executed bearing of one is set by the two primitives' measured speeds, not
        by the command — but it does stop the choice of direction depending on which of
        two independent thresholds a component happened to fall under.

        Yaw keeps its own scalar deadband. It is a third axis, not part of the linear
        vector, and the vendor gives it its own dead zone.

        Gating the vector makes *more* commands execute, not fewer: ``(0.040, 0.040)``
        m/s used to fall under both per-axis gates and emit nothing. So
        ``input_deadband.linear_m_s`` is not a small-command filter — it is the
        commanded magnitude at which a full-speed primitive fires. Set it from that.

        The shared navigator uses positive lateral velocity and positive yaw for left. The
        vendor moving-mode axes use positive raw values for right, so those two axes invert.
        """
        forward_sign, lateral_sign, yaw_sign = self._signs(vx, vy, yaw)
        return AxisValues(
            forward=self._primitive(
                forward_sign, self.forward_positive, self.forward_negative, "forward",
            ),
            lateral=self._primitive(
                lateral_sign, self.lateral_negative, self.lateral_positive, "lateral",
            ),
            yaw=self._primitive(
                yaw_sign, self.yaw_negative, self.yaw_positive, "yaw",
            ),
        )

    def executed_velocity(self, vx: float, vy: float, yaw: float) -> ExecutedVelocity:
        """The velocity the LEGS will receive for this command, in the command's units.

        ⚠️ **THIS IS THE QUESTION EVERY SAFETY CHECK ABOVE THIS TRANSPORT WAS ASKING
        THE WRONG WAY.** :meth:`map_velocity` answers "which bytes go on the wire", and
        the bytes carry no magnitude. Nothing between the planner and the legs converted
        those bytes back into metres per second, so a stopping-distance cap, a
        feasibility rollout and a gait floor were each applied to the number the planner
        typed rather than to the speed the robot would produce. Probed against the
        shipped example profile — 0.05 m/s linear deadband, a forward primitive
        evidenced at 0.30 m/s — 0.049 m/s emits nothing and 0.050 m/s walks at 0.30, as
        do 0.10, 0.20, 0.34 and 0.55. A crawl and a lunge are the same command here.

        Derived from the SAME :meth:`_signs` as :meth:`map_velocity`, so the speed and
        the raw value can never come from different primitives; see
        :data:`_DELIVERING_PRIMITIVE` for the two rows that cross, and why.

        Raises the same :class:`AxisProfileError` :meth:`map_velocity` does for a
        direction with no evidenced primitive, because a command this transport cannot
        express has no executed velocity to report either.

        A primitive that fires with no ``measured_m_s``/``measured_rad_s`` entry gives
        ``nan`` on its axis and its name in
        :attr:`ExecutedVelocity.unmeasured`. That is not a failure of this method: the
        profile genuinely does not say, ``axis_primitive_probe.py`` is what makes it
        say, and reporting a plausible number instead is how a measurement becomes a
        preference. ``measured_rad_s`` is undeclared on every profile in this repository
        today and deliberately so — that probe refuses to time yaw while
        ``Segment.yaw_change_deg`` can report a turn through pi backwards — so the yaw
        axis reaches this branch on a real robot and the forward axis should not.
        """
        forward_sign, lateral_sign, yaw_sign = self._signs(vx, vy, yaw)
        speeds = self.measured_speeds
        executed = []
        unmeasured = []
        for axis, sign, positive, negative in (
            ("forward", forward_sign, self.forward_positive, self.forward_negative),
            ("lateral", lateral_sign, self.lateral_negative, self.lateral_positive),
            ("yaw", yaw_sign, self.yaw_negative, self.yaw_positive),
        ):
            # Called for its refusal, not its value: a direction with no evidenced
            # primitive must fail here exactly as it fails in `map_velocity`, or a
            # caller could validate a command the transport will then refuse to send.
            self._primitive(sign, positive, negative, axis)
            if sign == 0:
                executed.append(0.0)
                continue
            name = _DELIVERING_PRIMITIVE[(axis, sign)]
            speed = speeds.get(name)
            if speed is None:
                unmeasured.append(name)
                executed.append(math.nan)
            else:
                # `sign * speed`, and it is the sign of the NAVIGATOR's axis: every
                # `measured_*` entry is a positive magnitude (`_validate_measured`
                # refuses anything else), and `_DELIVERING_PRIMITIVE` has already
                # resolved which side's magnitude this is.
                executed.append(sign * float(speed))
        return ExecutedVelocity(executed[0], executed[1], executed[2],
                                tuple(unmeasured))

    def _signs(self, vx: float, vy: float, yaw: float) -> tuple[int, int, int]:
        """The three navigator-frame signs this command reduces to, and nothing else.

        Shared by :meth:`map_velocity` and :meth:`executed_velocity` so that the raw
        value sent and the speed predicted are two readings of one decision. Deriving
        them twice is how they would come to disagree about a boundary case — the
        octant snap and the two deadbands each have one — and a disagreement there is a
        safety check validating a different direction from the one the legs take.
        """
        for name, value in (("forward", vx), ("lateral", vy), ("yaw", yaw)):
            if not math.isfinite(value):
                raise AxisProfileError(f"{name} navigation component must be finite")
        forward_sign, lateral_sign = self._linear_direction(vx, vy)
        yaw_sign = 0 if abs(yaw) < self.yaw_deadband_rad_s else (1 if yaw > 0.0 else -1)
        return forward_sign, lateral_sign, yaw_sign

    def _linear_direction(self, vx: float, vy: float) -> tuple[int, int]:
        """Gate the linear pair on its magnitude, then snap it to the nearest of eight."""
        if math.hypot(vx, vy) < self.linear_deadband_m_s:
            return 0, 0
        # Half-up, not round(): round() breaks ties to even, which resolves the boundary
        # at 22.5 degrees clockwise and the one at 112.5 counter-clockwise. One rule at
        # every boundary is worth one addition. (Which way a tie goes does not matter;
        # a float landing exactly on one is not a case worth a test.)
        octants = math.floor(math.atan2(vy, vx) / (math.pi / 4.0) + 0.5)
        return _LINEAR_DIRECTIONS[octants % len(_LINEAR_DIRECTIONS)]

    @staticmethod
    def _primitive(sign: int, positive: int | None, negative: int | None,
                   name: str) -> int:
        if sign == 0:
            return 0
        primitive = positive if sign > 0 else negative
        if primitive is None:
            direction = "positive" if sign > 0 else "negative"
            raise AxisProfileError(
                f"axis profile has no physically evidenced {direction} {name} primitive"
            )
        return primitive


@dataclass(frozen=True)
class SignOnlyAxisTransport:
    """What the planner has to know about this transport, and nothing about this robot.

    ⚠️ **THE PLANNER'S SEAM FOR ISSUE #145.** ``avoidance.DynamicWindowPlanner`` samples
    velocities, rolls each one forward and refuses the ones that end inside something.
    Every one of those rollouts is of a velocity this transport does not execute: the
    mapping is sign-only, so a sampled 0.05 m/s and a sampled 0.55 m/s are the same
    0.30 m/s of legs. The planner therefore validated a robot that does not exist —
    a 0.05 m/s crawl past a bin 0.72 m away is a full-speed walk into it, and
    ``is_feasible`` said yes.

    This object is the answer to the only question that fixes that: **given this
    command, what will the legs do?** The planner asks it before it rolls anything
    forward, so the geometry it checks is the geometry that happens. On a robot whose
    transport honours magnitudes the answer is "the command" and the planner's
    behaviour is unchanged to the bit — see ``avoidance.PROPORTIONAL``.

    Sequences in, sequences out, and no numpy import: this module is imported on the
    robot beside a vendor SDK and has never needed one. The planner converts.

    :attr:`is_proportional` is ``False`` and is read rather than inferred, because
    "executed == commanded for every command I happened to try" is not the same claim
    as "this transport honours magnitudes", and only the second one licenses the
    planner to skip the work.
    """

    profile: AxisProfile
    #: ``ClassVar``, not a field. As a field it would be a keyword argument that turns
    #: issue #145's whole fix off — a sign-only transport claiming to be proportional
    #: puts the planner straight back to rolling out velocities the legs never receive.
    is_proportional: ClassVar[bool] = False

    def executed(self, commands) -> tuple[list, list]:
        """``(executed rows, known flags)`` for a sequence of ``(vx, vy, wz)`` commands.

        A ``False`` flag means this command has no executed velocity anyone can name,
        so the planner must not choose it. Two ways to get one, and they are different
        failures:

        * **A LINEAR primitive fires with no ``measured_m_s``.** The robot will move,
          at a speed nobody has timed, and no rollout of it means anything.
          ``Lite3Bindings`` refuses a live run in this state rather than leaving it to
          be discovered here — ``axis_primitive_probe.py`` exists to produce exactly
          that field, and the deployment SOP tells the operator to paste it in.
        * **The YAW primitive fires with no ``measured_rad_s`` WHILE THE ROBOT IS ALSO
          TRANSLATING.** Then the arc's shape is unknown and so is where it ends.

        A pure turn with an unmeasured rate is allowed, and that carve-out is load
        bearing rather than a convenience: ``measured_rad_s`` is undeclared on every
        profile in this repository, deliberately (``axis_primitive_probe.py`` refuses to
        time yaw while ``Segment.yaw_change_deg`` can report a turn through pi
        backwards), and the deployment SOP's live command runs at ``--max-wz 0.90``.
        Refusing every command that turns would leave a robot that can only walk in a
        straight line, which is a worse failure than the one being fixed. A pure turn's
        rollout is a POINT — ``_rollout`` holds ``x`` and ``y`` constant when ``vx`` and
        ``vy`` are zero — so its positions do not depend on the rate at all, and the
        rate reaches only the heading cost. The rule that generalises: **an unmeasured
        axis may influence cost, never geometry.**

        The requested ``wz`` is what such a row carries, because the cost function needs
        a number and this one is at least the operator's intent. It is not a prediction
        and nothing safety-bearing reads it.

        An unknown row is returned as a stop rather than as ``nan``. ``nan`` would
        poison ``gap_m`` reporting for every candidate through ``gaps.max()``, and a
        stop is a rollout the geometry can actually evaluate; the ``known`` flag is what
        makes the row unreachable, and it is applied where feasibility is computed so it
        cannot be forgotten one branch at a time.
        """
        rows = []
        known = []
        for vx, vy, wz in commands:
            try:
                result = self.profile.executed_velocity(vx, vy, wz)
            except AxisProfileError:
                # A DIRECTION THIS PROFILE CANNOT EXPRESS, e.g. a turn on a profile with
                # no evidenced yaw primitive. `set_velocity` raises for that and must:
                # there the command is real and the caller has to be told. Here the
                # caller is a planner asking a hypothetical about 330 sampled
                # velocities, and "this transport has no such command" is an answer
                # rather than an error. Raising would abort the control loop on a tick
                # where the right outcome is simply not to choose that candidate.
                #
                # `Lite3Bindings._validate_axis_profile_for_envelope` already refuses a
                # live run whose envelope enables an axis the profile cannot drive, so
                # this is the second line rather than the first — but the first one is a
                # pre-flight, and a pre-flight that is ever bypassed must not leave the
                # loop able to crash.
                rows.append((0.0, 0.0, 0.0))
                known.append(False)
                continue
            turning_blind = bool(result.unmeasured) and (
                result.translates or not self._only_yaw(result.unmeasured))
            if turning_blind:
                rows.append((0.0, 0.0, 0.0))
                known.append(False)
                continue
            rows.append((result.vx, result.vy,
                         wz if result.unmeasured else result.yaw))
            known.append(True)
        return rows, known

    @staticmethod
    def _only_yaw(unmeasured: tuple[str, ...]) -> bool:
        return all(name in YAW_PRIMITIVES for name in unmeasured)

    def describe(self) -> str:
        """One line naming the executable set, for the top of a run log.

        A run that never says this leaves ``--gait-floor`` looking like the bottom of a
        range. On this transport there is no range — see issue #42, which is the same
        confusion one field along.
        """
        speeds = self.profile.measured_speeds
        forward = speeds.get("forward_positive")
        parts = ["sign-only transport: the executable forward set is "
                 + ("{0} and one UNMEASURED primitive speed" if forward is None
                    else f"{{0, {forward:.3f}}} m/s — TWO VALUES, not a range")]
        # Named by the direction the ROBOT goes, not by the primitive's name: nav +y is
        # left and the primitive that delivers it is `lateral_negative`. Both sides are
        # reported when both are measured, because a profile may have one and not the
        # other and an omitted line reads as an axis that does not exist.
        for side, name in (("left", "lateral_negative"), ("right", "lateral_positive")):
            if name in speeds:
                parts.append(f"{side} strafe {speeds[name]:.3f} m/s")
        if not any(name in speeds for name in YAW_PRIMITIVES):
            parts.append("yaw rate NOT MEASURED, so a turn is never combined with a step")
        return "; ".join(parts)


@dataclass(frozen=True)
class _AxisSetpoint:
    values: AxisValues
    updated_at: float


class AxisStreamSender:
    """Continuously send the freshest valid three-axis command with TTL-based zeroing."""

    def __init__(self, *, host: str = DEFAULT_MOTION_HOST,
                 port: int = DEFAULT_COMMAND_PORT, source_address: str | None = None,
                 local_port: int = DEFAULT_LOCAL_PORT, axis_rate_hz: float = AXIS_RATE_HZ,
                 heartbeat_hz: float = HEARTBEAT_HZ, command_ttl_s: float = COMMAND_TTL_S,
                 stop_seconds: float = STOP_SECONDS, socket_factory=socket.socket,
                 clock=time.monotonic, sleep=time.sleep) -> None:
        self._validate_rate(axis_rate_hz, AXIS_RATE_HZ, "axis rate")
        self._validate_rate(heartbeat_hz, 2.0, "heartbeat rate")
        if not math.isfinite(command_ttl_s) or not 0.0 < command_ttl_s < 0.25:
            raise ValueError("command TTL must be finite, positive, and below the 250 ms watchdog")
        if not math.isfinite(stop_seconds) or stop_seconds <= 0.0:
            raise ValueError("stop seconds must be finite and positive")
        if not 0 <= local_port <= 65535:
            raise ValueError("local port must be within 0..65535")

        self._host = host
        self._port = port
        self._source_address = source_address
        self._local_port = local_port
        self._axis_interval = 1.0 / axis_rate_hz
        self._heartbeat_interval = 1.0 / heartbeat_hz
        self._command_ttl_s = command_ttl_s
        self._stop_seconds = stop_seconds
        self._socket_factory = socket_factory
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._setpoint = _AxisSetpoint(AxisValues(), float("-inf"))
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._failure: OSError | None = None

    @staticmethod
    def _validate_rate(value: float, minimum: float, name: str) -> None:
        if not math.isfinite(value) or value < minimum:
            raise ValueError(f"{name} must be finite and at least {minimum:g} Hz")

    @property
    def local_port(self) -> int:
        """The actual bound source port."""
        return self._local_port

    def start(self) -> None:
        """Bind once and start the independent 20 Hz axis/heartbeat stream."""
        if self._socket is not None:
            raise RuntimeError("axis stream is already running")
        sock = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self._source_address or "0.0.0.0", self._local_port))
        except OSError:
            sock.close()
            raise
        self._socket = sock
        self._local_port = sock.getsockname()[1]
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="lite3-axis-stream", daemon=True)
        self._thread.start()

    def set_axes(self, values: AxisValues) -> None:
        """Publish the newest intended axes; the worker zeroes them after the command TTL."""
        if self._socket is None:
            raise RuntimeError("start() first")
        self._raise_if_failed()
        with self._lock:
            self._setpoint = _AxisSetpoint(values, self._clock())

    def stop(self) -> None:
        """Replace the current setpoint with zero axes without stopping the safety streamer."""
        if self._socket is None:
            return
        with self._lock:
            self._setpoint = _AxisSetpoint(AxisValues(), self._clock())

    def shutdown(self) -> None:
        """Stop the worker, stream zeros for the cleanup interval, and close the socket."""
        sock = self._socket
        if sock is None:
            return
        self.stop()
        self._running.clear()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._socket = None

        cleanup_error = None
        try:
            self._send_zeros_for_duration(sock)
        except OSError as error:
            cleanup_error = error
        finally:
            sock.close()
        self._raise_if_failed()
        if cleanup_error is not None:
            raise cleanup_error

    def effective_axes(self, now: float | None = None) -> AxisValues:
        """Return the latest command only while it is younger than the application TTL."""
        now = self._clock() if now is None else now
        with self._lock:
            setpoint = self._setpoint
        return setpoint.values if now - setpoint.updated_at <= self._command_ttl_s else AxisValues()

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(f"axis stream failed: {self._failure}") from self._failure

    def _run(self) -> None:
        sock = self._socket
        if sock is None:
            return
        next_axis = self._clock()
        next_heartbeat = next_axis
        while self._running.is_set():
            now = self._clock()
            try:
                self._send_axes(sock, self.effective_axes(now))
                if now >= next_heartbeat:
                    self._send_heartbeat(sock)
                    while next_heartbeat <= now:
                        next_heartbeat += self._heartbeat_interval
            except OSError as error:
                self._failure = error
                self._running.clear()
                return

            next_axis += self._axis_interval
            now = self._clock()
            while next_axis <= now:
                next_axis += self._axis_interval
            self._sleep(max(0.0, next_axis - now))

    def _send_zeros_for_duration(self, sock: socket.socket) -> None:
        deadline = self._clock() + self._stop_seconds
        next_axis = self._clock()
        next_heartbeat = next_axis
        errors: list[OSError] = []
        while self._clock() < deadline:
            now = self._clock()
            try:
                self._send_axes(sock, AxisValues())
                if now >= next_heartbeat:
                    self._send_heartbeat(sock)
                    while next_heartbeat <= now:
                        next_heartbeat += self._heartbeat_interval
            except OSError as error:
                errors.append(error)
            next_axis += self._axis_interval
            now = self._clock()
            while next_axis <= now:
                next_axis += self._axis_interval
            self._sleep(min(max(0.0, next_axis - now), max(0.0, deadline - now)))
        if errors:
            raise OSError(f"{len(errors)} zero-axis cleanup frame(s) failed") from errors[-1]

    def _send_axes(self, sock: socket.socket, values: AxisValues) -> None:
        sock.sendto(axis_packet(FORWARD_AXIS_CODE, values.forward), (self._host, self._port))
        sock.sendto(axis_packet(LATERAL_AXIS_CODE, values.lateral), (self._host, self._port))
        sock.sendto(axis_packet(YAW_AXIS_CODE, values.yaw), (self._host, self._port))

    def _send_heartbeat(self, sock: socket.socket) -> None:
        packet = struct.Struct("<3I").pack(HEARTBEAT_CODE, 0, 0)
        sock.sendto(packet, (self._host, self._port))


class Lite3AxisLocomotion(Lite3UdpLocomotion):
    """Navigator locomotion interface with inherited state decoding and profile-gated axes."""

    def __init__(self, *, axis_profile: AxisProfile | None,
                 axis_source_address: str | None = None,
                 axis_local_port: int = DEFAULT_LOCAL_PORT,
                 axis_rate_hz: float = AXIS_RATE_HZ,
                 heartbeat_hz: float = HEARTBEAT_HZ,
                 command_ttl_s: float = COMMAND_TTL_S,
                 streamer_factory=AxisStreamSender, **kwargs) -> None:
        super().__init__(**kwargs)
        self._axis_profile = axis_profile
        self._axis_source_address = axis_source_address
        self._axis_local_port = axis_local_port
        self._axis_rate_hz = axis_rate_hz
        self._heartbeat_hz = heartbeat_hz
        self._command_ttl_s = command_ttl_s
        self._streamer_factory = streamer_factory
        self._streamer: AxisStreamSender | None = None
        #: The raw axes the most recent ``set_velocity`` mapped to — what the transport
        #: actually accepted, after the sign-only profile mapping has discarded the
        #: commanded magnitudes. ``None`` until the first call. Read by the shared
        #: navigator's telemetry, which otherwise records the requested velocity and
        #: can never show what reached the wire.
        self._last_axes: AxisValues | None = None

    def transport_axes(self) -> dict | None:
        """The raw axes the last accepted command mapped to, or ``None`` before it.

        A method rather than an attribute so a caller on a DIFFERENT backend can
        ``getattr`` for it without an attribute-error dance — the Go2 transport has no
        such notion, and telemetry must never be the thing that ends a run.
        """
        axes = self._last_axes
        if axes is None:
            return None
        return {"forward": axes.forward, "lateral": axes.lateral, "yaw": axes.yaw}

    def connect(self) -> None:
        """Connect the inherited state reader without retaining a legacy command socket."""
        super().connect()
        legacy_socket, self._command_socket = self._command_socket, None
        if legacy_socket is not None:
            legacy_socket.close()

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Map velocity intent to profile primitives and stream only while state is fresh."""
        profile = self._axis_profile
        if profile is None:
            raise AxisProfileError("a local axis profile is required before axis commands")
        axes = profile.map_velocity(vx, vy, vyaw)
        # Recorded BEFORE the early return, so a zero command is as visible in the
        # telemetry as a moving one — the record is of what the backend ACCEPTED, and
        # it accepted a stop.
        self._last_axes = axes
        if axes.is_zero and self._streamer is None:
            return
        if not axes.is_zero:
            age = self.state_age()
            if age is None or age > self._state_timeout_s:
                raise Lite3LinkLost(
                    f"the Lite3 state stream has been silent for "
                    f"{'ever' if age is None else f'{age:.2f}s'}; refusing to command "
                    "axis motion blind"
                )
            self.assert_axis_state_ready()
        if self._streamer is None:
            self._streamer = self._streamer_factory(
                host=self._motion_host,
                port=self._command_port,
                source_address=self._axis_source_address,
                local_port=self._axis_local_port,
                axis_rate_hz=self._axis_rate_hz,
                heartbeat_hz=self._heartbeat_hz,
                command_ttl_s=self._command_ttl_s,
            )
            self._streamer.start()
        self._streamer.set_axes(axes)

    def assert_axis_state_ready(self) -> None:
        """Require the documented manual/moving state before nonzero axis motion.

        The freshness bound is part of the gate, not an extra: ``prepare_motion`` calls
        this at pre-flight with no age check ahead of it, so a link that died between
        ``connect()`` and the gate would otherwise authorise motion from a frozen
        ``basic=6`` recorded seconds earlier. ``set_velocity`` checks the age first and
        then calls this, so the check is redundant on that path and load-bearing on the
        other.
        """
        profile = self._axis_profile
        if profile is None:
            raise AxisProfileError("a local axis profile is required before axis commands")
        state = self._require_fresh_state()
        basic, gait, policy, motion = state.mode
        if state.error_state != 0:
            raise Lite3LinkLost(f"Lite3 error_state={state.error_state}; refusing axis motion")
        if basic != 6:
            raise Lite3LinkLost(
                f"Lite3 basic_state={basic}; axis motion requires documented force-control state 6"
            )
        # ``None`` means this firmware omits ``robot_policy_state`` from its state frame
        # (see ``_ROBOT_STATE_NO_POLICY``), so the check is unenforceable rather than
        # passed: there is no measurement to gate on. The rest of the gate -- error_state,
        # force-control basic 6, the profile's gait set, and motion -- still applies, and
        # it is those that bound the motion this authorises.
        if policy is not None and policy != 0:
            raise Lite3LinkLost(
                f"Lite3 policy_state={policy}; profile-gated manual moving mode requires policy 0"
            )
        if gait not in profile.allowed_gait_states:
            raise Lite3LinkLost(
                f"Lite3 gait_state={gait}; axis profile allows {profile.allowed_gait_states}"
            )
        if motion not in (0, 1):
            raise Lite3LinkLost(
                f"Lite3 motion_state={motion}; refusing axis motion outside stationary/stepping"
            )

    def stop(self) -> None:
        """Set zero axes immediately; cleanup streaming occurs in :meth:`shutdown`."""
        if self._streamer is not None:
            self._streamer.stop()

    def shutdown(self) -> None:
        """Zero/disarm the axis stream before releasing inherited state resources."""
        streamer, self._streamer = self._streamer, None
        try:
            if streamer is not None:
                streamer.shutdown()
        finally:
            super().shutdown()


def axis_locomotion_factory(*, axis_profile: AxisProfile | None,
                            axis_source_address: str | None = None,
                            axis_local_port: int = DEFAULT_LOCAL_PORT,
                            axis_rate_hz: float = AXIS_RATE_HZ,
                            heartbeat_hz: float = HEARTBEAT_HZ,
                            command_ttl_s: float = COMMAND_TTL_S,
                            motion_host: str = DEFAULT_MOTION_HOST,
                            command_port: int = DEFAULT_COMMAND_PORT,
                            state_port: int = DEFAULT_STATE_PORT,
                            bind: str = "0.0.0.0"):
    """Build the shared navigator factory for the profile-gated simple-axis transport."""

    def factory(*, cmd_vel_topic=None, odom_topic=None, stamped=None, node_name=None):
        return Lite3AxisLocomotion(
            axis_profile=axis_profile,
            axis_source_address=axis_source_address,
            axis_local_port=axis_local_port,
            axis_rate_hz=axis_rate_hz,
            heartbeat_hz=heartbeat_hz,
            command_ttl_s=command_ttl_s,
            motion_host=motion_host,
            command_port=command_port,
            state_port=state_port,
            bind=bind,
        )

    return factory
