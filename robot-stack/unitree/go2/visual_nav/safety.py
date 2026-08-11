# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight and in-flight guards for walking this Go2 with the D1 arm fitted.

This unit carries a 3.15 kg D1 arm cantilevered over its back, and that changes what
"safe to walk" means. Two hazards are specific enough to be worth code:

**Thermal.** The hind-leg motors work against the arm's moment continuously, and a
previous session on this robot found them saturating badly enough that it could not
hold a stand for 60 s and squatted without warning. An unannounced squat with an
extended arm drives the arm into whatever it is over. :class:`HealthMonitor` watches
``rt/lowstate`` so a run ends on a temperature trend rather than on a collapse, and it
is why the navigator keeps the robot PRONE except while actually walking.

**Arm pose.** A folded arm rides over the trunk; an extended one is a swinging 3 kg
lever that changes the centre of mass and can strike furniture the camera never looked
at. :class:`ArmStowMonitor` refuses the walk unless the arm is compact, judged by
forward kinematics rather than by joint angles — a reach number means the same thing
whatever combination of joints produced it. Measured on this unit: the parked pose
puts the jaw **0.137 m** from the arm base, a zero pose 0.552 m, an extended pose
0.637 m, against a 0.733 m maximum reach. :data:`STOWED_REACH_M` sits in the wide gap
between parked and everything else.

**Arm swivel — why a stowed arm is still not a settled arm.** An unpowered D1
back-drives, and its base yaw (``angle0``) is the one joint with nothing to rest
against. Measured on this unit during a turning test, it crept **6.2° → 19.6°** — 13°
of a 3.15 kg mass swinging off-centre while the robot turned, which is a real balance
disturbance and not a rounding error. :func:`latch_arm` removes most of it.

Swivel is gated by :data:`STOWED_YAW_DEG` (3.0°, owner-set), checked as ``|angle0|``
rather than as the jaw's lateral offset — the jaw is the lightest point on the arm and
the least sensitive to the rotation that matters. The limit is ABSOLUTE, so creep
accumulates across runs: the arm measured 1.5° before this robot's first two walking
runs and 2.5° after them, which is 83% of the budget spent in 35 s of walking. Expect
to re-pose between sessions rather than assuming a latched arm stays put.

Everything here READS the arm except :func:`latch_arm`, which is opt-in and issues the
single safest command this arm has: funcode 5 damp-enable, which holds each joint at
the angle it is **already at**. It never commands a trajectory, so it cannot fling the
arm anywhere. That matters, because the alternatives are worse than they sound — a
funcode-6 "panic" *energises* this arm rather than stopping it, and discharging a
RAISED arm drops it (both catalogued in ``../d1_arm``).
"""

from __future__ import annotations

import contextlib
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# The D1 helpers live one directory up; import them by package path so this module
# does not depend on the caller's sys.path layout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOWSTATE_TOPIC = "rt/lowstate"
ARM_FEEDBACK_TOPIC = "rt/arm_Feedback"

# Conservative operating limits for the ARM-LOADED case, not vendor maxima. The point
# is to stop on a trend, long before the motors' own protection would act — with the
# arm fitted there is no margin to spend. Idle on this unit reads ~30 C.
MOTOR_TEMP_WARN_C = 55.0
MOTOR_TEMP_ABORT_C = 70.0
BATTERY_SOC_ABORT_PCT = 20.0

# Jaw distance from the arm base below which the arm counts as stowed. See the module
# docstring for the measured poses this sits between.
STOWED_REACH_M = 0.30

#: The dorsal-stow pose, hand-set on this unit and confirmed by forward kinematics.
#: Joint angles in DEGREES ``[J0 base-yaw, J1 shoulder, J2 elbow, J3 elbow-roll,
#: J4 wrist-pitch, J5 wrist-roll]``. J0 ≈ 0 is what puts the arm on the dorsal
#: centreline; the measured lateral offset at this pose is ``y = +0.002 m``.
#:
#: This is a REFERENCE, not a set-point. Nothing commands the arm to these angles —
#: the operator hand-poses it (a discharged D1 back-drives freely) and
#: :func:`latch_arm` freezes it wherever they left it. Use these numbers to check a
#: pose by eye, or to recognise when the arm has drifted out of stow.
DORSAL_STOW_ANGLES_DEG = (1.4, -90.5, 88.0, 1.3, 20.0, -0.5)

#: Jaw position at :data:`DORSAL_STOW_ANGLES_DEG`, in the arm-base frame, metres:
#: ``x`` forward, ``y`` left, ``z`` up. Reach 0.138 m against a 0.733 m maximum.
DORSAL_STOW_JAW_XYZ_M = (0.074, 0.002, 0.116)

#: Joint drift tolerated between issuing the latch and confirming it took.
LATCH_DRIFT_TOLERANCE_DEG = 2.0

#: Maximum D1 SWAY — base-yaw magnitude, in degrees — for the arm to count as stowed
#: along the dorsal line. This is the PRIMARY centreline gate. Owner-set at 3.0 deg.
#:
#: WHY BASE YAW AND NOT THE JAW'S LATERAL OFFSET. ``angle0`` is the axis the 3.15 kg
#: mass actually rotates about, and it is the unit every recorded creep measurement on
#: this robot is already expressed in. Jaw ``y`` was tried first and is a poor sensor
#: for it, measured on this unit by forward kinematics:
#:
#:   * ``y`` is not zero at ``J0 = 0``. The jaw sits -1.93 mm off axis there, so ``y``
#:     and ``J0`` disagree about where the centreline is by 1.43 deg.
#:   * ``J3`` (elbow-roll) moves the jaw 0.66 mm/deg — nearly TWICE ``J0``'s 0.34 —
#:     so ``y`` blends three joints and tracks none of them.
#:   * A folded arm is compact, so ``y`` barely responds to the rotation that matters:
#:     :data:`STOWED_LATERAL_M` at 0.05 m does not trip until ``J0`` is about 41 deg.
#:
#: MEASURED AGAINST THIS LIMIT (all on this unit, latched unless stated):
#:
#:   ===========================  ========  ==============================
#:   condition                    base yaw  vs the 3.0 deg gate
#:   ===========================  ========  ==============================
#:   reference stow               1.4 deg   passes
#:   creep over 35 s of WALKING   1.0 deg   passes (measured 2026-08-11)
#:   creep over one TURNING test  5.3 deg   FAILS — turning exceeds this gate
#:   arm unpowered, turning       13.4 deg  fails
#:   ===========================  ========  ==============================
#:
#: So a walking run holds inside 3 deg comfortably, and a turning one does not. The gate
#: is ABSOLUTE — measured from the joint zero, not from wherever the operator posed it —
#: which means creep accumulates across runs and the arm needs re-posing when it trips,
#: rather than the limit quietly following the drift.
STOWED_YAW_DEG = 3.0

#: Jaw offset from the dorsal centreline, kept as a SECONDARY cross-check on
#: :data:`STOWED_YAW_DEG`. It catches the case base yaw cannot see: a wrist or
#: elbow-roll re-posed far enough to throw the jaw out over the flank while ``J0``
#: stays near zero.
#:
#: PROVISIONAL — derived from pose geometry, not measured against gait behaviour, and
#: loose enough (J0 ~41 deg) that it is a gross-mispose backstop rather than a
#: centreline gate. Deliberately not re-derived from the 3 deg figure: that would be
#: inventing a second number from the same geometry instead of measuring one.
STOWED_LATERAL_M = 0.05


@dataclass(frozen=True)
class Health:
    """One sample of the robot's physical condition."""

    max_motor_temp_c: float
    hottest_motor: int
    battery_soc_pct: float
    sample_time: float

    @property
    def age(self) -> float:
        return time.monotonic() - self.sample_time


class HealthMonitor:
    """Watches ``rt/lowstate`` for motor temperature and battery state of charge."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._health: Health | None = None
        self._sub = None

    def start(self, wait_s: float = 3.0) -> None:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

        self._sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        self._sub.Init(self._on_state, 10)

        deadline = time.monotonic() + wait_s
        while self.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.latest() is None:
            self.stop()
            raise RuntimeError(f"no {LOWSTATE_TOPIC} in {wait_s:.1f}s — is the robot on?")

    def stop(self) -> None:
        if self._sub is not None:
            with contextlib.suppress(Exception):
                self._sub.Close()
            self._sub = None

    def _on_state(self, msg) -> None:
        # Only the 12 leg motors matter here; the message carries spare slots.
        temps = [float(m.temperature) for m in msg.motor_state[:12]]
        hottest = max(range(len(temps)), key=temps.__getitem__)
        health = Health(max_motor_temp_c=temps[hottest], hottest_motor=hottest,
                        battery_soc_pct=float(msg.bms_state.soc),
                        sample_time=time.monotonic())
        with self._lock:
            self._health = health

    def latest(self) -> Health | None:
        with self._lock:
            return self._health

    def abort_reason(self) -> str | None:
        """A string if the robot should stop walking now, else ``None``."""
        health = self.latest()
        if health is None:
            return "no rt/lowstate"
        if health.age > 2.0:
            return f"rt/lowstate stale by {health.age:.1f}s"
        if health.max_motor_temp_c >= MOTOR_TEMP_ABORT_C:
            return (f"motor {health.hottest_motor} at {health.max_motor_temp_c:.0f}C "
                    f"(limit {MOTOR_TEMP_ABORT_C:.0f}C)")
        if health.battery_soc_pct <= BATTERY_SOC_ABORT_PCT:
            return f"battery {health.battery_soc_pct:.0f}%"
        return None

    def warning_reason(self) -> str | None:
        """A string if the robot is heading for an abort, else ``None``.

        :data:`MOTOR_TEMP_WARN_C` is the number the pre-flight checklist asks an
        operator to read off by hand. Having the constant here but never evaluating it
        left that check to a human's memory of a threshold the code already knew, so
        the navigator now prints this before it walks and again whenever it crosses.
        """
        health = self.latest()
        if health is None or health.max_motor_temp_c < MOTOR_TEMP_WARN_C:
            return None
        return (f"motor {health.hottest_motor} at {health.max_motor_temp_c:.0f}C is "
                f"past the {MOTOR_TEMP_WARN_C:.0f}C warning mark "
                f"(abort at {MOTOR_TEMP_ABORT_C:.0f}C)")


class ArmStowMonitor:
    """Reads D1 joint angles and reports whether the arm is folded compactly.

    Read-only: it never commands the arm. Construction is cheap, but :meth:`start`
    loads the D1 URDF for forward kinematics.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._angles: list[float] | None = None
        self._sample_time = 0.0
        self._sub = None
        self._kinematics = None
        self._power_status: int | None = None
        self._enable_status: int | None = None
        self._error_status: int | None = None

    def powered(self) -> bool | None:
        """Whether the D1 reports its SERVO BUS live. ``None`` if it has not said.

        Measured on this robot: an arm whose comms board is up but whose servos are not
        powered publishes ``{"enable_status":0,"power_status":0,"error_status":0}`` on
        ``rt/arm_Feedback`` while still reporting joint angles at 10 Hz — so angles are
        not evidence of drive power, and neither is a successful ack.
        """
        with self._lock:
            return None if self._power_status is None else bool(self._power_status)

    def energised(self) -> bool | None:
        """Whether the D1 reports its joints ENABLED (holding under power)."""
        with self._lock:
            return None if self._enable_status is None else bool(self._enable_status)

    def status_word(self) -> str:
        """The three status flags, for logs."""
        with self._lock:
            return (f"power={self._power_status} enable={self._enable_status} "
                    f"error={self._error_status}")

    def start(self, wait_s: float = 3.0) -> None:
        from d1_arm._arm_idl import ArmString_
        from d1_arm.d1_fk import D1Kinematics
        from unitree_sdk2py.core.channel import ChannelSubscriber

        self._kinematics = D1Kinematics()
        self._sub = ChannelSubscriber(ARM_FEEDBACK_TOPIC, ArmString_)
        self._sub.Init(self._on_feedback, 10)

        deadline = time.monotonic() + wait_s
        while self.angles() is None and time.monotonic() < deadline:
            time.sleep(0.02)

    def stop(self) -> None:
        if self._sub is not None:
            with contextlib.suppress(Exception):
                self._sub.Close()
            self._sub = None

    def _on_feedback(self, msg) -> None:
        from d1_arm.d1_arm import parse_feedback

        try:
            parsed = parse_feedback(msg.data_)
        except Exception:
            return
        if parsed["kind"] == "arm_status":
            # (address 2, funcode 3) — the arm's own account of whether its servo bus is
            # live. Previously discarded along with the ack frames, which is how an
            # UNPOWERED arm passed the latch check: see LatchResult.
            data = parsed["data"] or {}
            with self._lock:
                self._power_status = data.get("power_status")
                self._enable_status = data.get("enable_status")
                self._error_status = data.get("error_status")
            return
        if parsed["kind"] != "joint_angles":
            return  # the same topic also carries ack frames
        data = parsed["data"] or {}
        try:
            angles = [float(data[f"angle{i}"]) for i in range(6)]
        except (KeyError, TypeError, ValueError):
            return
        with self._lock:
            self._angles = angles
            self._sample_time = time.monotonic()

    def angles(self) -> list[float] | None:
        """The six joint angles in DEGREES, or ``None`` if the arm is not reporting."""
        with self._lock:
            return None if self._angles is None else list(self._angles)

    def reach_m(self) -> float | None:
        """Distance from the arm base to the jaw, in metres."""
        angles = self.angles()
        if angles is None or self._kinematics is None:
            return None
        position = self._kinematics.jaw_xyz(angles)
        return float(math.sqrt(sum(float(v) ** 2 for v in position[:3])))

    def sway_deg(self) -> float | None:
        """How far the arm has swung off the dorsal line, in degrees. ``None`` if silent.

        This is ``|angle0|`` — the base-yaw magnitude — which is the axis the arm's mass
        rotates about and the unit every creep measurement on this robot is recorded in.
        See :data:`STOWED_YAW_DEG` for why this rather than the jaw's lateral offset.
        """
        angles = self.angles()
        return None if angles is None else abs(float(angles[0]))

    def blocking_reason(self, required: bool = True) -> str | None:
        """A string if the arm should stop the walk, else ``None``.

        With ``required=False`` a silent arm is accepted (it may be physically
        removed); the reach check still applies whenever the arm IS reporting.
        """
        reach = self.reach_m()
        if reach is None:
            if required:
                return (f"no {ARM_FEEDBACK_TOPIC} joint angles — cannot confirm the arm "
                        f"is stowed (pass --no-require-arm if the arm is removed)")
            return None
        if reach > STOWED_REACH_M:
            return (f"D1 arm is extended: jaw {reach:.3f} m from base, "
                    f"limit {STOWED_REACH_M:.2f} m — fold it before walking")
        sway = self.sway_deg()
        if sway is not None and sway > STOWED_YAW_DEG:
            return (f"D1 arm has swayed {sway:.1f} deg off the dorsal line, limit "
                    f"{STOWED_YAW_DEG:.1f} deg — 3.15 kg this far off centre unbalances "
                    f"the vendor gait controller. Hand-pose it flat along the spine and "
                    f"re-latch. Note the limit is ABSOLUTE, so creep accumulates across "
                    f"runs: an arm that passed last run can fail this one")
        jaw = self.jaw_xyz()
        if jaw is not None and abs(jaw[1]) > STOWED_LATERAL_M:
            return (f"D1 arm is off the dorsal centreline: jaw y={jaw[1]:+.3f} m, "
                    f"limit +-{STOWED_LATERAL_M:.2f} m — it is folded compactly enough "
                    f"to pass the reach check but its mass sits out over the flank, "
                    f"which unbalances the gait. Re-pose it along the spine")
        return None

    def jaw_xyz(self) -> tuple[float, float, float] | None:
        """Jaw position in the arm-base frame (x forward, y left, z up), metres.

        Compare against :data:`DORSAL_STOW_JAW_XYZ_M`. The ``y`` component is the
        useful one for stow: it is how far off the dorsal centreline the mass sits.
        """
        angles = self.angles()
        if angles is None or self._kinematics is None:
            return None
        position = self._kinematics.jaw_xyz(angles)
        return tuple(float(v) for v in position[:3])


def stand_up(loco, settle_s: float = 2.0, balance_s: float = 1.0) -> None:
    """Get the robot up and ready to accept Move commands.

    Two calls, not one: ``recover()`` (RecoveryStand) is what gets a prone robot onto
    its feet, and ``stand()`` (BalanceStand) is what puts it in the mode that accepts a
    velocity. The sleeps are the vendor sequences' own settling time.

    Shared by the navigator and the calibration sweep because getting an arm-loaded Go2
    up is a safety sequence, not a convenience: this is the moment the hind legs take
    the D1's moment, and the two callers had drifted into separate copies of it.

    NOTE FOR CALLERS: this blocks for ``settle_s + balance_s`` seconds. Any plan made
    before calling it is that much staler afterwards — see ``VisualNavigator.run``,
    which deliberately discards its command and re-plans rather than acting on one
    decided while the robot was still lying down.
    """
    loco.recover()          # RecoveryStand: gets up from lying
    time.sleep(settle_s)
    loco.stand()            # BalanceStand: ready to accept Move
    time.sleep(balance_s)


def lie_down(loco, settle_s: float = 0.3) -> None:
    """Stop and fold the legs. Best-effort — never raises, so it is safe on an exit path.

    The stop comes first and is given a moment to take effect: folding the legs out of a
    moving gait is how a walking robot ends up on its side, and with 3.15 kg cantilevered
    over its back that is not a recoverable posture.
    """
    try:
        loco.stop()
        time.sleep(settle_s)
        loco.stand_down()
    except Exception as exc:
        print(f"[safety] lie-down failed: {exc!r}")


@dataclass(frozen=True)
class LatchResult:
    """Outcome of :func:`latch_arm` — above all, whether the joints actually stopped.

    A structured result rather than a status line because callers have to ACT on it:
    walking with the D1 unlatched is not a warning, it is a refusal (see
    ``visual_nav.py``). ``__str__`` still renders the operator-facing line, so printing
    it reads the same as it always did.
    """

    held: bool
    drift_deg: float
    angles_deg: tuple
    jaw_xyz_m: tuple
    reach_m: float
    powered: bool | None = None      # the arm's own power_status, None if it never said
    reason: str = ""                 # why held is False

    def __str__(self) -> str:
        verdict = "HELD" if self.held else f"NOT HELD — {self.reason}"
        return (f"arm latched at {[round(a, 1) for a in self.angles_deg]} deg, jaw "
                f"x={self.jaw_xyz_m[0]:+.3f} y={self.jaw_xyz_m[1]:+.3f} "
                f"z={self.jaw_xyz_m[2]:+.3f} m (reach {self.reach_m:.3f} m), drift "
                f"{self.drift_deg:.2f} deg, power={self.powered} — {verdict}")


def latch_arm(monitor: ArmStowMonitor, iface: str = "eth0",
              init_dds: bool = False, settle_s: float = 2.5) -> LatchResult:
    """Freeze the D1 at the angles it is ALREADY holding.

    THE PROCEDURE THIS IMPLEMENTS (the operator does step 1, this does 2—4):

      1. **Hand-pose the arm.** A discharged D1 back-drives, so it is placed by hand,
         flat along the dorsal centreline and as low as it will go. Support its
         weight — do not let it drop, and never lift the robot by the arm.
      2. **Verify before latching.** Refuses if forward kinematics put the jaw beyond
         :data:`STOWED_REACH_M`. Latching an extended arm would freeze a 3 kg lever
         out over the robot's side.
      3. **Latch.** ``D1Arm.enable()`` — funcode 5 damp-enable. Each joint holds the
         angle it is at, so no trajectory is commanded and the arm cannot move to get
         there. This is the only D1 command on this unit that cannot fling it.
      4. **Confirm it took**, which needs TWO checks, not one.

    WHY A DRIFT CHECK ALONE IS VACUOUS. This function used to prove the latch by
    re-reading the angles and asserting they had stopped moving. **An arm that cannot
    move shows zero drift whether or not the latch took.** Measured on this robot: with
    the D1's servo bus unpowered, ``latch_arm`` reported ``drift 0.00 deg — HELD`` on
    every run, including the two live walking runs, while a commanded 3 deg base-yaw
    move produced 0.00 deg of motion in every mode. The hard requirement that the arm be
    latched was being satisfied by a check that could not fail.

    So the arm's own status word is now the primary evidence and drift is the secondary.
    The D1 publishes ``{"enable_status":E,"power_status":P,"error_status":X}`` on
    ``rt/arm_Feedback`` (address 2, funcode 3), and an unpowered arm reports ``P=0``
    while still streaming joint angles at 10 Hz and still ACKing commands — so neither
    angles nor an ack is evidence of drive power. Note this contradicts the previous
    comment here, which claimed ``enable_status`` was unreliable; what was actually
    happening is that the flags were being discarded by the feedback filter and never
    read at all.

    Measured on this unit: drift immediately after latching **0.10°**; base-yaw creep
    across a subsequent turning test **5.3°**, against **13.4°** for the same test with
    the arm unpowered. TREAT THOSE TWO NUMBERS WITH SUSPICION — they were recorded
    through the vacuous check, so the "latched" figure may describe an unpowered arm
    too. Re-measure once ``powered`` reports true.

    To undo it, discharge the arm — but only while it is folded low, as it is here.
    Discharging a raised arm drops it.
    """
    before = monitor.angles()
    if before is None:
        raise RuntimeError(f"no joint angles on {ARM_FEEDBACK_TOPIC} — refusing to "
                           f"latch an arm whose pose cannot be confirmed")
    blocking = monitor.blocking_reason()
    if blocking is not None:
        raise RuntimeError(f"refusing to latch: {blocking}")

    from d1_arm.d1_arm import D1Arm

    arm = D1Arm(iface=iface, init_dds=init_dds)
    arm.connect()
    time.sleep(0.5)
    arm.enable()
    time.sleep(settle_s)

    after = monitor.angles()
    if after is None:
        # A silent feed must NOT read as a successful latch. With no angles there is no
        # drift evidence at all, so this fails closed rather than reporting the zero
        # drift of an arm that simply stopped talking.
        drift, still = float("nan"), False
        reason = "the joint feed went silent, so the pose cannot be confirmed"
    else:
        drift = max(abs(a - b) for a, b in zip(after, before))
        still = drift < LATCH_DRIFT_TOLERANCE_DEG
        reason = "" if still else (f"joints drifted {drift:.2f} deg after enable, "
                                   f"tolerance {LATCH_DRIFT_TOLERANCE_DEG:.1f} deg")

    # PRIMARY evidence: the arm's own power flag. Stillness is necessary but nowhere
    # near sufficient — an unpowered arm is perfectly still. Only `powered is True` is
    # accepted; None (the arm never published a status frame) fails closed, because
    # "it did not say" must not read as "it is fine".
    powered = monitor.powered()
    if powered is not True:
        held = False
        detail = ("its servo bus is NOT powered" if powered is False
                  else "it never published a status frame")
        reason = (f"{detail} ({monitor.status_word()}), so the arm cannot hold anything "
                  f"— a still arm is not a latched arm")
    else:
        held = still

    reach = monitor.reach_m()
    return LatchResult(held=held, drift_deg=drift, angles_deg=tuple(before),
                       jaw_xyz_m=tuple(monitor.jaw_xyz() or (float("nan"),) * 3),
                       reach_m=float("nan") if reach is None else reach,
                       powered=powered, reason=reason)
