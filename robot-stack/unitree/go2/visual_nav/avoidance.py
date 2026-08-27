# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Window planner that avoids obstacles *where they are going to be*.

``lib/navigation.py`` already plans in this repo — A* over an inflated occupancy grid
from a LiDAR cloud. That is the right tool for walls and furniture and the wrong one
for a walking person: a grid is a snapshot, so a person is re-planned around after
they have already moved, and the robot chases their trail. Avoiding a mover needs the
prediction to happen in *velocity* space, which is what this module does and why it
sits alongside rather than inside the A* planner.

Each control tick it samples reachable ``(vx, vy, wz)`` commands, rolls each one
forward for a few seconds, rolls every tracked person forward on their estimated
velocity for the same few seconds, and scores the pair. The robot therefore commits to
a gap that will still be open when it gets there — it walks BEHIND someone crossing
left-to-right rather than into the space they are vacating.

WHAT IT DOES WHEN NOTHING IS SAFE, and why. The tempting answer is "take the
highest-clearance command", but that lunges a 15 kg robot sideways on a monocular
estimate at the exact moment its estimate is worst, and this robot has no rear or side
sensing to lunge into. So the policy is **swerve early, stop late**: the horizon is
long enough (:data:`PlannerConfig.horizon_s`) that the graceful sidestep happens while
there is still room, and once no sampled command clears the hard gap the planner
commands a stop. A stationary robot is also the case a person handles best — people
walk around stopped obstacles instinctively, and a freeze is legible in a way that a
sudden dodge is not.

Two further safety properties are enforced here rather than left to the caller:

  * **Never outrun the stopping distance.** Forward speed is capped at
    ``sqrt(2·a·gap)``, so the robot is always able to stop inside the space it can
    currently see to be clear.
  * **Uncertainty widens obstacles.** The tracker's position sigma is added to each
    obstacle's radius, so a person who has not been seen for a second is given a
    berth that grows with how stale the estimate is.

THE GAIT FLOOR IS A THIRD, AND IT IS ABOUT WHAT THE ROBOT CAN DO RATHER THAN WHAT IT
SHOULD. Slowing down is how this planner buys itself room to steer, and on a robot with a
minimum walking speed that is how it stops: below :attr:`Limits.gait_floor` the legs stop
swinging, the robot stands still, and nothing faults. Those two facts are incompatible in
tight space, which is issue #26, and until it the stack could not even tell that they had
met — a crawl and a stop are the same stationary robot, and the only difference in the
record was a number you had to know the platform to read.

So :meth:`DynamicWindowPlanner._gait_floor_stop` watches every command that leaves, and
when one has been under the floor for :data:`SUB_FLOOR_WINDOW_S` while the POSE went
nowhere it stops the robot on purpose and says which speed it refused. It does not scale,
clamp, filter or invent a velocity — four attempts to do that are recorded in that
method, all measured and all worse than the stall — and it triggers on a fact rather than
on the floor's value, because that value is a guess this repository's own evidence
contradicts.

Body frame ``+x`` forward / ``+y`` left; planning frame is the estimator's odom frame.
Pure numpy, no robot needed — ``python3 test_avoidance.py``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from geometry import wrap_pi

#: How long an emitted command may stay under the gait floor while the robot's POSE goes
#: nowhere before :meth:`DynamicWindowPlanner._gait_floor_stop` commands a stop, seconds.
#:
#: SHORTER THAN THE STACK'S OWN STALL GATE ON PURPOSE. ``visual_nav.PROGRESS_WINDOW_S``
#: is 4.0 s and when it fires it ENDS THE RUN naming the tether, so an explanation that
#: arrives after that outcome line is an explanation nobody reads. Long enough that a
#: legitimate standing start is not caught: the accel-limited ramp from rest to the floor
#: is seven ticks, about 0.7 s at the nominal 10 Hz.
SUB_FLOOR_WINDOW_S = 2.0

#: Fraction of the commanded travel the pose must actually cover over that window.
#: ``visual_nav.PROGRESS_FRACTION``'s value, low on purpose in both places: the failure
#: being caught is total, not marginal — 0.012 m of a commanded 0.17 m.
SUB_FLOOR_PROGRESS_FRACTION = 0.20

#: Soft gap for a MAPPED STATIC obstacle, metres — the counterpart to
#: :attr:`PlannerConfig.soft_gap_m`, which is a person's. The soft gap measures how
#: unpredictable a thing is, not how big it is: a bin cannot step sideways, so what it
#: needs is a smooth path around it rather than a berth, and the berth is paid for in
#: corridor width this robot has no way to sense. Measured in the closed-loop test, a
#: robot giving the staged bin a person's 1.20 m swung 1.76 m off the direct line to
#: clear an obstacle needing 0.63 m — in the 1.5 m lane it was staged in, that is a wall
#: rather than a detour.
#:
#: Measured again against the closed-loop test after a live run walked into that wall:
#: below about 0.25 m this stops mattering at all, because the swerve is then set by the
#: HARD constraint (obstacle radius + hard_gap_m + robot_radius_m = 0.90 m for the staged
#: bin) rather than by comfort. 0.45 was costing 0.11 m of extra lateral travel for
#: nothing. Anyone wanting a narrower pass than 0.90 m has to argue with a safety margin,
#: not with this number.
STATIC_SOFT_GAP_M = 0.20

#: Hard gap for a MAPPED STATIC obstacle, metres — the counterpart to
#: :attr:`PlannerConfig.hard_gap_m`, which is a person's. Smaller because the margin is
#: added on top of a radius that ALREADY carries the landmark's position uncertainty, so
#: a person's value counts the same caution twice, and because the thing it guards
#: against — the obstacle moving into the gap after the robot has committed — is
#: something a person can do and a bin cannot.
STATIC_HARD_GAP_M = 0.12


#: ⚠️ THE GO2 WILL NOT WALK BELOW THIS. Commanded forward speeds under about this value
#: do not produce a gait: the robot stands up, takes a few asymmetric one-or-two-leg
#: steps, and then stands perfectly still while still being commanded forward. It does
#: not fall over and it reports no fault — it simply does not go.
#:
#: MEASURED 2026-08-14, on carpet, with the 3.15 kg D1 arm fitted:
#:
#:   | commanded | outcome                                              | runs |
#:   | 0.21 m/s  | 0.04-0.08 m/s in bursts, then 3 s at 0.0 deg of knee | 5    |
#:   |           | swing. Travelled 0.34-0.43 m and stopped.            |      |
#:   | 0.35 m/s  | 0.23-0.25 m/s sustained, continuous stepping, 2.07 m | 1    |
#:   |           | in 9 s, arrived at the goal.                         |      |
#:
#: Anything BETWEEN 0.21 and 0.35 is untested — this is the lowest speed observed to
#: work, not a measured threshold, so treat it as the floor rather than as a boundary.
#:
#: Why this is worth a constant rather than a footnote: the failure is indistinguishable
#: from being physically stuck. The joint encoders read 0.0 deg of swing, the state
#: estimator correctly reports no motion, and :data:`PROGRESS_FRACTION` then fires with
#: "something is holding the robot — check the tether". Five runs and two controllers
#: were spent on tethers, walls and obstacle-avoidance settings before the speed was
#: tested, because every instrument agreed and every one of them was pointing away from
#: the cause. `--derate` reaches the same place by a different road.
MIN_GAIT_COMMAND_M_S = 0.35

#: THE GO2'S forward ceiling, m/s — the arm-fitted conservative profile of ``SKILL.md``.
#: It is the same number as :data:`MIN_GAIT_COMMAND_M_S`, and that is not a coincidence
#: to be tidied away: this robot's floor and its demo ceiling meet, so the Go2 has
#: exactly ONE usable forward speed and ``--derate`` below 1.0 walks it off the bottom of
#: its own gait. The 0.35 itself is measured — 5 of 5 runs stalled at 0.21, one run
#: walked 2.07 m in 9 s at 0.35 — and contested by this repository's own evidence:
#: `evidence/2026-08-18-threading-two-bins/` sustained 0.295 m/s for 54 of 54 ticks.
#: That is issue #26, not this constant.
GO2_MAX_VX_M_S = 0.35

#: THE GO2'S strafe ceiling, m/s. It is also this robot's measured LATERAL GAIT FLOOR:
#: on 2026-08-19, vy 0.15 travelled 0.010 m with no gait and vy 0.20 walked 3 of 3
#: (0.076-0.087 m each) against a forward control in the same session — issue #42, and
#: the attribution issues #83 and #101 had to put back after it was written down as a
#: Lite3 measurement. Both floors meeting their own axis limit is what makes
#: ``mappo_drive``'s floor ELLIPSE and the envelope ellipse the same curve.
GO2_MAX_VY_M_S = 0.20

#: THE GO2'S yaw ceiling, rad/s, from the same arm-fitted profile. Unlike the two above
#: this is a CHOSEN setting bracketed by measurements rather than a measured point: on
#: this robot 0.30 commanded achieves 0.02-0.04 and below ~0.4 (``SPIN_DEADBAND_RAD_S``)
#: it does not reliably initiate a turn at all, 0.80 achieves 0.45-0.49, and 1.50
#: saturates at 0.55-0.58. Nothing was run at 0.70. See ``SKILL.md``.
GO2_MAX_WZ_RAD_S = 0.70

#: Value of :attr:`Command.feasible` / :attr:`Command.evaluated` when the producer did
#: not state them. NOT ``0``, and that is the whole point: ``0`` is a REAL answer this
#: planner gives — ``feasible=0`` is the hold branch reporting that nothing cleared the
#: hard gap, i.e. the robot is boxed in — so defaulting to it makes "nobody said" and
#: "boxed in" the same bytes in the telemetry, and the second is an alarm.
#:
#: That is not hypothetical. `evidence/2026-08-17-corridor-and-room-runs/` records
#: ``feasible=0 evaluated=0`` on all 58 policy-driven ticks of the successful run, beside
#: vetoed ticks reading ``330 of 330``, because one branch of ``MappoPlanner.plan`` did
#: not forward what the planner had already counted. Issue #20 fixed those branches; this
#: constant is what makes the NEXT one visible rather than plausible, and
#: ``test_avoidance.test_every_branch_that_returns_a_command_states_its_search`` is what
#: makes it fail. Negative because no count can be, so a consumer needs no convention
#: beyond ``< 0``.
COUNT_NOT_RECORDED = -1

#: Every word this planner puts in :attr:`Command.reason`. A closed vocabulary, and the
#: only thing :func:`base_reason` will strip a qualifier down to.
PLANNER_REASONS = ("goal", "avoid", "hold", "arrived")


def base_reason(reason: str) -> str:
    """The planner's own word for why, with a wrapper's qualifier removed.

    A wrapper may QUALIFY a reason rather than replace it, so that a log says both what
    the planner decided and who overrode it: ``integration/mappo_drive.py`` returns the
    planner's own command with ``reason="veto-hold"`` when the policy's command was
    refused and the planner's answer was a hold. The full string is what belongs in
    telemetry. It is not what belongs in an ``if``, and this repository shipped the
    difference five times in three files — :meth:`DynamicWindowPlanner.plan` used
    ``last_reason == "hold"`` and ``last_reason == "avoid"`` for its two Schmitt
    triggers; ``visual_nav.blocked_stop``'s ``reason == "hold"`` was what started the
    rest-after-blocked timer; and ``mappo_bridge`` read the same word twice, once to
    decide whether a MOVER was holding the robot. ``"veto-hold" != "hold"``, so on every
    policy-driven run all five were inert and the Go2 stood braced under a 3.15 kg arm
    until the run timed out. Nothing faulted; both stall gates decline a zero command by
    construction.

    STRICTER THAN A SUBSTRING TEST, deliberately. ``"hold" in reason`` also fires on
    ``"threshold"`` and on ``"holding_pattern"`` — a different decision, taken because
    two strings share five letters, which is a worse bug than the one it fixes. This
    strips at most ONE leading ``<qualifier>-``, and only when what follows is exactly a
    word in :data:`PLANNER_REASONS`. So ``veto-hold`` becomes ``hold``; ``holding``,
    ``threshold``, ``policy`` and ``veto-policy`` are all returned unchanged and are
    compared as themselves. ``test_avoidance`` pins the whole table.
    """
    _, separator, tail = reason.partition("-")
    return tail if separator and tail in PLANNER_REASONS else reason


@dataclass(frozen=True)
class Limits:
    """Velocity and acceleration envelope the planner may command.

    ⚠️ ``max_vx`` has a FLOOR as well as a ceiling — see :data:`MIN_GAIT_COMMAND_M_S`.
    Derating below it (``--derate``, ``--max-vx``) produces a robot that stands still.

    ⚠️ **THE THREE VELOCITY DEFAULTS ARE ONE ROBOT'S MEASUREMENTS, NOT A NEUTRAL
    ENVELOPE.** They are the Unitree Go2's, named above so that a reader of ``Limits()``
    can see whose they are; every other field here is a planner property that is not
    platform-specific in the same way. A second platform that takes this class as its
    envelope inherits a Go2 measurement silently, and this repository has now shipped
    that mistake four times (issues #83, #96, #101). The seam where a platform blanks
    them is its bindings' ``add_navigation_arguments``; see
    ``deep_robotics/lite3/visual_nav/robot_bindings.py``, which sets ``--max-vx`` /
    ``--max-vy`` / ``--max-wz`` to ``None`` beside ``--robot-radius`` for exactly this
    reason.

    They are NOT ``None``-defaulted here, which is the ``#96`` treatment given to
    ``FisheyeCamera.height_m``. That was tried and measured: a bare ``None`` default
    takes 5 test files down and 184 of 1113 tests with them, and two of those files are
    ``integration/test_closed_loop_sim.py`` and ``integration/test_mappo_drive.py``,
    whose ``Limits()`` calls are the Go2 drive path legitimately asking for the Go2's
    envelope. A default that is correct for the platform this file belongs to is not
    the defect. Being inherited without being named is.

    Accelerations bound how far the command may move in ONE control period; that is
    what makes the sampled window "dynamic" and keeps the vendor gait controller from
    being handed a step it cannot follow.
    """

    max_vx: float = GO2_MAX_VX_M_S    # m/s forward. Reverse is never sampled (below).
    max_vy: float = GO2_MAX_VY_M_S    # m/s strafe — the Go2 can crab, which is the
    #                                   cheapest sidestep available to it.
    max_wz: float = GO2_MAX_WZ_RAD_S  # rad/s yaw
    accel_x: float = 0.50         # m/s^2
    accel_y: float = 0.40         # m/s^2
    accel_wz: float = 1.50        # rad/s^2

    #: THE SLOWEST FORWARD SPEED THIS ROBOT CAN ACTUALLY WALK AT, m/s, or ``0.0`` for a
    #: platform that has not measured one. See :data:`MIN_GAIT_COMMAND_M_S`, which is the
    #: Go2's and is this default for the same reason the three ceilings above are Go2
    #: numbers — this file is the Go2's stack, and a default that is correct for the
    #: platform it belongs to is not the defect. A Lite3 states its own with
    #: ``--gait-floor``, which its bindings already require on a live run.
    #:
    #: NOT A SECOND CEILING AND NOT PART OF THE ENVELOPE. The envelope says what the
    #: planner is ALLOWED to command; this says what the robot is ABLE to execute, and
    #: :meth:`DynamicWindowPlanner._gait_floor_stop` is the only thing that reads it.
    #: The two are independent: ``--derate 0.6`` puts the ceiling 0.21 UNDER this
    #: floor, which is a real configuration and is why nothing here may assume ordering.
    gait_floor: float = MIN_GAIT_COMMAND_M_S

    def scaled(self, factor: float) -> Limits:
        """A uniformly derated envelope (the arm-fitted conservative profile).

        ``gait_floor`` IS NOT SCALED, and that is the whole point of it being here rather
        than beside the ceilings. A derate is an instruction to the planner to ask for
        less; it is not a change to the robot, and a floor that shrank with the envelope
        would make ``--derate 0.6`` report a 0.21 m/s command — the speed measured to
        stall 5 runs of 5 — as comfortably above the floor. Scaling it is how a physical
        measurement quietly becomes a preference.
        """
        return Limits(max_vx=self.max_vx * factor, max_vy=self.max_vy * factor,
                      max_wz=self.max_wz * factor, accel_x=self.accel_x * factor,
                      accel_y=self.accel_y * factor, accel_wz=self.accel_wz * factor,
                      gait_floor=self.gait_floor)


@dataclass(frozen=True)
class PlannerConfig:
    """Rollout geometry, clearances and cost weights."""

    horizon_s: float = 2.5        # long enough that the swerve beats the freeze
    dt_s: float = 0.125           # rollout step
    samples_vx: int = 6
    samples_vy: int = 5
    samples_wz: int = 11

    robot_radius_m: float = 0.40  # Go2 is ~0.70 x 0.31 m; half-diagonal, rounded up
    obstacle_radius_m: float = 0.35   # a person's personal footprint
    hard_gap_m: float = 0.25      # free space that must remain between the two discs

    # Below this, closeness starts costing. A PERSON's comfort distance, which is why
    # it is this large and why Obstacle carries a per-obstacle override: the soft gap
    # is about how unpredictable a thing is, not how big it is. Applied to a 0.30 m bin
    # it is actively harmful — measured in the closed-loop test, a robot giving a bin
    # 1.20 m of soft gap detoured 1.76 m off the direct line to clear an obstacle that
    # needs 0.63 m, which in a real 1.5 m corridor is a wall rather than a detour.
    soft_gap_m: float = 1.20

    # Gap below which the chosen path is REPORTED as `avoid`. Reporting only, never
    # planning — it is deliberately not the soft gap, because that number sizes the
    # berth and this one decides what a human reads in the log and the overlay. They
    # were the same field until tightening the berth for a narrow corridor silently
    # stopped a run that visibly swerved around a bin from ever saying `avoid`, which is
    # the one word that makes the footage legible. A berth is a control decision; a label
    # is an explanation, and tuning one must not quietly delete the other.
    avoid_report_gap_m: float = 1.20

    # Extra margin required to CHANGE the reported reason, in metres of gap — a Schmitt
    # trigger on both `hold` and `avoid`. Observed live: `hold` and `avoid` alternated
    # on 9 consecutive ticks, and the simulated approach to the staged bin flipped
    # `goal`/`avoid` 44 times. No cost weight can damp either. `hold` is not a sampled
    # candidate at all — it is a fallback taken when the feasible set is empty, so it
    # never competes on cost — and `avoid` is a LABEL applied after the choice, so
    # weight_smooth cannot see it. Both oscillations are a bare threshold sitting on the
    # noise of a size-prior gap estimate, and the fix is the standard one: make changing
    # your mind cost more than keeping it. Sized at a third of the hard gap, comfortably
    # above frame-to-frame range jitter and well inside one tick of travel.
    reason_hysteresis_m: float = 0.08

    # Costs are summed after each is normalised to roughly 0..1, so these weights are
    # directly comparable. Goal dominates in the open; clearance dominates near people.
    weight_goal: float = 1.0
    weight_heading: float = 0.6
    weight_clearance: float = 2.0
    weight_speed: float = 0.25
    weight_smooth: float = 0.15

    decel_for_stopping_m_s2: float = 0.5   # used for the stopping-distance speed cap


@dataclass(frozen=True)
class Obstacle:
    """An obstacle in odom coordinates, as the planner sees it.

    ``radius_m`` is already inflated for uncertainty by whoever built this — the planner
    treats it as a hard footprint. A static obstacle is simply one with zero velocity;
    the rollout maths needs no special case for it.
    """

    x: float
    y: float
    vx: float
    vy: float
    radius_m: float
    label: str = "person"      # for logs and the overlay; the planner ignores it
    #: Whether this obstacle must STOP the robot rather than be routed to the MAPPO
    #: policy. Judged on box SHAPE, not on ``label`` — see
    #: ``person_detector.RangedDetection.person_shaped``. The PLANNER ignores this too;
    #: it exists for ``mappo_bridge.holds_the_robot``. Defaults True so any producer
    #: that does not set it lands on the stopping side.
    person_shaped: bool = True
    #: Which subsystem produced this — ``"tracked"`` (a mover from the tracker) or
    #: ``"static"`` (a landmark from the map). The planner ignores it, but a consumer
    #: must not have to infer it. ``label`` cannot stand in: it is a CLASS name, and it
    #: separates the two here only because this scene happens to have one mapped prop
    #: and one detector class. A stopped person still has ``label="person"`` and a
    #: velocity of zero, and telling those apart is the difference between "path around
    #: it" and "wait for it to move".
    kind: str = "tracked"
    #: Stable identity across ticks, ``None`` if the producer has none. A consumer that
    #: re-associates by position instead will merge two objects that pass within its
    #: matching threshold. Both producers already have one — ``Track.track_id`` and
    #: ``Landmark.landmark_id`` — so this only carries what exists.
    object_id: str | None = None
    #: Distance below which closeness starts costing, for THIS obstacle. ``None`` takes
    #: the planner's default, which is a person's. A landmark wants a much smaller one:
    #: a bin that cannot move needs a smooth path around it, not a wide berth, and the
    #: berth is what the robot spends corridor width on.
    soft_gap_m: float | None = None
    #: Free space that must REMAIN between the two discs for a path to be allowed at all,
    #: for THIS obstacle. ``None`` takes the planner's default.
    #:
    #: Per-obstacle for the same reason the soft gap is, and the reason is stronger here:
    #: this margin is added ON TOP of a radius that already carries the object's position
    #: uncertainty, so applying a person's value to a mapped landmark counts the caution
    #: twice. Measured live — the robot stopped with a reported gap of 0.21 m against a
    #: 0.25 m threshold, which was 0.43 m of actual air between robot and bin, in a lane
    #: where it needed every centimetre. A person can step sideways into the gap while
    #: the robot is committed to it; a bin cannot, and that difference is exactly what
    #: this margin is for.
    hard_gap_m: float | None = None


@dataclass(frozen=True)
class Command:
    """A velocity command plus why the planner chose it."""

    vx: float
    vy: float
    wz: float
    #: Why this command was chosen: one of :data:`PLANNER_REASONS`, or one of those
    #: QUALIFIED by a wrapper that overrode the planner (``mappo_drive`` writes
    #: ``veto-hold``). Compare it with :func:`base_reason`, never with ``==`` against a
    #: bare word and never with ``in``; log it whole.
    reason: str
    gap_m: float           # predicted worst free gap over the horizon (inf if clear)
    #: How many sampled commands cleared the hard gap, and how many were sampled. They
    #: describe the PLANNER's search, so a wrapper that returns a command of its own has
    #: to forward the search it wrapped rather than let these default —
    #: :data:`COUNT_NOT_RECORDED` says "not stated", and ``0`` says "boxed in".
    feasible: int = COUNT_NOT_RECORDED
    evaluated: int = COUNT_NOT_RECORDED
    #: WHAT SEPARATES A DELIBERATE STOP FROM A COMMAND THAT BECAME ONE. ``None`` on
    #: every ordinary tick. Set only on a stop the gait-floor guard commanded, to the
    #: sub-floor speed the robot was measured not to walk at
    #: (:meth:`DynamicWindowPlanner._gait_floor_stop`).
    #:
    #: Three states a reader could not tell apart before issue #26, all of which leave a
    #: stationary robot in the record and one of which is an alarm:
    #:
    #:   * ``is_stop`` and this is ``None`` — boxed in, or told to stop. Nothing cleared
    #:     the hard gap, or a wrapper zeroed the command for a reason of its own.
    #:   * ``is_stop`` and this is set — A FLOOR STOP. The robot was commanded
    #:     ``floor_reach_m_s`` m/s for :data:`SUB_FLOOR_WINDOW_S`, went nowhere, and is
    #:     now being stopped on purpose. It is not the tether.
    #:   * ``vx`` under the floor and this is ``None`` — A CREEP, and still a legitimate
    #:     command: the planner chose a slow speed with faster ones in its window, and
    #:     run C of 2026-08-18 walked 3 m doing exactly that. The guard is watching it.
    #:
    #: It describes the PLANNER's verdict, so a wrapper that returns a command of its own
    #: forwards it exactly as it forwards :attr:`feasible` and :attr:`evaluated`, and
    #: ``test_avoidance.test_every_branch_that_returns_a_command_states_its_search``
    #: fails on a branch that does not.
    floor_reach_m_s: float | None = None

    @property
    def is_stop(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0 and self.wz == 0.0

    @property
    def search_recorded(self) -> bool:
        """Whether this command states the search that produced it.

        ``False`` means the producer did not forward the counts, NOT that the search
        found nothing — see :data:`COUNT_NOT_RECORDED`.
        """
        return (self.feasible != COUNT_NOT_RECORDED
                and self.evaluated != COUNT_NOT_RECORDED)


class DynamicWindowPlanner:
    """Samples reachable velocities and scores them against predicted obstacle motion."""

    def __init__(self, limits: Limits | None = None,
                 config: PlannerConfig | None = None) -> None:
        self.limits = limits or Limits()
        self.config = config or PlannerConfig()
        #: The gait-floor guard's only state; see :meth:`_gait_floor_stop`. NOT named
        #: ``_sub_floor``: ``integration/mappo_drive.MappoPlanner`` subclasses this and
        #: already binds that name to its own reporting window, so the two collided
        #: silently — its ``__init__`` runs after ``super().__init__`` and replaced this
        #: deque with ``None``, which surfaced as ``AttributeError: 'NoneType' object has
        #: no attribute 'clear'`` rather than as a wrong answer, but only because a deque
        #: and a list are not the same shape. ``plan`` is
        #: otherwise a pure function of its arguments, which is why ``last_reason`` is
        #: passed in rather than remembered, and this is the deliberate exception: the
        #: guard's question — "has this robot moved while being commanded to?" — cannot
        #: be answered from one tick. Both are cleared by :meth:`reset_gait_floor_guard`.
        self._floor_samples: deque = deque()
        self._floor_stalled: float | None = None

    # ── The gait-floor guard ───────────────────────────────────────
    def reset_gait_floor_guard(self) -> None:
        """Forget everything the guard has observed. Call between runs."""
        self._floor_samples.clear()
        self._floor_stalled = None

    def _gait_floor_stop(self, chosen: tuple[float, float, float],
                         pose: tuple[float, float, float], now: float) -> float | None:
        """``None``, or the sub-floor speed this robot has just PROVEN it cannot walk at.

        ⚠️ **THE GUARD FOR ISSUE #26, AND THE ONLY THING IN THIS PLANNER THAT OVERRIDES
        A CHOSEN COMMAND.** Below :attr:`Limits.gait_floor` the robot produces no gait,
        stands still, and reports NO FAULT — the joint encoders read 0.0 deg of swing,
        the state estimator agrees, and four seconds later the stack's stall gate fires
        with *"something is holding the robot — check the tether"*. Every instrument is
        right and every one of them points away from the cause.

        MEASURED 2026-08-27, Go2 run ``20260827T012702Z-00652ea``, live. 31 of 37
        planner-issued ticks were under the floor and 90% of all commanded ticks were.
        Perception ran at ~3 Hz against a 10 Hz loop, so ``perception_timeout_s``
        commanded a stop on 26 of 86 ticks and each one re-anchored :meth:`_window` at
        zero; the accel-limited ramp restarted eighteen times and never once got past
        0.104 m/s. The robot moved 0.012 m in 3.4 s, 0.72 m from a bin, and the run ended
        blaming the tether. The stopping-distance cap was NOT the constraint — it stood
        at 0.38-0.39 m/s throughout, above the floor.

        WHAT IT TRIGGERS ON IS A FACT, NOT THE FLOOR'S VALUE, and that is the whole
        design. ``MIN_GAIT_COMMAND_M_S`` is a GUESS — "the lowest speed observed to work"
        on 2026-08-14 — and this repository's own run C of 2026-08-18 contradicts it: a
        sustained 0.295 m/s with 54 of 54 ticks below 0.35, minimum 0.189, 3 m walked,
        arrived. **A rule that refused sub-floor commands on their value would have
        killed that run.** So the trigger is *commanded to move and not moving*, and the
        floor is used only to decide which of the two explanations to believe: above the
        floor and stationary is something holding the robot, below it and stationary is
        this. That makes the guard independent of the number issue #26 proposal 2 exists
        to replace.

        MEASURED FROM POSE, NOT FROM THE VELOCITY ESTIMATE, for the reason
        ``lateral_floor_probe.py`` gives: the reported velocity on this unit has been
        caught reading 0.17 m/s of noise as signal, which is larger than the whole
        0.137 m/s signal being judged. Net displacement between two poses is the same
        quantity the stack's own stall gate compares against.

        IT CAN ONLY EVER MAKE A COMMAND SLOWER. Issue #26 records two attempts to make a
        sub-floor command FASTER, both measured and both reverted, and a third was
        measured on 2026-08-27 while writing this:

        * Filtering sub-floor speeds out of the window leaves only zero from rest.
        * Allowing them only while ramping up drops the lateral offset around a bin from
          the required 0.88 m to 0.55 m and clips it.
        * Replacing a standstill's whole forward window with the floor — so the robot
          steps off at 0.35 rather than crawling at 0.05 — makes it refuse to START.
          ``test_closed_loop_sim.test_the_planner_reaches_the_goal_past_the_bin`` went to
          ``timeout`` with ``path_m=0.0``: one control period of the only walkable speed
          commits the robot to 0.875 m over the 2.5 s feasibility horizon, and a bin
          1.3 m away makes every candidate infeasible, so it held for all 600 ticks.

        And a fourth, which is the one that looks most like what an issue asks for:
        refusing any chosen command under the floor and stopping instead. Rolled against
        ``test_avoidance.test_it_commits_to_one_side_rather_than_splitting_the_
        difference``, the planner wanting 0.34 m/s had it zeroed, could then only offer
        itself 0.35 from the resulting standstill, found that infeasible at that range,
        and **stopped 1.60 m from the bin for the remaining 588 ticks** — 0.04 m of
        lateral offset against the 0.88 m the geometry needs. Refusing against a floor
        that is 1.85x too high refuses speeds this robot demonstrably walks at.

        So: nothing is scaled, nothing is filtered, and no velocity is invented. The one
        thing this does is write down the zero the legs were going to produce anyway,
        two seconds before the stall gate misattributes it.

        NOT A COST TERM AND NOT A SAMPLED CANDIDATE. :meth:`_window` deliberately holds
        no stop row, because a stationary rollout never approaches anything and its
        clearance term is zero by construction, so it is unbeatable near a hazard — the
        measured example is in ``_window``'s own comment. This is a branch taken after
        the choice, exactly like the two holds above it, so "do nothing" still never
        competes with "do something" on cost.

        Latches, and clears the moment the planner's own choice reaches the floor again —
        so a person walking out of the way frees the robot on the next tick without
        anything having to time out, and a robot that is genuinely wedged stays stopped
        rather than resuming the crawl every two seconds.
        """
        floor = self.limits.gait_floor
        if floor <= 0.0:
            return None                       # no floor measured for this robot
        speed = math.hypot(chosen[0], chosen[1])
        walkable = speed >= floor
        if self._floor_stalled is not None:
            # Latched. Only a command that reaches the floor lets the legs go again;
            # anything else is a speed already proven not to move this robot.
            if walkable:
                self._floor_stalled = None
                self._floor_samples.clear()
                return None
            return self._floor_stalled
        if speed <= 0.0 or walkable:
            # A stop is a stop and a walkable command walks. Neither is evidence about
            # the floor, and both end the run of sub-floor ticks.
            self._floor_samples.clear()
            return None

        self._floor_samples.append((now, pose[0], pose[1], speed))
        # SLIDING, NOT TUMBLING, and the difference decided whether this guard was worth
        # writing. A window that resets on every verdict throws away the ticks between
        # one verdict and the next arming, and it judges a collapse against travel from
        # before the collapse. Replayed against the 2026-08-27 run, a tumbling window
        # spent its first 2.3 s excusing the crawl with 0.18 m of coasting that happened
        # while the command was still 0.24 m/s, re-armed 0.44 s late, and fired at
        # t=13.37 — 0.11 s before the run ended on the stall gate's own clock, which is
        # to say it fired too late to be read. Sliding, the same run fires at t=12.03.
        #
        # Pruned by discarding a sample only once the one BEHIND it has aged past the
        # window, which keeps the newest sample that is still a full window old. Dropping
        # everything older than the window instead — the obvious spelling — leaves the
        # oldest survivor younger than the window by construction, so "have I got a full
        # window yet?" can only pass on a tick landing exactly on the boundary. That
        # spelling is why `visual_nav._blocked_reason` once sat through a 12 s stall
        # without judging it, and a unit test with dt=0.5 hid it because 0.5 divides 4.0.
        while (len(self._floor_samples) > 1
               and now - self._floor_samples[1][0] >= SUB_FLOOR_WINDOW_S):
            self._floor_samples.popleft()
        span = now - self._floor_samples[0][0]
        if span < SUB_FLOOR_WINDOW_S:
            return None
        # Integrated over the samples rather than `speed * span`, which is the stack's
        # model and takes the LAST tick's command as though it had been held all window.
        # At an irregular ~3 Hz under a planner that is actively slowing down those are
        # different numbers: the 2026-08-27 run fell from 0.245 to 0.052 across one tick.
        # Each sample's command governs the interval that FOLLOWS it, because the loop
        # issues a command and then sleeps.
        samples = list(self._floor_samples)
        travel = sum(held * (later[0] - when)
                     for (when, _x, _y, held), later in zip(samples, samples[1:]))
        moved = math.hypot(pose[0] - self._floor_samples[0][1],
                           pose[1] - self._floor_samples[0][2])
        if moved >= SUB_FLOOR_PROGRESS_FRACTION * travel:
            # Sub-floor AND getting somewhere: run C of 2026-08-18, 54 of 54 ticks under
            # the floor and 3 m walked. Nothing is latched and the window keeps sliding,
            # so a robot that recovers and stalls again is judged again.
            return None
        self._floor_stalled = float(speed)
        return self._floor_stalled

    # ── Sampling ───────────────────────────────────────────────────────
    def _window(self, current: tuple[float, float, float], control_dt: float
                ) -> np.ndarray:
        """Candidate ``(vx, vy, wz)`` triples reachable within one control period.

        ``current`` is the last COMMANDED velocity, not the measured one: the vendor
        gait controller lags its target by design, and feeding the lag back in would
        ratchet the window shut and stop the robot ever reaching commanded speed.
        """
        cfg, lim = self.config, self.limits
        vx0, vy0, wz0 = current

        # Reverse is deliberately not sampled. The Go2 has no rear-facing sensing on
        # this unit, so backing away from a person means moving blind into space this
        # pipeline has never observed.
        vx = np.linspace(max(0.0, vx0 - lim.accel_x * control_dt),
                         min(lim.max_vx, vx0 + lim.accel_x * control_dt),
                         cfg.samples_vx)
        vy = np.linspace(max(-lim.max_vy, vy0 - lim.accel_y * control_dt),
                         min(lim.max_vy, vy0 + lim.accel_y * control_dt),
                         cfg.samples_vy)
        wz = np.linspace(max(-lim.max_wz, wz0 - lim.accel_wz * control_dt),
                         min(lim.max_wz, wz0 + lim.accel_wz * control_dt),
                         cfg.samples_wz)

        grid = np.stack(np.meshgrid(vx, vy, wz, indexing="ij"), axis=-1)
        # DELIBERATELY NOT a full-stop row appended here, though one used to be, "so
        # the planner can never be forced to keep moving". It did two things wrong.
        #
        # It broke the dynamic window. Cruising at 0.30 m/s the reachable set is
        # [0.25, 0.35]; a zero row is a command the robot cannot execute in one control
        # period, which is the single constraint this whole sampling scheme exists to
        # respect.
        #
        # And it competed on COST, where standing still is unbeatable: a stationary
        # rollout never approaches anything, so its clearance term is zero by
        # construction while every real command pays. Measured on the staged scene —
        # bin 2.15 m ahead, goal behind it, the lane passable and all 331 candidates
        # feasible — the stop row scored 1.317 against 1.393 for the best moving
        # command, and the planner returned ``reason="goal"`` with ``v=(0,0,0)``. That
        # is worse than a hold: the log claims the robot is driving at the goal, and
        # because the reason is not "hold" the rest-after-blocked timer never starts,
        # so it stands there under load until the run times out.
        #
        # Stopping is still always available, and is still never forced: both `hold`
        # branches in `plan` command a full stop directly, which is where an
        # acceleration-limit-breaking emergency stop belongs. Decelerating for a normal
        # obstacle is handled better without the row — the stopping-distance cap
        # ratchets the window down over successive ticks, which is a smooth slowdown
        # rather than a step to zero.
        return grid.reshape(-1, 3)

    # ── Rollout ─────────────────────────────────────────────────────────────
    def _rollout(self, candidates: np.ndarray, pose: tuple[float, float, float]
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Forward-simulate every candidate.

        Returns ``(xy, yaw)`` with shapes ``(N, K, 2)`` and ``(N, K)`` for K steps.
        Body velocities are held constant, so each path is an arc; integrating rather
        than closed-forming keeps strafe-plus-yaw correct.
        """
        cfg = self.config
        steps = max(1, round(cfg.horizon_s / cfg.dt_s))
        x0, y0, yaw0 = pose
        n = candidates.shape[0]

        vx, vy, wz = candidates[:, 0], candidates[:, 1], candidates[:, 2]
        xy = np.empty((n, steps, 2))
        yaws = np.empty((n, steps))

        x = np.full(n, x0)
        y = np.full(n, y0)
        yaw = np.full(n, yaw0)
        for k in range(steps):
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            x = x + (vx * cos_y - vy * sin_y) * cfg.dt_s
            y = y + (vx * sin_y + vy * cos_y) * cfg.dt_s
            yaw = yaw + wz * cfg.dt_s
            xy[:, k, 0], xy[:, k, 1] = x, y
            yaws[:, k] = yaw
        return xy, yaws

    def _gaps(self, xy: np.ndarray, obstacles: list[Obstacle]) -> np.ndarray:
        """Worst free gap over the horizon, per candidate PER OBSTACLE: ``(N, M)``.

        "Gap" is surface-to-surface: centre distance minus both radii. Obstacles are
        advanced on their estimated velocity at each step, which is the whole point —
        the robot is scored against where the person WILL be.

        Kept per-obstacle rather than collapsed to a single minimum because the soft gap
        is per-obstacle: a candidate 0.5 m from a bin and 2 m from a person is close for
        one and comfortable for the other, and a single worst-gap number cannot say
        which. Feasibility, which uses one hard gap for everything, takes the row
        minimum at the call site.
        """
        steps = xy.shape[1]
        if not obstacles:
            return np.full((xy.shape[0], 0), math.inf)

        times = (np.arange(steps) + 1) * self.config.dt_s          # (K,)
        centres = np.array([[o.x, o.y] for o in obstacles])         # (M, 2)
        velocities = np.array([[o.vx, o.vy] for o in obstacles])    # (M, 2)
        radii = np.array([o.radius_m for o in obstacles])           # (M,)

        # (M, K, 2): each obstacle's predicted position at each rollout step.
        predicted = centres[:, None, :] + velocities[:, None, :] * times[None, :, None]
        # (N, M, K): distance from candidate n at step k to obstacle m at step k.
        delta = xy[:, None, :, :] - predicted[None, :, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        gaps = distance - radii[None, :, None] - self.config.robot_radius_m
        return gaps.min(axis=2)

    def _soft_gaps(self, obstacles: list[Obstacle]) -> np.ndarray:
        """Each obstacle's soft gap, defaulting to the planner's, shape ``(M,)``."""
        default = self.config.soft_gap_m
        return np.array([default if o.soft_gap_m is None else o.soft_gap_m
                         for o in obstacles], dtype=float)

    def _hard_gaps(self, obstacles: list[Obstacle]) -> np.ndarray:
        """Each obstacle's hard gap, defaulting to the planner's, shape ``(M,)``."""
        default = self.config.hard_gap_m
        return np.array([default if o.hard_gap_m is None else o.hard_gap_m
                         for o in obstacles], dtype=float)

    @staticmethod
    def _worst_gap(gaps: np.ndarray) -> np.ndarray:
        """Collapse ``(N, M)`` per-obstacle gaps to the worst per candidate, ``(N,)``."""
        if gaps.shape[1] == 0:
            return np.full(gaps.shape[0], math.inf)
        return gaps.min(axis=1)

    def current_gap(self, pose: tuple[float, float, float],
                    obstacles: list[Obstacle]) -> float:
        """Free gap to the nearest obstacle right now (``inf`` if there are none).

        Evaluated at t=0, unlike :meth:`_gaps` which scores a whole rollout, because
        the stopping-distance cap must reflect the space that is clear *now*.
        """
        if not obstacles:
            return math.inf
        return min(
            math.hypot(o.x - pose[0], o.y - pose[1])
            - o.radius_m - self.config.robot_radius_m
            for o in obstacles)

    def is_feasible(self, pose: tuple[float, float, float],
                    command: tuple[float, float, float], obstacles: list[Obstacle],
                    horizon_s: float | None = None) -> bool:
        """Whether holding ``command`` from ``pose`` keeps every obstacle's hard gap.

        The same feasibility test :meth:`plan` applies to every sampled candidate, asked
        about ONE command instead. That is the natural question for a supervisory veto
        over an external controller — let the other controller choose, and refuse only
        what this planner's own geometry says ends inside something. Without it a
        consumer has to reach into :meth:`_rollout`, :meth:`_gaps` and
        :meth:`_hard_gaps`, and then a rename here becomes an ``AttributeError`` there.

        Per-obstacle hard gaps, like :meth:`plan`: a mapped landmark and a person do not
        want the same margin, which is why :class:`Obstacle` carries the override.

        ``horizon_s`` defaults to :attr:`PlannerConfig.horizon_s`, and the default is the
        one to use. A shorter window is tempting for a command that will be re-chosen
        100 ms later — "what if the robot held this, blind, for 2.5 s" is not a question
        anyone is actually asking — but it was swept downstream against a static obstacle
        and shortening it cost collisions without even firing less often. A veto that
        intervenes late lets the robot get closer, where it then has to intervene more.
        The parameter is here so that argument can be re-run, not so it can be tuned
        casually.
        """
        if not obstacles:
            return True
        xy, _ = self._rollout(np.asarray([list(command)], dtype=float), pose)
        if horizon_s is not None:
            # Truncating the path is equivalent to re-rolling with a shorter horizon and
            # cheaper: _gaps derives its per-step obstacle prediction times from
            # xy.shape[1], so slicing keeps the moving obstacles consistent with it.
            xy = xy[:, :max(1, round(horizon_s / self.config.dt_s)), :]
        return bool(np.all(self._gaps(xy, obstacles)[0] >= self._hard_gaps(obstacles)))

    # ── Planning ────────────────────────────────────────────────────────────
    def plan(self, pose: tuple[float, float, float], goal: tuple[float, float],
             last_command: tuple[float, float, float], obstacles: list[Obstacle],
             control_dt: float = 0.1, last_reason: str = "goal",
             now: float | None = None) -> Command:
        """Choose a velocity command for this tick.

        Args:
            pose: robot ``(x, y, yaw)`` in odom.
            goal: target ``(x, y)`` in odom.
            last_command: previously commanded ``(vx, vy, wz)``.
            obstacles: predicted obstacles, radius already inflated for uncertainty.
            control_dt: seconds until the next command — sets the dynamic window.
            last_reason: the previous tick's :attr:`Command.reason`. Only the two
                hysteresis decisions below read it, and passing it keeps :meth:`plan` a
                pure function of its arguments — the alternative, a ``self._holding``
                flag, would make every test depend on call order. It may arrive
                QUALIFIED by a wrapper, so it is read through :func:`base_reason`.
            now: monotonic seconds, defaulting to the clock. Read ONLY by the gait-floor
                guard, which is the one thing here that needs more than this tick — an
                argument rather than a call inside it so a test can drive the guard
                without sleeping, which is the difference between a guard that has been
                watched fire and one that has not.
        """
        cfg = self.config
        clock = time.monotonic() if now is None else now
        candidates = self._window(last_command, control_dt)
        xy, yaws = self._rollout(candidates, pose)
        per_obstacle_gaps = self._gaps(xy, obstacles)
        gaps = self._worst_gap(per_obstacle_gaps)

        # Starting is harder than continuing: see PlannerConfig.reason_hysteresis_m.
        # Read through `base_reason` because a wrapper may hand back the word it got
        # with a qualifier on it. `last_reason == "hold"` was False for every tick of
        # every policy-driven run, where the previous reason is `veto-hold`, so both
        # Schmitt triggers were off exactly where the chatter was first measured:
        # against a 0.35 m blocker at 0.85-0.92 m this planner holds on `hold` and
        # leaves the hold on `veto-hold`.
        previous = base_reason(last_reason)
        margin = cfg.reason_hysteresis_m if previous == "hold" else 0.0
        if obstacles:
            # Per-obstacle: a candidate is feasible when it clears EVERY obstacle by
            # that obstacle's own hard gap, not when its worst gap clears a single
            # planner-wide number that was chosen for people.
            required = self._hard_gaps(obstacles)[None, :] + margin
            feasible = (per_obstacle_gaps >= required).all(axis=1)
        else:
            feasible = np.ones(gaps.shape[0], dtype=bool)
        if not feasible.any():
            # Swerve early, stop late — see the module docstring.
            # The guard sees this stop too. Not for the verdict — a hold is not evidence
            # about the floor — but because it has to CLEAR the window, and a `return`
            # above it leaves one open across the hold. The travel integral then credits
            # the last sub-floor command to the seconds the robot spent deliberately
            # stationary, which inflates the expected distance and makes the guard fire
            # on a robot that was told to stand still. It fails in the dangerous
            # direction, so it goes through the same call as everything else.
            return Command(0.0, 0.0, 0.0, reason="hold",
                           gap_m=float(gaps.max()), feasible=0,
                           evaluated=int(candidates.shape[0]),
                           floor_reach_m_s=self._gait_floor_stop(
                               (0.0, 0.0, 0.0), pose, clock))

        # Never commit to a speed that outruns the space known to be clear. The margin
        # is NOT added here: this cap is about stopping distance, a physical quantity,
        # and inflating it while holding would make the robot creep out of a hold more
        # slowly the longer it held.
        current_gap = self.current_gap(pose, obstacles)
        nearest_hard_gap = (float(self._hard_gaps(obstacles).min()) if obstacles
                            else cfg.hard_gap_m)
        stopping_cap = math.sqrt(max(0.0, 2.0 * cfg.decel_for_stopping_m_s2
                                     * max(0.0, current_gap - nearest_hard_gap)))
        feasible &= candidates[:, 0] <= max(stopping_cap, 1e-6)
        if not feasible.any():
            return Command(0.0, 0.0, 0.0, reason="hold", gap_m=float(current_gap),
                           feasible=0, evaluated=int(candidates.shape[0]),
                           floor_reach_m_s=self._gait_floor_stop(
                               (0.0, 0.0, 0.0), pose, clock))

        # Built once and shared: the clearance cost and the avoid decision are two
        # views of the same quantity, and deriving the soft gaps twice invites them to
        # drift apart under a later edit.
        soft_gaps = self._soft_gaps(obstacles)
        clearance = self._clearance_cost(per_obstacle_gaps, soft_gaps)
        cost = self._cost(candidates, xy, yaws, clearance, goal, pose, last_command)
        cost = np.where(feasible, cost, np.inf)
        best = int(np.argmin(cost))

        # "Avoiding" means an obstacle is close enough to be shaping the chosen path,
        # with the same hysteresis the hold decision gets: `avoid` is a label applied
        # after the choice, so weight_smooth cannot damp it and a bare threshold
        # chatters.
        avoid_threshold = cfg.avoid_report_gap_m
        if previous == "avoid":
            avoid_threshold += cfg.reason_hysteresis_m

        # THE GAIT-FLOOR GUARD, on the command that is actually going out. Read
        # `_gait_floor_stop` before touching this; the short version is that a crawl the
        # robot has been measured not to walk is a stop already, and this is where that
        # stop gets written down instead of inferred four seconds later from the wrong
        # cause. It never raises a speed and it is not a candidate, so it can neither
        # lunge nor win on cost.
        chosen = (float(candidates[best, 0]), float(candidates[best, 1]),
                  float(candidates[best, 2]))
        stalled = self._gait_floor_stop(chosen, pose, clock)
        if stalled is not None:
            # `hold`, not this tick's `avoid`/`goal` label. A stop IS a hold in this
            # vocabulary, and `hold` is the word `visual_nav.blocked_stop` reads to put
            # the Go2 prone rather than leave it standing braced under a 3.15 kg arm for
            # the rest of the run. `floor_reach_m_s` keeps the speed that was refused, so
            # the record still says which of the two kinds of stop this is.
            return Command(0.0, 0.0, 0.0, reason="hold", gap_m=float(gaps[best]),
                           feasible=int(feasible.sum()),
                           evaluated=int(candidates.shape[0]),
                           floor_reach_m_s=stalled)

        return Command(
            vx=chosen[0], vy=chosen[1], wz=chosen[2],
            reason="avoid" if gaps[best] < avoid_threshold else "goal",
            gap_m=float(gaps[best]), feasible=int(feasible.sum()),
            evaluated=int(candidates.shape[0]), floor_reach_m_s=None)

    @staticmethod
    def _clearance_cost(per_obstacle_gaps: np.ndarray,
                        soft_gaps: np.ndarray) -> np.ndarray:
        """Normalised 0..1 crowding per candidate, worst obstacle wins, ``(N,)``.

        Each obstacle is normalised by ITS OWN soft gap before the maximum is taken, so
        the number stays comparable across obstacles of very different kinds — being
        halfway inside a person's 1.20 m and halfway inside a bin's 0.45 m both score
        0.5, which is what makes one weight sensible for both.

        Same zero-length-axis guard as :meth:`_soft_slack`.
        """
        if soft_gaps.size == 0:
            return np.zeros(per_obstacle_gaps.shape[0])
        soft = soft_gaps[None, :]
        crowding = np.clip((soft - per_obstacle_gaps) / soft, 0.0, 1.0)
        return crowding.max(axis=1)

    def _cost(self, candidates: np.ndarray, xy: np.ndarray, yaws: np.ndarray,
              clearance_cost: np.ndarray, goal: tuple[float, float],
              pose: tuple[float, float, float],
              last_command: tuple[float, float, float]) -> np.ndarray:
        """Total cost per candidate. Every term is normalised to ~0..1 first.

        ``clearance_cost`` arrives already normalised from :meth:`_clearance_cost`,
        which is the one term that needs per-obstacle knowledge.
        """
        cfg, lim = self.config, self.limits
        goal_xy = np.asarray(goal, dtype=float)

        start_distance = max(float(np.hypot(goal_xy[0] - pose[0],
                                            goal_xy[1] - pose[1])), 1e-6)
        end_xy = xy[:, -1, :]
        end_distance = np.linalg.norm(end_xy - goal_xy, axis=-1)
        # Fraction of the remaining distance still left at the end of the rollout.
        # Clipped because a candidate can overshoot past the goal.
        goal_cost = np.clip(end_distance / start_distance, 0.0, 2.0)

        # Point the nose where we are going: without this the planner is happy to
        # crab sideways to the goal and arrive facing a wall, which also means the
        # camera stops looking where the robot is heading.
        bearing_to_goal = np.arctan2(goal_xy[1] - end_xy[:, 1],
                                     goal_xy[0] - end_xy[:, 0])
        heading_cost = np.abs(wrap_pi(bearing_to_goal - yaws[:, -1])) / math.pi

        speed_cost = (lim.max_vx - candidates[:, 0]) / max(lim.max_vx, 1e-6)

        # Penalise jerky command changes — the vendor gait tracks a smooth target far
        # better, and it stops the planner dithering between two equal-cost swerves.
        smooth_cost = (
            np.abs(candidates[:, 0] - last_command[0]) / max(lim.max_vx, 1e-6)
            + np.abs(candidates[:, 1] - last_command[1]) / max(lim.max_vy, 1e-6)
            + np.abs(candidates[:, 2] - last_command[2]) / max(lim.max_wz, 1e-6)
        ) / 3.0

        return (cfg.weight_goal * goal_cost
                + cfg.weight_heading * heading_cost
                + cfg.weight_clearance * clearance_cost
                + cfg.weight_speed * speed_cost
                + cfg.weight_smooth * smooth_cost)
