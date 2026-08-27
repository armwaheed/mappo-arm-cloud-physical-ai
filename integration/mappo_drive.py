#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Let the MAPPO policy drive a supported quadruped, with the planner keeping a veto.

⛔ **READ `robot-stack/SAFETY.md` AND `deploy/README.md` FIRST.** This is the only file in
the repository that can put the policy in charge of the legs, and it inherits every gate
`visual_nav.py` has — `--live` is still what moves the robot, the D1 latch still refuses a
run, and an operator still stays on the remote.

## What it substitutes, and what it deliberately does not

Everything except the choice of velocity. The camera, the detector, the tracker, the odom
map, the goal source, the health monitor, the arm gate, the telemetry writer, the
recording, the stall detector and the whole teardown path are the vendored stack's,
unchanged and untouched. :class:`MappoPlanner` replaces exactly one method —
``DynamicWindowPlanner.plan`` — and subclasses it so the rest of the planner's surface,
including its rollout geometry, is still there to veto with.

## The veto

The policy proposes; the planner's OWN feasibility test decides. The proposed velocity is
rolled forward over the planner's 2.5 s horizon and must keep every obstacle's hard gap.
If it does not, the planner's own command is issued instead.

Deliberately NOT "forward the planner's hold": the planner holds for the BIN as well as
for people, so that would zero the policy in the one scene it exists to solve. Rolling out
the actual command separates "the planner would rather wait" from "this command ends
inside the bin", and only the second is worth overriding.

## The gait floor, which is a second gate and not part of the envelope

The envelope is a CEILING and this robot also has a FLOOR: below about 0.35 m/s the Go2
produces no gait, stands still, and reports no fault. Until issue #26 the floor was
checked once, at startup, against the envelope's ceiling — so a per-tick command below it
was never checked at all, and the stall gate four seconds later named the tether.

Every command leaving ``plan()`` is now compared to the floor the ROBOT's bindings
report, on every path, and a run of sub-floor ticks that covers no ground says so before
the stall gate gets to misattribute it. It changes no velocity: two ways of making a
sub-floor command faster were measured and both were worse than the stall (issue #26).
``--policy-gait-floor`` still raises a sub-floor POLICY command onto the floor ellipse,
and stays opt-in.

**The DECISION now lives one level down, in the shared planner.** This file's check was
only ever on the policy-driven path, and the bug it is about reaches every platform and
both drive modes — a Lite3 run with no policy at all got nothing. So
``avoidance.DynamicWindowPlanner._gait_floor_stop`` is what stops the legs, using the
same trigger this file already used and the floor now carried on ``Limits``; what stays
here is the banner, because a guard that acts without saying why is the defect one level
along. ``_note_sub_floor`` recognises the planner's floor stop and prints for it rather
than treating it as an ordinary zero — without that branch the fix would have deleted
its own diagnostic.

## Peer robots, with no perception at all

``--peer-odom-align`` turns on a second obstacle source: peer poses published over the
Device Connect mesh by ``dashboard/robot_driver.py --publish-pose`` and spooled locally by
``dashboard/peer_link.py``. No detector, no marker, no training — and it is the faithful
deployment of this policy rather than a fallback, because the VMAS agents it was trained
against observed each other's true positions.

The peers arrive as ordinary ``kind="tracked"`` obstacles in the list ``_obstacles``
already builds, so the veto, the telemetry, the overlay and the policy all consume them
through the paths they already had. The only new decision in the control loop is what to
do when the link goes quiet, and that is a HOLD — see ``peer_source``, which argues it.

**The transform is the enabling flag on purpose.** Two robots' odom frames both start at
their own power-on pose and have no relationship until somebody measures one, so there is
no code path in which peer avoidance is on and the frames have not been declared. A
default of identity would be a claim that both robots were switched on at the same spot,
which is true at a bench and false in a room, and would put the obstacle somewhere
plausible and wrong.

``--policy-mode raw`` removes the veto. The closed-loop simulation's numbers for the two
are in `deploy/README.md`; raw collided and supervised did not, so raw is for a
deliberately empty arena and nothing else.

## How it substitutes: a supported seam, no longer a monkey-patch

`visual_nav.main()` takes a ``planner_factory``, and ``DynamicWindowPlanner`` has a public
``is_feasible``. Both landed upstream and were re-vendored, which is why this file is now
about forty lines shorter than the version that swapped three module globals and reached
into the planner's rollout internals. Everything the vendored ``main()`` does — the
arm-latch refusal, the health gate, the recorder's codec check, the telemetry header, the
teardown ordering — happens exactly as it does for an ordinary run.

The one thing the seam does not carry is the MEASURED velocity, which ``plan()``'s
signature has no room for and which matters here because this robot delivers about 0.45 of
what it is commanded. The factory closes over ``loco``, which is the shape the upstream
docstring suggests.

    python3 mappo_drive.py --live --telemetry run.jsonl --record run.mp4 \\
        --static-prop bin --goal-class chair --goal-height 1.067 \\
        --robot-radius 0.25 --no-latch-arm

Every ``visual_nav.py`` flag still applies. ``python3 test_mappo_drive.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from mappo_policy import (
    DEFAULT_PACKAGE,
    GOAL,
    SERVO_MODES,
    TRAVEL,
    VETO_HORIZON_S,
    HeadingServo,
    PolicyRunner,
    tick_from_state,
)
from peer_source import DEFAULT_SPOOL, PEER_TIMEOUT_S, Alignment, PeerSource
from turn_drive_supervisor import TurnDriveSupervisor

_STACK = Path(__file__).resolve().parent.parent / "robot-stack" / "unitree" / "go2"
# `go2/` so `locomotion.*` and `d1_arm.*` resolve as the packages the stack imports them
# as, and `visual_nav/` for this file's own sibling-module imports. NOT the package
# directories themselves: `go2/d1_arm` on the path makes `import d1_arm` find the MODULE
# `d1_arm.py` sitting inside it and shadow the namespace package, so safety.py's
# `from d1_arm._arm_idl import ArmString_` raises "not a package" — and that is the
# ARM-STOW MONITOR, which exists to refuse a run whose arm has crept off the dorsal line.
# It failed as an import error before the pre-flight, so it read as a missing dependency
# rather than as a safety check that had been disabled.
for _directory in (_STACK, _STACK / "visual_nav"):
    sys.path.insert(0, str(_directory))

from avoidance import (  # noqa: E402
    SUB_FLOOR_PROGRESS_FRACTION,
    SUB_FLOOR_WINDOW_S,
    Command,
    DynamicWindowPlanner,
    Obstacle,
)

#: Statuses that mean "stop", mapped to the reason string the vendored loop understands.
#: The word is not cosmetic there. ``visual_nav.blocked_stop`` rests the legs on a stop
#: whose reason is anything but ``arrived``, so ``STOP_GOAL_REACHED`` is the one status
#: in this table that leaves the robot standing and the other three put it prone rather
#: than braced under the arm. A status the table does not know falls back to ``hold``,
#: i.e. to resting, which is the safe direction for an unrecognised stop.
_STOP_REASONS = {
    "STOP_EXTERNAL_HOLD": "hold",
    "STOP_STALE_INPUT": "hold",
    "STOP_CLOCK_ERROR": "hold",
    "STOP_GOAL_REACHED": "arrived",
}


#: Where a refused run leaves its trace. A refusal happens BEFORE the telemetry writer
#: exists, so without this a run that never started leaves nothing behind at all — and
#: "why did the 14:32 run not happen" is then unanswerable an hour later, which on a demo
#: day is exactly when it gets asked. One JSON object per line, not prose: this repository
#: already learned that a console log is not an interface.
DEFAULT_REFUSAL_LOG = Path.home() / ".mappo-refusals.jsonl"


def _record_refusal(path: Path | None, reason: str, detail: dict) -> None:
    """Append one refusal record. Never raises — a failed log must not mask the refusal.

    The refusal itself is the safety mechanism; this is the audit trail. If the disk is
    full or the path is unwritable the run must still be refused, and loudly, so the
    failure to record is reported and swallowed rather than propagated.

    ``TypeError`` and ``ValueError`` are caught alongside ``OSError`` because the failure
    here is not only the filesystem: ``json.dumps`` raises ``TypeError`` on anything
    unserialisable, and ``detail`` is built by the caller. Catching only ``OSError`` made
    the docstring's promise true by luck of the current call site rather than by
    construction — and the cost of being wrong is that a traceback replaces the refusal
    message written specifically so a demo-day operator knows why the run did not start.
    ``default=str`` means an unexpected value is recorded approximately rather than
    losing the whole record.
    """
    path = DEFAULT_REFUSAL_LOG if path is None else Path(path)
    record = {"wall_time": time.time(), "reason": reason,
              "argv": sys.argv[1:], **detail}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        print(f"[mappo_drive] refusal recorded in {path}", file=sys.stderr)
    except (OSError, TypeError, ValueError) as exc:
        print(f"[mappo_drive] could not record the refusal in {path}: {exc}",
              file=sys.stderr)


#: ``SUB_FLOOR_WINDOW_S`` and ``SUB_FLOOR_PROGRESS_FRACTION`` now live in ``avoidance``
#: and are imported above rather than restated here. They used to be defined in this file
#: with a comment explaining that ``visual_nav.PROGRESS_FRACTION`` could not be imported,
#: because ``visual_nav`` needs OpenCV and a robot and this file must stay importable
#: without either. ``avoidance`` needs neither — it is pure numpy, and it is where the
#: guard that acts on them now lives (issue #26), so one definition serves both.


def _axis_reach(value: float, limit: float) -> float:
    """One axis's share of an ellipse; ``0`` for an axis that is switched off.

    A zero ``limit`` means the robot cannot move on this axis at all, and the envelope
    clamp in :meth:`MappoPlanner.plan` has already forced ``value`` to zero to match, so
    reporting no extent is a statement of fact rather than a fallback. Dividing instead
    raises ``ZeroDivisionError`` even for ``0.0 / 0.0``.
    """
    return 0.0 if limit <= 0.0 else value / limit


class MappoPlanner(DynamicWindowPlanner):
    """A planner-shaped object that asks the policy first.

    Subclasses rather than wraps so that ``config``, ``limits`` and the rollout internals
    are all genuinely the planner's — ``visual_nav`` reads ``planner.config`` when it
    builds its obstacle list, and the veto needs the rollout.
    """

    def __init__(self, limits, config, runner: PolicyRunner, supervised: bool = True,
                 refusal_log: Path | None = None, scale_override: bool = False,
                 veto_horizon_s: float | None = VETO_HORIZON_S,
                 gait_floor_m_s: float = 0.0,
                 platform_floor_m_s: float = 0.0,
                 execution_supervisor: TurnDriveSupervisor | None = None,
                 axis_preview=None):
        super().__init__(limits=limits, config=config)
        self._runner = runner
        self._supervised = supervised
        #: The turn-drive execution supervisor, or ``None``. When it is set and a
        #: mapped static obstacle blocks the straight line to the goal, its command —
        #: built only from the primitives this robot was measured to perform —
        #: replaces the policy's, and the SAME veto is then applied to it. See
        #: ``--execution-supervisor``.
        self._execution_supervisor = execution_supervisor
        #: Optional ``(vx, vy, wz) -> dict`` translating a command into the raw
        #: transport axes it WOULD leave as, for the decision record. The Lite3 axis
        #: mapping is sign-only, so the record says so rather than implying the
        #: magnitudes survive.
        self._axis_preview = axis_preview
        #: The per-tick decision record, rebuilt by every ``plan()`` call that reaches
        #: the policy. Read by ``visual_nav`` on the tick that planned; a tick that
        #: never called ``plan()`` has nothing to inherit because the navigator only
        #: reads this on the planning path.
        self._last_decision: dict | None = None
        #: How far ahead the veto rolls a proposed command. ``None`` is the planner's own
        #: horizon (2.5 s). This is the parameter that decides WHERE the planner takes the
        #: avoidance off the policy, and it is not the policy's sensing horizon: measured
        #: live on 2026-08-18, the veto fired at 0.900 m from the bin's surface while the
        #: policy's horizon was 0.700 m, so the planner had committed the escape three
        #: ticks before the policy could see anything. Shortening the policy's horizon
        #: alone cannot narrow the pass — it only hands the planner more of the job.
        self._veto_horizon_s = veto_horizon_s
        #: True when the operator set the scale by hand with ``--policy-scale``, which
        #: turns the calibration refusal below into a warning. See
        #: :meth:`_check_radius_calibration`.
        self._scale_override = scale_override
        self._loco = None
        #: The mesh peer source, or ``None`` when peer avoidance is off. Read here rather
        #: than re-derived, so the hold and the obstacle list are one tick's decision:
        #: ``PeerNavigator._obstacles`` refreshes it, and the vendored loop calls that
        #: before every ``plan()``.
        self._peers = None
        #: Counted and printed at the end of a run. A veto that never fires and a veto
        #: that fires on every tick are both worth knowing about, and neither is visible
        #: in a log of velocities.
        self.counts: dict = {"ticks": 0, "policy": 0, "vetoed": 0, "stopped": 0,
                             "velocity_unavailable": 0, "speed_raised": 0,
                             "peer_held": 0, "floor_unreachable": 0,
                             "raised_below_floor": 0, "sub_floor": 0,
                             "sub_floor_stalled": 0, "transport_refused": 0,
                             "turn_drive": 0}
        #: Commanded speeds below this are scaled up, direction preserved — see
        #: :meth:`_at_least_walking_pace`. Zero disables it. This is
        #: ``--policy-gait-floor``: an OPT-IN override of what the policy asked for, and
        #: it stays opt-in. Issue #106 has just moved the heading servo the other way
        #: after its default drove the robot into a wall three times.
        self._gait_floor_m_s = gait_floor_m_s
        #: THIS ROBOT'S measured gait floor, from its bindings — ``MIN_GAIT_COMMAND_M_S``
        #: on a Go2, ``--gait-floor`` on a Lite3. Nothing is scaled by it and no leg
        #: moves differently because of it: it is what :meth:`_note_sub_floor` judges the
        #: EMITTED command against, on every path out of :meth:`plan`. Separate from the
        #: field above because raising a command is a decision and reporting one is not,
        #: and because they are two different numbers on any robot that is not a Go2.
        self._platform_floor_m_s = platform_floor_m_s
        #: ``None``, or ``[since, x0, y0, commanded_travel_m, last_now, last_speed]``
        #: for the run of consecutive sub-floor ticks in progress. See
        #: :meth:`_note_sub_floor`.
        self._sub_floor: list | None = None
        #: The banner is printed once per run; the COUNT keeps rising after it.
        self._sub_floor_announced = False
        #: The same, for the planner's transport refusal. A SEPARATE latch, not a shared
        #: one: the two say different things about different subsystems, and the first
        #: to fire would otherwise silence the other for the rest of the run.
        self._transport_refusal_announced = False
        self._check_radius_calibration(refusal_log)
        self._announce_floor_against_envelope()

    def _announce_floor_against_envelope(self) -> None:
        """State how the two gates compose, at the top of the run log, in one place.

        THE TWO GATES ARE A CEILING AND A FLOOR AND NOTHING PUTS THEM SIDE BY SIDE.
        ``limits`` is the envelope — ``--max-vx``/``--max-vy``/``--max-wz`` scaled by
        ``--derate`` — which issue #103 made a per-robot, refusable number carrying named
        Go2 provenance. ``_platform_floor_m_s`` is the floor. Before this line the only
        report of either was ``warn_if_below_gait_floor``, which fires once, at startup,
        and only when the CEILING is under the floor.

        Three cases, and only the first is the one every recorded run was made in:

        * floor == the envelope's forward ceiling. The Go2's, and not a coincidence —
          ``avoidance`` says so at :data:`MIN_GAIT_COMMAND_M_S`. The robot has exactly
          one usable forward speed, and the floor ellipse and the envelope ellipse
          coincide, which is the premise :meth:`_floor_axes` exists to stop assuming.
        * floor INSIDE the envelope. A Lite3 states ``--max-vx 0.55`` and measures
          ``--gait-floor 0.30`` (``DEPLOYMENT-SOP.md``). The two ellipses are then
          different curves, and projecting onto the outer one turns a 0.05 m/s crawl
          into 0.55 m/s. See :meth:`_at_least_walking_pace`.
        * floor ABOVE the envelope. ``--derate 0.6`` on the shipped Go2 profile is
          exactly the 0.21 m/s measured to stall 5 runs of 5. NO command this run can
          make reaches the floor, so it is worth a banner rather than a line.
        """
        floor = self._platform_floor_m_s
        if floor <= 0.0:
            return
        ceiling = self.limits.max_vx
        if ceiling > 0.0 and floor > ceiling:
            print("!" * 78)
            print(f"[mappo_drive] ⚠️  GAIT FLOOR {floor:.3f} m/s IS ABOVE THIS RUN'S "
                  f"ENVELOPE (vx<={ceiling:.3f} m/s)")
            print("    No command this run can make is at or above the floor, so the "
                  "robot may stand")
            print("    still and report no fault. Raise --max-vx / --derate, or do not "
                  "run.")
            print("!" * 78)
            return
        shape = ("the same number as the envelope, so a sub-floor command has no faster "
                 "version" if math.isclose(floor, ceiling, rel_tol=1e-6)
                 else "INSIDE the envelope, so the floor and the envelope are "
                      "different curves")
        print(f"[mappo_drive] gait floor {floor:.3f} m/s against envelope "
              f"vx<={ceiling:.3f} vy<={self.limits.max_vy:.3f} — {shape}")

    def _check_radius_calibration(self, refusal_log: Path | None) -> None:
        """Refuse the run if the policy's scale was calibrated for a different robot size.

        ``meters_per_vmas_unit`` is not a free parameter: it is the radius the PLANNER
        plans with, divided by the radius the policy was TRAINED with. The trap is that
        those two live in different places and only one of them is in the policy's config
        — the vendored planner's default ``robot_radius_m`` is 0.40 m, every recorded run
        passed ``--robot-radius 0.25``, and the shipped 2.5 is 0.25 / 0.10. Run at the
        default and the policy's sensing horizon is 1.4 m rather than 0.875 m, every
        closed-loop number stops applying, and **nothing anywhere says so**.

        This runs in ``__init__``, which ``visual_nav.main()`` reaches after its own
        pre-flight but before sport mode is selected and before the control loop starts —
        so the robot is still lying down and nothing has moved.

        ``--policy-scale`` is the escape hatch, and it has to be checked separately rather
        than by comparing the two numbers: the override is applied to the config BEFORE
        this runs, so it moves ``configured`` and the mismatch it creates is exactly the
        one this guard fires on. Refusing then makes the documented escape hatch trip the
        gate it is supposed to open, and the refusal's own advice — "pass --policy-scale
        2.50" — asks the operator to undo the change they deliberately made. The trap
        being guarded against is a SILENT mismatch, i.e. forgetting ``--robot-radius`` and
        getting the vendored 0.40 default; an operator who typed the scale by hand has
        already made the choice this gate exists to force. So a deliberate override warns
        with the same numbers and runs.
        """
        trained = self._runner.controller.actor.metadata.get("training_agent_radius_vmas")
        if trained is None:
            print("[mappo_drive] WARNING: this checkpoint does not record its trained "
                  "agent radius, so the scale calibration cannot be checked. Confirm "
                  "--robot-radius by hand against policy/PROVENANCE.md.")
            return
        implied = self.config.robot_radius_m / trained
        configured = self._runner.config.meters_per_vmas_unit
        if math.isclose(implied, configured, rel_tol=0.02):
            return

        detail = {
            "planner_robot_radius_m": self.config.robot_radius_m,
            "trained_agent_radius_vmas": trained,
            "implied_scale": round(implied, 3),
            "configured_scale": configured,
        }
        if self._scale_override:
            print(
                "\n" + "!" * 78 + "\n"
                "[mappo_drive] ⚠️  DELIBERATE SCALE OVERRIDE — the policy is being told "
                "the robot is\n"
                "              a different size from the one the planner plans with.\n"
                + "!" * 78 + "\n"
                f"  planner robot radius   {self.config.robot_radius_m:.3f} m   "
                f"(--robot-radius)\n"
                f"  trained agent radius   {trained:.3f} VMAS   (from the checkpoint)\n"
                f"  calibrated scale       {implied:.2f} m/unit   "
                f"(horizon {implied * self._runner.config.lidar_range_vmas:.3f} m)\n"
                f"  running at             {configured:.2f} m/unit   "
                f"(horizon {self._runner.config.lidar_range_m:.3f} m)\n"
                "\n"
                f"  A SMALLER scale makes the policy react LATER and pass CLOSER; the "
                f"planner's\n"
                f"  veto is then the only thing holding the gap. The closed-loop "
                f"evidence in\n"
                f"  deploy/README.md was measured at {implied:.2f} and does not "
                f"describe this run.\n"
                + "!" * 78, file=sys.stderr)
            _record_refusal(refusal_log, "robot_radius_scale_override", detail)
            return

        _record_refusal(refusal_log, "robot_radius_scale_mismatch", detail)
        raise SystemExit(
            "\n" + "!" * 78 + "\n"
            "[mappo_drive] REFUSING TO RUN — the policy's scale was calibrated for a "
            "different robot size\n" + "!" * 78 + "\n"
            f"  planner robot radius   {self.config.robot_radius_m:.3f} m   "
            f"(--robot-radius)\n"
            f"  trained agent radius   {trained:.3f} VMAS   (from the checkpoint)\n"
            f"  implied scale          {implied:.2f} m/unit\n"
            f"  config says            {configured:.2f} m/unit   "
            f"(horizon {self._runner.config.lidar_range_m:.3f} m)\n"
            "\n"
            f"  The closed-loop evidence in deploy/README.md is only valid at "
            f"{configured:.2f}.\n"
            f"  Either pass --robot-radius "
            f"{configured * trained:.2f}, or pass --policy-scale {implied:.2f} and "
            f"re-run\n"
            f"  the simulation before trusting anything it does.\n"
            + "!" * 78)

    def attach_peers(self, peers) -> None:
        """Give the planner the peer source, so a lost link can stop the robot here."""
        self._peers = peers

    def attach(self, loco) -> None:
        """Give the planner the locomotion client, for the MEASURED velocity.

        The commanded velocity is not a substitute and the difference is a measured
        factor of two on this robot — it delivers about 0.45 of what it is told. The
        policy's observation carries velocity, so feeding it the command would tell it it
        is moving twice as fast as it is.
        """
        self._loco = loco

    def plan(self, pose, goal, last_command, obstacles, control_dt: float = 0.1,
             last_reason: str = "goal") -> Command:
        """Choose a command, then judge the one that is actually going out.

        SPLIT IN TWO SO THAT THE JUDGEMENT CANNOT BE SKIPPED. :meth:`_choose` has five
        exits — a goal-less tick, a peer hold, a policy stop, a vetoed tick and a
        policy-driven one — and issue #26 is about a gate that ran on ONE of them.
        ``_at_least_walking_pace`` sits inside the policy branch, so the planner's own
        command, which is the one measured crawling at 0.137 m/s through a 0.93 m gap,
        has never been compared to the floor at all. A single exit is the only way "on
        every path" is a property of the code rather than a claim in a comment.
        """
        command = self._choose(pose, goal, last_command, obstacles,
                               control_dt=control_dt, last_reason=last_reason)
        self._note_sub_floor(command, pose, time.monotonic())
        self._note_transport_refusal(command)
        return command

    def decision_record(self) -> dict | None:
        """The decision layers of the LAST ``plan()`` call, for the telemetry tick.

        ``None`` until the first call and whenever the last call never reached the
        policy — which is the property that stops a stale, goal-less or stand-switch
        tick inheriting a decision it never made. The record names every layer the
        command passed through — the policy's raw action, the envelope clamp, the gait
        floor, the execution supervisor, the command that left, and the transport axes
        it translates to — because the 2026-08-26 live runs were undiagnosable in
        exactly these terms: their telemetry held a reason string and nothing else.
        """
        return self._last_decision

    def _choose(self, pose, goal, last_command, obstacles, control_dt: float = 0.1,
                last_reason: str = "goal") -> Command:
        # Rebuilt on every call, so a tick that returns before the policy ran — no
        # goal, a peer hold — leaves NOTHING behind rather than last tick's record.
        self._last_decision = None
        # The incumbent is computed on EVERY tick, used or not, so that its acceleration
        # window and reason hysteresis stay continuous. A planner consulted for the first
        # time mid-run plans from a standstill the robot is not in.
        planned = self._incumbent(pose, goal, last_command, obstacles,
                                  control_dt=control_dt, last_reason=last_reason)
        self.counts["ticks"] += 1

        if self._loco is None:
            self.counts["velocity_unavailable"] += 1
            measured = (0.0, 0.0, 0.0)
        else:
            measured = self._loco.velocity()

        # The peer link's verdict is computed by `PeerNavigator._obstacles`, which the
        # vendored loop calls immediately before this on every tick. Reading the stored
        # snapshot rather than re-reading the spool keeps the obstacle list and the hold
        # as one tick's decision — otherwise the policy could be handed a peer that the
        # hold has already given up on, half a spool-write apart.
        #
        # A source that has been ATTACHED but never READ is a hold and not an absent link,
        # and the difference is the whole file: `{}` means "no peers configured", which is
        # the fail-open reading of "peers are configured and I have not looked". Found by
        # mutation-testing `test_mappo_drive.py`, where deleting the `last is None` guard
        # left every test green while a run with peers configured but a navigator that
        # never called `_obstacles` drove straight through them.
        if self._peers is None:
            peer_link = {}
        elif self._peers.last is None:
            peer_link = {"lost": True,
                         "reason": "peer link: configured, but nothing has been read yet"}
        else:
            peer_link = {"lost": self._peers.last.holds,
                         "reason": self._peers.last.reason()}

        step = self._runner.step(
            tick_from_state(time.monotonic(), pose, goal, obstacles, measured=measured,
                            reason=last_reason, peer_link=peer_link),
            monotonic_s=time.monotonic())
        if step is None:                      # no goal; the vendored loop filters these
            return planned

        # BELT AND BRACES, and the braces are the important half. `peer_link` above
        # reaches the policy through `mappo_bridge.external_hold`, which is what makes the
        # policy's own view of the world consistent — it is told the link is gone rather
        # than silently handed a world with one fewer robot in it. But that path routes a
        # SAFETY property through the vendored policy package, and nothing in this
        # repository controls whether a future checkpoint or runner honours
        # `external_hold`. This check does not go through it. A peer whose position is
        # unknown stops the legs whether or not anything downstream agrees, in supervised
        # mode and in raw, and `test_mappo_drive` proves it against a runner that ignores
        # external_hold entirely.
        if peer_link.get("lost"):
            self.counts["peer_held"] += 1
            return Command(0.0, 0.0, 0.0, reason="hold", gap_m=planned.gap_m,
                           feasible=planned.feasible, evaluated=planned.evaluated,
                           floor_reach_m_s=planned.floor_reach_m_s,
                           transport_refusal=planned.transport_refusal)

        if step.status != "COMMAND":
            self.counts["stopped"] += 1
            return Command(0.0, 0.0, 0.0, reason=_STOP_REASONS.get(step.status, "hold"),
                           gap_m=planned.gap_m,
                           feasible=planned.feasible, evaluated=planned.evaluated,
                           floor_reach_m_s=planned.floor_reach_m_s,
                           transport_refusal=planned.transport_refusal)

        # Clamp to the STACK's envelope, which may be derated below the policy config's
        # own ceilings by --derate or --max-vx. The policy does not know about those and
        # must not be able to out-run them.
        # FORWARD ONLY, matching the rule the vendored planner states for itself:
        # "Reverse is deliberately not sampled. The Go2 has no rear-facing sensing on
        # this unit, so backing away from a person means moving blind into space this
        # pipeline has never observed." That rule was applied to the planner's sampling
        # and NOT to this path, so the policy — a holonomic agent with no notion of which
        # way the sensors point — was free to command up to -0.35 m/s. Measured live on
        # 2026-08-18: approaching two bins the policy commanded v=(-0.35, -0.03) and the
        # goal distance grew from 2.64 m to 2.73 m while the robot reversed toward
        # unobserved floor. The camera is an 85-degree forward cone; there is nothing
        # behind it but the optimistic default that unseen bearings read as clear.
        clamped = (
            max(0.0, min(self.limits.max_vx, step.vx_mps)),
            max(-self.limits.max_vy, min(self.limits.max_vy, step.vy_mps)),
            max(-self.limits.max_wz, min(self.limits.max_wz, step.wz_radps)))

        proposed = self._at_least_walking_pace(clamped)

        decision = {
            "policy_raw": {"vx": step.vx_mps, "vy": step.vy_mps,
                           "wz": step.wz_radps},
            "after_limits": {"vx": clamped[0], "vy": clamped[1], "wz": clamped[2]},
            "after_gait_floor": {"vx": proposed[0], "vy": proposed[1],
                                 "wz": proposed[2]},
        }

        # The execution supervisor gets the scene AFTER the policy has answered: it
        # exists to re-express avoidance in primitives this robot can perform, not to
        # drive the run. ``None`` from it — no static blocker on the goal line, or no
        # clear detour — leaves the policy's command exactly where it was.
        override = (None if self._execution_supervisor is None
                    else self._execution_supervisor.command(pose, goal, obstacles))
        if override is not None:
            candidate = (override.vx, override.vy, override.wz)
            decision["supervisor"] = {
                "phase": override.phase, "side": override.side,
                "waypoint": [override.waypoint[0], override.waypoint[1]],
                "blocker": override.blocker_id,
                "command": {"vx": candidate[0], "vy": candidate[1],
                            "wz": candidate[2]},
            }
            # THE SAME VETO, ON THE OVERRIDE. The first version of this bypass
            # returned the supervisor's command directly, which routed AROUND the
            # dynamic-obstacle check — a person stepping onto the detour path would
            # have been walked into by the safety layer's own replacement. The
            # supervisor only ever looks at the static map; everything that moves is
            # judged here, on the command that is actually going out.
            if self._supervised and not self.is_feasible(
                    pose, candidate, obstacles, horizon_s=self._veto_horizon_s):
                self.counts["vetoed"] += 1
                decision["supervisor"]["vetoed"] = True
                # `floor_reach_m_s` and `transport_refusal` travel HERE and not on the
                # branch below, and the difference is which velocity is going out. This
                # branch re-emits THE PLANNER'S OWN command, so both fields describe it
                # and dropping them would strip the state off the exact velocity that
                # carries it — issue #26 and issue #145, the same argument the policy
                # veto below makes.
                return self._finalise(decision, Command(
                    planned.vx, planned.vy, planned.wz,
                    reason=f"veto-{planned.reason}", gap_m=planned.gap_m,
                    feasible=planned.feasible, evaluated=planned.evaluated,
                    floor_reach_m_s=planned.floor_reach_m_s,
                    transport_refusal=planned.transport_refusal))
            self.counts["turn_drive"] += 1
            # The reason is `exec-<phase>`, a word OUTSIDE PLANNER_REASONS on purpose:
            # `base_reason` returns it unchanged, so the rest-after-blocked dispatch,
            # both Schmitt triggers and the bridge's mover hold all read it as itself
            # and none of them mistake a supervised detour for a planner hold.
            #
            # ⚠️ **`floor_reach_m_s` AND `transport_refusal` ARE STATED AS `None` HERE
            # RATHER THAN FORWARDED, AND THAT IS THE WHOLE COMPOSITION OF #145 AND
            # #13.** Both describe `planned` — the command the shared planner would
            # have issued — and `planned` is precisely what this branch REPLACES.
            # `visual_nav` ends the run on either of them (`stopped by the transport`,
            # `stopped below the gait floor`), so forwarding them would end the run on
            # the exact tick the supervisor found the way around: the planner refuses a
            # straight line it cannot execute, the supervisor answers with a turn and a
            # drive it can, and the refusal that has just been superseded would still be
            # the last word. Measured on the scene in
            # `test_the_supervisors_detour_survives_the_transport_aware_veto`: 18 of the
            # run's 51 supervisor ticks carry a planner refusal, the first at tick 24 of
            # 104, and forwarding it ends the run on the detour's first drive leg.
            #
            # STATED rather than omitted, because `test_every_branch_that_returns_a_
            # command_states_its_search` requires every producer to say what it means —
            # and "deliberately nothing to report" and "forgot to forward" must not be
            # the same diff. `test_a_planner_transport_refusal_does_not_end_a_supervised
            # _detour` is what keeps the value right.
            return self._finalise(decision, Command(
                candidate[0], candidate[1], candidate[2],
                reason=f"exec-{override.phase}", gap_m=planned.gap_m,
                feasible=planned.feasible, evaluated=planned.evaluated,
                floor_reach_m_s=None, transport_refusal=None))

        if self._supervised and not self.is_feasible(pose, proposed, obstacles,
                                                     horizon_s=self._veto_horizon_s):
            self.counts["vetoed"] += 1
            # The reason is QUALIFIED, not replaced, so that one string says both what
            # the planner decided and that the policy's command was refused. That makes
            # it wrong to compare against a vocabulary word with ``==``: five reads in
            # three files did, `"veto-hold" != "hold"`, and both of the planner's
            # Schmitt triggers, the whole rest-after-blocked timer and the bridge's
            # mover hold were inert on every policy-driven run (issue #118).
            # `avoidance.base_reason` is what a consumer reads it
            # through, and `test_the_reason_a_veto_writes_is_one_base_reason_can_read`
            # is what keeps this end of the contract honest.
            # `floor_reach_m_s` travels with the other two and for the stronger
            # reason: this branch RE-EMITS THE PLANNER'S OWN COMMAND, so dropping it
            # would strip the gait-floor state off the exact velocity that carries it.
            # The 2026-08-27 run that reopened issue #26 was policy-driven, and 37 of its
            # 38 planner-issued ticks came out of this branch.
            return self._finalise(decision, Command(
                planned.vx, planned.vy, planned.wz,
                reason=f"veto-{planned.reason}", gap_m=planned.gap_m,
                feasible=planned.feasible, evaluated=planned.evaluated,
                floor_reach_m_s=planned.floor_reach_m_s,
                transport_refusal=planned.transport_refusal))

        self.counts["policy"] += 1
        # `feasible`/`evaluated` describe the PLANNER's search, which ran on this tick
        # whatever the outcome, so they are forwarded on every branch. Letting them fall
        # back to Command's defaults writes `0 of 0` into the telemetry, and that is not
        # an absent value — it reads as "the planner sampled nothing and found nothing
        # feasible", i.e. the robot was boxed in. The recorded runs of 2026-08-17 say
        # exactly that on all 58 policy-driven ticks of the successful one, while the
        # vetoed ticks beside them record 330 of 330. See issue #20.
        return self._finalise(decision, Command(
            proposed[0], proposed[1], proposed[2], reason="policy",
            gap_m=planned.gap_m,
            feasible=planned.feasible, evaluated=planned.evaluated,
            floor_reach_m_s=planned.floor_reach_m_s,
            transport_refusal=planned.transport_refusal))

    def _note_transport_refusal(self, command: Command) -> None:
        """Say, once, that the TRANSPORT stopped the robot rather than the room.

        ``avoidance.DynamicWindowPlanner`` already refused the command — its feasibility
        is computed on what the legs will receive, so a command whose EXECUTION ends
        inside an obstacle is never chosen (issue #145). What that leaves in the log is
        a `hold`, and a `hold` is what a robot boxed in by furniture also writes. The
        two want opposite responses: one wants a wider lane, and this one wants the
        operator to know the robot has one forward gear and the geometry does not fit it.

        Counted every tick and printed once, the shape ``_announce_sub_floor`` uses and
        for the same reason: a banner per tick at 10 Hz is a banner nobody reads.
        """
        if command.transport_refusal is None:
            return
        self.counts["transport_refused"] += 1
        if self._transport_refusal_announced:
            return
        self._transport_refusal_announced = True
        print("!" * 78)
        print("[mappo_drive] ⚠️  THE TRANSPORT REFUSED THIS COMMAND — IT IS NOT THE "
              "ROOM AND NOT THE TETHER")
        print(f"    {command.transport_refusal}")
        print("    This robot's transport is sign-only: it has no slow. Raising a "
              "ceiling will not help,")
        print("    and lowering one will not either — the executable set does not "
              "change with --max-vx or")
        print("    --derate. What changes it is the geometry: more lane, a smaller "
              "--robot-radius if the")
        print("    loaded body measures smaller, or an approach that does not need a "
              "crawl. See issue #145.")
        print("!" * 78)

    def _incumbent(self, pose, goal, last_command, obstacles, control_dt: float,
                   last_reason: str) -> Command:
        """The SHARED planner's own answer for this tick — what the robot would do
        with no policy and no supervisor at all.

        A named seam rather than a bare ``super().plan`` call at the top of
        :meth:`_choose`, because it is the fallback every refusal path returns and
        the carrier of ``floor_reach_m_s`` and ``transport_refusal``, and a test that
        wants to pin what those branches forward has to be able to state what came in.
        Provoking a transport refusal from a real scene cannot do that: on any scene
        where the supervisor is refused there is a mover present, and
        ``_transport_refusal`` correctly declines to blame the transport when a
        proportional planner would have held too — so both fields are ``None`` and
        the assertion is vacuous. See
        ``test_the_supervisors_veto_fallback_still_carries_the_planners_state``.
        """
        return super().plan(pose, goal, last_command, obstacles,
                            control_dt=control_dt, last_reason=last_reason)

    def _finalise(self, decision: dict, command: Command) -> Command:
        """Close out the tick's decision record and return the command it describes.

        ONE exit helper rather than a record written at four return sites, because a
        layer that some paths forget to write is the defect this record exists to
        close: the 2026-08-26 runs could not say whether a `hold` was the policy's,
        the envelope's or the transport's. The axis preview is computed HERE, on the
        command actually leaving, so the record never shows the axes a DIFFERENT
        candidate would have produced.
        """
        decision["final"] = {"vx": command.vx, "vy": command.vy, "wz": command.wz,
                             "reason": command.reason}
        if self._axis_preview is not None:
            decision["axis_preview"] = self._axis_preview(command.vx, command.vy,
                                                          command.wz)
        self._last_decision = decision
        return command

    def _at_least_walking_pace(self, proposed: tuple) -> tuple:
        """Scale a policy command up to the gait floor, KEEPING ITS DIRECTION.

        ⚠️ **ON A SIGN-ONLY TRANSPORT THIS CHANGES NOTHING THE LEGS SEE**, and that is
        worth knowing before reaching for ``--policy-gait-floor`` on a Lite3. Raising
        0.05 m/s onto the floor ellipse produces a larger number with the same signs, and
        the axis mapping keeps only the signs — both commands fire the same primitive at
        the same speed. It is still applied, because it is applied before the veto and
        the veto now rolls out the EXECUTION, so the two agree either way; it is simply
        not a fix for anything on that robot. Issue #145.

        The delivered checkpoint is a holonomic VMAS agent: it can move at any speed in
        any direction, and it never saw a robot with a minimum speed. The Go2 has one —
        below roughly :data:`MIN_GAIT_COMMAND_M_S` it produces no gait at all, stands
        still, and reports no fault. Threading a 0.93 m gap between two bins on
        2026-08-18 the policy went strongly lateral, its forward component collapsed to
        ``0.40 * 0.35 = 0.14 m/s``, and the robot stood there for 4 s moving 0.09 m of an
        expected 0.68 m while the stall gate blamed the tether.

        Why scaling is right HERE and was wrong for the planner. When the planner slows
        near an obstacle that is a deliberate safety decision, and overriding it was
        measured to cost clearance — the lateral offset around a bin fell from the
        required 0.88 m to 0.55 m and it clipped the bin (issue #26). The policy's
        magnitude is not a decision of that kind: the network's output is a DIRECTION,
        and the speed is whatever the envelope mapping happens to make of it. Scaling the
        vector preserves every bit of intent the policy actually expressed.

        Direction is preserved by scaling ``vx`` and ``vy`` together — scaling only the
        forward axis would rotate the command toward straight ahead, which near an
        obstacle is the one direction the policy was steering away from. Clamping one
        axis afterwards turns the command for the same reason, which is why the floor is
        projected onto as an ELLIPSE rather than scaled to as a circle; see the worked
        example below.

        Applied BEFORE the veto on purpose. A command clamped on the way OUT would mean
        ``is_feasible`` validated a velocity different from the one the legs receive, and
        a safety check that no longer describes the robot is the failure mode this
        repository keeps finding. Here the veto sees exactly what gets sent.

        ``wz`` is untouched: yaw has no gait floor, and the servo (when on) derives it
        from the direction, which this does not change.
        """
        floor = self._gait_floor_m_s
        if floor <= 0.0:
            return proposed
        vx, vy, wz = proposed
        speed = math.hypot(vx, vy)
        # A genuine stop stays a stop. Only a command the policy meant as MOTION is
        # scaled, or a zeroed status tick would be turned into a walk.
        if speed <= 0.0 or speed >= floor:
            return proposed
        # Never scale a command that is going BACKWARDS. Scaling multiplies the whole
        # vector, so without this a timid backward twitch becomes a committed one at full
        # speed — and the only direction this robot senses is ahead. Observed live before
        # the forward-only clamp above existed: a -0.03 m/s drift was scaled into a
        # 0.35 m/s reverse.
        #
        # CORRECTED 2026-08-19: this read `vx <= 0.0`, which also refused a PURE STRAFE,
        # on the stated grounds that a strafe "cannot reach the floor anyway (max_vy 0.20
        # < 0.35)". That compares max_vy against the FORWARD gait floor. The lateral floor
        # is not the same number and had never been measured. It is now: 0.15 m/s does not
        # walk this robot, 0.20 does — three repeats out of three, 0.076-0.087 m of travel
        # each, against a forward control in the same session. So the shipped envelope can
        # strafe, and this guard was refusing the one command that would have helped.
        #
        # Three live runs stalled on exactly that. Each escape was v=(+0.000,-0.150) with
        # vx EXACTLY zero, so `vx <= 0.0` held, no scaling happened, 0.150 m/s went out,
        # and 0.150 is below the lateral floor. The robot stood still inside its own hard
        # gap with an escape available and no way to execute it. Scaled, the same command
        # leaves as 0.20 m/s of strafe, which is measured to walk.
        #
        # `< 0.0` not `<= 0.0`: a sideways step is not a reverse.
        if vx < 0.0:
            return proposed

        # CORRECTED 2026-08-25: the floor is an ELLIPSE, and treating it as a circle
        # rotated the command this method's own docstring promises to preserve.
        #
        # The two floors are different numbers — 0.35 m/s forward, 0.20 m/s lateral —
        # so `floor / speed` overshoots on the lateral axis and the `max_vy` clamp then
        # trims only THAT component. Trimming one component of a vector turns it. Worked
        # example, and it is not a rounding error: the policy proposes (0.05, 0.108),
        # 65.2 deg off the nose. `scale` is 0.35/0.119 = 2.94, giving (0.147, 0.318);
        # the clamp cuts vy to 0.20 and leaves vx at 0.147, which is 53.7 deg. The
        # command arrives 11.5 deg closer to straight ahead — and near an obstacle,
        # straight ahead is the one direction the policy was steering away from. A
        # command at 80 deg is rotated by 34.8 deg, the worst case.
        #
        # Both floors happen to equal their own axis limit (MIN_GAIT_COMMAND_M_S 0.35 ==
        # max_vx, lateral floor 0.20 == max_vy), so the floor ellipse and the envelope
        # ellipse are the SAME curve. That makes the fix a projection onto it: normalise
        # each axis by its own limit, and scale the whole vector by one scalar until the
        # normalised radius reaches 1. Multiplying both components by a single number
        # cannot change their ratio, so the direction survives exactly, and the result
        # lands on the boundary rather than needing a clamp afterwards.
        #
        # The floor is measured only on the two axes; in between it is interpolated, and
        # that assumption is worth stating because it could be wrong. It does not need to
        # be right. Landing on the envelope boundary is the FASTEST command available in
        # the requested direction — nothing in that direction can be commanded harder. So
        # if the projected command does not walk, no command in that direction would, and
        # the only alternative left is to turn the command into a different one, which is
        # precisely the bug above. Optimal-in-direction holds whatever shape the true
        # floor has between the axes.
        #
        # A pure strafe is the case this exists to serve: (0.000, 0.108) has a normalised
        # radius of 0.54, so it scales to (0.000, 0.200) — the measured lateral floor,
        # direction untouched. That is the swerve, executed as a crab step.
        # A zero limit is an AXIS THE ROBOT DOES NOT HAVE, and dividing by it raised
        # ZeroDivisionError right here — `0.0 / 0.0` raises like any other. `--max-vy 0`
        # is how `robot-stack/deep_robotics/lite3/DEPLOYMENT-SOP.md` switches the strafe
        # axis off on a Lite3, and `--policy-gait-floor` is what reaches this line at all
        # (`deploy/run-peer-supervised.sh` ships it at 0.35), so the two together are a
        # configuration somebody will assemble.
        #
        # Contributing zero is the correct answer rather than a convenient one. The
        # envelope clamp in `plan()` runs BEFORE this and has already forced the
        # component on a zero-limit axis to zero, so the vector arriving here has no
        # extent on that axis to normalise. The ellipse degenerates to a segment on the
        # surviving axis, and scaling the whole vector — which is what happens below —
        # is the projection onto it. A pure strafe under `--max-vy 0` never gets this
        # far: the clamp zeroes it and `speed <= 0.0` returns above, which is right,
        # because a robot with no lateral axis has no way to execute one.
        #
        # Symmetric on both axes. `--max-vx 0` is not documented as a thing anyone does,
        # but the degenerate case is identical and a one-sided guard would be a second
        # rule to remember.
        #
        # CORRECTED 2026-08-26 (issue #26): the ellipse projected onto is the FLOOR's,
        # not the ENVELOPE's. Everything above is unchanged and still true on a Go2,
        # where they are one curve — see `_floor_axes`, which is where the two come
        # apart, and why they no longer can be assumed to be the same one.
        floor_x, floor_y, clipped = self._floor_axes(floor)
        reach = math.hypot(_axis_reach(vx, floor_x), _axis_reach(vy, floor_y))
        if reach >= 1.0:
            # Already on or outside the ellipse: it walks as proposed. `speed < floor`
            # can still hold here, because a mostly-lateral command reaches the lateral
            # floor well below the forward one.
            return proposed
        if reach <= 0.0:
            self.counts["floor_unreachable"] += 1
            return proposed
        scale = 1.0 / reach
        self.counts["speed_raised"] += 1
        if clipped:
            # The envelope is under the floor, so this landed on the envelope and is
            # STILL sub-floor. Counted apart from `speed_raised`, because "scaled up to
            # the gait floor" would be a false sentence about it, and a count that
            # misdescribes what it counted is worse than no count.
            self.counts["raised_below_floor"] += 1
        return (vx * scale, vy * scale, wz)

    def _floor_axes(self, floor: float, clip: bool = True) -> tuple:
        """``(forward floor, lateral floor, is the envelope below the floor)``.

        WHY THIS IS NOT JUST ``(self.limits.max_vx, self.limits.max_vy)``, which is what
        it was until issue #26. That spelling rested on one measured coincidence, stated
        in :meth:`_at_least_walking_pace` and in ``avoidance.GO2_MAX_VX_M_S``: on the Go2
        the forward floor 0.35 IS ``max_vx`` and the lateral floor 0.20 IS ``max_vy``, so
        the floor ellipse and the envelope ellipse are the same curve and it does not
        matter which one you project onto.

        Issue #103 ended that. The envelope is now a per-robot, refusable number — a
        Lite3 must STATE ``--max-vx``/``--max-vy``/``--max-wz`` on a live run, and
        ``robot-stack/deep_robotics/lite3/DEPLOYMENT-SOP.md`` states ``--max-vx 0.55``
        beside a measured ``--gait-floor 0.30``. Against that envelope the old spelling
        scaled a 0.050 m/s policy command to **0.550 m/s, an 11x amplification** of a
        command the policy meant as a crawl, and the veto then validated the sprint. The
        floor it should have reached is 0.30.

        So: take the envelope ellipse and shrink it by ``floor / max_vx`` — the one
        factor that puts the forward semi-axis on the floor — and never grow it past the
        envelope. Multiplying both semi-axes by one scalar keeps the ellipse's SHAPE, so
        the lateral floor is still interpolated from the envelope's aspect ratio, which
        is the assumption :meth:`_at_least_walking_pace` already documents and defends.

        Reduces exactly to the old behaviour wherever the old behaviour was measured:

        * Go2, ``floor 0.35`` vs ``max_vx 0.35`` → ``k = 1`` → ``(0.35, 0.20)``, the two
          measured floors, unchanged to the bit.
        * ``--derate 0.6`` → ``max_vx 0.21`` under the 0.35 floor → ``k`` clips at 1 →
          ``(0.21, 0.12)``, the envelope, again unchanged — but ``clipped`` is now True,
          so the caller stops claiming the result reached the floor.

        ``clip`` IS THE DIFFERENCE BETWEEN TWO QUESTIONS AND THEY HAVE DIFFERENT ANSWERS.
        "How fast can this command be made, in this direction?" is bounded by the
        envelope, so :meth:`_at_least_walking_pace` clips. "Is this command below the
        floor?" is not: under ``--derate 0.6`` the envelope ceiling 0.21 m/s is ITSELF
        the speed measured to stall 5 runs of 5, and a clipped test would call a command
        AT that ceiling "at the floor" and report ``0/58 ticks below the floor`` on a run
        that was under it from the first tick to the last. :meth:`_note_sub_floor` asks
        the second question and passes ``clip=False``.
        * ``--max-vx 0``: no forward axis, so ``floor / max_vx`` has no value. ``k = 1``
          leaves the lateral floor at ``max_vy``, which is what the symmetric-degenerate
          test pins today. A robot with no forward axis has no measured forward floor to
          scale a lateral one from, and inventing one to divide by would be worse than
          keeping the envelope.
        """
        ceiling = self.limits.max_vx
        if floor <= 0.0 or ceiling <= 0.0:
            return self.limits.max_vx, self.limits.max_vy, False
        # The floor ellipse is the envelope ellipse scaled by this one factor, which is
        # what keeps its SHAPE — and therefore the interpolated lateral floor — intact.
        scale = floor / ceiling
        clipped = scale > 1.0
        if clipped and clip:
            scale = 1.0
        return ceiling * scale, self.limits.max_vy * scale, clipped

    def _note_sub_floor(self, command: Command, pose, now: float) -> None:
        """Judge the command that is actually going out against this robot's gait floor.

        THE DEFECT THIS EXISTS FOR. Below :data:`avoidance.MIN_GAIT_COMMAND_M_S` the Go2
        produces no gait, stands still, and reports NO FAULT — the joint encoders read
        0.0 deg of swing, the state estimator agrees, and the stack's stall gate then
        fires four seconds later with *"something is holding the robot — check the
        tether"*. Every instrument is right and every one of them points away from the
        cause; five runs and two controllers went on tethers and walls before the speed
        was tested. The floor was checked once, at startup, against the envelope CEILING
        (``visual_nav`` line 1433, and this file's own startup line) and never against a
        command. This is that check, on the command, on every tick.

        IT DOES NOT GATE ON THE FLOOR'S VALUE, and that is deliberate, because the value
        is a guess. 0.35 is "the lowest speed observed to work" on 2026-08-14, not a
        measured threshold, and this repository's own run C of 2026-08-18 contradicts it:
        a sustained mean of 0.295 m/s with **54 of 54 ticks below 0.35**, minimum 0.189,
        3 m walked, arrived. A rule that stopped or faulted on "sub-floor" alone would
        have killed that run. So the trigger is a FACT — commanded to move and not
        moving — and the floor is used only to choose which of the two explanations of
        that fact to print. Above the floor and stationary is the tether. Below it and
        stationary is this.

        MEASURED FROM POSE, NOT FROM THE VELOCITY ESTIMATE, for the reason
        ``lateral_floor_probe.py`` gives: *"the reported velocity on this unit has been
        caught reading 0.17 m/s of noise as signal"*, which is larger than the whole
        0.137 m/s signal being judged. ``plan()`` is handed the odom pose every tick, and
        net displacement between two of them is the same quantity the stack's own stall
        gate compares against.

        IT NEVER TOUCHES THE LEGS. It does not scale, clamp, hold or refuse. Two measured
        attempts to make a sub-floor command into a faster one both failed and are
        recorded in issue #26 — filtering the dynamic window leaves only zero from rest,
        and allowing sub-floor only while ramping up drops the lateral offset around a
        bin from the required 0.88 m to 0.55 m and CLIPS IT. Clamping on the way out is
        worse again: ``is_feasible`` would then have validated a velocity the legs never
        received. And commanding a stop here would suppress the stack's stall gate, whose
        ``commanded <= PROGRESS_MIN_COMMAND_M_S`` branch returns "not asking it to go
        anywhere" — the run would stop ENDING and just stand there instead. So the run
        still ends exactly as it does today, with the cause printed above the outcome
        line rather than absent from it.
        """
        if self._platform_floor_m_s <= 0.0:
            return                            # no floor measured for this robot
        # `clip=False`: the question here is whether the command is under the FLOOR,
        # which does not stop being true because the envelope cannot reach it. See
        # `_floor_axes` — a clipped test reports `--derate 0.6` as fully compliant.
        floor_x, floor_y, _clipped = self._floor_axes(self._platform_floor_m_s,
                                                      clip=False)
        # JUDGED ON WHAT THE LEGS RECEIVE, not on what was typed — issue #145, and the
        # same category error one gate along. A gait floor is a fact about the legs, so
        # comparing the commanded number against it is only the right question on a
        # transport that delivers the commanded number. On the Lite3's sign-only axis
        # mapping it is not: the planner asks for 0.05 m/s, the legs walk at the
        # primitive's 0.30, and this would have counted every moving tick of every run
        # as sub-floor and printed "COMMANDED BELOW THE GAIT FLOOR AND NOT MOVING" at an
        # operator whose robot was walking. `None` means the transport cannot name an
        # executed velocity for this command, and there is then nothing to judge.
        executed = self.executed_velocity((command.vx, command.vy, command.wz))
        if executed is None:
            self._sub_floor = None
            return
        speed = math.hypot(executed[0], executed[1])
        reach = math.hypot(_axis_reach(executed[0], floor_x),
                           _axis_reach(executed[1], floor_y))
        if command.is_stop and command.floor_reach_m_s is not None:
            # A FLOOR STOP, which is a stop this file must not treat as one. The planner's
            # own guard has already measured two seconds of sub-floor commanding that
            # covered no ground and has stopped the legs on purpose
            # (`avoidance.DynamicWindowPlanner._gait_floor_stop`). Falling through to the
            # branch below would clear the window and print nothing, so the fix would
            # have silently deleted the only banner issue #26 has — the guard would act
            # and nobody would be told why.
            self.counts["sub_floor_stalled"] += 1
            self._announce_sub_floor(command.floor_reach_m_s, SUB_FLOOR_WINDOW_S, 0.0)
            self._sub_floor = None
            return
        if speed <= 0.0 or reach >= 1.0:
            # A stop is a stop, and a command on or outside the floor ellipse walks. Both
            # end the run of sub-floor ticks; neither is evidence about the floor.
            self._sub_floor = None
            return
        self.counts["sub_floor"] += 1
        if self._sub_floor is None:
            self._sub_floor = [now, pose[0], pose[1], 0.0, now, speed]
            return
        since, x0, y0, travel, previous, held = self._sub_floor
        # Integrated rather than `speed * span`, which is the stack's model and takes the
        # LAST tick's command as though it had been held all window. At an irregular
        # ~3 Hz under a planner that is actively slowing down those are different numbers.
        # `held` and not `speed`: the command in force over [previous, now] is the one
        # decided at `previous`, because the loop issues a command and then sleeps.
        # `speed` here would credit this tick's command to an interval it did not
        # govern, and on a collapsing command that is a factor of five, not a rounding.
        travel += held * max(0.0, now - previous)
        span = now - since
        if span < SUB_FLOOR_WINDOW_S:
            self._sub_floor = [since, x0, y0, travel, now, speed]
            return
        moved = math.hypot(pose[0] - x0, pose[1] - y0)
        # Judged, so the next window starts here whatever the verdict. Re-arming on a
        # PASS is what keeps a long productive-but-sub-floor run — run C — quiet; not
        # re-arming on a FAIL would make this a once-per-run gate that a robot which
        # recovered and stalled again would never trip twice.
        self._sub_floor = None
        if moved >= SUB_FLOOR_PROGRESS_FRACTION * travel:
            return
        self.counts["sub_floor_stalled"] += 1
        self._announce_sub_floor(speed, span, moved, travel)

    def _announce_sub_floor(self, speed: float, span: float, moved: float,
                            travel: float | None = None) -> None:
        """Say it once, loudly, whichever of the two detectors got there first.

        Shared with the planner's own guard, which reaches this through
        :meth:`_note_sub_floor`'s floor-stop branch. One message rather than two, because
        two would race: the planner stops the legs on the tick it decides, and this
        file's window would then see a zero command and reset without printing.
        """
        if self._sub_floor_announced:
            return
        self._sub_floor_announced = True
        print("!" * 78)
        print("[mappo_drive] ⚠️  COMMANDED BELOW THE GAIT FLOOR AND NOT MOVING — "
              "IT IS NOT THE TETHER")
        print(f"    commanded {speed:.3f} m/s for {span:.1f}s and moved {moved:.3f} m"
              + ("." if travel is None else f" of an expected {travel:.3f} m."))
        print(f"    This robot's measured gait floor is "
              f"{self._platform_floor_m_s:.3f} m/s; below it the legs stop swinging "
              f"and")
        print("    nothing reports a fault. When the stall gate fires it names the "
              "tether; below 0.05 m/s")
        print("    commanded it does not fire at all, and the robot simply stands "
              "there. It is this.")
        print("    Fixes, cheapest first: --robot-radius 0.20 so the planner never "
              "feels squeezed;")
        print(f"    --policy-gait-floor {self._platform_floor_m_s:.2f} so a sub-floor "
              f"POLICY command is raised; then")
        print("    widen the lane. The planner's own command is not raised, on purpose "
              "— see issue #26.")
        print("!" * 78)

    def report(self) -> str:
        counts = self.counts
        return (f"[mappo_drive] {counts['policy']}/{counts['ticks']} ticks driven by the "
                f"policy, {counts['vetoed']} vetoed, {counts['stopped']} stopped"
                + (f", {counts['turn_drive']} taken by the turn-drive supervisor"
                   if counts["turn_drive"] else "")
                + (f", {counts['speed_raised']} scaled up to the gait floor"
                   if counts["speed_raised"] else "")
                + (f", {counts['raised_below_floor']} of those only as far as an "
                   f"envelope that is itself below the floor"
                   if counts["raised_below_floor"] else "")
                + (f", {counts['floor_unreachable']} with a direction the envelope "
                   f"cannot reach" if counts["floor_unreachable"] else "")
                # Printed even when it is zero, unlike every other term here. A run in
                # which NO tick was commanded below the floor is the control that makes
                # a run in which many were mean something, and issue #26's proposal 2 —
                # measure the real floor — is a comparison of those two numbers across
                # runs. An absent line reads as "not measured", which is the one thing
                # it is not.
                + (f", {counts['sub_floor']}/{counts['ticks']} ticks commanded below "
                   f"the gait floor" if self._platform_floor_m_s > 0.0
                   else ", and no gait floor was known to judge the commands against")
                + (f" ({counts['sub_floor_stalled']} "
                   f"{SUB_FLOOR_WINDOW_S:.0f}s "
                   f"{'window' if counts['sub_floor_stalled'] == 1 else 'windows'} "
                   f"of it covering no ground — THE GAIT FLOOR, NOT THE TETHER)"
                   if counts["sub_floor_stalled"] else "")
                + (f", {counts['transport_refused']} refused because the legs' version "
                   f"of the command was not the one the planner validated"
                   if counts["transport_refused"] else "")
                + (f", {counts['peer_held']} held for a peer this robot could not locate"
                   if counts["peer_held"] else "")
                + (f", {counts['velocity_unavailable']} with no measured velocity"
                   if counts["velocity_unavailable"] else ""))


def peer_navigator(base, peers: PeerSource):
    """A ``VisualNavigator`` subclass whose obstacle list also carries the mesh peers.

    ``_obstacles`` is the seam and not ``plan()``, and the difference is what a consumer
    sees. ``visual_nav``'s loop builds the obstacle list ONCE per tick and hands the same
    object to the planner, the recorder, the console log and the telemetry writer — so
    adding a peer here puts it in all four, on every path through the loop, including the
    stale-frame and goal-search ticks where ``plan()`` is never called. Appending inside
    ``plan()`` instead would have been fewer lines and would have left the peer out of the
    record on exactly the ticks a reader would later want to explain.

    ``robot-stack/`` is a vendored copy and ``PROVENANCE.md`` is emphatic that editing it
    in place is how this project has lost fixes three times, so this is a subclass through
    the ``navigator_factory`` seam ``visual_nav.main()`` already offers rather than a
    change to ``_obstacles`` itself.

    A factory returning a class, rather than a class taking a source, because
    ``navigator_factory`` is called with the vendored constructor's positional arguments
    and this must not add one.
    """

    class PeerNavigator(base):
        def _obstacles(self, now: float) -> list:
            obstacles = super()._obstacles(now)
            # `now` is the vendored loop's `time.monotonic()`, which is the clock the
            # spool's stamps are in. That is not a coincidence to rely on quietly: see
            # `PeerSource.read`, which says what happens if it is ever a wall clock.
            link = peers.read(now)
            obstacles.extend(
                Obstacle(x=record["x"], y=record["y"],
                         vx=record["vx"], vy=record["vy"],
                         radius_m=record["radius_m"], label=record["label"],
                         kind=record["kind"], object_id=record["id"])
                for record in link.obstacles)
            return obstacles

    return PeerNavigator


def _add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("MAPPO policy")
    group.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                       help="the policy package directory")
    group.add_argument("--policy-config", type=Path,
                       help="alternative policy config.json")
    group.add_argument("--policy-mode", choices=("supervised", "raw"),
                       default="supervised",
                       help="supervised keeps the planner's feasibility veto (default)")
    group.add_argument("--policy-scale", type=float, metavar="M_PER_UNIT",
                       help="override meters_per_vmas_unit for this run")
    group.add_argument("--policy-command-scale", type=float, metavar="FRACTION",
                       help="override command_scale for this run")
    group.add_argument("--refusal-log", type=Path, default=DEFAULT_REFUSAL_LOG,
                       metavar="PATH.jsonl",
                       help="where a refused run records why. A refusal happens before "
                            "the telemetry writer exists, so without this a run that "
                            "never started leaves no trace at all")
    group.add_argument("--veto-horizon", type=float, metavar="SECONDS",
                       help="how far ahead the planner's veto rolls the policy's "
                            "proposed command. Default is the planner's own 2.5 s, which "
                            "fires the veto about 0.9 m from a bin's surface — outside "
                            "the policy's sensing horizon, so the planner commits the "
                            "escape before the policy has seen the obstacle. Lower this "
                            "to let the policy own the avoidance and pass closer; it is "
                            "also the safety margin, so it trades directly against it")
    group.add_argument("--policy-gait-floor", type=float, default=0.0,
                       metavar="M_PER_S",
                       help="scale a POLICY command slower than this UP to it, keeping "
                            "its direction. The checkpoint is a holonomic agent that "
                            "never saw a minimum speed, so a slow sideways manoeuvre — "
                            "exactly what threading a gap needs — comes out below the "
                            "Go2's gait floor and the robot stands still reporting no "
                            "fault. 0 (the default) leaves the command alone. This is "
                            "the RAISING knob only, and it does not have to be set for "
                            "a sub-floor command to be reported: the floor a command is "
                            "judged against comes from the robot's own bindings "
                            "(MIN_GAIT_COMMAND_M_S on a Go2, --gait-floor on a Lite3) "
                            "and is checked on every tick and on every path. Pass the "
                            "same number this robot measured, or do not pass it")
    group.add_argument("--peer-odom-align", metavar="DX,DY,DYAW_DEG",
                       help="TURNS ON peer avoidance over the Device Connect mesh, and "
                            "declares where the peer's odom frame is relative to this "
                            "robot's. Both frames start at their own robot's power-on "
                            "pose and have no relationship until it is measured, so this "
                            "is the enabling flag rather than an option on one: there is "
                            "no path where peers are on and the frames are undeclared. "
                            "DX/DY are where the peer was standing when it was switched "
                            "on, in metres, in this robot's start frame (+x ahead, +y "
                            "left); DYAW_DEG is its start heading minus this robot's, CCW "
                            "positive. A peer 2 m ahead, 1 m left, facing back at this "
                            "robot is 2.0,1.0,180. Pass 0,0,0 for a bench where both "
                            "really do start on the same spot")
    group.add_argument("--peer-spool", type=Path, default=DEFAULT_SPOOL,
                       metavar="PATH.json",
                       help="where dashboard/peer_link.py is writing peer poses. Must "
                            "match its --spool")
    group.add_argument("--peer-radius", type=float, metavar="METRES",
                       help="override the footprint a peer is modelled as. The default is "
                            "whatever the peer PUBLISHES for itself, which is the "
                            "half-diagonal of its own platform — 0.40 m for a Go2. Pass "
                            "this only with a tape measure in hand: a Go2 Wheel carries "
                            "modules the published 0.70 x 0.31 m does not describe")
    group.add_argument("--peer-timeout", type=float, default=PEER_TIMEOUT_S,
                       metavar="SECONDS",
                       help=f"how old a peer pose may be before the robot HOLDS. Default "
                            f"{PEER_TIMEOUT_S}, which is the stack's own "
                            f"perception_timeout_s: a peer link that has gone quiet is "
                            f"the same blindness as a camera that has. Raising it does "
                            f"not make a peer's position better known, it makes the robot "
                            f"act for longer on a position it no longer has")
    group.add_argument("--execution-supervisor", choices=("none", "turn-drive"),
                       default="none",
                       help="how avoidance is re-expressed when a mapped static "
                            "obstacle blocks the straight line to the goal. "
                            "'turn-drive' (the Lite3's answer) replaces the policy's "
                            "command with a two-segment detour executed as pure turns "
                            "and pure drives — the only motions this Lite3's measured "
                            "axis profile can perform, since its lateral and reverse "
                            "primitives are unmeasured and --max-vy 0 deletes the "
                            "holonomic policy's strafe intent at the envelope clamp. "
                            "The planner's veto still runs on the supervisor's command, "
                            "so a person stepping onto the detour holds the robot. "
                            "'none' (the default) is today's behaviour: the policy's "
                            "command, clamped, is what the veto judges")
    group.add_argument("--heading-servo", choices=("off", *SERVO_MODES), default="off",
                       help="turn the nose towards something the policy does not steer "
                            "for. The policy commands no yaw at all, so with the servo "
                            "OFF — the default — the robot crabs and its 85-degree "
                            f"camera never looks anywhere new. {GOAL!r} faces the goal "
                            f"bearing; {TRAVEL!r} faces the direction of travel and is "
                            "the law that put the robot into a wall three times on "
                            "2026-08-17 (issue #16). Default is off because no robot has "
                            f"yet been driven with {GOAL!r}")
    # The old spelling, kept working rather than broken: it appears in operator command
    # lines and in deploy/. It always meant "off", and off is now what you get anyway, so
    # it is a no-op that costs nothing to honour. Hidden from --help so the new flag is
    # the one people copy.
    group.add_argument("--no-heading-servo", dest="heading_servo",
                       action="store_const", const="off", help=argparse.SUPPRESS)
    return parser


def split_argv(argv, stack_parser) -> tuple:
    """Return ``(policy args, the argv the vendored parser should see)``.

    Two parsers read the same command line and only one of them is ours, so the policy
    flags have to be consumed here and removed. ``visual_nav.build_parser()`` knows
    nothing about ``--policy-mode``, and argparse does not ignore an option it does not
    recognise — it prints a usage message and exits 2. Handing it the raw argv therefore
    turned EVERY policy flag into a run that never started, ``--policy-command-scale``
    included, which is the first thing the runbook reaches for when the robot barely
    moves. Loud, but at the wrong altitude: it reads as a typo in the command line rather
    than as a seam that does not compose.

    The first parse is against the FULL parser so the policy flags are validated in the
    presence of the stack's and ``--help`` still shows the whole set; the second is
    against a policy-only parser purely to find what is left over.
    """
    args, _ = _add_arguments(stack_parser).parse_known_args(argv)
    stripper = _add_arguments(argparse.ArgumentParser(add_help=False))
    _, vendored_argv = stripper.parse_known_args(
        sys.argv[1:] if argv is None else list(argv))
    return args, vendored_argv


def platform_gait_floor(bindings, args) -> float:
    """This robot's own measured gait floor in m/s, or ``0.0`` if it has none.

    IT COMES FROM THE ROBOT, NOT FROM THE COMMAND LINE, which is the half of issue #26
    that a flag could not fix. ``Go2Bindings.gait_floor`` is ``MIN_GAIT_COMMAND_M_S``;
    ``Lite3Bindings.gait_floor`` is the operator's measured ``--gait-floor``, which that
    binding's own pre-flight already requires on a live run. So a Go2 gets the per-tick
    check with no flag at all, and every recorded run gets it retroactively.

    ``None`` — a Lite3 dry run — resolves to ``0.0`` and NOTHING IS JUDGED. An unmeasured
    floor is an absence, not a floor of zero, and inventing 0.35 for a robot that never
    produced it is the mistake issues #83, #96, #101 and #103 have each been about. The
    absence is printed, because a check that silently did not run is the other half of
    this repository's recurring defect.

    Split out of ``main`` so it can be exercised against both bindings without a robot,
    a camera or a policy package — the same reason ``visual_nav.build_goal_source`` is.
    """
    floor = bindings.gait_floor(args) or 0.0
    if not floor:
        print("[mappo_drive] no gait floor is known for this robot, so a command below "
              "it cannot be reported. Pass this platform's measured floor.")
        return 0.0
    override = getattr(args, "policy_gait_floor", 0.0)
    if override and not math.isclose(override, floor, rel_tol=0.02):
        # Two numbers that both call themselves the gait floor, set in two places, and
        # until now nothing compared them. `deploy/run-peer-supervised.sh` hard-codes
        # `--policy-gait-floor 0.35` — right for a Go2, wrong for the 0.30 a Lite3
        # measures — and that script is the thing people copy onto the next robot.
        print(f"[mappo_drive] ⚠️  --policy-gait-floor {override:.3f} is not this robot's "
              f"measured floor of {floor:.3f} m/s.")
        print(f"[mappo_drive]   A policy command will be raised to {override:.3f}; a "
              f"command is reported as sub-floor against {floor:.3f}.")
        print("[mappo_drive]   Pass one number, or neither.")
    return floor


#: The deployed tree's own root. ``integration/`` sits directly under it, which is also
#: what ``_STACK`` above assumes, so the two cannot drift apart without both breaking.
_ROOT = Path(__file__).resolve().parent.parent


def stamp_verdict(root=None, on_robot=None, announce=None):
    """Refuse a run whose tree does not match the commit its stamp names.

    Imported here rather than at module scope for the reason ``drive_bridge`` gives: this
    file is imported by an offline suite on machines that have no ``robot-stack``
    deployed beside it, and a module-level import would make the guard the thing that
    breaks the tests it is supposed to survive.

    ``on_robot`` comes from ``venv_guard``'s positive-only host evidence so that both
    pre-flights share one definition of "robot" -- an unstamped tree is a *finding* in a
    checkout and a *refusal* on a robot, and two definitions of which machine is which is
    how one of them ends up never firing.

    The verdict is printed either way. A guard that is silent when it passes cannot be
    told apart from a guard that is not running, and the tree id it prints is the only
    thing that lets a run log be resolved to a commit afterwards.
    """
    root = _ROOT if root is None else root
    announce = print if announce is None else announce
    preflight = str(Path(root) / "robot-stack" / "preflight")
    if preflight not in sys.path:
        sys.path.insert(0, preflight)
    from tree_stamp import describe, require_stamped_tree
    from venv_guard import robot_host_evidence
    if on_robot is None:
        on_robot = robot_host_evidence() is not None
    verdict = require_stamped_tree("mappo_drive", str(root), on_robot=on_robot)
    announce("[mappo_drive] " + describe(verdict))
    return verdict


def main(argv=None, bindings=None) -> int:
    # Parsed twice on purpose: once here to build the policy before the vendored main()
    # runs, and once by that main() for everything else.
    # Before anything else, and before the vendored stack is imported: a run that
    # cannot name its own commit produces a measurement nobody can attribute.
    stamp_verdict()

    import visual_nav

    bindings = bindings or visual_nav.Go2Bindings()
    args, vendored_argv = split_argv(argv, visual_nav.build_parser(bindings))

    overrides = {}
    if args.policy_scale is not None:
        overrides["meters_per_vmas_unit"] = args.policy_scale
    if args.policy_command_scale is not None:
        overrides["command_scale"] = args.policy_command_scale

    from replay_mappo import derived_config

    base = args.policy_config or args.package / "config.json"
    with derived_config(base, **overrides) as config:
        runner = PolicyRunner(
            args.package, config,
            servo=(None if args.heading_servo == "off"
                   else HeadingServo(mode=args.heading_servo)))
        cfg = runner.config
        print(f"[mappo_drive] policy {args.policy_mode}, scale "
              f"{cfg.meters_per_vmas_unit} m/unit, horizon {cfg.lidar_range_m:.3f} m, "
              f"command_scale {cfg.command_scale}")
        top_speed = cfg.max_vx_mps * cfg.command_scale
        print(f"[mappo_drive] {bindings.actuation_summary(top_speed, args)}")
        # The gait floor is a property of the ROBOT, not of the policy, so the warning is
        # the control stack's and is reused rather than restated — a second copy of that
        # text would drift from the constant it explains. Only the sentence naming the
        # knob that gets you here is ours, because `command_scale` reaches the floor by
        # multiplying where `--derate` reaches it by scaling.
        if bindings.warn_if_below_gait_floor(top_speed, args):
            floor = bindings.gait_floor(args)
            print(f"    Here that is command_scale {cfg.command_scale} x max_vx_mps "
                  f"{cfg.max_vx_mps} = {top_speed:.3f}. Pass --policy-command-scale "
                  f"{floor / cfg.max_vx_mps:.2f} or higher.")
        platform_floor = platform_gait_floor(bindings, args)
        if args.policy_mode == "raw":
            print("[mappo_drive] ⚠️  NO VETO. In the closed-loop simulation the raw "
                  "policy collided and the supervised one did not. Empty arena only.")
        if args.heading_servo == "off":
            print("[mappo_drive] heading servo off (the default): the robot will crab "
                  "and will not turn to look where it is going. --heading-servo "
                  f"{GOAL} faces the goal instead; it is simulated, not yet driven.")
        elif args.heading_servo == TRAVEL:
            print("[mappo_drive] ⚠️  --heading-servo travel is issue #16's control law. "
                  "It saturated the yaw rate and put this robot into a cubicle panel or "
                  "a cabinet on three runs out of four on 2026-08-17. It is here so "
                  "those runs stay reproducible. Empty arena only.")

        planners: list = []

        def planner_factory(limits, config):
            """Called by the shared run loop in place of ``DynamicWindowPlanner``."""
            supervisor = None
            if args.execution_supervisor == "turn-drive":
                # The detour is planned with the planner's OWN robot disc and executed
                # at the envelope ceilings. On the sign-only axis transport those
                # ceilings ARE the executed speeds to within the measured primitives'
                # accuracy — any command past the deadband leaves at the primitive's
                # measured rate — so the veto's rollout of a supervisor command
                # describes the motion the robot will actually make.
                supervisor = TurnDriveSupervisor(
                    robot_radius_m=config.robot_radius_m,
                    drive_speed_m_s=limits.max_vx,
                    turn_rate_rad_s=limits.max_wz)
            planner = MappoPlanner(limits, config, runner,
                                   supervised=args.policy_mode == "supervised",
                                   refusal_log=args.refusal_log,
                                   scale_override=args.policy_scale is not None,
                                   veto_horizon_s=(VETO_HORIZON_S
                                                   if args.veto_horizon is None
                                                   else args.veto_horizon),
                                   gait_floor_m_s=args.policy_gait_floor,
                                   platform_floor_m_s=platform_floor,
                                   execution_supervisor=supervisor,
                                   axis_preview=getattr(bindings, "axis_preview",
                                                        None))
            planners.append(planner)
            return planner

        peers = None
        if args.peer_odom_align is not None:
            peers = PeerSource(args.peer_spool, Alignment.parse(args.peer_odom_align),
                               timeout_s=args.peer_timeout,
                               **({} if args.peer_radius is None
                                  else {"radius_m": args.peer_radius}))
            print(f"[mappo_drive] peer link: {args.peer_spool}, align "
                  f"{args.peer_odom_align}, hold after {args.peer_timeout:.2f} s")

        # The measured velocity is the one thing plan() cannot reach, and it is not
        # optional: the commanded and achieved velocities differ by about a factor of two
        # on this robot, so a policy fed the command believes it is moving twice as fast
        # as it is. VisualNavigator is where loco and the planner are both in scope.
        real_navigator = visual_nav.VisualNavigator

        def navigator(loco, perception, planner, *rest, **kwargs):
            if isinstance(planner, MappoPlanner):
                planner.attach(loco)
                planner.attach_peers(peers)
            if peers is None:
                return real_navigator(loco, perception, planner, *rest, **kwargs)
            return peer_navigator(real_navigator, peers)(
                loco, perception, planner, *rest, **kwargs)

        try:
            visual_nav.main(
                argv=vendored_argv,
                planner_factory=planner_factory,
                navigator_factory=navigator,
                bindings=bindings,
            )
        finally:
            for planner in planners:
                print(planner.report())
            if peers is not None:
                print(peers.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
