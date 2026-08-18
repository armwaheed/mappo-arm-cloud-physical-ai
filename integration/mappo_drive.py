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
    VETO_HORIZON_S,
    HeadingServo,
    PolicyRunner,
    tick_from_state,
)

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
    Command,
    DynamicWindowPlanner,
)

#: Statuses that mean "stop", mapped to the reason string the vendored loop understands.
#: ``hold`` is not cosmetic there: it is what starts the rest-after-blocked timer that
#: puts the robot prone instead of standing braced, and the arm makes standing expensive.
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


class MappoPlanner(DynamicWindowPlanner):
    """A planner-shaped object that asks the policy first.

    Subclasses rather than wraps so that ``config``, ``limits`` and the rollout internals
    are all genuinely the planner's — ``visual_nav`` reads ``planner.config`` when it
    builds its obstacle list, and the veto needs the rollout.
    """

    def __init__(self, limits, config, runner: PolicyRunner, supervised: bool = True,
                 refusal_log: Path | None = None, scale_override: bool = False,
                 veto_horizon_s: float | None = VETO_HORIZON_S):
        super().__init__(limits=limits, config=config)
        self._runner = runner
        self._supervised = supervised
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
        #: Counted and printed at the end of a run. A veto that never fires and a veto
        #: that fires on every tick are both worth knowing about, and neither is visible
        #: in a log of velocities.
        self.counts: dict = {"ticks": 0, "policy": 0, "vetoed": 0, "stopped": 0,
                             "velocity_unavailable": 0}
        self._check_radius_calibration(refusal_log)

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
        # The incumbent is computed on EVERY tick, used or not, so that its acceleration
        # window and reason hysteresis stay continuous. A planner consulted for the first
        # time mid-run plans from a standstill the robot is not in.
        planned = super().plan(pose, goal, last_command, obstacles,
                               control_dt=control_dt, last_reason=last_reason)
        self.counts["ticks"] += 1

        if self._loco is None:
            self.counts["velocity_unavailable"] += 1
            measured = (0.0, 0.0, 0.0)
        else:
            measured = self._loco.velocity()

        step = self._runner.step(
            tick_from_state(time.monotonic(), pose, goal, obstacles, measured=measured,
                            reason=last_reason),
            monotonic_s=time.monotonic())
        if step is None:                      # no goal; the vendored loop filters these
            return planned

        if step.status != "COMMAND":
            self.counts["stopped"] += 1
            return Command(0.0, 0.0, 0.0, reason=_STOP_REASONS.get(step.status, "hold"),
                           gap_m=planned.gap_m,
                           feasible=planned.feasible, evaluated=planned.evaluated)

        # Clamp to the STACK's envelope, which may be derated below the policy config's
        # own ceilings by --derate or --max-vx. The policy does not know about those and
        # must not be able to out-run them.
        proposed = (
            max(-self.limits.max_vx, min(self.limits.max_vx, step.vx_mps)),
            max(-self.limits.max_vy, min(self.limits.max_vy, step.vy_mps)),
            max(-self.limits.max_wz, min(self.limits.max_wz, step.wz_radps)))

        if self._supervised and not self.is_feasible(pose, proposed, obstacles,
                                                     horizon_s=self._veto_horizon_s):
            self.counts["vetoed"] += 1
            return Command(planned.vx, planned.vy, planned.wz,
                           reason=f"veto-{planned.reason}", gap_m=planned.gap_m,
                           feasible=planned.feasible, evaluated=planned.evaluated)

        self.counts["policy"] += 1
        # `feasible`/`evaluated` describe the PLANNER's search, which ran on this tick
        # whatever the outcome, so they are forwarded on every branch. Letting them fall
        # back to Command's defaults writes `0 of 0` into the telemetry, and that is not
        # an absent value — it reads as "the planner sampled nothing and found nothing
        # feasible", i.e. the robot was boxed in. The recorded runs of 2026-08-17 say
        # exactly that on all 58 policy-driven ticks of the successful one, while the
        # vetoed ticks beside them record 330 of 330. See issue #20.
        return Command(proposed[0], proposed[1], proposed[2], reason="policy",
                       gap_m=planned.gap_m,
                       feasible=planned.feasible, evaluated=planned.evaluated)

    def report(self) -> str:
        counts = self.counts
        return (f"[mappo_drive] {counts['policy']}/{counts['ticks']} ticks driven by the "
                f"policy, {counts['vetoed']} vetoed, {counts['stopped']} stopped"
                + (f", {counts['velocity_unavailable']} with no measured velocity"
                   if counts["velocity_unavailable"] else ""))


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
    group.add_argument("--no-heading-servo", action="store_true",
                       help="do not turn the nose towards the direction of travel. The "
                            "policy commands no yaw at all, so without the servo the "
                            "robot crabs and its 85-degree camera never looks anywhere "
                            "new")
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


def main(argv=None, bindings=None) -> int:
    # Parsed twice on purpose: once here to build the policy before the vendored main()
    # runs, and once by that main() for everything else.
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
            servo=None if args.no_heading_servo else HeadingServo())
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
        if args.policy_mode == "raw":
            print("[mappo_drive] ⚠️  NO VETO. In the closed-loop simulation the raw "
                  "policy collided and the supervised one did not. Empty arena only.")
        if args.no_heading_servo:
            print("[mappo_drive] ⚠️  heading servo off: the robot will crab and will "
                  "not turn to look where it is going.")

        planners: list = []

        def planner_factory(limits, config):
            """Called by the shared run loop in place of ``DynamicWindowPlanner``."""
            planner = MappoPlanner(limits, config, runner,
                                   supervised=args.policy_mode == "supervised",
                                   refusal_log=args.refusal_log,
                                   scale_override=args.policy_scale is not None,
                                   veto_horizon_s=(VETO_HORIZON_S
                                                   if args.veto_horizon is None
                                                   else args.veto_horizon))
            planners.append(planner)
            return planner

        # The measured velocity is the one thing plan() cannot reach, and it is not
        # optional: the commanded and achieved velocities differ by about a factor of two
        # on this robot, so a policy fed the command believes it is moving twice as fast
        # as it is. VisualNavigator is where loco and the planner are both in scope.
        real_navigator = visual_nav.VisualNavigator

        def navigator(loco, perception, planner, *rest, **kwargs):
            if isinstance(planner, MappoPlanner):
                planner.attach(loco)
            return real_navigator(loco, perception, planner, *rest, **kwargs)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
