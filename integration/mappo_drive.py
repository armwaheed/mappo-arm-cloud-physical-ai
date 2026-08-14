#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Let the MAPPO policy drive the Go2, with the shipped planner keeping a veto.

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

## ⚠️ Why this file monkey-patches, and what should replace it

`robot-stack/` is VENDORED and `PROVENANCE.md` forbids editing it in place — that rule
exists because this project has been bitten three times by fixes that lived only in a
copy. The vendored `main()` has no seam to pass a planner through, so the three module
globals are swapped before it runs. The alternative was to copy 150 lines of pre-flight,
including the arm-latch refusal and the health gate, and a duplicated safety check is
strictly worse than a documented patch. **The real fix is upstream: let `main()` accept a
planner.** Until then a re-vendor that renames any of the three raises an AttributeError
here rather than silently reverting to the shipped planner — see
:func:`_install`, which asserts each one exists before replacing it.

    python3 mappo_drive.py --live --telemetry run.jsonl --record run.mp4 \\
        --static-prop bin --goal-class chair --goal-height 1.067 \\
        --robot-radius 0.25 --no-latch-arm

Every ``visual_nav.py`` flag still applies. ``python3 test_mappo_drive.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mappo_policy import (
    DEFAULT_PACKAGE,
    HeadingServo,
    PolicyRunner,
    rollout_is_feasible,
    tick_from_state,
)

_STACK = Path(__file__).resolve().parent.parent / "robot-stack" / "unitree" / "go2"
for _directory in ("visual_nav", "locomotion", "d1_arm"):
    sys.path.insert(0, str(_STACK / _directory))

from avoidance import Command, DynamicWindowPlanner  # noqa: E402

#: Statuses that mean "stop", mapped to the reason string the vendored loop understands.
#: ``hold`` is not cosmetic there: it is what starts the rest-after-blocked timer that
#: puts the robot prone instead of standing braced, and the arm makes standing expensive.
_STOP_REASONS = {
    "STOP_EXTERNAL_HOLD": "hold",
    "STOP_STALE_INPUT": "hold",
    "STOP_CLOCK_ERROR": "hold",
    "STOP_GOAL_REACHED": "arrived",
}


class MappoPlanner(DynamicWindowPlanner):
    """A planner-shaped object that asks the policy first.

    Subclasses rather than wraps so that ``config``, ``limits`` and the rollout internals
    are all genuinely the planner's — ``visual_nav`` reads ``planner.config`` when it
    builds its obstacle list, and the veto needs the rollout.
    """

    def __init__(self, limits, config, runner: PolicyRunner, supervised: bool = True):
        super().__init__(limits=limits, config=config)
        self._runner = runner
        self._supervised = supervised
        self._loco = None
        #: Counted and printed at the end of a run. A veto that never fires and a veto
        #: that fires on every tick are both worth knowing about, and neither is visible
        #: in a log of velocities.
        self.counts: dict = {"ticks": 0, "policy": 0, "vetoed": 0, "stopped": 0,
                             "velocity_unavailable": 0}

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
                           gap_m=planned.gap_m)

        # Clamp to the STACK's envelope, which may be derated below the policy config's
        # own ceilings by --derate or --max-vx. The policy does not know about those and
        # must not be able to out-run them.
        proposed = (
            max(-self.limits.max_vx, min(self.limits.max_vx, step.vx_mps)),
            max(-self.limits.max_vy, min(self.limits.max_vy, step.vy_mps)),
            max(-self.limits.max_wz, min(self.limits.max_wz, step.wz_radps)))

        if self._supervised and not rollout_is_feasible(self, pose, proposed,
                                                        obstacles):
            self.counts["vetoed"] += 1
            return Command(planned.vx, planned.vy, planned.wz,
                           reason=f"veto-{planned.reason}", gap_m=planned.gap_m,
                           feasible=planned.feasible, evaluated=planned.evaluated)

        self.counts["policy"] += 1
        return Command(proposed[0], proposed[1], proposed[2], reason="policy",
                       gap_m=planned.gap_m)

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
                       help="override command_scale for this run. The delivered 0.3 is "
                            "0.047 m/s on the floor once the robot's measured 0.45 "
                            "actuator gain is applied")
    group.add_argument("--no-heading-servo", action="store_true",
                       help="do not turn the nose towards the direction of travel. The "
                            "policy commands no yaw at all, so without the servo the "
                            "robot crabs and its 85-degree camera never looks anywhere "
                            "new")
    return parser


def _install(module, planner_factory, on_navigator):
    """Swap the three module globals, refusing if any of them has moved.

    An AttributeError here is the intended behaviour after a re-vendor that renames one
    of them. The alternative — patching what is present and skipping what is not — leaves
    the shipped planner quietly in charge of a run the operator believes is being driven
    by the policy, which is the worst of the available failures.
    """
    for name in ("DynamicWindowPlanner", "VisualNavigator", "build_parser"):
        if not hasattr(module, name):
            raise SystemExit(
                f"[mappo_drive] visual_nav has no {name!r}: the vendored stack has moved "
                f"and this substitution is no longer valid. Fix it here before running.")

    real_parser, real_navigator = module.build_parser, module.VisualNavigator
    module.DynamicWindowPlanner = planner_factory
    module.build_parser = lambda: _add_arguments(real_parser())

    def navigator(loco, perception, planner, *args, **kwargs):
        on_navigator(loco, planner)
        return real_navigator(loco, perception, planner, *args, **kwargs)

    module.VisualNavigator = navigator


def main(argv=None) -> int:
    # Parsed twice on purpose: once here to build the policy before the vendored main()
    # runs, and once by that main() for everything else. `parse_known_args` on a copy of
    # the real parser means the policy flags are validated and `--help` still shows the
    # whole set.
    import visual_nav

    args, _ = _add_arguments(visual_nav.build_parser()).parse_known_args(argv)

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
        print(f"[mappo_drive] top commanded speed "
              f"{cfg.max_vx_mps * cfg.command_scale:.3f} m/s; this robot has measured "
              f"about 0.45 of what it is commanded")
        if args.policy_mode == "raw":
            print("[mappo_drive] ⚠️  NO VETO. In the closed-loop simulation the raw "
                  "policy collided and the supervised one did not. Empty arena only.")
        if args.no_heading_servo:
            print("[mappo_drive] ⚠️  heading servo off: the robot will crab and will "
                  "not turn to look where it is going.")

        planners: list = []

        def factory(limits, config):
            planner = MappoPlanner(limits, config, runner,
                                   supervised=args.policy_mode == "supervised")
            planners.append(planner)
            return planner

        def attach(loco, planner):
            if isinstance(planner, MappoPlanner):
                planner.attach(loco)

        _install(visual_nav, factory, attach)
        try:
            visual_nav.main()
        finally:
            for planner in planners:
                print(planner.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_STOP_REASONS", "MappoPlanner", "_add_arguments", "_install", "main"]
