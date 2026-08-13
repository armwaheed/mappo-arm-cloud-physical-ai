#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Drive the MAPPO policy from a recorded run and report what it would have done.

THIS IS THE TEST THAT THE LOG FORMAT IS ACTUALLY ALIGNED. A field-by-field table
comparing telemetry keys to ``RobotInput`` keys proves nothing: it cannot catch a frame
convention, a unit, or a field that is present but means something else. Replaying a real
run through the real checkpoint does, because every gap shows up as either a crash, a
:class:`BridgeReport` count, or a number that does not make sense.

It also answers the question that matters more than the format — whether the policy can
see the obstacle before it hits it. The policy's world is scaled: ``meters_per_vmas_unit``
maps a trained 0.35-unit lidar range onto 0.525 m of real floor, measured to the
obstacle's SURFACE.

EVERY RUN IS PAIRED WITH ITS OWN CONTROL. The same ticks are replayed twice, through two
independent controllers: once as recorded, and once with ``stationary_objects`` emptied.
Without that, "the policy steered 36 degrees off the goal bearing" is not evidence of
anything — this checkpoint carries a systematic heading bias of 6-16 degrees with no
obstacle anywhere near it, so an absolute deflection number credits the bias to
avoidance. The difference between the two runs is the part the obstacle actually caused.

    python3 replay_mappo.py ../evidence/sample_telemetry.jsonl \\
        --package ~/physicalai_mappo_go2

Needs the policy package (and its numpy) on the path; everything else here is stdlib.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from mappo_bridge import BridgeReport, audit, is_stationary, robot_input
from observation import wrap_pi
from telemetry_reader import read_run


def _load_policy(package: Path):
    """Import the policy package from wherever it was unpacked.

    It is not pip-installable and is not vendored here — it is a checkpoint plus an
    adapter that the policy owner ships as a directory, so the path is an argument.
    """
    sys.path.insert(0, str(package))
    try:
        from physical_ai_mappo import MappoController, RobotInput, StationaryObject
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the policy package from {package}: {exc}\n"
            f"pass --package <dir containing physical_ai_mappo.py>") from exc
    return MappoController, RobotInput, StationaryObject


def _goal_bearing(tick: dict) -> float:
    """Bearing to the goal in the body frame — where a robot with no obstacles goes."""
    pose = tick["pose"]
    return wrap_pi(math.atan2(tick["goal"]["y"] - pose["y"],
                              tick["goal"]["x"] - pose["x"]) - pose["yaw"])


def _nearest_surface_m(tick: dict) -> float:
    """Distance to the closest static obstacle's SURFACE, or inf if none is mapped.

    Surface rather than centre because that is what the policy's range vector measures
    and what its lidar range is therefore comparable against.
    """
    pose = tick["pose"]
    return min((math.hypot(o["x"] - pose["x"], o["y"] - pose["y"]) - o["radius_m"]
                for o in tick.get("obstacles", []) if is_stationary(o)),
               default=float("inf"))


def replay(run, package: Path, config: Path | None = None,
           verbose: bool = False) -> dict:
    """Step two controllers once per tick: as recorded, and with obstacles ablated."""
    MappoController, RobotInput, StationaryObject = _load_policy(package)
    config = config or package / "config.json"
    live = MappoController(config)
    control = MappoController(config)
    horizon_m = live.cfg.lidar_range_m

    report = BridgeReport()
    rows: list = []
    origin_yaw = 0.0
    for tick in run.ticks:
        report = report.merge(**audit(tick))
        kwargs = robot_input(tick, reset_run=not rows)
        if kwargs is None:
            continue
        objects = [StationaryObject(**o) for o in kwargs["stationary_objects"]]
        if not rows:
            origin_yaw = tick["pose"]["yaw"]   # the frame both controllers reset into

        out = live.step(RobotInput(**{**kwargs, "stationary_objects": objects}))
        blind = control.step(RobotInput(**{**kwargs, "stationary_objects": []}))

        # The policy's own heading, before the command mapping touches it. The two
        # differ by construction and not by a little: the mapping scales x by
        # max_vx_mps and y by max_vy_mps, which are not equal (0.35 vs 0.20), so a
        # 45-degree intent leaves as a 30-degree command. Reporting only the command
        # would credit that distortion to the policy, or hide a real avoidance in it.
        # The action is in the run-local frame the policy reset into, so it comes back
        # to body by the yaw turned since that reset.
        turned = tick["pose"]["yaw"] - origin_yaw
        intent = wrap_pi(math.atan2(out.action_y, out.action_x) - turned)
        blind_intent = wrap_pi(math.atan2(blind.action_y, blind.action_x) - turned)

        surface = _nearest_surface_m(tick)
        rows.append({
            "t": tick["t"],
            "status": out.status,
            "stack_reason": (tick.get("command") or {}).get("reason"),
            "nearest_surface_m": surface,
            "visible": surface <= horizon_m,
            "intent_deflection_deg": math.degrees(wrap_pi(intent - _goal_bearing(tick))),
            # How far the obstacle moved the policy. This, not the absolute deflection,
            # is the avoidance signal.
            "obstacle_effect_deg": math.degrees(wrap_pi(intent - blind_intent)),
        })
        if verbose:
            print(f"  t={tick['t']:6.2f}  {out.status:<18} "
                  f"v=({out.vx_mps:+.3f},{out.vy_mps:+.3f})  "
                  f"stack {rows[-1]['stack_reason']!s:<6} "
                  f"nearest={surface:5.2f}m {'SEEN  ' if rows[-1]['visible'] else 'unseen'}"
                  f" effect={rows[-1]['obstacle_effect_deg']:+6.1f}deg")

    return {"rows": rows, "report": report, "horizon_m": horizon_m, "config": live.cfg}


def _tally(rows: list, key: str) -> str:
    counts: dict = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return ", ".join(f"{name}={count}"
                     for name, count in sorted(counts.items(), key=lambda kv: -kv[1]))


def summarise(result: dict) -> None:
    rows = result["rows"]
    if not rows:
        print("no tick carried a goal — nothing to replay")
        return
    horizon = result["horizon_m"]
    config = result["config"]

    print()
    print("=" * 72)
    print("REPLAY SUMMARY")
    print("=" * 72)
    print(f"  ticks replayed               {len(rows)}")
    print(f"  policy status                {_tally(rows, 'status')}")
    print(f"  shipped planner meanwhile    {_tally(rows, 'stack_reason')}")

    mapped = [r for r in rows if math.isfinite(r["nearest_surface_m"])]
    seen = [r for r in mapped if r["visible"]]
    print()
    print(f"  policy sensing horizon       {horizon:.3f} m to the obstacle SURFACE")
    if mapped:
        print(f"  ticks with a static obstacle {len(mapped)}")
        print(f"  ... inside that horizon      {len(seen)} "
              f"({100.0 * len(seen) / len(mapped):.0f}%)")
        print(f"  closest surface all run      "
              f"{min(r['nearest_surface_m'] for r in mapped):.3f} m")

    effects = [abs(r["obstacle_effect_deg"]) for r in rows]
    intents = [abs(r["intent_deflection_deg"]) for r in rows]
    print()
    print("  steering, in degrees:")
    print(f"    off the goal bearing       max {max(intents):5.1f}, "
          f"mean {sum(intents) / len(intents):5.1f}   <- includes this checkpoint's "
          f"own bias")
    print(f"    CAUSED by the obstacle     max {max(effects):5.1f}, "
          f"mean {sum(effects) / len(effects):5.1f}   <- vs the ablated control")
    if seen:
        inside = [abs(r["obstacle_effect_deg"]) for r in seen]
        print(f"    ... on the {len(seen)} ticks it could see one: "
              f"max {max(inside):5.1f}, mean {sum(inside) / len(inside):5.1f}")
    unseen = [abs(r["obstacle_effect_deg"]) for r in mapped if not r["visible"]]
    if unseen:
        print(f"    ... on the {len(unseen)} ticks it could not: "
              f"max {max(unseen):5.1f}, mean {sum(unseen) / len(unseen):5.1f}")

    # A response that is near-zero outside the horizon and near-saturated inside it is a
    # THRESHOLD, and a threshold with no hysteresis chatters. The shipped planner carries
    # `reason_hysteresis_m` for exactly this failure, found live. Counting the reversals
    # says whether the policy will need the same treatment before it drives real legs.
    swerves = [r["obstacle_effect_deg"] for r in rows if abs(r["obstacle_effect_deg"]) > 10.0]
    reversals = sum(1 for a, b in zip(swerves, swerves[1:]) if a * b < 0.0)
    print(f"    direction reversals        {reversals} over {len(swerves)} swerving ticks")

    print()
    print(f"  command mapping caps vy at {config.max_vy_mps} m/s against vx's "
          f"{config.max_vx_mps},")
    print(f"  so a 45-degree intent is issued as "
          f"{math.degrees(math.atan2(config.max_vy_mps, config.max_vx_mps)):.0f} degrees.")

    lines = result["report"].lines()
    print()
    print("  BRIDGE GAPS" if lines else "  bridge mapping was clean")
    for line in lines:
        print(f"    - {line}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry", type=Path, help="a run's .jsonl")
    parser.add_argument("--package", type=Path, required=True,
                        help="directory holding physical_ai_mappo.py and config.json")
    parser.add_argument("--config", type=Path,
                        help="alternative config.json — for sweeping a constant such as "
                             "meters_per_vmas_unit without editing the package")
    parser.add_argument("--verbose", action="store_true", help="print every tick")
    args = parser.parse_args(argv)

    run = read_run(args.telemetry)
    print(f"{args.telemetry.name}: schema {run.schema}, {len(run.ticks)} ticks, "
          f"{'completed' if run.completed else 'TRUNCATED'}")
    summarise(replay(run, args.package.expanduser().resolve(),
                     config=args.config, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
