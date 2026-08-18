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
maps the trained 0.35-unit lidar range onto real floor, measured to the obstacle's
SURFACE. At the shipped 2.5 that is 0.875 m. ``--scale`` sweeps it.

EVERY RUN IS PAIRED WITH ITS OWN CONTROL. The same ticks are replayed twice, through two
independent controllers: once as recorded, and once with ``stationary_objects`` emptied.
Without that, "the policy steered 36 degrees off the goal bearing" is not evidence of
anything — this checkpoint carries a systematic heading bias of 6-16 degrees with no
obstacle anywhere near it, so an absolute deflection number credits the bias to
avoidance. The difference between the two runs is the part the obstacle actually caused.

    python3 replay_mappo.py ../evidence/sample_telemetry.jsonl
    python3 replay_mappo.py ../evidence/sample_telemetry.jsonl --config sweep.json

Needs the policy package's numpy; everything else here is stdlib.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import tempfile
from pathlib import Path

from mappo_bridge import BridgeReport, audit, is_stationary, robot_input
from observation import wrap_pi
from telemetry_reader import read_run

#: The vendored policy package. It used to be an unpacked zip somewhere in a home
#: directory, which meant every replay in an issue or a PR quoted numbers nobody else
#: could reproduce. ``--package`` still overrides it, which is how a newly delivered
#: checkpoint gets compared against this one.
DEFAULT_PACKAGE = Path(__file__).resolve().parent.parent / "policy"


def _load_policy(package: Path):
    """Import the policy package from wherever it was unpacked.

    It is not pip-installable: it is a checkpoint plus an adapter that the policy owner
    ships as a directory, so the path stays an argument even though the delivered one is
    now in the tree.
    """
    sys.path.insert(0, str(package))
    try:
        from physical_ai_mappo import MappoController, RobotInput, StationaryObject
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the policy package from {package}: {exc}\n"
            f"pass --package <dir containing physical_ai_mappo.py>") from exc
    return MappoController, RobotInput, StationaryObject


@contextlib.contextmanager
def derived_config(base: Path, **overrides):
    """Yield a path to ``base`` with some fields replaced, for sweeping a constant.

    Sweeping ``meters_per_vmas_unit`` is the first thing issue #4 asks for, and
    ``command_scale`` turned out to matter as much, so this is a helper rather than
    something each reader re-improvises with a temp file. The policy package takes a
    config PATH, not a config object, which is why this writes a file at all.

    ``model_path`` is made absolute on the way out. It is stored relative to the config
    that names it, so a derived config written to a temp directory would otherwise look
    for the checkpoint next to itself and fail with a confusing "no such file".
    """
    data = json.loads(base.read_text())
    data.update(overrides)
    model = Path(data.get("model_path", ""))
    if not model.is_absolute():
        data["model_path"] = str((base.parent / model).resolve())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(data))
        yield path


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


#: Where the range fan starts in the observation vector. The checkpoint records its layout
#: as ``[x, y, vx, vy, x-gx, y-gy, *lidar]``, so the first six entries are state and
#: everything after them is the fan. Named rather than inlined because slicing at the wrong
#: offset yields a plausible number instead of an error.
OBSERVATION_STATE_WIDTH = 6


def policy_sight(controller) -> tuple:
    """``(visible, nearest_surface_m)`` as the POLICY saw it on the step just taken.

    THE OBSERVATION IS THE POLICY'S PERCEPTION, so it is the only honest source for
    "could the policy see an obstacle". :class:`MappoController` steers on its own
    retained obstacle map, not on the telemetry's per-tick list, and the two diverge
    whenever a landmark leaves the telemetry while the controller still remembers it —
    ``static_obstacle_ttl_s`` is 120 s against runs that last 20.

    This tool used to answer the question from the tick instead. Over the 70-tick run of
    2026-08-17 that under-reported visibility on **16 ticks**, every one in the same
    direction: it called the policy blind while an obstacle sat inside its fan, and then
    attributed the resulting deflection to ticks it had labelled unseeing. Issue #17.

    ``lidar`` is PROXIMITY — ``lidar_range_vmas - range_vmas`` — so a positive entry means
    something is inside the horizon and the LARGEST entry is the nearest thing.
    """
    obs = controller.last_observation
    if obs is None:
        return False, float("inf")          # stepped zero times, or no goal yet
    cfg = controller.cfg
    peak = float(max(obs[OBSERVATION_STATE_WIDTH:]))
    if peak <= 0.0:
        return False, float("inf")
    return True, (cfg.lidar_range_vmas - peak) * cfg.meters_per_vmas_unit


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
        visible, policy_surface = policy_sight(live)
        rows.append({
            "t": tick["t"],
            "status": out.status,
            "stack_reason": (tick.get("command") or {}).get("reason"),
            #: What the TELEMETRY reported this tick. Kept alongside the policy's own view
            #: rather than replaced by it, because the DIFFERENCE between the two is the
            #: only offline detector for the controller retaining obstacles perception no
            #: longer reports (issue #19). Collapsing them would hide that.
            "nearest_surface_m": surface,
            "policy_surface_m": policy_surface,
            "visible": visible,
            #: Inside the policy's fan while the telemetry had nothing there. A ghost.
            "remembered_only": visible and not (math.isfinite(surface)
                                                and surface <= horizon_m),
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

    return {"rows": rows, "report": report, "horizon_m": horizon_m, "config": live.cfg,
            # The TRAINED agent radius at this scale, read out of the checkpoint rather
            # than written down here. It is the number the scale is calibrated against.
            "agent_radius_m": live.agent_radius_m}


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
    # Visibility is the POLICY's, read out of its own observation — not the telemetry's
    # view of the same tick. See policy_sight() and issue #17.
    seen = [r for r in rows if r["visible"]]
    ghosts = [r for r in rows if r["remembered_only"]]
    print()
    print(f"  policy sensing horizon       {horizon:.3f} m to the obstacle SURFACE")
    print(f"  ticks the POLICY could see one {len(seen)}/{len(rows)} "
          f"({100.0 * len(seen) / len(rows):.0f}%)")
    if mapped:
        print(f"  ticks the TELEMETRY mapped one {len(mapped)}")
        print(f"  closest surface all run      "
              f"{min(r['nearest_surface_m'] for r in mapped):.3f} m")
    if ghosts:
        print()
        print(f"  ⚠️  {len(ghosts)}/{len(rows)} ticks had an obstacle inside the policy's "
              f"fan that the")
        print("      telemetry did not report — the controller is steering on objects it "
              "remembers")
        print(f"      but perception no longer sees. Closest such ghost: "
              f"{min(r['policy_surface_m'] for r in ghosts):.3f} m. Issue #19.")

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
    unseen = [abs(r["obstacle_effect_deg"]) for r in rows if not r["visible"]]
    if unseen:
        print(f"    ... on the {len(unseen)} ticks it could not: "
              f"max {max(unseen):5.1f}, mean {sum(unseen) / len(unseen):5.1f}")
        print("        (this row should be near zero. A large one means the policy "
              "steered for")
        print("         something absent from its own observation, which is a defect "
              "in THIS tool")
        print("         or in the controller, not a property of the checkpoint.)")

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


def sweep(run, package: Path, base_config: Path, scales: list) -> None:
    """One row per scale: what the horizon buys and what it costs.

    The whole point of the table is that the two columns move in opposite directions.
    Raising the scale lets the policy see the obstacle sooner, and the response it then
    makes is the SAME response — saturated at around 100 degrees at every scale measured —
    so what is being bought is warning, not proportionality. Reversals are the price, and
    they are the open-loop shadow of a chatter that only a closed-loop run can really
    measure (issue #5).
    """
    print()
    print(f"{'m/unit':>7}  {'horizon':>8}  {'agent r':>8}  {'seen':>10}  "
          f"{'effect inside':>14}  {'reversals':>10}")
    for scale in scales:
        with derived_config(base_config, meters_per_vmas_unit=scale) as config:
            result = replay(run, package, config=config)
        rows = result["rows"]
        # The policy's own view, not the telemetry's — see policy_sight() and issue #17.
        seen = [r for r in rows if r["visible"]]
        inside = [abs(r["obstacle_effect_deg"]) for r in seen]
        swerves = [r["obstacle_effect_deg"] for r in rows
                   if abs(r["obstacle_effect_deg"]) > 10.0]
        reversals = sum(1 for a, b in zip(swerves, swerves[1:]) if a * b < 0.0)
        share = f"{len(seen)}/{len(rows)}" if rows else "-"
        mean_inside = f"{sum(inside) / len(inside):.1f} deg" if inside else "-"
        radius = result["agent_radius_m"]
        radius_text = "      ?" if radius is None else format(radius, ">7.3f")
        print(f"{scale:>7.2f}  {result['horizon_m']:>7.3f}m  {radius_text}m  "
              f"{share:>10}  {mean_inside:>14}  "
              f"{reversals:>4} / {len(swerves):<3}")
    print()
    print("  agent r is the TRAINED 0.10 VMAS radius at that scale. Match it to the "
          "radius the")
    print("  planner is run with (--robot-radius, 0.25 m in the recorded runs) and the "
          "policy's")
    print("  idea of how much room it needs matches the robot's. See issue #4.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry", type=Path, help="a run's .jsonl")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE,
                        help="directory holding physical_ai_mappo.py and config.json "
                             "(default: the vendored ../policy)")
    parser.add_argument("--config", type=Path,
                        help="alternative config.json, in place of the package's own")
    parser.add_argument("--scale", type=float, nargs="+", metavar="M_PER_UNIT",
                        help="sweep meters_per_vmas_unit over these values and print a "
                             "comparison table instead of one run's detail")
    parser.add_argument("--verbose", action="store_true", help="print every tick")
    args = parser.parse_args(argv)

    package = args.package.expanduser().resolve()
    run = read_run(args.telemetry)
    print(f"{args.telemetry.name}: schema {run.schema}, {len(run.ticks)} ticks, "
          f"{'completed' if run.completed else 'TRUNCATED'}")
    if args.scale:
        sweep(run, package, args.config or package / "config.json", args.scale)
    else:
        summarise(replay(run, package, config=args.config, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
