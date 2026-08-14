#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the policy against a LIVE run without letting it touch the robot.

The shipped planner drives, exactly as it does today. This process tails the telemetry
file that run is writing, puts every tick through the same bridge and the same policy the
drive path would, and records what the policy WOULD have commanded beside what the stack
actually did.

**THIS FILE CANNOT MOVE A LEG.** It holds no locomotion client, opens no DDS channel and
imports nothing from ``locomotion``. It reads a file. That is the entire safety argument,
and it is why this is the first thing to run on the robot rather than the last.

What it buys that ``replay_mappo.py`` cannot: real perception, at the real rate, with the
real detection latency and the real dropouts, on the day and in the room. What it still
does not buy is any evidence that the policy NAVIGATES — the path is the planner's, so
this is open-loop in exactly the sense issue #5 means. ``closed_loop_sim.py`` is the
harness for that question.

    # terminal 1, on the robot
    python3 visual_nav.py --live --telemetry run.jsonl ...
    # terminal 2, on the robot or on a laptop with the file synced
    python3 mappo_shadow.py run.jsonl --follow --out shadow.jsonl

Needs the policy package's numpy. ``python3 test_mappo_shadow.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from mappo_bridge import BridgeReport, audit
from mappo_policy import DEFAULT_PACKAGE, PolicyRunner
from observation import wrap_pi

#: How long to wait for a new line before deciding a followed run has finished. Sized
#: well above the stack's 10 Hz tick and above the longest gap a stale-perception hold
#: produces, because a run that pauses is not a run that ended.
FOLLOW_IDLE_TIMEOUT_S = 20.0

#: Poll interval while following. The consumer is not the bottleneck and a tighter loop
#: would only spend the Jetson's CPU, which is already the scarce thing at ~130 ms/frame.
FOLLOW_POLL_S = 0.05


def follow(path: Path, follow_live: bool, idle_timeout_s: float = FOLLOW_IDLE_TIMEOUT_S):
    """Yield decoded records, optionally waiting for a writer to append more.

    A partially written final line is NOT yielded and NOT skipped: the read position is
    rewound to the start of it so the next poll sees it whole. Consuming half a line and
    moving on would drop a tick silently, which for a shadow run means a missing decision
    exactly where the interesting thing happened.
    """
    with path.open("r") as handle:
        idle_since = time.monotonic()
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line.endswith("\n"):
                handle.seek(position)
                if not follow_live:
                    return
                if time.monotonic() - idle_since > idle_timeout_s:
                    return
                time.sleep(FOLLOW_POLL_S)
                continue
            idle_since = time.monotonic()
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [shadow] skipping unparseable line: {exc}")


def _goal_bearing(tick: dict) -> float:
    pose = tick["pose"]
    return wrap_pi(math.atan2(tick["goal"]["y"] - pose["y"],
                              tick["goal"]["x"] - pose["x"]) - pose["yaw"])


def shadow(path: Path, package: Path = DEFAULT_PACKAGE, config: Path | None = None,
           follow_live: bool = False, out: Path | None = None,
           quiet: bool = False) -> dict:
    """Tail ``path`` through the policy. Returns the summary the CLI prints."""
    runner = PolicyRunner(package, config)
    report = BridgeReport()
    rows: list = []
    writer = None if out is None else out.open("w")
    try:
        if writer is not None:
            writer.write(json.dumps({
                "type": "header", "schema": "go2.mappo.shadow/1",
                "source": str(path), "wall_time": time.time(),
                "config": {"meters_per_vmas_unit": runner.config.meters_per_vmas_unit,
                           "command_scale": runner.config.command_scale,
                           "lidar_range_m": runner.config.lidar_range_m,
                           "velocity_frame": runner.config.velocity_frame}}) + "\n")
        for record in follow(path, follow_live):
            if record.get("type") != "tick":
                if not quiet and record.get("type") == "header":
                    print(f"  [shadow] following {record.get('schema')}, "
                          f"live={record.get('live')}, goal={record.get('goal')!r}")
                continue
            report = report.merge(**audit(record))
            # The policy compares this against its own clock, so it must be a monotonic
            # reading taken NOW — the tick's `t` is seconds since the run started and its
            # `wall_time` is an epoch. Both would disable the staleness gate.
            step = runner.step(record, monotonic_s=time.monotonic())
            if step is None:
                continue
            stack = record.get("command") or {}
            row = {
                "t": record["t"],
                "policy_status": step.status,
                "policy": [round(step.vx_mps, 4), round(step.vy_mps, 4),
                           round(step.wz_radps, 4)],
                "stack": [stack.get("vx"), stack.get("vy"), stack.get("wz")],
                "stack_reason": stack.get("reason"),
                "goal_bearing_deg": math.degrees(_goal_bearing(record)),
                "intent_deg": (None if step.intent_bearing_rad is None
                               else math.degrees(step.intent_bearing_rad)),
                "observation": [round(v, 5) for v in step.observation],
            }
            row["disagreement_deg"] = (
                None if row["intent_deg"] is None
                else math.degrees(wrap_pi(step.intent_bearing_rad
                                          - _goal_bearing(record))))
            rows.append(row)
            if writer is not None:
                writer.write(json.dumps(row) + "\n")
            if not quiet:
                print(f"  t={record['t']:6.2f}  stack {stack.get('reason')!s:<6} "
                      f"({_fmt(stack.get('vx'))},{_fmt(stack.get('vy'))})   "
                      f"policy {step.status:<18} "
                      f"({step.vx_mps:+.2f},{step.vy_mps:+.2f})  "
                      f"intent {_fmt_deg(row['intent_deg'])} vs goal "
                      f"{row['goal_bearing_deg']:+6.1f}deg")
    finally:
        if writer is not None:
            writer.close()
    return {"rows": rows, "report": report, "config": runner.config}


def _fmt(value) -> str:
    return "  n/a" if value is None else f"{value:+.2f}"


def _fmt_deg(value) -> str:
    return "   n/a" if value is None else f"{value:+6.1f}"


def summarise(result: dict) -> None:
    rows = result["rows"]
    if not rows:
        print("no tick carried a goal — nothing to shadow")
        return
    statuses: dict = {}
    for row in rows:
        statuses[row["policy_status"]] = statuses.get(row["policy_status"], 0) + 1
    disagreements = [abs(r["disagreement_deg"]) for r in rows
                     if r["disagreement_deg"] is not None]

    print()
    print("=" * 72)
    print("SHADOW SUMMARY — the policy was NOT driving")
    print("=" * 72)
    print(f"  ticks with a goal            {len(rows)}")
    print(f"  policy status                "
          f"{', '.join(f'{k}={v}' for k, v in sorted(statuses.items(), key=lambda kv: -kv[1]))}")
    print(f"  horizon                      {result['config'].lidar_range_m:.3f} m "
          f"(scale {result['config'].meters_per_vmas_unit})")
    if disagreements:
        wide = sum(1 for d in disagreements if d > 45.0)
        print()
        print("  policy intent vs the straight-line goal bearing, in degrees:")
        print(f"    mean {sum(disagreements) / len(disagreements):5.1f}, "
              f"max {max(disagreements):5.1f}, "
              f"over 45 deg on {wide}/{len(disagreements)} ticks")
        print("    ⚠️  this is deflection, NOT avoidance. Some of it is this "
              "checkpoint's own")
        print("       6-16 deg heading bias with an empty scene. Only a paired ablated "
              "control")
        print("       separates the two — replay_mappo.py does that, this cannot.")

    lines = result["report"].lines()
    print()
    print("  BRIDGE GAPS" if lines else "  bridge mapping was clean")
    for line in lines:
        print(f"    - {line}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry", type=Path, help="the run's .jsonl, live or finished")
    parser.add_argument("--follow", action="store_true",
                        help=f"wait for the writer to append; stops after "
                             f"{FOLLOW_IDLE_TIMEOUT_S:.0f}s of silence")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--config", type=Path, help="alternative policy config.json")
    parser.add_argument("--out", type=Path, metavar="PATH.jsonl",
                        help="write the policy's per-tick decision and observation here")
    parser.add_argument("--quiet", action="store_true", help="summary only")
    args = parser.parse_args(argv)

    result = shadow(args.telemetry, args.package.expanduser().resolve(), args.config,
                    follow_live=args.follow, out=args.out, quiet=args.quiet)
    summarise(result)
    if args.out is not None:
        print(f"\n  wrote {args.out.resolve()} ({len(result['rows'])} decisions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
