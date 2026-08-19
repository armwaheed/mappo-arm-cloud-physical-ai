#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay the two-bin runs with the radius rule BEFORE and AFTER correction 6.

The before/after has to be reproducible from one checkout or nobody will re-run it, so
this restores the delivered ``max(match.radius, radius)`` by patching the method rather
than by asking the reader to check out a parent commit. The "after" path is the shipped
code, untouched — so if the correction is ever reverted, both columns here read the same
and the table stops making its point, which is the failure mode worth having.

    python3 radius_latch.py

Everything printed is measured on the recorded telemetry in
``evidence/2026-08-18-threading-two-bins/``. No robot, no network access.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "integration"))
sys.path.insert(0, str(ROOT / "policy"))

import physical_ai_mappo as policy                              # noqa: E402
from mappo_bridge import robot_input                            # noqa: E402
from observation import window_containing, window_is_sampled    # noqa: E402
from telemetry_reader import read_run                           # noqa: E402

RUNS = ("run11-SUCCESS-threaded-the-gap",
        "run10-forward-clamp-pure-strafe",
        "run14-ray0-blocked-policy-retreats",
        "run15-aimed-8deg-off-corridor")

SHIPPED = policy.MappoController._update_obstacles


def _latched(self, inp, x, y, yaw, now):
    """The delivered rule: a mapped radius can only ever grow.

    Re-imposed on top of the corrected update rather than reimplemented, so the two
    columns differ in exactly one thing and nothing else can drift between them.
    """
    SHIPPED(self, inp, x, y, yaw, now)
    for obstacle in self._obstacles:
        # Keyed on the disc itself, not on its id: the delivered high-water mark was per
        # mapped obstacle, and two anonymous ones would share the key `None`.
        opening = self._opening_radius.setdefault(id(obstacle), obstacle.radius)
        obstacle.radius = max(opening, obstacle.radius)


def replay(name: str, latch: bool) -> dict:
    policy.MappoController._update_obstacles = _latched if latch else SHIPPED
    run = read_run(ROOT / "evidence/2026-08-18-threading-two-bins" / f"{name}.jsonl")
    controller = policy.MappoController(ROOT / "policy/config.json")
    controller._opening_radius = {}

    driven, widths, ray0_blocked, sampled = [], [], 0, {12: 0, 16: 0, 24: 0}
    both_in_horizon = 0
    for tick in run.ticks:
        kwargs = robot_input(tick, reset_run=not driven)
        if kwargs is None:
            continue
        objects = [policy.StationaryObject(**o) for o in kwargs["stationary_objects"]]
        inp = policy.RobotInput(**{**kwargs, "stationary_objects": objects})
        out = controller.step(inp)
        x, y, _yaw, _vx, _vy, gx, gy = controller._local_state(inp)
        driven.append(out)

        horizon = controller.cfg.lidar_range_m
        both_in_horizon += sum(
            math.hypot(o.x - x, o.y - y) - o.radius <= horizon
            for o in controller._obstacles) >= 2
        ray0_blocked += controller.last_observation[6] > 0.0

        window = window_containing(x, y, [(o.x, o.y, o.radius)
                                          for o in controller._obstacles],
                                   math.atan2(gy - y, gx - x))
        if window is None:
            continue
        widths.append(math.degrees(window[1] - window[0]))
        for rays in sampled:
            sampled[rays] += window_is_sampled(window, rays)

    commanding = [o for o in driven if o.status == "COMMAND"]
    return {
        "ticks": len(driven),
        "radii": sorted(round(o.radius, 3) for o in controller._obstacles),
        "both_in_horizon": both_in_horizon,
        "ray0_blocked": ray0_blocked,
        "mean_vx": (sum(o.vx_mps for o in commanding) / len(commanding)
                    if commanding else float("nan")),
        "median_width": (sorted(widths)[len(widths) // 2] if widths else float("nan")),
        "open": len(widths),
        "sampled": sampled,
    }


def main() -> int:
    horizon = policy.MappoController(ROOT / "policy/config.json").cfg.lidar_range_m
    print("Recorded two-bin runs of 2026-08-18, replayed through the real checkpoint.")
    print(f"Policy horizon {horizon:.3f} m. Bin separation measured 1.27-1.43 m "
          f"centre to centre.\n")

    for name in RUNS:
        before, after = replay(name, latch=True), replay(name, latch=False)
        print(f"{name}   ({after['ticks']} driven ticks)")
        print(f"{'':24}{'latched (delivered)':>22}{'converged (corrected)':>24}")
        rows = [
            ("retained radii, m", str(before["radii"]), str(after["radii"])),
            ("both bins in horizon", f"{before['both_in_horizon']} ticks",
             f"{after['both_in_horizon']} ticks"),
            ("ray 0 blocked", f"{before['ray0_blocked']} ticks",
             f"{after['ray0_blocked']} ticks"),
            ("mean commanded vx", f"{before['mean_vx']:+.3f} m/s",
             f"{after['mean_vx']:+.3f} m/s"),
            ("median clear window", f"{before['median_width']:.1f} deg",
             f"{after['median_width']:.1f} deg"),
        ]
        for rays in (12, 16, 24):
            rows.append((f"{rays}-ray fan samples it",
                         f"{before['sampled'][rays]}/{before['open']}",
                         f"{after['sampled'][rays]}/{after['open']}"))
        for label, left, right in rows:
            print(f"  {label:<22}{left:>22}{right:>24}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
