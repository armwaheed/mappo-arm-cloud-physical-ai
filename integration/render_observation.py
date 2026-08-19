#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Draw what the policy saw, next to what the camera saw, for every tick of a run.

**The question this exists to answer is "why did it do that".** A range vector is twelve
floats, and twelve floats do not tell anyone whether the robot was looking at a wall, a
gap, or its own stale map. Reading the numbers out of a replay is not enough either: the
observation is built in a RUN-LOCAL frame from a RETAINED obstacle map, so two of the
three things that decide it are not in the tick you are looking at.

Each frame is three panels:

* **the camera**, as ``visual_nav`` annotated it — pulled by ``perception.video_frame``,
  which indexes the run's own MP4. This is the only panel that shows the room rather than
  the robot's model of it, and comparing the two is the whole point.
* **the fan**, top-down in the run-local frame: every ray drawn to what it hit, the
  retained obstacle discs, the horizon, and the clear angular window toward the goal.
* **the observation**, as a bar per ray, in the PROXIMITY convention the checkpoint uses
  (bigger = closer, zero = clear) so the plot matches the vector the network is handed.

Two things it is built to make impossible to miss, because both were argued about from
the numbers alone before anyone drew them:

1. **The fan does not turn with the robot.** Ray 0 points along the heading the run reset
   into, for the whole run. The nose is drawn separately, so the two can be seen to
   disagree.
2. **The retained radius is not the telemetry's radius.** The control stack's radius is
   ``radius_m + position_sigma`` and converges downward as sightings accumulate; the
   controller's copy is drawn solid and the tick's current value dashed, so a map that
   has stopped tracking its own producer is visible rather than inferred.

Desk tool: it imports matplotlib, which is not a deployment dependency and is imported
inside the function that needs it so the module stays importable without it.

    # numbers only, no plotting and no frames needed
    python3 render_observation.py ../evidence/.../run14.jsonl --summary

    # full contact sheet, with the camera panel
    python3 render_observation.py ../evidence/.../run14.jsonl --out /tmp/run14 \\
        --frames /tmp/fr14 --rays 12 24
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from mappo_bridge import robot_input
from observation import (
    fan_bearings,
    rays_to_sample,
    window_containing,
    window_is_sampled,
)
from replay_mappo import OBSERVATION_STATE_WIDTH
from telemetry_reader import read_run

#: The vendored policy package, matching ``replay_mappo``.
DEFAULT_PACKAGE = Path(__file__).resolve().parent.parent / "policy"


def _load_policy(package: Path):
    sys.path.insert(0, str(package))
    try:
        from physical_ai_mappo import MappoController, RobotInput, StationaryObject
    except ImportError as exc:
        raise SystemExit(f"cannot import the policy package from {package}: {exc}") from exc
    return MappoController, RobotInput, StationaryObject


def walk(run, package: Path, config: Path | None = None) -> list:
    """Step the real controller over a run, returning one record per driven tick.

    Everything a panel draws comes from the controller's own state after its own
    :meth:`step`, not from a reconstruction: the retained map, the origin the run reset
    into, and the observation it last handed the network.
    """
    MappoController, RobotInput, StationaryObject = _load_policy(package)
    controller = MappoController(config or package / "config.json")

    records: list = []
    for tick in run.ticks:
        kwargs = robot_input(tick, reset_run=not records)
        if kwargs is None:
            continue
        objects = [StationaryObject(**o) for o in kwargs["stationary_objects"]]
        inp = RobotInput(**{**kwargs, "stationary_objects": objects})
        out = controller.step(inp)
        x, y, yaw, _vx, _vy, gx, gy = controller._local_state(inp)

        discs = [(o.x, o.y, o.radius) for o in controller._obstacles]
        goal_bearing = math.atan2(gy - y, gx - x)
        window = window_containing(x, y, discs, goal_bearing) if discs else None
        # What the tick itself says the radii are now, keyed by id, so the panel can show
        # the controller's copy against its producer's current estimate.
        reported = {o["id"]: o["radius_m"] for o in tick.get("obstacles", [])
                    if o.get("id") and o.get("kind") == "static"}

        records.append({
            "t": tick["t"],
            "video_frame": (tick.get("perception") or {}).get("video_frame"),
            "status": out.status,
            "action": (out.action_x, out.action_y),
            "command": (out.vx_mps, out.vy_mps),
            "pose_local": (x, y, yaw),
            "goal_local": (gx, gy),
            "goal_bearing": goal_bearing,
            "obstacles": [(o.x, o.y, o.radius, o.object_id, reported.get(o.object_id))
                          for o in controller._obstacles],
            "lidar": [float(v)
                      for v in controller.last_observation[OBSERVATION_STATE_WIDTH:]],
            "window": window,
            "horizon_m": controller.cfg.lidar_range_m,
            #: Metres per VMAS unit, carried on the record so a panel cannot be drawn
            #: against a different config from the one that produced its numbers.
            "scale_m": controller.cfg.meters_per_vmas_unit,
        })
    return records


def summarise(records: list, ray_counts) -> None:
    """Print the aperture the goal lies in, and which fans would sample it."""
    print(f"{'t':>6} {'status':<18} {'vx':>7} {'aperture (deg)':>22} {'width':>7} "
          f"{'needs':>6}  " + " ".join(f"N={n}" for n in ray_counts))
    counts = dict.fromkeys(ray_counts, 0)
    open_ticks = 0
    for record in records:
        window = record["window"]
        if window is None:
            print(f"{record['t']:6.1f} {record['status']:<18} "
                  f"{record['command'][0]:+7.3f} {'blocked toward the goal':>22}")
            continue
        open_ticks += 1
        width = window[1] - window[0]
        marks = []
        for n in ray_counts:
            sampled = window_is_sampled(window, n)
            counts[n] += sampled
            marks.append(" yes" if sampled else "  - ")
        print(f"{record['t']:6.1f} {record['status']:<18} {record['command'][0]:+7.3f} "
              f"{f'[{math.degrees(window[0]):+6.1f},{math.degrees(window[1]):+6.1f}]':>22} "
              f"{math.degrees(width):6.1f}  {rays_to_sample(width):5d}  " + " ".join(marks))
    print()
    print(f"  {open_ticks}/{len(records)} driven ticks had an open window toward the goal")
    for n in ray_counts:
        share = counts[n] / open_ticks if open_ticks else 0.0
        print(f"    a {n:>3}-ray fan samples it on {counts[n]:3d} of them ({share:4.0%})")


def _draw(record: dict, frame_path: Path | None, ray_counts, out_path: Path,
          title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge

    x, y, yaw = record["pose_local"]
    horizon = record["horizon_m"]
    rays = len(record["lidar"])

    figure = plt.figure(figsize=(15.5, 5.4))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.32, 1.0, 0.86), wspace=0.22)

    camera = figure.add_subplot(grid[0, 0])
    camera.axis("off")
    if frame_path is not None and frame_path.exists():
        camera.imshow(plt.imread(str(frame_path)))
        stale = "" if record["video_frame"] is not None else "  (held: no new frame)"
        camera.set_title(f"camera, as visual_nav annotated it — "
                         f"frame {record.get('frame_shown')}{stale}", fontsize=8.5)
    else:
        camera.text(0.5, 0.5, "no camera frame\n(pass --frames)",
                    ha="center", va="center", fontsize=9, color="0.45")
        camera.set_title("camera", fontsize=8.5)

    # ---- top-down, run-local ------------------------------------------------------
    top = figure.add_subplot(grid[0, 1])
    top.set_aspect("equal")
    top.add_patch(Circle((x, y), horizon, fill=False, ls=":", color="0.55", lw=1.0))

    for cx, cy, radius, name, reported in record["obstacles"]:
        top.add_patch(Circle((cx, cy), radius, color="tab:red", alpha=0.22))
        top.add_patch(Circle((cx, cy), radius, fill=False, color="tab:red", lw=1.2))
        if reported is not None and abs(reported - radius) > 1e-3:
            top.add_patch(Circle((cx, cy), reported, fill=False, color="tab:red",
                                 ls="--", lw=1.0, alpha=0.9))
        top.annotate(f"{name}\nr={radius:.2f}"
                     + (f" (now {reported:.2f})" if reported is not None
                        and abs(reported - radius) > 1e-3 else ""),
                     (cx, cy), fontsize=6.5, ha="center", va="center")

    window = record["window"]
    if window is not None:
        top.add_patch(Wedge((x, y), horizon * 0.98, math.degrees(window[0]),
                            math.degrees(window[1]), color="tab:green", alpha=0.16))

    for index, bearing in enumerate(fan_bearings(rays)):
        proximity = record["lidar"][index]
        # proximity = lidar_range_vmas - range/scale, so range = horizon - proximity*scale
        reach = horizon if proximity <= 0.0 else max(
            0.0, horizon - proximity * record["scale_m"])
        blocked = proximity > 0.0
        top.plot([x, x + reach * math.cos(bearing)], [y, y + reach * math.sin(bearing)],
                 color="tab:red" if blocked else "0.72",
                 lw=1.6 if blocked else 0.8, zorder=3)
        label = x + (horizon + 0.10) * math.cos(bearing), \
            y + (horizon + 0.10) * math.sin(bearing)
        top.annotate(str(index), label, fontsize=6, ha="center", va="center",
                     color="tab:red" if blocked else "0.55")

    top.plot([x, x + 0.42 * math.cos(yaw)], [y, y + 0.42 * math.sin(yaw)],
             color="tab:blue", lw=2.4, zorder=4)
    top.plot([x], [y], "o", color="tab:blue", ms=6, zorder=5)
    span = horizon + 0.6
    gx, gy = record["goal_local"]
    if abs(gx - x) <= span and abs(gy - y) <= span:
        top.plot([gx], [gy], "*", color="tab:green", ms=13, zorder=5)
        top.annotate("goal", (gx, gy), fontsize=7, xytext=(4, 5),
                     textcoords="offset points")
    else:
        # Off the square, which is the normal case: the goal is metres away and the fan
        # is under a metre deep. Draw the BEARING, since that is the part of the goal the
        # observation actually carries into the network.
        bearing = record["goal_bearing"]
        reach = span * 0.92
        top.annotate("", (x + reach * math.cos(bearing), y + reach * math.sin(bearing)),
                     xytext=(x, y), arrowprops={"arrowstyle": "-|>", "lw": 1.4,
                                                "color": "tab:green"}, zorder=5)
        top.annotate(f"goal {math.hypot(gx - x, gy - y):.1f} m",
                     (x + reach * math.cos(bearing), y + reach * math.sin(bearing)),
                     fontsize=7, color="tab:green", ha="center",
                     xytext=(0, 7), textcoords="offset points")

    top.set_xlim(x - span, x + span)
    top.set_ylim(y - span, y + span)
    top.set_title(f"the {rays}-ray fan, run-local frame\n"
                  f"ray 0 = start heading (blue = nose, {math.degrees(yaw):+.0f} deg off)",
                  fontsize=9)
    top.tick_params(labelsize=6)

    # ---- the observation itself ----------------------------------------------------
    bars = figure.add_subplot(grid[0, 2])
    values = record["lidar"]
    bars.bar(range(rays), values,
             color=["tab:red" if v > 0 else "0.8" for v in values])
    bars.set_xticks(range(rays))
    bars.set_xticklabels([f"{i}\n{round(math.degrees(b))}" for i, b
                          in enumerate(fan_bearings(rays))], fontsize=5.5)
    bars.set_ylim(0, max(0.36, max(values) * 1.25 if values else 0.36))
    bars.set_title("the observation handed to the network\n"
                   "proximity: bigger = closer, 0 = clear", fontsize=9)
    bars.tick_params(axis="y", labelsize=6)
    bars.grid(axis="y", lw=0.4, alpha=0.4)

    if window is not None:
        width = window[1] - window[0]
        verdict = "  ".join(
            f"N={n}:{'sees it' if window_is_sampled(window, n) else 'BLIND'}"
            for n in ray_counts)
        note = (f"clear window toward the goal {math.degrees(width):.1f} deg "
                f"-> needs {rays_to_sample(width)} rays to be sure   |   {verdict}")
    else:
        note = "no clear window toward the goal: the discs meet across it"

    figure.suptitle(
        f"{title}    t={record['t']:.1f}s   {record['status']}   "
        f"action=({record['action'][0]:+.2f},{record['action'][1]:+.2f})   "
        f"command=({record['command'][0]:+.2f},{record['command'][1]:+.2f}) m/s\n{note}",
        fontsize=9.5)
    figure.savefig(out_path, dpi=115, bbox_inches="tight")
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry", type=Path, help="a run's .jsonl")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--config", type=Path,
                        help="policy config; defaults to the package's config.json")
    parser.add_argument("--out", type=Path,
                        help="directory for one PNG per tick; omit for numbers only")
    parser.add_argument("--frames", type=Path,
                        help="directory of f%%04d.jpg extracted from the run's MP4, "
                             "indexed by perception.video_frame")
    parser.add_argument("--rays", type=int, nargs="+", default=[12, 16, 24],
                        help="fans to test the clear window against")
    parser.add_argument("--summary", action="store_true",
                        help="print the per-tick aperture table")
    args = parser.parse_args(argv)

    run = read_run(args.telemetry)
    records = walk(run, args.package, args.config)
    if not records:
        print("no tick carried a goal — nothing to draw")
        return 1

    if args.summary or not args.out:
        summarise(records, args.rays)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        title = args.telemetry.stem
        # Perception runs slower than control, so most ticks carry no NEW frame. Holding
        # the last one keeps the camera panel populated and labels it as held, which is
        # honest; blanking it would suggest the robot was flying blind between frames.
        held = None
        for index, record in enumerate(records):
            if args.frames is not None and record["video_frame"] is not None:
                candidate = args.frames / f"f{record['video_frame']:04d}.jpg"
                if candidate.exists():
                    held = (candidate, record["video_frame"])
            record["frame_shown"] = held[1] if held else None
            _draw(record, held[0] if held else None, args.rays,
                  args.out / f"tick{index:03d}.png", title)
        print(f"\nwrote {len(records)} frames to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
