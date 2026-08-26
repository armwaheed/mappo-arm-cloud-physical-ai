#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Why one run cleared the peer and the other drove through it.

Regenerates every number in this directory's README from the two telemetry files.
No robot, no network, nothing outside the standard library:

    python3 bearings.py

THE CLAIM UNDER TEST. The hero run clears the Go2 Wheel. Is that the policy steering
around it, or the goal happening to sit on the other side of the nose? Two things
would support avoidance: a lateral command that appears (or grows) when the peer is
first detected, or one that tracks the peer's RANGE. Neither appears; the lateral
command correlates with the distance still to run to the GOAL at r=+0.95, and with
the peer's range at r=+0.05.

READING THE OUTPUT. Bearings are body-frame, positive to the LEFT of the nose.
`command.vy` is positive to the left, so a swerve away from a peer on the right is
POSITIVE vy.

The per-track table is printed raw and deliberately un-filtered. The tracker fragments
the peer across several short-lived ids with inconsistent labels -- that is the finding
of PR #73, not a defect in this script -- and it also carries near-field junk tracks
whose bearings swing through 180 degrees. Do not read a track's label as an identity.
In the hero run the peer is `track-3`, identified from the frame itself
(`hero-contact-sheet.jpg`, top middle: the Go2 Wheel boxed at 1.3 m), and `track-7`
closing to 0.02 m is the goal chair being ARRIVED at, not a collision.
"""

from __future__ import annotations

import collections
import json
import math
import pathlib

RUNS = (
    ("hero     peer off-axis, cleared", "hero-run-telemetry.jsonl"),
    ("contrast peer on the goal bearing, hit", "contrast-run-telemetry.jsonl"),
)

# The peer's track id in the hero run, read off the annotated frame, not off the label.
HERO_PEER_TRACK = "track-3"


def wrap_deg(radians: float) -> float:
    """Radians to degrees on (-180, 180]."""
    return (math.degrees(radians) + 180.0) % 360.0 - 180.0


def bearing_to(pose: dict, point: dict) -> tuple[float, float]:
    """Body-frame (range_m, bearing_deg) from an odom pose to an odom point."""
    dx = point["x"] - pose["x"]
    dy = point["y"] - pose["y"]
    return math.hypot(dx, dy), wrap_deg(math.atan2(dy, dx) - pose["yaw"])


def correlation(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx and sy else float("nan")


def load(path: pathlib.Path) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return records[0], [r for r in records if r.get("type") != "header"]


def report(title: str, path: pathlib.Path) -> None:
    header, ticks = load(path)
    dt = 1.0 / header.get("control_hz", 10.0)

    goal_bearings: list[float] = []
    goal_distances: list[float] = []
    lateral_cmds: list[float] = []
    peer_ranges: list[tuple[float, float]] = []
    tracks: dict[str, list[tuple[int, float, float]]] = collections.defaultdict(list)
    forward = lateral = 0.0

    for index, tick in enumerate(ticks):
        command, pose, goal = tick.get("command"), tick.get("pose"), tick.get("goal")
        if command:
            forward += command["vx"] * dt
            lateral += command["vy"] * dt
        if not pose:
            continue
        for obstacle in tick.get("obstacles") or []:
            range_m, bearing = bearing_to(pose, obstacle)
            tracks[obstacle["id"]].append((index, range_m, bearing))
            if command and obstacle["id"] == HERO_PEER_TRACK:
                peer_ranges.append((range_m, command["vy"]))
        if not (goal and command):
            continue
        distance, bearing = bearing_to(pose, goal)
        goal_distances.append(distance)
        goal_bearings.append(bearing)
        lateral_cmds.append(command["vy"])

    # Measured velocity is what the legs delivered; command is what the policy asked for.
    delivered = sum(t["measured"]["vy"] * dt for t in ticks if t.get("measured"))
    left = sum(1 for v in lateral_cmds if v > 0)

    print(f"\n{title}")
    print(f"  {path.name}, robot-side {header.get('video')}, {len(ticks)} ticks")
    print(f"  goal bearing        {min(goal_bearings):+.1f} .. {max(goal_bearings):+.1f} deg"
          f"   (mean {sum(goal_bearings) / len(goal_bearings):+.1f})")
    print(f"  commanded lateral   leftward on {left}/{len(lateral_cmds)} ticks,"
          f" {min(lateral_cmds):+.3f} .. {max(lateral_cmds):+.3f} m/s")
    print(f"  integrated          {forward:.2f} m forward,"
          f" {lateral:+.2f} m lateral commanded, {delivered:+.2f} m delivered"
          f" ({abs(delivered / lateral):.0%} of it)")
    print(f"  corr(vy, goal distance remaining) = {correlation(lateral_cmds, goal_distances):+.3f}")
    print(f"  corr(vy, goal bearing)            = {correlation(lateral_cmds, goal_bearings):+.3f}")
    if peer_ranges:
        print(f"  corr(vy, peer range)              ="
              f" {correlation([r for r, _ in peer_ranges], [v for _, v in peer_ranges]):+.3f}"
              f"   over the {len(peer_ranges)} ticks {HERO_PEER_TRACK} was tracked")

    print("  tracks (raw, unfiltered -- labels are noise, see the docstring):")
    for track_id, samples in sorted(tracks.items(), key=lambda kv: -len(kv[1])):
        ranges = [s[1] for s in samples]
        bearings = [s[2] for s in samples]
        print(f"    {track_id:<9} n={len(samples):>3}  ticks {samples[0][0]:>3}-{samples[-1][0]:<3}"
              f"  range {min(ranges):.2f}..{max(ranges):.2f} m"
              f"  bearing {min(bearings):+7.1f}..{max(bearings):+7.1f} deg")


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    for title, name in RUNS:
        report(title, here / name)

    _, ticks = load(here / "hero-run-telemetry.jsonl")
    first = next(i for i, t in enumerate(ticks)
                 if any(o["id"] == HERO_PEER_TRACK for o in t.get("obstacles") or []))
    before = [t["command"]["vy"] for t in ticks[:first] if t.get("command")]
    during = [t["command"]["vy"] for t in ticks[first:] if t.get("command")
              and any(o["id"] == HERO_PEER_TRACK for o in t.get("obstacles") or [])]
    print(f"\nHero run, the decisive comparison. The peer is first tracked at tick {first}.")
    print(f"  lateral command BEFORE the peer is ever seen:"
          f" {min(before):+.3f} .. {max(before):+.3f} m/s")
    print(f"  lateral command WHILE the peer is tracked:   "
          f" {min(during):+.3f} .. {max(during):+.3f} m/s")
    print("  The leftward command predates the peer and does not grow when it appears.")


if __name__ == "__main__":
    main()
