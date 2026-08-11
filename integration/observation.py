# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn one telemetry tick into the observation a MAPPO policy expects.

The policy was trained in a simulator whose agents see a **range vector** — N rays cast
out from the robot, each reporting the distance to the first thing it hits. The control
stack does not produce that; it produces a short list of *objects* with positions and
radii, which is a richer and smaller representation. :func:`range_vector` is the adapter.

IT IS EXACT RAY-VERSUS-DISC GEOMETRY, not "the nearest ray gets the object", because a
sampled fan is coarser than it looks and the staged scene lands right on the limit: the
bin is 0.46 m across at 2.15 m, which subtends **13.2°**, while 16 rays over a full
circle sit **22.5°** apart. Bucketing by centre bearing would put it on one ray and leave
its neighbours reading clear; spreading the fan over 360° at all makes it invisible
between two rays. :func:`reliable_range_m` turns that into a number you can check a
checkpoint against, and :data:`DEFAULT_FOV_RAD` is the camera's real field of view for
exactly this reason.

WHAT THIS ADAPTER CANNOT INVENT, and the reason it is worth stating in code rather than
in a slide:

  * **It sees only what the stack detected.** A range vector from a real LiDAR reports
    walls, table legs and doorframes. This one reports tracked people and one named
    coloured prop, and reads "clear" everywhere else. Free space here means *nothing was
    recognised*, not *nothing is there*.
  * **It has no rear view.** The camera spans ~85° and the robot never reverses. Rays
    outside the field of view are reported as unknown (``max_range_m``), which is the
    optimistic direction — a policy that learned to back out of a dead end will believe
    the space behind it is clear.
  * **Peers are not in it.** Another quadruped is not a detector class and not a colour
    profile, so a multi-robot policy's most important obstacle is invisible.

Pure stdlib; no numpy so it drops into any environment. ``python3 test_observation.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Rays in the fan. Matches the common MAPPO lidar-observation width. Whatever the
#: checkpoint wants, check it against :func:`reliable_range_m` — see the module docstring
#: for why a fan can look fine and still be blind to the one obstacle in the scene.
DEFAULT_RAYS = 16

#: Default angular span of the fan. The CAMERA's measured field of view, not 360, and
#: that is a deliberate difference from the simulator. A 16-ray fan spread over the full
#: circle puts its rays 22.5 deg apart, while the staged 0.30 m bin at 2 m subtends
#: 13.2 deg — it falls cleanly BETWEEN two rays and the policy sees open floor where the
#: only obstacle in the scene is. Over 85.27 deg the same 16 rays are 5.3 deg apart and
#: the bin is caught out to 4.95 m. Widening this back to 2*pi without raising `rays` is
#: the single easiest way to make this adapter quietly useless.
DEFAULT_FOV_RAD = math.radians(85.27)

#: Beyond this a ray reports "clear". Set from the DETECTOR's usable band, not the
#: camera's: person recall thins past ~6 m, so a longer vector would be reporting
#: confidence the perception stack does not have.
DEFAULT_MAX_RANGE_M = 6.0


@dataclass(frozen=True)
class Observation:
    """One policy input, in the robot's BODY frame (+x forward, +y left)."""

    goal_range_m: float
    goal_bearing_rad: float
    ranges_m: tuple            # DEFAULT_RAYS entries, clockwise from straight ahead
    ray_bearings_rad: tuple
    speed: tuple               # the velocity actually commanded last tick (vx, vy, wz)

    def as_vector(self) -> list:
        """Flat float list: goal polar, then the range fan, then the last command."""
        return [self.goal_range_m, self.goal_bearing_rad, *self.ranges_m, *self.speed]


def reliable_range_m(radius_m: float, rays: int = DEFAULT_RAYS,
                     fov_rad: float = DEFAULT_FOV_RAD) -> float:
    """Range beyond which an object of ``radius_m`` can slip between two rays.

    A fan samples; it does not integrate. An object is only *guaranteed* to be hit while
    the half-angle it subtends, ``asin(r/d)``, is at least half the ray spacing — so the
    guarantee holds out to ``r / sin(spacing/2)`` and no further. Past that it depends on
    where the object happens to sit relative to the rays, which is not a property anyone
    should be relying on.

    For the shipped defaults and the staged bin (r = 0.23 m including its uncertainty
    margin) this is 4.95 m, comfortably past the detector's own ~6 m band. Spread the
    same 16 rays over a full circle instead and it collapses to 1.18 m.
    """
    if rays <= 0 or fov_rad <= 0.0:
        raise ValueError("rays and fov_rad must be positive")
    half_spacing = (fov_rad / rays) / 2.0
    if half_spacing >= math.pi / 2.0:
        return 0.0
    return radius_m / math.sin(half_spacing)


def wrap_pi(angle: float) -> float:
    """Wrap to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def to_body_frame(pose: dict, x: float, y: float) -> tuple:
    """An odom point in the robot's body frame.

    Telemetry is in ODOM because that is the only frame in which a mapped landmark stays
    put while the robot walks. A policy wants body-relative, so the transform happens
    here — once, explicitly — rather than being open-coded at each call site with a sign
    error waiting in it.
    """
    dx, dy = x - pose["x"], y - pose["y"]
    cos_yaw, sin_yaw = math.cos(-pose["yaw"]), math.sin(-pose["yaw"])
    return (dx * cos_yaw - dy * sin_yaw, dx * sin_yaw + dy * cos_yaw)


def _ray_hits_disc(bearing_rad: float, cx: float, cy: float, radius_m: float
                   ) -> float | None:
    """Distance along a ray from the origin to a disc, or ``None`` if it misses.

    Solves ``|t*d - c|^2 = r^2`` for the near root, so a ray that merely clips the edge
    of a disc still returns the near face rather than the centre distance.

    INSIDE THE DISC IS ZERO, not the exit distance. The near root goes negative there and
    the far root is where the ray leaves the far side — 0.55 m, for a robot sitting 0.05 m
    inside a 0.5 m disc. Reporting that would tell the policy it has half a metre of clear
    space while it is already inside the obstacle, which is the one direction an
    observation must never be wrong in. This is reachable in normal operation because the
    radius carries the map's uncertainty margin, so the robot can be inside the DISC while
    nowhere near the bin.
    """
    dx, dy = math.cos(bearing_rad), math.sin(bearing_rad)
    projection = dx * cx + dy * cy            # d . c
    centre_distance_sq = cx * cx + cy * cy
    if centre_distance_sq <= radius_m ** 2:
        return 0.0
    discriminant = projection ** 2 - (centre_distance_sq - radius_m ** 2)
    if discriminant < 0.0:
        return None
    near = projection - math.sqrt(discriminant)
    return near if near >= 0.0 else None       # the disc is entirely behind the ray


def range_vector(tick: dict, rays: int = DEFAULT_RAYS,
                 max_range_m: float = DEFAULT_MAX_RANGE_M,
                 fov_rad: float = DEFAULT_FOV_RAD) -> tuple:
    """``(ranges_m, ray_bearings_rad)`` — the LiDAR-like fan for one tick.

    Rays are spread evenly over ``fov_rad`` centred on the nose, and default to the
    camera's real field of view rather than a simulator's full circle. If the checkpoint
    needs 360 degrees, pass ``fov_rad=2*math.pi`` — and raise ``rays`` to match, because
    the spacing is what decides whether an obstacle is seen at all
    (:func:`reliable_range_m`). Bearings the camera cannot see report ``max_range_m``,
    i.e. "clear", which the module docstring flags as the optimistic direction.
    """
    pose = tick["pose"]
    discs = []
    for obstacle in tick.get("obstacles", []):
        bx, by = to_body_frame(pose, obstacle["x"], obstacle["y"])
        discs.append((bx, by, obstacle["radius_m"]))

    ranges = []
    bearings = []
    for index in range(rays):
        bearing = wrap_pi(-fov_rad / 2.0 + fov_rad * (index + 0.5) / rays)
        bearings.append(bearing)
        nearest = max_range_m
        for cx, cy, radius in discs:
            hit = _ray_hits_disc(bearing, cx, cy, radius)
            if hit is not None and hit < nearest:
                nearest = hit
        ranges.append(nearest)
    return tuple(ranges), tuple(bearings)


def observation_from_tick(tick: dict, rays: int = DEFAULT_RAYS,
                          max_range_m: float = DEFAULT_MAX_RANGE_M,
                          fov_rad: float = DEFAULT_FOV_RAD) -> Observation | None:
    """Build an :class:`Observation`, or ``None`` if the tick has no goal yet.

    A tick without a goal is a real part of the episode — the robot is searching, and
    the telemetry records it deliberately — but it has no policy input, so the caller
    decides what to do about it rather than being handed a zero-filled vector that looks
    like a goal at the origin.
    """
    if not tick.get("goal"):
        return None
    pose = tick["pose"]
    goal_x, goal_y = to_body_frame(pose, tick["goal"]["x"], tick["goal"]["y"])
    command = tick.get("command") or {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    ranges, bearings = range_vector(tick, rays=rays, max_range_m=max_range_m,
                                    fov_rad=fov_rad)
    return Observation(
        goal_range_m=math.hypot(goal_x, goal_y),
        goal_bearing_rad=math.atan2(goal_y, goal_x),
        ranges_m=ranges,
        ray_bearings_rad=bearings,
        speed=(command["vx"], command["vy"], command["wz"]))
