# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable record of what the robot believed and did, one line per control tick.

THIS EXISTS BECAUSE THE CONSOLE LOG IS NOT AN INTERFACE. The human log is genuinely
useful — it is how the phantom-person false positive was caught — but it is prose, and
prose is the wrong thing to hand a learning agent. Two specific failures made that
concrete rather than theoretical:

  * **It does not carry what a consumer needs.** An integrator read a live run's log and
    reported it had "odometry/pose, goal info, camera data and the motion commands". It
    has the commands, and the goal as a *scalar distance*. The robot's pose appears
    ONCE, in a start-up banner, and never again across 107 control ticks. There is no
    camera data at all — the ``lat=235ms`` field is a frame's AGE. An observation vector
    cannot be built from it, and that is not discoverable without counting the lines.
  * **It is not stable.** ``people=0`` became ``obst=[binx1,personx1]`` in the same week,
    for a good reason (a bare count cannot distinguish a mapped bin from a ghost). Any
    parser written against the old shape was silently broken by an improvement to a
    human-facing string, which is exactly the coupling a format should not create.

So: JSONL, one object per line, a versioned schema, and a header record that pins the
run's configuration. Fields are added freely; anything renamed or removed bumps
:data:`SCHEMA`.

FLUSHED EVERY LINE, deliberately. The normal way one of these runs ends is Ctrl-C or a
safety abort, and a buffered writer loses the last few seconds — which is precisely the
window that explains why the run ended. The cost is a write syscall at 10 Hz.

JOINING TO THE VIDEO. ``video_frame`` is the index of the annotated frame written by
``--record`` at this tick, or null on ticks that wrote none (the recorder emits one frame
per PERCEPTION cycle and the controller runs faster). That is the join key between this
file and the MP4 — the honest way to carry "camera data" without putting pixels in a log.

Pure stdlib, no robot: ``python3 test_telemetry.py``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

#: Bump the major on any rename or removal; additions do not bump it. Consumers should
#: refuse a major they do not know rather than guess.
SCHEMA = "go2.visual_nav.telemetry/1"

#: WHICH FRAME EVERY VECTOR IN THIS FILE IS IN. Written into the header because it is a
#: property of the schema, not of a run, and because it is the one thing a consumer
#: cannot recover from the data: odom and body agree exactly while the robot faces its
#: start heading and diverge as it turns, so an integration built on the wrong
#: assumption passes every bench test and fails in the first corner. An integration note
#: for the MAPPO policy package proposed reading `measured` as odom; it is body, and
#: nothing in the file said so.
FRAMES = {
    "pose": "odom",                 # x, y metres; yaw radians CCW
    "goal": "odom",
    "obstacles": "odom",            # position AND velocity
    "command": "body",              # vx forward, vy left, wz CCW
    "measured": "body",             # the estimator's own body-frame velocity
}


def _finite(value):
    """JSON has no infinity. Emit ``null`` instead of a token no strict parser accepts.

    ``json.dump`` writes bare ``Infinity`` by default, which is valid JavaScript and
    invalid JSON — Python reads it back, most other stacks reject the line. ``gap_m`` is
    infinite whenever the lane is clear, i.e. on most ticks of a good run, so this is the
    common case rather than an edge one.
    """
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class TelemetryWriter:
    """Append one JSON object per control tick to a file.

    Args:
        path: output ``.jsonl``. Overwritten, not appended to — one file per run.
        clock: wall-clock source, injected for tests. Used only for ``wall_time``, so a
            reader can line these up against logs from another machine.

    A failure to write must never take down a run that is driving a robot, so
    :meth:`write` swallows its exceptions after saying so once. The run is the product;
    the telemetry is a record of it.
    """

    def __init__(self, path: str | Path, clock=time.time) -> None:
        self._path = Path(path)
        self._clock = clock
        self._handle = self._path.open("w", encoding="utf-8")
        self._records = 0
        self._failed = False

    @property
    def records(self) -> int:
        """Lines written, header included — for the run summary."""
        return self._records

    def _emit(self, record: dict) -> None:
        if self._handle is None or self._failed:
            return
        try:
            self._handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._handle.flush()
            self._records += 1
        except Exception as exc:
            self._failed = True
            print(f"[telemetry] write failed, continuing without it: {exc!r}")

    def write_header(self, **config) -> None:
        """First line: the schema, the wall clock, and whatever pins this run.

        Everything a consumer needs to interpret the ticks that follow — the camera
        model, the envelope, the priors — belongs here rather than being re-sent 900
        times, and belongs in the FILE rather than in a sibling document that can be
        separated from it. :data:`FRAMES` is added unconditionally for that last reason:
        it is not a run parameter and no caller should be able to omit it.
        """
        self._emit({"type": "header", "schema": SCHEMA, "frames": FRAMES,
                    "wall_time": self._clock(), **config})

    def write_tick(self, *, elapsed_s: float, pose, goal_xy, goal_distance_m,
                   command, obstacles, frame_age_s: float, perception_seq: int,
                   detect_ms: float, standing: bool, live: bool,
                   video_frame: int | None = None, stale: bool = False,
                   measured=None, health=None, sightings=(),
                   goal_crop: float | None = None) -> None:
        """One control tick, whether or not it commanded motion.

        EVERY tick is written, including holds, stale-perception skips and the
        goal-search phase. A learning agent needs the gaps as much as the motion — "the
        robot stood still for 1.4 s" is a training signal, and a file that only contains
        the interesting ticks silently re-times the whole episode.
        """
        self._emit({
            "type": "tick",
            "t": round(float(elapsed_s), 4),
            "wall_time": self._clock(),
            "pose": {"x": _finite(pose[0]), "y": _finite(pose[1]),
                     "yaw": _finite(pose[2])},
            "goal": (None if goal_xy is None else
                     {"x": _finite(goal_xy[0]), "y": _finite(goal_xy[1]),
                      "distance_m": _finite(goal_distance_m)}),
            "command": (None if command is None else {
                "vx": _finite(command.vx), "vy": _finite(command.vy),
                "wz": _finite(command.wz), "reason": command.reason,
                "gap_m": _finite(command.gap_m),
                "feasible": int(command.feasible),
                "evaluated": int(command.evaluated)}),
            # Positions and radii, not just a count: a consumer building an occupancy or
            # range vector needs the geometry. `kind` separates a mapped static prop
            # from a tracked mover and `label` does NOT — label is a class name, and it
            # happened to work only while the scene had exactly one mapped prop and one
            # detector class. `id` is the identity that lets a consumer follow one
            # object across ticks instead of re-associating by position.
            "obstacles": [{"label": o.label, "kind": o.kind, "id": o.object_id,
                           "x": _finite(o.x), "y": _finite(o.y),
                           "vx": _finite(o.vx), "vy": _finite(o.vy),
                           "radius_m": _finite(o.radius_m)}
                          for o in obstacles],
            # THE RAW MEASUREMENT, before the map fuses it. Everything in `obstacles`
            # above is a Kalman estimate in odom, so a range recomputed from it is just
            # the map re-derived and cannot audit the map. Two open questions needed this
            # and could not be answered without it: whether the size-prior range scale is
            # right (compare `range_m` against odometry over an approach), and how a
            # detection ranged at 0.8 m became a landmark 0.18 m from the robot. `source`
            # is which prior produced it — "height", "width" or "frame-fill" — because a
            # frame-fill reading is a constant, not a measurement, and a consumer must be
            # able to tell those apart.
            "sightings": [{"label": item.detection.label, "range_m": _finite(item.range_m),
                           "bearing_rad": _finite(item.bearing_rad), "source": item.source,
                           "score": _finite(item.detection.score),
                           "box": [_finite(item.detection.x1), _finite(item.detection.y1),
                                   _finite(item.detection.x2), _finite(item.detection.y2)]}
                          for item in sightings],
            "perception": {"seq": int(perception_seq),
                           # The crop the goal pass ACTUALLY used this tick, which is no
                           # longer the flag the operator passed: it widens as the goal
                           # nears so the target stops being clipped. A goal that jumps
                           # is the first thing anyone suspects, and without this there
                           # is no way to tell a moving crop from a hopping detection.
                           "goal_crop": _finite(goal_crop) if goal_crop is not None
                           else None,
                           "frame_age_s": round(float(frame_age_s), 4),
                           "detect_ms": round(float(detect_ms), 2),
                           "video_frame": video_frame,
                           # The robot is BLIND this tick and holding, but it has not
                           # forgotten its goal. Distinguishing the two matters: a null
                           # goal means "never acquired / lost", which is a different
                           # event entirely.
                           "stale": bool(stale)},
            # What the robot ACTUALLY did, beside what it was told to do. Recording only
            # the command makes a whole class of failure undiagnosable from the file: a
            # run that commanded 0.12 m/s forward and 0.20 m/s of strafe for fifty
            # seconds and moved nothing reads, in the command alone, exactly like a run
            # that was walking. The two are told apart here and nowhere else.
            "measured": (None if measured is None else
                         {"vx": _finite(measured[0]), "vy": _finite(measured[1]),
                          "wz": _finite(measured[2])}),
            "posture": ("standing" if standing else "prone"),
            "live": bool(live),
            "health": (None if health is None else {
                "motor_temp_c": _finite(getattr(health, "max_motor_temp_c", None)),
                "battery_pct": _finite(getattr(health, "battery_soc_pct", None))}),
        })

    def write_outcome(self, outcome: str, **extra) -> None:
        """Last line: how the run ended, so a truncated file is distinguishable from an
        aborted one."""
        self._emit({"type": "outcome", "outcome": outcome,
                    "wall_time": self._clock(), **extra})

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> TelemetryWriter:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
