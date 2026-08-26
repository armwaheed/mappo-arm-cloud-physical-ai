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

WHERE THE TICK WENT. ``profile`` carries the wall clock of each stage of one control
tick, because issue #18 spent eight days unable to say. The loop measured 3.15 Hz and
3.46 Hz against a configured 10 Hz on the 2026-08-25 runs, and the tree contained no
``perf_counter``, no ``cProfile`` and no stage timer at all: the only number anywhere was
``detect_ms``, which times the PERCEPTION thread and therefore explains none of the
control tick. It is written on every tick rather than behind a flag, because five clock
reads cost microseconds and a profiler you have to remember to switch on is off during
every run anyone later wants to explain. See :class:`TickProfiler`.

READING ONE BACK is ``python3 telemetry.py <run.jsonl>`` — the rate, the staleness margin
and where the tick went. It lives in this file, beside the writer, so the field names have
exactly one definition; a reader in another module is a second copy of the schema, and
this repository has already shipped a manifest and a frame-key convention that drifted
from the code meant to read them.

Pure stdlib, no robot: ``python3 test_telemetry.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import sys
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


#: Stages of one control tick, in the order ``VisualNavigator.run`` executes them.
#: ENUMERATED HERE rather than left to the call sites, for two reasons. A reader of a
#: telemetry file needs one place that says what each name covers; and a stage that did not
#: run on a given tick — ``plan`` on a stale-perception hold, ``record`` on the four ticks
#: in five that write no frame — must still appear, as ``0.0``, or every consumer has to
#: decide for itself whether a missing key means "free" or "did not happen".
TICK_STAGES = (
    "perceive",     # latest() — reading the newest result off the perception thread
    "tracker",      # consuming a new result: tracker predict/update, static map observe
    "obstacles",    # pose() + extrapolating and inflating the tracks
    "plan",         # planner.plan() — for MAPPO this is the policy and both rollouts
    "command",      # handing the velocity to the locomotion transport
    "record",       # overlay drawing and the MP4 encode, on the CONTROL thread
)


class TickProfiler:
    """Wall clock of each stage of one control tick, for the record.

    WHAT THE RECORDED TELEMETRY ALREADY PROVES, and why this is still worth having. Across
    the committed runs, every tick that wrote a video frame took 173-299 ms and every tick
    that did not took 100.3-104.0 ms — the configured period, to the jitter. The two dry
    runs, which pass no ``--record`` at all, hold 9.86 Hz for 195 and 145 ticks while
    carrying the HIGHEST ``detect_ms`` in the corpus (262 and 269 ms median). So detection
    does not enter the control tick, and the deficit is on the ticks that record.

    ``_record``'s gate and the tracker-update gate are the same predicate
    (``result.seq > <last>``), and both #112 and #116 concluded from that that a recorded
    file cannot separate the mp4v encode from the tracker update. WITHIN one ``--record``
    run that is true. Across the corpus it is not, because the dry runs still consume
    results and still run the tracker — the recorder is the only thing they take out.
    Every tick of every committed run, split on the loop's own predicate:

    ======================================================  =====  ==========
    tick population, all 23 distinct committed runs             n      median
    ======================================================  =====  ==========
    ``--record``, consumed a new result (tracker + encode)   1023    246.4 ms
    ``--record``, reused the result                           453    101.4 ms
    NO ``--record``, consumed a new result (tracker only)     126    100.6 ms
    NO ``--record``, reused the result                        212    100.9 ms
    ======================================================  =====  ==========

    The tracker-and-static-map path costs the same as no path at all once the recorder is
    absent, and the two dry runs are interleaved in time with the recorded ones on the same
    2026-08-17 session (wall clocks 582662, 583045 ... 584060, **584591**, 584701) rather
    than being a recalled baseline. So ``_record`` is **at least 145 ms** of the control
    tick — a lower bound, because the two 100 ms buckets sit on the sleep floor and only
    bound their own bodies from above. That is issue #18's item 2, answered from data that
    was already committed, and ``test_telemetry`` recomputes the whole table on every run.

    What is left for this class is the part a recording cannot do: which HALF of
    ``_record`` — the overlay draw or the mp4v encode. ``tracker`` is timed under its own
    name so that the next live run confirms or breaks the table above rather than assuming
    it; #116 was right that the two want separate names even though the corpus settles it.

    Usage, once per tick::

        profiler.begin()
        with profiler.stage("plan"):
            command = planner.plan(...)
        ...
        telemetry.write_tick(..., profile=profiler.snapshot())
        profiler.wrote(ms)

    Args:
        clock: monotonic source, injected for tests.
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._stages: dict = {}
        self._write_ms = 0.0

    def begin(self, at: float | None = None) -> None:
        """Start a tick. ``at`` is the loop's own ``tick_start``, so ``tick_ms`` is
        measured against the same instant the trailing sleep is."""
        self._started = self._clock() if at is None else at
        self._stages = {}

    @contextlib.contextmanager
    def stage(self, name: str):
        """Time one stage. Re-entering a name within a tick ADDS, so a stage the loop runs
        on two branches is not silently reduced to whichever ran last."""
        if name not in TICK_STAGES:
            raise KeyError(f"unknown tick stage {name!r}; add it to TICK_STAGES")
        started = self._clock()
        try:
            yield
        finally:
            elapsed = (self._clock() - started) * 1000.0
            self._stages[name] = self._stages.get(name, 0.0) + elapsed

    def wrote(self, milliseconds: float) -> None:
        """How long the last telemetry write took, for the NEXT tick to carry."""
        self._write_ms = float(milliseconds)

    def snapshot(self) -> dict:
        """The tick's profile, as it goes into the record.

        ``write_prev_ms`` is the PREVIOUS tick's telemetry write, and the name says so
        because a record cannot contain the time it took to write itself. It is here at
        all because the write is a ``json.dumps`` and a deliberate ``flush`` at 10 Hz —
        the module docstring argues for that syscall, and nothing had ever priced it.

        ``other_ms`` is the remainder: everything in the tick body that no stage covers.
        A large one usually means the stage list has a hole in it, which is the failure
        mode of every hand-placed profiler. One large one is expected and correct — the
        tick that stands the robot up blocks for about three seconds on the vendor
        RecoveryStand and BalanceStand, which is a posture change and not a stage of a
        control tick.
        """
        elapsed = (self._clock() - self._started) * 1000.0
        stages = {name: round(self._stages.get(name, 0.0), 3) for name in TICK_STAGES}
        return {"tick_ms": round(elapsed, 3),
                "other_ms": round(elapsed - sum(self._stages.values()), 3),
                "write_prev_ms": round(self._write_ms, 3),
                "stages": stages}


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
                   goal_crop: float | None = None, profile: dict | None = None,
                   cycle_ms: float | None = None, wait_ms: float | None = None,
                   pass_ms: dict | None = None) -> None:
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
            # `person_shaped` is the ROUTING decision and is deliberately separate from
            # `label`: the label is what VOC guessed, which on this peer was `person` on
            # 12 of 12 live frames, and `mappo_bridge` must not route on it.
            "obstacles": [{"label": o.label, "kind": o.kind, "id": o.object_id,
                           "person_shaped": bool(o.person_shaped),
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
            # is which estimator produced it, because two of the values are CONSTANTS
            # rather than measurements and a consumer must be able to tell those apart:
            # "frame-fill" is a fixed near-range and "width-capped" is the fit-range cap,
            # which deadlocked a live run against a number that could not move. Both
            # appear in the committed 2026-08-25 runs: of 87 and 36 sightings,
            # "frame-fill" 6 and 6, "width-capped" 4 and 2.
            #
            # ⚠️ THE FULL SET IS EIGHT AND GROWING, AND NO DOCSTRING HAS EVER LISTED IT.
            # `person_detector.estimate_range` says "height", "width" or "frame-fill" and
            # its own code returns "width-capped" too; `GroundRanger` added "ground",
            # "ground-clipped", "ground-horizon" and "ground-far". Only five can reach this
            # field — `range_detections` drops every non-finite range, which is the three
            # `ground-*` refusals — leaving height, width, width-capped, frame-fill and
            # ground. `test_telemetry.test_every_ranging_source_is_named_here` walks
            # person_detector.py and fails if a ninth appears without this comment moving,
            # because a rule enforced by a comment is worth nothing.
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
                           # The WHOLE perception cycle, and the part of it spent blocked
                           # waiting for the camera. `detect_ms` covers the goal pass, the
                           # tiered detect and colour segmentation and nothing else, so on
                           # its own it cannot say whether that thread is compute-bound or
                           # camera-bound — and the 2026-08-25 runs show a third of the
                           # cycle outside it (317 ms per cycle against 202 ms of detect).
                           "cycle_ms": (None if cycle_ms is None
                                        else round(float(cycle_ms), 2)),
                           "wait_ms": (None if wait_ms is None
                                       else round(float(wait_ms), 2)),
                           # `detect_ms` SPLIT INTO THE THREE PASSES IT SPANS — goal,
                           # detect, colour — because one number for three passes cannot
                           # say which of them owns the 202 ms, and that is what decides
                           # whether the 300x300 SSD input is the lever or the goal pass's
                           # cadence is. Sums to `detect_ms` above, so a consumer of that
                           # field is not handed a different number under the same name.
                           # Null on a producer that does not measure the split.
                           "pass_ms": (pass_ms or None),
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
            # Where this tick's wall clock went. Null on a file written by anything that
            # does not profile itself; see TICK_STAGES for what each name covers.
            "profile": profile,
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


# ── Reading one back ────────────────────────────────────────────────────────
def read_run(path: str | Path) -> tuple[dict, list]:
    """``(header, ticks)`` from a run file, tolerating a truncated last line.

    The writer flushes every record, but a run that ends on a power cut can still lose
    its final one — and the normal way one of these ends is Ctrl-C or a safety abort, so
    refusing a whole file over its last byte would throw away the run being explained.
    """
    header: dict = {}
    ticks: list = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "header":
                header = record
            elif record.get("type") == "tick":
                ticks.append(record)
    return header, ticks


def _percentile(values: list, fraction: float) -> float:
    """Nearest-rank percentile. Not interpolated: these are latencies with a handful of
    samples, and an interpolated p90 of eight ticks is a number nobody measured."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summarise(header: dict, ticks: list) -> dict:
    """Every number the report prints, so a caller can assert on them instead of on prose."""
    times = [tick["t"] for tick in ticks]
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    intervals = [b - a for a, b in zip(times, times[1:])]
    perception = [tick.get("perception", {}) for tick in ticks]

    def field(name):
        return [p[name] for p in perception if p.get(name) is not None]

    recorded = [intervals[i] for i in range(len(intervals))
                if perception[i].get("video_frame") is not None]
    plain = [intervals[i] for i in range(len(intervals))
             if perception[i].get("video_frame") is None]

    # DID THIS TICK CONSUME A NEW PERCEPTION RESULT — that is, run the tracker and the
    # static map? Reconstructed from `seq` with the loop's own predicate. This is the split
    # the recorder gate coincides with inside a `--record` run and DIVERGES from in a run
    # with none: a dry run still consumes results and still updates the tracker, so its
    # new-result ticks are the price of that path with the encode taken out. That is what
    # attributes the deficit to the recorder rather than merely localising it.
    consumed, highest = [], None
    for entry in perception:
        seq = entry.get("seq")
        consumed.append(highest is None or (seq is not None and seq > highest))
        if seq is not None:
            highest = seq if highest is None else max(highest, seq)
    fresh = [intervals[i] for i in range(len(intervals)) if consumed[i]]
    reused = [intervals[i] for i in range(len(intervals)) if not consumed[i]]
    profiles = [tick["profile"] for tick in ticks if tick.get("profile")]
    stages = {name: statistics.median([p.get("stages", {}).get(name, 0.0)
                                       for p in profiles])
              for name in TICK_STAGES} if profiles else {}
    return {
        "ticks": len(ticks),
        "span_s": span,
        "measured_hz": (len(intervals) / span) if span else float("nan"),
        "configured_hz": header.get("control_hz"),
        "interval_ms": [x * 1000.0 for x in intervals],
        "stale": sum(1 for p in perception if p.get("stale")),
        "cycles": len({p["seq"] for p in perception if p.get("seq") is not None}),
        "detect_ms": field("detect_ms"),
        "cycle_ms": field("cycle_ms"),
        "wait_ms": field("wait_ms"),
        "frame_age_s": field("frame_age_s"),
        "recorded_ms": [x * 1000.0 for x in recorded],
        "plain_ms": [x * 1000.0 for x in plain],
        "new_result_ms": [x * 1000.0 for x in fresh],
        "reused_result_ms": [x * 1000.0 for x in reused],
        "recorder": (header.get("video") is not None
                     or header.get("video_raw") is not None),
        "profiles": profiles,
        "stages_ms": stages,
    }


def _line(label: str, values: list, out) -> None:
    if not values:
        return
    print(f"  {label:<16} median {statistics.median(values):7.1f}   "
          f"p90 {_percentile(values, 0.9):7.1f}   max {max(values):7.1f}", file=out)


def report(header: dict, ticks: list, out=sys.stdout) -> int:
    """Print where the loop's time went. Returns a process exit code."""
    if not ticks:
        print("no ticks in this file", file=out)
        return 1
    summary = summarise(header, ticks)
    configured = summary["configured_hz"]
    print(f"{summary['ticks']} ticks over {summary['span_s']:.2f} s — "
          f"{summary['measured_hz']:.2f} Hz measured"
          + (f", {configured:.1f} Hz configured" if configured else ""), file=out)
    _line("tick interval", summary["interval_ms"], out)
    _line("detect_ms", summary["detect_ms"], out)
    _line("perception cycle", summary["cycle_ms"], out)
    _line("  of it, waiting", summary["wait_ms"], out)
    ages = [x * 1000.0 for x in summary["frame_age_s"]]
    _line("frame age", ages, out)
    print(f"  {'perception':<16} {summary['cycles']} cycles, "
          f"{summary['cycles'] / summary['span_s'] if summary['span_s'] else 0:.2f} Hz; "
          f"{summary['stale']} of {summary['ticks']} ticks stale", file=out)
    print("  detect_ms times the PERCEPTION thread, not the control tick.", file=out)

    print("\nwhere the tick went", file=out)
    if summary["profiles"]:
        for name in TICK_STAGES:
            print(f"  {name:<16} {summary['stages_ms'][name]:7.1f} ms", file=out)
        for name in ("other_ms", "write_prev_ms", "tick_ms"):
            values = [p[name] for p in summary["profiles"] if name in p]
            label = {"other_ms": "(unaccounted)", "write_prev_ms": "telemetry write",
                     "tick_ms": "tick body"}[name]
            if values:
                print(f"  {label:<16} {statistics.median(values):7.1f} ms", file=out)
        return 0

    # No profile: what the two gates a recorded file DOES carry can be made to say. Both,
    # because they coincide in a `--record` run and diverge in a run with none, and that
    # divergence is the only thing in the corpus that separates the encode from the tracker.
    print("  this file carries no per-stage profile. What its gates still say:", file=out)
    buckets = (("wrote a frame", summary["recorded_ms"]),
               ("wrote none", summary["plain_ms"]),
               ("consumed a new result", summary["new_result_ms"]),
               ("reused the result", summary["reused_result_ms"]))
    for label, values in buckets:
        if values:
            print(f"    {len(values):3d} tick(s) that {label:<22} "
                  f"median {statistics.median(values):7.1f} ms", file=out)
    if summary["recorder"] and summary["new_result_ms"] and summary["reused_result_ms"]:
        delta = (statistics.median(summary["new_result_ms"])
                 - statistics.median(summary["reused_result_ms"]))
        print(f"    THIS run cannot subdivide that {delta:.1f} ms: the recorder gate and "
              f"the tracker gate are\n    the same predicate, so 'wrote a frame' and "
              f"'consumed a new result' are one indicator\n    for two candidates. The "
              f"CORPUS can, and does — the two runs with no --record still\n    consume "
              f"results and still update the tracker, and their new-result ticks measure\n"
              f"    100.6 ms. So this is the recorder. Run this over\n"
              f"    evidence/2026-08-17-corridor-and-room-runs/*.jsonl to see both halves.",
              file=out)
    elif not summary["recorder"] and summary["new_result_ms"]:
        print(f"    NO RECORDER on this run, which makes it the control: its new-result "
              f"ticks price the\n    tracker and the static map with the encode taken "
              f"out, at {statistics.median(summary['new_result_ms']):.1f} ms against a "
              f"{1000.0 / summary['configured_hz']:.0f} ms period.", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Where a recorded run's control loop spent its time.")
    parser.add_argument("run", type=Path, nargs="+", help="telemetry .jsonl")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing = [str(path) for path in args.run if not path.is_file()]
    if missing:
        raise SystemExit("not a file: " + ", ".join(missing))
    code = 0
    for path in args.run:
        if len(args.run) > 1:
            print(f"\n=== {path}")
        header, ticks = read_run(path)
        code |= report(header, ticks)
    return code


if __name__ == "__main__":
    sys.exit(main())
