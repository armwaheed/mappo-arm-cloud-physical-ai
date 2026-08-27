# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inventory the corpus pulled off the lab Go2, and measure what it can still prove.

Run against the pulled corpus directory (the one holding `history/`, `telemetry/` and
`tmp-evidence/`). It writes `inventory.tsv` and `failure-modes.json` beside itself and
prints the summary tables quoted in README.md.

    python3 inventory.py --corpus /path/to/robot-pull

Three things here are load-bearing and are why this is a script and not a spreadsheet.

**`wall_time` is two different clocks.** Tonight's runs stamp epoch seconds. Every older
run stamps the robot's monotonic uptime instead, and the rsync that pulled them did not
preserve mtimes, so the files all carry the pull time. A run on the uptime clock cannot
be dated at all, only ordered.

**Runs come from six different code generations**, told apart by which fields the writing
code emitted. This matters because the older schemas carry an `obstacles` array but never
wrote `sightings`: measuring detector recall over those runs scores the schema, not the
detector, and it is silently a 27-point error (see `_emits_sightings`).

**A range whose `source` is in UNRANGEABLE_SOURCES is a constant, not a measurement.**
`robot-stack/unitree/go2/visual_nav/person_detector.py` names four; the deployed tree
predates the GroundRanger that emits two of them, so only `width-capped` and `frame-fill`
appear in this corpus. They are counted and excluded, never averaged in.
"""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import math
import os

#: Above this, `wall_time` is epoch seconds; below it, robot uptime. 2020-01-01.
EPOCH_FLOOR = 1_577_836_800

#: Verbatim from person_detector.py (PR #134). A range carrying one of these is a
#: substituted constant and must not be trained or evaluated against.
UNRANGEABLE_SOURCES = ("frame-fill", "width-capped", "ground-clipped", "ground-horizon")

SUBDIRS = ("history", "telemetry", "tmp-evidence")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def generation(obstacle_keys, sighting_keys, has_profile):
    """Which code generation wrote this run, by the fields it emitted.

    Ordered oldest to newest by feature accretion. G3 is the tree preserved by PR #138:
    it emits `sightings` but neither `person_shaped` nor a tick `profile`, which is
    exactly what that manifest's 134 files produce.
    """
    if not obstacle_keys:
        return "G0"
    if "id" not in obstacle_keys:
        return "G1"
    if not sighting_keys:
        return "G2"
    if "person_shaped" not in obstacle_keys:
        return "G3"
    if not has_profile:
        return "G4"
    return "G5"


def parse(path):
    """Everything one run file yields, in a single pass."""
    run = {
        "file": path, "bytes": os.path.getsize(path), "sha256": sha256(path),
        "lines": 0, "ticks": 0, "bad_lines": 0, "header": None, "outcome": None,
        "sighting_rows": 0, "ticks_with_sightings": 0,
        "labels": collections.Counter(), "sources": collections.Counter(),
        "wall_first": None, "obstacle_keys": set(), "sighting_keys": set(),
        "has_profile": False, "person_track_ticks": 0, "person_dropped_ticks": 0,
    }
    with open(path, errors="replace") as handle:
        lines = handle.read().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        run["lines"] += 1
        try:
            rec = json.loads(line)
        except ValueError:
            run["bad_lines"] += 1
            continue
        when = rec.get("wall_time")
        if isinstance(when, (int, float)) and run["wall_first"] is None:
            run["wall_first"] = when
        kind = rec.get("type")
        if kind == "header":
            run["header"] = rec
        elif kind == "outcome":
            run["outcome"] = rec
        elif kind == "tick":
            run["ticks"] += 1
            if "profile" in rec:
                run["has_profile"] = True
            sightings = rec.get("sightings") or []
            obstacles = [o for o in (rec.get("obstacles") or []) if isinstance(o, dict)]
            for obstacle in obstacles:
                run["obstacle_keys"] |= set(obstacle.keys())
            if sightings:
                run["ticks_with_sightings"] += 1
            for sighting in sightings:
                run["sighting_keys"] |= set(sighting.keys())
                run["sighting_rows"] += 1
                run["labels"][sighting.get("label")] += 1
                run["sources"][sighting.get("source")] += 1
            # Track continuity: the tracker carries a person across ticks, so a tick
            # holding a person track with no person sighting is one the detector
            # dropped. Only meaningful where the schema can express a sighting.
            if any(o.get("label") == "person" for o in obstacles):
                run["person_track_ticks"] += 1
                if not any(s.get("label") == "person" for s in sightings):
                    run["person_dropped_ticks"] += 1
    run["clock"] = ("none" if run["wall_first"] is None else
                    "epoch" if run["wall_first"] > EPOCH_FLOOR else "uptime")
    run["gen"] = (generation(run["obstacle_keys"], run["sighting_keys"],
                             run["has_profile"]) if run["ticks"] else "-")
    return run


def _emits_sightings(run):
    """Whether this run's code generation could write a `sightings` array at all.

    G0-G2 cannot. Scoring recall over them counts every tick as a dropped detection.
    On this corpus 7 such runs carry a person track, contributing 424 person-track
    ticks that would ALL score as drops; including them moves the headline recall from
    474/678 = 69.9% to 474/1102 = 43.0% while measuring nothing about the detector.
    """
    return run["gen"] in ("G3", "G4", "G5")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True,
                    help="directory holding history/, telemetry/, tmp-evidence/")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    runs, seen = [], {}
    for sub in SUBDIRS:
        for path in sorted(glob.glob(os.path.join(args.corpus, sub, "*.jsonl"))):
            run = parse(path)
            run["dir"] = sub
            run["duplicate_of"] = seen.get(run["sha256"])
            if run["duplicate_of"] is None:
                seen[run["sha256"]] = os.path.relpath(path, args.corpus)
            runs.append(run)
    unique = [r for r in runs if r["duplicate_of"] is None]

    labels, sources = collections.Counter(), collections.Counter()
    track = dropped = 0
    skipped_runs = skipped_ticks = 0
    for run in unique:
        labels.update(run["labels"])
        sources.update(run["sources"])
        if _emits_sightings(run):
            track += run["person_track_ticks"]
            dropped += run["person_dropped_ticks"]
        elif run["person_track_ticks"]:
            skipped_runs += 1
            skipped_ticks += run["person_track_ticks"]

    cols = ["run", "dir", "gen", "clock", "utc", "uptime_s", "ticks", "sightings",
            "person", "bin", "live", "classes", "person_track_ticks",
            "person_dropped_ticks", "jsonl_bytes", "sha256_12"]
    rows = []
    for run in unique:
        header = run["header"] or {}
        when = run["wall_first"]
        rows.append({
            "run": os.path.splitext(os.path.basename(run["file"]))[0],
            "dir": run["dir"], "gen": run["gen"], "clock": run["clock"],
            "utc": (datetime.datetime.utcfromtimestamp(when).strftime("%Y-%m-%dT%H:%M:%SZ")
                    if run["clock"] == "epoch" else ""),
            "uptime_s": f"{when:.0f}" if run["clock"] == "uptime" else "",
            "ticks": run["ticks"], "sightings": run["sighting_rows"],
            "person": run["labels"].get("person", 0), "bin": run["labels"].get("bin", 0),
            "live": header.get("live"), "classes": len(header.get("classes") or []),
            "person_track_ticks": run["person_track_ticks"],
            "person_dropped_ticks": run["person_dropped_ticks"],
            "jsonl_bytes": run["bytes"], "sha256_12": run["sha256"][:12],
        })
    tsv = os.path.join(args.out, "inventory.tsv")
    with open(tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in sorted(rows, key=lambda r: (r["gen"], r["run"])):
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")

    report = {
        "files_seen": len(runs), "unique_runs": len(unique),
        "exact_duplicates": len(runs) - len(unique),
        "ticks": sum(r["ticks"] for r in unique),
        "sighting_rows": sum(r["sighting_rows"] for r in unique),
        "ticks_with_sightings": sum(r["ticks_with_sightings"] for r in unique),
        "labels": dict(labels), "range_sources": dict(sources),
        "unrangeable_rows": sum(v for k, v in sources.items()
                                if k in UNRANGEABLE_SOURCES),
        "by_generation": dict(collections.Counter(r["gen"] for r in unique)),
        "by_clock": dict(collections.Counter(r["clock"] for r in unique)),
        "person_track_ticks": track, "person_dropped_ticks": dropped,
        "recall_on_track": round(1 - dropped / track, 4) if track else None,
        "excluded_runs_schema_cannot_sight": skipped_runs,
        "excluded_person_track_ticks_schema_cannot_sight": skipped_ticks,
        "recall_if_those_were_counted": (
            round(1 - (dropped + skipped_ticks) / (track + skipped_ticks), 4)
            if track + skipped_ticks else None),
    }
    with open(os.path.join(args.out, "failure-modes.json"), "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)

    print(f"files {report['files_seen']}  unique {report['unique_runs']}  "
          f"exact duplicates {report['exact_duplicates']}")
    print(f"ticks {report['ticks']}  sighting rows {report['sighting_rows']}  "
          f"ticks with a sighting {report['ticks_with_sightings']}")
    print("by generation:", report["by_generation"])
    print("by clock:", report["by_clock"])
    print("labels:", report["labels"])
    print("range sources:", report["range_sources"],
          "-> unrangeable", report["unrangeable_rows"])
    print(f"person track ticks {report['person_track_ticks']}  "
          f"dropped {report['person_dropped_ticks']}  "
          f"recall {report['recall_on_track']}")
    print(f"excluded, schema cannot express a sighting: "
          f"{report['excluded_runs_schema_cannot_sight']} runs / "
          f"{report['excluded_person_track_ticks_schema_cannot_sight']} person-track "
          f"ticks  (counting them would read {report['recall_if_those_were_counted']})")
    assert math.isclose(sum(labels.values()), report["sighting_rows"])


if __name__ == "__main__":
    main()
