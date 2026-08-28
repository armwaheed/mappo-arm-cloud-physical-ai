#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn the scored checkpoints into the tables the README publishes.

    python3 summarise_ratio.py

FOUR TABLES, AND THREE OF THEM EXIST TO STOP THE FOURTH FROM BEING READ ALONE.

``incumbent``        the shipped ``mnssd22`` weights, scored by the same script at the same
    preprocessing. It is the gate floor, and it is re-measured rather than recalled.
``best lite3``       the epoch with the most quadruped hits, and whatever person retention
    that epoch happens to have. This is what a checkpoint sweep reaches for and on its own
    it is selection, not measurement — the argmax of one metric with the other merely
    reported at that point.
``best under the gate``   the epoch with the most quadruped hits **among epochs keeping at
    least as many people as the incumbent at the same preprocessing**. This is the only
    table that can recommend anything, because the standing gate is *lose zero of the
    people the shipped network sees*.
``matched-step endpoint``  each arm's LAST epoch. The arms run 144/87/40 epochs precisely so
    that this row is ~5,200 gradient steps for all of them, on one learning-rate curve in
    step space. It is the row where the ratio is the only thing that differs, and it is
    reported whether or not it flatters.

⚠️ THE TWO COLUMNS ARE NOT THE SAME KIND OF NUMBER.

``lite3``   **same-session**. A held-out time block of the same six tripod shots the
    training frames come from — one room, one morning, 0.0–1.0 px of camera motion, 456
    distinct views. It does not generalise and nothing here claims it does.
``people``  **cross-day**. The 2026-08-20 Go2 manifest: another day, another building. It is
    the only column with a day boundary in it, and it is the gate.

TWO MORE TABLES, BECAUSE A SINGLE EPOCH IS A SINGLE EPOCH.

``epochs clearing the floor``  how many of an arm's own checkpoints keep at least the
    incumbent's people. Every table above reports ONE epoch, chosen by an argmax over 40 to
    144 candidates, and an argmax over 144 noisy numbers finds a high one whether or not the
    run is any good. A count cannot be won by one lucky epoch.
``median over all epochs``     the same idea for the size of the effect rather than its
    existence. Reported for both columns, and it is the number the ratio conclusion rests
    on — not the endpoint row, which is also one epoch.
``floor sensitivity``          the best lite3 still available as the person floor is raised
    one person at a time. A candidate that clears by +0 and collapses when the floor moves
    by one is a different kind of result from one that survives +3, and the two are
    indistinguishable in the gate table.

``replicates``  the same condition, re-run on seeds 1 and 2, grouped so the spread is
    visible. This is the table that decides how much of everything above is real: if the
    seed moves a number more than the conditions do, that number cannot rank the conditions,
    however carefully it was measured.

⚠️ RUN-TO-RUN NOISE ON THE PERSON COLUMN IS ±1–3 PEOPLE, measured on this project's own
duplicate r/s pair on the Go2 corpus. A ``+2`` is not a win and this script prints the band
under every table rather than leaving the reader to remember it.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
#: Order profiles are reported in: the one the peer launcher opens at, then the rest.
PROFILE_ORDER = ("go2-peer-supervised", "go2-navigator-default", "go2-run-smoke")
#: The band inside which a difference in the person column is not a result. Measured on the
#: duplicate r/s pair of the Go2 corpus, and quoted by every wave since.
PERSON_NOISE = 3
#: Arms, in the order the argument runs: ratio sweep at the deployed square, then the two
#: 300 px controls that separate the ratio from the square.
RUN_LABEL = {
    "r1x1_224": "1:1  @224  run 1",
    "r1x3_224": "1:3  @224",
    "r1x9_224": "1:9  @224  run 1",
    "r1x1_300": "1:1  @300  run 1",
    "r1x9_300": "1:9  @300  (wave 7's `b`)",
    #: Replicates. Not further conditions -- the SAME conditions re-run, which is what
    #: established that a single run of any of them ranks nothing.
    "r1x1_224_s1": "1:1  @224  run 2",
    "r1x1_224_s2": "1:1  @224  run 3",
    "r1x9_224_s1": "1:9  @224  run 2",
    "r1x9_224_s2": "1:9  @224  run 3",
    "r1x1_300_s1": "1:1  @300  run 2",
    "r1x1_300_s2": "1:1  @300  run 3",
}


def load() -> tuple:
    """``(runs, incumbent)``: scores keyed by run then profile, and the floor per profile."""
    runs: dict = {}
    incumbent: dict = {}
    for path in sorted(HERE.glob("scored_*.json")):
        data = json.loads(path.read_text())
        profile = data["preprocessing"]
        name = path.stem.replace("scored_", "").replace(f"_{profile}", "")
        runs.setdefault(name, {})[profile] = data
    for path in sorted(HERE.glob("incumbent_*.json")):
        data = json.loads(path.read_text())
        incumbent[data["preprocessing"]] = data["results"][0]
    return runs, incumbent


def row(name: str, profile: str, result: dict, floor: int) -> str:
    delta = result["person"]["people_kept"] - floor
    return (f"  {RUN_LABEL.get(name, name):26s}{profile:24s}"
            f"{result['model'][5:8]:>7s}"
            f"{result['lite3']['class_hit']:>6d}/{result['lite3']['frames']:<3d}"
            f"{result['person']['people_kept']:>7d}/{floor:<4d}{delta:+d}"
            f"{'' if abs(delta) > PERSON_NOISE else '  (inside the noise band)'}")


def main() -> None:
    runs, incumbent = load()
    if not runs:
        print("no scored_*.json beside this script; run score_ratio_wave.sh first")
        return

    print("INCUMBENT — the mnssd22 weights every arm starts from, re-scored here")
    print(f"  {'profile':24s}{'px/conf':>10s}{'people (cross-day)':>21s}")
    for profile in PROFILE_ORDER:
        result = incumbent.get(profile)
        if result:
            print(f"  {profile:24s}{result['input_size']:>6d}/{result['confidence']:<4}"
                  f"{result['person']['people_kept']:>15d}/{result['person']['frames']}")

    header = f"  {'arm':26s}{'profile':24s}{'ep':>7s}{'lite3':>10s}{'people':>12s}"
    for rule in ("best lite3", "best under the gate", "matched-step endpoint"):
        note = {"best lite3": "(argmax of one metric; the other is only reported)",
                "best under the gate":
                    "(most lite3 among epochs keeping >= the incumbent's people)",
                "matched-step endpoint":
                    "(each arm's LAST epoch: ~5,200 steps, one LR curve, ratio the only "
                    "difference)"}[rule]
        print(f"\n{rule.upper()}  {note}")
        print(header)
        for name in RUN_LABEL:
            for profile in PROFILE_ORDER:
                data = runs.get(name, {}).get(profile)
                if not data:
                    continue
                floor = incumbent[profile]["person"]["people_kept"]
                results = data["results"]
                if rule == "matched-step endpoint":
                    print(row(name, profile, results[-1], floor))
                    continue
                if rule == "best under the gate":
                    results = [r for r in results
                               if r["person"]["people_kept"] >= floor]
                if not results:
                    print(f"  {RUN_LABEL[name]:26s}{profile:24s}{'-':>7s}{'-':>10s}"
                          f"      loses people at every epoch")
                    continue
                print(row(name, profile, max(results, key=lambda r: (
                    r["lite3"]["class_hit"], r["person"]["people_kept"])), floor))

    print("\nMOST PEOPLE ANY EPOCH OF EACH ARM KEEPS  (the ratio's effect on the gate "
          "column, with lite3 not selected on at all)")
    print(header)
    for name in RUN_LABEL:
        for profile in PROFILE_ORDER:
            data = runs.get(name, {}).get(profile)
            if data:
                floor = incumbent[profile]["person"]["people_kept"]
                print(row(name, profile,
                          max(data["results"], key=lambda r: r["person"]["people_kept"]),
                          floor))

    print("\nEPOCHS CLEARING THE INCUMBENT'S PEOPLE, out of the arm's own epochs  (a count "
          "cannot be won by one lucky epoch)")
    print(f"  {'arm':26s}" + "".join(f"{p:>24s}" for p in PROFILE_ORDER))
    for name in RUN_LABEL:
        line = f"  {RUN_LABEL[name]:26s}"
        for profile in PROFILE_ORDER:
            data = runs.get(name, {}).get(profile)
            if not data:
                line += f"{'-':>24s}"
                continue
            floor = incumbent[profile]["person"]["people_kept"]
            rows = data["results"]
            passing = sum(1 for r in rows if r["person"]["people_kept"] >= floor)
            line += f"{passing:>13d}/{len(rows):<4d}{100 * passing / len(rows):>5.1f}%"
        print(line)

    print("\nMEDIAN OVER ALL EPOCHS  (people, then lite3 — the ratio conclusion rests on "
          "this, not on any single epoch)")
    print(f"  {'arm':26s}" + "".join(f"{p:>24s}" for p in PROFILE_ORDER))
    for name in RUN_LABEL:
        line = f"  {RUN_LABEL[name]:26s}"
        for profile in PROFILE_ORDER:
            data = runs.get(name, {}).get(profile)
            if not data:
                line += f"{'-':>24s}"
                continue
            floor = incumbent[profile]["person"]["people_kept"]
            rows = data["results"]
            people = median(r["person"]["people_kept"] for r in rows)
            lite3 = median(r["lite3"]["class_hit"] for r in rows)
            line += f"{people:>10.1f}/{floor:<4d}{lite3:>6.1f}/36"
        print(line)

    print("\nFLOOR SENSITIVITY  (best lite3 still available as the person floor rises; a "
          "candidate that dies at +1 is not the same result as one that survives +3)")
    print(f"  {'arm':26s}{'profile':24s}"
          + "".join(f"{'+' + str(d):>10s}" for d in range(0, 4)))
    for name in RUN_LABEL:
        for profile in PROFILE_ORDER:
            data = runs.get(name, {}).get(profile)
            if not data:
                continue
            floor = incumbent[profile]["person"]["people_kept"]
            line = f"  {RUN_LABEL[name]:26s}{profile:24s}"
            for delta in range(0, 4):
                rows = [r for r in data["results"]
                        if r["person"]["people_kept"] >= floor + delta]
                best = max((r["lite3"]["class_hit"] for r in rows), default=None)
                line += f"{'none' if best is None else str(best) + '/36':>10s}"
            print(line)

    conditions: dict = {}
    for name in runs:
        conditions.setdefault(name.split("_s")[0], []).append(name)
    replicated = {c: sorted(v) for c, v in conditions.items() if len(v) > 1}
    if replicated:
        print("\nREPLICATES — the same condition, three runs  (`--seed` differs but does "
              "not decide the outcome: see probe_determinism.sh. If re-running moves a "
              "number more than the conditions do, that number cannot rank them)")
        print(f"  {'condition':14s}{'profile':24s}{'run':>6}"
              f"{'median people':>16}{'clears':>12}{'best lite3 u/gate':>20}")
        for condition, names in sorted(replicated.items()):
            for profile in PROFILE_ORDER:
                floor = incumbent[profile]["person"]["people_kept"]
                for run, name in enumerate(names, start=1):
                    rows = runs[name].get(profile)
                    if not rows:
                        continue
                    rows = rows["results"]
                    passing = [r for r in rows if r["person"]["people_kept"] >= floor]
                    best = (max(passing, key=lambda r: r["lite3"]["class_hit"])
                            if passing else None)
                    people = median(r["person"]["people_kept"] for r in rows)
                    print(f"  {condition if run == 1 else '':14s}"
                          f"{profile if run == 1 else '':24s}{run:>6d}"
                          f"{people:>12.1f}/{floor:<3d}"
                          f"{len(passing):>8d}/{len(rows):<3d}"
                          + (f"{best['lite3']['class_hit']:>15d}/36" if best
                             else f"{'none':>18s}"))

    print("\n  lite3 is SAME-SESSION and does not generalise; people is CROSS-DAY and is")
    print(f"  the gate. Person differences within +/-{PERSON_NOISE} are inside this "
          f"project's")
    print("  own run-to-run noise and are not results.")


if __name__ == "__main__":
    main()
