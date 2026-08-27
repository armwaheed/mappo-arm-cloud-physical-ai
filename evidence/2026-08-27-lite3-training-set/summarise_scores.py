#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn the scored checkpoints into the tables the README publishes.

    python3 summarise_scores.py

TWO SELECTION RULES, BOTH REPORTED, BECAUSE ONE OF THEM FLATTERS.

``best lite3``   the epoch with the most quadruped hits, and whatever person retention that
    epoch happens to have. This is the number a checkpoint sweep reaches for, and on its own
    it is close to meaningless: picking the argmax of one metric and then quoting the other
    at that point is selection, not measurement.
``best under the gate``   the epoch with the most quadruped hits **among epochs that keep at
    least as many people as the incumbent does at the same preprocessing**. This is the
    number that decides whether anything ships, because the project's standing gate is *lose
    zero of the people the shipped network sees*. Prior best checkpoints kept 3 and 5 of a
    base 17 across a day boundary and were refused for it.

⚠️ THE TWO COLUMNS ARE NOT THE SAME KIND OF NUMBER.

``lite3``  is **same-session**: a held-out time block of the same tripod shots the training
    frames come from — same room, same morning, same camera pose. It does not generalise and
    the README says so in its own section.
``people`` is **cross-day**: the 2026-08-20 Go2 manifest, a different day and a different
    building. It is the only column here that generalises, and it is the gate.

The incumbent row is the same `mnssd22` weights the fine-tunes start from, scored by this
same script at the same preprocessing, so the comparison is like-for-like rather than a
recalled figure.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Order the profiles are reported in: the one production launches, then the rest.
PROFILE_ORDER = ("go2-peer-supervised", "go2-navigator-default", "go2-run-smoke")
RUN_LABEL = {
    "a_ws_real": "a  real only",
    "b_ws_synth": "b  + synthetic",
    "c_ws_synth_aug": "c  + synthetic + wave-6 flags",
}


def load_scores() -> tuple[dict, dict]:
    """``(runs, incumbent)`` keyed by run name then profile, and profile."""
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


def main() -> None:
    runs, incumbent = load_scores()
    if not runs:
        print("no scored_*.json beside this script")
        return

    print("INCUMBENT — the mnssd22 weights every run starts from, scored here")
    print(f"  {'profile':24s}{'px/conf':>10s}{'people (cross-day)':>21s}")
    for profile in PROFILE_ORDER:
        row = incumbent.get(profile)
        if row:
            print(f"  {profile:24s}{row['input_size']:>6d}/{row['confidence']:<4}"
                  f"{row['person']['people_kept']:>15d}/{row['person']['frames']}")

    for rule in ("best lite3", "best under the gate"):
        print(f"\n{rule.upper()}"
              + ("  (argmax of one metric; the other is not chosen, only reported)"
                 if rule == "best lite3" else
                 "  (most lite3 among epochs keeping >= the incumbent's people)"))
        print(f"  {'run':32s}{'profile':24s}{'epoch':>9s}{'lite3':>9s}{'people':>11s}")
        for name in sorted(runs):
            for profile in PROFILE_ORDER:
                data = runs[name].get(profile)
                if not data:
                    continue
                floor = incumbent[profile]["person"]["people_kept"]
                rows = data["results"]
                if rule == "best under the gate":
                    rows = [r for r in rows
                            if r["person"]["people_kept"] >= floor]
                if not rows:
                    print(f"  {RUN_LABEL.get(name, name):32s}{profile:24s}"
                          f"{'-':>9s}{'-':>10s}   loses people at every epoch")
                    continue
                best = max(rows, key=lambda r: (r["lite3"]["class_hit"],
                                                r["person"]["people_kept"]))
                delta = best["person"]["people_kept"] - floor
                print(f"  {RUN_LABEL.get(name, name):32s}{profile:24s}"
                      f"{best['model'][5:8]:>9s}"
                      f"{best['lite3']['class_hit']:>6d}/{best['lite3']['frames']:<3d}"
                      f"{best['person']['people_kept']:>7d}/{floor:<3d}{delta:+d}")

    print("\n  lite3 is SAME-SESSION and does not generalise; people is CROSS-DAY and is the")
    print("  gate. A run with no row under the gate loses people at every epoch and is not")
    print("  a candidate, however good its lite3 column looks above it.")


if __name__ == "__main__":
    main()
