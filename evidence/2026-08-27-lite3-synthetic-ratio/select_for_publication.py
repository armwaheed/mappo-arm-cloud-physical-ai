#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Decide, by rule, which checkpoints are published — and record what is left behind.

    python3 select_for_publication.py --out published_checkpoints.json

⛔ **THE OMISSION IS THE POINT.** 743 checkpoints exist; publishing all of them is 17 GB
over a link measured at 0.4 MB/s, for weights nobody will cite. Publishing a hand-picked
few is worse than either extreme, because a reader cannot tell whether the ones missing were
uninteresting or inconvenient. So the selection is a RULE, applied to every arm equally, and
the output file names every checkpoint that was left on the training host and why.

THE RULE. A checkpoint is published if it is one a reader needs to check a published claim:

1. **it clears its profile's incumbent floor** at any of the three deployments — the gate is
   the decision, and every checkpoint that passes it must be inspectable, including the ones
   whose `lite3` column makes them useless;
2. **it is an arm's best `lite3`** at any profile — the argmax rows in the README;
3. **it is an arm's best person retention** at any profile — the other argmax;
4. **it is an arm's last epoch** — the matched-step endpoint row, which is the comparison
   the epoch counts were chosen to make.

Everything else is a point on a loss curve. It stays on the training host, it is listed in
the output, and `history.json` — which IS published, for every arm — carries its loss.

⚠️ **A NEGATIVE RESULT NEEDS ITS WEIGHTS.** Rule 1 deliberately publishes the arms that
FAILED, including every epoch of `r1x1_224` that cleared the floor on the seed that worked
and the seed that did not. The replication result is this wave's main finding and it is
unreproducible without both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = ("go2-peer-supervised", "go2-navigator-default", "go2-run-smoke")
#: Bytes per checkpoint. Every `.caffemodel` this trainer writes is the same size — the
#: graph is fixed at 22 classes and only the weights differ. Measured, not assumed.
CHECKPOINT_BYTES = 23_206_074


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    incumbent = {}
    for path in HERE.glob("incumbent_*.json"):
        data = json.loads(path.read_text())
        incumbent[data["preprocessing"]] = data["results"][0]["person"]["people_kept"]

    arms: dict = {}
    for path in sorted(HERE.glob("scored_*.json")):
        data = json.loads(path.read_text())
        profile = data["preprocessing"]
        arm = path.stem.replace("scored_", "").replace(f"_{profile}", "")
        arms.setdefault(arm, {})[profile] = data["results"]

    keep: dict = {}
    total: dict = {}

    def mark(arm: str, model: str, reason: str) -> None:
        keep.setdefault(arm, {}).setdefault(model, []).append(reason)

    for arm, byprofile in sorted(arms.items()):
        total[arm] = len(next(iter(byprofile.values())))
        for profile, rows in byprofile.items():
            floor = incumbent[profile]
            for row in rows:
                if row["person"]["people_kept"] >= floor:
                    mark(arm, row["model"],
                         f"clears the {profile} floor ({row['person']['people_kept']}/"
                         f"{floor} people, {row['lite3']['class_hit']}/36 lite3)")
            best = max(rows, key=lambda r: r["lite3"]["class_hit"])
            mark(arm, best["model"], f"best lite3 at {profile} "
                                     f"({best['lite3']['class_hit']}/36)")
            most = max(rows, key=lambda r: r["person"]["people_kept"])
            mark(arm, most["model"], f"most people at {profile} "
                                     f"({most['person']['people_kept']}/284)")
            mark(arm, rows[-1]["model"], f"matched-step endpoint at {profile}")

    published = sum(len(v) for v in keep.values())
    held = sum(total.values()) - published
    payload = {
        "rule": __doc__.split("THE RULE.")[1].split("Everything else")[0].strip(),
        "published_checkpoints": published,
        "held_on_training_host": held,
        "published_bytes": published * CHECKPOINT_BYTES,
        "held_bytes": held * CHECKPOINT_BYTES,
        "training_host_path": "~/lite3-ratio-20260827/runs/<arm>/epoch<NNN>.caffemodel",
        "also_published_for_every_arm": [
            "history.json — per-epoch conf/loc/distil loss and the full argv",
            "pseudo_labels_224.json / pseudo_labels_300.json — the teacher's old-class "
            "supervision, which differs by square and cannot be rebuilt from the manifests",
        ],
        "arms": {arm: {"epochs_trained": total[arm],
                       "published": len(keep.get(arm, {})),
                       "held_back": total[arm] - len(keep.get(arm, {})),
                       "checkpoints": dict(sorted(keep.get(arm, {}).items()))}
                 for arm in sorted(total)},
    }
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"  {'arm':16s}{'trained':>9}{'published':>11}{'held back':>11}")
    for arm in sorted(total):
        n = len(keep.get(arm, {}))
        print(f"  {arm:16s}{total[arm]:>9d}{n:>11d}{total[arm] - n:>11d}")
    print(f"\n  {published} checkpoints published "
          f"({published * CHECKPOINT_BYTES / 1e9:.2f} GB), {held} held on the training host "
          f"({held * CHECKPOINT_BYTES / 1e9:.2f} GB)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
