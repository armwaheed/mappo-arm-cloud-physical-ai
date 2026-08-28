#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cut the synthetic half of the 2026-08-27 Lite3 set down to a stated real:synthetic ratio.

    python3 subsample_synthetic.py --ratio 1 --out lite3_train_r1x1_20260827.json
    python3 subsample_synthetic.py --ratio 3 --out lite3_train_r1x3_20260827.json
    python3 subsample_synthetic.py --check          # every arm, counts only, writes nothing

⛔ **NOTHING IS REBUILT AND NO IMAGE IS TOUCHED.** The frames, the boxes and the three
synthetic families are exactly the ones
[`../2026-08-27-lite3-training-set`](../2026-08-27-lite3-training-set/README.md) shipped;
this only decides which of the 2,542 synthetic records go into a manifest. The staged
images on the training host are already a superset of every arm.

WHY A RATIO IS WORTH SWEEPING AT ALL. That wave's ablation is monotone in both directions
at every resolution: each step that adds augmentation adds `lite3` hits and removes people.
The real-only control is the only arm at either resolution that keeps people, and the one
thing that separates it from the arms that do not is that it holds 283 real positives
against 2,542 synthetic — **1 : 9.0**. Resolution is not the variable that does it, because
the people collapse at 300 as well.

## The three arms are NESTED, which is what makes this an ablation and not three datasets

``make_synthetic.py`` emits up to three variants (``_0``, ``_1``, ``_2``) of each of the
three families for each of the 283 real parents. Every parent has a ``_0`` in all three
families -- checked, not assumed, by :func:`build` -- so:

    1:9   all 2,542 synthetic records            = the previous wave's set, unchanged
    1:3   variant ``_0`` of all three families   = 849, one per family per parent
    1:1   variant ``_0`` of ONE family per       = 283, the family rotating by parent
          parent

and ``1:1 ⊂ 1:3 ⊂ 1:9``. Removing synthetic records is the only difference between the
arms: no parent is dropped, no family is dropped, and the real 283 are in all three.

**The family rotates rather than being chosen.** At 1:1 there is one synthetic slot per
parent and three families competing for it, so keeping "the kinds as they are" means
spreading them: parents in sorted order take shear, colour-slice, occlude, shear, ...
That lands 95 / 94 / 94. Picking one family instead would have confounded *how many* with
*which kind*, which is the one thing this sweep is not allowed to move.

⚠️ **NO RNG ANYWHERE.** The selection is a function of the sorted parent list, so this file
plus the two committed manifests reproduce every arm byte for byte on any machine. A seeded
draw would have been reproducible too and would still have made "which images" a thing a
reader has to run code to see; this way it is a sentence.

⚠️ **IT ADDS 0 VIEWPOINTS AND REMOVES 0.** Every arm is the same 456 distinct views from
the same 13 minutes of one morning. This changes how often the trainer is shown them, and
nothing else.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: The wave that built the set. Read, never written.
SOURCE_DIR = HERE.parent / "2026-08-27-lite3-training-set"
REAL_MANIFEST = SOURCE_DIR / "lite3_train_20260827.json"
AUG_MANIFEST = SOURCE_DIR / "lite3_train_aug_20260827.json"

#: ``synth_<family>_<parent stem>_<variant>.jpg``, the name ``make_synthetic.py`` writes.
SYNTH_NAME = re.compile(r"^synth_(?P<family>[a-z-]+)_(?P<parent>.+)_(?P<variant>\d+)\.jpg$")

#: The order families are handed out in at 1:1. Alphabetical, so it is not a preference.
FAMILIES = ("colour-slice", "occlude", "shear")

#: Ratios the wave runs. 9 is the previous wave's set in full; it is listed so that the
#: control arm comes out of the same code path as the arms it is a control for.
ARMS = (1, 3, 9)


def split(records: list) -> tuple[list, list]:
    """``(real, synthetic)``. Synthetic records are the ones ``make_synthetic.py`` named."""
    real = [r for r in records if not r["image"].startswith("synth_")]
    synthetic = [r for r in records if r["image"].startswith("synth_")]
    return real, synthetic


def parse(record: dict) -> dict:
    """``{family, parent, variant}`` for one synthetic record, checked against its own fields.

    ``make_synthetic.py`` writes the parent twice -- into the filename and into a ``parent``
    field -- and this asserts they agree rather than trusting the cheaper one. A filename
    that has drifted from its metadata is how a stratified subsample silently stops being
    stratified.
    """
    found = SYNTH_NAME.match(record["image"])
    if not found:
        raise ValueError(f"not a name make_synthetic.py writes: {record['image']!r}")
    if f"{found['parent']}.jpg" != record["parent"]:
        raise ValueError(f"{record['image']!r} names parent {found['parent']!r} but its "
                         f"parent field says {record['parent']!r}")
    if record["derivation"] != f"synthetic:{found['family']}":
        raise ValueError(f"{record['image']!r} is family {found['family']!r} by name and "
                         f"{record['derivation']!r} by derivation")
    return {"family": found["family"], "parent": record["parent"],
            "variant": int(found["variant"])}


def build(ratio: int, real: list, synthetic: list) -> tuple[list, dict]:
    """``(records, provenance)`` for one arm. See the module docstring for the three rules."""
    parsed = {r["image"]: parse(r) for r in synthetic}
    families = sorted({p["family"] for p in parsed.values()})
    if families != sorted(FAMILIES):
        raise ValueError(f"families in the manifest are {families}, this file expects "
                         f"{sorted(FAMILIES)}; the arms would not be nested")

    parents = sorted(r["image"] for r in real)
    first = {(p["parent"], p["family"]): image
             for image, p in parsed.items() if p["variant"] == 0}
    absent = [(parent, family) for parent in parents for family in FAMILIES
              if (parent, family) not in first]
    if absent:
        # Every parent has a variant 0 in every family in the shipped set. If that ever
        # stops being true the nesting argument above is void, so this refuses rather than
        # falling back to another variant and quietly changing what the arms mean.
        raise ValueError(f"{len(absent)} (parent, family) pairs have no variant 0, e.g. "
                         f"{absent[:3]}. 1:1 and 1:3 are defined as variant 0, so they "
                         f"would not be subsets of each other.")

    if ratio == 9:
        keep = {r["image"] for r in synthetic}
        rule = "every synthetic record, i.e. the previous wave's set unchanged"
    elif ratio == 3:
        keep = set(first.values())
        rule = "variant 0 of all three families, one per family per parent"
    elif ratio == 1:
        keep = {first[(parent, FAMILIES[index % len(FAMILIES)])]
                for index, parent in enumerate(parents)}
        rule = ("variant 0 of ONE family per parent, the family rotating through "
                + "/".join(FAMILIES) + " in sorted parent order")
    else:
        raise ValueError(f"ratio {ratio} is not one of {ARMS}")

    chosen = [r for r in synthetic if r["image"] in keep]
    counts: dict = {}
    for record in chosen:
        counts[parsed[record["image"]]["family"]] = 1 + counts.get(
            parsed[record["image"]]["family"], 0)
    return real + chosen, {
        "real": len(real), "synthetic": len(chosen),
        "ratio_requested": f"1 : {ratio}",
        "ratio_actual": f"1 : {len(chosen) / len(real):.2f}",
        "rule": rule, "per_family": dict(sorted(counts.items())),
        "parents_covered": len({parsed[r['image']]['parent'] for r in chosen}),
        "viewpoints_added_by_synthesis": 0,
        "distinct_views_behind_all_of_it": 456,
        "from": [str(REAL_MANIFEST.name), str(AUG_MANIFEST.name)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratio", type=int, choices=ARMS,
                        help="synthetic records per real record")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="print every arm's counts and the nesting check; write nothing")
    args = parser.parse_args()

    aug = json.loads(AUG_MANIFEST.read_text())
    real, synthetic = split(aug["records"])
    declared = json.loads(REAL_MANIFEST.read_text())
    if len(real) != declared["count"]:
        raise SystemExit(f"{AUG_MANIFEST.name} holds {len(real)} real records and "
                         f"{REAL_MANIFEST.name} declares {declared['count']}")

    if args.check:
        previous: set = set()
        for ratio in ARMS:
            records, provenance = build(ratio, real, synthetic)
            names = {r["image"] for r in records}
            nested = "subset of the next" if previous <= names else "NOT NESTED"
            previous = names
            print(f"  1:{ratio:<2d} {provenance['real']:>4d} real + "
                  f"{provenance['synthetic']:>5d} synthetic = {len(records):>5d}  "
                  f"({provenance['ratio_actual']})  {provenance['per_family']}  {nested}")
        return

    if args.ratio is None or args.out is None:
        raise SystemExit("--ratio and --out are both required without --check")
    records, provenance = build(args.ratio, real, synthetic)
    args.out.write_text(json.dumps(
        {"label": "lite3",
         "source": {**aug["source"], "what": f"{provenance['real']} real + "
                                             f"{provenance['synthetic']} synthetic lite3 "
                                             f"boxes", "subsample": provenance},
         "count": len(records), "records": records}, indent=1))
    print(f"  wrote {args.out}  {provenance['real']} real + {provenance['synthetic']} "
          f"synthetic = {len(records)} ({provenance['ratio_actual']})")


if __name__ == "__main__":
    main()
