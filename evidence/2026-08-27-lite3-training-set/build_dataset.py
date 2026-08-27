#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn ``sam_labels.json`` into the manifests the Spark trainer and scorer read.

    python3 build_dataset.py --labels sam_labels.json --distinct distinct_views.json \\
        --out-dir . --frames-root DIR --stage-dir DIR

WHAT COMES OUT, AND WHAT EACH IS FOR.

``lite3_train_20260827.json``   the ``--labels`` file for ``finetune_ssd.py``. ONE class,
    ``lite3``, because the trainer grows the net from 21 classes to 22 -- one new class.
    ``person`` is not a new class and is not trained from these boxes: it is class 15 of the
    base net already, and it is PRESERVED by ``--pseudo-labels``/``--distil``, which is the
    mechanism every wave-5/6 run used.
``lite3_eval_20260827.json``    a held-out TIME BLOCK of the same clips. See the warning.
``person_20260827.json``        the person boxes, for measuring what this footage contains.
    Not a training input; the cross-day Go2 manifest is what person retention is scored on.
``negatives_20260827.json``     in-domain negatives -- the same room, same session, NO
    quadruped. The previous Lite3 set had ZERO of these and trained against Go2 corridor
    frames from another building. Frames containing PEOPLE are deliberately included:
    entering with ``box: null`` they are a negative for ``lite3`` and, through
    ``--pseudo-labels``, person supervision at the same time.

⚠️ **THE SPLIT IS TEMPORAL, AND IT IS STILL SAME-SESSION.** Frames are split by index
within each clip, not at random, because at 15 fps a random split puts frame *n* in train and
*n+1* in eval -- the same photograph twice. A temporal block at least separates the robot's
poses. It does NOT separate the day, the room, the camera pose or the lighting, because
there is only one of each. This project has measured 0/705 same-session against 60/159
cross-day for one model. Nothing scored on this split predicts tomorrow.

⚠️ **NEAR-DUPLICATES DO NOT RIDE ALONG, AND THAT IS A MEASURED CHOICE.** Of 5,854 frames only
456 are distinct views. The other 5,398 could each inherit their keyframe's box for free.
``neighbour_drift.json`` measures what that would cost: the frames around a sample of
keyframes were re-labelled with the same pipeline and IoU'd against their keyframe's box.
At **+/-1 frame the median IoU is 0.954**; by +4 it is 0.873 with a p10 of 0.496, i.e. one
box in ten has moved off its object. So ``--ride-along 1`` (offsets -1 and +1) is sound and
is what the shipped dataset uses -- it triples the positives for a measured 0.954 median
overlap. Anything wider is not: it would be pasting a stale box onto a moved subject.

Even at +/-1 this multiplies EXAMPLES, not views. The distinct-view count stays 456 and the
session stays one morning.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

#: Fraction of each clip's frame range that becomes the training block.
TRAIN_FRACTION = 0.70
#: In-domain negatives sampled per non-quadruped scene.
NEGATIVES_PER_SCENE = 175


#: Short, UNIQUE tag per scene. A truncation like ``scene[:5]`` collapses all three
#: ``light-*`` scenes onto one prefix and silently overwrites frames across them; that bug
#: staged 449 images into 420 files before this table replaced it.
SCENE_TAG = {
    "dim-box-in-different-distance-and-angle": "dimbox",
    "dim-lite3-in-difference-distance-and-angle": "dimlite3",
    "dim-people-walk-around": "dimppl",
    "light-box-in-different-distance-and-angle": "litbox",
    "light-lite3-in-difference-distance-and-angle": "litlite3",
    "light-people-walkaround": "litppl",
}


def flat(image: str) -> str:
    """``scene/f01234.jpg`` -> one UNIQUE filename, because --images is a flat directory."""
    scene, name = image.split("/")
    return f"{SCENE_TAG[scene]}_{name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--distinct", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--ride-along", type=int, default=0)
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text())
    distinct = json.loads(args.distinct.read_text())
    records = labels["records"]

    # --- temporal split, per clip
    cuts = {}
    for scene, block in distinct["scenes"].items():
        keys = block["keyframes"]
        cuts[scene] = keys[int(TRAIN_FRACTION * len(keys))] if keys else 0

    lite3 = [r for r in records if r["label"] == "lite3"]
    person = [r for r in records if r["label"] == "person"]
    train = [r for r in lite3 if r["frame"] < cuts[r["scene"]]]
    evalset = [r for r in lite3 if r["frame"] >= cuts[r["scene"]]]

    if args.ride_along:
        extra = []
        offsets = [o for n in range(1, args.ride_along + 1) for o in (-n, n)]
        for record in list(train):
            for offset in offsets:
                path = (args.frames_root / record["scene"]
                        / f"f{record['frame'] + offset:05d}.jpg")
                if path.exists():
                    extra.append({**record, "frame": record["frame"] + offset,
                                  "image": f"{record['scene']}/{path.name}",
                                  "derivation": record["derivation"] + "+ride-along"})
        train += extra

    # --- in-domain negatives: same room, same session, no quadruped
    negatives = []
    for scene, block in sorted(distinct["scenes"].items()):
        if "-lite3-" in scene:
            continue
        keys = block["keyframes"]
        step = max(1, len(keys) // NEGATIVES_PER_SCENE)
        negatives += [f"{scene}/f{k:05d}.jpg" for k in keys[::step]]

    def emit(name: str, rows: list, label: str, what: str) -> None:
        payload = {"label": label, "source": {
            "what": what,
            "supervision": "owlv2 phrase -> box, sam2 box -> mask, mask extent -> label; "
                           "class from the scene folder name and the phrase prompted",
            "geometry": "none — no range_m, focal_px, height_m or hfov_deg is read",
            "split": "TEMPORAL and SAME-SESSION; see the module docstring"},
            "count": len(rows), "records": [
                {"image": flat(r["image"]), "label": label, "box": r["box"],
                 "derivation": r["derivation"], "scene": r["scene"],
                 "frame": r["frame"], "owl_score": r["owl_score"]} for r in rows]}
        (args.out_dir / name).write_text(json.dumps(payload, indent=1))
        print(f"  {name:38s}{len(rows):>6d}")

    emit("lite3_train_20260827.json", train, "lite3", "training split, real frames")
    emit("lite3_eval_20260827.json", evalset, "lite3", "held-out TIME BLOCK, same session")
    emit("person_20260827.json", person, "person",
         "person boxes in this footage; not a training input")
    (args.out_dir / "negatives_20260827.json").write_text(json.dumps(
        {"what": "in-domain negatives: same room, same session, no quadruped present. "
                 "Frames with PEOPLE are included on purpose — see build_dataset.py.",
         "count": len(negatives), "files": ["neg_" + flat(f) for f in negatives]}, indent=1))
    print(f"  {'negatives_20260827.json':38s}{len(negatives):>6d}")

    if args.stage_dir:
        args.stage_dir.mkdir(parents=True, exist_ok=True)
        staged = 0
        # Only the quadruped positives and the negatives are training inputs. The person
        # boxes are a description of the footage, not a --labels file, so they are not
        # staged: their frames reach the trainer as negatives, where --pseudo-labels turns
        # the people in them into old-class supervision.
        # Positives are staged under exactly the name their manifest record carries, because
        # finetune_ssd.py resolves each record as `--images / record["image"]`. Negatives get
        # the `neg_` prefix that --negatives-glob matches, and the manifest stores that name.
        for image, prefix in ([(r["image"], "") for r in train + evalset]
                              + [(n, "neg_") for n in negatives]):
            source = args.frames_root / image
            if source.exists():
                shutil.copyfile(source, args.stage_dir / (prefix + flat(image)))
                staged += 1
        print(f"  staged {staged} images -> {args.stage_dir}")

    print(f"\n  distinct views {distinct['total_distinct']} of {distinct['total_frames']} "
          f"frames; lite3 {len(lite3)} (train {len(train)} / eval {len(evalset)}), "
          f"person {len(person)}, negatives {len(negatives)}")


if __name__ == "__main__":
    main()
