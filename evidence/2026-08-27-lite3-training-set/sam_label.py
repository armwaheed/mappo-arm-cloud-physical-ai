#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Label the distinct views with an open-vocabulary detector and SAM 2, in IMAGE mode.

    python3 sam_label.py --frames-root DIR --distinct distinct_views.json \\
        --queries scene_queries.json --out sam_labels.json

WHY IMAGE MODE AND NOT SAM 2 VIDEO PROPAGATION. Because there is almost no video here.
Every clip is a tripod shot -- 0.0-1.0 px median camera displacement, measured -- so of
5,854 frames only **456 are distinct views** under ``distinct_views.py``'s rule, about 8%.
Propagating a mask through 5,398 near-duplicates buys nothing and costs the one failure
mode that has no internal signal: a propagated mask drifting onto the wrong object. Image
mode has no propagation state and therefore cannot drift.

THE TWO MODELS, AND WHAT EACH IS AND IS NOT ALLOWED TO DECIDE.

``google/owlv2-base-patch16-ensemble`` turns a PHRASE into boxes. It is used only to LOCATE
    -- to answer "where in this frame is the thing I asked for". Apache 2.0.
``facebook/sam2.1-hiera-large`` turns a box into a mask, and the mask's extent is the label.
    A mask box is TIGHTER than a detector's regressed rectangle, because it is the object's
    silhouette rather than a guess at one. Apache 2.0.

⛔ **NEITHER MODEL NAMES A CLASS.** SAM is class-agnostic by construction, and OWLv2's
"class" is just the phrase it was handed. Every label comes from ``scene_queries.json`` --
the scene folder name plus the object prompted -- and that file is committed next to the
labels so a wrong class is a wrong line someone can read.

⛔ NO GEOMETRY. Nothing here reads ``range_m``, ``focal_px``, ``height_m`` or ``hfov_deg``;
the camera block in these recordings is wrong three ways and ``robot-stack/CAMERA-GEOMETRY.md``
has the record. Boxes are pixels of the 1280x720 frame.

⚠️ **OWLv2 WILL CONFIDENTLY BOX A CHAIR, A SHADOW, OR THE OPERATOR'S LEG**, and SAM will
segment whatever it is handed without complaint. Nothing in this file can detect that. It is
why ``handcheck.json`` records a stratified sample per class -- negatives included -- and why
the README quotes the hand-checked rate and not the produced count.

⚠️ **456 VIEWS FROM 13 MINUTES OF ONE MORNING IN ONE ROOM IS THE CEILING.** Better labels do
not add a day, a room or a viewpoint. Every number carries the split that produced it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

#: Floor for the OWLv2 sweep. Each query then applies its OWN `threshold` from
#: scene_queries.json, because the two classes are not on the same scale: `lite3` scores a
#: median 0.517 and `person` a median 0.228, and one global number either floods `person`
#: with false boxes or throws away 95% of `lite3`. Both per-class thresholds were set from
#: a hand-check, not from the distribution -- see the README.
OWL_THRESHOLD = 0.10
#: Smallest mask, in pixels, that counts as a real object.
MIN_MASK_PX = 150
#: Largest fraction of the frame a mask may fill before it is the room, not an object in it.
MAX_AREA_FRAC = 0.60
#: Boxes overlapping this much are the same object seen twice by two phrases.
NMS_IOU = 0.55


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_to_box(mask: np.ndarray) -> list[int] | None:
    """Axis-aligned extent of a boolean mask -- tighter than the box that seeded it."""
    if mask.sum() < MIN_MASK_PX:
        return None
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def locate(processor, model, image: Image.Image, queries: list[dict], device: str):
    """Every box OWLv2 returns for these phrases, above threshold, best first."""
    texts = [[q["text"] for q in queries]]
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([[image.height, image.width]], device=device)
    result = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target, threshold=OWL_THRESHOLD)[0]
    found = []
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        query = queries[int(label)]
        if float(score) < query.get("threshold", OWL_THRESHOLD):
            continue
        found.append({"score": float(score), "query": query,
                      "box": [float(v) for v in box]})
    found.sort(key=lambda f: -f["score"])
    kept: list[dict] = []
    per_label: dict[str, int] = {}
    for candidate in found:
        label = candidate["query"]["label"]
        if per_label.get(label, 0) >= candidate["query"]["max_instances"]:
            continue
        if any(iou(candidate["box"], k["box"]) > NMS_IOU for k in kept):
            continue
        kept.append(candidate)
        per_label[label] = per_label.get(label, 0) + 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--distinct", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    spec = json.loads(args.queries.read_text())
    distinct = json.loads(args.distinct.read_text())
    processor = Owlv2Processor.from_pretrained(spec["detector"])
    owl = Owlv2ForObjectDetection.from_pretrained(spec["detector"]).to(args.device).eval()
    sam = SAM2ImagePredictor.from_pretrained(spec["segmenter"], device=args.device)

    records = []
    stats: dict[str, dict] = {}
    for scene, block in sorted(distinct["scenes"].items()):
        queries = spec["scenes"][scene]["queries"]
        counts: dict[str, int] = {}
        located = refined = 0
        for frame in block["keyframes"]:
            path = args.frames_root / scene / f"f{frame:05d}.jpg"
            image = Image.open(path).convert("RGB")
            found = locate(processor, owl, image, queries, args.device)
            located += len(found)
            if not found:
                continue
            array = np.array(image)
            sam.set_image(array)
            boxes = np.array([f["box"] for f in found], dtype=np.float32)
            masks, scores, _ = sam.predict(box=boxes, multimask_output=False)
            masks = np.asarray(masks)
            if masks.ndim == 3:
                masks = masks[:, None]
            for i, candidate in enumerate(found):
                mask = masks[i, 0] > 0.0
                box = mask_to_box(mask)
                if box is None:
                    continue
                area = (box[2] - box[0]) * (box[3] - box[1]) / (image.width * image.height)
                if area > MAX_AREA_FRAC:
                    continue
                label = candidate["query"]["label"]
                records.append({
                    "image": f"{scene}/f{frame:05d}.jpg", "label": label, "box": box,
                    "scene": scene, "frame": frame,
                    "derivation": f"owlv2+sam2:{candidate['query']['text']}",
                    "owl_score": round(candidate["score"], 4),
                    "owl_box": [round(v, 1) for v in candidate["box"]],
                    "sam_score": round(float(np.ravel(scores)[i]), 4),
                    "area_frac": round(area, 4),
                    "aspect": round((box[3] - box[1]) / max(1, box[2] - box[0]), 3)})
                counts[label] = counts.get(label, 0) + 1
                refined += 1
        stats[scene] = {"keyframes": len(block["keyframes"]), "owl_boxes": located,
                        "labels": refined, "by_class": counts}
        print(f"  {scene[:44]:46s} keyframes {len(block['keyframes']):>4d}  "
              f"owl {located:>4d}  labels {refined:>4d}  {counts}", flush=True)

    args.out.write_text(json.dumps(
        {"detector": spec["detector"], "segmenter": spec["segmenter"],
         "owl_threshold": OWL_THRESHOLD, "distinct_method": distinct["method"],
         "stats": stats, "count": len(records), "records": records}, indent=1))
    print(f"wrote {args.out} ({len(records)} labels)")


if __name__ == "__main__":
    main()
