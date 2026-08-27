#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synthesise extra Lite3 frames from the real ones, and move the boxes with them.

    python3 make_synthetic.py --labels lite3_weak_lite3_20260827.json \
        --frames DIR --out-images DIR --out-labels lite3_synth_20260827.json

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT. ``~/ssdft/code/finetune_ssd.py``
already augments online, every epoch, with a different draw each time: photometric jitter,
expand, crop, horizontal flip, JPEG recompression, and — at the wave-5/6 settings —
``--motion-blur 0.5 --sensor-noise 0.5 --composite 0.3``. Generating those offline would
inflate the synthetic count without adding one bit of information, and would freeze into
files a draw the trainer is better off making fresh.

So this generates only the three families the trainer has NO operator for:

``shear``      affine tilt/skew. The trainer crops and flips but never shears, and shear is
               the only one of these that changes apparent VIEWPOINT at all.
``colour-slice`` vertical colour-filter bands. The trainer's photometric jitter is global;
               this varies illumination ACROSS the frame, which is what a room with a window
               at one end actually does.
``occlude``    rectangular cut-outs filled with local median. The trainer has no occlusion
               operator, and at avoidance range the peer is usually partly hidden.

⚠️ **NONE OF THIS MANUFACTURES A SECOND CAMERA POSE, and the shear family is the one most
likely to be mistaken for doing so.** ``evidence/2026-08-27-lite3-pov-clip-audit`` put it
plainly for the previous clip and it is equally true here: all six of these recordings are
tripod shots (measured: 0.0–1.0 px median displacement over every scene), so the set holds
at most six viewpoints and no augmentation adds a seventh. A shear is a 2-D warp of one
photograph, not a new photograph.

BOXES MOVE WITH THE PIXELS. ``shear`` maps the box's four corners through the same affine
matrix and takes their axis-aligned hull, then clips to the frame — the same "visible
extent, clipped to the frame" convention ``detector/labels/LABELLING.md`` sets.
``colour-slice`` and ``occlude`` leave geometry alone, so the box is copied unchanged. An
occlusion that
would hide more than ``MAX_OCCLUSION`` of the box is redrawn rather than kept, because a
box around a mostly-hidden object is a label that argues with itself.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

#: Shear, as a fraction of frame height/width. Beyond this the office furniture stops
#: looking like furniture and the frame stops being in-domain.
SHEAR_MAX = 0.18
#: Rotation paired with the shear, in degrees.
ROTATE_MAX = 6.0
#: Number of vertical bands a colour-slice frame is cut into.
SLICE_BANDS = (3, 4, 5)
#: Per-band multiplicative gain range, per channel.
SLICE_GAIN = (0.62, 1.38)
#: Occluder size, as a fraction of the shorter frame side.
OCCLUDE_FRAC = (0.08, 0.22)
#: Most of a box an occluder may cover before the sample is redrawn.
MAX_OCCLUSION = 0.45
#: Occluders per frame.
OCCLUDE_COUNT = (1, 3)


def _hull(box: list[int], matrix: np.ndarray, width: int, height: int) -> list[int] | None:
    """The axis-aligned hull of ``box``'s corners under ``matrix``, clipped to the frame."""
    x1, y1, x2, y2 = box
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    moved = cv2.transform(corners, matrix).reshape(-1, 2)
    nx1, ny1 = moved[:, 0].min(), moved[:, 1].min()
    nx2, ny2 = moved[:, 0].max(), moved[:, 1].max()
    nx1, ny1 = max(0, int(round(nx1))), max(0, int(round(ny1)))
    nx2, ny2 = min(width, int(round(nx2))), min(height, int(round(ny2)))
    if nx2 - nx1 < 12 or ny2 - ny1 < 12:
        return None
    return [nx1, ny1, nx2, ny2]


def shear(image: np.ndarray, box: list[int], rng: random.Random):
    """Affine tilt/skew. The only family here that changes apparent viewpoint at all."""
    height, width = image.shape[:2]
    shear_x = rng.uniform(-SHEAR_MAX, SHEAR_MAX)
    shear_y = rng.uniform(-SHEAR_MAX / 2, SHEAR_MAX / 2)
    angle = rng.uniform(-ROTATE_MAX, ROTATE_MAX)
    rot = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    matrix = np.float32([[1, shear_x, -shear_x * height / 2],
                         [shear_y, 1, -shear_y * width / 2]])
    combined = np.float32([
        [rot[0, 0] + rot[0, 1] * matrix[1, 0], rot[0, 0] * matrix[0, 1] + rot[0, 1],
         rot[0, 0] * matrix[0, 2] + rot[0, 1] * matrix[1, 2] + rot[0, 2]],
        [rot[1, 0] + rot[1, 1] * matrix[1, 0], rot[1, 0] * matrix[0, 1] + rot[1, 1],
         rot[1, 0] * matrix[0, 2] + rot[1, 1] * matrix[1, 2] + rot[1, 2]]])
    out = cv2.warpAffine(image, combined, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    new_box = _hull(box, combined, width, height)
    return (out, new_box) if new_box else (None, None)


def colour_slice(image: np.ndarray, box: list[int], rng: random.Random):
    """Vertical colour-filter bands — illumination that varies ACROSS the frame."""
    height, width = image.shape[:2]
    out = image.astype(np.float32)
    cuts = sorted(rng.sample(range(80, width - 80), rng.choice(SLICE_BANDS) - 1))
    edges = [0, *cuts, width]
    for left, right in zip(edges, edges[1:]):
        gain = np.float32([rng.uniform(*SLICE_GAIN) for _ in range(3)])
        out[:, left:right] *= gain
    return np.clip(out, 0, 255).astype(np.uint8), list(box)


def occlude(image: np.ndarray, box: list[int], rng: random.Random):
    """Rectangular cut-outs filled with the local median — a partly hidden peer."""
    height, width = image.shape[:2]
    out = image.copy()
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    covered = 0
    for _ in range(rng.randint(*OCCLUDE_COUNT)):
        size = int(min(height, width) * rng.uniform(*OCCLUDE_FRAC))
        ox = rng.randint(max(0, x1 - size // 2), min(width - size, max(1, x2 - size // 2)))
        oy = rng.randint(max(0, y1 - size // 2), min(height - size, max(1, y2 - size // 2)))
        patch = out[oy:oy + size, ox:ox + size]
        if patch.size == 0:
            continue
        overlap_x = max(0, min(x2, ox + size) - max(x1, ox))
        overlap_y = max(0, min(y2, oy + size) - max(y1, oy))
        covered += overlap_x * overlap_y
        out[oy:oy + size, ox:ox + size] = np.median(
            patch.reshape(-1, 3), axis=0).astype(np.uint8)
    if covered / area > MAX_OCCLUSION:
        return None, None
    return out, list(box)


FAMILIES = {"shear": shear, "colour-slice": colour_slice, "occlude": occlude}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out-images", type=Path, required=True)
    parser.add_argument("--out-labels", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=2,
                        help="synthetic frames per real frame per family")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = json.loads(args.labels.read_text())
    args.out_images.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    out_records = []
    tally = dict.fromkeys(FAMILIES, 0)
    redrawn = 0
    for record in data["records"]:
        image = cv2.imread(str(args.frames / record["image"]))
        if image is None:
            continue
        stem = record["image"].replace("/", "__").rsplit(".", 1)[0]
        for name, fn in FAMILIES.items():
            made = 0
            for _attempt in range(args.per_family * 4):
                if made >= args.per_family:
                    break
                new_image, new_box = fn(image, record["box"], rng)
                if new_image is None:
                    redrawn += 1
                    continue
                filename = f"synth_{name}_{stem}_{made}.jpg"
                cv2.imwrite(str(args.out_images / filename), new_image,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                out_records.append({
                    "image": filename, "label": data["label"], "box": new_box,
                    "derivation": f"synthetic:{name}", "parent": record["image"],
                    "parent_derivation": record["derivation"]})
                tally[name] += 1
                made += 1

    payload = {
        "label": data["label"],
        "source": {
            "what": f"{len(out_records)} synthetic frames warped from "
                    f"{data['count']} real ones. Images are NOT in this repository.",
            "families": sorted(FAMILIES),
            "not_generated": "photometric, expand, crop, flip, jpeg, motion-blur, "
                             "sensor-noise and composite — finetune_ssd.py does those "
                             "online, with a fresh draw every epoch",
            "viewpoints_added": 0,
            "seed": args.seed,
        },
        "count": len(out_records),
        "records": out_records,
    }
    args.out_labels.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out_labels}  ({len(out_records)} synthetic from "
          f"{data['count']} real; {redrawn} redrawn)")
    for name, count in sorted(tally.items()):
        print(f"  {name:14s}{count:>6d}")


if __name__ == "__main__":
    main()
