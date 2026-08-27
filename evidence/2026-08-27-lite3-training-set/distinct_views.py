#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""How many of the 5,854 frames are actually different? Answer: about 6% of them.

    python3 distinct_views.py --frames-root DIR --out distinct_views.json

Every one of the six clips is a TRIPOD SHOT -- measured at 0.0-1.0 px median camera
displacement -- so consecutive frames differ only by whatever moved in the room. At 15 fps
with a walking subject that is very little, and a set of 5,854 frames is not 5,854 samples.

THE RULE, stated so it can be argued with. Sample every ``STEP``th frame, downscale to
160x90 greyscale, and keep a frame when it differs from the LAST KEPT frame -- not from its
immediate predecessor -- by more than ``THRESHOLD`` mean grey levels. Comparing against the
last kept frame is what makes it a running selection rather than a change detector: a
subject drifting slowly across the room accumulates difference until it trips the threshold,
where a frame-to-frame test would score every step as small and keep nothing.

⚠️ **THIS IS A CRUDE INSTRUMENT AND ITS NUMBER MOVES WITH ITS KNOBS.** 3.0 mean grey levels
on a 160x90 thumbnail is not a principled quantity; it is a working threshold. The
three DIM scenes sweep 28-83 mean luminance on their own, so a global brightness change
counts as a new view there whether or not anything moved -- which is why dim-lite3 scores
more distinct views than light-lite3 despite holding the same robot doing the same thing.
The count is reported with the knobs that produced it, and both are in the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

#: Sample every Nth frame before comparing.
STEP = 5
#: Mean grey-level difference from the LAST KEPT frame that makes a frame a new view.
THRESHOLD = 3.0
#: Thumbnail the comparison is made on.
THUMB = (160, 90)


def distinct(paths: list[Path]) -> list[int]:
    """Indices (into ``paths``) of the frames that are new views under the rule above."""
    kept: list[int] = []
    last: np.ndarray | None = None
    for index in range(0, len(paths), STEP):
        thumb = cv2.resize(cv2.imread(str(paths[index]), cv2.IMREAD_GRAYSCALE),
                           THUMB).astype(np.float32)
        if last is None or float(np.abs(thumb - last).mean()) > THRESHOLD:
            kept.append(index)
            last = thumb
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = {"method": {"step": STEP, "threshold_mean_grey": THRESHOLD,
                      "thumbnail": list(THUMB), "compared_against": "last kept frame"},
           "scenes": {}}
    total_frames = total_kept = 0
    for scene_dir in sorted(d for d in args.frames_root.iterdir() if d.is_dir()):
        paths = sorted(scene_dir.glob("*.jpg"))
        kept = distinct(paths)
        out["scenes"][scene_dir.name] = {
            "frames": len(paths), "distinct": len(kept),
            "keyframes": [int(p.stem[1:]) for p in (paths[i] for i in kept)]}
        total_frames += len(paths)
        total_kept += len(kept)
        print(f"  {scene_dir.name[:44]:46s}{len(paths):>7d}{len(kept):>10d}"
              f"{100 * len(kept) / len(paths):>8.1f}%")
    out["total_frames"] = total_frames
    out["total_distinct"] = total_kept
    args.out.write_text(json.dumps(out, indent=1))
    print(f"  {'TOTAL':46s}{total_frames:>7d}{total_kept:>10d}"
          f"{100 * total_kept / total_frames:>8.1f}%")


if __name__ == "__main__":
    main()
