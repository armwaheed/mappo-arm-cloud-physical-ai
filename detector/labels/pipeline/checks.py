"""Diagnostics that justify the propagation method, printed per segment.

  drift   - global camera motion within a segment (phase correlation vs frame 0)
  change  - how much the frame content changes at all (mean abs difference vs frame 0)
  roi     - change confined to the labelled box, i.e. whether the peer itself moved

These are what say a single box may legitimately be reused across a whole static segment.
"""
import glob
import sys

import cv2
import numpy as np

SCRATCH = (
    "/private/tmp/claude-501/-Users-wahbro01-workspaces-git/"
    "ae5beebd-3312-48c6-92c7-3538b392af3f/scratchpad/"
)
SRC = SCRATCH + "peercap/"

TAGS = [
    "p1_close_broadside", "p2_close_headon_stand", "p3_close_rearon_stand",
    "p4_mid_sweep_stand", "p5_1_far_left_stand", "p5_23_far_centre_then_right_stand",
    "p6_1_trunc_left_stand", "p6_2_trunc_right_stand", "p6_3_trunc_half_left_stand",
    "peer01", "smoke",
]
BOXES = {
    "p1_close_broadside": (232, 0, 1783, 1042),
    "p2_close_headon_stand": (655, 268, 1665, 1080),
    "p3_close_rearon_stand": (595, 308, 1790, 1080),
    "p5_1_far_left_stand": (618, 527, 792, 682),
    "p5_23_far_centre_then_right_stand": (898, 548, 1078, 706),
    "p6_1_trunc_left_stand": (0, 311, 249, 1080),
    "p6_2_trunc_right_stand": (1577, 495, 1920, 1080),
    "p6_3_trunc_half_left_stand": (1046, 297, 1920, 1080),
    "peer01": (852, 403, 1332, 778),
}


def files_for(tag: str) -> list[str]:
    return sorted(glob.glob(SRC + tag + "_[0-9][0-9][0-9][0-9].jpg"))


def windowed(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return g * cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)


def drift() -> None:
    for tag in TAGS:
        f = files_for(tag)
        ref = windowed(f[0])
        worst = max((cv2.phaseCorrelate(ref, windowed(f[i]))[0]
                     for i in (1, 5, 20, len(f) // 2, len(f) - 1)),
                    key=lambda d: abs(d[0]) + abs(d[1]))
        print(f"drift  {tag:38s} n={len(f):4d} worst=({worst[0]:+.1f},{worst[1]:+.1f}) px")


def change() -> None:
    for tag in TAGS:
        f = files_for(tag)
        ref = cv2.imread(f[0], cv2.IMREAD_GRAYSCALE).astype(np.int16)
        vals = [float(np.abs(cv2.imread(f[i], cv2.IMREAD_GRAYSCALE).astype(np.int16) - ref).mean())
                for i in (1, len(f) // 2, len(f) - 1)]
        print(f"change {tag:38s} mean|diff| vs frame0: " + " ".join(f"{v:.1f}" for v in vals))


def roi() -> None:
    for tag, (x0, y0, x1, y1) in BOXES.items():
        f = files_for(tag)
        prev = cv2.imread(f[0], cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1].astype(np.int16)
        worst = 0.0
        for path in f[1:]:
            cur = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[y0:y1, x0:x1].astype(np.int16)
            worst = max(worst, float((np.abs(cur - prev) > 25).mean()) * 100)
            prev = cur
        print(f"roi    {tag:38s} worst consecutive change inside box: {worst:.2f}%")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "drift"):
        drift()
    if which in ("all", "change"):
        change()
    if which in ("all", "roi"):
        roi()


main()
