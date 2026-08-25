"""Per-frame peer boxes for p4, the one segment where the peer actually crosses the frame.

p4 gets its own plate built from its own frames, so registration is exact and the difference
is clean enough to use directly as the box.  Cast shadow is rejected only in the floor band:
a grey robot against a light wall satisfies the same colour-ratio test, so applying it
frame-wide eats the robot's own top edge.
"""
import glob
import json

import cv2
import numpy as np

SCRATCH = (
    "/private/tmp/claude-501/-Users-wahbro01-workspaces-git/"
    "ae5beebd-3312-48c6-92c7-3538b392af3f/scratchpad/"
)
SRC = SCRATCH + "peercap/"
WORK = SCRATCH + "peercap_work/"

THR, MIN_AREA, MERGE_FRAC, FLOOR_Y = 28, 1500, 0.12, 950


def main() -> None:
    plate = np.load(WORK + "plate_p4.npy")
    k = np.ones((3, 3), np.uint8)
    lo, hi = cv2.erode(plate, k).astype(np.int16), cv2.dilate(plate, k).astype(np.int16)
    pf = plate.astype(np.float32) + 1.0
    files = sorted(glob.glob(SRC + "p4_mid_sweep_stand_[0-9][0-9][0-9][0-9].jpg"))
    out = {}
    for f in files:
        raw = cv2.imread(f)
        img = raw.astype(np.int16)
        d = np.maximum(img - hi, lo - img).clip(0).max(axis=2).astype(np.uint8)
        ratio = raw.astype(np.float32) / pf
        shadow = ((ratio > 0.45) & (ratio < 0.92)).all(axis=2) & \
                 (ratio.max(axis=2) - ratio.min(axis=2) < 0.16)
        shadow[:FLOOR_Y] = False
        d[shadow] = 0
        d = cv2.GaussianBlur(d, (7, 7), 0)
        mask = cv2.morphologyEx((d > THR).astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        blobs = sorted((s for s in stats[1:] if s[4] >= MIN_AREA), key=lambda s: -s[4])
        if not blobs:
            out[f.rsplit("/", 1)[1]] = None
            continue
        keep = [b for b in blobs if b[4] >= MERGE_FRAC * blobs[0][4]]
        out[f.rsplit("/", 1)[1]] = [int(min(b[0] for b in keep)), int(min(b[1] for b in keep)),
                                    int(max(b[0] + b[2] for b in keep)),
                                    int(max(b[1] + b[3] for b in keep))]
    with open(WORK + "p4_boxes.json", "w") as fh:
        json.dump(out, fh)
    for i, name in enumerate(sorted(out)):
        if i % 20 == 0:
            print(name.split("_")[-1], out[name])
    print("frames with no blob:", sum(1 for v in out.values() if v is None))


main()
