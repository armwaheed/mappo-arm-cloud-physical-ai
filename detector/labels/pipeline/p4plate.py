"""Composite background plate for p4, built from p4's own frames -- zero registration error.

The peer is on the left early in the run and parked on the right at the end, so the left of
the plate comes from the late frames and the right from the early ones.  Both halves share a
camera pose, so no warping (and no parallax error) is involved.
"""
import glob

import cv2
import numpy as np

SCRATCH = (
    "/private/tmp/claude-501/-Users-wahbro01-workspaces-git/"
    "ae5beebd-3312-48c6-92c7-3538b392af3f/scratchpad/"
)
SRC = SCRATCH + "peercap/"
WORK = SCRATCH + "peercap_work/"

SPLIT = 1150


def med(files: list[str]) -> np.ndarray:
    return np.median(np.stack([cv2.imread(f) for f in files]), axis=0).astype(np.uint8)


def main() -> None:
    files = sorted(glob.glob(SRC + "p4_mid_sweep_stand_[0-9][0-9][0-9][0-9].jpg"))
    plate = med(files[150:208])
    plate[:, SPLIT:] = med(files[0:26])[:, SPLIT:]
    cv2.imwrite(WORK + "plate_p4.jpg", plate, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(WORK + "plate_p4.npy", plate)
    print(f"plate_p4: left of x={SPLIT} from frames 150-207, right from frames 0-25")


main()
