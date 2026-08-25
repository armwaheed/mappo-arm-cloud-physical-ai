"""Verification sheet: sample frames of one segment, draw the final box, tile them."""
import json
import sys

import cv2
import numpy as np

SCRATCH = (
    "/private/tmp/claude-501/-Users-wahbro01-workspaces-git/"
    "ae5beebd-3312-48c6-92c7-3538b392af3f/scratchpad/"
)
SRC = SCRATCH + "peercap/"
OUT = SCRATCH + "peercap_labelled/"


def main() -> None:
    tag, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(OUT + "annotations.json") as fh:
        doc = json.load(fh)
    recs = sorted((r for r in doc["records"] if r["image"].startswith(tag + "_")),
                  key=lambda r: r["image"])
    picks = [recs[round(i * (len(recs) - 1) / (n - 1))] for i in range(n)]
    tiles = []
    for r in picks:
        img = cv2.imread(SRC + r["image"])
        b = r["box"]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 4)
        cv2.putText(img, f"{r['image'].rsplit('_', 1)[1]} {b}", (16, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        tiles.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    cols = 2 if n <= 4 else 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, len(recs), "records")


main()
