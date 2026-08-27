#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure the Lite3 point-of-view clip and write every raw number to JSON.

WHAT THIS IS FOR. The clip is 20 MB and cannot go in git, so this script runs once
against the video and writes ``clip_measurements.json``; ``audit.py`` then recomputes
every published number from that file with no video present. Nothing in the README is
derived by hand.

WHAT IT MEASURES.
  * frame count and rate, four independent ways, because the container disagrees with
    itself: ``r_frame_rate`` says 7 and ``avg_frame_rate`` says 27/5.
  * CAMERA motion, by ORB+RANSAC homography against frame 1, masked to exclude the two
    HUD panels. This is the count of viewpoints; pixel change is not, because a person
    walking through a fixed frame changes pixels without adding a viewpoint.
  * which frames carry a burned-in detection outline, by the overlay's own hue.
  * per-region deviation from the block median, with inert regions as controls, so
    "the peer did not move" is a comparison and not an assertion.

USAGE. ``python3 measure_clip.py --video CLIP.mp4 --out clip_measurements.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

#: The overlay's own geometry, from robot-stack/unitree/go2/visual_nav/overlay.py:
#: draw_plan_view() places a size_px=300 panel at (width - size_px - 16, 16).
RADAR_SIZE, RADAR_MARGIN = 300, 16
#: The bottom-left status plate, measured from the pixels (draw_status()).
PLATE = (0, 630, 470, 90)
#: Every overlay pixel sampled has hue exactly 17; saturation varies because the stroke
#: is alpha-blended over the scene.
HUE_LO, HUE_HI = np.array([15, 140, 180]), np.array([19, 255, 255])
#: A scene edge is not 90 px of perfectly straight constant hue; an overlay stroke is.
MIN_RUN = 90


def probe(video: Path) -> dict:
    def ff(*args):
        return subprocess.run(["ffprobe", "-v", "error", *args, str(video)],
                              capture_output=True, text=True).stdout.strip()
    return {
        "container_nb_frames": ff("-select_streams", "v:0", "-show_entries",
                                  "stream=nb_frames", "-of", "csv=p=0"),
        "decoded_frames": ff("-select_streams", "v:0", "-count_frames", "-show_entries",
                             "stream=nb_read_frames", "-of", "csv=p=0"),
        "packets": ff("-select_streams", "v:0", "-count_packets", "-show_entries",
                      "stream=nb_read_packets", "-of", "csv=p=0"),
        "r_frame_rate": ff("-select_streams", "v:0", "-show_entries",
                           "stream=r_frame_rate", "-of", "csv=p=0"),
        "avg_frame_rate": ff("-select_streams", "v:0", "-show_entries",
                             "stream=avg_frame_rate", "-of", "csv=p=0"),
        "duration_s": ff("-show_entries", "format=duration", "-of", "csv=p=0"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("clip_measurements.json"))
    args = ap.parse_args()

    out: dict = {"video": args.video.name,
                 "md5": hashlib.md5(args.video.read_bytes()).hexdigest(),
                 "ffprobe": probe(args.video)}

    cap = cv2.VideoCapture(str(args.video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n = len(frames)
    H, W = frames[0].shape[:2]
    radar = (W - RADAR_SIZE - RADAR_MARGIN, RADAR_MARGIN, RADAR_SIZE, RADAR_SIZE)
    out["opencv_frames"] = n
    out["width"], out["height"] = W, H
    out["fps_measured"] = n / float(out["ffprobe"]["duration_s"])
    out["radar_from_overlay_source"] = list(radar)
    out["plate_measured"] = list(PLATE)

    def hud_mask(m):
        x, y, w, h = radar
        m[y:y+h, x:x+w] = 0
        x, y, w, h = PLATE
        m[y:y+h, x:x+w] = 0
        return m

    # ---- camera motion: ORB + RANSAC homography against frame 1 ----
    keep = np.full((H, W), 255, np.uint8)
    hud_mask(keep)
    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    kp0, des0 = orb.detectAndCompute(g0, keep)
    corners = np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]]).reshape(-1, 1, 2)
    disp = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(g, keep)
        d = None
        if des is not None and len(kp) > 20:
            mt = bf.match(des0, des)
            if len(mt) >= 20:
                src = np.float32([kp0[x.queryIdx].pt for x in mt]).reshape(-1, 1, 2)
                dst = np.float32([kp[x.trainIdx].pt for x in mt]).reshape(-1, 1, 2)
                Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if Hm is not None:
                    mv = cv2.perspectiveTransform(corners, Hm)
                    d = float(np.median(np.linalg.norm(mv - corners, axis=2)))
        disp.append(d)
    out["camera_displacement_px"] = disp

    # ---- which frames carry an overlay outline on the SCENE ----
    contaminated = []
    for i, f in enumerate(frames):
        m = hud_mask(cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), HUE_LO, HUE_HI))
        v = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((MIN_RUN, 1), np.uint8))
        z = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, MIN_RUN), np.uint8))
        if v.any() or z.any():
            contaminated.append(i)
    out["outline_frames"] = contaminated

    # ---- within the outline-free block, per-region deviation from that block's median ----
    clean_idx = [i for i in range(n) if i not in set(contaminated)]
    block = np.stack([frames[i] for i in clean_idx])
    med = np.median(block, axis=0).astype(np.int16)
    regions = {
        "peer": (470, 400, 460, 320),
        "glass_wall_inert": (0, 0, 400, 400),
        "ceiling_inert": (600, 0, 300, 120),
        "carpet_inert": (150, 560, 250, 150),
        "radar_synthetic": (980, 40, 260, 260),
    }
    out["clean_block"] = clean_idx
    out["region_deviation"] = {
        k: float(np.median(np.abs(block[:, y:y+h, x:x+w].astype(np.int16)
                                  - med[y:y+h, x:x+w]).mean(axis=(1, 2, 3))))
        for k, (x, y, w, h) in regions.items()}

    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}  ({n} frames, {len(contaminated)} with an outline)")


if __name__ == "__main__":
    main()
