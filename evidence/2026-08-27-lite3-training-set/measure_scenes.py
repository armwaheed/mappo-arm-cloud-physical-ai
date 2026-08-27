#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure the six 2026-08-27 Lite3 recordings, and write the JSON ``audit.py`` reads.

    python3 measure_scenes.py --frames DIR --recordings DIR --models DIR \
        --out scene_measurements.json

WHAT THIS READS AND WHY IT IS NOT IN GIT. Six ``.mp4`` / ``.jsonl`` pairs, 5,854 frames,
199 MB of video. ``--frames`` is that video already decoded to
JPEG, one directory per scene, ``f%05d.jpg`` from 0. ``--models`` holds the
published ``MobileNetSSD_deploy`` prototxt and caffemodel. None of the three is committed;
this script is how the committed
JSON was produced, and ``audit.py`` recomputes every published number from that JSON with
no video, no model and no network.

FOUR MEASUREMENTS, IN THE ORDER THEY DECIDE WHAT THE FOOTAGE IS WORTH.

1. **Camera motion**, ORB+RANSAC homography against each scene's own first frame — the same
   instrument ``evidence/2026-08-27-lite3-pov-clip-audit`` used on the previous clip, so the
   two are comparable. It answers whether "in different distance and angle" describes the
   camera or only the subject.

2. **Illumination stability**, mean luminance per frame. This is not decoration: a
   background-subtraction gate on a static camera assumes the room's brightness holds
   still, and on three of these six scenes it does not.

3. **A detector pass at a stated preprocessing** — ``mobilenet-ssd-trained``: 300 px,
   scale 1/127.5, mean 127.5, ``swapRB=False``, and the prototxt's own ``DetectionOutput``
   floor of 0.25, with NO class filter. The recordings were written at
   ``go2-navigator-default`` (300 px, 0.4, ``classes=('person',)``), and that class filter
   is why the telemetry carries zero Lite3 boxes. Re-running the SAME weights over the SAME
   pixels without the filter recovers a measurement that was discarded; it does not invent
   one. A fresh ``cv2.dnn.Net`` is built per scene because ``cv2.dnn.Net`` retains state
   across an input-size change.

4. **A background-subtraction motion fraction per box.** The camera does not move, so a
   per-scene temporal median is the empty room. Motion is used ONLY to accept or reject a
   box the detector already produced — it never creates one, because a created box would be
   a fabricated label.

The telemetry join is ``frame = round((t - perception.frame_age_s) * 15)``:
``perception.video_frame`` is null in all 3,896 ticks of all six files, so the documented
join does not exist and the frame the detector saw has to be recovered from its own age.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

#: The published MobileNet-SSD label set, in index order.
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
#: mobilenet-ssd-trained, from robot-stack/unitree/go2/visual_nav/inference_profile.py.
INPUT_SIZE, SCALE, MEAN, SWAP_RB = 300, 1.0 / 127.5, 127.5, False
#: The video rate every one of the six files was written at, from ffprobe.
FPS = 15.0
#: Grey levels a pixel must differ from the empty-room median to count as moving.
MOTION_DIFF = 25
#: Frames sampled to build the empty-room median.
PLATE_FRAMES = 120
#: Homography is measured every Nth frame; 3 is 5 Hz on 15 fps footage.
POSE_STEP = 3


def camera_motion(paths: list[Path]) -> list[float]:
    """Median corner displacement in px against frame 0, by ORB+RANSAC homography."""
    orb = cv2.ORB_create(4000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ref = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    height, width = ref.shape
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    kp_ref, des_ref = orb.detectAndCompute(ref, None)
    out = []
    for path in paths[::POSE_STEP]:
        grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(grey, None)
        value = float("nan")
        if des is not None and des_ref is not None and len(des) > 10:
            matches = matcher.match(des_ref, des)
            if len(matches) >= 12:
                matches = sorted(matches, key=lambda m: m.distance)[:200]
                src = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if homography is not None:
                    moved = cv2.perspectiveTransform(corners, homography)
                    value = float(np.median(np.linalg.norm(moved - corners, axis=2)))
        out.append(value)
    return out


def empty_room(paths: list[Path]) -> np.ndarray:
    """The per-pixel temporal median — valid only because the camera does not move."""
    idx = np.linspace(0, len(paths) - 1, min(PLATE_FRAMES, len(paths))).astype(int)
    stack = np.stack([cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE) for i in idx])
    return np.median(stack, axis=0).astype(np.uint8)


def detect(net, image: np.ndarray) -> list[dict]:
    """Every box the net emits above the prototxt's own floor, with no class filter."""
    height, width = image.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(
        image, SCALE, (INPUT_SIZE, INPUT_SIZE), MEAN, swapRB=SWAP_RB))
    raw = net.forward()
    out = []
    for i in range(raw.shape[2]):
        score = float(raw[0, 0, i, 2])
        class_id = int(raw[0, 0, i, 1])
        x1, y1, x2, y2 = (float(v) for v in raw[0, 0, i, 3:7])
        out.append({
            "cls": VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else "?",
            "score": round(score, 4),
            "box": [round(x1 * width, 2), round(y1 * height, 2),
                    round(x2 * width, 2), round(y2 * height, 2)]})
    return out


def telemetry_sightings(jsonl: Path) -> tuple[list[dict], dict]:
    """Every live sighting, with the video frame the detector actually saw."""
    lines = [json.loads(line) for line in jsonl.open()]
    header = next((t for t in lines if t.get("type") == "header"), {})
    ticks = [t for t in lines if t.get("type") == "tick"]
    rows = []
    joined = 0
    for tick in ticks:
        perception = tick.get("perception") or {}
        age = perception.get("frame_age_s")
        for sighting in (tick.get("sightings") or []):
            frame = None
            if age is not None:
                frame = int(round((tick["t"] - age) * FPS))
                joined += 1
            rows.append({"frame": frame, "t": tick["t"], "label": sighting["label"],
                         "score": round(float(sighting["score"]), 4),
                         "box": [round(float(v), 2) for v in sighting["box"]]})
    meta = {"ticks": len(ticks), "sightings": len(rows), "joined": joined,
            "video_frame_non_null": sum(
                1 for t in ticks
                if (t.get("perception") or {}).get("video_frame") is not None),
            "header_classes": header.get("classes"),
            "header_confidence": header.get("confidence"),
            "header_camera": header.get("camera")}
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    scenes = sorted(d.name for d in args.frames.iterdir()
                    if d.is_dir() and not d.name.startswith("_"))
    out: dict = {"profile": {"name": "mobilenet-ssd-trained", "input_size": INPUT_SIZE,
                             "scale": SCALE, "mean": MEAN, "swap_rb": SWAP_RB,
                             "prototxt_floor": 0.25, "classes": "all VOC, no filter"},
                 "fps": FPS, "motion_diff": MOTION_DIFF, "pose_step": POSE_STEP,
                 "scenes": {}}
    for scene in scenes:
        paths = sorted((args.frames / scene).glob("f*.jpg"))
        jsonl = next((args.recordings / scene).glob("*.jsonl"))
        sightings, meta = telemetry_sightings(jsonl)

        luminance = [float(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).mean())
                     for p in paths]
        plate = empty_room(paths)
        # A fresh net per scene: cv2.dnn.Net retains state across an input-size change.
        net = cv2.dnn.readNetFromCaffe(
            str(args.models / "MobileNetSSD_deploy.prototxt"),
            str(args.models / "MobileNetSSD_deploy.caffemodel"))
        frames = []
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = (cv2.absdiff(grey, plate) > MOTION_DIFF).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            dets = []
            for det in detect(net, image):
                x1, y1, x2, y2 = (int(round(v)) for v in det["box"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                det["motion_frac"] = round(float(mask[y1:y2, x1:x2].mean()), 4)
                dets.append(det)
            frames.append({"frame": index, "file": path.name,
                           "moving_px": int(mask.sum()), "det": dets})

        out["scenes"][scene] = {
            "n_frames": len(paths), "luminance": [round(v, 2) for v in luminance],
            "camera_displacement_px": camera_motion(paths),
            "telemetry": meta, "telemetry_sightings": sightings, "frames": frames}
        print(f"  {scene}: {len(paths)} frames", flush=True)

    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
