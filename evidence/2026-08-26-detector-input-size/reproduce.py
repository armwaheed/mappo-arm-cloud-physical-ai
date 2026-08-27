#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Every number in this directory's README, regenerated.

    python3 reproduce.py --model-dir ~/go2_models

WHAT THIS MEASURES. The shipped MobileNet-SSD, scored twice on the same frames with the
same rule, changing exactly one thing: the square the frame is squashed into before it
reaches the network. 300 is what every scorer in ``detector/`` hardcodes. 224 is what
``deploy/run-peer-supervised.sh`` passes to ``--input-size``, so it is what the peer runs
this project reports actually execute. Nothing else differs between the two columns --
same weights, same 0.25 floor, same frames, same decode, one process.

WHY THAT IS WORTH A DIRECTORY. The 2026-08-26 checkpoint sweep ranks 94 checkpoints
against a shipped baseline of "68% recall, 49% false alarms". This script measures that
same baseline at 224 and gets 50% / 26%. The candidate detectors were never scored at
224, so the margin they were selected on is a margin over a baseline the robot does not
run.

WHAT IT NEEDS. The weights are not vendored, the same way they are not vendored for
``detector/eval_class_agnostic.py`` -- pass ``--model-dir``, holding
``MobileNetSSD_deploy.prototxt`` and ``MobileNetSSD_deploy.caffemodel``. The eval frames
ARE reachable from a clone: the seven 2026-08-20 clips are recovered from the commit
``detector/labels/CROSSDAY.md`` names, decoded here at the quality that file specifies,
and checked against the committed manifest before anything is scored.

OPENCV 4.x IS REQUIRED. OpenCV 5 removed ``readNetFromCaffe``; the robot runs 4.2.

The live section needs no model and no robot. It re-derives its aggregates from
``live_detections.json``, which is this directory's committed copy of the network's own
output over 149 frames pulled from the robot's frame server on 2026-08-26. The pixels are
27 MB and are not committed; the detections they produced are.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from collections import Counter

#: The commit `detector/labels/CROSSDAY.md` names. The clips are NOT on the default
#: branch -- the branch that carried them was never merged -- so they are addressed by
#: commit, which is the one route that works from any clone.
CLIPS_COMMIT = "f7b158f3bf18ba9868a40305985f75dc42374a7b"
CLIPS_PATH = "evidence/2026-08-20-peer-avoidance/scene-captures/_raw"
CLIPS = ("peer_cross1", "peer_cross5", "chair1", "gs-0.6-300-0.35",
         "gs-1.0-300-0.50", "peer_baseline", "smoke1")

#: VOC-21, in the order the shipped prototxt emits class ids.
VOC = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
       "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
       "pottedplant", "sheep", "sofa", "train", "tvmonitor")

#: Baked into the published weights. Not tunable, and not the thing under test here.
SSD_SCALE, SSD_MEAN = 1.0 / 127.5, 127.5

#: `deploy/run-peer-supervised.sh` launches with `--confidence 0.25`, and the prototxt's
#: own DetectionOutput floor is 0.25 as well, so this is both the layer's floor and the
#: caller's.
CONF = 0.25

#: `person_detector.PERSON_ASPECT_MIN`. At or above this a box STOPS the robot; below it
#: the box is routed to the policy as an obstacle. Shape decides that, not the VOC label
#: -- see `RangedDetection.person_shaped` for the twelve-frame measurement behind it.
ASPECT_MIN = 2.0

REPO = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPO / "detector" / "labels" / "peer_crossday_20260820.json"


def recover_clips(work: pathlib.Path) -> None:
    """`git show` the seven clips out of the commit that still carries them."""
    raw = work / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        target = raw / f"{clip}.mp4"
        if target.exists() and target.stat().st_size:
            continue
        spec = f"{CLIPS_COMMIT}:{CLIPS_PATH}/{clip}.mp4"
        with target.open("wb") as handle:
            done = subprocess.run(["git", "show", spec], cwd=REPO, stdout=handle)
        if done.returncode:
            target.unlink(missing_ok=True)
            raise SystemExit(
                f"could not recover {spec}.\n"
                f"Fetch the commit first:  git fetch origin {CLIPS_COMMIT}")


def decode(work: pathlib.Path) -> pathlib.Path:
    """Decode every frame of every clip at the quality CROSSDAY.md specifies.

    QUALITY 95 IS NOT COSMETIC. That file measured the same model at 56% and 60%
    false-positive rate on the same frames extracted twice at different JPEG qualities,
    so the extraction is part of the measurement and is pinned here rather than left to
    a default.
    """
    import cv2

    out = work / "xday"
    out.mkdir(parents=True, exist_ok=True)
    if len(list(out.glob("*.jpg"))) == 284:
        return out
    for clip in CLIPS:
        cap = cv2.VideoCapture(str(work / "_raw" / f"{clip}.mp4"))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out / f"{clip}_{index:03d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            index += 1
        cap.release()
    return out


def check_against_manifest(frames_dir: pathlib.Path, rows: list) -> None:
    """Refuse to score an extraction the manifest does not describe exactly.

    Both directions matter. A frame the manifest names and the directory lacks is an
    obvious failure; a JPEG in the directory the manifest does NOT name is the quiet one,
    because it would join the peer-free denominator and lower every false-alarm rate on
    the page.
    """
    named = {f"{r['clip']}_{r['index']:03d}.jpg" for r in rows}
    present = {p.name for p in frames_dir.glob("*.jpg")}
    missing, extra = sorted(named - present), sorted(present - named)
    if missing or extra:
        raise SystemExit(f"extraction disagrees with the manifest: "
                         f"{len(missing)} named-but-absent {missing[:3]}, "
                         f"{len(extra)} present-but-unnamed {extra[:3]}")
    print(f"  manifest: {len(named)} frames named, {len(present)} present, 0 either way")


def detect(net, image, size: int) -> list:
    """``(label, score, aspect)`` per row, boxes clamped into the frame first.

    CLAMP BEFORE ASPECT, which is the order `person_detector.detect_tiered` uses. SSD box
    regression lands outside the image often enough to matter, and an unclamped box that
    ran off the top of the frame reports a height the lens never imaged -- which would
    inflate the aspect and turn an obstacle into a person-shaped hold.
    """
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(image, SSD_SCALE, (size, size), SSD_MEAN))
    rows = []
    for row in net.forward()[0, 0]:
        class_id, score = int(row[1]), float(row[2])
        if class_id <= 0 or score < CONF:
            continue
        x1, y1, x2, y2 = (row[3:7] * np.array([width, height, width, height])).tolist()
        x1, x2 = min(max(x1, 0.0), width - 1.0), min(max(x2, 0.0), width - 1.0)
        y1, y2 = min(max(y1, 0.0), height - 1.0), min(max(y2, 0.0), height - 1.0)
        box_w, box_h = x2 - x1, y2 - y1
        label = VOC[class_id] if class_id < len(VOC) else f"?{class_id}"
        rows.append((label, score, (box_h / box_w) if box_w > 0 else 0.0))
    return rows


def score_crossday(model_dir: pathlib.Path, frames_dir: pathlib.Path,
                   rows: list, sizes: tuple) -> dict:
    """The one rule, every size, every split."""
    import cv2

    proto = model_dir / "MobileNetSSD_deploy.prototxt"
    weights = model_dir / "MobileNetSSD_deploy.caffemodel"
    for path in (proto, weights):
        if not path.is_file():
            raise SystemExit(f"{path} not found -- see this directory's README for where "
                             f"the weights come from")
    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))

    report = {}
    for size in sizes:
        per = {}
        for record in rows:
            name = f"{record['clip']}_{record['index']:03d}.jpg"
            image = cv2.imread(str(frames_dir / name))
            if image is None:
                raise SystemExit(f"unreadable: {name}")
            kept = detect(net, image, size)
            per[name] = {
                "split": record["split"],
                "present": record["present"],
                # A frame "fires" when the stack would route SOMETHING to the policy as an
                # obstacle: any detection that is not person-shaped.
                "fire": any(a < ASPECT_MIN for _, _, a in kept),
                # Label only -- the denominator `wave6_wholeday.json` uses.
                "person": any(lb == "person" for lb, _, _ in kept),
                # Label AND shape -- the subset that actually reaches the hold path.
                "person_shaped": any(lb == "person" and a >= ASPECT_MIN
                                     for lb, _, a in kept),
                # Shape alone, label ignored. This is what the peer runs really hold on:
                # they pass all 21 VOC classes to --classes, so every detection is a
                # mover and only the aspect gate decides hold-vs-route.
                "hold": any(a >= ASPECT_MIN for _, _, a in kept),
            }
        report[size] = {}
        for split in ("test", "select", "whole"):
            chosen = [v for v in per.values()
                      if split == "whole" or v["split"] == split]
            positive = [v for v in chosen if v["present"] is True]
            negative = [v for v in chosen if v["present"] is False]
            report[size][split] = {
                "recall_n": sum(v["fire"] for v in positive), "recall_d": len(positive),
                "fp_n": sum(v["fire"] for v in negative), "fp_d": len(negative),
                "person": sum(v["person"] for v in chosen),
                "person_shaped": sum(v["person_shaped"] for v in chosen),
                "hold": sum(v["hold"] for v in chosen),
            }
    return report


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} = {100.0 * n / d:.0f}%" if d else "n/a"


def print_crossday(report: dict) -> None:
    print("\n== The shipped weights on the 2026-08-20 cross-day split ==")
    print("   fire = any detection >= 0.25 whose clamped box aspect h/w < 2.0\n")
    header = f"{'split':8s} {'size':>5s}  {'peer recall':>16s}  {'false alarms':>16s}"
    print(header + f"  {'person':>7s} {'+shaped':>8s} {'any-shaped':>11s}")
    for split in ("test", "select", "whole"):
        for size in sorted(report, reverse=True):
            v = report[size][split]
            print(f"{split:8s} {size:5d}  {_pct(v['recall_n'], v['recall_d']):>16s}  "
                  f"{_pct(v['fp_n'], v['fp_d']):>16s}  {v['person']:>7d} "
                  f"{v['person_shaped']:>8d} {v['hold']:>11d}")
        print()


def print_live(path: pathlib.Path) -> None:
    """Aggregate the committed live detections. No model, no robot, no network."""
    data = json.loads(path.read_text())
    print("== The live Go2 scene, 2026-08-26 ==")
    print(f"   {data['frames']} frames pulled over HTTP from the robot's frame server.")
    print("   Detections below are from a prototxt whose DetectionOutput floor was")
    print("   patched 0.25 -> 0.01, so the sub-threshold structure is visible.\n")
    for size in sorted(data["sizes"], key=int, reverse=True):
        per = data["sizes"][size]
        rows = [d for frame in per.values() for d in frame]
        shipped = [d for d in rows if d[1] >= CONF]
        person = [d for d in rows if d[0] == "person"]
        fired = sum(1 for frame in per.values()
                    if any(d[1] >= CONF and d[2] < ASPECT_MIN for d in frame))
        held = sum(1 for frame in per.values()
                   if any(d[1] >= CONF and d[2] >= ASPECT_MIN for d in frame))
        labels = Counter(d[0] for d in rows)
        print(f"  size {size}: {len(rows):,} detections >= 0.01, "
              f"{len(shipped)} of them >= 0.25")
        print(f"    at the shipped 0.25 floor: {fired}/{len(per)} frames fire, "
              f"{held}/{len(per)} hold")
        print(f"    best score any class {max(d[1] for d in rows):.4f};  "
              f"best `person` {max(d[1] for d in person):.4f}  "
              f"({len(person):,} sub-floor person rows)")
        print(f"    top labels: {dict(labels.most_common(5))}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=pathlib.Path,
                        help="holds MobileNetSSD_deploy.prototxt/.caffemodel. Omit to "
                             "run the live section only.")
    parser.add_argument("--work-dir", type=pathlib.Path,
                        default=pathlib.Path("_reproduce"),
                        help="scratch for the recovered clips and decoded frames")
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 224])
    parser.add_argument("--out", type=pathlib.Path,
                        help="write the cross-day report as JSON")
    args = parser.parse_args(argv)

    here = pathlib.Path(__file__).resolve().parent
    print_live(here / "live_detections.json")

    if args.model_dir is None:
        print("no --model-dir given, so the cross-day section was SKIPPED. That is the "
              "half that decides the question; see this directory's README.")
        return 0

    rows = json.loads(MANIFEST.read_text())["frames"]
    work = args.work_dir if args.work_dir.is_absolute() else here / args.work_dir
    print(f"recovering the 2026-08-20 clips into {work}")
    recover_clips(work)
    frames_dir = decode(work)
    check_against_manifest(frames_dir, rows)

    report = score_crossday(args.model_dir, frames_dir, rows, tuple(args.sizes))
    print_crossday(report)
    if args.out:
        args.out.write_text(json.dumps({str(k): v for k, v in report.items()}, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
