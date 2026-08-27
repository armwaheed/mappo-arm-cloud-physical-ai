#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Score detector checkpoints on the 2026-08-20 cross-day day, per DEPLOYED configuration.

    # what the peer-avoidance runs compute:
    python3 score_crossday.py --frames-dir XDAY --preprocessing go2-peer-supervised \\
        --proto ~/go2_models/MobileNetSSD_deploy.prototxt \\
        --model  ~/go2_models/MobileNetSSD_deploy.caffemodel --out peer.json

    # what the smoke/berth/chair runs compute, and the sweep's own configuration, from ONE
    # forward pass because they share a square:
    python3 score_crossday.py --frames-dir XDAY --proto base/mnssd22.prototxt \\
        --models 'runs/*/epoch*.caffemodel' \\
        --preprocessing go2-run-smoke --preprocessing go2-navigator-default \\
        --preprocessing mobilenet-ssd-trained \\
        --allow-preprocessing-mismatch 'reason' --out at300.json

WHAT THIS IS FOR. ``eval_detector.py`` answers "did the new class learn?"; this answers
"would the robot behave better?" -- and since **this robot runs three different detectors
depending on which script starts it**, that question has to be asked once per launcher. It
exists because the 2026-08-26 checkpoint sweep asked it once, through a configuration no
launcher runs (issue #129).

THERE IS NO DEFAULT PREPROCESSING, AND THAT IS THE POINT. ``--preprocessing`` is required
and comes from ``robot-stack/unitree/go2/visual_nav/inference_profile.py``, which is the
same object ``deploy/run-peer-supervised.sh`` takes the robot's own flags from. Naming a
configuration no launcher runs is possible and needs a reason, which is written into the
output file beside the numbers it qualifies.

**Profiles that share an input size are scored from one forward pass.** They differ only in
what happens after ``forward()`` -- the Python-side score floor and the class list -- so
this is not an optimisation: it guarantees a difference between two of those rows is that
post-processing and cannot be an inference difference.

THE RULE, one rule for every model including the incumbent. A detection counts when its
score clears the profile's floor; its box is clamped into the frame FIRST -- the order
``person_detector.detect_tiered`` uses, because an unclamped box that ran off the top of the
frame reports a height the lens never imaged and would turn an obstacle into a hold. Then,
per frame:

* ``fire``  -- any detection this profile's ``--classes`` lets through whose clamped aspect
  h/w is BELOW ``person_detector.PERSON_ASPECT_MIN``. That is what the stack routes to the
  policy as an obstacle, so on a peer-present frame it is recall and on a peer-free frame it
  is a false alarm.
* ``hold`` -- any detection this profile's ``--classes`` lets through, at or above that
  aspect. **This is the denominator the hold path acts on, and it is per profile.** A peer
  run passes all twenty VOC labels, so the shape gate alone decides; the other launchers
  pass ``("person",)``, so on those runs `hold` is the person-labelled subset and a `chair`
  box is not an obstacle at all.
* ``person`` -- a box LABELLED ``person``, at any aspect. The denominator
  ``wave6_wholeday.json`` used. Reported so the two can be compared; not the one the robot
  stops on. ``person_detector``'s docstring records the peer being labelled ``person`` on 12
  consecutive live frames.
* ``person_shaped`` -- labelled ``person`` AND person-shaped. The intersection.

OpenCV 4.x. OpenCV 5 removed ``readNetFromCaffe``, and the robot runs 4.2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from glob import glob
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robot-stack" / "unitree"
                       / "go2" / "visual_nav"))
import inference_profile
import person_detector
from inference_profile import PreprocessingMismatch

#: Every split name this script will aggregate. ``whole`` is the union and is what the
#: input-size evidence reports; ``test`` and ``select`` are the manifest's own clip-wise
#: split, and the checkpoint sweep reported ``test``. Both are printed so a row here can
#: be lined up against either published table without re-running anything.
SPLITS = ("whole", "test", "select")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_against_manifest(frames_dir: Path, rows: Sequence[dict]) -> None:
    """Refuse to score an extraction the manifest does not describe exactly.

    Both directions matter, and the quiet one is the second: a JPEG the manifest does NOT
    name would join the peer-free denominator and lower every false-alarm rate on the
    page. Same check ``evidence/2026-08-26-detector-input-size/reproduce.py`` makes.
    """
    named = {f"{r['clip']}_{r['index']:03d}.jpg" for r in rows}
    present = {p.name for p in frames_dir.glob("*.jpg")}
    missing, extra = sorted(named - present), sorted(present - named)
    if missing or extra:
        raise SystemExit(
            f"extraction disagrees with the manifest: {len(missing)} named-but-absent "
            f"{missing[:3]}, {len(extra)} present-but-unnamed {extra[:3]}")
    print(f"  manifest: {len(named)} frames named, {len(present)} present, 0 either way")


def frame_digest(frames_dir: Path, rows: Sequence[dict]) -> str:
    """One digest over the pixels actually scored, in manifest order.

    The frames are decoded from video, and ``detector/labels/CROSSDAY.md`` measured the
    same model at 56% and 60% false-positive rate on the same frames extracted twice at
    different JPEG qualities. So the extraction is part of the measurement: this is what
    lets a run on one machine be compared with a run on another.
    """
    digest = hashlib.sha256()
    for row in rows:
        name = f"{row['clip']}_{row['index']:03d}.jpg"
        digest.update(name.encode())
        digest.update(hashlib.sha256((frames_dir / name).read_bytes()).digest())
    return digest.hexdigest()


def load_frames(frames_dir: Path, rows: Sequence[dict]) -> list[np.ndarray]:
    """Decode once. Inference is 1.3 s per checkpoint and decoding is 4 s, so decoding
    per checkpoint would spend three quarters of a 40-minute sweep re-reading JPEGs."""
    images = []
    for row in rows:
        name = f"{row['clip']}_{row['index']:03d}.jpg"
        image = cv2.imread(str(frames_dir / name))
        if image is None:
            raise SystemExit(f"unreadable: {name}")
        images.append(image)
    return images


#: Baked into the published weights. Every declared profile carries the same three, and
#: ``assert_matches_person_detector`` checks them against ``person_detector``'s own; they
#: are pulled out here so one forward pass can serve several profiles.
_SCALE = inference_profile.GO2_PEER_SUPERVISED.scale
_MEAN = inference_profile.GO2_PEER_SUPERVISED.mean
_SWAP_RB = inference_profile.GO2_PEER_SUPERVISED.swap_rb


def frame_rows(net, image: np.ndarray, size: int, floor: float) -> list:
    """``(score, aspect, label)`` for every detection at or above ``floor``.

    One forward pass. ``floor`` is the LOWEST threshold any caller will apply, so a single
    pass serves every profile at this input size. The LABEL is carried rather than a
    ``person``/not-``person`` flag because the profiles differ in which labels reach the
    planner at all, not only in the threshold.
    """
    height, width = image.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(image, _SCALE, (size, size), _MEAN, swapRB=_SWAP_RB))
    rows = []
    for row in net.forward()[0, 0]:
        class_id, score = int(row[1]), float(row[2])
        if class_id <= 0 or score < floor:
            continue
        x1, y1, x2, y2 = (row[3:7] * np.array([width, height, width, height])).tolist()
        x1, x2 = min(max(x1, 0.0), width - 1.0), min(max(x2, 0.0), width - 1.0)
        y1, y2 = min(max(y1, 0.0), height - 1.0), min(max(y2, 0.0), height - 1.0)
        box_w, box_h = x2 - x1, y2 - y1
        label = (person_detector.VOC_CLASSES[class_id]
                 if class_id < len(person_detector.VOC_CLASSES) else f"?{class_id}")
        rows.append((score, (box_h / box_w) if box_w > 0 else 0.0, label))
    return rows


def verdict(rows: Sequence[tuple], profile) -> dict:
    """The four per-frame booleans, for one profile.

    ``fire`` and ``hold`` are what THIS profile's stack would do, so they are filtered by
    ``profile.classes`` first: ``PersonDetector.detect_tiered`` drops a detection whose
    label is not in that list before the tracker, the map or the planner can see it. A
    peer run passes all twenty VOC labels, so nothing is dropped and the aspect gate alone
    decides; the smoke/berth/chair runs and a bare navigator pass ``("person",)``, so a
    `chair` box is not an obstacle on those runs however well it fits the peer.

    ``person`` and ``person_shaped`` are deliberately NOT class-filtered. They are
    properties of the network's output rather than of one stack's configuration, and
    ``person`` is the denominator the checkpoint sweep published, which has to stay
    comparable across profiles.
    """
    allowed = frozenset(profile.classes)
    fire = hold = person = person_shaped = False
    for score, aspect, label in rows:
        if score < profile.confidence:
            continue
        shaped = aspect >= person_detector.PERSON_ASPECT_MIN
        if label in allowed:
            fire = fire or not shaped
            hold = hold or shaped
        if label == "person":
            person = True
            person_shaped = person_shaped or shaped
    return {"fire": fire, "hold": hold, "person": person, "person_shaped": person_shaped}


def aggregate(per_frame: Sequence[dict], rows: Sequence[dict]) -> dict[str, dict]:
    """Recall, false alarms and the three person denominators, per split."""
    out = {}
    for split in SPLITS:
        chosen = [(r, v) for r, v in zip(rows, per_frame)
                  if split == "whole" or r["split"] == split]
        positive = [v for r, v in chosen if r["present"] is True]
        negative = [v for r, v in chosen if r["present"] is False]
        out[split] = {
            "recall_n": sum(v["fire"] for v in positive), "recall_d": len(positive),
            "fp_n": sum(v["fire"] for v in negative), "fp_d": len(negative),
            "person": sum(v["person"] for _, v in chosen),
            "person_shaped": sum(v["person_shaped"] for _, v in chosen),
            "hold": sum(v["hold"] for _, v in chosen),
        }
    return out


def score_model(proto: Path, weights: Path, images: Sequence[np.ndarray],
                rows: Sequence[dict], profiles: Sequence) -> dict:
    """One entry, carrying one result set per profile.

    Grouped by input size, so profiles that differ only in what happens AFTER ``forward()``
    -- the score floor and the class list -- share one pass. See this module's docstring
    for why that is a correctness property and not a speed one.
    """
    by_size: dict = {}
    for profile in profiles:
        by_size.setdefault(profile.input_size, []).append(profile)
    results = {}
    for size, group in sorted(by_size.items()):
        # ⚠️ A FRESH Net PER INPUT SIZE, AND THIS IS NOT TIDINESS. `cv2.dnn.Net` keeps
        # state across a change of input blob size: measured on the shipped weights and
        # this corpus, scoring 300 px on a Net that has already run 224 px gives 42/60
        # peer recall and hold 41, where a Net that has only ever seen 300 gives 41/60 and
        # hold 40. Same weights, same frames, same threshold — one frame of difference
        # that depends on the ORDER the sizes were swept in. Reusing the Net would make
        # this scorer's answer a function of its own argument order, which is the same
        # class of defect as #129 and would be far harder to see.
        net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        floor = min(p.confidence for p in group)
        per_frame = [frame_rows(net, image, size, floor) for image in images]
        for profile in group:
            results[profile.name] = aggregate(
                [verdict(f, profile) for f in per_frame], rows)
    return {"model": str(weights), "name": f"{weights.parent.name}/{weights.stem}",
            "sha256": sha256(weights), "results": results}


def dump(header: dict, entries: Sequence[dict]) -> str:
    """The results file: everything pretty, one line per checkpoint.

    800 checkpoints at ``indent=1`` is 27,000 lines of JSON in a review diff, and a diff
    nobody reads is a diff nobody checks. One line per model stays diffable — a checkpoint
    whose numbers moved is a line that moved — and a reader can grep it. Compact
    throughout would be one 500 kB line, which is what this directory's
    ``live_detections.json`` is and is not a precedent worth extending.
    """
    parts = [f" {json.dumps(key)}: {json.dumps(value, indent=1)}"
             for key, value in header.items()]
    rows = ",\n  ".join(json.dumps(entry, sort_keys=True) for entry in entries)
    parts.append(f' "models": [\n  {rows}\n ]')
    return "{\n" + ",\n".join(parts) + "\n}\n"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} = {100.0 * n / d:.0f}%" if d else "n/a"


def print_row(entry: dict, profile_name: str, split: str) -> None:
    v = entry["results"][profile_name][split]
    print(f"  {entry['name']:<34s} recall {_pct(v['recall_n'], v['recall_d']):>14s}   "
          f"false {_pct(v['fp_n'], v['fp_d']):>15s}   hold {v['hold']:>3d}   "
          f"person {v['person']:>3d}   +shaped {v['person_shaped']:>3d}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent / "labels"
                        / "peer_crossday_20260820.json")
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="<clip>_<index>.jpg, decoded at the quality the manifest's "
                             "own 'extract' field names")
    parser.add_argument("--proto", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", default=None,
                        help="one .caffemodel; repeatable")
    parser.add_argument("--models", default="", help="glob of .caffemodel files")
    parser.add_argument("--inventory-glob", default="",
                        help="everything that COULD have been scored. What matched this "
                             "and was not scored is written into --out as `coverage`, so "
                             "a partial sweep cannot be reported as a complete one — "
                             "which is the error the 2026-08-26 sweep made when 627 of "
                             "640 checkpoints went unscored and unmentioned.")
    parser.add_argument("--report-split", default="whole", choices=SPLITS,
                        help="which split the progress lines print (all are in --out)")
    parser.add_argument("--out", type=Path, required=True)
    inference_profile.add_arguments(parser, multiple=True)
    args = parser.parse_args(argv)

    # ONE except FOR BOTH REFUSALS. The profile guard and the prototxt floor guard raise the
    # same exception and are the same kind of answer -- "this measurement would not describe
    # a run" -- so they leave by the same door, as a message and exit 2 rather than as a
    # traceback. A traceback reads as a crash, and a crash is something a reader works
    # around.
    try:
        profiles, reason = inference_profile.resolve_many(args)
        inference_profile.assert_matches_person_detector(person_detector)
        floors = {p.name: inference_profile.assert_prototxt_floor(args.proto, p)
                  for p in profiles}
    except PreprocessingMismatch as refusal:
        parser.exit(2, f"\nREFUSED\n{refusal}\n\n")

    rows = json.loads(args.manifest.read_text())["frames"]
    check_against_manifest(args.frames_dir, rows)
    for profile in profiles:
        layer = floors[profile.name]
        print(f"  {profile.name:<24s} {profile.input_size} px, confidence "
              f"{profile.confidence}"
              + ("" if profile.confidence == layer
                 else f" (stricter than the prototxt layer's {layer})")
              + (f"  <- {', '.join(profile.deployments)}" if profile.is_deployed
                 else f"  <- RUN BY NO LAUNCHER: {reason}"))

    weights = [Path(p) for p in sorted(glob(args.models))] if args.models else []
    weights += list(args.model or [])
    if not weights:
        parser.error("give --model or --models")
    images = load_frames(args.frames_dir, rows)
    sizes = sorted({p.input_size for p in profiles})
    print(f"  {len(images)} frames decoded once; {len(weights)} checkpoints x "
          f"{len(sizes)} forward pass(es) each ({sizes} px)\n")

    entries = []
    for index, weight in enumerate(weights, 1):
        entry = score_model(args.proto, weight, images, rows, profiles)
        entries.append(entry)
        for profile in profiles:
            print(f"  [{index:>3d}/{len(weights)}] {profile.name:<22s}", end="")
            print_row(entry, profile.name, args.report_split)

    coverage = {"scored": len(weights)}
    if args.inventory_glob:
        available = sorted(glob(args.inventory_glob))
        scored = {str(w) for w in weights}
        skipped = [p for p in available if p not in scored]
        coverage.update({"glob": args.inventory_glob, "available": len(available),
                         "not_scored": skipped})
        print(f"\n  coverage: {len(weights)} scored of {len(available)} matching "
              f"{args.inventory_glob}; {len(skipped)} not scored"
              + (f" — {skipped}" if 0 < len(skipped) <= 12 else ""))

    args.out.write_text(dump({
        "preprocessing": [dict(inference_profile.stamp(p, reason),
                               prototxt_layer_floor=floors[p.name]) for p in profiles],
        "prototxt": {"path": str(args.proto), "sha256": sha256(args.proto)},
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest),
                     "frames": len(rows)},
        "frames": {"dir": str(args.frames_dir),
                   "digest": frame_digest(args.frames_dir, rows)},
        "coverage": coverage,
        "rule": {"aspect_min": person_detector.PERSON_ASPECT_MIN,
                 "source": "person_detector.PERSON_ASPECT_MIN"},
    }, entries))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
