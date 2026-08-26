#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn one recorded run into a label manifest, by joining its telemetry to its raw video.

Every control tick already writes the pixel box, the label, the score, the range, the
bearing and which size prior produced it — and ``perception.video_frame``, the index of
the frame ``--record``/``--record-raw`` wrote on that tick::

    "perception": {"video_frame": 12, "seq": 112, "detect_ms": 162.5},
    "sightings": [{"box": [1176.1, 327.7, 1804.0, 859.3], "label": "person",
                   "range_m": 1.267, "bearing_rad": -0.411, "score": 0.392,
                   "source": "height"}]

Nothing consumed it. Before this file, ``grep -rl 'video_frame\\|sightings' detector/``
returned one hit and it was a README. So a recorded run was a video and a log, and never a
dataset — which is the sentence issue #77 opens with.

This walks the JSONL, maps each ``video_frame`` onto the frame at that index in the raw
MP4, writes the JPEGs, and writes a manifest in the ``records`` shape the two manifests in
this directory already use, so ``check_manifest.py`` and ``eval_class_agnostic.py`` read it
unchanged::

    python3 autolabel_run.py run.jsonl --frames-dir OUT --manifest OUT/labels.json \\
        --classes person --label go2wheel
    python3 check_manifest.py OUT/labels.json --frames-dir OUT      # passes, both directions

⛔ **THESE ARE DETECTOR BOXES, NOT GROUND TRUTH.** Every box here was produced by the same
MobileNet-SSD the robot ran, so this manifest inherits that network's recall exactly: 64%
class-agnostic at the deployed 0.25 threshold on the 2026-08-24 corpus (`detector/README.md`).
It is an auto-label for the frames the detector already found and says NOTHING about the
ones it missed. Two consequences, and neither is negotiable:

* **Not a recall benchmark.** Scoring the same network against labels the same network
  produced measures agreement with itself and returns ~100% by construction. A test set has
  to be hand-labelled, or labelled by something else.
* **A frame with no sighting is not a peer-free frame.** It is a frame nobody has labelled.
  So frames without a surviving sighting are NOT extracted and NOT named here — writing
  them would hand ``eval_class_agnostic.py`` a false-alarm denominator built out of the
  detector's own misses. The count is printed instead; hand-label those if you need them.

What it IS good for: range and aspect statistics over real runs (every record carries
``range_m``, ``bearing_rad`` and the prior that produced them), a much bigger pool of
pseudo-labels than a staged capture, and turning a fleet of recorded runs into frames a
human can correct rather than draw from scratch.

⚠️ **Point it at the RAW video.** ``--record`` burns an orange box around every detection,
at exactly the place the label goes; training on those frames teaches the network to find
an orange rectangle. ``--record-raw`` (issue #77, landed in #84) writes the undecorated
frame with the same frame counter, so frame *n* of both files is the same tick. This
refuses an annotated video it can recognise, and cannot recognise one you rename.

Nuisance variables — corridor, lighting, peer pose, date — are still recorded nowhere, so a
corpus pooled from several runs still cannot be split into an honest cross-session holdout.
What the run header does carry (camera, confidence, envelope, goal, wall clock) is copied
into ``source.run`` so at least the run is identifiable. That is item 4 of issue #77 and it
is not solved here.

Run the tests with ``python3 test_autolabel_run.py``. This module is pure stdlib until it
actually decodes a video; ``cv2`` is imported inside the two functions that need it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Filename the manifest names and the extractor writes. Four digits to match
#: ``peer_go2wheel_20260824.json`` (``p1_close_broadside_0000.jpg``), and the number IS the
#: ``perception.video_frame`` it came from, so the join stays legible from the filename
#: alone. ``%04d`` widens past 9999 rather than truncating.
FRAME_KEY = "{prefix}_{index:04d}.jpg"

#: What the two committed manifests' own ``extract`` one-liner uses.
JPEG_QUALITY = 95

#: Written into every record and every manifest built here, so no reader can mistake these
#: for hand-drawn boxes. ``check_manifest.py`` ignores keys it does not know.
PROVENANCE = (
    "auto-labelled from detector output recorded in the run's own telemetry — NOT ground "
    "truth. Inherits the detector's recall (64% class-agnostic at 0.25), so it cannot "
    "benchmark that detector and its frames-without-a-box are unlabelled, not peer-free."
)


class Refused(SystemExit):
    """Exit with a reason. Subclassed only so the tests can tell it from a crash."""


# ── Reading the run ─────────────────────────────────────────────────────────
def read_run(path: Path) -> tuple[dict, list[dict]]:
    """``(header, ticks)`` from a telemetry JSONL.

    Tolerates a truncated last line: the writer flushes every record but a run that ends
    on a power cut can still lose one, and refusing the whole file over its final byte
    would throw away the run it is meant to salvage.
    """
    header: dict = {}
    ticks: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"  line {number} is not JSON, ignored (truncated run?)")
            continue
        if record.get("type") == "header":
            header = record
        elif record.get("type") == "tick":
            ticks.append(record)
    if not ticks:
        raise Refused(f"{path} carries no ticks")
    return header, ticks


def _usable(sighting: dict) -> bool:
    """A sighting whose box can be written into a manifest at all.

    ``telemetry._finite`` writes ``null`` for anything non-finite, and a box that is not
    four finite numbers with positive extent is precisely what ``check_manifest.check_boxes``
    exists to catch — it scores IoU 0 against every prediction and reads as a miss.
    """
    box = sighting.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if any(not isinstance(value, (int, float)) for value in box):
        return False
    return box[2] > box[0] and box[3] > box[1]


def sightings_by_frame(ticks: list[dict], classes: frozenset | None = None,
                       min_score: float = 0.0) -> tuple[dict, dict]:
    """``({video_frame: [sighting, ...]}, counts)`` for the ticks that wrote a frame.

    ``counts`` is the audit trail: how many ticks wrote a frame, how many of those kept a
    sighting, and how many sightings each filter dropped. It is printed rather than
    summarised, because "how many frames did this run actually yield" is the number a
    reader has to be able to check without re-running anything.

    A ``video_frame`` appearing twice means the file was not written by
    ``VisualNavigator._record`` — that gate advances the index once per perception cycle
    and returns ``None`` on every other tick — so it is refused rather than merged.
    """
    kept: dict = {}
    counts = {"ticks": len(ticks), "recorded": 0, "dropped_class": 0,
              "dropped_score": 0, "dropped_box": 0, "frames_without_a_box": 0}
    for tick in ticks:
        index = tick.get("perception", {}).get("video_frame")
        if index is None:
            continue
        counts["recorded"] += 1
        if index in kept:
            raise Refused(
                f"video_frame {index} appears on two ticks. The recorder advances that "
                f"index once per perception cycle, so this file was not written by "
                f"visual_nav.py --record; refusing rather than guessing which tick owns "
                f"the frame.")
        surviving = []
        for sighting in tick.get("sightings", ()):
            if classes is not None and sighting.get("label") not in classes:
                counts["dropped_class"] += 1
                continue
            score = sighting.get("score")
            if score is None or score < min_score:
                counts["dropped_score"] += 1
                continue
            if not _usable(sighting):
                counts["dropped_box"] += 1
                continue
            surviving.append(dict(sighting, t=tick.get("t")))
        if surviving:
            kept[index] = sorted(surviving, key=lambda s: -s["score"])
        else:
            counts["frames_without_a_box"] += 1
    return kept, counts


# ── Writing the manifest ────────────────────────────────────────────────────
def record_for(prefix: str, index: int, sightings: list[dict], label: str) -> dict:
    """One manifest record: the ``records`` shape, plus everything the join recovered.

    ``box`` is the HIGHEST-SCORING surviving sighting and there is exactly one per frame,
    because both readers of this shape assume that — ``check_manifest.check_unique`` fails
    a repeated key and ``eval_class_agnostic.load_frames`` keys its boxes by image name.
    The rest are kept under ``sightings`` so nothing measured is thrown away.
    """
    best = sightings[0]
    return {
        "image": FRAME_KEY.format(prefix=prefix, index=index),
        "label": label,
        "box": [float(value) for value in best["box"]],
        "video_frame": index,
        "t": best.get("t"),
        "detector_label": best.get("label"),
        "score": best.get("score"),
        "range_m": best.get("range_m"),
        "bearing_rad": best.get("bearing_rad"),
        "range_source": best.get("source"),
        "provenance": "auto",
        "sightings": sightings,
    }


#: Copied out of the run header. An allow-list, not the whole header: the header is a
#: growing set of run parameters and blindly embedding it would make this manifest change
#: shape every time an unrelated flag is added to visual_nav.py.
RUN_KEYS = ("wall_time", "live", "goal", "classes", "confidence", "control_hz",
            "camera", "envelope", "video", "video_raw", "static_prop")


def build_manifest(records: list[dict], *, label: str, run_path: Path, video: Path,
                   header: dict, counts: dict, filters: dict, manifest_path: Path,
                   frames_dir: Path) -> dict:
    """The manifest, with its own provenance and the command that checks it.

    ``count`` and ``boxes`` are the two ``check_manifest.COUNTERS`` keys this shape can
    declare, and they are recomputed there rather than trusted — which is the whole reason
    they are declared at all.
    """
    return {
        "label": label,
        "source": {
            "what": (f"{len(records)} frames auto-labelled from one recorded run. Every "
                     f"box is detector output joined to the raw video by "
                     f"perception.video_frame; see autolabel_run.py."),
            "provenance": PROVENANCE,
            "telemetry": str(run_path),
            "video": str(video),
            "filters": filters,
            "yield": counts,
            "run": {key: header[key] for key in RUN_KEYS if key in header},
            "nuisance_variables": (
                "corridor, lighting, peer pose and operator are NOT recorded by any run — "
                "issue #77 item 4. Pooling several of these manifests does not give a "
                "cross-session split, only a bigger single-session one."),
        },
        "check": f"python3 check_manifest.py {manifest_path} --frames-dir {frames_dir}",
        "count": len(records),
        "boxes": sum(1 for record in records if record["box"] is not None),
        "records": records,
    }


# ── The video ───────────────────────────────────────────────────────────────
def decode_frames(path: Path):
    """Yield ``(index, image)`` for every frame of an MP4, in the order it was written.

    ``cv2`` is imported here rather than at module scope so the join above stays importable
    on an interpreter that has no OpenCV — which is the one the robot's tests run under,
    and the one this directory's other test modules are careful to stay inside.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise Refused(f"cannot open {path} for reading")
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield index, frame
            index += 1
    finally:
        capture.release()


def write_jpeg(path: Path, image) -> None:
    """One frame to disk at the quality the committed manifests' own extractor uses."""
    import cv2

    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise Refused(f"cannot write {path}")


def extract(wanted: dict, video: Path, frames_dir: Path, prefix: str,
            decoder=decode_frames, writer=write_jpeg) -> list[int]:
    """Write every wanted frame out of ``video``; return the indices actually found.

    Decoded in one forward pass rather than seeking per frame: ``CAP_PROP_POS_FRAMES`` on a
    B-frame codec lands on the wrong picture often enough that the index in the filename
    would stop meaning the index in the telemetry, and that is the only thing holding this
    join together.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    found = []
    for index, image in decoder(video):
        if index not in wanted:
            continue
        writer(frames_dir / FRAME_KEY.format(prefix=prefix, index=index), image)
        found.append(index)
    return found


def check_video_choice(video: Path, header: dict, allow_annotated: bool) -> None:
    """Refuse the annotated recording, which has the answer drawn on it.

    Matched on basename because the file is copied off the robot and the header's path is
    the robot's. This can only catch a video that still has the name the run gave it — it
    is a guard against the obvious mistake, not a detector for drawn-on pixels.
    """
    if allow_annotated:
        return
    annotated = header.get("video")
    raw = header.get("video_raw")
    if not annotated or Path(annotated).name != video.name:
        return
    if raw and Path(raw).name == video.name:
        return          # the run wrote both to one path; nothing to warn about
    raise Refused(
        f"{video.name} is the run's ANNOTATED recording (header 'video'). Every detection "
        f"has an orange box drawn round it, at exactly the place the label goes, so a "
        f"model trained on these frames learns to find the box. Re-run with --record-raw "
        f"(issue #77, landed in #84), or pass --allow-annotated if you know these frames "
        f"are not going near a training set.")


def resolve_video(args, header: dict) -> Path:
    """The video to decode: ``--video`` if given, else the raw one the run recorded."""
    if args.video is not None:
        return args.video
    raw = header.get("video_raw")
    if not raw:
        raise Refused(
            "this run recorded no raw video (header 'video_raw' is absent or null), so "
            "there is nothing to join to. Re-run visual_nav.py with --record-raw, or pass "
            "--video explicitly if you have the file under another path.")
    return Path(raw)


# ── CLI ─────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("telemetry", type=Path, help="the run's .jsonl")
    parser.add_argument("--video", type=Path, default=None,
                        help="raw MP4. Default: the run header's 'video_raw'")
    parser.add_argument("--frames-dir", type=Path, required=True,
                        help="where the extracted JPEGs go")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="where the label manifest goes")
    parser.add_argument("--prefix", default=None,
                        help="filename stem for the frames. Default: the telemetry's")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="keep only sightings the detector gave these labels. "
                             "Default: every label")
    parser.add_argument("--label", default="detection",
                        help="the class name written into the manifest, e.g. go2wheel")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="drop sightings below this detector score")
    parser.add_argument("--allow-annotated", action="store_true",
                        help="permit the --record video, which has boxes drawn on it")
    return parser


def main(argv: list[str] | None = None, decoder=decode_frames, writer=write_jpeg) -> int:
    """``decoder`` and ``writer`` are parameters so the tests can drive the whole path
    without OpenCV — every other module in this directory is careful to stay importable on
    the robot's Python, and a test that reaches in and rebinds a module attribute would
    silently stop working the day a default argument captured the original."""
    args = build_parser().parse_args(argv)
    header, ticks = read_run(args.telemetry)
    video = resolve_video(args, header)
    check_video_choice(video, header, args.allow_annotated)
    prefix = args.prefix or args.telemetry.stem
    classes = None if args.classes is None else frozenset(args.classes)

    wanted, counts = sightings_by_frame(ticks, classes, args.min_score)
    found = extract(wanted, video, args.frames_dir, prefix, decoder, writer)
    missing = sorted(set(wanted) - set(found))

    records = [record_for(prefix, index, wanted[index], args.label)
               for index in sorted(found)]
    manifest = build_manifest(
        records, label=args.label, run_path=args.telemetry, video=video, header=header,
        counts=counts, manifest_path=args.manifest, frames_dir=args.frames_dir,
        filters={"classes": None if classes is None else sorted(classes),
                 "min_score": args.min_score})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"{counts['ticks']} ticks, {counts['recorded']} wrote a video frame")
    print(f"  dropped: {counts['dropped_class']} sighting(s) by class, "
          f"{counts['dropped_score']} by score, {counts['dropped_box']} for an unusable box")
    print(f"  {counts['frames_without_a_box']} recorded frame(s) kept no sighting and were "
          f"NOT written — a detector miss is not a peer-free frame")
    print(f"  wrote {len(records)} frame(s) to {args.frames_dir} and "
          f"{args.manifest}")
    print(f"  check with: {manifest['check']}")
    if missing:
        print(f"\nFAILED — {len(missing)} frame(s) the telemetry names are not in "
              f"{video}: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        print("The manifest names only the frames that were written, so it is still "
              "self-consistent; this is a truncated or mismatched recording.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
