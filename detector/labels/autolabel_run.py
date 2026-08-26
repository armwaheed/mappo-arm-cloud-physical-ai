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
        --unlabelled-dir MISSED --classes person --label go2wheel
    python3 check_manifest.py OUT/labels.json --frames-dir OUT      # passes, both directions

ONE RECORD PER BOX, not per frame. Two peers in one frame are two records naming one image,
because that is what the reader does with them: ``eval_class_agnostic.load_frames``
accumulates a list per image name and scores each frame against the best match in it. The
first version of this file kept only the top-scoring sighting per frame, and over the two
committed runs that carry ``sightings`` that discarded 23 of 118 boxes into a key nothing
reads.

⛔ **THESE ARE DETECTOR BOXES, NOT GROUND TRUTH.** Every box here was produced by the same
MobileNet-SSD the robot ran, so this manifest inherits that network's recall exactly: 64%
class-agnostic at the deployed 0.25 threshold on the 2026-08-24 corpus (`detector/README.md`).
It is an auto-label for the frames the detector already found and says NOTHING about the
ones it missed. Two consequences, and neither is negotiable:

* **Not a recall benchmark.** Scoring the same network against labels the same network
  produced measures agreement with itself and returns ~100% by construction. A test set has
  to be hand-labelled, or labelled by something else.
* **A frame with no sighting is not a peer-free frame.** It is a frame nobody has labelled.
  So it is never named by this manifest and never written beside the labelled ones — that
  would hand ``eval_class_agnostic.py`` a false-alarm denominator built out of the
  detector's own misses. ``--unlabelled-dir`` puts those pixels somewhere else, which is
  what issue #77 is actually about ("we keep the labels and discard the pixels"); the two
  directories may not be the same one and this refuses if they are.

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
                       min_score: float = 0.0) -> tuple[dict, list, dict]:
    """``({video_frame: [sighting, ...]}, [video_frame the detector missed], counts)``.

    ``counts`` is the audit trail: how many ticks wrote a frame, how many of those kept a
    sighting, and how many sightings each filter dropped. It is printed rather than
    summarised, because "how many frames did this run actually yield" is the number a
    reader has to be able to check without re-running anything.

    A ``video_frame`` appearing twice means the file was not written by
    ``VisualNavigator._record`` — that gate advances the index once per perception cycle
    and returns ``None`` on every other tick — so it is refused rather than merged.
    """
    kept: dict = {}
    empty: list = []
    # `highest_frame` is the largest index the recorder wrote, which is one less than the
    # number of frames in this run's own recording — `_record` advances one shared counter
    # per perception cycle and never skips. `recorded` counts TICKS, and the two differ when
    # the telemetry tail was lost, so they are not interchangeable. See check_frame_count.
    counts = {"ticks": len(ticks), "recorded": 0, "highest_frame": None,
              "dropped_class": 0, "dropped_score": 0, "dropped_box": 0,
              "frames_without_a_box": 0}
    for tick in ticks:
        index = tick.get("perception", {}).get("video_frame")
        if index is None:
            continue
        counts["recorded"] += 1
        if counts["highest_frame"] is None or index > counts["highest_frame"]:
            counts["highest_frame"] = index
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
            empty.append(index)
            counts["frames_without_a_box"] += 1
    return kept, empty, counts


# ── Writing the manifest ────────────────────────────────────────────────────
def records_for(prefix: str, index: int, sightings: list[dict], label: str) -> list[dict]:
    """ONE RECORD PER SURVIVING SIGHTING, all naming the same image.

    This used to emit one record per FRAME carrying only the highest-scoring sighting, on
    the stated grounds that "both readers of this shape assume that —
    ``check_manifest.check_unique`` fails a repeated key and
    ``eval_class_agnostic.load_frames`` keys its boxes by image name". Only the first half
    was ever true and it is no longer: ``check_unique`` is shape-aware, and ``load_frames``
    builds ``boxes[record["image"]].append(box)`` — a list per image — scored with
    ``max(iou(box, t) for t in truth)``. Two peers in one frame were always readable.

    ⚠️ ITS PREDECESSOR LEFT THIS CALL OPEN — "it changes what every existing consumer of
    this manifest reads, and that is a call for whoever needs the second box". Making it,
    with the number: over the two committed runs that carry ``sightings``, **21.1% of the
    frames with a box hold more than one, and the top-box rule discarded 23 of 118 boxes**
    (on the hero run, 4 of 36) into a ``sightings`` key nothing reads. A detection landing
    on one of those scored as a false alarm against a frame that did contain an object.

    The consumers are two and both were checked: ``check_manifest.report`` accepts it (a
    test drives it end to end) and ``eval_class_agnostic.load_frames`` accumulates. No
    manifest of this shape produced by this script exists in the tree yet, so nothing
    already committed changes meaning.

    Sorted highest score first, so the first record for an image is the one the old shape
    would have emitted and a diff against an older manifest reads.
    """
    image = FRAME_KEY.format(prefix=prefix, index=index)
    return [{
        "image": image,
        "label": label,
        "box": [float(value) for value in sighting["box"]],
        "video_frame": index,
        "t": sighting.get("t"),
        "detector_label": sighting.get("label"),
        "score": sighting.get("score"),
        "range_m": sighting.get("range_m"),
        "bearing_rad": sighting.get("bearing_rad"),
        "range_source": sighting.get("source"),
        "provenance": "auto",
    } for sighting in sightings]


#: Copied out of the run header. An allow-list, not the whole header: the header is a
#: growing set of run parameters and blindly embedding it would make this manifest change
#: shape every time an unrelated flag is added to visual_nav.py.
RUN_KEYS = ("wall_time", "live", "goal", "classes", "confidence", "control_hz",
            "camera", "envelope", "video", "video_raw", "static_prop")


def build_manifest(records: list[dict], *, label: str, run_path: Path, video: Path,
                   header: dict, counts: dict, filters: dict, manifest_path: Path,
                   frames_dir: Path, frame_count_unverified: bool = False,
                   frames_are_annotated: bool = False) -> dict:
    """The manifest, with its own provenance and the command that checks it.

    ``count`` and ``boxes`` are the two ``check_manifest.COUNTERS`` keys this shape can
    declare, and they are recomputed there rather than trusted — which is the whole reason
    they are declared at all.
    """
    source_extra = {}
    if frame_count_unverified:
        # A reader must not have to know which flag was passed. This is the one condition
        # under which the pixels are not certainly the pixels the boxes were measured on.
        source_extra["frame_count_unverified"] = (
            "the video's frame count is not the number of frames this run recorded, and "
            "--allow-frame-count-mismatch was passed. Correct only if the TELEMETRY is "
            "the truncated half; otherwise every box is on the wrong frame.")
    if frames_are_annotated:
        # The other waived refusal, stamped for the same reason: these frames may carry the
        # run's own detection boxes and HUD burned in, and this file is all a later reader
        # of the directory gets.
        source_extra["frames_are_annotated"] = (
            "--allow-annotated was passed, so these frames may carry the run's own "
            "detection boxes and HUD burned into the pixels. Usable for range and aspect "
            "statistics, where a drawn box does not matter; NOT usable as training data — "
            "the label is drawn exactly where the label goes.")
    return {
        "label": label,
        "source": {
            "what": (f"{len(records)} boxes over "
                     f"{len({record['image'] for record in records})} frames auto-labelled "
                     f"from one recorded run. Every box is detector output joined to the "
                     f"raw video by perception.video_frame; see autolabel_run.py."),
            **source_extra,
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
def open_capture(path: Path):
    """The ``cv2.VideoCapture`` for a file, opened and checked.

    ``cv2`` is imported here rather than at module scope so the join above stays importable
    on an interpreter that has no OpenCV — which is the one the robot's tests run under,
    and the one this directory's other test modules are careful to stay inside.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise Refused(f"cannot open {path} for reading")
    return capture


#: ``cv2.CAP_PROP_FRAME_COUNT``, written out as the literal it has been since OpenCV 2.4 so
#: that asking a container its length does not drag ``cv2`` into this module's import. The
#: test module asserts the two agree wherever ``cv2`` IS importable.
CAP_PROP_FRAME_COUNT = 7


def declared_frame_count(path: Path, opener=open_capture) -> int:
    """What the container's header says it holds, or ``0`` if it will not say.

    ONLY EVER USED TO REFUSE, never to accept, because it is a header field a re-encode can
    leave wrong — which is why :func:`extract` still returns the decoded count and
    :func:`check_frame_count` runs again on that. It is asked FIRST because the decoded
    count is only knowable once a directory of possibly-wrong JPEGs already exists under
    right-looking names, and on the committed hero video the header is right: 423.
    """
    capture = opener(path)
    try:
        return max(0, int(capture.get(CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()


def decode_frames(path: Path, opener=open_capture):
    """Yield ``(index, image)`` for every frame of an MP4, in the order it was written.

    The index is this loop's own counter and it is the join key, so ``opener`` is a
    parameter: the counter can then be exercised without OpenCV. It is the one thing in
    this module a test can be wrong about and still look right — every other test injects
    a decoder, so before this the real counter was never executed by anything.
    """
    capture = opener(path)
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


def check_frame_count(video: Path, decoded: int, expected: int, allow: bool,
                      measured: bool = True) -> None:
    """🔴 REFUSE A VIDEO WITH MORE FRAMES THAN THIS RUN RECORDED.

    ONE DIRECTION ONLY, deliberately. A video that is too SHORT is already handled: ``main``
    reports the frames the telemetry names and the decode never reached, exits 1, and leaves
    a manifest that is short rather than wrong. Too LONG was caught by nothing, and it is
    the dangerous direction because it is silent — every index still resolves, so every
    frame is written, every box lands on a picture, and every picture is the wrong one.

    Measured against this script as it stood, on files committed to this repository:

        python3 autolabel_run.py evidence/2026-08-25-peer-runs/hero-run-telemetry.jsonl \
            --video evidence/2026-08-25-peer-runs/hero-clears-peer-on-right.mp4 ...

    wrote **32 auto-labels, exited 0, and passed ``check_manifest.py`` in both
    directions**. That run recorded **58** frames and that file holds **423**, because the
    committed videos are edited hero cuts and not the recorder's output. Nothing in the
    manifest said so, and nothing could have: the frames it names all exist.

    A recording written by ``--record``/``--record-raw`` holds exactly one frame per
    recorded index, because ``VisualNavigator._record`` advances one shared counter and
    ``visual_nav.main``'s ``finally`` releases the writer. So the file holds exactly
    ``max(video_frame) + 1`` frames, and more than that is a different file — or a telemetry
    file whose tail was lost, which is what ``--allow-frame-count-mismatch`` is for.

    ⚠️ ``expected`` is ``max(video_frame) + 1`` and NOT the number of recorded ticks. The two
    are equal for any file the recorder wrote, and they diverge exactly when the telemetry
    is the incomplete half — which is the case this must not misdiagnose as a wrong video.
    """
    if not decoded or decoded <= expected:
        return
    holds = "holds" if measured else "declares"
    if allow:
        print(f"⚠️  {video.name} {holds} {decoded} frames and this run recorded {expected}. "
              f"--allow-frame-count-mismatch was passed, so the pixels may not be the "
              f"pixels these boxes were measured on.")
        return
    raise Refused(
        f"REFUSING: {video.name} {holds} {decoded} frames and this run recorded "
        f"{expected}. "
        f"A recording written by --record/--record-raw holds exactly one frame per "
        f"recorded index, so this is a different file: an edited cut, a re-encode, or "
        f"another run. Every index would still resolve and every box would land on the "
        f"wrong frame — measured, on this repository's own committed hero video, as 32 "
        f"auto-labels that passed check_manifest.py in both directions and meant nothing. "
        f"Pass --allow-frame-count-mismatch if the TELEMETRY is the truncated half; it is "
        f"recorded in the manifest.")


def extract(wanted: dict, video: Path, frames_dir: Path, prefix: str,
            decoder=decode_frames, writer=write_jpeg,
            unlabelled=(), unlabelled_dir: Path | None = None) -> tuple[list, list, int]:
    """Write the wanted frames out of ``video``; return
    ``(indices found, unlabelled indices found, frames decoded)``.

    Decoded in one forward pass rather than seeking per frame: ``CAP_PROP_POS_FRAMES`` on a
    B-frame codec lands on the wrong picture often enough that the index in the filename
    would stop meaning the index in the telemetry, and that is the only thing holding this
    join together.

    ``unlabelled_dir`` is a SEPARATE directory or nothing, never ``frames_dir``;
    :func:`resolve_unlabelled_dir` enforces that and its docstring says why.

    The decoded count comes back because it is the only honest measure of how long the file
    is — a container's declared count is a header field a re-encode can leave wrong — and
    :func:`check_frame_count` needs it to tell this run's recording from another file.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    if unlabelled_dir is not None:
        unlabelled_dir.mkdir(parents=True, exist_ok=True)
    unlabelled = set(unlabelled)
    found, found_unlabelled, decoded = [], [], 0
    for index, image in decoder(video):
        decoded = index + 1
        if index in wanted:
            writer(frames_dir / FRAME_KEY.format(prefix=prefix, index=index), image)
            found.append(index)
        elif unlabelled_dir is not None and index in unlabelled:
            writer(unlabelled_dir / FRAME_KEY.format(prefix=prefix, index=index), image)
            found_unlabelled.append(index)
    return found, found_unlabelled, decoded


def resolve_unlabelled_dir(unlabelled_dir: Path | None, frames_dir: Path) -> Path | None:
    """``unlabelled_dir``, refused if it is the labelled directory under another spelling.

    ⛔ THIS IS THE ONE WAY THE FLAG CAN POISON WHAT IT EXISTS TO PROTECT.
    ``eval_class_agnostic.load_frames`` builds its peer-free set as every JPEG in a
    directory that the manifest does not name. Frames the detector found nothing in are
    unlabelled, NOT peer-free — at 64% class-agnostic recall roughly a third of them still
    hold the object — so writing them beside the labelled ones files the detector's own
    misses into its own false-alarm denominator. #89 has already been burned once by a
    negative set that was not what it said it was.

    Keeping the pixels at all is the point of issue #77 — "the bug is that we keep the
    labels and discard the pixels" — and these are the frames a human should label next.
    They just cannot live in the scored directory, and pointing both flags at one path is
    one tab-completion away. Compared as RESOLVED paths, because ``Path`` collapses neither
    ``..`` nor a symlink and a plain ``==`` would pass ``OUT/../OUT``.
    """
    if unlabelled_dir is None:
        return None
    if unlabelled_dir.resolve() == frames_dir.resolve():
        raise Refused(
            f"--unlabelled-dir and --frames-dir are the same directory ({frames_dir}). "
            f"eval_class_agnostic.py scores every JPEG the manifest does not name as a "
            f"peer-free frame, so putting the detector's misses in there files them as its "
            f"own false alarms. Give them a directory of their own, or leave the flag off "
            f"and they are dropped.")
    return unlabelled_dir


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
    parser.add_argument("--unlabelled-dir", type=Path, default=None,
                        help="ALSO write the recorded frames the detector found nothing "
                             "in, to this directory. Must not be --frames-dir: they are "
                             "unlabelled, not peer-free. Default: they are dropped")
    parser.add_argument("--allow-frame-count-mismatch", action="store_true",
                        help="accept a video whose frame count is not the number of frames "
                             "the run recorded. Only correct when the TELEMETRY is the "
                             "truncated half; stamped into the manifest")
    parser.add_argument("--allow-annotated", action="store_true",
                        help="permit the --record video, which has boxes drawn on it")
    return parser


def main(argv: list[str] | None = None, decoder=decode_frames, writer=write_jpeg,
         counter=declared_frame_count) -> int:
    """``decoder``, ``writer`` and ``counter`` are parameters so the tests drive the whole path
    without OpenCV — every other module in this directory is careful to stay importable on
    the robot's Python, and a test that reaches in and rebinds a module attribute would
    silently stop working the day a default argument captured the original."""
    args = build_parser().parse_args(argv)
    header, ticks = read_run(args.telemetry)
    video = resolve_video(args, header)
    check_video_choice(video, header, args.allow_annotated)
    unlabelled_dir = resolve_unlabelled_dir(args.unlabelled_dir, args.frames_dir)
    prefix = args.prefix or args.telemetry.stem
    classes = None if args.classes is None else frozenset(args.classes)

    wanted, empty, counts = sightings_by_frame(ticks, classes, args.min_score)
    expected = 0 if counts["highest_frame"] is None else counts["highest_frame"] + 1
    # ASKED TWICE, ON PURPOSE. The decoded count below is the trustworthy one — a
    # container's declared count is a header field a re-encode can leave wrong — but it is
    # only knowable once a directory of possibly-wrong JPEGs exists under right-looking
    # names. So the header is asked first and used ONLY to refuse; on the committed hero
    # video it is right (423) and nothing reaches the disk.
    if not args.allow_frame_count_mismatch:
        check_frame_count(video, counter(video), expected, False, measured=False)

    found, found_unlabelled, decoded = extract(
        wanted, video, args.frames_dir, prefix, decoder, writer,
        unlabelled=empty, unlabelled_dir=unlabelled_dir)
    # BEFORE the manifest, because a directory of wrong JPEGs with no manifest beside it
    # reads as a failed run rather than as a corpus.
    check_frame_count(video, decoded, expected, args.allow_frame_count_mismatch)
    missing = sorted(set(wanted) - set(found))

    records = [record for index in sorted(found)
               for record in records_for(prefix, index, wanted[index], args.label)]
    manifest = build_manifest(
        records, label=args.label, run_path=args.telemetry, video=video, header=header,
        counts=counts, manifest_path=args.manifest, frames_dir=args.frames_dir,
        filters={"classes": None if classes is None else sorted(classes),
                 "min_score": args.min_score},
        frame_count_unverified=(args.allow_frame_count_mismatch and decoded > expected),
        frames_are_annotated=args.allow_annotated)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"{counts['ticks']} ticks, {counts['recorded']} wrote a video frame")
    print(f"  dropped: {counts['dropped_class']} sighting(s) by class, "
          f"{counts['dropped_score']} by score, {counts['dropped_box']} for an unusable box")
    if unlabelled_dir is None:
        print(f"  {counts['frames_without_a_box']} recorded frame(s) kept no sighting and "
              f"were NOT written — a detector miss is not a peer-free frame")
    else:
        print(f"  {len(found_unlabelled)} recorded frame(s) kept no sighting -> "
              f"{unlabelled_dir}, NOT named by the manifest and NOT beside the labelled "
              f"ones — a detector miss is not a peer-free frame")
    print(f"  wrote {len(records)} box(es) over {len(found)} frame(s) to "
          f"{args.frames_dir} and {args.manifest}")
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
