#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for autolabel_run.py — the join from a recorded run to a label manifest.

Two of these are the ones that matter, and both are about a manifest making a claim it
cannot support:

* ``test_a_frame_the_detector_found_nothing_in_is_neither_written_nor_named`` pins the
  refusal that makes this honest. ``eval_class_agnostic.py`` builds its negative set as
  *every JPEG the manifest does not name*, so extracting a frame the detector missed would
  silently file that frame as peer-free — the detector's own recall failures becoming its
  own false-alarm denominator. #89 has already been burned once by a negative set that was
  not what it said it was.
* ``test_the_manifest_this_writes_passes_check_manifest_in_both_directions`` runs the real
  checker over the real output. Asserting the shape by eye is how a manifest ends up
  self-consistent and unreadable by the thing that has to read it.

The rest are the filters and the refusals. ``cv2`` is never imported: the decoder and the
JPEG writer are injected, which is also the only way these run on the robot's Python.

Run: ``python3 test_autolabel_run.py``.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autolabel_run
import check_manifest

RAW = "/home/unitree/run-raw.mp4"
ANNOTATED = "/home/unitree/run.mp4"


def _sighting(**overrides):
    sighting = {"label": "person", "score": 0.5, "source": "height",
                "range_m": 1.27, "bearing_rad": -0.41,
                "box": [100.0, 50.0, 300.0, 500.0]}
    sighting.update(overrides)
    return sighting


def _tick(video_frame, sightings=(), t=0.0):
    return {"type": "tick", "t": t,
            "perception": {"seq": 1, "video_frame": video_frame, "detect_ms": 160.0},
            "sightings": list(sightings)}


def _run_file(directory, ticks, header=None):
    """Write a telemetry JSONL and return its path."""
    path = Path(directory) / "run.jsonl"
    lines = [json.dumps(header if header is not None
                        else {"type": "header", "video": ANNOTATED, "video_raw": RAW})]
    lines += [json.dumps(tick) for tick in ticks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _Video:
    """A decoder over a fixed number of frames, and a writer that really makes files.

    The files are empty: nothing downstream of this join reads a pixel — ``check_manifest``
    globs names and ``eval_class_agnostic`` opens them itself — so a real encoder here would
    buy nothing and would cost this suite its "needs neither cv2 nor numpy" property.
    """

    def __init__(self, frames: int):
        self.frames = frames
        self.written: list[str] = []

    def decode(self, _path):
        for index in range(self.frames):
            yield index, f"frame-{index}"

    def write(self, path: Path, image):
        Path(path).write_bytes(b"")
        self.written.append((Path(path).name, image))


def _extract(wanted, video, frames_dir, prefix="run", frames=8):
    video_io = _Video(frames)
    # `extract` returns (indices found, frames decoded) — the second is what
    # `check_frame_count` needs to tell this run's recording from another file.
    found, _decoded = autolabel_run.extract(wanted, Path(video), Path(frames_dir), prefix,
                                            decoder=video_io.decode, writer=video_io.write)
    return found, video_io


# ── The join ────────────────────────────────────────────────────────────────
def test_the_frame_written_is_the_one_at_that_video_frame_index():
    """The whole join. ``video_frame`` is an index into the recorder's write order, and
    the decoder yields in that same order, so frame *n* of the MP4 is the frame the tick
    with ``video_frame == n`` saw. The index is carried into the FILENAME so a reader can
    check the join without the manifest."""
    with tempfile.TemporaryDirectory() as directory:
        wanted = {3: [_sighting()], 5: [_sighting()]}
        found, video = _extract(wanted, "raw.mp4", directory)
        assert found == [3, 5], found
        assert [name for name, _ in video.written] == ["run_0003.jpg", "run_0005.jpg"]
        assert [image for _, image in video.written] == ["frame-3", "frame-5"]


def test_a_frame_the_detector_found_nothing_in_is_neither_written_nor_named():
    """A detector miss is not a peer-free frame, and this is where that is enforced.

    ``eval_class_agnostic.load_frames`` files every JPEG the manifest does not name as a
    negative. Extract an unlabelled frame and it joins the false-alarm denominator on the
    strength of the same network having missed it."""
    with tempfile.TemporaryDirectory() as directory:
        ticks = [_tick(0, [_sighting()]), _tick(1, []), _tick(2, [])]
        wanted, counts = autolabel_run.sightings_by_frame(ticks)
        assert sorted(wanted) == [0], wanted
        assert counts["frames_without_a_box"] == 2, counts
        found, video = _extract(wanted, "raw.mp4", directory)
        assert [name for name, _ in video.written] == ["run_0000.jpg"]
        records = [autolabel_run.record_for("run", i, wanted[i], "go2wheel") for i in found]
        assert [r["image"] for r in records] == ["run_0000.jpg"]


def test_ticks_that_wrote_no_frame_are_skipped():
    """The recorder advances once per PERCEPTION cycle and the controller runs faster, so
    most ticks of a healthy run carry ``video_frame: null`` and have no pixels to join to."""
    ticks = [_tick(None, [_sighting()]), _tick(4, [_sighting()])]
    wanted, counts = autolabel_run.sightings_by_frame(ticks)
    assert sorted(wanted) == [4], wanted
    assert counts["ticks"] == 2 and counts["recorded"] == 1, counts


def test_a_video_frame_on_two_ticks_is_refused():
    """``VisualNavigator._record`` returns an index at most once, so a repeat means the
    file was not written by it. Merging two ticks' sightings onto one frame would invent a
    label; refusing says which assumption broke."""
    ticks = [_tick(2, [_sighting()]), _tick(2, [_sighting()])]
    try:
        autolabel_run.sightings_by_frame(ticks)
    except autolabel_run.Refused as refusal:
        assert "video_frame 2" in str(refusal), refusal
    else:
        raise AssertionError("a repeated video_frame should be refused")


# ── What survives the filters ───────────────────────────────────────────────
def test_the_highest_scoring_sighting_becomes_the_box_and_the_others_are_kept():
    """One box per image, because both readers of this shape assume it —
    ``check_manifest.check_unique`` fails a repeated key and ``eval_class_agnostic``
    keys its boxes by image name. Nothing measured is thrown away for that."""
    ticks = [_tick(0, [_sighting(score=0.3, box=[1.0, 1.0, 2.0, 2.0]),
                       _sighting(score=0.9, box=[3.0, 3.0, 9.0, 9.0])])]
    wanted, _ = autolabel_run.sightings_by_frame(ticks)
    record = autolabel_run.record_for("run", 0, wanted[0], "go2wheel")
    assert record["box"] == [3.0, 3.0, 9.0, 9.0], record["box"]
    assert record["score"] == 0.9
    assert len(record["sightings"]) == 2
    assert [s["score"] for s in record["sightings"]] == [0.9, 0.3]


def test_the_record_carries_the_range_the_bearing_and_which_prior_produced_it():
    """The reason this manifest is worth building at all. A hand-drawn box is a box; these
    carry the range and the prior that made it, which is what makes a run usable for range
    and aspect statistics."""
    ticks = [_tick(0, [_sighting(range_m=2.5, bearing_rad=0.75, source="width")], t=4.25)]
    wanted, _ = autolabel_run.sightings_by_frame(ticks)
    record = autolabel_run.record_for("run", 0, wanted[0], "go2wheel")
    assert record["range_m"] == 2.5
    assert record["bearing_rad"] == 0.75
    assert record["range_source"] == "width"
    assert record["detector_label"] == "person"
    assert record["t"] == 4.25
    assert record["video_frame"] == 0


def test_every_record_says_it_was_auto_labelled():
    """A reader who opens one record must not have to find the docstring to learn that a
    detector drew this box. The manifest header carries the caveat in full."""
    ticks = [_tick(0, [_sighting()])]
    wanted, _ = autolabel_run.sightings_by_frame(ticks)
    assert autolabel_run.record_for("run", 0, wanted[0], "go2wheel")["provenance"] == "auto"
    manifest = autolabel_run.build_manifest(
        [], label="go2wheel", run_path=Path("r.jsonl"), video=Path("v.mp4"), header={},
        counts={}, filters={}, manifest_path=Path("m.json"), frames_dir=Path("F"))
    assert "NOT ground truth" in manifest["source"]["provenance"]
    assert "recall" in manifest["source"]["provenance"]


def test_a_class_filter_and_a_score_floor_drop_sightings_and_say_how_many():
    """Both counted rather than silently applied: how many frames a run yielded, and why
    the rest went, is the number a reader has to be able to check."""
    ticks = [_tick(0, [_sighting(label="person", score=0.9),
                       _sighting(label="chair", score=0.9),
                       _sighting(label="person", score=0.1)])]
    wanted, counts = autolabel_run.sightings_by_frame(ticks, frozenset({"person"}), 0.25)
    assert [s["score"] for s in wanted[0]] == [0.9]
    assert counts["dropped_class"] == 1 and counts["dropped_score"] == 1, counts


def test_a_box_without_positive_extent_is_dropped_rather_than_written():
    """``check_manifest.check_boxes`` fails on one, and downstream it scores IoU 0 against
    every prediction and reads as a detector that missed. A ``null`` coordinate is the same
    class: ``telemetry._finite`` writes one for any value JSON cannot carry."""
    ticks = [_tick(0, [_sighting(box=[10.0, 10.0, 10.0, 40.0])]),
             _tick(1, [_sighting(box=[10.0, 10.0, None, 40.0])]),
             _tick(2, [_sighting(box=[10.0, 10.0])])]
    wanted, counts = autolabel_run.sightings_by_frame(ticks)
    assert wanted == {}, wanted
    assert counts["dropped_box"] == 3, counts


# ── Refusals ────────────────────────────────────────────────────────────────
def test_the_annotated_recording_is_refused_by_name():
    """``--record`` draws an orange box round every detection, at exactly the place the
    label goes. Training on those frames is label leakage in its purest form — the model
    learns to find the rectangle — and it is what issue #77 was reopened about."""
    header = {"type": "header", "video": ANNOTATED, "video_raw": RAW}
    try:
        autolabel_run.check_video_choice(Path("/tmp/run.mp4"), header, False)
    except autolabel_run.Refused as refusal:
        assert "ANNOTATED" in str(refusal) and "--record-raw" in str(refusal), refusal
    else:
        raise AssertionError("the annotated video should be refused")
    autolabel_run.check_video_choice(Path("/tmp/run.mp4"), header, True)
    autolabel_run.check_video_choice(Path("/tmp/run-raw.mp4"), header, False)


def test_a_run_that_recorded_no_raw_video_is_refused_and_says_which_flag():
    """Every run before #84 is this one. The refusal has to name ``--record-raw``, because
    the answer is to re-record, not to point the tool somewhere else."""
    parser = autolabel_run.build_parser()
    args = parser.parse_args(["r.jsonl", "--frames-dir", "F", "--manifest", "m.json"])
    try:
        autolabel_run.resolve_video(args, {"video": ANNOTATED})
    except autolabel_run.Refused as refusal:
        assert "--record-raw" in str(refusal), refusal
    else:
        raise AssertionError("a run with no raw video should be refused")
    assert autolabel_run.resolve_video(args, {"video_raw": RAW}) == Path(RAW)


def test_a_run_with_no_ticks_is_refused():
    """A header-only file is a run that died before its first tick, not an empty dataset."""
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [])
        try:
            autolabel_run.read_run(path)
        except autolabel_run.Refused as refusal:
            assert "no ticks" in str(refusal), refusal
        else:
            raise AssertionError("a file with no ticks should be refused")


def test_a_truncated_last_line_costs_one_tick_and_not_the_run():
    """The writer flushes every line, but a run that ends on a power cut can still lose
    its last one. Refusing the whole file over its final byte throws away the run this
    exists to salvage."""
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_tick(0, [_sighting()])])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"type": "tick", "t": 0.2, "percep')
        with redirect_stdout(io.StringIO()):
            _header, ticks = autolabel_run.read_run(path)
        assert len(ticks) == 1, ticks


# ── End to end ──────────────────────────────────────────────────────────────
def test_the_manifest_this_writes_passes_check_manifest_in_both_directions():
    """The contract. Named-but-absent shrinks the positive denominator; present-but-unnamed
    is silently ADDED to the negative one by ``eval_class_agnostic.py``. A manifest can be
    perfectly self-consistent and still be scored against the wrong denominator, so the
    real checker is run over the real output rather than the shape being eyeballed."""
    with tempfile.TemporaryDirectory() as directory:
        frames_dir = Path(directory) / "frames"
        ticks = [_tick(0, [_sighting()]), _tick(1, []), _tick(3, [_sighting(score=0.8)])]
        wanted, counts = autolabel_run.sightings_by_frame(ticks)
        found, _ = _extract(wanted, "raw.mp4", frames_dir)
        records = [autolabel_run.record_for("run", i, wanted[i], "go2wheel") for i in found]
        manifest = autolabel_run.build_manifest(
            records, label="go2wheel", run_path=Path("run.jsonl"), video=Path("raw.mp4"),
            header={"live": True, "confidence": 0.25}, counts=counts, filters={},
            manifest_path=Path(directory) / "m.json", frames_dir=frames_dir)

        out = io.StringIO()
        code = check_manifest.report(manifest, check_manifest.rows_of(manifest),
                                     frames_dir, None, out=out)
        assert code == 0, out.getvalue()
        assert "0 named-but-absent, 0 present-but-unnamed" in out.getvalue()
        assert manifest["count"] == 2 and manifest["boxes"] == 2
        assert manifest["source"]["run"] == {"live": True, "confidence": 0.25}


def test_a_frame_the_telemetry_names_but_the_video_lacks_fails_the_run():
    """A recording that stopped early, or the wrong file. The manifest still names only
    what was written — it is not wrong, it is short — and the exit code is what says so."""
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_tick(0, [_sighting()]), _tick(9, [_sighting()])])
        video = _Video(4)
        argv = [str(path), "--video", RAW, "--frames-dir", str(Path(directory) / "f"),
                "--manifest", str(Path(directory) / "m.json"), "--prefix", "run"]
        out = io.StringIO()
        with redirect_stdout(out):
            code = autolabel_run.main(argv, decoder=video.decode, writer=video.write)
        assert code == 1, out.getvalue()
        assert "[9]" in out.getvalue(), out.getvalue()
        manifest = json.loads((Path(directory) / "m.json").read_text())
        assert [r["image"] for r in manifest["records"]] == ["run_0000.jpg"]


# ── A video that is not this run's own recording ────────────────────────────
# 🔴 FOUND BY RUNNING THIS SCRIPT AGAINST FILES ALREADY IN THIS REPOSITORY. Before the
# check these tests pin:
#
#   python3 autolabel_run.py evidence/2026-08-25-peer-runs/hero-run-telemetry.jsonl \
#       --video evidence/2026-08-25-peer-runs/hero-clears-peer-on-right.mp4 ...
#
# wrote 32 auto-labels, exited 0, and passed `check_manifest.py` in BOTH directions. That
# run recorded 58 frames; that file holds 423, because the committed hero videos are edited
# cuts and not the recorder's output. Every index resolved, every box landed on a frame, and
# every frame was the wrong one.
def test_a_video_longer_than_the_run_recorded_is_refused():
    """The silent direction. Too SHORT already fails with the indices named; too LONG left
    a manifest that was self-consistent, checkable, and meaningless."""
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_tick(0, [_sighting()]), _tick(1, [_sighting()])])
        video = _Video(40)                       # the run recorded 2
        argv = [str(path), "--video", RAW, "--frames-dir", str(Path(directory) / "f"),
                "--manifest", str(Path(directory) / "m.json"), "--prefix", "run"]
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                autolabel_run.main(argv, decoder=video.decode, writer=video.write)
        except autolabel_run.Refused as refusal:
            assert "holds 40 frames and this run recorded 2" in str(refusal), refusal
        else:
            raise AssertionError("it joined against a video with the wrong frame count")
        assert not (Path(directory) / "m.json").exists(), (
            "a manifest was written beside pixels that are not this run's")


def test_the_expected_count_is_the_highest_index_and_not_the_tick_count():
    """⚠️ THE TRAP IN THE OBVIOUS SPELLING, and it would have broken the case above it.

    A truncated telemetry file has FEWER recorded ticks than the recording has frames, so
    counting ticks would call a perfectly good recording "too long". The recorder advances
    one contiguous index per written frame, so the file holds `max(video_frame) + 1` frames
    and that is the only number to compare against. Here two ticks name indices 0 and 5 —
    what a lost tail looks like — and a six-frame video is exactly right.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_tick(0, [_sighting()]), _tick(5, [_sighting()])])
        video = _Video(6)
        argv = [str(path), "--video", RAW, "--frames-dir", str(Path(directory) / "f"),
                "--manifest", str(Path(directory) / "m.json"), "--prefix", "run"]
        out = io.StringIO()
        with redirect_stdout(out):
            code = autolabel_run.main(argv, decoder=video.decode, writer=video.write)
        assert code == 0, out.getvalue()
        assert [r["video_frame"] for r in
                json.loads((Path(directory) / "m.json").read_text())["records"]] == [0, 5]


def test_a_short_video_still_fails_the_old_way_rather_than_being_refused():
    """The new check adds ONE direction and must not take over the other. A recording that
    stopped early leaves a manifest that is short rather than wrong, exits 1, and names the
    frames it could not reach — which is more useful than a refusal, because the frames it
    did write are real."""
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_tick(0, [_sighting()]), _tick(9, [_sighting()])])
        video = _Video(4)
        argv = [str(path), "--video", RAW, "--frames-dir", str(Path(directory) / "f"),
                "--manifest", str(Path(directory) / "m.json"), "--prefix", "run"]
        out = io.StringIO()
        with redirect_stdout(out):
            code = autolabel_run.main(argv, decoder=video.decode, writer=video.write)
        assert code == 1, out.getvalue()
        assert "[9]" in out.getvalue(), out.getvalue()
        assert (Path(directory) / "m.json").exists()


def test_the_override_is_stamped_into_the_manifest():
    """A truncated telemetry file with a complete video is a real case, so the escape hatch
    is real — and a reader of the manifest must not have to know which flag was passed."""
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_tick(0, [_sighting()])])
        video = _Video(40)
        argv = [str(path), "--video", RAW, "--frames-dir", str(Path(directory) / "f"),
                "--manifest", str(Path(directory) / "m.json"), "--prefix", "run",
                "--allow-frame-count-mismatch"]
        out = io.StringIO()
        with redirect_stdout(out):
            code = autolabel_run.main(argv, decoder=video.decode, writer=video.write)
        assert code == 0, out.getvalue()
        assert "holds 40 frames" in out.getvalue(), out.getvalue()
        source = json.loads((Path(directory) / "m.json").read_text())["source"]
        assert "frame_count_unverified" in source, source


def test_a_correct_recording_is_not_stamped():
    """The positive control for the test above. A stamp that is always there says nothing,
    and this repository has shipped a gate that could never fire."""
    with tempfile.TemporaryDirectory() as directory:
        path = _run_file(directory, [_tick(0, [_sighting()]), _tick(1, [_sighting()])])
        video = _Video(2)
        argv = [str(path), "--video", RAW, "--frames-dir", str(Path(directory) / "f"),
                "--manifest", str(Path(directory) / "m.json"), "--prefix", "run",
                "--allow-frame-count-mismatch"]
        out = io.StringIO()
        with redirect_stdout(out):
            assert autolabel_run.main(argv, decoder=video.decode,
                                      writer=video.write) == 0, out.getvalue()
        source = json.loads((Path(directory) / "m.json").read_text())["source"]
        assert "frame_count_unverified" not in source, source
        assert "holds" not in out.getvalue()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"autolabel_run: {len(tests)}/{len(tests)} passed")
