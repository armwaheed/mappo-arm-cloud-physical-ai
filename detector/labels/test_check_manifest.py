#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for check_manifest.py, plus two pins on the manifests this repository ships.

The last two tests are the ones that matter operationally: they assert the shipped
manifests are internally consistent and that their scoring denominators are the numbers
the docs quote. A manifest is edited by hand, and the failure mode is not a crash — it is
a recall figure quietly computed over the wrong denominator.

Run: ``python3 test_check_manifest.py``.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_manifest

HERE = Path(os.path.dirname(os.path.abspath(__file__)))


def _frames_manifest(**overrides):
    """Three frames of one clip: one peer with a box, one peer-free, one undecidable."""
    manifest = {
        "label": "go2wheel",
        "present_true": 1,
        "present_false": 1,
        "present_null": 1,
        "boxes": 1,
        "frames": [
            {"clip": "c", "index": 0, "present": True, "box": [10, 20, 30, 40], "split": "test"},
            {"clip": "c", "index": 1, "present": False, "box": None, "split": "test"},
            {"clip": "c", "index": 2, "present": None, "box": None, "split": "select"},
        ],
    }
    manifest.update(overrides)
    return manifest


def _records_manifest(**overrides):
    manifest = {
        "label": "go2wheel",
        "count": 2,
        "records": [
            {"image": "a_0000.jpg", "label": "go2wheel", "box": [1, 2, 3, 4]},
            {"image": "a_0001.jpg", "label": "go2wheel", "box": [5, 6, 7, 8]},
        ],
    }
    manifest.update(overrides)
    return manifest


def _run(manifest, frames_dir=None, split=None):
    """``(exit_code, printed_text)``."""
    out = io.StringIO()
    rows = check_manifest.rows_of(manifest)
    code = check_manifest.report(manifest, rows, frames_dir, split, out=out)
    return code, out.getvalue()


def test_a_consistent_manifest_passes():
    code, text = _run(_frames_manifest())
    assert code == 0, text
    assert "OK" in text


def test_the_frame_key_is_the_one_eval_detector_builds():
    """eval_detector.py:244 builds f"{clip}_{index:03d}.jpg". If this checker built any
    other name it would validate a set of files nothing ever scores, and report a clean
    directory while the real one was missing every frame."""
    rows = check_manifest.rows_of(_frames_manifest())
    assert rows[0].key == "c_000.jpg"
    assert rows[2].key == "c_002.jpg"


def test_a_declared_count_that_disagrees_with_the_records_fails():
    """The counts are hand-written and nothing else recomputes them."""
    for key, wrong in (("present_true", 2), ("present_false", 0),
                       ("present_null", 9), ("boxes", 3)):
        code, text = _run(_frames_manifest(**{key: wrong}))
        assert code == 1, f"{key}={wrong} was accepted:\n{text}"
        assert f"header says {key}={wrong}" in text, text


def test_a_records_manifest_checks_its_own_count():
    code, text = _run(_records_manifest(count=7))
    assert code == 1, text
    assert "header says count=7, records give 2" in text


def test_a_frame_named_twice_in_the_frames_shape_is_reported():
    """In THAT shape presence is declared per frame, so two rows can declare it two
    different ways — and `denominators` would count both."""
    manifest = _frames_manifest()
    manifest["frames"].append(dict(manifest["frames"][0]))
    manifest["present_true"] = 2
    manifest["boxes"] = 2
    code, text = _run(manifest)
    assert code == 1, text
    assert "named more than once" in text and "c_000.jpg" in text


def test_an_inverted_or_empty_box_is_reported():
    """It raises nowhere downstream — it scores IoU 0 and reads as a detector miss."""
    for box in ([30, 20, 10, 40], [10, 20, 10, 40], [10, 40, 30, 20]):
        manifest = _frames_manifest()
        manifest["frames"][0]["box"] = box
        code, text = _run(manifest)
        assert code == 1, f"accepted box {box}:\n{text}"
        assert "no positive extent" in text
    manifest = _frames_manifest()
    manifest["frames"][0]["box"] = [10, 20, 30]
    code, text = _run(manifest)
    assert code == 1, text
    assert "not four numbers" in text


def test_a_named_frame_that_is_absent_is_reported():
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "c_000.jpg").write_bytes(b"")
        (Path(directory) / "c_001.jpg").write_bytes(b"")
        # c_002.jpg deliberately not written.
        code, text = _run(_frames_manifest(), frames_dir=Path(directory))
        assert code == 1, text
        assert "absent from" in text and "c_002.jpg" in text
        assert "1 named-but-absent" in text


def test_a_frame_present_but_unnamed_is_reported():
    """The quiet one. eval_class_agnostic.py builds its negatives as every JPEG the
    manifest does not name, so a stray file is silently scored as peer-free and moves the
    false-alarm rate. A manifest can be perfectly self-consistent and still be scored
    against a denominator nobody chose."""
    with tempfile.TemporaryDirectory() as directory:
        for name in ("c_000.jpg", "c_001.jpg", "c_002.jpg", "stray_999.jpg"):
            (Path(directory) / name).write_bytes(b"")
        code, text = _run(_frames_manifest(), frames_dir=Path(directory))
        assert code == 1, text
        assert "NOT named by the manifest" in text and "stray_999.jpg" in text
        assert "1 present-but-unnamed" in text


def test_a_complete_directory_passes_both_directions():
    with tempfile.TemporaryDirectory() as directory:
        for name in ("c_000.jpg", "c_001.jpg", "c_002.jpg"):
            (Path(directory) / name).write_bytes(b"")
        code, text = _run(_frames_manifest(), frames_dir=Path(directory))
        assert code == 0, text
        assert "0 named-but-absent, 0 present-but-unnamed" in text


def test_without_a_frames_dir_it_says_it_did_not_check_the_frames():
    """A green line that did not earn itself is how this whole class of error survives."""
    code, text = _run(_frames_manifest())
    assert code == 0
    assert "the frames themselves were NOT checked" in text


def test_a_manifest_with_neither_list_is_refused():
    try:
        check_manifest.rows_of({"label": "go2wheel"})
    except SystemExit as error:
        assert "frames" in str(error) and "records" in str(error)
    else:
        raise AssertionError("accepted a manifest with no records")


def test_the_shipped_manifests_are_internally_consistent():
    """Runs the real check over the two files this directory ships."""
    for name in ("peer_crossday_20260820.json", "peer_go2wheel_20260824.json"):
        manifest = json.loads((HERE / name).read_text())
        code, text = _run(manifest)
        assert code == 0, f"{name} failed its own checker:\n{text}"


def test_the_crossday_test_split_denominators_are_the_reported_ones():
    """CROSSDAY.md and every cross-day figure quote these. 136 negatives, not 134: all
    seven clips decode to exactly the frame count this manifest names, verified against
    the clips themselves on 2026-08-26."""
    manifest = json.loads((HERE / "peer_crossday_20260820.json").read_text())
    rows = check_manifest.rows_of(manifest)
    test = check_manifest.denominators(rows, "test")
    assert test == {"rows": 185, "present": 47, "absent": 136, "null": 2, "boxes": 4}, test
    select = check_manifest.denominators(rows, "select")
    assert select == {"rows": 99, "present": 13, "absent": 85, "null": 1, "boxes": 2}, select
    whole = check_manifest.denominators(rows, None)
    assert whole["rows"] == 284 and whole["boxes"] == 6


# ── The records shape allows a second object in a frame; the frames shape does not ──
# 🔴 This check used to reject both, on the stated grounds that "a repeated key is two
# records scoring one file, which double-counts it". That is false for the records shape:
# `eval_class_agnostic.load_frames` builds `boxes[image].append(box)` — a LIST per image —
# and scores recall per frame with `max(iou(box, t) for t in truth)`. The checker was
# stricter than the script it protects, and it would have forced the auto-labelling join
# of issue #77 to throw away every second peer in a frame.
def test_two_boxes_on_one_image_are_a_second_object_and_not_an_error():
    """The consumer handles it correctly, so the checker must not reject it."""
    manifest = _records_manifest()
    manifest["records"].append(
        {"image": "a_0000.jpg", "label": "go2wheel", "box": [40, 50, 60, 70]})
    manifest["count"] = 3
    code, text = _run(manifest)
    assert code == 0, text
    assert "3 rows" in text


def test_the_consumer_really_does_collect_a_list_per_image():
    """The pin for the test above, against the real read site rather than against a belief
    about it. `check_unique` was relaxed BECAUSE of this line; if it ever becomes
    `boxes[image] = box` the relaxation is wrong and this fails."""
    text = (HERE.parent / "eval_class_agnostic.py").read_text()
    assert 'boxes[record["image"]].append(' in text, (
        "eval_class_agnostic no longer accumulates a list of boxes per image, so a "
        "repeated image in a records manifest is no longer safe")
    assert "max(iou(box, t) for t in truth)" in text, (
        "recall is no longer scored per frame over every truth box")


def test_one_frame_declared_two_different_ways_is_an_error_in_the_frames_shape():
    """The reason the two shapes are checked differently, in one case.

    A `records` row is one BOX, so a repeat with a different box is a second object. A
    `frames` row declares that frame's PRESENCE, so a repeat with a different verdict is a
    contradiction — and `denominators`, which is what a reader quotes, would count it
    twice under two answers. Relaxing this shape the way the records shape was relaxed
    would let that through.
    """
    manifest = _frames_manifest()
    manifest["frames"].append({"clip": "c", "index": 0, "present": False,
                               "box": None, "split": "test"})
    manifest["present_false"] = 2
    code, text = _run(manifest)
    assert code == 1, text
    assert "named more than once" in text and "c_000.jpg" in text


def test_the_same_box_twice_on_one_image_is_still_an_error():
    """A copy is a copy in either shape. It does duplicate IoU work and it inflates the
    label tally `eval_class_agnostic` prints beside the rate."""
    manifest = _records_manifest()
    manifest["records"].append(dict(manifest["records"][0]))
    manifest["count"] = 3
    code, text = _run(manifest)
    assert code == 1, text
    assert "duplicated box" in text and "a_0000.jpg" in text


def test_the_shape_is_named_rather_than_inferred_twice():
    """`rows_of` and `check_unique` must agree about which shape they are looking at, and
    a manifest carrying neither list is a refusal, not a silent empty check."""
    assert check_manifest.shape_of(_frames_manifest()) == "frames"
    assert check_manifest.shape_of(_records_manifest()) == "records"
    try:
        check_manifest.shape_of({"label": "go2wheel"})
    except SystemExit:
        pass
    else:
        raise AssertionError("a manifest with no rows at all was accepted")

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"check_manifest: {len(tests)}/{len(tests)} passed")
