#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a label manifest against the frames it names, before anything scores against it.

Why this exists, concretely. `peer_go2wheel_20260824.json` recorded its `source_dir` as a
scratchpad path that no longer existed, and that one dead string was read as *the corpus is
gone* — 2,800 frames written off in an issue before anyone looked at the machine the
training had run on. `CROSSDAY.md` said "every frame is in this repository already, as
video" while the clips sat on an unmerged branch. Neither is a subtle bug. Both are a
manifest asserting something about the world that nothing checked.

Three classes of error it catches, all of which move a reported number:

1. **Declared counts that disagree with the records.** `present_true` and friends are what
   a reader quotes; they are written by hand and nothing recomputes them.
2. **Named frames that are not there.** Missing positives shrink recall's denominator.
3. **Frames present but NOT named.** This one is the quiet one. `eval_class_agnostic.py`
   builds its negative set as *every JPEG in the directory that the manifest does not
   name*, so one stray file silently becomes a peer-free frame and moves the false-alarm
   rate. A manifest can be perfectly self-consistent and still be scored against the wrong
   denominator.

Two manifest shapes are read, because the repo has two:

* ``{"records": [{"image", "label", "box"}]}``  — one file name per record, boxes only.
* ``{"frames":  [{"clip", "index", "present", "box", "split"}]}`` — presence over whole
  clips; the frame file is ``<clip>_<index>03d.jpg``, which is what ``eval_detector.py``
  builds and therefore what this must build too.

Usage::

    python3 check_manifest.py peer_crossday_20260820.json
    python3 check_manifest.py peer_crossday_20260820.json --frames-dir XDAY --split test
    python3 check_manifest.py peer_go2wheel_20260824.json --frames-dir PEERCAP

Exits non-zero if any check fails, so it can gate a scoring run. With no ``--frames-dir``
it checks only what the file can prove about itself, and says so rather than reporting a
clean bill of health it did not earn.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: What ``eval_detector.py:244`` builds. Duplicated deliberately: if that convention ever
#: changes, this checker must fail loudly rather than validate a different set of names.
FRAME_KEY = "{clip}_{index:03d}.jpg"

#: Header keys that declare a count, mapped to how to recount them from the records. A
#: manifest is only checked against the ones it actually declares.
COUNTERS = {
    "count": lambda rows: len(rows),
    "frames_count": lambda rows: len(rows),
    "present_true": lambda rows: sum(1 for r in rows if r.present is True),
    "present_false": lambda rows: sum(1 for r in rows if r.present is False),
    "present_null": lambda rows: sum(1 for r in rows if r.present is None),
    "boxes": lambda rows: sum(1 for r in rows if r.box is not None),
}


class Row:
    """One manifest entry, normalised across the two shapes."""

    __slots__ = ("box", "key", "present", "split")

    def __init__(self, key: str, present, box, split: str | None) -> None:
        self.key = key
        self.present = present
        self.box = box
        self.split = split

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Row({self.key!r}, present={self.present!r}, split={self.split!r})"


def rows_of(manifest: dict) -> list[Row]:
    """Normalise either manifest shape into ``Row`` objects.

    ``records`` entries carry a box and nothing else, so presence is inferred from the box
    being there — that is exactly what ``eval_class_agnostic.load_frames`` does with them,
    and inventing a different rule here would validate a set nothing scores.
    """
    if "frames" in manifest:
        rows = []
        for entry in manifest["frames"]:
            key = FRAME_KEY.format(clip=entry["clip"], index=entry["index"])
            rows.append(Row(key, entry.get("present"), entry.get("box"), entry.get("split")))
        return rows
    if "records" in manifest:
        return [Row(entry["image"], entry.get("box") is not None, entry.get("box"), None)
                for entry in manifest["records"]]
    raise SystemExit("manifest has neither a 'frames' nor a 'records' list")


def check_counts(manifest: dict, rows: list[Row]) -> list[str]:
    """Every declared count, recomputed. Silent when a manifest declares none."""
    problems = []
    for key, recount in COUNTERS.items():
        if key not in manifest:
            continue
        declared, actual = manifest[key], recount(rows)
        if declared != actual:
            problems.append(f"header says {key}={declared}, records give {actual}")
    return problems


def check_unique(rows: list[Row]) -> list[str]:
    """A repeated key is two records scoring one file, which double-counts it."""
    seen, repeated = set(), []
    for row in rows:
        if row.key in seen:
            repeated.append(row.key)
        seen.add(row.key)
    if not repeated:
        return []
    shown = ", ".join(sorted(set(repeated))[:5])
    return [f"{len(set(repeated))} frame(s) named more than once: {shown}"]


def check_boxes(rows: list[Row]) -> list[str]:
    """A box must be four numbers with positive extent.

    An inverted or zero-area box does not raise anywhere downstream — it silently scores
    IoU 0 against every prediction, and reads as a detector that missed.
    """
    problems = []
    for row in rows:
        if row.box is None:
            continue
        if not isinstance(row.box, (list, tuple)) or len(row.box) != 4:
            problems.append(f"{row.key}: box is not four numbers: {row.box!r}")
            continue
        x0, y0, x1, y1 = row.box
        if x1 <= x0 or y1 <= y0:
            problems.append(f"{row.key}: box has no positive extent: {row.box!r}")
    return problems


def check_files(rows: list[Row], frames_dir: Path) -> tuple[list[str], list[str]]:
    """``(missing, extra)`` — named-but-absent, and present-but-unnamed.

    Both directions matter and they fail differently: missing frames shrink the positive
    denominator, extra ones are silently *added* to the negative denominator by
    ``eval_class_agnostic.py``.
    """
    if not frames_dir.is_dir():
        raise SystemExit(f"--frames-dir {frames_dir} is not a directory")
    on_disk = {path.name for path in frames_dir.glob("*.jpg")}
    named = {row.key for row in rows}
    return sorted(named - on_disk), sorted(on_disk - named)


def denominators(rows: list[Row], split: str | None) -> dict:
    """The numbers a scoring run divides by, printed so they cannot be assumed."""
    chosen = [r for r in rows if split is None or r.split == split]
    return {
        "rows": len(chosen),
        "present": sum(1 for r in chosen if r.present is True),
        "absent": sum(1 for r in chosen if r.present is False),
        "null": sum(1 for r in chosen if r.present is None),
        "boxes": sum(1 for r in chosen if r.box is not None),
    }


def report(manifest: dict, rows: list[Row], frames_dir: Path | None,
           split: str | None, out=sys.stdout) -> int:
    """Run every check. Returns the process exit code."""
    problems = check_counts(manifest, rows) + check_unique(rows) + check_boxes(rows)
    missing: list[str] = []
    extra: list[str] = []
    if frames_dir is not None:
        missing, extra = check_files(rows, frames_dir)
        for name in missing[:10]:
            problems.append(f"named by the manifest, absent from {frames_dir}: {name}")
        if len(missing) > 10:
            problems.append(f"...and {len(missing) - 10} more missing")
        for name in extra[:10]:
            problems.append(
                f"present in {frames_dir}, NOT named by the manifest: {name} "
                f"(eval_class_agnostic.py would score it as peer-free)")
        if len(extra) > 10:
            problems.append(f"...and {len(extra) - 10} more unnamed")

    counts = denominators(rows, split)
    scope = f"split {split!r}" if split else "whole manifest"
    print(f"{scope}: {counts['rows']} rows — present {counts['present']}, "
          f"absent {counts['absent']}, null {counts['null']}, boxes {counts['boxes']}",
          file=out)
    if frames_dir is None:
        print("no --frames-dir given: the frames themselves were NOT checked", file=out)
    else:
        print(f"{frames_dir}: {len(missing)} named-but-absent, {len(extra)} present-but-unnamed",
              file=out)

    if problems:
        print(f"\nFAILED — {len(problems)} problem(s):", file=out)
        for problem in problems:
            print(f"  - {problem}", file=out)
        return 1
    print("OK", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frames-dir", type=Path, default=None,
                        help="directory of the JPEGs the manifest names. Without it only "
                             "the manifest's internal consistency is checked")
    parser.add_argument("--split", default=None,
                        help="restrict the reported denominators to one split")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read manifest {args.manifest}: {error}") from None
    return report(manifest, rows_of(manifest), args.frames_dir, args.split)


if __name__ == "__main__":
    sys.exit(main())
