#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pull what each arm actually trained on out of its own log, into one committed file.

    python3 collect_run_facts.py --runs ~/lite3-ratio-20260827/runs --out run_facts.json

The training logs stay on the Spark -- they are 144 lines of loss per arm and none of it is
evidence. Four numbers in each log's HEADER are, and they are not derivable from the
manifests:

``priors``          1917 at 300 px and 1014 at 224. PriorBox takes its ``img_width`` from
    the data blob, so the square changes how many boxes the loss is written against and how
    large each one is relative to the frame.
``teacher_boxes``   the old-class supervision that ``--pseudo-labels 0.3`` carries. It is
    the shipped network's own detections over the training frames, so it is ALSO a function
    of the square -- and that is the confound in "224 versus 300" that no manifest shows.
``matched``         priors matched per batch, averaged over the last epoch. Reported because
    it moves with the square for the same reason.
``final_loss``      conf and loc at the last epoch, so an arm that diverged is visible here
    rather than only in a score table.

It also records the **md5 of the four source files the training host actually ran**, so
``audit.py`` can assert that the committed trainer IS the trainer that produced these
weights. That is the same argument ``robot-stack/preflight/tree_stamp.py`` makes for the
robot: a deployed tree is not a checkout, and "deployed from main" is not a claim anyone
can check. Here it is four hashes, and they are checked.

⚠️ **A LOG IS NOT AN INTERFACE.** These are parsed out of lines ``finetune_ssd.py`` prints
for a human, so the patterns below are pinned to the exact strings and this refuses on a
line it cannot read rather than defaulting a field to zero. If the trainer's printing
changes, this must fail loudly -- a silently missing ``teacher_boxes`` would read as an arm
that had no old-class supervision at all, which is the one condition wave 5 measured as
catastrophic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

#: The files that decide what a checkpoint is, relative to the deployed tree root. If the
#: committed copy of any of these differs from what ran, every number below describes a
#: program that is not in the pull request.
TRAINER_FILES = (
    "detector/finetune_ssd.py",
    "detector/ssd_torch.py",
    "detector/add_class.py",
    "robot-stack/unitree/go2/visual_nav/inference_profile.py",
)

PATTERNS = {
    "input_size": re.compile(r"^training at (\d+) px"),
    "priors": re.compile(r"^(\d+) priors from the deployed network"),
    "positives": re.compile(r"^(\d+) labelled \+ (\d+) peer-free frames"),
    "teacher": re.compile(r"^(\d+) teacher boxes over (\d+) frames"),
}
EPOCH = re.compile(r"^epoch\s+(\d+)/(\d+)\s+conf ([\d.]+)\s+loc ([\d.]+)\s+"
                   r"distil ([\d.]+)\s+matched/batch ([\d.]+)")


def read(log: Path) -> dict:
    """One arm's facts, or a hard failure. Every field below is required."""
    facts: dict = {}
    last = None
    for line in log.read_text().splitlines():
        for name, pattern in PATTERNS.items():
            found = pattern.match(line)
            if not found:
                continue
            if name == "positives":
                facts["positives"], facts["negatives"] = (int(g) for g in found.groups())
            elif name == "teacher":
                facts["teacher_boxes"], facts["teacher_frames"] = (
                    int(g) for g in found.groups())
            else:
                facts[name] = int(found.group(1))
        found = EPOCH.match(line)
        if found:
            last = found
    if last is None:
        raise SystemExit(f"{log}: no epoch lines — the arm did not train")
    facts["epochs_run"], facts["epochs_declared"] = int(last.group(1)), int(last.group(2))
    facts["final_conf"], facts["final_loc"] = float(last.group(3)), float(last.group(4))
    facts["matched_per_batch"] = float(last.group(6))
    missing = [k for k in ("input_size", "priors", "positives", "negatives",
                           "teacher_boxes", "teacher_frames") if k not in facts]
    if missing:
        raise SystemExit(f"{log}: could not read {missing}. finetune_ssd.py's header lines "
                         f"have changed; fix the patterns rather than the defaults.")
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True,
                        help="the deployed source tree these runs were launched from")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    trainer = {}
    for name in TRAINER_FILES:
        path = args.tree / name
        if not path.exists():
            raise SystemExit(f"{path} is not in the deployed tree; the hash that proves "
                             f"the committed trainer is the one that ran cannot be taken")
        trainer[name] = hashlib.md5(path.read_bytes()).hexdigest()
        print(f"  {trainer[name]}  {name}")

    out: dict = {"trainer_md5": trainer}
    for log in sorted(args.runs.glob("*.log")):
        facts = read(log)
        history = args.runs / log.stem / "history.json"
        if history.exists():
            facts["steps_per_epoch"] = (facts["positives"] + facts["negatives"]) // int(
                json.loads(history.read_text())["args"]["batch_size"])
            facts["total_steps"] = facts["steps_per_epoch"] * facts["epochs_run"]
        facts["checkpoints"] = len(list((args.runs / log.stem).glob("epoch*.caffemodel")))
        out.setdefault("arms", {})[log.stem] = facts
        print(f"  {log.stem:12s}{facts['input_size']:>5d} px  {facts['priors']:>5d} priors "
              f"{facts['positives']:>5d}+{facts['negatives']} frames  "
              f"{facts['teacher_boxes']:>4d} teacher boxes over "
              f"{facts['teacher_frames']:>4d} frames  {facts['epochs_run']:>4d} epochs  "
              f"{facts.get('total_steps', 0):>6d} steps  "
              f"{facts['checkpoints']:>4d} checkpoints")
    args.out.write_text(json.dumps(out, indent=1))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
