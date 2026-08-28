#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recompute every number in this README from the committed JSON. No video, model, network.

    python3 audit.py

WHAT IT CHECKS, AND WHY EACH ONE IS HERE.

1. **The scorer's preprocessing table against ``inference_profile.py`` itself.**
   ``score_checkpoints.py`` carries a hand-written ``PROFILES`` dict "copied from
   inference_profile.py by NAME so a reader can check them against it". Nothing was
   checking them. #147's whole finding is that a copy of a preprocessing constant is how a
   94-checkpoint sweep came to be scored through a configuration no launcher runs, so this
   turns that copy into a checked copy. It is read with ``ast`` rather than imported,
   because importing it needs ``cv2`` and this file must run without one.

2. **The stamp inside every results file against the profile it names.** Each row records
   its own ``input_size`` and ``confidence``; if a row's numbers disagree with the profile
   in its filename, the table above it describes a different detector than it says.

3. **The arms.** Positives, steps per epoch, total steps and the nesting
   ``1:1 ⊂ 1:3 ⊂ 1:9`` are recomputed from the manifests rather than restated, because the
   equal-step claim is the one thing separating this wave's ablation from wave 7's.

4. **Checkpoint counts against the declared epochs.** An arm that died at epoch 90 and was
   summarised anyway would otherwise read as an arm that finished.

5. **That the square is threaded through every call site in the trainer, not most of them.**
   ``--input-size`` is this wave's one change to ``detector/``, and a half-done version of
   it is worse than none: priors read at 224 with the dataset still resizing to 300 trains
   the loc head against boxes the network will never emit, and ``verify_head_assembly``
   would not catch it because the mirror is checked at one square in isolation. A grep is a
   crude test and it is the one that fails on the mistake actually available here.

6. **What each arm actually trained on, from its own log.** ``run_facts.json`` carries the
   prior count, the old-class teacher supervision and the step count each arm printed at
   startup. The equal-step claim and the "the teacher is also a function of the square"
   finding are both checked here against the run rather than against the intention.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WAVE7 = HERE.parent / "2026-08-27-lite3-training-set"
SCORER = WAVE7 / "score_checkpoints.py"

sys.path.insert(0, str(REPO / "robot-stack" / "unitree" / "go2" / "visual_nav"))
import inference_profile  # noqa: E402  (needs the sys.path line above)

#: ``name -> (real, synthetic, epochs, input_size)``. The manifests and the run script are
#: the source; this is what they are checked against.
ARMS = {
    "r1x1_224": ("lite3_train_r1x1_20260827.json", 144, 224),
    "r1x3_224": ("lite3_train_r1x3_20260827.json", 87, 224),
    "r1x9_224": ("lite3_train_aug_20260827.json", 40, 224),
    "r1x1_300": ("lite3_train_r1x1_20260827.json", 144, 300),
    "r1x9_300": ("lite3_train_aug_20260827.json", 40, 300),
    # Two more seeds of the arm that cleared the gate, to test whether the RECIPE clears it
    # or whether one epoch of one seed did. See run_seed_replicates.sh.
    "r1x1_224_s1": ("lite3_train_r1x1_20260827.json", 144, 224),
    "r1x1_224_s2": ("lite3_train_r1x1_20260827.json", 144, 224),
    "r1x9_224_s1": ("lite3_train_aug_20260827.json", 40, 224),
    "r1x9_224_s2": ("lite3_train_aug_20260827.json", 40, 224),
    "r1x1_300_s1": ("lite3_train_r1x1_20260827.json", 144, 300),
    "r1x1_300_s2": ("lite3_train_r1x1_20260827.json", 144, 300),
}
#: ``finetune_ssd.py``'s defaults for this wave, and what turns records into steps.
BATCH_SIZE = 24
#: In-domain negatives, from wave 7's own manifest. They are in every arm and identical.
NEGATIVES = WAVE7 / "negatives_20260827.json"

failures: list = []


def check(condition: bool, message: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {message}")
    if not condition:
        failures.append(message)


def scorer_profiles() -> dict:
    """``score_checkpoints.py``'s PROFILES literal, read without importing ``cv2``."""
    tree = ast.parse(SCORER.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "PROFILES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"no PROFILES assignment in {SCORER}")


def manifest(name: str) -> dict:
    """A manifest from this directory, falling back to wave 7's for the unchanged ones."""
    local = HERE / name
    return json.loads((local if local.exists() else WAVE7 / name).read_text())


def main() -> int:
    print("\nthe scorer's preprocessing table vs inference_profile.py")
    theirs = scorer_profiles()
    check(set(theirs) == set(inference_profile.PROFILES),
          f"same profile names: {sorted(theirs)}")
    for name, values in sorted(theirs.items()):
        declared = inference_profile.PROFILES.get(name)
        check(declared is not None and values["input_size"] == declared.input_size
              and values["confidence"] == declared.confidence,
              f"{name}: {values['input_size']} px at {values['confidence']}"
              + ("" if declared is None else
                 f" == inference_profile ({declared.source})"))

    print("\nevery results file's own stamp vs the profile it names")
    scored = sorted(HERE.glob("scored_*.json")) + sorted(HERE.glob("incumbent_*.json"))
    if not scored:
        print("  ..    no results files yet; run score_ratio_wave.sh")
    for path in scored:
        data = json.loads(path.read_text())
        profile = inference_profile.PROFILES[data["preprocessing"]]
        rows = data["results"]
        agree = all(r["input_size"] == profile.input_size
                    and r["confidence"] == profile.confidence for r in rows)
        check(agree and path.name.endswith(f"_{data['preprocessing']}.json"),
              f"{path.name}: {len(rows):>3d} rows, all at {profile.input_size} px / "
              f"{profile.confidence}")

    print("\nthe arms: positives, steps and the nesting")
    negatives = json.loads(NEGATIVES.read_text())["count"]
    check(negatives == 318, f"{negatives} in-domain negatives, identical in every arm")
    images: dict = {}
    totals: list = []
    for name, (labels, epochs, size) in ARMS.items():
        records = manifest(labels)["records"]
        real = [r for r in records if not r["image"].startswith("synth_")]
        images[labels] = {r["image"] for r in records}
        steps = (len(records) + negatives) // BATCH_SIZE
        totals.append(steps * epochs)
        check(len(real) == 283,
              f"{name}: {len(real)} real + {len(records) - len(real)} synthetic "
              f"(1 : {(len(records) - len(real)) / len(real):.2f}), {size} px, "
              f"{epochs} epochs x {steps} steps = {steps * epochs} steps")
    spread = (max(totals) - min(totals)) / max(totals)
    check(spread < 0.01, f"total steps agree to {spread * 100:.1f}% across all five arms "
                         f"({min(totals)}-{max(totals)})")
    order = ["lite3_train_r1x1_20260827.json", "lite3_train_r1x3_20260827.json",
             "lite3_train_aug_20260827.json"]
    for smaller, larger in zip(order, order[1:]):
        check(images[smaller] <= images[larger],
              f"{smaller.split('_')[2]} is a subset of {larger.split('_')[2]}")

    print("\ncheckpoints on disk vs the epochs each arm declares")
    for name, (_labels, epochs, _size) in ARMS.items():
        found = {len(json.loads(p.read_text())["results"])
                 for p in HERE.glob(f"scored_{name}_*.json")}
        if not found:
            print(f"  ..    {name}: not scored yet")
            continue
        check(found == {epochs},
              f"{name}: {found.pop()} checkpoints scored, {epochs} epochs declared")

    print("\nthe square is threaded through the trainer, not left hardcoded")
    trainer = (REPO / "detector" / "finetune_ssd.py").read_text()
    # The four places the square decides what is computed: priors, the mirror check, the
    # teacher pass and the dataset resize. Each must read a parameter, not the constant.
    check("(INPUT_SIZE, INPUT_SIZE)" not in trainer,
          "no `(INPUT_SIZE, INPUT_SIZE)` blob or resize left in finetune_ssd.py")
    check(trainer.count("input_size: int = INPUT_SIZE") == 4,
          f"{trainer.count('input_size: int = INPUT_SIZE')} functions take the square as a "
          f"parameter defaulting to the module constant")
    check('parser.add_argument("--input-size"' in trainer,
          "finetune_ssd.py has an --input-size flag")

    print("\nthe determinism probe, which is this wave's headline")
    probe = HERE / "determinism.json"
    if not probe.exists():
        print("  ..    no determinism.json; run probe_determinism.sh on the training host")
    else:
        runs = json.loads(probe.read_text())["runs"]
        digests = {r["weights_md5"] for r in runs}
        check(len(digests) == len(runs),
              f"{len(runs)} identical invocations at --seed 0 produced {len(digests)} "
              f"distinct .caffemodel files")
        matched = {r["matched_per_batch"] for r in runs}
        check(len(matched) == 1,
              f"matched/batch is {matched.pop() if len(matched) == 1 else matched} in all "
              f"of them — the sampler and the augmentation ARE deterministic, so the "
              f"divergence is in GPU compute and not in the RNG stream")

    print("\nthe committed trainer is the trainer that ran")
    facts_file = HERE / "run_facts.json"
    if not facts_file.exists():
        print("  ..    no run_facts.json; run collect_run_facts.py on the training host")
    else:
        for name, digest in json.loads(
                facts_file.read_text())["trainer_md5"].items():
            here = hashlib.md5((REPO / name).read_bytes()).hexdigest()
            check(here == digest, f"{name}: {here[:12]} on the training host and in this "
                                  f"pull request")

    print("\nwhat each arm printed at startup, vs what this wave says it ran")
    facts_path = HERE / "run_facts.json"
    if not facts_path.exists():
        print("  ..    no run_facts.json; run collect_run_facts.py on the training host")
    else:
        facts = json.loads(facts_path.read_text())["arms"]
        check(set(facts) == set(ARMS), f"{len(facts)} arms recorded: {sorted(facts)}")
        for name, (_labels, epochs, size) in ARMS.items():
            arm = facts.get(name, {})
            check(arm.get("input_size") == size and arm.get("epochs_run") == epochs
                  and arm.get("checkpoints") == epochs,
                  f"{name}: trained at {arm.get('input_size')} px for "
                  f"{arm.get('epochs_run')} epochs, {arm.get('checkpoints')} checkpoints")
            check(arm.get("priors") == (1014 if size == 224 else 1917),
                  f"{name}: {arm.get('priors')} priors at {size} px")
        step_counts = [a["total_steps"] for a in facts.values()]
        ran = (max(step_counts) - min(step_counts)) / max(step_counts)
        check(ran < 0.01, f"total steps AS RUN agree to {ran * 100:.1f}% "
                          f"({min(step_counts)}-{max(step_counts)})")
        # The teacher runs at the training square, so the old-class supervision the arms get
        # is not the same between 224 and 300 even on identical records. Named rather than
        # buried: it is the confound inside "224 versus 300".
        for ratio in ("r1x1", "r1x9"):
            small, large = facts[f"{ratio}_224"], facts[f"{ratio}_300"]
            check(small["positives"] == large["positives"],
                  f"{ratio}: same {small['positives']} training frames at both squares, and "
                  f"{small['teacher_boxes']} teacher boxes at 224 vs "
                  f"{large['teacher_boxes']} at 300 "
                  f"({100 * (1 - small['teacher_boxes'] / large['teacher_boxes']):.0f}% "
                  f"less old-class supervision at 224)")

    print(f"\n{len(failures)} failures" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
