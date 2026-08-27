#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Which phrase actually finds a Lite3? Sweep twelve, on five keyframes of each clip.

    python3 probe_queries.py            # on the Spark, with the keyframes staged

THE RESULT THIS EXISTS TO RECORD. ``a robot dog`` -- the phrase anyone would reach for
first, and the one the first labelling pass used -- scores **0.000 on all ten frames**. So
do ``a robotic dog``, ``a dog``, and on nine of ten ``a quadruped robot``. ``a robot`` fires
at 0.068-0.169 and puts its box on a ceiling fitting. ``a small white four-legged machine``
scores **0.305-0.679** and puts it on the robot.

A phrase that reads like the object's NAME loses to a phrase that reads like its
DESCRIPTION. On ``light-lite3`` that is the difference between 0 and 60 labelled frames,
which is the difference between having a quadruped class and not having one.

The threshold here is 0.02, deliberately below anything shippable, because the question is
which phrases fire AT ALL -- not which of them pass a gate.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

HOME = Path.home()
KEYFRAMES = HOME / "lite3sam" / "keyframes"
#: Deliberately below anything shippable. See the module docstring.
THRESHOLD = 0.02
PHRASES = [
    "a robot dog", "a robot", "a quadruped robot", "a four-legged robot",
    "a robotic dog", "a white robot", "a mechanical animal", "a toy dog", "a dog",
    "a machine on four legs", "a legged robot", "a small white four-legged machine",
]
MODEL_ID = "google/owlv2-base-patch16-ensemble"
#: Fractions through each clip's keyframe list to sample.
SAMPLE_AT = (0.1, 0.3, 0.5, 0.7, 0.9)


def best_per_phrase(processor, model, image: Image.Image) -> dict:
    """The highest-scoring box each phrase returns on one frame."""
    inputs = processor(text=[PHRASES], images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    target = torch.tensor([[image.height, image.width]], device="cuda")
    result = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target, threshold=THRESHOLD)[0]
    best: dict = {}
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        phrase = PHRASES[int(label)]
        if float(score) > best.get(phrase, (0.0, None))[0]:
            best[phrase] = (float(score), [round(float(v)) for v in box])
    return best


def main() -> None:
    distinct = json.loads((HOME / "lite3sam" / "distinct_views.json").read_text())
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to("cuda").eval()

    out: dict = {"threshold": THRESHOLD, "phrases": PHRASES, "model": MODEL_ID,
                 "scenes": {}}
    for scene in sorted(s for s in distinct["scenes"] if "-lite3-" in s):
        keys = distinct["scenes"][scene]["keyframes"]
        chosen = [keys[int(f * (len(keys) - 1))] for f in SAMPLE_AT]
        print(f"\n=== {scene} ({chosen})")
        table: dict = {phrase: [] for phrase in PHRASES}
        for frame in chosen:
            image = Image.open(KEYFRAMES / scene / f"f{frame:05d}.jpg").convert("RGB")
            best = best_per_phrase(processor, model, image)
            for phrase in PHRASES:
                table[phrase].append(best.get(phrase, (0.0, None)))
        out["scenes"][scene] = {
            phrase: [{"score": round(score, 4), "box": box}
                     for score, box in table[phrase]] for phrase in PHRASES}
        for phrase in PHRASES:
            scores = " ".join(f"{score:.3f}" for score, _ in table[phrase])
            print(f"  {phrase:34s} {scores}   best box {max(table[phrase])[1]}")

    here = Path(__file__).resolve().parent
    (here / "phrase_sweep.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {here / 'phrase_sweep.json'}")


if __name__ == "__main__":
    main()
