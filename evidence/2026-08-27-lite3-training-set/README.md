<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-27 — six raw Lite3 clips, 5,854 frames, 456 views, and a labeller that had to be audited twice

Six `--record-raw` recordings arrived from Shanghai with their telemetry, answering the
three asks in [`evidence/2026-08-27-lite3-pov-clip-audit`](../2026-08-27-lite3-pov-clip-audit/README.md)
in order: no burned-in overlay, the `.jsonl` alongside, and a subject that moves. They are a
real improvement on the tripod clip that preceded them, and they are still one morning.

Three measurements decide what they are worth, and none of them is the frame count.

**The camera never moves — in any of the six.** ORB+RANSAC homography against each clip's
own first frame gives a median displacement of **0.0–1.0 px** per clip; the steadiest moves
0.18 px in 90 seconds. "in different distance and angle" in the folder names describes the
*subject*. So the set holds at most **six viewpoints**, not 5,854.

**5,854 frames contain 456 distinct views — 7.8%.** Sampling every 5th frame and keeping one
only when it differs from the last kept frame by more than 3.0 mean grey levels on a 160x90
thumbnail. The instrument is crude and its number moves with its knobs, which is why both are
printed together; the conclusion does not move at all.

**The telemetry cannot label any of it.** Every header reads `classes: ["person"]`,
`confidence: 0.4` — `go2-navigator-default`. Across **2,701 frames of quadruped footage the
telemetry carries 3 boxes**. The network was not blind; the class filter discarded what it
saw.

Reproduce every number below with no video, no model and no network:

```bash
python3 audit.py
```

## ⛔ The weak-supervision route was tried, measured, and abandoned

Before reaching for a segmenter, the obvious cheap route was tested: re-run the *same*
shipped MobileNet-SSD over the *same* pixels with the class filter removed, and gate the
result on box aspect and on background subtraction (the camera is static, so a temporal
median is the empty room). It recovers the Lite3 as `chair`, exactly as the previous audit
predicted at 300 px.

| clip | hand-checked | verdict |
| --- | --- | --- |
| `light-lite3` | **12 / 12** on the robot | looks like a working method |
| `dim-lite3` | **0 / 12** — every box on the same office-chair cluster, robot visible and unboxed beside it | it is not |

The dim clip sweeps **28 → 83 mean luminance** with frame-to-frame steps up to **37.9 grey
levels**, so the "empty room" median is not a background and the motion gate **fails open**.
One rule scoring 12/12 and 0/12 on two clips of the same robot in the same room is not a rule.
That result, not a preference, is why the labelling moved to a segmenter.

## What labels the set now

`google/owlv2-base-patch16-ensemble` turns a **phrase** into boxes; `facebook/sam2.1-hiera-large`
turns a box into a **mask**, and the mask's extent is the label — tighter than a regressed
rectangle because it is a silhouette. Both Apache 2.0; neither is vendored and no weights are
committed.

**Image mode, on the 456 distinct views — not SAM 2 video propagation.** Propagating through
5,398 near-duplicates buys nothing and costs the one failure mode with no internal signal: a
mask drifting onto the wrong object. Image mode has no propagation state and cannot drift.

⛔ **Neither model names a class.** SAM is class-agnostic by construction and OWLv2's "class"
is only the phrase it was handed. Every label's class comes from
[`scene_queries.json`](scene_queries.json) — the scene folder name plus the object prompted —
which is committed beside the labels so a wrong class is a wrong line someone can read. The
`*-box-*` scenes contain both Timo and a cardboard box; only the person is prompted, and the
box is correctly left unlabelled.

### ⚠️ The phrase that reads like the object's name scores zero

Twelve phrasings, scored on five keyframes of each quadruped clip:

| phrase | light-lite3 | dim-lite3 |
| --- | --- | --- |
| `a robot dog` | **0.000 on all five** | **0.000 on all five** |
| `a robotic dog` | 0.000 | 0.000 |
| `a quadruped robot` | 0.000 | 0.000 on four of five |
| `a dog` | 0.000 | 0.000 |
| `a robot` | 0.112–0.157, box on a ceiling fitting | 0.068–0.169 |
| **`a small white four-legged machine`** | **0.305–0.629** | **0.376–0.574** |

A phrase that reads like the object's **name** loses to one that reads like its
**description**. This is the difference between 0 and 60 labelled frames in `light-lite3`, and
it was found by sweeping, not by guessing. `probe_queries.py` is the sweep.

### ⚠️ The first threshold passed the quadruped and failed the person, and only the hand-check said so

At the initial global floor of 0.12 the two classes are not on the same scale — `lite3` scores
a median 0.517, `person` a median 0.228 — and one number cannot serve both:

| class | at 0.12, hand-checked | failures scored | successes scored |
| --- | --- | --- | --- |
| `lite3` | 23 / 24 | 0.13 | 0.31–0.67 |
| `person` | **9 / 16** | 0.12–0.20 | 0.21+ |

The floor is **0.22 per query**, set at the point the hand-check turned — **not** from the
score distribution. It keeps 95% of `lite3` (131/138) and 51% of `person` (259/505). Half the
person boxes are discarded on purpose: a wrong box poisons a training set, a missing one only
makes it smaller.

## The hand-check, which is the number that decides everything

| sheet | route | on-subject |
| --- | --- | --- |
| [`handcheck-lite3.jpg`](handcheck-lite3.jpg) | owlv2+sam2 @0.22, 8 frames per quadruped clip | **16 / 16** |
| [`handcheck-person.jpg`](handcheck-person.jpg) | owlv2+sam2 @0.22, 4 frames per person clip | **16 / 16** |
| [`handcheck-synthetic.jpg`](handcheck-synthetic.jpg) | shear / colour-slice / occlude | **12 / 12** |
| [`handcheck-telemetry-join.jpg`](handcheck-telemetry-join.jpg) | the recovered telemetry join | **17 / 17** |
| (superseded) | weak supervision, `dim-lite3` | **0 / 12** |
| (superseded) | owlv2+sam2 @0.12, `person` | **9 / 16** |

**44 of 44** at the shipped configuration. The two superseded rows are kept deliberately —
they are the measurements that chose the method, and both were found by looking at frames the
method called *good*, which is the only way either would have been caught.

## The dataset

| | count | |
| --- | ---: | --- |
| frames delivered | 5,854 | six clips, 1280x720, 15 fps |
| **distinct views** | **456** | 7.8% — the rest are copies |
| `lite3` boxes on keyframes | 131 | 71 dim + 60 light |
| `lite3` train, with ±1 ride-along | 283 | measured at 0.954 median IoU |
| `lite3` eval, held-out time block | 36 | **same session** |
| synthetic `lite3` | 2,542 | shear 849, colour-slice 849, occlude 844 |
| **train + synthetic** | **2,825** | real : synthetic = **1 : 9.0** |
| `person` boxes | 259 | a description of the footage, not a training input |
| in-domain negatives | 318 | same room, no quadruped — the previous set had **zero** |
| **frames carrying no label** | **5,464** | 5,854 − 390 labelled |

**Unlabelled, stated plainly.** Of 2,701 quadruped frames, 131 carry a box and **2,570 do
not** — but 2,570 is not 2,570 lost samples: only 138 of those frames are distinct views, and
**every distinct view in both quadruped clips got a box** (78/78 and 60/60 before thresholding;
71 and 60 after). The unlabelled remainder is near-duplicate, not missed.

### Near-duplicates ride along at ±1 frame, and no further

Re-labelling the frames around a sample of keyframes with the same pipeline and IoU-ing them
against the keyframe's own box:

| offset | n | median IoU | p10 | below 0.75 |
| ---: | ---: | ---: | ---: | ---: |
| +1 | 59 | **0.954** | 0.757 | 5 |
| +2 | 56 | 0.926 | 0.669 | 12 |
| +4 | 59 | 0.873 | **0.496** | 22 |

±1 is shipped. At +4 one box in ten has slid off its object, so nothing wider rides along —
that would be pasting a stale box onto a moved subject. `lite3` drifts less than `person`
(0.961 vs 0.873 median), because a walking person outruns a walking robot.

### What the synthetic half is and is not

Three families the trainer has **no operator for**: `shear` (tilt/skew), `colour-slice`
(vertical bands — illumination that varies *across* the frame, where the trainer's photometric
jitter is global), and `occlude`. Photometric, expand, crop, flip, JPEG, motion-blur,
sensor-noise and composite are **deliberately not generated**: `finetune_ssd.py` does those
online with a fresh draw every epoch, and freezing them into files would inflate the count
without adding information.

⚠️ **It adds 0 viewpoints, 0 rooms and 0 days.** Augmentation multiplies examples. 456 views
from 13 minutes of one morning is the ceiling, and a 1:9 ratio does not move it.

## The three fine-tunes, and the gate two of them fail everywhere

`k_full_pseudo03` is the best detector this project has produced — 89% recall at 12% false
alarms against the shipped weights' 68%/56%, keeping 17 of 22 people, and it ran with **no
augmentation flags**. Its hyperparameters are the constant across all three runs
(`--freeze-through= --backbone-lr-scale 0.5 --pseudo-labels 0.3 --distil 0.1`, 40 epochs);
one thing moves at a time.

| run | positives | what it isolates |
| --- | ---: | --- |
| `a_ws_real` | 283 | the contemporaneous paired control |
| `b_ws_synth` | 2,825 | + the offline synthetic half |
| `c_ws_synth_aug` | 2,825 | + the wave-6 `--motion-blur/--sensor-noise/--composite` flags |

`a` is a real control, not a citation: these operators call `rng.random()` even at
probability 0, so no earlier run is byte-reproducible under this code.

```bash
python3 summarise_scores.py
```

### Incumbent, scored here by the same script

| preprocessing | | people kept, cross-day |
| --- | --- | ---: |
| `go2-peer-supervised` — **what production launches** | 224 px / 0.25 | **25 / 284** |
| `go2-navigator-default` | 300 px / 0.4 | 24 / 284 |
| `go2-run-smoke` | 300 px / 0.45 | 20 / 284 |

### Best `lite3` epoch, and what it costs in people

⚠️ This table is an argmax of one metric with the other merely *reported* at that point. It
is how a checkpoint sweep picks a winner and it is not a basis for shipping anything.

| run | 224 px / 0.25 (production) | 300 px / 0.4 | 300 px / 0.45 |
| --- | --- | --- | --- |
| `a` real only | 4/36 lite3, **21** people (−4) | 7/36, **29** (+5) | 7/36, **25** (+5) |
| `b` + synthetic | 15/36, **7** (−18) | 18/36, **18** (−6) | 17/36, **16** (−4) |
| `c` + synthetic + flags | 15/36, **1** (−24) | 19/36, **0** (−24) | 18/36, **2** (−18) |

The ablation is monotone in both directions at once: **every step that adds augmentation adds
`lite3` hits and removes people.** `c` at 300 px finds the robot in 19 of 36 frames and has
almost stopped seeing people at all — 0 of 284. That reproduces wave 6's own finding, where
adding the same three flags to `k_full_pseudo03` lost 4–13 people at every matched epoch.

### Best epoch that keeps at least as many people as the incumbent

This is the column that decides shipping, because the standing gate is *lose zero of the
people the shipped network sees*.

| run | 224 px / 0.25 (production) | 300 px / 0.4 | 300 px / 0.45 |
| --- | --- | --- | --- |
| `a` real only | **no epoch passes** | ep026 — **7/36**, 29 people (+5) | ep026 — 7/36, 25 (+5) |
| `b` + synthetic | ep001 — 0/36, 29 (+4) | **no epoch passes** | ep002 — 3/36, 22 (+2) |
| `c` + synthetic + flags | ep001 — 0/36, 34 (+9) | ep001 — 2/36, 26 (+2) | **no epoch passes** |

`ep001` is the base weights barely moved, so a row reading "ep001, 0/36" is the gate saying
*nothing here*.

### ⛔ Nothing is recommended, and here is the one bright corner

**At 224 px, the size `deploy/run-peer-supervised.sh` actually launches, no checkpoint from
any of the three runs both detects the Lite3 and holds its people.** The best that detects
anything at all, `b` at 15/36, keeps **7 of 284** people against the incumbent's 25. That is
a worse failure than the previous Lite3 wave's 3-and-5-of-17, and it is refused on the same
grounds.

The one real gain: **`a_ws_real` epoch 026 at `go2-navigator-default` finds the Lite3 in 7 of
36 held-out frames while keeping 29 of 284 people against the incumbent's 24** — it gains 5.
That is a checkpoint that learned a new class without paying for it in the old one. Three
caveats, all of them load-bearing:

* the 7/36 is **same-session**, and 19% recall is weak besides;
* 300 px / 0.4 is `go2-navigator-default`, which is what a launcher that passes *no flags*
  produces — **not** what the peer-avoidance launcher passes;
* 29 vs 24 people is +5 on a base of 284, and the r/s duplicate pair on the Go2 corpus puts
  this project's own run-to-run noise at ±1–3 people. +5 is outside that, but not by much.

**The 300-vs-224 split is still the blocker, and this is now the third wave to hit it.** The
previous Lite3 wave measured every checkpoint at 168/168 at 300 and 0/168 at 224.
`finetune_ssd.py` resizes every training image to 300; the launcher opens at 224; the class
the training produces is weakest exactly where it has to run.

## ⛔ Pixels only

Nothing here reads `range_m`, `focal_px`, `height_m` or `hfov_deg`. The camera block embedded
in these recordings is wrong three ways — `height_m` 0.40 against Timo's measured 0.37 m
standing, no pitch field where the mount is ~11°, and a `focal_px` implying 107.46° against a
stated `hfov_deg` of 156.16° — and [`robot-stack/CAMERA-GEOMETRY.md`](../../robot-stack/CAMERA-GEOMETRY.md)
holds the record. A detector outputs boxes in pixels and transfers across cameras; nothing in
this set should inherit a focal length that describes no real camera.

## Two defects in the recording tool, found here

* **`perception.video_frame` is `null` in all 3,896 ticks of all six files.** It is documented
  as the telemetry↔video join and it does not exist. The join was recovered as
  `round((t - frame_age_s) * 15)` and validated by hand on 17 frames across all six clips;
  it reproduces 80.4% of the live sightings at IoU ≥ 0.5, median IoU 0.793. The shortfall is
  concentrated in the fast-moving people clips, which is why the labels here are computed on
  the extracted frame itself and carry **no join error at all**.
* **The detector ran `classes: ['person']`**, which is why 2,701 frames of quadruped footage
  produced 3 boxes.

Both are written up for the operator, bilingually, in
[`robot-stack/deep_robotics/lite3/RECORDING-TRAINING-FOOTAGE.md`](../../robot-stack/deep_robotics/lite3/RECORDING-TRAINING-FOOTAGE.md).

## Files

| file | what |
| --- | --- |
| `audit.py` | recomputes every number above from the committed JSON. No video, model or network. |
| `measure_scenes.py` | reads the video and writes `scene_measurements.json` (camera motion, luminance, the class-filter-free detector pass) |
| `distinct_views.py` | the 456 |
| `probe_queries.py` | the phrase sweep |
| `sam_label.py` | owlv2 + sam2, image mode |
| `build_dataset.py` | labels → train/eval/negative manifests |
| `make_synthetic.py` | the three synthetic families |
| `score_checkpoints.py` | scores checkpoints at a **named** preprocessing, fresh net per configuration |
| `summarise_scores.py` | the two score tables above, both selection rules |
| `scored_*.json`, `incumbent_*.json` | every checkpoint at every profile |
| `run_lite3_ws.sh` | the Spark wave |
| `scene_queries.json` | **what each class means** — folder name + prompted phrase |
| `handcheck.json` | every hand-inspected frame and its verdict |
| `neighbour_drift.json` | the ride-along measurement |

No video, no frames and no weights are committed.
