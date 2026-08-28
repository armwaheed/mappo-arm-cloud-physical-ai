<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-27 — five identical runs, five different detectors: this project has been ranking n=1 arms

This wave was asked to settle whether the `real : synthetic` ratio, rather than the input
square, is what makes a fine-tuned detector stop seeing people. Five arms at matched gradient
steps gave a clean answer. Then the arms were re-run, and **the answer did not survive its own
replication** — because the instrument is noisier than the effect it was built to measure.

The measurement that matters here is therefore not a ranking. It is the finding that no
ranking this project has published could have been trusted, including the one this document
was drafted to make.

## ⛔ The trainer is not reproducible at a fixed seed

Five byte-identical invocations of `finetune_ssd.py`, `--seed 0` in every one, same machine,
same data, one epoch each:

| run | conf loss | loc loss | `matched/batch` | md5 of the exported weights |
| ---: | ---: | ---: | ---: | --- |
| 1 | 5.4547 | 1.7811 | 98.3 | `9e4091e06858d928…` |
| 2 | 5.4540 | 1.7885 | 98.3 | `05f85fbc40beecf1…` |
| 3 | 5.4411 | 1.7717 | 98.3 | `a1d919985d4db335…` |
| 4 | 5.3261 | 1.7566 | 98.3 | `c65bb9b4fc0b48af…` |
| 5 | 5.4765 | 1.7988 | 98.3 | `463e53568e27cc9c…` |

**Five runs, five different `.caffemodel` files.** The classification loss spans
5.3261–5.4765 — a 2.82% spread
after a single epoch — and no two exported checkpoints share a hash.

**`matched/batch` is identical in all five.** That is the control, and it is what makes this
a specific finding rather than a shrug: `matched/batch` is a property of the data pipeline —
which images, in which order, with which augmentation — so the sampler and the augmentation
RNG *are* deterministic. The divergence is in GPU compute, and it is already visible after
one epoch.

⚠️ **Wave 7 diagnosed this and got it wrong, and the wrong diagnosis is the dangerous part.**
[`run_lite3_ws.sh`](../2026-08-27-lite3-training-set/run_lite3_ws.sh) and that wave's README
both state that "these operators call `rng.random()` even at probability 0, so no earlier run
is byte-reproducible under this code" — attributing irreproducibility to the **augmentation
RNG stream**, which implies that pinning the stream would fix it. It would not. The stream is
already deterministic, as `matched/batch` shows, and the weights diverge anyway. Someone
acting on wave 7's sentence would pin the stream, re-run, still get different weights, and
have no idea why.

### What that does to every ranking this project has published

Run one arm three times, changing nothing, and score all 432 checkpoints:

| profile | median people, three runs | epochs clearing the floor | best `lite3` under the gate |
| --- | --- | --- | --- |
| `go2-peer-supervised` 224/0.25 | 12.5 / 6.0 / 16.0 | 4 / 1 / 2 | 15/36 / 0/36 / 12/36 |
| `go2-navigator-default` 300/0.4 | 13.0 / 18.0 / 22.0 | 1 / 0 / 42 | 3/36 / none / 19/36 |
| `go2-run-smoke` 300/0.45 | 10.0 / 15.5 / 21.0 | 1 / 0 / 108 | 1/36 / none / 18/36 |

The spread **within one condition** is wider than any difference this wave was built to
detect between conditions. Every wave in this project — including the one that produced the
shipped detector's ranking — has compared arms at **n=1**.

> **Had I run one seed, as every previous wave did, this directory would have published a
> gate-clearing production checkpoint.**

`r1x1_224` epoch 037 clears the production gate at 15/36 `lite3` with exactly 25 of 284
people. It is a real measurement, it re-scores to 25 every time, and it is an artefact of
which run produced it. That is the whole argument for replicates, and it cost two extra runs
of an arm that had already finished.

## ⛔ The ratio finding did not survive replication either

This wave was run to test whether `real : synthetic` decides person retention. On one run per
condition it looked decisive — median people at 224 px / 0.25 of **12.5** at 1:1 against
**3.5** at 1:9, with the square moving it 0.5–2.5. That table was written, and then both ends
of the sweep were re-run twice more.

**Median people over every epoch, three runs per condition, at 224 px:**

| profile | 1:1, three runs | 1:9, three runs | means |
| --- | --- | --- | --- |
| `go2-peer-supervised` 224/0.25 | 12.5 / 6.0 / 16.0 | 3.5 / **16.0** / **14.0** | 11.5 vs 11.2 — **+0.3** |
| `go2-navigator-default` 300/0.4 | 13.0 / 18.0 / 22.0 | 9.5 / **23.0** / 5.0 | 17.7 vs 12.5 — +5.2 |
| `go2-run-smoke` 300/0.45 | 10.0 / 15.5 / 21.0 | 6.5 / **21.5** / 2.0 | 15.5 vs 10.0 — +5.5 |

**The distributions overlap completely at every profile**, and at the square production
launches the two conditions are indistinguishable: 11.5 against 11.2, on within-condition
standard deviations of 5.1 and 6.7. The 3.5 that made the original table was a low draw of
1:9; its two replicates score 16.0 and 14.0, which is squarely inside the 1:1 range. Gate
clearing behaves the same way — one 1:9 run clears **18 of 40** epochs at 300 px / 0.4 and
**22 of 40** at 300 px / 0.45, better rates than two of the three 1:1 runs.

⛔ **So the answer to the question this wave was commissioned to settle is that the
experiment cannot settle it.** Not "the ratio does not matter" — the within-condition spread
is 5 to 12 people and the between-condition difference is 0.3 to 5.5, so an effect of the
size being looked for is invisible under the noise of the instrument. Distinguishing them
needs many more runs per condition than five arms of one, and that is a different experiment
from the one that was briefed.

### ⚠️ The two columns are not equally noisy, and that is a mechanism, not a curiosity

The same runs that scatter the person column by 12.5 people give a **stable** `lite3` column:

| condition, three runs each | median `lite3` at 224 px / 0.25 | mean | range |
| --- | --- | ---: | ---: |
| 1:1 @224 | 16.0 / 11.0 / 16.0 | 14.3 | 5.0 |
| 1:9 @224 | 19.0 / 17.0 / 18.0 | **18.0** | **2.0** |

`lite3` is the class being **trained**; `person` is a class being **preserved** by
distillation and pseudo-labels, and every retained person box is a marginal detection sitting
near a score threshold. GPU-level nondeterminism moves marginal detections across a threshold
and leaves confident ones alone. That is why the gate column — the only cross-day column, and
the one every wave has been refused on — is the noisiest number in the experiment.

**What does survive**, because it lives in the stable column:

* **more synthetic data detects the robot better**: 18.0 against 14.3 median `lite3`, with
  per-condition ranges of 2.0 and 5.0. That is the direction wave 7 reported, and it holds.
* **training at 224 buys Lite3 recall at 224**: about **+6 of 36** — median 19/36 trained at
  224 against 12/36 trained at 300, at 1:9; 16/36 against 10/36 at 1:1. The train/deploy
  square mismatch (#129) was real and it was costing detection.

**What does not survive**: any claim that a training condition in this wave moved person
retention, in either direction, including the one this document was going to lead with.

## ✅ One thing does replicate, and it is at 300 px

`r1x1_300` — 283 real positives against 283 synthetic, trained at 300 — was run three times.
Best `lite3` **among epochs keeping at least the incumbent's people**, with the epoch given
so that a near-untouched network cannot masquerade as a trained one:

| profile | run 1 | run 2 | run 3 | incumbent |
| --- | --- | --- | --- | --- |
| `go2-navigator-default` 300/0.4 | **17/36** @ep031 | **8/36** @ep016 | **14/36** @ep036 | 0/36 |
| `go2-run-smoke` 300/0.45 | **17/36** @ep039 | **11/36** @ep033 | **13/36** @ep037 | 0/36 |
| `go2-peer-supervised` 224/0.25 | 6/36 @ep022 | 2/36 @ep016 | 1/36 @ep005 — base weights | 0/36 |

**At both 300 px deployments, three runs out of three produce a *trained* checkpoint that
detects the Lite3 while keeping at least as many cross-day people as the shipped network.**
Every winning epoch is between 31 and 39 of 144 — a third of the way into the schedule, loss
down 2.7x — not an `ep001` row. The shipped weights find the Lite3 in **0** of 36.

That is a capability, and it is the only claim in this document that survived being attacked.
What does **not** survive is the number: 17, 8 and 14 of 36 is a factor-of-two spread, so no
single checkpoint is "the" result and `epoch031`'s 17/36 is the top of a range, not a value.

⛔ **And it does not hold at 224 px, the square `deploy/run-peer-supervised.sh` opens.** The
same three runs give 6, 2 and 1 of 36 there, and the third is `ep005` — the base weights
barely moved. Whatever this arm learns, the peer launcher cannot see it.

### What this does to wave 7, stated precisely

Wave 7's ablation is "monotone in both directions" on n=1 per arm. Its person column at
224 px / 0.25 reads 21 → 7 → 1 across `a` → `b` → `c`. Against the within-condition spread
measured here — standard deviations of 5.1 and 6.7, ranges up to 12.5 — the **`a` → `b` step
of 14 people is larger than the noise; the `b` → `c` step of 6 is not**. So its headline is
half-supported and half inside the band, and nothing about which half was knowable from a
single run of each. Its `lite3` column, in the stable metric, is fine.

## ⛔ And the recipe was never the blocker

Two facts about the corpus, and together they say the same thing.

**The best checkpoint this project has produced finds the Lite3 in 17 of 36 frames of the
morning it was trained on, while its cross-day person retention sits at parity with the
shipped network.** 47% recall on a held-out time block of the same six tripod shots — one
room, thirteen minutes, 456 distinct views, 0.0–1.0 px of median camera displacement —
predicts nothing about a Tuesday in another room.

**And the gate cannot resolve a training change at all.** Person retention is measured on
284 cross-day frames, the incumbent keeps 25 of them, and re-running one arm moves that
number by up to 12.5. A metric whose run-to-run spread is half its own baseline cannot rank
recipes — which is why five carefully matched arms, 743 checkpoints and two variables
separated at equal gradient steps produced no usable ranking. **You cannot tune what you
cannot measure**, and four waves have now tried.

So the next useful measurement is not another sweep of the recipe. It is more data, of a
kind this corpus does not contain — more viewpoints, a second room, a second day — which
widens the eval set as much as the training set and is the only thing that makes the gate
sharp enough to decide anything.

A venue recording session is scheduled for **Monday 31 August 2026** at the MGM Shanghai
West Bund hotel to break exactly that, and
[`RECORDING-TRAINING-FOOTAGE.md`](../../robot-stack/deep_robotics/lite3/RECORDING-TRAINING-FOOTAGE.md)
§2 and §3 — move the camera, come back on a second day — are what it should be run against.
**Nothing in this directory is recommended for deployment.**

## Why this is a sibling directory and not more files in wave 7's

[`evidence/2026-08-27-lite3-training-set`](../2026-08-27-lite3-training-set/README.md)
built the dataset and this wave does not change one frame, one box or one synthetic image
of it. Two concrete things decided the split rather than a preference:

* **`summarise_scores.py` globs `scored_*.json` beside itself.** Dropping five arms of
  224-trained scores into that directory would put them under wave 7's headings and its
  `RUN_LABEL` map, and its published table would silently start mixing checkpoints trained
  at two different squares. The breakage is in a committed script, not hypothetical.
* **An evidence directory is an archive of one run.** Wave 7's README is a finished
  argument about *how the labels were made*; this is an argument about *what to train on*.
  Appending would make the first one harder to read and the second one impossible to date.

What is shared is shared by reference, not by copy: `subsample_synthetic.py` reads wave 7's
own manifests from `../2026-08-27-lite3-training-set/`, `score_ratio_wave.sh` runs wave 7's
own `score_checkpoints.py`, and `audit.py` checks that scorer's hand-copied preprocessing
table against `inference_profile.py` itself. Nothing here is a fork of anything there.

```bash
python3 audit.py            # every number below, from committed JSON. No video/model/net.
python3 summarise_ratio.py  # the score tables
```

## What moved, and what did not

Every number is at a **named** preprocessing, and the three are different detectors on one
robot (#129/#147). The incumbent row is the same `mnssd22` weights every arm starts from,
re-scored here by the same script rather than recalled.

| | `go2-peer-supervised` 224/0.25 | `go2-navigator-default` 300/0.4 | `go2-run-smoke` 300/0.45 |
| --- | --- | --- | --- |
| **incumbent** | 0/36 lite3, **25**/284 people | 0/36, **24**/284 | 0/36, **20**/284 |
| 1:1 @224 | ep037 — 15/36, **25** (+0) | ep005 — 3/36, 29 (+5) | ep005 — 1/36, 22 (+2) |
| 1:3 @224 | *loses people at every epoch* | *every epoch* | *every epoch* |
| 1:9 @224 | *loses people at every epoch* | *every epoch* | *every epoch* |
| **1:1 @300** | ep022 — 6/36, 26 (+1) | **ep031 — 17/36, 25 (+1)** | **ep039 — 17/36, 22 (+2)** |
| 1:9 @300 | ep001 — 1/36, 25 (+0) | *loses people at every epoch* | *every epoch* |

Best `lite3` **among epochs keeping at least the incumbent's people**. The rule matters: the
argmax of `lite3` alone picks `1:9 @224` ep016 at 21/36 — which keeps **5 of 284 people** and
is not a candidate for anything.

### ⚠️ Two of these rows are the incumbent barely damaged, and two are not

`ep001` and `ep005` are the base weights a few hundred steps in. A row reading "ep005 —
3/36, 29 people" is the gate saying *nothing here yet*, and it should not be read as a
trained gain. The loss curves say which is which:

| arm | epoch | conf loss | vs its own ep001 | verdict |
| --- | ---: | ---: | --- | --- |
| 1:1 @300 | ep001 | 4.9932 | — | untouched |
| 1:1 @224 | ep005 | 3.7622 | 5.4208 → 3.7622 | barely moved |
| **1:1 @300** | **ep031** | **1.8766** | 4.9932 → 1.8766, **2.7x** | trained |
| **1:1 @224** | **ep037** | **1.6736** | 5.4208 → 1.6736, **3.2x** | trained |
| 1:1 @300 | ep144 | 0.8707 | 5.7x | trained to the end |

The two candidates worth arguing about — `1:1 @300` ep031 and `1:1 @224` ep037 — are a third
of the way into a 144-epoch cosine schedule with the classification loss down 2.7x and 3.2x.
Neither is an untouched network.

### ⚠️ How the floor sensitivity looked before the arms were re-run

On one run per condition, these two candidates looked like different kinds of result — one
clearing by exactly zero, one surviving a floor three people higher:

| candidate, run 1 only | epochs clearing the floor | best `lite3` at floor +0 / +1 / +2 / +3 |
| --- | ---: | --- |
| `1:1 @224` at 224 px / 0.25 | 4 of 144 | 15/36 → 11/36 → **1/36** → 1/36 |
| `1:1 @300` at 300 px / 0.4 | 26 of 144 | 17/36 → 17/36 → 15/36 → **13/36** |
| `1:1 @300` at 300 px / 0.45 | 34 of 144 | 17/36 → 17/36 → 17/36 → **14/36** |

⛔ **Both readings are draws.** Re-running each arm twice more gives 4 / 1 / 2 clearing
epochs for the first and 26 / 9 / 126 for the second. The contrast this table appears to
draw — fragile versus robust — is not a property of the two conditions; it is two samples
from two wide distributions. It is kept because it is what a single-run sweep produces, and
because every published table in this project's history looks exactly like it.

Two things are called noise and only one applies to a checkpoint: *re-scoring* epoch 037
returns 25 people every time, because `cv2` runs a fixed graph over fixed frames. The
variation is in **re-running the training**, and — see the top of this document — that is not
a seed effect that a seed could fix.

### ⛔ The 224 result is falsified by its own replicate

`r1x1_224`, three seeds, everything else identical — same manifest, same square, same 144
epochs, same hyperparameters, same committed trainer, and the same 282 teacher boxes over
258 frames in all three, because that pass is deterministic. Scored at 224 px / 0.25:

| seed | epochs clearing 25 people | best `lite3` under the gate | most people any epoch |
| ---: | ---: | --- | ---: |
| 0 | 4 of 144 | 15/36 at ep037 | 34/284 |
| 1 | **1 of 144** | **0/36, and it is ep001** | 27/284 |
| 2 | 2 of 144 | 12/36 at ep026 | 26/284 |

| seed | floor +0 | +1 | +2 | +3 |
| ---: | --- | --- | --- | --- |
| 0 | 15/36 | 11/36 | 1/36 | 1/36 |
| 1 | **0/36** | 0/36 | 0/36 | none |
| 2 | 12/36 | 1/36 | none | none |

**One seed in three produces nothing but the base weights.** The best `lite3` under the gate
across three runs of one recipe is 0/36, 12/36 and 15/36 — a spread wider than the difference
between this wave and wave 7. A pass that a re-run does not reproduce is a property of the
seed, and 15/36 was the number this document was going to lead with.

That is what the ±1–3 band was warning about, and it is worse than the band suggests: the
band is quoted on the person *count*, and here it moves the person count by enough to change
**which epochs exist** under the floor, which then moves the `lite3` number by 15.

⛔ **So nothing at 224 px is recommended, and the reason is replication rather than
judgement.** Had only seed 0 been run — which is what every previous wave in this project
did, including wave 7 — this directory would have published a 15/36 checkpoint that clears
the production gate.

## The five arms

The dataset is wave 7's, untouched. What changes is how much of its synthetic half each arm
sees, and the square the trainer warps to.

| arm | real : synthetic | px | epochs | frames/epoch | steps/epoch | total steps | priors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r1x1_224` | 1 : 1.00 | 224 | 144 | 884 | 36 | 5,184 | 1,014 |
| `r1x3_224` | 1 : 3.00 | 224 | 87 | 1,450 | 60 | 5,220 | 1,014 |
| `r1x9_224` | 1 : 8.98 | 224 | 40 | 3,143 | 130 | 5,200 | 1,014 |
| `r1x1_300` | 1 : 1.00 | 300 | 144 | 884 | 36 | 5,184 | 1,917 |
| `r1x9_300` | 1 : 8.98 | 300 | 40 | 3,143 | 130 | 5,200 | 1,917 |

Everything else is `k_full_pseudo03`'s recipe, unchanged and unchosen:
`--freeze-through= --backbone-lr-scale 0.5 --pseudo-labels 0.3 --distil 0.1`, batch 24,
learning rate 1e-3, and **no `--motion-blur`/`--sensor-noise`/`--composite`** — those cost
the most people in every prior run, and turning them on here would confound the ratio.

### ⛔ The epoch counts are the experiment, not a knob

Wave 7 ran a 283-positive arm and a 2,825-positive arm for the same 40 epochs. Its `a` took
**a tenth of the gradient steps** its `b` did, so "real only" and "barely trained" moved
together and the ablation could not separate them.

Here every arm takes **~5,200 steps, within 0.7%**, and `CosineAnnealingLR` anneals over
each arm's own epoch count — so the learning rate as a function of *step* is the same curve
in all five and the composition of the training set is the only thing that differs. Each
arm's own epoch 40 is still on disk and still scored, so the equal-*epoch* reading is
available; it is just not the headline. `audit.py` recomputes the step counts from the
manifests **and** from what each arm printed at startup, and fails if the two disagree.

### ⚠️ One number moves three shares, and the third one is not synthetic

The 318 in-domain negatives are identical in every arm — same files, same count — so
cutting the synthetic half does not only cut the synthetic half. It changes what a batch is
made of:

| arm | real positives | synthetic | negatives | real frames seen per arm |
| --- | ---: | ---: | ---: | ---: |
| 1:1 | **32.0%** | 32.0% | **36.0%** | 144 passes |
| 1:3 | 19.5% | 58.6% | 21.9% | 87 passes |
| 1:9 | 9.0% | 80.9% | 10.1% | 40 passes |

**The negatives are person supervision.** They are the same room with no quadruped, they
deliberately include frames containing people, and `--pseudo-labels 0.3` turns the shipped
network's detections on them into old-class ground truth. So a 1:1 arm does not merely see
less synthetic data per step — it sees **3.6x more of the frames that hold `person` in
place**. If the ratio moves the person column, that share is the most likely mechanism, and
it is not separable from "less synthetic" by any arm in this wave.

Holding the negative share fixed would have meant *deleting* negatives from the 1:1 arm —
down to 32 of them — which degrades it for a different reason. It was not done, and this
paragraph is what stands in for the arm that would have shown it.

At matched steps the 1:1 arm also makes **144 passes over the same 283 real images** against
the 1:9 arm's 40. More exposure to less data is what a ratio change *is*; the equal-epoch
rows (each arm's own epoch 40) are the reading where that exposure is matched instead.

### The three ratios are NESTED, so this is an ablation and not three datasets

`make_synthetic.py` writes up to three variants (`_0`, `_1`, `_2`) of each of three families
for each of the 283 real parents, and every parent has a `_0` in all three families —
checked, not assumed. So:

| arm | rule | per family |
| --- | --- | --- |
| 1:9 | every synthetic record | shear 849, colour-slice 849, occlude 844 |
| 1:3 | variant `_0` of all three families | 283 / 283 / 283 |
| 1:1 | variant `_0` of ONE family per parent, rotating | 94 / 95 / 94 |

`1:1 ⊂ 1:3 ⊂ 1:9`, no parent is dropped, no family is dropped, and no RNG is involved. At
1:1 the family **rotates by sorted parent index** rather than being picked, because keeping
"the kinds as they are" while there is one slot per parent means spreading them; choosing
one family would have confounded *how many* with *which kind*.

## Training at 224 is a different network, not a resize

`finetune_ssd.py` had no `--input-size`: `INPUT_SIZE = 300` was a module constant used by
the prior generator, the mirror check, the teacher pass and the dataset resize. This wave
threads a parameter through all four, and `audit.py` greps for a half-done version of that
edit, because priors read at 224 with the dataset still resizing to 300 would train the loc
head against boxes `DetectionOutput` never emits and no existing check would catch it.

Two consequences are measured rather than assumed, and both are in every arm's own log:

**PriorBox takes its `img_width` from the data blob.** A `min_size` written in absolute
pixels is a larger fraction of a smaller frame — the 60 px prior that covers 0.20 of a 300
px frame covers 0.27 of a 224 px one — and the head-source feature maps go from 19/10/5/3/2/1
cells to 14/7/4/2/1/1. **1,917 priors at 300 against 1,014 at 224.** The torch mirror still
reproduces `cv2` at 224, to 6.4e-4 against a 2e-3 tolerance; that check runs at the training
square at the start of every arm, and it is the reason this was attempted at all.

**⚠️ The teacher runs at the training square too, and that is a confound inside "224 vs
300".** `--pseudo-labels 0.3` carries the *shipped network's own* detections as old-class
ground truth, and it is the only thing standing between a fine-tune and a robot that has
stopped seeing people. On identical training frames:

| arm | identical training frames | teacher boxes at 224 | at 300 | |
| --- | ---: | ---: | ---: | --- |
| 1:1 | 884 | 282, over 258 frames | 330, over 309 | **15% less** old-class supervision at 224 |
| 1:9 | 3,143 | 458, over 427 frames | 617, over 571 | **26% less** at 224 |

That is not a bug and it is not avoidable by keeping the teacher at 300 — a box the shipped
network cannot find at 224 is a box the student cannot be taught to find at 224 either. It
does mean the 224 arms get less person supervision than the 300 arms, and any 224-vs-300
difference in the person column is partly this.

## ⚠️ The single-run tables, kept because they are what was nearly published

Everything in this section is **one run per condition** and is superseded by the replicates
above. It is retained rather than deleted because the argument of this document is that
tables of exactly this shape — carefully matched, honestly computed, internally consistent —
were not enough, and a reader should be able to see the version that looked convincing.

### Which variable appeared to move the people

Both readings below come from **every epoch of every arm**, not from one selected epoch,
because an argmax over 144 noisy numbers finds a high one whether or not the arm is any
good. `summarise_ratio.py` prints them and four more.

**Epochs clearing the incumbent's people, out of the arm's own epochs.** A count cannot be
won by a lucky epoch:

| arm | 224/0.25 | 300/0.4 | 300/0.45 |
| --- | ---: | ---: | ---: |
| 1:1 @224 | 4/144 (2.8%) | 1/144 | 1/144 |
| 1:3 @224 | **0/87** | **0/87** | **0/87** |
| 1:9 @224 | **0/40** | **0/40** | **0/40** |
| **1:1 @300** | **13/144 (9.0%)** | **26/144 (18.1%)** | **34/144 (23.6%)** |
| 1:9 @300 | 1/40 — and it is ep001 | 0/40 | 0/40 |

**Median people over all epochs**, against the incumbent at the same preprocessing:

| arm | 224/0.25 (inc. 25) | 300/0.4 (inc. 24) | 300/0.45 (inc. 20) |
| --- | ---: | ---: | ---: |
| 1:1 @224 | **12.5** | 13.0 | 10.0 |
| 1:3 @224 | 4.0 | 7.0 | 5.0 |
| 1:9 @224 | 3.5 | 9.5 | 6.5 |
| 1:1 @300 | **10.0** | **19.0** | **15.0** |
| 1:9 @300 | 3.0 | 7.0 | 5.0 |

**The ratio moves the person column by 6 to 12 people; the square moves it by 0.5 to 2.5,
which is inside the ±3 band.** At the production square, going from 1:9 to 1:1 takes the
median from 3.5 to 12.5 at 224 and from 3.0 to 10.0 at 300 — the same effect, the same size,
on both sides of the resolution question. Going from 224 to 300 at a fixed ratio moves it
2.5 (at 1:1) and 0.5 (at 1:9). One of those is a result and the other is not.

**1:3 is not halfway.** It clears the floor at **no epoch of any profile**, and its median
person retention (4.0 at production) is nearer 1:9's 3.5 than 1:1's 12.5. Whatever this is,
it is not linear in the ratio, and 1:1 is not "a bit better than 1:3" — it is the only arm
in the sweep that keeps people at all.

## ⛔ Did training at 224 change Lite3 detection at 224? Yes — and it is the smaller finding

Separate question, separate answer. Scored at 224 px / 0.25 in both cases:

| | trained at 224 | trained at 300 | difference |
| --- | ---: | ---: | ---: |
| median `lite3` over all epochs, 1:9 | **19/36** | 12/36 | **+7** |
| median `lite3` over all epochs, 1:1 | **16/36** | 10/36 | **+6** |
| best epoch, 1:9 | **21/36** | 16/36 | **+5** |
| matched-step endpoint, 1:9 | **19/36** | 15/36 | **+4** |

**Training at the square the peer launcher opens at is worth roughly +6 of 36 frames — about
17 points of same-session recall — and it is consistent across ratios and selection rules.**
The train/deploy mismatch was real and it was costing detection. It simply was not what
destroyed the people, which is what wave 7 assumed and what this wave was run to test.

`r1x9_300` also reproduces wave 7's `b` closely enough to trust the comparison: 16/36 best
at production against `b`'s 15/36, from a different trainer and a different RNG stream. That
+1 is inside the noise, which is the point of running it contemporaneously rather than
citing it.

⚠️ And it does not point the same way as the person column. At the **300 px** deployments,
training at 300 keeps *more* people (median 19.0 vs 13.0 at `go2-navigator-default`), which
is consistent with the teacher confound above — 15% more old-class supervision at 1:1. So
the best candidate in this wave is trained at 300 and scored at 300, and the 224 alignment,
though real, buys detection at a cost in the column that is the gate.

## ⛔ What is recommended: nothing, and the reason is not the person column

Three separable statements, because collapsing them is how the last four waves each produced
a different answer.

**1. A capability is real and replicated.** At both 300 px deployments, three runs out of
three of `r1x1_300` produce a trained checkpoint that finds the Lite3 while keeping at least
as many cross-day people as the shipped network, which finds it in 0 of 36. That is the first
thing this project has produced that survived an attempt to break it.

**2. No checkpoint should be deployed, and no number here should be quoted as a value.** The
same three runs give 17, 8 and 14 of 36 — a factor of two. And every one of those numbers is
**same-session**: a held-out time block of the same six tripod shots, one room, thirteen
minutes, 456 distinct views, 0.0–1.0 px of median camera displacement. 47% recall on the
morning it trained on is not a detection rate for a demo, and the only cross-day column is at
parity rather than ahead.

**3. Nothing at 224 px works, which is the square production launches.** Three runs give 6, 2
and 1 of 36 under the gate, and the third is `ep005` — the base weights barely moved.
`deploy/run-peer-supervised.sh` cannot see whatever the 300 px arm learns. Issue #129's split
is therefore still open and is now costed: the checkpoint that clears the gate does so only
under launchers that pass no `--input-size`.

**If someone runs this anyway**, the checkpoint is `runs/r1x1_300/epoch031.caffemodel` on the
training host, and **the launcher decides whether it passes or fails its own gate**:

| the same file, launched three ways | `lite3` | people | vs incumbent |
| --- | ---: | ---: | --- |
| `go2-navigator-default` — 300 px / 0.4, any launcher passing no `--input-size` | **17/36** | **25**/284 | 24 → **+1, passes** |
| `go2-run-smoke` — 300 px / 0.45 | 14/36 | 23/284 | 20 → +3, passes |
| `deploy/run-peer-supervised.sh` — 224 px / 0.25 | 8/36 | 21/284 | 25 → **−4, fails** |

One `.caffemodel`, three deployments, clearing the gate at two of them and losing four people
at the third. That is #147's argument as a shipping decision: there is no "the" score for a
checkpoint, and a recommendation without a launcher name attached is not one. Its 17/36 is
same-session either way, and its two replicates scored 8/36 and 14/36 at the same profile.

## ⚠️ Which of these numbers has a day boundary in it

| column | what it is | generalises? |
| --- | --- | --- |
| `lite3` | 36 frames, a held-out **time block** of the same six tripod shots the training frames come from — one room, one morning, 13 minutes, 456 distinct views, 0.0–1.0 px of camera motion | **no.** Same session. This project has measured 0/705 same-session against 60/159 cross-day for one model |
| `people` | 284 frames of the 2026-08-20 Go2 manifest — **another day, another building** | yes, and it is the gate |

Every conclusion about the **Lite3** in this document is same-session. Every conclusion about
**people** is cross-day. Person differences within ±3 are inside this project's own
run-to-run noise, measured on the duplicate r/s pair of the Go2 corpus, and
`summarise_ratio.py` marks them in its own output rather than leaving it to the reader.

## Two findings about the code, neither fixed here

**`detector/finetune_ssd.py` on `main` is not the file the previous waves ran.** The Spark's
copy is 1,000 lines and carries `motion_blur`, `sensor_noise` and `composite_onto_background`
— the operators behind `--motion-blur/--sensor-noise/--composite`. The committed copy is 811
lines and implements none of them; the flags were **never committed**. Function for function
the committed copy is a strict subset (three extra functions on the Spark, three differing
lines). Every arm here runs the **committed** trainer, which is what makes "every number
from a committed script" true — and it is also why `r1x9_300` is a contemporaneous control
rather than a citation of wave 7's `b`. `collect_run_facts.py` records the md5 of all four
source files the training host ran and `audit.py` checks them against this pull request.

**`ssd_torch.verify_against_cv2` has no call site in the repository.** `export_caffemodel`'s
docstring says the export "is checked immediately afterwards by re-running
`verify_against_cv2` on the file that was written, which is the only evidence that the robot
would run what was trained", and `UNFROZEN-FINE-TUNE.md` publishes a table of its results.
`grep -rn verify_against_cv2 --include=*.py` finds the definition and nothing else. The
table's numbers are real; whatever produced them is not in the tree. This wave gives that
function the `input_size` parameter a fix would need and does not otherwise touch it, because
wiring it in would change the trainer mid-wave.

## Files

```bash
python3 audit.py            # every number below, from committed JSON. No video/model/net.
python3 summarise_ratio.py  # seven tables, both selection rules, and the replicate spread
```

| file | what |
| --- | --- |
| `audit.py` | recomputes every number here from committed JSON: the scorer's preprocessing table against `inference_profile.py`, the arms' step counts against what each arm printed, the committed trainer's md5 against the training host's, and the determinism probe's two claims. Mutation-tested. |
| `probe_determinism.sh`, `determinism.json` | **the headline** — five identical invocations, five different checkpoints |
| `subsample_synthetic.py` | the two new ratio manifests. No RNG; `--check` prints the nesting proof |
| `run_ratio_wave.sh` | the five original arms |
| `run_seed_replicates.sh`, `run_seed_replicates_300.sh`, `run_seed_replicates_1x9.sh` | the six replicate arms — three separate files because each one's bytes are the record of what it ran |
| `score_ratio_wave.sh`, `score_seed_replicates*.sh` | every checkpoint at every deployed preprocessing, through wave 7's own `score_checkpoints.py` |
| `collect_run_facts.py` | what each arm printed at startup — priors, teacher boxes, steps — plus the trainer hashes |
| `summarise_ratio.py` | the tables |
| `select_for_publication.py` | which checkpoints go to the corpus repo, **by rule**, and what is left behind |
| `publish_to_hub.py` | the upload. Token from stdin; refuses if the repo is not private |
| `lite3_train_r1x1_20260827.json`, `lite3_train_r1x3_20260827.json` | the two new manifests |
| `run_facts.json`, `published_checkpoints.json`, `scored_*.json`, `incumbent_*.json` | eleven arms at three profiles, every epoch |

No video, no frames and no weights are committed to git. **1,111 checkpoints** live on the
training host under `~/lite3-ratio-20260827/runs/`, one directory per arm, with each arm's
`history.json` and `pseudo_labels_{224,300}.json`.

**359 of them, the dataset, and every scored JSON are published** to the private dataset
repo `armwaheed/go2-peer-detection` under `lite3_20260827/`, which already holds
`aug20_crossday/` — the 284-frame corpus every person number here is measured against.
Weights and the frames they were scored on sit behind one access grant.
`published_checkpoints.json` names every checkpoint that was **not** published and the rule
that excluded it; `select_for_publication.py` is that rule. A negative result whose weights
are gone cannot be checked by anyone, and — see the top of this document — re-running the
trainer does not reproduce them.

⚠️ **Two completion markers, in two different files**, which cost one session an hour of
waiting on the wrong one: `run_ratio_wave.sh` writes `LITE3_RATIO_WAVE_DONE` to `wave.log`,
and `score_ratio_wave.sh` writes `LITE3_RATIO_SCORING_DONE` to whatever its caller
redirected. Grepping the wrong file reads as "still running" rather than as an error.

⚠️ **`~/.cache/huggingface/{xet,hub,datasets}` are root-owned on the training host**, so the
Xet uploader cannot write its chunk cache and dies with `Permission denied (os error 13)`
seconds into a transfer. `publish_to_hub.py` points `HF_XET_CACHE` and `HF_HUB_CACHE` at a
user-owned directory. The token file itself is the user's and is readable.
