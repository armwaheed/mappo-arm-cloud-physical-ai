<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The fine-tune clears the frozen-feature ceiling, and still loses to the stock model

`FROZEN-FEATURE-CEILING.md` said what would work: unfreeze the backbone, multibox loss,
prior matching, hard negative mining, augmentation, then export back through the
`.caffemodel` path. `finetune_ssd.py` is that, and on the axis that page named, it delivers —
**72% cross-day recall at a 4% false-positive rate, against 74% at 60%.** The linear-probe
limit is real and unfreezing lifts it.

> ⛔ **It is still not worth deploying, and this page ends in a recommendation not to.**
> Scored against the robot's own **stock, unmodified** 21-class weights over the same
> 2,800-frame Aug-24 capture — the parsimony test in
> [`eval_class_agnostic.py`](eval_class_agnostic.py), which takes any VOC label as an
> obstacle because the planner needs a box and not a name — the best checkpoint here gets
> **53% recall at 38% false alarms** against the stock model's **64% at 18%**. It loses on
> both axes, from a training run that cost a day of GB10 time, to a file that was already on
> the robot. See *The comparison that ended this* below.
>
> ⚠️ **Corrected 2026-08-26. Both rows in that block are the Aug-24 capture, which is this
> page's own training day, and the stock model had never been scored on the held-out day at
> all.** Scored on the Aug-20 split with one rule applied to every model, the shipped weights
> get **68% recall at 57% false alarms** and *every* checkpoint on this page beats them on the
> peer. The half of the recommendation that survives is the other half: every one of those
> checkpoints loses people the shipped network keeps, and none of them is deployable. See
> *Correction, 2026-08-26* below. The block is left standing because it is what was believed,
> and on what evidence.
>
> ⛔ **And the checkpoint this page recommends is not the best one it produced.** Sixteen runs
> x 40 epochs left **640 checkpoints** on disk; thirteen model/epoch rows had ever been scored.
> 64 were scored on 2026-08-26 and the winner is `k_full_pseudo03`, a run no table on this page
> names; a finer pass over it reads **89% peer recall at 12% false alarms**. It keeps 17 of 22
> people, so it does not clear the gate either. See *The sweep finished* below and
> [`evidence/2026-08-26-checkpoint-sweep/`](../evidence/2026-08-26-checkpoint-sweep/).

Two things on this page outlive the recommendation, and both are the reason it is being
merged rather than deleted: **fine-tuning on a one-class corpus teaches the network that
people are background**, which is a hazard for any future run over a partially-labelled
corpus; and **47 held-out positives could not rank anything**, which is why the number in the
first paragraph and the number in the block above are the same checkpoint.

## ⛔ The comparison that ended this

Same weights, same `cv2.dnn` path, same 2,800 frames of the Aug-24 capture — 1,903 with a
hand-labelled peer, 897 peer-free — at 0.25, counting a hit as any detection overlapping the
labelled box at IoU >= 0.30:

| | reads | box on the peer | fires on a peer-free frame |
| --- | --- | --- | --- |
| **stock 21-class, unmodified** | **any VOC label** | **64%** of 1,903 | **18%** of 897 |
| this fine-tune, `runs/best/epoch015` | its own `go2wheel` | 53% | 38% |

Labels the stock model puts on the peer: `motorbike` 613, `chair` 372, `aeroplane` 200,
`person` 109. Nonsense names, correctly placed boxes.

**The fine-tune's row is the generous one.** 1,343 of those 1,903 peer frames and 705 of
those 897 peer-free frames were in its own training set. The stock weights have never seen a
frame of the corpus, so its row is clean out-of-sample and the trap that invalidated the
refusal gate — negatives sharing a session with the training frames — cannot apply to it.
Reading the fine-tune class-agnostically as well would only add its remaining twenty classes'
alarms on top of the 38%.

### ⚠️ 72% and 53% are the same checkpoint

`runs/best/epoch015` scores 72% recall at 4% false positives on the 47-positive Aug-20 test
split, and 53% at 38% on 1,903 Aug-24 positives it was largely trained on. A model does not
normally do *worse* on its own training day than on a held-out one, and no reading of the two
sets reconciles them as a ranking. The safe reading is the one the denominators force: 47
positives and 136 negatives could not rank anything, and 1,903 and 897 can. Every table on
this page below is the 47-positive split, and should be read as ordering runs against each
other rather than as a claim about the world.

That is the same failure as the refusal gate one level up, and the same failure as the
fifteen stills one level down: **the evidence set was too small and too like itself, three
times in a row, and each time it flattered the thing being tested.**

## ⛔ Correction, 2026-08-26: the comparison that ended this was never scored cross-day

Both rows of the table above are the Aug-24 capture. **1,343 of those 1,903 positives and 705 of
those 897 negatives are this fine-tune's own training frames** — the section above says so, calls
its own row "the generous one", and then rules on it anyway. The one number on this page that is
honestly out-of-sample for the fine-tune is the Aug-20 split, 72% at 4%, and **the stock model had
no Aug-20 number to put beside it.** Nobody had ever run it there. So the recommendation compared a
training-day row against a training-day row, with the cross-day row of one model and no cross-day
row of the other, and the section above it admits the two cannot be reconciled.

It has now been run. **Aug-20 held-out split: 47 peer-present frames and 134 peer-free**, scored
through the deployed `cv2.dnn` path at the prototxt's own `confidence_threshold: 0.25`, with one
rule applied identically to every row — **a frame counts as a fire if any detection at >= 0.25 has
box aspect h/w < 2.0.** That is not person-shaped, which is what `mappo_bridge` routes to the
policy as an obstacle instead of holding for (`person_detector.PERSON_ASPECT_MIN`, PR #73), so the
column below is the thing the deployed stack would actually do. `people kept` counts, out of the 15
test frames where the **shipped** network detects a person at >= 0.45 — the confidence the
navigator runs at — how many the candidate still detects.

| model | peer recall | false alarms | precision | people kept |
| --- | ---: | ---: | ---: | ---: |
| **stock 21-class, as shipped** | **68%** (32/47) | **57%** (76/134) | 30% | **15/15** |
| `l_full_bb02` ep015 | 70% | 19% | 56% | 13/15 |
| `j_full_distil03` ep015 | 66% | 6% | 79% | 11/15 |
| `i_full_pseudo` ep020 | 74% | 8% | 76% | 8/15 |
| `f_full_distil01` ep020 | 74% | 1% | 95% | 4/15 |

**On the peer, every fine-tune beats the shipped weights.** Recall 70, 66, 74 and 74 against 68 —
three of the four above it and the fourth two points under. False alarms 19, 6, 8 and 1 against
**57**. Precision 56-95% against 30%. There is no reading of the peer axis on which the shipped
weights win, and the recommendation below is wrong on the axis it was argued.

### ⚠️ 18% and 57% are the same model on two different negative sets

The 18% that this fine-tune was measured against is not a property of the network. **705 of those
897 Aug-24 peer-free frames are the `neg_prone` and `neg_standing` sets** — corridor deliberately
cleared and shot as training negatives. The Aug-20 negatives are the same corridor furnished and
in use, ArUco office chair included, and the same weights at the same threshold through the same
code path fire on 57% of them.

That is this page's own lesson pointed the other way — *a test that shares its conditions with the
thing it is testing measures nothing*, and negatives staged to be empty share a condition with
nothing the robot drives through. An empty-corridor false-alarm rate is not a false-alarm rate for
a room.

### The half of the recommendation that holds, and it is the safety half

**Every fine-tune buys its peer numbers with people.** `f_full_distil01` — the best peer precision
on the table, 95% — keeps **4 of the 15** people the shipped network sees at the navigator's own
0.45. The warning below, *Fine-tuning on a one-class corpus teaches the network that people are
background*, was right and understated: it was written from 2 of 15 lost on one checkpoint, and
the spread across four checkpoints is 2 to 11 of 15.

**So nothing on this page is deployable, and nothing above changes that.** A checkpoint that finds
more peers and drops a person is not an improvement to a stack whose first job is not to walk into
someone. The open objective is the one no run has hit: **hold every person the shipped network
sees while keeping the peer gains.**

That wave-5 sweep has now finished, and so has a sweep of the checkpoints that already existed.
**Neither cleared the gate.** Of 64 checkpoints scored, the best keeps **20 of 22** people and is
*below* the shipped weights on peer recall; every checkpoint that beats them on both peer axes
loses at least three. See *The sweep finished* below.

### ⚠️ Read the denominators before quoting any of this

**47 positives and 134 negatives, one corridor, one day.** The same objection that invalidated the
refusal gate (0 of 705 on same-session negatives), the fifteen stills (nine of which held no peer),
and the 47-positive ranking above applies to the table in this section: it is small, and it is one
building. What it is good for is the comparison it makes — every row scored the same way on the
same frames on the same day — and not for the absolute values.

~~**Two frames are unaccounted for.** The manifest's test split holds **136** peer-free frames and
this sweep scored **134**; the difference is not explained here.~~ **Explained by
[#91](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/pull/91), 2026-08-26.** The
manifest in this repository was right and the *derived* split on the training host — the file the
evaluator actually opens — had its clip indices shifted by one: it named `peer_baseline_045.jpg`
and `smoke1_058.jpg`, neither of which exists, and omitted `peer_baseline_000.jpg` and
`smoke1_000.jpg`, which do. Repaired and all thirteen rows rescored, **every numerator is
identical** and only the denominator moved: the shipped weights go from 76/134 = 57% to
**76/136 = 56%**, and every other rate by at most one point. The table above is left at 134 and
57% because that is what it was measured on; the 2026-08-26 sweep below is scored on the
repaired 136 and reads 56%.

## ⛔ The sweep finished, and the winner is a run this page never names — 2026-08-26

Everything above ranks the handful of checkpoints somebody happened to score. **Sixteen runs at
40 epochs each is 640 checkpoints, and thirteen model/epoch rows had ever been evaluated** — the
thirteen [#91](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/pull/91) rescored when it
repaired the split, one of which is the shipped model itself. So at least **627 of the 640 had
never been evaluated**, and wave 5 was launched before they were.

64 of them have now been scored — epochs 10/20/30/40 of every run — against the shipped weights,
on the Aug-20 cross-day split, under this page's own rule. **The best detector was already on
disk.** `git grep` finds four of the sixteen run names in this repository; the winner is one of
the twelve it does not.

**Aug-20 held-out split, 47 peer-present and 136 peer-free** (the denominator #91 repaired), one
rule for every model — a frame fires if any detection at >= 0.25 has box aspect h/w < 2.0.
`people` counts, of the **22** frames where the shipped network sees a person at **0.25**, how
many the candidate still sees. 0.25, not the 0.45 this page quotes elsewhere, because 0.25 is
what `deploy/run-peer-supervised.sh` launches the peer runs with; the denominator grows from 15
to 22 and the gate gets harder, not easier.

| model | peer recall | false alarms | people |
| --- | ---: | ---: | ---: |
| **stock 21-class, as shipped** | 68% (32/47) | **56%** (76/136) | **22/22** |
| `k_full_pseudo03` ep022 | **89%** | 12% | 17/22 |
| `k_full_pseudo03` ep020 | 85% (40/47) | 10% (14/136) | 17/22 |
| `k_full_pseudo03` ep017 | 83% | 12% | **19/22** |
| `k_full_pseudo03` ep010 | 77% (36/47) | 5% (7/136) | 18/22 |
| `p_bb02_d01_aug` ep020 | 70% (33/47) | 18% (25/136) | **19/22** |
| `l_full_bb02` ep040 | 72% (34/47) | 21% (29/136) | 16/22 |
| `f_full_distil01` ep020 | 74% (35/47) | **1%** (2/136) | 5/22 |

**13 of the 64 beat the shipped weights on both peer axes, and all 64 beat them on false alarms
alone** — 56% against a candidate range of 0% to 55%. The full 64-row grid, the raw
`sweep_all.json` and both figures are in
[`evidence/2026-08-26-checkpoint-sweep/`](../evidence/2026-08-26-checkpoint-sweep/).

⚠️ **`k_full_pseudo03` ep022 and ep017 are not in the archived sweep data.** That file is the
coarse pass — four epochs per run, 64 rows. The two odd epochs come from a finer pass over the
winning run made afterwards on the training host, and are recorded as attributed rather than
verified from this repository. The archived data also reads **5/22** for `f_full_distil01` ep020
where the published table says 4 — 4 is that run's ep010 value, one row up.

### The lever is `--pseudo-labels`, and this page already identified it

`k_full_pseudo03` is the **only run of the sixteen at `--pseudo-labels 0.3`**. Every other run
used 0.5, or carried no old-class labels at all.

The mechanism is the one measured further down this page, under *What fixes it is labelling the
old objects*: the threshold decides how many of the starting network's own confident detections
are carried into training as old-class ground truth, and nearly all of them are `person`. Lower
threshold, more `person` supervision — which is precisely the axis every run has failed on. That
section measured 0.5 to 0.3 as worth five `person` detections **and** two points of new-class
recall, and called it *"the single most effective knob in this work"*.

**It was right, and the knob was then never turned again.** Nobody had ever swept below 0.3.
A wave doing exactly that — 0.1 / 0.2 / 0.3, with a paired control on the winner's configuration —
is running as this is written, and **no conclusion about its outcome is recorded here.**

### Augmentation moved the metric, against a contemporaneous control

`l2_bb02_d01_control` and `p_bb02_d01_aug` share every hyperparameter — `--backbone-lr-scale 0.2
--pseudo-labels 0.5 --distil 0.1`, 40 epochs, same corpus — and differ in exactly three
augmentations no earlier wave ran: **motion blur, sensor noise, and compositing the peer onto a
peer-free frame.** `l2` is a control run in the same wave rather than a citation of `l_full_bb02`,
which it had to be: the new operators call `rng.random()` even at probability 0, so the stream
shifted and the earlier runs are no longer byte-reproducible under wave-5 code.

Epoch-matched at 040, both at **68% peer recall: 15/22 people to 18/22, +3**, for 4 points of
false-alarm rate. The published figure states **+4** by comparing the control's epoch040 against
the augmented run's epoch015 — also at 68% recall, but not the same epoch. The augmented run
keeps more people at every epoch from 020 on; the epoch-matched +3 is the number to quote.

The composite stands the peer on a plane fitted from the corpus,
`contact_row = 0.472 x box_height + 0.585`, r2 = 0.702 over 1,256 boxes whose bottom edge is
inside the frame. That fit is re-derivable from `labels/peer_go2wheel_20260824.json` in this
repository, and the evidence directory carries the command.

### ⛔ What it does not change: nothing clears the gate

**No checkpoint of the 64 keeps all 22 people.** The best keeps 20, reached by ten rows across
four runs, and every one of those ten is *below* the shipped weights on peer recall. The set of
checkpoints that beat the shipped weights on both peer axes and keep at least 20 people is
**empty**. The best peer row, `k_full_pseudo03` ep022, loses five.

The warning further down this page — *Fine-tuning on a one-class corpus teaches the network that
people are background* — is now measured across 64 checkpoints instead of four, and it holds
across all of them. It is the constraint on this whole line of work, not a property of one run.

### ⚠️ The transferable lesson is about order of work, not about weights

The best detector in this project was on disk, unscored, while a fifth training wave was queued
on the same host. Measuring 640 existing checkpoints costs an evaluation pass; producing four
more runs costs a day of GB10 time and produced nothing that beat what was already there.

**Measure what you have before you make more of it.** That is the same shape as this page's
other two lessons — a test that shares its conditions with what it tests measures nothing, and
47 positives could not rank anything — and it is the cheaper of the three to act on.

## The cross-day number, on its own terms

Every row is the **held-out test split**: 47 frames containing a Go2 Wheel and 136 containing
none, all recorded 2026-08-20, none trained on, scored through the deployed `cv2.dnn` path at
the prototxt's own `confidence_threshold: 0.25`. `person lost` counts, out of the 15 test
frames where the shipped network still detects a person, those on which the candidate drops
it below **0.45** — the confidence the navigator actually runs at, from every Aug-20
telemetry header.

| model | recall | false positives | precision | `person` lost |
| --- | --- | --- | --- | --- |
| frozen head (`train_new_class.py`, on the record) | 74% (35/47) | **60%** (81/136) | 30% | 1 of 15 |
| **fully unfrozen + old-class labels** | **72%** (34/47) | **4%** (6/136) | **85%** | **2 of 15** |
| fully unfrozen, old-class labels at 0.5 not 0.3 | 70% (33/47) | 1% (1/136) | 97% | 7 of 15 |
| unfrozen from `conv6`, backbone LR x0.1 | 55% (26/47) | 1% (2/136) | 93% | 4 of 15 |
| backbone frozen through `conv13` | 47% (22/47) | 8% (11/136) | 67% | 1 of 15 |
| untrained 22-class seed (floor) | 0% | 0% | — | 0 of 15 |

The recommended checkpoint in full, across the reachable thresholds:

| threshold | recall | recall (IoU>=0.5) | false positives | precision |
| --- | --- | --- | --- | --- |
| 0.25 | 72% (34/47) | **100%** (4/4) | 4% (6/136) | 85% |
| 0.50 | 57% (27/47) | 75% (3/4) | 1% (2/136) | 93% |
| 0.70 | 6% (3/47) | 25% (1/4) | 0% (0/136) | 100% |
| 0.90 | 0% | 0% | 0% | — |

Nothing below 0.25 is reachable without editing the deployed prototxt: `DetectionOutput`
carries `confidence_threshold: 0.25` and has already discarded weaker boxes before
`forward()` returns.

**Does it beat the frozen head? Yes, and the axis is precision.** 30% -> 85% at the same
recall; 60% -> 4% false positives. It does not beat the stock model **on the Aug-24 capture** —
see above; on the held-out Aug-20 day, with the deployed shape rule applied to both, it does, and
so does every other checkpoint here. See *Correction, 2026-08-26*.
On the original fifteen-still protocol — its original
labels, its original JPEGs, so nothing is re-baselined — the fine-tune scores **5 of 15 at
0 of 159 false positives** against the published **8 of 15 at 60 of 159**. Fewer fires, and
not one of them on the 159 frames that hold nothing. Precision 12% -> 100%.

## What the published recall column actually meant

Nine of the fifteen "held-out peer frames" contain no peer — see `labels/CROSSDAY.md`. The
frozen head's 53% was 8 fires over a denominator of 15 of which **6** held a robot, against
its own 38% false-positive rate; firing at random would have scored about 6 of 15. Scored
against a box, on the six frames that carry one, the frozen head localised **1**. The
fine-tune localises **4 of 4** in the test split at 0.25.

That is why the table above is 47 positives, not 6: every frame of the two peer clips is now
labelled, from video this repository already carries.

## ⚠️ Fine-tuning on a one-class corpus teaches the network that people are background

The corpus is labelled for `go2wheel` and nothing else, and by the labeller's own account the
human operator stands in a large share of the frames. Every prior on that operator therefore
carries the label *background*. Hard negative mining then does exactly its job: it selects
the priors the model is most confident about, which are the operator's, and trains them down.

Measured on the first run that scored well on the new class:

```
person, on the 15 test frames the shipped model still detects one:
    0.819 -> 0.179 mean,  worst single drop 0.987,  below the deployed 0.45 on 13 of 15
```

The new class was the best it had ever been on that same run.

**Distillation alone does not fix it, and the reason is mechanical.** A distillation term pins
the old classes' *logits* to the starting network's; the cross-entropy is free to raise class
0 above them, and softmax does the rest. Two runs at `--distil 1.0` confirm it — and with
class 0 *in* the distillation target the term pins the background logit at the value a
network that had never seen this object assigned it, which is high on every prior. The new
class then has to beat a logit the loss is holding up, and it does not: those runs reached
15% frame recall and stayed there for 25 epochs.

**What fixes it is labelling the old objects.** `--pseudo-labels` runs the starting network
over every training frame and carries its own confident detections as ground truth. At 0.3,
312 boxes over 311 of the 2,048 frames, nearly all `person`. The operator is then labelled
`person` instead of mined as a hard negative.

| run | old-class labels | `person` mean | below 0.45 |
| --- | --- | --- | --- |
| shipped 22-class seed | — | 0.819 | 0 of 15 |
| frozen head | untouched by construction | 0.734 | 1 of 15 |
| trunk frozen, none | none | 0.179 | 13 of 15 |
| from `conv6`, none | none | 0.330 | 10 of 15 |
| fully unfrozen, 209 boxes at 0.5 | 209 | 0.459 | 7 of 15 |
| fully unfrozen, `--distil` 0.3 | 209 | 0.646 | 3 of 15 |
| fully unfrozen, backbone LR x0.2 | 209 | 0.672 | 3 of 15 |
| **fully unfrozen, 312 boxes at 0.3** | **312** | **0.724** | **2 of 15** |
| trunk frozen, 209 boxes at 0.5 | 209 | 0.720 | 1 of 15 |

Lowering the pseudo-label threshold from 0.5 to 0.3 bought back five `person` detections AND
two points of new-class recall, at the price of three points of false-positive rate. It is
the single most effective knob in this work, and it costs one number in a command line.

## The freeze ladder does not say what the brief expected

The warning was that unfreezing 5.7M parameters against 1,343 frames of one corridor would
overfit harder, not less. Measured on the **selection** split, so the test numbers above stay
clean:

| trainable | training conf loss | cross-day recall @0.25 |
| --- | --- | --- |
| trunk frozen (`conv14_1`..heads) | 2.52 | 46% |
| from `conv6`, LR x0.1 | 1.92 | 46% |
| everything, LR x0.5, **no old-class labels** | **0.87** | **8%** |
| everything, LR x0.5, old-class labels | 1.11 | 46% |

Row 3 is the textbook picture — the lowest training loss of any run and the worst
generalisation of any run that fired at all. Row 4 has nearly the same trainable set and
nearly the same training loss and does not collapse. **What separated them was not how much
of the backbone moved but whether the loss was telling the truth about the other twenty
classes.** On the test split the fully-unfrozen runs are the best by 17 to 25 recall points.

So: unfreezing is not the hazard the ladder was built to find, and it is what produced the
result. Labelling one class on a corpus full of unlabelled objects is the hazard, and it
punishes a bigger trainable set harder because there is more of the network available to
learn the wrong thing with.

⚠️ **The selection split holds 13 positives**, which is small enough that its ranking should
not be trusted and was not: it tied three runs at 46% while the test split separated them by
25 points. Its only job is choosing the epoch.

## What was borrowed rather than re-derived

**Priors** come out of `cv2`, from the deployed network's own `mbox_priorbox` blob: 1917
boxes, variances 0.1/0.1/0.2/0.2, unclipped (`clip: false` — row 0 really does start at
-0.0737). PriorBox is not re-implemented: a half-cell offset there is invisible in the loss
and fatal at inference.

**The head layout is asserted, not assumed.** `verify_head_assembly` checks that this
module's re-assembly of the six `mbox_loc` / `mbox_conf` convolutions reproduces `cv2`'s own
`mbox_loc` and `mbox_conf` blobs on real input, every time training starts. Measured:
**2.9e-05** on loc, **3.7e-04** on conf. Had the cell/slot ordering been wrong the loss would
still have gone down; it would just have trained the wrong prior for every box.

**Targets are encoded the way `DetectionOutput` decodes** — `CENTER_SIZE`, centre offsets
divided by 0.1 and log size ratios by 0.2.

## Export, which is the half that has to survive

Every checkpoint is written straight back to `.caffemodel` through the protobuf path
`add_class.py` proved, and `ssd_torch.verify_against_cv2` then runs **on the file that was
written**:

| model | layers | params | worst abs error vs `cv2` | bytes |
| --- | --- | --- | --- | --- |
| fully unfrozen + 312 labels, e015 | 47 | 5,798,042 | 1.98e-04 | 23,206,074 |
| fully unfrozen + `--distil` 0.3, e022 | 47 | 5,798,042 | 3.97e-04 | 23,206,074 |
| trunk frozen, e020 | 47 | 5,798,042 | 2.14e-04 | 23,206,074 |

Same prototxt, same 300x300 preprocessing, same file size as the model the Jetson loads
today. Nothing on the robot changes.

## How a checkpoint is chosen

**The earliest epoch maximising (frame recall − false-positive rate) at 0.50 on the SELECTION
split.** The split is by clip, so selection frames and reported frames share no moment of
video — 7 fps footage makes a random split meaningless. Early stopping on a cross-day set is
exactly the "tuned until it looks good" hazard; a clip-disjoint split is what stands against
it.

## Augmentation, and the one that matters

Photometric jitter, SSD's IoU-constrained sample crop, horizontal flip with the boxes, and
random expand. **Expand is doing the work.** The corpus has the peer at nine staged distances
and nothing beyond about 4 m; the deployment case is a peer arriving down a corridor. Expand
is the only operator that manufactures a smaller apparent robot. No random channel swap: the
object is achromatic and the corridor's colour is one of the few cues separating a grey robot
from a grey floor.

Stock SSD mines `3 * num_pos` negatives per image, which is **zero** for an image with no
object — and 705 of the 2,048 training frames are deliberately peer-free corridor, the most
valuable frames in the set. `--neg-floor 32` is what lets them contribute.

## Not done, and the honest caveats

* **No hardware run, and no robot has seen these weights.** The `.caffemodel` is verified
  against `cv2` and nothing more.
* **`person` is close to par but not at it.** 2 of 15 lost against the frozen head's 1. Two
  frames is two frames. This was written up as a shadow-run candidate; it is not one any
  more, because the stock model needs no shadow run and costs `person` nothing at all.
  ⚠️ *Two frames was the best case, not the case.* The four checkpoints scored cross-day on
  2026-08-26 lose 2, 4, 7 and 11 of the same 15. "Close to par" describes one checkpoint of
  four, and the sentence should never have been written about the approach.
* **A new false `person`, on the peer.** On `peer_cross5` frame 039 the shipped model calls
  the peer `chair 0.291` and this model calls it **`person 0.543`** — above the deployed
  threshold. It fails safe (the navigator holds) but routes a peer into the always-hold path
  rather than to the policy, which is the behaviour the peer work exists to avoid. `dog` also
  fires on the robot at 0.31-0.62, which is harmless because the deployed stack reads only
  `person`, and is what a `dog`-seeded class looks like.
* **The misses are the far frames.** The model does not fire on the peer at the far end of the
  corridor, which is where a navigator needs it earliest. A visual audit of nine test
  positives found eight detected with the box on the robot, none anywhere else, and the one
  miss was the most distant frame in the sample.
* **Four boxes.** `recall (IoU>=0.5)` in the test split is over the four frames that carry a
  box. The visual audit is the qualitative version of the same claim over more frames.
* **The held-out frames carry the navigator's burned-in overlay** and the training frames do
  not — the confound `fix/ceiling-overlay-caveat` measured at four points of false-positive
  rate. Masking the two fixed regions moved this evidence by the same small amount (frozen
  head 60% -> 52% at 0.25). It is real, it is minor, and it cannot explain a 56-point gap.
* **JPEG re-encoding is worth a few points.** The same model scored 56% and 60% false
  positives on the same frames extracted twice at different qualities. Do not read these
  numbers to the percent.
* **One corridor, one day of training data, one day of test data**, 47 held-out positives.
  Everything above is a claim about this building.

## ⛔ The recommendation as published, 2026-08-25 — half of it is now wrong

Kept verbatim, because it is what the repository acted on for a day and the correction is only
readable next to it. The peer half is reversed by *Correction, 2026-08-26* above; the
`person` half is not.

**Do not fine-tune this detector, and do not deploy any checkpoint on this page.** Read the
stock one class-agnostically instead: it wins on both axes over 40x the frames, it costs no
training, it cannot regress `person` because it *is* the model that defines `person`, and it
is already installed on every robot. `FROZEN-FEATURE-CEILING.md` carries the full argument
and `eval_class_agnostic.py` carries the check.

What the class-agnostic route still owes, and none of it is a training problem: a range prior
for a box with no meaningful label — `estimate_range` currently picks between a person's
height and shoulder width, and a `motorbike` box on a peer robot ranged against a standing
adult reads **far**, in the dangerous direction — and a decision about which of the 18% is
furniture the static map should hold rather than the tracker carry as a mover.

**Everything in `finetune_ssd.py` stays and stays runnable.** It is how the negative result
was produced; a conclusion nobody can re-derive is an opinion. It also remains the only route
if a class ever has to be added for a reason no stock VOC label can serve, and the
people-are-background finding above applies to any such run.

## The recommendation, corrected 2026-08-26

**Still do not deploy any checkpoint on this page — and the reason is `person`, not the peer.**
The three sentences that changed:

* **"It loses to the stock model."** It does not. On the held-out day, with the deployed shape
  rule applied to both, `l_full_bb02` reads 70% of peers at 19% false alarms against the shipped
  weights' 68% at 57%. Every checkpoint here beats them on the peer.
* **"18% false alarms."** That was staged empty corridor. The shipped weights fire on **57%** of
  furnished peer-free frames, which is the number the class-agnostic route has to answer for.
* **"It costs `person` two frames."** Across four checkpoints it costs between 2 and 11 of 15,
  and that is the blocker. The shipped weights keep 15 of 15 by construction.

So fine-tuning is an open line of work again, with one gate on it, and the gate is not a peer
number: **lose none of the people the shipped network sees.** Nothing that drops a person is a
candidate, however good its peer columns. ⚠️ *Restated 2026-08-26:* the gate is **22 people at
0.25**, not 15 at 0.45 — 0.25 is what `deploy/run-peer-supervised.sh` launches the peer runs
with, and the larger denominator makes the gate harder. Wave 5 and the 64-checkpoint sweep have
both now reported and **neither cleared it**; the best of 64 keeps 20 of 22 and is below the
shipped weights on peer recall. See *The sweep finished* above.

Neither route is deployable today. What the class-agnostic stock read still owes is unchanged
except in size: a range prior for a box with no meaningful label, and a decision about which of
the false alarms are furniture the static map should hold rather than the tracker carry as a
mover — 18% of them on staged empty corridor, **57%** in a furnished room.

## Reproducing

Twelve runs, 40 epochs each, about 35 s per epoch with three in parallel on one GB10. ⚠️ *Wave 5
added four more in the same shape, so sixteen runs and 640 checkpoints exist; the sweep above
measured 64 of them.*

```sh
add_class.py --in-proto MobileNetSSD_deploy.prototxt \
             --in-model MobileNetSSD_deploy.caffemodel \
             --out-proto mnssd22.prototxt --out-model mnssd22.caffemodel \
             --classes go2wheel

finetune_ssd.py --proto mnssd22.prototxt --model mnssd22.caffemodel \
                --labels labels/peer_go2wheel_20260824.json --images PEERCAP \
                --negatives-glob 'PEERCAP/neg_*.jpg' --caffe-pb2 PB \
                --freeze-through '' --backbone-lr-scale 0.5 \
                --pseudo-labels 0.3 --distil 0.1 --epochs 40 --out-dir runs/best

eval_detector.py --proto mnssd22.prototxt --models 'runs/best/epoch*.caffemodel' \
                 --manifest labels/peer_crossday_20260820.json --frames-dir XDAY \
                 --split select                      # pick the epoch
eval_detector.py --proto mnssd22.prototxt --model runs/best/epoch015.caffemodel \
                 --reference mnssd22.caffemodel \
                 --manifest labels/peer_crossday_20260820.json --frames-dir XDAY \
                 --split test                        # report it
```

And the comparison that ended it, which needs no training run and no `.caffemodel` that is
not already on the robot:

```sh
eval_class_agnostic.py --model-dir ROBOTMODELS \
                       --labels labels/peer_go2wheel_20260824.json --frames-dir PEERCAP
```

`XDAY` is filled by the extraction command in `labels/CROSSDAY.md`. `PEERCAP` is the 394 MB
of training JPEG, which is not in this repository — see
`evidence/2026-08-24-peer-capture-and-gait-sweeps`.
