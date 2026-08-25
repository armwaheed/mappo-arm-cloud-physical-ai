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
recall; 60% -> 4% false positives. It does not beat the stock model — see above. On the original fifteen-still protocol — its original
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

## The recommendation

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

## Reproducing

Twelve runs, 40 epochs each, about 35 s per epoch with three in parallel on one GB10.

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
