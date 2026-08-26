<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fitting a linear head on frozen MobileNet-SSD features does not work for this class

A negative result, recorded because it cost a day and the next person will otherwise
spend the same day. **The approach in `train_new_class.py` has a ceiling, and 1,343 real
in-domain training frames were enough to establish that we are at it.**

The tooling is not wasted — `add_class.py`, the caffemodel write-back, the labelled
dataset and the refusal gate all carry forward. What does not carry forward is the
assumption that the six `*_mbox_conf` heads can be re-fitted on their own.

> ⛔ **This page's recommendation has been reversed. Do not fine-tune this detector.**
> Measured 2026-08-25 on this page's own corpus: the robot's **stock, unmodified** 21-class
> weights put a box on the peer in **64%** of 1,903 peer frames, and fire on **18%** of 897
> peer-free frames — if you stop asking the model for a *name* and take any VOC label as an
> obstacle. The section this page used to end with, *"What would work: unfreeze the
> backbone"*, is struck through below and replaced by that measurement. The diagnosis on
> this page is still correct; the prescription was not.
>
> ⚠️ **That reversal is itself half-reversed, 2026-08-26.** Both numbers in it are the Aug-24
> capture — the day the fine-tunes were trained on — and the stock model had never been scored on
> the held-out Aug-20 day at all. On Aug-20, with one rule applied to every model, the stock
> weights read **68% recall at 57% false alarms** and every fine-tune beats them on the peer. The
> 18% was measured against staged empty corridor. What survives is `person`: the stock weights
> keep all 15 people the navigator sees, and every fine-tune loses between 2 and 11 of them, so
> nothing is deployable either way. See *The bar was measured on the wrong day*, below.
>
> ⚠️ **And the four fine-tunes in that correction were not the best four, 2026-08-26.** They were
> four of the thirteen model/epoch rows anybody had scored, out of **640 checkpoints that
> existed**. 64 have now been scored, and the best run's finer pass reads **89% peer recall at 12%
> false alarms** — well past the 72% this page credits unfreezing with. It keeps 17 of 22 people,
> so it does not clear the gate either. See *The ceiling was cleared by more than this page
> records*, below.

## ⚠️ Two corrections, both found later, neither of which changes the conclusion

**The recall column below is measured against a denominator that is wrong.** Nine of the
fifteen "held-out peer frames" contain no peer — see [`labels/CROSSDAY.md`](labels/CROSSDAY.md).
Six do. So "8 of 15" is 8 fires over a set of which 6 held a robot, against this model's own
38% false-positive rate; firing at random would have scored about 6 of 15. Scored against a
box on the six frames that carry one, this head localised **1**. The ceiling is deeper than
this page claimed, not shallower.

**The ceiling was cleared, and it did not matter.** Unfreezing the backbone — item 1 of
"What would work", below — does reach **72% recall at 4% false positives** on the corrected
cross-day set, so the linear-probe limit this page diagnoses is real and unfreezing is what
lifts it. Scored against the stock model on the same 2,800 frames, that same checkpoint gets
**53% at 38%** and loses on both axes. See [`UNFROZEN-FINE-TUNE.md`](UNFROZEN-FINE-TUNE.md),
including the part where it costs `person` and the part where 47 held-out positives could not
rank anything.

⚠️ **And "72%" is not the ceiling either.** It is the best of thirteen checkpoints anybody had
scored. Sixteen runs at 40 epochs left **640**, and a sweep of 64 of them on 2026-08-26 found
**89% recall at 12% false alarms** in a run this page does not name. The linear-probe limit this
page diagnoses is cleared by a wider margin than this page ever recorded — and still by nothing
deployable. See below, and
[`evidence/2026-08-26-checkpoint-sweep/`](../evidence/2026-08-26-checkpoint-sweep/).

⚠️ **"It did not matter" was the wrong conclusion, and the last clause of that paragraph is why.**
The 53%-at-38% row is the fine-tune's own training day; the 72%-at-4% row is the held-out day; the
page below knows the two cannot be ranked against each other and rules on the training-day row
anyway. Scored on the held-out day against a stock model finally measured there too, the ceiling
being cleared *does* matter — on the peer. It still does not produce anything deployable, because
of `person`. *The bar was measured on the wrong day*, below.

Everything else on this page stands. The three falsified hypotheses are still falsified, and
the reason they were falsified is still the right one: a linear probe on frozen VOC features
cannot separate this class.

## The result

Trained on **1,343 labelled frames** of a Go2 Wheel from nine staged positions, against
the **robot's own weights**, with 705 peer-free frames from the same session as training
negatives. Evaluated on **Aug-20 footage of the same corridor on a different day**, which
was never trained on.

| threshold | recall | false-positive rate | precision |
| --- | --- | --- | --- |
| 0.25 | 53% | **38%** | 12% |
| 0.50 | 47% | 24% | 16% |
| 0.70 | 40% | 18% | 17% |
| 0.90 | 20% | 7% | 21% |
| 0.99 | 13% | 0% | 100% |

There is no usable operating point. Precision never exceeds 21% at any recall above 20%.

**The score distributions are the diagnosis.** Highest `go2wheel` scores on frames
containing a peer: `0.995, 0.992, 0.987`. Highest on frames containing **no peer at all**:
`0.987, 0.975, 0.969`. The model is as confident on empty corridor as on the robot.

## ⚠️ The gate was measuring the wrong thing, and this is the transferable lesson

The same model scored **0 of 705** on its gate and **60 of 159** on another day. Both
numbers are real; they measure different things.

The gate's negatives came from the same session as the training frames — same corridor,
same hour, same light, same camera pose. A gate whose negatives share those with the
training set reports **memorisation**. `--gate-negatives` now exists so held-out negatives
can be given separately, and the trainer prints a loud warning when they are not.

This is the same error, one level down, that the 20% figure exposed: *a test that shares
its conditions with the thing it is testing measures nothing.*

## Three hypotheses, all tested, all falsified

**"It needs more regularisation."** `--l2` 0.05 → 0.15 made it **worse**: 37.7% → 48.4%.
L2 shrinks the new class's weights toward zero, so its logit collapses toward its **bias** —
a near-constant, which becomes the arg-max on any prior whose trained logits fall below it.
That is the same flat-floor failure the seeded-class init was introduced to avoid, arriving
by a different road.

**"The negatives are not diverse enough."** Adding another day's negatives to training
(705 + 101, split by source clip so near-duplicate frames could not straddle the split)
made it **worse**: 44.8% and 51.7% against 37.7%. Negative diversity is not the constraint.

**"One recurring object is causing it."** The labelling pass flagged a plausible culprit —
an office chair carrying an ArUco marker, at almost exactly the far peer's apparent size,
present in every corridor frame. Measured: the false detections **do not cluster**. Centre-x
standard deviation 511 px across a 1920 px frame, no cell of a 6x4 grid holding more than
20%. It is not a confusable object; the head fires everywhere.

## Why it cannot work, stated plainly

The backbone is frozen, so this fits a **linear probe** on features MobileNet-SSD learned
for twenty PASCAL VOC classes. Those features have no representation of a wheeled
quadruped. Adding examples a linear probe cannot separate does not make it separable, and
the overlapping score distributions above are what that limit looks like from outside.

The tell that rules out the obvious alternative explanation: **recall is 53% on the same
corridor**, one day apart. If this were background overfitting, recall on familiar
backgrounds would be high and false positives would concentrate on novel ones. Neither
holds.


## ⚠️ Nothing on this page was measured below 0.25, including the rows that say it was

The deployed `MobileNetSSD_deploy.prototxt` carries `confidence_threshold: 0.25` inside its
`detection_output_param`. `DetectionOutput` applies it **before** `forward()` returns, so no
box weaker than that has ever left this network — here, on the robot, or in any table in
this repository. Every "at any confidence down to 0.02" claim is really the 0.25 row under
another name, and a sweep that reports 0.02, 0.05 and 0.10 is reporting 0.25 three times.

It does not change the conclusion below — 0.25 is the operating point that exists — but it
does mean the shipped model was never given the chance those sentences claimed for it.

## ⛔ "What would work: unfreeze the backbone" — superseded, 2026-08-25

This section used to say that unfreezing the backbone was the route, and it named it first
of three. It was the wrong question. Nobody had asked whether a new class was needed at all.

**The planner does not need a name. It needs a box.** A detection reaches
`mappo_bridge.stationary_objects` as geometry — centre, span, the range derived from them —
and the label's only job upstream of that is deciding whether the detection is forwarded.
`person_detector.PersonDetector` is constructed with `classes=DYNAMIC_CLASSES`, which is
`("person",)`, and drops every other label on the floor before anything downstream sees it.

So the parsimony test is one run of the robot's own unmodified weights with that filter
removed: accept **every** VOC label, and ask whether a box lands on the peer.

### The stock 21-class model already finds the peer, on two-thirds of the frames

Measured on the same Aug-24 capture this page trained on — 2,800 frames, **1,903 with a
hand-labelled peer and 897 peer-free** — with `MobileNetSSD_deploy.caffemodel` exactly as it
sits on the robot, at 0.25, counting a hit as any detection overlapping the labelled box at
IoU >= 0.30:

| | stock 21-class, unmodified, any label |
| --- | --- |
| a box lands on the peer | **64%** of 1,903 peer frames |
| anything at all fires | **18%** of 897 peer-free frames |

The labels are nonsense and the boxes are not. Every detection that landed on the peer,
by the name it was given:

```
motorbike 613    chair 372    aeroplane 200    person 109
```

`motorbike` is the wheels read side-on — already noted on the record as firing 2/2 on live
broadside frames. `chair` and `aeroplane` are a low horizontal body on legs. Not one of them
is a correct name for a wheeled quadruped, and not one of them needs to be.

**This is a clean out-of-sample number in a way none of the fine-tune's are.** The stock
weights have never seen a frame of this corpus, so the memorisation trap that invalidated the
gate above cannot apply here: same-session negatives are legitimate against a model that was
not trained on the session.

### What this costs, stated rather than skipped

Widening the class filter is not free, and the comment in `person_detector.py` that narrowed
it to `person` gives the reason: *"a chair detected as an obstacle-with-velocity would be
noise."* That is exactly what the 18% is. Two things have to be decided before any of this
reaches the policy, and neither is a training problem:

* **Ranging has no prior for an unnamed box.** `estimate_range` picks between a person's
  height and shoulder width. A `motorbike` box on a peer robot ranged against a standing
  adult reads far, in the dangerous direction. A class-agnostic obstacle needs either its own
  prior or a fixed conservative range.
* **18% of peer-free frames fire on something.** Some of that is real furniture, which is a
  static obstacle the map should hold rather than a track. `kind` and the tracker decide
  whether that is a false alarm or a correct one, and that is the piece of work this replaces
  the fine-tune with. ⚠️ *18% is the staged-negative number.* Cross-day, on furnished corridor,
  it is **57%** — the work is the same work, three times the size.

### ⛔ The bar every fine-tune failed to clear — superseded 2026-08-26, wrong day

**64% recall at 18% false alarms, on these frames, from a model that costs nothing and is
already installed.** Nothing on this page was ever measured against it — the tables above are
the Aug-20 cross-day set, a different day and a different denominator, so they do not compare
to it directly.

The one model that *was* measured against it is the best checkpoint of the unfrozen
fine-tune, `runs/best/epoch015`, scored the same way over the same 2,800 frames:

| | reads | box on the peer | fires on a peer-free frame |
| --- | --- | --- | --- |
| **stock 21-class, unmodified** | any VOC label | **64%** of 1,903 | **18%** of 897 |
| unfrozen fine-tune, `epoch015` | its own `go2wheel` | 53% | 38% |

**It loses on both axes, and its row is the generous one.** 1,343 of those 1,903 peer frames
and 705 of those 897 peer-free frames were in its training set; the stock model has never
seen any of them. See [`UNFROZEN-FINE-TUNE.md`](UNFROZEN-FINE-TUNE.md).

Reproduce the bar with [`eval_class_agnostic.py`](eval_class_agnostic.py); it is the script
that produced the two rows above, and it needs no training run and no `.caffemodel` that is
not already on the robot.

## ⛔ The bar was measured on the wrong day — corrected 2026-08-26

**"64% recall at 18% false alarms" is a real number and it is not a bar.** Both rows of the table
above are the Aug-24 capture. That is the day the fine-tunes were trained on — 1,343 of the 1,903
positives and 705 of the 897 negatives are in the fine-tune's training set, which the section above
states plainly and then rules on anyway. The fine-tune's honest out-of-sample number is the Aug-20
split, and **the stock model had never been scored on Aug-20.** The comparison that closed this
work had a training-day row for both models, a cross-day row for one, and no cross-day row for the
model it recommended.

Scored now, on the **Aug-20 held-out split — 47 peer-present frames and 134 peer-free** — with one
rule applied identically to every model: a frame counts as a fire if any detection at >= 0.25 has
box aspect **h/w < 2.0**, i.e. it is not person-shaped, so `mappo_bridge` hands it to the policy as
an obstacle rather than holding (`person_detector.PERSON_ASPECT_MIN`). `people kept` counts, of the
15 frames where the **shipped** network sees a person at >= 0.45 — the confidence the navigator
runs at — how many the candidate still sees.

| model | peer recall | false alarms | precision | people kept |
| --- | ---: | ---: | ---: | ---: |
| **stock 21-class, as shipped** | **68%** (32/47) | **57%** (76/134) | 30% | **15/15** |
| `l_full_bb02` ep015 | 70% | 19% | 56% | 13/15 |
| `j_full_distil03` ep015 | 66% | 6% | 79% | 11/15 |
| `i_full_pseudo` ep020 | 74% | 8% | 76% | 8/15 |
| `f_full_distil01` ep020 | 74% | 1% | 95% | 4/15 |

**On the peer, every fine-tune beats the shipped weights.** Recall 70, 66, 74 and 74 against 68 —
three of the four above it, the fourth two points under. False alarms 19, 6, 8 and 1 against
**57**. So "do not fine-tune this detector" is wrong on the axis it was argued, and the section
above is superseded on that axis.

### ⚠️ The 18% was staged empty corridor

The stock model did not get worse. **705 of the 897 Aug-24 peer-free frames are `neg_prone` and
`neg_standing`** — corridor deliberately cleared and shot as training negatives, which is exactly
what they were made for. The Aug-20 negatives are the same corridor furnished and in use, ArUco
office chair and all. Same weights, same 0.25, same `cv2.dnn` path: **18% on the staged set, 57% on
the furnished one.**

This page already carries that lesson, one level up, about the refusal gate that scored 0 of 705:
*a test that shares its conditions with the thing it is testing measures nothing.* The 18% is the
same error pointed the other way — an empty-room false-alarm rate quoted as if it described a room.

### What did not change, and it is the part that blocks deployment

`⚠️ Fine-tuning on a one-class corpus teaches the network that people are background` — the warning
in [`UNFROZEN-FINE-TUNE.md`](UNFROZEN-FINE-TUNE.md) — was right, and it was understated. It was
written from one checkpoint losing 2 of 15 people. Across the four checkpoints above the loss runs
**2, 4, 7 and 11 of 15**, and the checkpoint with the best peer precision on the table is the one
that loses eleven. The shipped weights lose none, by construction: they are the model that defines
`person`.

**So nothing here is deployable, on either route.** The open objective is the combination no run has
produced: hold every person the shipped network sees while keeping the peer gains.

### ⚠️ 47 and 134 are small, and this repository has been burned three times

The refusal gate (0 of 705 same-session negatives), the fifteen stills (nine of which held no
peer), and the 47-positive ranking in `UNFROZEN-FINE-TUNE.md` were each an evidence set too small
and too like itself, and each flattered the thing being tested. The table above is one corridor,
one held-out day, 47 positives and 134 negatives. It is good for the comparison it makes — every
row scored the same way on the same frames — and it is not a claim about buildings in general.
~~Note also that the manifest's test split holds **136** peer-free frames and this sweep scored
**134**; two frames are unaccounted for.~~ **Answered by
[#91](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/pull/91):** the manifest in this
repository was right and the *derived* split on the training host was shifted by one clip index,
naming two frames that do not exist and omitting two that do. Repaired and all thirteen rows
rescored, every numerator is identical and only the denominator moved — the shipped weights go
from 76/134 = 57% to **76/136 = 56%**. The table above is left at 134 and 57% because that is
what it was measured on; the 2026-08-26 sweep is scored on the repaired 136.

### The ceiling was cleared by more than this page records — swept 2026-08-26

That sweep has reported. It scored 64 of the **640** checkpoints sixteen runs at 40 epochs had
left on disk — thirteen model/epoch rows had ever been evaluated — and the best is
`k_full_pseudo03`, a run named in no document in this repository before 2026-08-26. The two odd
epochs below are from a finer pass over that run and are not among the 64:

| model | peer recall | false alarms | people at 0.25 |
| --- | ---: | ---: | ---: |
| **stock 21-class, as shipped** | 68% (32/47) | **56%** (76/136) | **22/22** |
| `k_full_pseudo03` ep022 | **89%** | 12% | 17/22 |
| `k_full_pseudo03` ep017 | 83% | 12% | **19/22** |
| `f_full_distil01` ep020 | 74% (35/47) | **1%** (2/136) | 5/22 |

**13 of the 64 beat the shipped weights on both peer axes; all 64 beat them on false alarms.**
So the answer to this page's central question — *can a fine-tune clear the frozen-feature
ceiling and beat the shipped model on the peer* — is yes, by 21 points of recall, and it was
already answered by a file sitting on the training host while a fifth wave was being queued
against it.

⛔ **It changes nothing about deployment.** No checkpoint of the 64 keeps all 22 people; the best
keeps 20 and is *below* the shipped weights on peer recall. The set that beats the shipped
weights on both peer axes and keeps at least 20 people is empty. This page's own warning —
one-class fine-tuning teaches the network that people are background — is now measured over 64
checkpoints instead of four and holds across all of them.

⚠️ The gate is restated with it: **lose none of the 22 people the shipped network sees at 0.25**,
not 15 at 0.45. 0.25 is what `deploy/run-peer-supervised.sh` launches the peer runs with. The
larger denominator makes the gate harder, and the 22 was computed on the training host and
cannot be re-derived from a clone.

**The transferable lesson is order of work, not weights: measure what you have before you make
more of it.** The full record, the 64-row grid and the raw data are in
[`evidence/2026-08-26-checkpoint-sweep/`](../evidence/2026-08-26-checkpoint-sweep/). A further
wave sweeping the lever that produced the winner — `--pseudo-labels` below 0.3 — is running as
this is written, and no conclusion about it belongs on this page until it has numbers.

## What survives

`add_class.py`, the `.caffemodel` write-back, the PyTorch mirror, the 1,903 labels and the
refusal gate are all still correct and all still work. What they are for has changed: they
are the apparatus that let a fine-tune be *falsified*, and they remain the only route if a
class ever has to be added for a reason a stock label cannot serve.

The labels in particular are the irreplaceable artefact. They are what made every number on
this page possible, including the one that ended it.
