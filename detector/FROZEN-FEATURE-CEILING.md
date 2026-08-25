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
  the fine-tune with.

### The bar any fine-tune now has to clear

**64% recall at 18% false alarms, on these frames, from a model that costs nothing and is
already installed.** Nothing on this page was ever measured against it — the tables above are
the Aug-20 cross-day set, a different day and a different denominator, so they do not compare
to it directly.

Reproduce the bar with [`eval_class_agnostic.py`](eval_class_agnostic.py); it is the script
that produced the two rows above, and it needs no training run and no `.caffemodel` that is
not already on the robot.

## What survives

`add_class.py`, the `.caffemodel` write-back, the PyTorch mirror, the 1,903 labels and the
refusal gate are all still correct and all still work. What they are for has changed: they
are the apparatus that let a fine-tune be *falsified*, and they remain the only route if a
class ever has to be added for a reason a stock label cannot serve.

The labels in particular are the irreplaceable artefact. They are what made every number on
this page possible, including the one that ended it.
