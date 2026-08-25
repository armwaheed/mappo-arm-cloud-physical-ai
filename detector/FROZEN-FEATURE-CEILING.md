<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fitting a linear head on frozen MobileNet-SSD features does not work for this class

A negative result, recorded because it cost a day and the next person will otherwise
spend the same day. **The approach in `train_new_class.py` has a ceiling, and 1,343 real
in-domain training frames were enough to establish that we are at it.**

The tooling is not wasted — `add_class.py`, the caffemodel write-back, the labelled
dataset and the refusal gate all carry forward to a real fine-tune. What does not carry
forward is the assumption that the six `*_mbox_conf` heads can be re-fitted on their own.

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

## A confound in this evidence, found and tested

The held-out Aug-20 frames were recorded with `--record`, which **burns the navigator's
overlay into the video** — a plan-view radar inset in the top-right, a status plate bottom
left, and detection rectangles. The 2026-08-24 training frames were captured straight from
the camera and carry none of it. Measured: the radar corner is 0.000 near-black in every
training frame and 0.900 in every held-out one.

So the numbers above compare a model trained on clean frames against frames carrying a
large synthetic artefact it had never seen, and "different day" was confounded with "has an
overlay".

**Tested by masking both fixed overlay regions with the frame's own median colour:**

| threshold | FP with overlay | FP masked | recall with | recall masked |
| --- | --- | --- | --- | --- |
| 0.25 | 38% | 34% | 53% | 40% |
| 0.50 | 24% | 23% | 47% | 33% |
| 0.90 | 7% | 4% | 20% | 20% |

**The confound is real and immaterial.** False positives move four points, recall drops,
and the best precision anywhere goes from 21% to 30%. There is still no usable operating
point. The conclusion below is unaffected — but the confound is recorded because it was a
genuine flaw in the evidence, and because the next person measuring against this corpus
needs to know the overlay is there.

⚠️ The same fact is a hazard for TRAINING, not just evaluation: compositing a synthetic
robot onto these frames would teach a detector that a peer arrives with an orange rectangle
attached.

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

## What would work, in order

1. **Unfreeze the backbone.** Real SSD training — multibox loss, prior matching, hard
   negative mining, augmentation — then export back through `add_class.py` into the
   `.caffemodel` the Jetson loads. The deployment path is already proven and unchanged;
   only the training half is missing.
2. **Then** more data, and background variety, both of which start mattering once the
   features can move. Synthetic background substitution belongs here and not before —
   before this, it adds examples a linear probe still cannot separate.
3. Real footage from other rooms beats generated backgrounds for the Go2 Wheel, since the
   robot and the building both exist. For the **Lite3** the reverse holds: nobody here can
   record in the deployment building, so synthesis competes against nothing.

## The fallback this makes more attractive

An ArUco marker on the peer needs **no training at all**, was measured at **6.43 m** on the
robot's own camera today, and reuses machinery that already ships in `goal.py`. It needs an
ArUco *obstacle* source — `ArucoGoal` only latches goals — which is about one file.

That was the recommendation before any of this was tried. It is a stronger recommendation
now, because the alternative has a measured ceiling rather than a suspected one.
