<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-26 — the floor contact point is usable, and the objection to it was measured at the wrong range

`camera_model` has argued since it was written that ranging from the floor contact point
is unusable on this robot, under the heading *WHY ANGULAR SIZE AND NOT THE GROUND PLANE*:

> the Go2's camera sits 0.32 m off the ground, so a person at 3 m is only 6° below the
> horizon and the range goes as `h/tan(elevation)` — a 2° trunk-pitch wobble (a trotting
> Go2 does more) swings the estimate from 2.3 m to 4.4 m.

**The arithmetic is exactly right** — 2.25 m to 4.49 m, reproduced below — and the
conclusion is right at 3 m. It is wrong as a verdict on the method, for two reasons that
are both measurable off recordings already in this repository, and this directory is that
measurement.

Reproduce every number below with no robot:

```bash
python3 ground_vs_prior.py
```

## 1. It was compared against an alternative that does not exist for an unnamed object

`estimate_range` divides an apparent size by a known one. For a person, or for a colour
prop an operator staged and measured, that is the better estimator and nothing here
changes it. For a chair an attendee walks in and puts down, there is no known size, and
substituting one scales the answer by the ratio of the two.

That is not hypothetical. Both live runs of
[`../2026-08-25-peer-runs/`](../2026-08-25-peer-runs/) ranged **every** detection with the
peer robot's 0.514 m prior — including the goal chair:

| hero run, goal chair | reported |
| --- | --- |
| size prior, told 0.514 m | 0.69–0.85 m |
| floor contact, told nothing | 0.88–1.69 m |
| height the contact point implies | **0.85 m** — an office chair |

The prior was 1.7x short and so was every range it produced. That error ran in the
harmless direction. An object **shorter** than the assumed prior reads **far**, which is
the direction that walks into it.

The same check run over the parked peer, where the prior *was* right, is what says the
contact point is not simply drifting: over the five sightings a 1.6°/18% ceiling keeps, it
implies **0.494 m** (sd 0.060) against the 0.514 m the run was told — an object dimension
recovered from geometry that never saw a size.

## 2. The band where it fails is a band no detection arrives from

The error is `|Δd/d| = δ·(d/h + h/d)`: linear in range for a fixed wobble. At 2°:

| range | far bound | error |
| ---: | ---: | ---: |
| 0.72 m | 0.79 m | +10.2% |
| 1.00 m | 1.14 m | +13.5% |
| **1.33 m** | 1.57 m | **+18.0%** |
| 1.50 m | 1.81 m | +20.5% |
| 2.00 m | 2.57 m | +28.6% |
| 2.70 m | 3.84 m | +42.4% |
| 3.00 m | 4.48 m | +49.2% |

18% is `tracker.RANGE_SIGMA_FRACTION` — what the filter already budgets for the size-prior
source it trusts **most**. And the detector's own recall, measured on the 1,903-frame
labelled corpus, is **0 of 315 beyond 2.7 m**, 80% at 1.5–1.9 m, 91% inside 1.1 m. The
3 m case the objection is built on is a range this detector does not produce detections
at.

## 3. How far the two estimators actually disagree, on a walking robot

`sightings` (#37) carries the raw per-detection **box**, so both estimators can be run
over the same recorded detection. Within one track of one object the ratio's mean is the
size prior's error; its scatter is the combined frame-to-frame noise of both, and so an
**upper bound** on the contact point's own.

| run | track | n | ratio mean | log-sd | implied height |
| --- | --- | ---: | ---: | ---: | ---: |
| hero | parked peer | 6 | 0.980 | **12.1%** | 0.504 m *(told 0.514)* |
| hero | goal chair | 9 | 2.117 | 45.3% | 1.088 m *(one 7.5 m outlier)* |
| contrast | t=4.6–6.4 | 12 | 1.286 | **7.9%** | 0.661 m |
| contrast | t=10.0–14.9 | 22 | 1.188 | 17.5% | 0.611 m |
| contrast | t=13.5–14.1 | 4 | 0.843 | **4.9%** | 0.434 m |
| contrast | t=15.2–19.6 | 18 | 1.571 | 33.3% | 0.808 m |

Attributing **all** of that residual to the contact point's elevation error — which
over-attributes, since the size prior's box-height noise is in it too — gives an implied
per-frame angular error of **1.6° median, 4.5° p90** over 71 sightings.

**So the 2° premise survives its own test.** What it buys, at 2° and an 18% budget, is a
usable band of 0.72–1.33 m; at the 1.6° median, 0.72–1.70 m. Narrow, and real.

| stated pitch error | eps=0.18 | eps=0.25 | eps=0.30 | eps=0.40 | eps=0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.0° | 2.77 | 3.64 | 4.21 | 5.22 | 6.10 |
| 1.6° | 1.70 | 2.25 | 2.61 | 3.25 | 3.80 |
| 2.0° | 1.33 | 1.79 | 2.08 | 2.59 | 3.03 |
| 3.0° | 0.83 | 1.15 | 1.35 | 1.70 | 2.00 |
| 4.5° | none | none | 0.85 | 1.09 | 1.30 |
| 6.0° | none | none | none | 0.78 | 0.94 |

`none` means the ceiling is inside the near wall, so the band is empty; `GroundRanger`
refuses to construct rather than reporting nothing on every frame.

## What this does NOT establish

* **Absolute accuracy is unmeasured.** Every number above is a comparison between two
  estimators, not against ground truth. Bearing-only triangulation against the robot's own
  odometry was tried as a third opinion and is ill-conditioned over these baselines
  (0.3–0.6 m of travel against objects at 0.7–1.8 m, most of the bearing sweep coming from
  yaw): the two runs prefer odometry scale factors of **1.9 and 0.5**. That indicts the
  triangulation, not either estimator, and it is why no absolute error is quoted here.
  A tape measure and twenty staged frames would settle it in an afternoon.
* **Trunk pitch under gait has still never been recorded.** 1.6°/4.5° is an upper bound
  contaminated by the other estimator's noise. The IMU is on the robot; nobody has logged
  it against the perception clock. That single measurement is what turns `pitch_error_rad`
  from a knob the caller guesses into a number.
* **Nothing here is a false-alarm rate.** These are two peer runs, not the 705-frame
  empty-corridor and 216-frame peer-absent sets PR #102 measured its gate on.

## The near wall is unchanged, and is worse than it is usually quoted

Below `h / tan(half_vfov)` the object's floor contact has left the bottom of the frame and
there is nothing to intersect. **0.719 m is the best case**: it is the centre column, and
the bottom-corner ray is shallower, so an object at the frame edge loses its contact point
at **0.806 m**. Both estimators die there — this is the wall the first comment on #6
predicted would be inherited, and it is.

`GroundRanger` refuses inside it rather than reporting a constant. The two constants that
already exist for that band, `FILLS_FRAME_RANGE_M` and the width-prior fit cap, between
them deadlocked a live run for five seconds on 2026-08-19, because a planner cannot tell a
constant from a measurement.
