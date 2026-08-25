<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Box expansion as a false-positive filter — what the corpus could and could not settle

Corpus: `go2-peer-dataset-20260824`, 2,800 frames from the Go2's own front camera,
2026-08-24, labels in `peer_go2wheel_20260824.json` (1,903 boxes). Everything below ran
the **robot's own weights** through the repo's own `person_detector.PersonDetector` and
`estimate_range`, class-agnostically: all 20 VOC classes count as an obstacle.

Two results are about the DATA and land before the filter is worth discussing. The rest
is what the filter is and what it is calibrated against.

---

## 1. ⛔ The 18% false-alarm rate does not exist. It is 192 mislabelled frames.

The scoring split the corpus by "is this filename in the label file", and treated
everything absent as peer-free. Four clips are absent. Two of them are negatives. The
other two are 192 frames **with a peer robot in them** that the labeller never covered.

| clip | frames | actually peer-free? |
|---|---|---|
| `neg_prone` | 434 | yes |
| `neg_standing` | 271 | yes |
| `p1b_close_broadside_STANDING` | 134 | **NO** — a Go2 Wheel filling most of the frame |
| `smoke` | 58 | **NO** — a peer at mid-corridor, plus a person |

Rescoring against the 705 genuinely peer-free frames:

| conf | recall on 1,903 labelled | "FA" on 897 as scored | FA on 705 truly empty | detection on the 192 |
|---|---|---|---|---|
| 0.25 | 1226 = 64.4% | 159 = 17.7% | **0 = 0.0%** | 159 = 82.8% |
| 0.40 (deployed) | 659 = 34.6% | 93 = 10.4% | **0 = 0.0%** | 93 = 48.4% |
| 0.50 | 405 = 21.3% | 44 = 4.9% | **0 = 0.0%** | 44 = 22.9% |

**Every single one of the 159 "false alarms" is a detection on a frame that contains a
peer.** Spot-checked by drawing them: `bicycle 0.45` on `p1b_..._0000` is the box
`[266, 29, 1737, 1077]`, which is the peer; `motorbike 0.32` on `smoke_0030` is
`[885, 411, 1336, 765]`, which is the peer; the `person 0.38` in the same frame is the
human standing in it.

So the class-agnostic operating point is **64% recall at 0% false alarm on this corridor**,
not 64% at 18%. The cost the filter was commissioned to reduce is not there to reduce.

### The threshold sweep below 0.25 was inert

`MobileNetSSD_deploy.prototxt` sets `confidence_threshold: 0.25` in its `detection_out`
layer, so the network never emits anything below it. A sweep "down to 0.10" — or the
`confidence=0.05` this work started with — measures nothing below 0.25.

Re-running with the prototxt threshold lowered to 0.01 gives the real floor:

| score floor | frames of 705 with ANY detection |
|---|---|
| 0.05 | 587 = 83.3% |
| 0.10 | 269 = 38.2% |
| **0.15** | **0 = 0.00%** |

The highest-scoring false positive on genuinely empty corridor scores **0.132** — a
`chair`, on the same fixed piece of corridor furniture in nearly every `neg_prone` frame.
The deployed threshold of 0.40 clears it by 3.0x.

⚠️ **Do not read that as "the detector is clean."** 705 frames from a PARKED robot in ONE
corridor at two camera heights is close to one independent sample, not 705. The same
over-read is already flagged in `person_detector.DEFAULT_CONFIDENCE`'s docstring for a
139-frame office set, and it is worth noting that the `train`-at-0.97 on a corridor
cabinet reported elsewhere in this repo does **not** appear anywhere in these 2,800
frames — 1 `train` detection in the whole corpus, at 0.016. Different viewpoint, different
answer. That is the point.

## 2. ⛔ The corpus cannot evaluate this filter at all, because the robot never moved

Phase-correlating consecutive frames, downscaled to 480 wide:

| clip | per-frame shift, median | net over the whole clip |
|---|---|---|
| `neg_prone` | 0.010 px | 13.9 px |
| `neg_standing` | 0.071 px | 0.1 px |
| `p1_close_broadside` | 0.015 px | 0.0 px |
| `p4_mid_sweep_stand` | 0.149 px | 14.7 px |
| `p1b_close_broadside_STANDING` | 0.158 px | 0.3 px |

14.7 px at 480 wide is 59 px at full resolution, 2.6° of yaw, over an entire clip. There
is no translation anywhere in this corpus — `p4_mid_sweep_stand` is the PEER being swept
between takes, not the robot. Ego-motion is the filter's entire input, so the corpus
measures the **noise** and can say nothing about the **signal**.

**What is needed: a walking sequence past a structure the detector fires on.** Nothing
short of that decides whether this filter catches anything.

---

## 3. What the corpus DOES settle: the noise floor, and the trap in it

Detections chained into tracks by IoU ≥ 0.30 within each clip, ranged by the repo's own
`estimate_range` with the deployed person prior. 94 tracks, 1,565 samples, median frame
interval 74 ms. The robot is parked, so every number here is noise.

| | median &#124;Δln R&#124; | p95 | p99 | sd |
|---|---|---|---|---|
| **ranging source HELD** (1,263 pairs) | 0.94% | 10.6% | 19.6% | 4.63% |
| **ranging source SWITCHED** (208 pairs) | **103.05%** | 158.9% | 160.5% | 107.5% |

The underlying box height moves 0.68% per frame with the source held, which independently
reproduces the 0.73% jitter figure this work was given.

**The dominant noise in the range series is not the detector. It is
`estimate_range` changing its mind**, between the height prior, the width prior and the
frame-fill constant — a factor of 2.8 between consecutive frames, on 14.1% of frame
pairs, on a stationary robot looking at a stationary peer. One clip,
`p6_2_trunc_right_stand`, alternates source almost every frame and has a *median* step of
102%. Any growth-rate fit spanning a switch measures the switch.

The filter therefore restarts its window on a source change, and abstains outright on the
two sources that report a constant.

### The window-slope null, which is what sets the threshold

Restricted to source-homogeneous runs (100 runs, 1,233 samples), per-sample
`sigma(ln R) = 3.07%`. Fitted slope over every window, with the true slope zero:

| n | span | windows | slope sd (1/s) | one-sided exceedance of +2σ / +3σ / +4σ / +5σ |
|---|---|---|---|---|
| 8 | 0.70 s | 595 | 0.0543 | 2.35% / 0.50% / **0.17%** / 0.00% |
| 12 | 1.11 s | 440 | 0.0325 | 2.95% / 0.45% / **0.00%** / 0.00% |
| 16 | 1.52 s | 349 | 0.0234 | 3.15% / 0.29% / **0.00%** / 0.00% |
| 20 | 1.95 s | 290 | 0.0166 | 1.38% / 0.00% / **0.00%** / 0.00% |
| 30 | 2.99 s | 208 | 0.0100 | 1.44% / 0.00% / **0.00%** / 0.00% |

Two things follow, and both went into the code:

* The measured slope sd sits within **0.81–1.15x** of what independent per-sample noise
  predicts over 0.7–3.0 s, so the noise is essentially white in `ln R` and the standard
  error of the fit is usable as written.
* The tail is **not** Gaussian. A normal distribution puts 0.13% beyond +3σ; the measured
  figure is 0.29–0.50%, three to four times that. `REJECT_SIGMAS` is 4.0 for that reason
  and not from a table. Note the "0.00%" cells are bounded by their sample count — at
  n=20 that is `<1/290`, i.e. under 0.35%, not zero.

---

## 4. The filter

`robot-stack/unitree/go2/visual_nav/expansion.py`. What it tests, and why it is worth
having even though §1 removed its stated motivation:

A class-agnostic detector gives up **what the object is, and therefore how big it is**.
`estimate_range` runs on a size prior, so the range is wrong by whatever factor the prior
is wrong by. Measured on this corpus: the `p1b` clip's peer, a ~0.40 m Go2 Wheel filling
the frame, is ranged at **2.05 m** by the 1.70 m person prior. `peer01` and `smoke` are
ranged at 6.05 m and 6.07 m. Those are size-prior errors of about 4x, and nothing in the
single-frame pipeline can see them.

Range from a size prior is exactly proportional to true range, so the *logarithmic* rate
is prior-free:

```
d(ln R_reported)/dt = d(ln R_true)/dt = -v_closing / R_true
```

The gate holds the odom point the track claims to occupy fixed, walks the robot's own
pose sequence past it, and compares the log-rate that predicts against the one observed.
It needs no camera model, no velocity input and no size prior — two range readings and
two robot poses.

A track is dropped only when its range falls **more slowly** than odometry demands, by
4σ, *and* the observed rate puts contact beyond 8 s. Growing FASTER — nearer than
reported, or walking at the robot — raises no verdict at all.

### Reach, given the measured sigma (30-sample window at ~7 Hz, noiseless)

Smallest `true / reported` range ratio that gets dropped:

| reported range | v = 0.35 | v = 0.50 | v = 0.80 |
|---|---|---|---|
| 0.8 m | 7.60x | never | never |
| 1.0 m | 4.90x | never | never |
| 1.5 m | 2.75x | 4.50x | never |
| 2.0 m | 2.05x | 3.00x | 6.35x |
| 3.0 m | 1.35x | 1.95x | 3.25x |
| 4.0 m | 1.20x | 1.45x | 2.35x |
| 6.0 m | 1.35x | 1.20x | 1.55x |

The same numbers as an absolute TRUE range are the more useful reading, because they are
nearly flat down the column: the gate starts dropping when the object is really **~4.1 m**
away at the 0.35 m/s gait floor, ~5.9 m at 0.5 and ~9.4 m at 0.8. It looks like a test of
the size prior and behaves like a test of true range — which is what makes the
one-sidedness safe, and is roughly `1.5 * v * TAU_HORIZON_S`.

⚠️ **Reach in REPORTED range is not monotone in speed**, and a test asserting it was is
what found this: walking faster sharpens the ego-motion signal and pushes the contact
horizon out at the same time, and past about 0.5 m/s the second wins. Below ~1.5 m
reported, at high speed, the robot walks through the claimed position before the window
fills and the gate abstains.

### Against the measured noise, in simulation

Injecting `sigma(ln R) = 3.07%` onto synthetic approaches. Reported range 2.0 m, gait
floor, 400 seeds per cell for the null, 200 for the rest.

| window | k=1.00 (false drops) | k=0.83 | k=0.67 | k=0.50 | k=0.33 | k=0.20 |
|---|---|---|---|---|---|---|
| 12 | 0/400 | 0.0% | 0.0% | 2.0% | 35.0% | 78.0% |
| 16 | 0/400 | 0.0% | 0.0% | 9.0% | 92.0% | 100% |
| 20 | 0/400 | 0.0% | 0.0% | 22.5% | 100% | 100% |
| 26 | 0/400 | 0.0% | 0.0% | 49.0% | 100% | 100% |
| 30 | 0/400 | 0.0% | 0.0% | 59.5% | 100% | 100% |

Monotone in both directions, and zero false drops at every range from 1.0 to 3.0 m.

**This is simulation, and it is the weakest evidence in this document.** It uses the
measured noise amplitude and the measured whiteness, but the approach itself is synthetic
and the world it walks through is a straight line. It says the ARITHMETIC works at the
measured noise. It does not say the filter catches a real false positive, because
nothing in this corpus can.

---

## 5. Three bugs the tests found, all of them the same bug

Recorded because each one presented as a *finding* until it was chased.

1. **A ghost rejected after 1.4 m of travel came back CONSISTENT after 3.3 m.** The
   prediction integrates `ln|anchor - robot|`, which falls to the closest approach and
   then RISES. A window spanning the turn fits a straight line to a V and the prediction
   flattens.
2. **The guard for that could never fire.** It refused a predicted range of zero or less,
   and `hypot` is non-negative — a robot 1.3 m short of the anchor and one 1.3 m past it
   are the same number. Replacing it with "the newest sample is still the closest" plus a
   floor took a 124-case reversal sweep to 86, not to 0.
3. **The residual reversals were the noise term, not the signal.** Both halves of the
   drop condition carry `REJECT_SIGMAS * sigma`, and sigma depends on how many samples
   the window currently holds. Traced exactly: a ghost at 10x its reported range held
   INCONSISTENT for 60 consecutive frames; the window went 12 samples to 11, 4σ went
   0.0725 to 0.0827, the contact-horizon comparison crossed, and the verdict reversed.
   Nothing about the world had changed.

Fixed by measuring the prediction's departure from its own straight line directly, and by
latching a rejection so it can only be contradicted — by the range starting to close as
odometry demands, by the window being discarded, or by the tracker deleting the track —
never merely forgotten. The reversal sweep is now 0 of 750.

The invariant took two attempts to state, too. "Never flips back" is too strong: a ghost
10 m off that the robot keeps walking toward genuinely comes inside the 8 s horizon, and
re-deciding is then correct. The invariant is restricted to the stretch where the object
is still comfortably beyond it.

## 6. What is still not established

* **No moving-robot data.** Everything in §4 is simulation over a measured noise
  amplitude. `--expansion-filter` is therefore **off by default**.
* **A mis-scaled ghost and a retreating real obstacle are the same measurement.** One
  monocular range series cannot separate them. What would is bearing parallax, which is
  immune to radial target motion — and degenerate straight ahead, which is where
  obstacles matter.
* **The `k > 1` half is untouched.** A person prior on a 0.40 m peer robot reports it
  ~4x too FAR, which is the dangerous direction, and this filter is deliberately silent
  on it. The gate already measures `R_reported/R_true` as `Verdict.scale`; using it to
  CORRECT the range rather than only to threshold it is the obvious next thing and is not
  done here.
* **The negative set is one corridor.** §1's zero is a real zero on these frames and says
  little about a different room.

## Reproducing

Data and weights are on the DGX Spark at `~/go2-peer-dataset-20260824/`; the scripts that
produced §1–§3 are in this directory and expect `~/cvenv/bin/python` (OpenCV 4.14).

```bash
python3 sweep_detections.py          # class-agnostic dump over all 2,800 frames
python3 rescore.py                   # section 1
python3 egomotion.py                 # section 2
python3 range_noise.py               # section 3
```

Sections 4 and 5 need no data:

```bash
cd robot-stack/unitree/go2/visual_nav && python3 test_expansion.py    # 38/38
```
