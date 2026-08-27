<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-26 — the range scale is right, and the tape was never the only way to find out

Reproduce every number below with no robot:

```bash
python3 scale_audit.py
```

## The answer

**The camera's range scale is correct to about 10% against this robot's own odometry.**
Two of issue [#35](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/35)'s
four estimates are wrong, and so is the correction that was posted to settle it.

| # | estimate | verdict |
| --- | --- | --- |
| 1 | size prior — the bin at 2.34 m | **the chain is right.** Its *prior* was the unaudited part, and section 4 puts that within 6% too |
| 2 | floor contact — 2.12 m | **not independent**, and the reason given for dismissing it has the pitch sign backwards (section 3) |
| 3 | landmark drift — "the map reads 25% **long**" | **falsified.** That predicts k ≈ 1.25; measured k = 1.02 and 1.09 |
| 4 | the operator — "about 2.9 m", i.e. the map reads 24% **short** | **falsified for this chain.** That predicts k ≈ 0.81 |

The 2026-08-20 correction on #35 concluded the opposite of #3 — *"the map reads SHORT, by
roughly 20-30%"*, k ≈ 0.72–0.82, methods 2 and 3 wrong and the operator right. **Its
verdict on 2 and 3 stands; its own conclusion does not.** The measurement below is 1.5
standard-error-widths from 1.0 and nowhere near 0.8.

## 1. The measurement, and why it needed no tape

`../2026-08-25-peer-runs/` recorded two runs driving at a printed ArUco marker. **A
printed marker's size is known by manufacture** — 0.10 m of black square, stated in the
run header — so the range the stack computes for it has no unaudited prior in it at all.
Its only inputs are the focal length, the projection model and the arithmetic.

`ArucoGoal` latches `pose + range·(cos, sin)(yaw + bearing)` on each sighting, and the
telemetry writes the latched pair, so **the raw per-fix range and bearing are recoverable
exactly** from files already committed. Fit them against odometry:

> for a static object at unknown `P`, reported range `R_i` is `k` times the true one
> exactly when `p_i + (R_i / k)·u_i = P` for every `i`
>
> — a three-parameter **linear** least squares in `(P_x, P_y, 1/k)`. No initial guess, no
> iteration, no ground truth.

| run | window | n | **k** | residual rms |
| --- | --- | ---: | ---: | ---: |
| hero | all fixes | 43 | **1.092** | 4.5 cm |
| hero | first half | 21 | 1.150 | 5.2 cm |
| hero | second half | 22 | 1.047 | 2.3 cm |
| hero | inside 2.0 m | 17 | **1.015** | 2.3 cm |
| contrast | all fixes | 52 | **1.020** | 4.8 cm |
| contrast | first half | 26 | 1.050 | 4.7 cm |
| contrast | second half | 26 | 0.899 | 4.0 cm |
| contrast | inside 2.0 m | 31 | **0.944** | 4.0 cm |

The marker closed 3.81 → 0.91 m over 2.63 m of odometry in one run and 3.37 → 1.08 m over
2.22 m in the other. **A 4–5 cm residual on a 2.5 m baseline is not a scale error of 25%.**

⚠️ **k is a RATIO of two scales.** k = 1.0 says the camera and the leg odometry agree, not
that either is separately perfect. A conspiracy in which both are wrong by the same factor
is not excluded by this and needs a tape. What *is* excluded is the case #35 was raised
about: a camera scale wrong on its own while the odometry is fine.

## 2. ⛔ #35's own proposed method does not work, and the missing field was never why

*"How to settle it"* item 2 proposed this same fit over the **raw per-sighting ranges**,
and named the missing `sightings` field as the blocker. The field landed in #37. **This is
the first tool in the tree that reads `sightings[].range_m`** — and the fit does not work
there:

| run | track | n | travel | **k** |
| --- | --- | ---: | ---: | ---: |
| hero | t=6.0–9.7 | 10 | 0.93 m | 1.386 |
| hero | t=12.7–18.4 | 21 | 1.30 m | 1.363 |
| contrast | t=4.6–8.5 | 24 | 0.92 m | 2.271 |
| contrast | t=10.0–14.9 | 26 | 0.66 m | 1.811 |
| contrast | t=15.1–17.9 | 12 | 0.24 m | **102.053** |

Those are not measurements of anything. The blocker was never the field: it is that
**0.2–1.2 m of baseline against objects at 0.8–1.7 m cannot outrun the detector's own
range noise** — median `|Δln R|` of 103% when `estimate_range` switches source, on 14.1%
of consecutive pairs, measured in `../2026-08-25-expansion-as-a-false-positive-filter/`.
The marker works precisely because it is one unambiguous object, cornered to sub-pixel,
over forty-odd fixes and a long baseline.

**That item should be closed as tried and retired**, not left open as unblocked.

## 3. The pitch confound runs the other way

`go2_front_camera.json` records `pitch_rad: 0.0`, and #35 correctly notes that has never
been independently verified. The 2026-08-20 correction then dismissed the floor-contact
estimate on this reasoning:

> *The Go2's front camera is on the chin and points **down**. A downward pitch puts the
> floor contact lower in the frame, which makes `0.32 × f / (y_bottom − c_y)` read
> **short**.*

**Tilting a camera down moves a ground point UP in the image**, toward `c_y`. The
denominator shrinks and the range reads **long**:

| row `v` | true, nose-down 2° | model assuming 0° | reported / true |
| ---: | ---: | ---: | ---: |
| 800 | 1.328 m | 1.566 m | **1.179** |
| 900 | 0.986 m | 1.117 m | 1.133 |
| 1000 | 0.775 m | 0.859 m | 1.108 |

So the confound offered **cannot** explain a floor-contact estimate that came out
*shorter* than the size prior's. Nor can the other one offered — a `val_min = 40` clipping
an unlit dark base raises the box's bottom edge, which also pushes the range long. Both
run the same way, and it is not the way that was needed.

The 2026-08-20 verdict that method 2 is *not independent* of the calibration it audits is
right and is unaffected; only the mechanism is wrong.

## 4. The bin, on a frame from the robot today

One frame pulled read-only over HTTP from the Go2's front camera on 2026-08-26, with the
staged blue bin in it, committed as `blue-bin-2026-08-26T19-33Z.jpg`. Two estimators over
the same blob:

| | |
| --- | --- |
| size prior, told `height_m = 0.3048` | **1.683 m** (`height`) |
| floor contact, told nothing | **1.784 m** |
| **implied bin height** | **0.323 m** — **+6.0%** on the 0.3048 m the profile states |

The implied height uses no size prior: it is the ground range times the angle the box
subtends. **`BLUE_BIN.height_m` is right to 6%**, which is the last place the 20–30% could
have hidden. For the map to read 24% short, the bin would have to be 0.380 m; the frame
says 0.323 m.

⚠️ **This assumes the lens is 0.32 m up and level.** It is not a free assumption, and the
frame checks it: at a prone 0.10 m lens the same arithmetic implies a **0.101 m** bin,
which is not a bin. And `val_min = 40` rejecting an unlit dark base would raise the box's
bottom edge and push the implied height **up**, so 0.323 m is if anything an over-estimate
— the direction that matters, since the claim under test is that the bin is *taller*.

## What this closes on #35, and what it does not

| "Done when" | |
| --- | --- |
| the parallax scale is computed from at least one existing recorded run, and reported | ✅ two runs, both sections 1 and 2 |
| the bin's true height is measured and `BLUE_BIN` updated, or **confirmed at 0.3048 m** | ⚠️ **confirmed to 6%** by geometry, not by a tape |
| `--prop-height` no longer silently leaves `radius_m` stale | ⛔ untouched here |
| the range scale is asserted by a test against a recorded run | ✅ `test_goal.py` |
| #29's horizon numbers restated at the corrected scale, **or confirmed unchanged** | ✅ **confirmed unchanged** — at k ≈ 1.0 there is no correction to apply |

And by the same argument the `README.md` figures the 2026-08-25 triage flagged as needing
a caveat — *"0.70 m of separation where it needed 0.60"*, *"arrived 0.77 m from the chair
after 2.78 m"*, *"needs ~0.3 m more lane"* — **do not need one.** They were quoted at a
scale that measures 1.0.

## ⛔ What is still open, and it is not nothing

* **The staged gate scene is a different scene.** The 2026-08-20 correction reasoned from
  photographs of the robot stalled between **two bins**, and this measures a marker in two
  peer runs plus one bin today. If those gate bins were a different bin, the map could
  still have read short *for them*, through their prior — and `evidence/live_run_telemetry.jsonl`
  carries no `sightings`, so it cannot be audited. **Section 4 is the recipe: one frame
  with the prop in it settles any prop in about a minute, with no tape and no run.**
* **A tape still bounds a conspiracy this cannot.** Camera and odometry wrong by the same
  factor would read k = 1.0. Nothing here excludes it.
* **Trunk pitch under gait is still unrecorded.** Section 3 corrects the *direction* of the
  confound; it does not measure the angle. One IMU log against the perception clock.
* **Nothing here ran the robot.** One read-only HTTP GET of a frame it was already
  serving. No motion, no lease, no writes.
