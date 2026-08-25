<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-24 — peer capture, and four gait-floor sweeps that kept overturning each other

Two things happened on the robot today. A **1,961-frame in-domain dataset** of a Go2 Wheel
peer was captured through the Go2 Walk's own camera, and a **gait-floor investigation**
produced three confident conclusions in a row, two of which were wrong.

## The peer dataset

⚠️ **The 1,961 frames are NOT in this repository.** They are 394 MB of JPEG — six times
the whole of `.git` — and the largest existing evidence directory here is 29 MB. `sample/`
holds one representative frame per segment; the full set lives outside git (see *Where the
data is*).

Captured with `tools/shot.py`, which stands the robot, films, and lies it back down **in one
process**. That shape is deliberate: the D1 arm loads the hind legs continuously, so the
earlier protocol — stand, run a separate capture, run a separate lie-down — held the legs
under load through two extra DDS connections and two extra arm latches. The operator
stopped a run over hot legs, which is what prompted it. The lie-down is in a `finally`, so a
capture that raises still parks the robot.

| segment | frames | posture | what it covers |
| --- | --- | --- | --- |
| `p1_close_broadside` | 97 | **prone** | ~0.6 m, side-on. Camera prone despite the name — a standing retake is owed |
| `p2_close_headon` | 98 | standing | ~0.6 m, nose-on. The 2.3x aspect case: 0.31 m wide vs 0.70 m long |
| `p3_close_rearon` | 133 | standing | ~0.6 m, tail-on — a silhouette a side-trained detector misses |
| `p4_mid_sweep` | 208 | standing | ~1.5 m crossing left→right. The only moving segment, and the closest thing here to the demo event |
| `p5_1_far_left` | 109 | standing | ~4 m, left of centre |
| `p5_23_far_centre` | 206 | standing | ~3.5–4.5 m, centre |
| `p6_1_trunc_left` | 140 | standing | clipped at the left edge, ~15% visible |
| `p6_2_trunc_right` | 136 | standing | clipped at the right edge, ~15% visible |
| `p6_3_trunc_half` | 136 | standing | **cut roughly in half** — the band that matters at avoidance range |
| `peer01` (aborted run) | 640 | prone | ~2 m broadside, static |

Truncation is over-represented on purpose. A peer close enough to swerve around is clipped
by the frame edge almost by definition, and it is the case the synthetic generator models
worst — its sprites are pinhole-rendered against an equidistant fisheye, and that error is
largest at the edges.

**Two hardware facts the session established.** The shipped detector returns **nothing** for
this robot at any confidence down to 0.02 — except from broadside, where it reads
`motorbike` at 0.57 on 2 of 2 live frames while firing on **0 of 15** of the earlier
peer frames. A viewpoint accident, not a detector. And the printed ArUco goal marker was
detected on the robot's own camera at **6.43 m**, confirming the sheet works at range.

## The gait sweeps, and the order in which they were wrong

`tools/gait_sweep.py`. Every velocity held steady, travel from `SportModeState` odometry,
`vx` never negative, robot parked prone in a `finally`.

**`sweep_z.json` is the null control and should be read first.** Standing, commanding zero,
it records **0.001 / 0.000 / 0.000 / 0.000 m**. Every other number is only interpretable
against it, and it did not exist until the third reversal — which is the lesson of the day.

### What was concluded, and then unconcluded

1. **"The forward floor is not 0.35."** Commands of 0.100 produced 0.114 m of travel
   (`sweep_a`). **Survives**, with a caveat below.
2. **"The floor ellipse fails at its own anchor."** 0.200 m/s lateral gave 0.010 m, twice
   (`sweep_b`, `sweep_b2`). **Overturned** — `sweep_c` then walked on the identical command.
3. **"It is position-dependent."** Four trials of the same command, 1 walked. **Overturned**
   — position was confounded with trial order, because forward trials carry the robot
   0.5–0.75 m and every trial therefore sampled somewhere new.
4. **"The first-trial walks are the robot settling out of `stand_up()`."** **Overturned by
   the null control**: a standing robot with zero command records 0.000 m, so those were
   real motion.

### What is actually established

* `MIN_GAIT_COMMAND_M_S = 0.35` **is not a floor.** 0.100 m/s produces travel.
* Commands **≥ 0.25 m/s forward walked in every trial**, regardless of order.
* **The same low command walks or stalls depending on the robot's motion state.** Every
  low-speed success was the FIRST trial after standing — 0.114, 0.120, and 0.127 m for
  lateral — while every repeat of those commands gave 0.004–0.013 m. A **starting** floor
  above a **sustaining** floor would explain it, and it echoes what #31 records about the
  dynamic window being `[0, 0.05]` from rest at 10 Hz.

### What is NOT established, and wants a designed experiment

The starting floor. The sustaining floor. Whether surface matters — the corridor has a
metal strip dividing polished concrete from carpet, which is a **hypothesis nobody tested**.
And whether the ellipse *shape* is right, independently of its numbers.

The experiment that would settle it holds **position fixed**, varies only motion state,
repeats every condition, and runs the null control **between every trial**. It needs a cold
robot: this session ended at 48 °C against a 50 °C ceiling, with the D1 having crept out of
its 3.0° stow gate twice.

## One more correction

`Go2Locomotion` warns that motion mode `'mcf'` is "not a sport mode — Move commands may be
ignored". Measured: the mode stayed `'mcf'` all session, `SelectMode('normal')` printed that
it was switching and silently **did not**, and every Move was obeyed — up to 0.707 m in 3 s.
The warning is misleading on both halves.

## Where the data is

`sample/` has one frame per segment. The full 1,961 frames are on the robot at
`~/peercap/` and archived off it — a robot's home directory is not storage, and this one has
been found reflashed before.
