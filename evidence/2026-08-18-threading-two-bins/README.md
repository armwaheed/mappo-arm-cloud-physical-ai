<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-18 — threading a 1.3 m gap between two bins

The robot had to drive **between** two staged bins to reach a chair goal. It took eleven
live runs, and the interesting part is that the first five failures each had a *different*
cause, none of which was the thing being tuned at the time.

## The run that worked

```
OUTCOME                       arrived, 0.77 m from goal
travelled                     3.38 m
ticks driven by the policy    63 / 67   (4 vetoed)
crossed the gate plane at     t = 13.5 s
lateral offset from the gap centre   +0.015 m
clearance                     0.205 m one side, 0.182 m the other
```

**1.5 cm off dead centre** through an aperture with a ±0.403 m envelope. It went down the
middle rather than squeezing past one side.

`run11-SUCCESS-threaded-the-gap.jsonl`. The configuration:

```bash
python3 mappo_drive.py --live \
    --policy-config <max_vy_mps 0.10> --policy-scale 2.0 \
    --static-prop bin --goal-class chair --goal-height 1.067 --goal-crop 0.6 \
    --robot-radius 0.20 --arrive 0.8 --policy-mode supervised \
    --no-heading-servo --veto-horizon 1.0 --policy-gait-floor 0.35 --max-seconds 40
```

| flag | what it fixed |
| --- | --- |
| `--no-heading-servo` | the unstable yaw feedback loop |
| `--veto-horizon 1.0` | the 2.5 s default rolls 0.875 m — wider than the gap |
| `--policy-gait-floor 0.35` | policy commands below the gait floor |
| `--goal-crop 0.6` | chair detection 0.981 vs 0.260 at the 0.5 default |
| a long approach | lets `position_sigma` converge so the mapped radii shrink |

## Finding 1 — the mapped radius, not the bins, decides whether the gap exists

`StaticLandmark.planning_radius_m` is `radius_m + position_sigma`. The bin's physical
radius is 0.150 m; `position_sigma` starts near 0.27 m after two sightings and falls below
0.10 m within ten. So the *same two bins*, never moved, present as:

| map state | mapped radius | surface gap | tolerance |
| --- | --- | --- | --- |
| converged (~10 sightings) | 0.23 m | 0.88 m | **±0.12 m** — passable |
| young (~2 sightings) | 0.34–0.40 m | 0.65 m | **±0.005 m** — impassable |

Centre-to-centre measured 1.38–1.40 m in every single run. **A 0.23 m swing in the
aperture came entirely from how sure the map was.** Every attempt that set off on a young
map failed, and the diagnosis each time looked like a control problem.

The practical consequence: give the robot a long enough approach to accumulate sightings
*and* parallax before it reaches the aperture. `run11` did; the earlier runs did not.

## Finding 2 — the veto's horizon was wider than the gap

`is_feasible` rolls a proposed command over the planner's 2.5 s horizon, which at
0.35 m/s is **0.875 m of travel — essentially the width of the 0.93 m aperture**. It was
therefore asking "will I still be clear a metre *past* the gap?" while the robot was still
approaching, and any small heading error over that distance breaches. It fired at 0.900 m
from a bin's surface, outside the policy's own 0.700 m horizon, so the planner committed
the escape before the policy had seen anything.

`--veto-horizon 1.0` (0.35 m of lookahead, still above the 0.32 m required clearance)
took the veto from 32 firing ticks to 0. `run1` vs `run7`.

## Finding 3 — a full-speed reverse into unsensed space. Fixed.

`run9-full-speed-reverse-hazard.jsonl`, and the most serious thing found today:

```
policy v=(-0.35, -0.03)     goal distance 2.64 -> 2.73 m
```

The vendored planner states its own rule plainly — *"Reverse is deliberately not sampled.
The Go2 has no rear-facing sensing on this unit, so backing away from a person means
moving blind into space this pipeline has never observed."* That rule was applied to the
planner's velocity sampling and **not** to the policy path, which clamped to
`max(-max_vx, min(max_vx, vx))` and so permitted −0.35 m/s. The policy is a holonomic
agent with no notion of where the sensors point, and it used it.

`--policy-gait-floor` made it worse before it was caught: the scaling multiplies the whole
vector, so a −0.03 m/s drift became a committed 0.35 m/s reverse.

Both fixed: the policy now gets the same forward-only clamp the planner has, and the
scaling refuses any command with `vx <= 0`. Two tests, and the reverse one was
mutation-checked — it fails on the pre-fix clamp.

## Finding 4 — a pure strafe can never be walkable on this robot

`max_vy` is 0.20 m/s. The gait floor is ~0.35 m/s. **A predominantly sideways command
cannot reach the floor by construction**, so `--policy-gait-floor` cannot rescue one — it
would only scale it into the fastest crab the envelope allows. `run10` shows the clamped
result: `v=(+0.00, -0.08)`, held for four seconds, robot stationary.

This matters because a slow lateral escape is exactly what the policy chooses in tight
space. It is a genuine policy/platform mismatch, not a tuning gap.

## What each failure actually was

| run | looked like | was |
| --- | --- | --- |
| 1 | berth too wide | planner crawling at 0.137 m/s under the gait floor |
| 2 | robot fault / corrupt state | the operator was driving it backwards |
| 5 | goal never sighted | `--goal-crop` 0.5 scored 0.260 against a 0.5 threshold |
| 7 | berth too wide | policy commanding 0.10 m/s, unwalkable |
| 9 | — | full-speed reverse into unsensed space |
| 10 | berth too wide | young map: aperture 0.65 m, tolerance ±5 mm |
| 11 | — | **arrived** |

Three separate failures presented as "it avoids too widely". None of them were.

## Open

* Goal overrun: `--arrive` is measured robot-centre to the goal's *estimated centre*,
  while the Go2's nose sits ~0.35 m ahead of its centre. Stopping at 0.77 m puts the nose
  inside a typical chair's footprint. See issue #25.
* `MIN_GAIT_COMMAND_M_S` is 0.35 but run C on the same day sustained 0.295 m/s with 54/54
  ticks below it and walked 3 m. The constant is a guess the data contradicts. Issue #26.
* `--policy-scale 2.0` remains unjustified — it never measurably changed anything. The
  calibrated 2.5 should be restored once someone has a run to spare.

---

# Addendum — why run 11 was the outlier, and what a retrain needs

Four further runs after the fixes above merged. Run 11 did **not** repeat, and finding out
why produced the clearest result of the day.

## The discriminator is a single ray

Replaying every run through the real checkpoint and reading the observation it was handed:

| | ray 0 (the START HEADING) blocked | commanded vx | outcome |
| --- | --- | --- | --- |
| run 11 | **0 / 49 ticks** | mean **+0.220** | threaded the gap |
| run 14 | **29 / 29 ticks** | mean **−0.097** | stalled, retreating |

Run 11's hot-ray pattern sweeps `(1,) → (1,2) → (1,2,3) → (2,3,4,10,11) → (5,8) → (8,)` —
the two bins rotating from ahead-left round to behind as the robot passed between them.
Run 14 only ever saw `(0,)` and `(0,1)`.

At the stall, run 14's observation was:

```
ray  0 (  0°)  0.1933      raw action: ax=-0.685 ay=-0.727  (back and right)
ray  1 ( 30°)  0.2010
ray  2-11      0.0000
```

**The policy did not see a gap. It saw a wall, and backing away was the correct response
to that observation.** Both bins landed on adjacent rays and the aperture between them fell
in the unsampled space between them. With 30° spacing there is no ray at 15° to report
clear.

## Two different tolerances, and the tighter one is not the sensor

The fan is anchored in the **run-local frame fixed at reset**, so the robot's yaw on the
first tick decides where every ray points for the whole run. That gives two separate
constraints, and they were conflated in the session that found this:

* **Ray latitude, ±15°** — beyond this the gap centre slides off ray 0 onto a bin and the
  policy stops being able to see the aperture at all.
* **Path corridor, ±5.2°** — `atan(0.173 / 1.9)` for a ±0.173 m corridor at 1.9 m. Beyond
  this the robot is inside the aperture's angular window but outside its lateral one.

Run 15 was aimed **8.4° off the gap centre**: inside the ray latitude, so the policy could
see the gap, and outside the path corridor, so driving along ray 0 walked it into the near
bin's flank. It stopped 0.446 m from landmark-1 having never reached landmark-2 (1.405 m).

## What this means for the checkpoint

Threading this aperture currently requires hand-aligning a **30°-resolution sensor** with a
**0.83–0.99 m gap** to within **5°**. That is achievable — run 11 did it — but it is not
robust, and one success in roughly six attempts is the honest hit rate.

The concrete ask for a retrain, with the numbers that justify it:

| fan | spacing | path corridor it can resolve |
| --- | --- | --- |
| 12 rays / 360° (**delivered**) | 30° | aperture invisible when it straddles two rays |
| 16 rays / 360° | 22.5° | ~1.3× better angular resolution |
| 24 rays / 360° | 15° | a ray lands in a 0.9 m gap at 1.9 m from any approach angle |

A finer fan is the only fix that removes the hand-alignment. Everything else in this
directory is a workaround for a sensor that cannot resolve the hole it is being asked to
drive through.

## Also observed

* **A `person`-labelled track inflated to a 1.67 m radius whose disc covered the goal**
  (run 13, 24 of 212 ticks, velocity ~0). It did not cause the stall — the bridge filters
  to `kind == "static"` before the policy sees anything, and the planner issued zero holds
  — but during a demo a person standing near the goal puts the goal inside an obstacle
  from the planner's point of view. The humanoid in that bay is a likely source.
* **The goal pass runs at `--goal-input-size 224`, not 300.** At 224 every crop fraction
  scored **below** the default 0.5 `min_score` (best 0.428), which is why the chair went
  unsighted repeatedly. `--goal-crop 0.8 --goal-input-size 300` scores 0.622 with a single
  candidate; the same crop at 224 scores 0.279. Any detector measurement taken at a
  different `input_size` from production describes nothing.
* **`--goal-crop 0.7` shows two chair candidates (0.428, 0.285)**, which is the documented
  goal-hopping condition. Measured live: the goal estimate jumped 0.84 m mid-run and the
  robot drove to the new one.
