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
