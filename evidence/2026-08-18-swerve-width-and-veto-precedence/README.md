<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-18 — why the swerve is wide, and which knob actually moves it

Three live runs on the Go2, staged bin between the robot and a chair goal, chasing one
complaint: **the robot swings about two feet wider than it needs to.**

The headline is that two of the three obvious explanations were wrong, and both were
wrong in a way the existing notes actively encouraged.

## Results

| run | veto | heading servo | **lateral swerve** | clearance | yaw range | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| A | on | on | 0.360 m (1.18 ft) | +0.311 m | 27.0° | arrived, 0.75 m |
| B | **off** | on | 0.543 m (1.78 ft) | +0.230 m | 53.5° | arrived, 0.75 m |
| C | on | **off** | **0.328 m (1.08 ft)** | +0.230 m | **16.3°** | timeout, 1.23 m |

`clearance` is free space past the robot's 0.25 m planning disc, to the bin's mapped
surface. `lateral swerve` is the span of cross-track deviation from the straight
start→goal line — it is the number that matches what an observer calls "veering".

All three ran `--policy-scale 2.0`, `max_vy_mps` 0.10, `command_scale` 1.0,
`--robot-radius 0.25`, supervised unless stated.

## Finding 1 — the policy's sensing horizon does not set the pass distance. The VETO does.

The starting theory was that the pass distance is set by where the policy first sees the
obstacle, so shortening its horizon (`meters_per_vmas_unit` 2.5 → 2.0, horizon 0.875 →
0.700 m) would narrow the pass. **It did nothing:** run A's 0.360 m swerve is the same as
run 5's 0.36 m on 2026-08-17 at the calibrated 2.5.

The tick trace says why:

```
 t=6.4  surf 0.900  vy -0.098   veto-avoid   <- PLANNER driving
 t=6.7  surf 0.799  vy -0.144   veto-avoid   <- PLANNER driving
 t=7.5  surf 0.651  vy -0.136   veto-avoid   <- PLANNER driving
 t=7.9  surf 0.606                policy     <- policy first sees it (horizon 0.700)
```

The veto fires at **0.900 m** from the bin's surface, well outside the policy's horizon,
because `is_feasible` rolls the proposed command forward over the planner's **2.5 s**
horizon. The planner has committed the escape three ticks before the policy can see
anything. Lowering the policy's horizon only hands the planner *more* of the job.

Those three veto ticks are 6% of the run and they are the decisive ones. "46/49 ticks
policy-driven" materially overstates how much of the avoidance was the policy's.

`--veto-horizon` now exposes the parameter that actually governs this.

## Finding 2 — removing the veto makes the path WIDER, not narrower

Run B tested the obvious follow-up: drop the veto entirely (`--policy-mode raw`) and let
the policy own the avoidance. It did pull the robot *closer to the bin* — clearance
0.311 → 0.230 m — but the **path got much wider**, 1.18 ft → 1.78 ft, and body rotation
doubled to 53.5°, which is the same signature as the three wall contacts of 2026-08-17.

Two numbers moved in opposite directions, which is why "avoids too widely" has to be
stated as *which* distance. Clearance to the obstacle and excursion from the line are
not the same complaint and do not have the same fix.

Raw also removes the only person-avoidance in the loop: `mappo_bridge` filters to
`kind == "static"`, so people never reach the policy at all.

## Finding 3 — the heading servo drives ROTATION, not width

Run C disabled the servo (`--no-heading-servo`, shipped since 2026-08-17 and never
tried). It did what the servo-instability note predicted for **yaw** — 27.0° → 16.3°,
and the residual is entirely the planner's 13 veto ticks, since the policy's own `wz` is
now hard zero.

It did **not** do much for width: 0.360 → 0.328 m, a 9% gain. So body rotation was not
what was making the path wide. The prediction that it would be was wrong.

What remains is genuine lateral translation: the policy commanding `vy` +0.05…+0.07
steadily, and the veto commanding −0.20 at the end.

## Finding 4 — `max_vy_mps` scales ONLY `vy`, and the note saying otherwise is wrong

`evidence/2026-08-17-corridor-and-room-runs/README.md:149` states the loop's gain lives
in the rotation, "so `max_vy_mps` cannot damp it — it scales `vx` and `vy` together and
leaves `atan2` unchanged". That is false. In `physical_ai_mappo.step()`, `max_vx_mps`
scales `vx` and `max_vy_mps` scales **only** `vy`:

```
max_vy_mps=0.20 -> vx=0.244 vy=-0.152  atan2=-31.9°  wz=-0.400
max_vy_mps=0.10 -> vx=0.244 vy=-0.076  atan2=-17.3°  wz=-0.362
max_vy_mps=0.05 -> vx=0.244 vy=-0.038  atan2= -8.9°  wz=-0.185
```

The claim was used to write off the lateral envelope as a lever after run 2 on
2026-08-17 (that run stalled on the gait floor, which is a `max_vx` property, so the
`max_vy` change was never cleanly evaluated). With the servo off there is no `atan2`
coupling left at all, which makes `max_vy_mps` a direct linear knob on swerve width and
the obvious next experiment.

## What to keep

* **`--no-heading-servo`** — removes a known unstable feedback loop for a real reduction
  in rotation. Costs camera aiming: the robot crabs and its 85° cone never looks anywhere
  new, which is fine for a staged run to a known goal and not fine for exploring.
* **Keep the veto on.** Run B is the argument.
* **Drop `--policy-scale 2.0` and restore the calibrated 2.5.** It bought nothing
  measurable and it de-calibrates the scale against `--robot-radius`.

## Open

* `max_vy_mps` 0.10 → 0.05 with the servo off is the untried experiment for width.
* Goal overrun: both arriving runs stopped 5 cm past the 0.80 m threshold, and that
  threshold is measured centre-to-centre while the Go2's nose is ~0.35 m ahead of its
  centre. See the follow-up issue.
* `policy/config.json`'s `goal_stop_distance_m` of 0.20 m is unreachable while the stack
  halts at `--arrive 0.8`. It fired on zero ticks across all three runs.
