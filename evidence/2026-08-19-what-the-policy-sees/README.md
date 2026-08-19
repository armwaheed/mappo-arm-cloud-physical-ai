<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-19 — what the policy actually sees, and what really stopped it

Answering @spsagar13's question on [issue #29](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/29):
*how does the robot perceive the scene, is the camera→lidar interface producing the right
angles and distances, and would a retrained model with more rays get stuck the same way?*

It is a good question and the honest answer moved the diagnosis. **The 12-ray fan is a
real limit, but it is the third of three things in the way, not the first — and one of
the other two was a bug in the adapter, which is now fixed.** The retrain is still worth
doing; the ask in #29 needs one number changed and one added.

Everything here is measured on the recorded telemetry of 2026-08-18. No robot was moved.
Reproduce the whole table with `python3 radius_latch.py`.

---

## 1. How a camera becomes twelve numbers

There is no LiDAR on this robot and no ray casting against an image. The chain is:

| stage | where | what it produces |
| --- | --- | --- |
| detect | `visual_nav` colour profile / VOC detector | a pixel box for a `bin` in one frame |
| range | `camera_model` | distance from the box's floor contact point and the known camera height (0.32 m), bearing from the pixel column and the measured 85.27° HFOV |
| fuse | `static_map.StaticObstacleMap` | a **landmark** in odom: `(x, y)` from a Kalman fuse over sightings, plus `planning_radius_m = radius_m + position_sigma` |
| hand over | `integration/mappo_bridge` | the subset with `kind == "static"`, as distance/bearing/radius/id |
| retain | `physical_ai_mappo._update_obstacles` | the controller's **own** map of discs, in a run-local frame |
| cast | `physical_ai_mappo._ranges` | 12 rays from the robot's position against those discs |
| encode | `physical_ai_mappo._observation` | `lidar = lidar_range_vmas − range/scale` |

Three properties of that chain are not the obvious ones, and all three matter for reading
the numbers in #29:

* **The rays do not point where the camera points, and they do not turn with the robot.**
  They sit at `2πi/12` in a frame fixed at the first `reset_run` tick. Ray 0 is the
  heading the run started in, for the whole run. Ray 1 is 30° left of it — not 30° left of
  the nose, and not 6° or any other camera-derived spacing. The trained VMAS agent is
  holonomic and never rotates, so this is faithful to training; it is just not the frame
  a camera person would assume.
* **The values are proximity, not distance.** `lidar_range_vmas − range/scale`, so
  **bigger means closer** and **0.000 means clear**, including "clear because it is past
  the horizon". A ray reading 0 tells you nothing about whether the space is empty.
* **The horizon is 0.875 m.** `lidar_range_vmas` 0.35 × `meters_per_vmas_unit` 2.5,
  measured to the obstacle's *surface*. Everything beyond that reads exactly as clear
  floor.

`integration/render_observation.py` draws all of it — the annotated camera frame beside
the fan and the observation vector, per tick. The four PNGs here came out of it.

## 2. Is the camera→map interface getting the angles and distances right?

**Yes, and it converges to about a centimetre.** Per landmark, over each run:

| run | landmark | sightings | drift from first fix | spread over last 10 | centre-to-centre (min…max) |
| --- | --- | --- | --- | --- | --- |
| 11 | landmark-1 | 78 | 0.289 m | 0.000 m | 1.260 … 1.333 m |
| 11 | landmark-4 | 71 | 0.252 m | 0.000 m | |
| 14 | landmark-1 | 49 | 0.251 m | 0.009 m | 1.379 … 1.419 m |
| 14 | landmark-2 | 49 | 0.120 m | 0.000 m | |
| 15 | landmark-1 | 41 | 0.213 m | 0.008 m | 1.372 … 1.429 m |
| 10 | landmark-1 | 45 | 0.198 m | 0.004 m | 1.368 … 1.391 m |

Each estimate walks 0.11–0.29 m from its first fix as parallax accumulates, then stops
moving. Runs 10, 14 and 15 staged the bins in the same place and independently recover
1.368–1.429 m between them, against a hand measurement of 1.38–1.40 m. Run 11 was a
different staging at 1.27 m.

So the geometry is sound. **The mapping is not the problem.** What happens to it next is.

## 3. Finding one: the policy never had both bins in range on any failing run

The single cleanest discriminator in the whole day, and it is not about rays at all:

| run | outcome | ticks with **both** bins inside the 0.875 m horizon |
| --- | --- | --- |
| **11** | **threaded the gap** | **33 of 79** |
| 10 | stalled, pure strafe | **0** of 46 |
| 14 | stalled, retreating | **0** of 50 |
| 15 | walked into a flank | **0** of 42 |

The bins are 1.27–1.43 m apart and the horizon is 0.875 m to the surface, so both sides of
the aperture are in range only from close to the midline. Off the midline the policy sees
**one** disc and open floor — and "drive between two things" is not a manoeuvre it can
choose, because only one of the two things exists in its observation.

`run14-t06.5-both-bins-in-view-observation-empty.png` is that in one frame: both bins are
plainly visible in the camera, 1.7 m and 2.7 m away, and the observation handed to the
network is **twelve zeros**.

This is a horizon problem, and no ray count fixes it. Sweeping the horizon against the
poses each run actually flew — how far the *further* bin's surface was, tick by tick:

| horizon | `lidar_range_vmas` at 2.5 m/unit | run 10 | run 14 | run 15 | run 11 |
| --- | --- | --- | --- | --- | --- |
| **0.875 m** (delivered) | 0.35 | 0/45 | 0/49 | 0/41 | 33/71 |
| 1.25 m | 0.50 | 21/45 | 0/49 | 0/41 | 46/71 |
| 1.50 m | 0.60 | 29/45 | 23/49 | 22/41 | 64/71 |
| 2.00 m | 0.80 | 43/45 | 35/49 | 38/41 | 71/71 |
| 2.50 m | 1.00 | 45/45 | 49/49 | 41/41 | 71/71 |

The closest the failing runs ever came to having both bins in range was 1.23 m (run 10),
1.42 m (run 15) and 1.45 m (run 14). Run 11 needed only 0.41 m, because it went down the
middle.

## 4. Finding two: the retained radius never converged — the adapter latched it

`planning_radius_m` is `radius_m + position_sigma` ([`static_map.py:161`](../../robot-stack/unitree/go2/visual_nav/static_map.py)),
an estimate that *starts large and shrinks* as sightings accumulate. Every landmark in
every run above converges from 0.38–0.47 m to **0.230 m**.

The controller kept the opening value. `_update_obstacles` combined a re-observation with
`match.radius = max(match.radius, radius)`, which makes the mapped radius a high-water
mark that can never come down:

| run | telemetry reports, first → last | controller retained, whole run |
| --- | --- | --- |
| 11 | 0.396 → 0.230, 0.418 → 0.230 | 0.396, 0.418 |
| 10 | 0.384 → 0.230, 0.450 → 0.230 | 0.384, 0.450 |
| 14 | 0.402 → 0.230, 0.472 → 0.230 | 0.402, 0.472 |
| 15 | 0.379 → 0.230, 0.451 → 0.230 | 0.379, 0.451 |

So the policy planned the entire run against the map's least certain moment, while the
planner beside it used the converged value. It is a silent failure in the way this
repository keeps finding: an over-large disc produces a completely well-formed range
vector. It just reports a gap the robot cannot fit through.

**It also inverts the fix that finding 1 of 2026-08-18 prescribed.** "Give the robot a
long approach so `position_sigma` converges and the mapped radii shrink" is correct, and
none of that convergence was reaching the policy.

Worse, it reverses the geometry on approach. Walking closer *should* open the aperture:

| run 14, t → | 4.6 s | 6.0 s | 12.0 s | 13.6 s |
| --- | --- | --- | --- | --- |
| distance to landmark-1 | 1.94 m | 1.73 m | 0.71 m | 0.69 m |
| its half-angle, latched | 12.0° | 13.4° | 34.5° | 35.6° |
| its half-angle, converged | 12.0° | 10.2° | 18.9° | 19.5° |
| **aperture width, latched** | 7.8° | 8.1° | **4.7°** | **3.2°** |
| **aperture width, converged** | 7.8° | 14.2° | **28.7°** | **27.8°** |

@spsagar13's intuition — *"when it walks closer, the angle should increase"* — is exactly
right for the obstacle's own half-angle, and it holds in both columns. But under the latch
the near bin's frozen radius swelled faster than the geometry opened the hole, so **the
aperture got narrower the closer the robot came.** That is the behaviour the question was
probing at, and it was a bug.

**Fixed** as CORRECTION 6 in `policy/physical_ai_mappo.py`: a re-observation of an object
the producer has *named* takes the radius it reports now; only an anonymous positional
merge still keeps the larger of the two, because there the disc genuinely has to cover
both detections. Two tests, both mutation-checked against the delivered rule.

`run14-t12.3-latched-radius-no-window.png` and
`run14-t12.3-converged-radius-window-unsampled.png` are the same tick before and after.
In the first, the goal arrow points straight into a disc that is not there.

### What the correction is worth, on the recorded runs

| | run 10 | run 14 | run 15 | run 11 |
| --- | --- | --- | --- | --- |
| mean commanded `vx`, latched | −0.037 | −0.007 | −0.039 | +0.269 |
| mean commanded `vx`, converged | **+0.265** | **+0.198** | **+0.271** | +0.326 |
| ticks with ray 0 blocked, latched | 34 | 31 | 25 | 0 |
| ticks with ray 0 blocked, converged | **0** | 24 | **0** | 0 |
| median clear window, latched | 13.7° | 8.2° | 10.2° | 165.8° |
| median clear window, converged | **39.7°** | **27.1°** | **28.4°** | 198.2° |

Three runs that were commanding a net reverse now command a net forward, from a one-line
change and no retrain.

⚠️ **The correction makes the policy react LATER, and that is the point.** A landmark's
retained surface moves 0.17 m further away (0.402 → 0.230 m of radius), so the ray that
was firing at 0.47 m of true clearance now fires at 0.65 m. The policy is not becoming
less careful than the rest of the stack — it is becoming *consistent* with it, because the
planner has always used the converged radius, and the planner's veto, unchanged, remains
the safety envelope. But it does mean the first hardware run after this change will see
the robot commit later than the last one did, and that should be expected rather than
diagnosed.

⚠️ **This is an open-loop replay.** The recorded poses are fixed, so "the command flips
forward" is not "it would have threaded the gap" — the robot never got to act on the
corrected observation. It needs a hardware run. Note also that the closed-loop simulator
**cannot** reproduce this: `closed_loop_sim.SimObstacle` deliberately carries the true
radius and does not model `position_sigma` inflation at all, which is why 30 seeded
scenarios never found it.

## 5. Finding three: with the radii corrected, 12 rays still miss the hole

Now the question #29 actually asked. Taking the clear angular window toward the goal and
asking whether any ray of an N-ray fan lands inside it:

| | run 10 | run 14 | run 15 | run 11 |
| --- | --- | --- | --- | --- |
| **12 rays** (delivered) | 34/46 | **3/50** | 41/42 | 79/79 |
| **16 rays** | 37/46 | **33/50** | 41/42 | 79/79 |
| **24 rays** | 46/46 | **48/50** | 42/42 | 79/79 |

Run 11 — the one that worked — had a 12-ray fan sampling its window on **every single
tick**. The fan was never its problem; its approach line put the aperture at a bearing the
existing rays happened to cover. Run 14's window sat at **[−29.3°, −0.7°]** for most of
its stall: between ray 11 at −30° and ray 0 at 0°, **missing each of them by under a
degree**. Twelve rays found it on 3 ticks out of 50.

**So: would a finer fan also get stuck?** On this evidence, at 24 rays, mostly no —
*provided the radius fix is in*. Without it, 24 rays sample run 14's window on only 15 of
27 open ticks, and **16 rays are no better than 12** (1/27 either way), which is the part
worth knowing before paying for a retrain.

## 6. What we would change in the ask on #29

The deliverable in #29 is unchanged — same `.npz` shape, same `metadata_json`, same
conventions. Two changes to what to train:

1. **`training_lidar_range_vmas` 0.35 is the binding constraint, not the ray count.**
   At the calibrated 2.5 m/unit it buys a 0.875 m horizon, and the policy needs to see
   both sides of a 1.4 m aperture from further out than that. **0.80** (a 2.0 m horizon)
   puts both bins in range on 35/49, 38/41 and 43/45 of the failing runs' ticks; 0.60 only
   manages about half of that and 1.00 covers everything. Our ask is **0.80, with 1.00 if
   it is free.** Raising `meters_per_vmas_unit` instead is not equivalent: it is
   calibrated as the planner's 0.25 m robot radius ÷ the trained 0.10 VMAS agent radius,
   so moving it de-calibrates the agent's own size.
2. **24 rays, and drop 16 from the fallback.** 16 rays measurably do not help on the run
   that failed. If 24 is expensive the honest ordering is horizon first, rays second.

The four platform constraints in #29 (gait floor 0.35 m/s, lateral cap 0.20 m/s below it,
no reverse, the +0.014…+0.023 m/s lateral bias) all still stand as written.

## 7. Also worth recording

* **A run's telemetry does not record the policy configuration it ran with.** The header
  carries the camera, planner and envelope, but not `--policy-scale`, `--veto-horizon`,
  `--policy-gait-floor`, `--max-vy` or `--no-heading-servo`. Every replay in this
  directory therefore runs at the repository default, which is the right baseline for
  comparing runs against each other and is *not* necessarily what the robot did. Worth
  fixing before the next hardware day.
* **The addendum in `../2026-08-18-threading-two-bins/README.md` had the mechanism wrong**
  and has been corrected in place. It read the stall as "both bins landed on adjacent
  rays". Only one did: at the stall landmark-2's centre was 1.69 m away and its surface
  1.22 m, against a 0.875 m horizon, so it was invisible. Rays 0 and 1 were both blocked by landmark-1 alone, whose latched 0.402 m
  radius made it subtend 69°. The conclusion — that the fan could not resolve the
  aperture — survives; the reason it could not was mostly the radius.

## Files

| | |
| --- | --- |
| `radius_latch.py` | the whole before/after table, from one checkout, no robot |
| `run14-t06.5-both-bins-in-view-observation-empty.png` | finding 1: the horizon |
| `run14-t12.3-latched-radius-no-window.png` | finding 2, before |
| `run14-t12.3-converged-radius-window-unsampled.png` | finding 2, after — and finding 3 |
| `run11-t12.1-between-the-bins.png` | the success: both bins mapped, rays 3–4 and 10–11 hot, a 106.9° window the 12-ray fan sees |

Frames for the camera panels are extracted from the run's own MP4 on the robot
(`perception.video_frame` indexes it directly); the panels above are committed because
the videos are not.
