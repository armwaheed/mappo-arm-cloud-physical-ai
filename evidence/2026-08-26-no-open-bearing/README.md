<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-26 — an unrangeable frame is not an empty one

Reproduce every number below with no robot:

```bash
python3 no_open_bearing.py
```

## The bug this is about is a gate that fails open

`avoidance.DynamicWindowPlanner.is_feasible` returns `True` unconditionally when the
obstacle list is empty, and `plan` marks every candidate feasible for the same reason.
That is correct when the arena is empty. It is also what happens when the camera is
looking straight at a peer robot it cannot range: `range_detections` drops what it
cannot locate, the drop leaves nothing behind, and **a frame the camera could not read
is bit-for-bit indistinguishable from a clear one.**

That is the shape of the collision in issue #72 — `live05`, 2026-08-25, `--policy-mode
raw`: 0 of 91 driven ticks had an open window toward the goal, and the robot drove into
the peer and pushed it the length of the corridor.

**A robot that keeps steering when no valid bearing exists is worse than one that
stops**, and this directory is the measurement behind making it stop.

## 1. Two floors, and the one people quote is not the binding one

| floor | value | what it is |
| --- | ---: | --- |
| aperture (`r_blind`, the issue's title) | **0.517 m** | a 0.35 m half-extent fills the 85.27° cone |
| contact point, centre-line | 0.721 m | the floor contact leaves the bottom of the frame |
| **contact point, frame corner** | **0.806 m** | the shallowest ray in the frame — **this one binds** |

`0.719 m` is quoted around this repository as *the* near wall. It is the centre column.
An obstacle at the frame edge — which is where an obstacle being swerved past sits —
loses its contact point 0.09 m further out, and **a robot with working ranging therefore
never reaches 0.517 m with anything left to steer by.** The aperture limit is real and
it is the deeper one; the operative floor is the camera's mounting height.

Both move only with the lens or the mounting. Neither moves with software, a better
detector, or more rays — issue #72 already measured that a 16- and a 24-ray fan find no
opening either.

## 2. What "no open bearing" is, as a number a run can be judged on

Four `source` strings in the telemetry are **not measurements of where anything is**:

| source | what it really is |
| --- | --- |
| `frame-fill` | `FILLS_FRAME_RANGE_M`, a constant — 0.8 m whatever is there |
| `width-capped` | `object_fit_range`'s cap. Read 0.719 m to three decimals for five seconds on 2026-08-19 while the bearing swung 12°, and deadlocked the run |
| `ground-clipped` | `GroundRanger` at the near wall |
| `ground-horizon` | a box floating above the skyline |

`ground-far` is **deliberately excluded**. It refuses a *distant* object, which is the
opposite situation, and counting it would fire this on 22 of the hero run's 59 ticks
instead of 1.

The arc those boxes leave open is answerable from the box alone — no range, no size
prior, no class — which is the entire point: **it is answerable exactly when ranging is
not.** The floor it is compared against is `asin(robot_radius_m / near_wall)`, computed
from the calibration and the planner's own radius rather than written down beside them,
so it follows a re-calibration and a change of platform. On this unit with the shipped
`robot_radius_m = 0.40 m` it is **29.75°**.

### Measured over both committed live runs

**16 of 150 ticks** carry a sighting the stack could not locate:

| | widest open bearing | verdict |
| --- | ---: | --- |
| `hero` t=17.53 | **1.05°** | hold |
| `contrast` t=23.31 | **15.37°** | hold |
| `contrast` t=23.14 | **22.64°** | hold |
| `contrast` t=24.24 | **22.99°** | hold |
| `hero` t=17.84 | 32.75° | drive |
| ...11 more | 38.60–65.15° | drive |

Held ticks leave 1.05–22.99°; driven ticks leave 32.75–65.15°. The floor sits **1.42×**
inside the gap between the two populations.

⚠️ **The stricter test was priced and rejected.** `2·asin(r/near_wall)` = 59.50° is the
arc the *whole robot* needs to pass through, and is the more obviously correct test of
passability. It holds **11 of the 16** — including three ticks of the hero run's
successful approach. This gate asks whether a steerable bearing *exists*, not whether
the robot fits through it; the planner's own feasibility check already answers the second
for anything it can range. The band between 29.75° and 59.50° is real and undecided by
any measurement here.

⚠️ **Neither committed run contains the collision this guard is for.** `live05` is not in
this repository. What is measured is that the threshold discriminates on the runs that
*are* committed, and that it costs the hero run **1 tick of 59**, at t=17.53, 0.9 s
before it arrived.

⚠️ **It is single-tick, like the staleness hold beside it, and deliberately not
confirmed over two frames.** A false hold costs 100 ms; a missed one costs a collision.

## 3. `contact_row = 0.472 × box_height + 0.585` is not a ranger, and could not be

That fit — `../2026-08-26-checkpoint-sweep/README.md`, r² = 0.702, n = 1,256 — is a
**data-augmentation placement rule**: where to paste a synthetic peer onto a peer-free
frame so it stands on a plausible ground plane. It reproduces exactly. Read as a *range*
estimator it fails three ways, and it is worth writing down because the arithmetic looks
like ranging:

**It predicts the contact row FROM the box height, which is a size prior wearing
geometry.** When the contact row is in frame you do not need it — you read the row. When
it is not, the row it predicts is an extrapolation of one object class's proportions,
which is exactly the substitution issue #6 exists to remove.

**n = 1,256 is 212 distinct boxes**, and one box is repeated **640 times**. The four
commonest account for 1,002 of the 1,256.

**Within a clip the relationship is absent.** Split by capture clip:

| clip | n | distinct | slope | r² |
| --- | ---: | ---: | ---: | ---: |
| `p4_mid_sweep_stand` | 204 | 183 | −0.051 | **0.007** |
| `p5_1_far_left_stand` | 109 | 4 | −1.206 | **−0.000** |
| `p5_23_far_centre_then_right_stand` | 206 | 23 | +3.823 | **0.000** |
| `p1_close_broadside` | 97 | 1 | — | — |
| `peer01` | 640 | 1 | — | — |

The r² = 0.702 is entirely *between* clips: it is fitting the fact that different staged
setups put the peer at different ranges. Within any clip where the peer actually moves,
it carries **no information at all**.

**And its residuals are worst where stopping distance is set.** Predicting the contact
row and then intersecting it with the floor, against the row the box actually has:

| true range | n | median error | p90 | worst |
| --- | ---: | ---: | ---: | ---: |
| **0.72–1.00 m** | 301 | **43.4%** | 52.0% | 63.2% |
| 1.50–2.00 m | 640 | 11.7% | 11.7% | 11.7% |
| 2.00–2.70 m | 206 | 0.2% | 1.6% | 9.5% |
| 2.70 m + | 109 | 14.0% | 14.0% | 14.0% |

**A fit that is good on average and 43% wrong at 0.9 m is not usable for obstacle
avoidance.** The estimator that is already in the tree —
`camera_model.ground_range`, ray-plane intersection through the box's lowest pixel —
needs no fit: where the contact row is in frame it is *read*, and where it is not, no
fit recovers it.

## What this does NOT establish

* **Nothing here ran on a robot.** Every number is off committed recordings and the
  committed calibration.
* **The threshold has never fired in the arena**, only in a test and against recorded
  boxes. The run that would exercise it is the one issue #72 asks for and nobody has
  done: a peer staged at 1.0–1.5 m in supervised mode, with a contemporaneous
  empty-arena control on the same ticks.
* **This is a stop, not an avoidance.** Issue #72's acceptance has two halves and this
  is the second one. The first — *a peer at 1.0 m is avoided repeatably* — is untouched,
  and issue #88 measured why: MAPPO halts rather than swerving even when handed the
  peer's exact position every tick.
* **Trunk pitch under gait is still unrecorded**, so `--static-detect-pitch-error-deg`
  has no default and the caller states it. That single measurement — one IMU log against
  the perception clock — is still the highest-value thing left on issue #6.
