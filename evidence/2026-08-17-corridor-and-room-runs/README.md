<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-17 — six live runs, two rooms, and the first policy-driven avoidance

Goal for the day: find out whether the MAPPO demo avoids **stationary objects, moving
objects and humans** on the Go2. The short answer is that it avoids a stationary object,
that this was demonstrated for the first time on run 5, and that the other two were never
reachable — not because they failed, but because nothing in the pipeline can represent
them. That is explained under *What was never testable* below.

All six telemetry files are here. The `.mp4` recordings stayed on the robot when it was
disconnected and should be collected next session (`~/run{0,1,2,3,4,5}*.mp4`).

## The runs

Staging A = office corridor, ~1.15 m wide, cubicles one side and meeting rooms the other.
Staging B = cramped equipment room. Both used the blue recycling bin as the obstacle and
an office chair as the goal. Every run: `--policy-mode supervised --robot-radius 0.25
--confidence 0.45 --arrive 0.8 --no-latch-arm`, `command_scale` 1.0, scale 2.5 m/unit.

| # | staging | controller | change from previous | outcome |
| --- | --- | --- | --- | --- |
| 0 | A | **planner** (incumbent) | — | **arrived, 0.80 m from goal** |
| 1 | A | MAPPO | first policy run | contact with a cubicle panel |
| 2 | A | MAPPO | `max_vy_mps` 0.20 → 0.10 | stalled below the gait floor |
| 3 | A | MAPPO | + `control_dt` fix | contact with a cubicle panel |
| 4 | B | MAPPO | new room | contact with a cabinet |
| 5 | B | MAPPO | bin repositioned | **success — 58/58 ticks policy-driven** |

### Run 0 — the contemporaneous baseline

Deliberately the incumbent planner, not MAPPO, and run first to validate a 307-line
refactor of `visual_nav.py` that had never moved a leg. It also gives a same-day paired
control rather than a number recorded three days earlier.

2.41 m forward, 2.56 m path, **0.58 m of lateral swerve**, +26.7° of yaw, minimum
clearance to the bin's surface **0.472 m** against the 0.37 m required. 3.7% of ticks
perception-stale. Arrived.

### Run 5 — the first genuinely policy-driven avoidance

| | |
| --- | --- |
| ticks driven by the policy | **58 / 58 — zero vetoed, zero stopped** |
| forward travel | 2.93 m (3.13 m path) |
| lateral swerve | 0.36 m |
| minimum clearance to the bin's surface | **0.532 m** vs 0.37 m required |
| peak yaw | −20.3° / +11.8° |
| measured speed | 0.222 m/s mean while commanded 0.35 (gain 0.63) |
| outcome | timed out at **0.89 m**, threshold 0.80 — 9 cm short |

Raise `--max-seconds` from 20 to 25 and it arrives. That is the only thing between this
run and clean footage.

**The swerve was caused by the bin, not coincidence.** `replay_mappo.py` re-runs the
recorded telemetry through the policy twice, identical but for whether the obstacle is in
the observation:

```
  ticks with a static obstacle 58
  ... inside that horizon      15 (26%)
  steering, in degrees:
    CAUSED by the obstacle     max  67.7, mean  15.9   <- vs the ablated control
    ... on the 15 ticks it could see one: max  54.4, mean  34.8
    direction reversals        0 over 26 swerving ticks
  bridge mapping was clean
```

**34.8° mean deflection attributable to the bin** on the ticks where it was within the
0.875 m sensing horizon, and zero chatter. This is the claim the demo needs, and it is
measured rather than asserted.

> **CORRECTED 2026-08-18.** The block above is what the tool printed on the day, and its
> seen/unseen split was measured from the wrong object — see `CODE-REVIEW.md` A3 and
> issue #17. With `policy_sight()` reading visibility out of the policy's own observation,
> the same telemetry gives **35.9° mean over 31 ticks**, not 34.8° over 15, and the
> "could not see one" row falls from 13.8° to **exactly 0.0°**, which is what it must be.
> The headline conclusion is unchanged and slightly stronger; only the breakdown moved.
> The tool now also reports the 16 ticks where the policy steered on a remembered
> obstacle the telemetry no longer carried (issue #19).

Two qualifications before anyone quotes it. Eleven of seventy ticks were
`STOP_EXTERNAL_HOLD` — the policy stopping itself on stale perception. And the replay
notes that *"command mapping caps vy at 0.1 m/s against vx's 0.35, so a 45-degree intent
is issued as 16 degrees"*: the policy is intending a substantially larger manoeuvre than
the robot executes.

**It is one run.** Run 0's baseline also succeeded, on different staging. Nothing here
supports a claim that the policy beats the incumbent; it supports the claim that the
policy can drive the avoidance unaided.

## Finding 1 — `control_dt` used the nominal loop period. FIXED.

`run()` passed `control_dt=period`, a hard-coded `1.0 / control_hz`, and the planner sizes
its dynamic window as `accel * control_dt`. That is only correct while the loop keeps up.
It does not: **the control loop measured 2.78 Hz against a nominal 10 Hz.**

So the planner allowed 0.05 m/s of velocity change per tick over an interval in which the
robot could deliver 0.18. The ramp from a standstill took 2.5 s instead of 0.7 s, and a
stale-perception hold — which zeroes the command and restarts the ramp — arrived every
1–2 s. The command never reached the 0.35 m/s gait floor, so the robot stood still while
commanded forward, and the stall gate reported *"something is holding the robot — check
the tether."* On run 2, **18 of 19 non-stale forward ticks were commanded below the gait
floor**.

Fixed by measuring the real interval, floored at the nominal period and capped so one long
tick cannot open the window to the whole envelope. Run 2 → run 3, same staging, minutes
apart:

| | run 2 | run 3 |
| --- | --- | --- |
| loop rate | 2.78 Hz | **3.58 Hz** |
| forward ticks below the gait floor | **95%** | **56%** |
| did the command ever hold 0.35? | never | **yes** |

**This is a defect in the upstream control stack, not in the policy**, and it degrades the
incumbent planner in exactly the same way. It is fixed here in `robot-stack/` and in the
upstream repository in the same session, per `PROVENANCE.md`.

## Finding 2 — the heading servo is an unstable feedback loop. NOT fixed.

Three of the four failures were the same event: the servo saturated at `wz = −0.40 rad/s`,
the robot rotated 34–54° to the right, and drove its shoulder into something.

| run | peak yaw | lateral travel | outcome |
| --- | --- | --- | --- |
| 0 (planner) | **+26.7°** | 0.57 m | arrived |
| 1 | −33.9° | 0.26 m | cubicle panel |
| 3 | −54.2° | 0.44 m | cubicle panel |
| 4 | −36.4° | 0.30 m | cabinet |

Note the failures moved **less** sideways than the success. The problem was never a wide
swerve — it was body rotation.

The mechanism: `mappo_policy.step()` rotates the policy's action from the run-local frame
into the body frame by `yaw_local = yaw − origin_yaw`. `HeadingServo` then yaws the robot
toward that body-frame direction. But yawing *changes* `yaw_local`, which rotates the
action further:

```
yaw ↑ → yaw_local ↑ → body_x = cos(yaw_local)·ax + sin(yaw_local)·ay  collapses
     → atan2(vy, vx) grows → servo saturates → yaw ↑ …
```

Forward speed collapsed to 0.03 m/s on run 3 while the robot spun. The loop's gain lives
in the **rotation**, so `max_vy_mps` cannot damp it — it scales `vx` and `vy` together and
leaves `atan2` unchanged. Halving it made run 3 *worse* than run 1 (−54° vs −34°) by
giving the loop longer to wind up.

The policy was trained as a non-rotating holonomic agent with rays at fixed world angles.
Bolting a heading servo onto it closes a loop the training never had.

`--no-heading-servo` is the existing flag that removes the mechanism and **was never
tried** — it is the single highest-value next experiment. The planner, which succeeded,
keeps its nose on the goal via `avoidance.py`'s `heading_cost` rather than on its
direction of travel; aiming the servo at the goal bearing is the candidate fix.

## Finding 3 — walls and cabinets are invisible, and read as the *emptiest* direction

The camera sees them as pixels; nothing converts those pixels into an obstacle. There are
exactly two producers: `PersonDetector`, whose `DYNAMIC_CLASSES = ("person",)`, and
`ColourBlobDetector`, whose `PROFILES = {"bin": BLUE_BIN}`. A grey cubicle panel fires
neither, and VOC has no wall class to widen to.

Even if one did fire, mono RGB cannot range a wall: `SizePrior` is *"the only reason a mono
camera can range it"*, and a wall has no characteristic size.

The compounding problem is that unseen bearings report `max_range_m` — **clear**, not
unknown. So the bin is the only object in the robot's world model, and every direction that
is not the bin reads as open floor. An avoidance policy escaping the only modelled obstacle
therefore steers, by construction, into unmodelled space. In a corridor that is a wall.

Run 0 survived this only because the planner never rotates, so its footprint stayed square
to the corridor. It is equally blind.

The Go2's L1 LiDAR would see this geometry and is not wired into `visual_nav`. **The
operator has ruled LiDAR out**, so this stands as an accepted limitation rather than a work
item. Note also that feeding walls to the *policy* is not a plumbing job:
`stationary_objects` are discs and a wall is a plane.

## Finding 4 — the loop runs at a third of its design rate

2.78–3.58 Hz against a nominal 10 Hz, and perception 4.1–5.9 Hz. **16–33% of ticks came
back perception-stale** against a 0.6 s timeout, with cycle latencies of 255–598 ms. Every
stale tick commands zero. The policy path roughly halves the rate versus the planner path
(5.1 Hz on run 0), so the policy's own cost is implicated but not isolated. Everything in
the stack is tuned for 10 Hz; Finding 1 is one bug this produced, and there may be others.

## Finding 5 — deflection attributed to an obstacle outside the sensing horizon

`replay_mappo.py` reports **13.8° mean and 67.7° max** of obstacle-caused deflection on the
43 run-5 ticks where the bin's surface was **beyond** the 0.875 m horizon. It should be
zero: outside the horizon the ranges clip and the observation should be identical to the
ablated control. First suspicion is `static_obstacle_ttl_s: 120.0`, which has the policy
remembering every obstacle for two minutes and possibly ray-casting against drifted ghosts
of it. Unresolved.

## What was never testable — moving objects and humans

Neither reaches the policy, so neither is a MAPPO test:

- `mappo_bridge.stationary_objects()` filters to `kind == "static"`, which
  `visual_nav._obstacles()` stamps only on colour-mapped landmarks. **People are
  `kind="tracked"` and are dropped.**
- A person reaches the policy as a single boolean, `external_hold`, which maps to *stop* —
  and only once the planner has already given up. Otherwise the policy drives straight at
  them and the **planner's veto** does the steering.
- A moving non-person object is not detected at all: the detector's dynamic class list is
  exactly `("person",)`. A moving *blue* object would map as a **static** landmark, which
  is the failure mode already on record.

So human avoidance in this system is 100% incumbent planner and 0% MAPPO. Worth stating
plainly before any footage is shown.

Incidentally, the G1 humanoid standing in staging B was **never** detected as a person
across 128 ticks — a useful negative result for the peer-invisibility gap in issue #6.

## Reproducing

```bash
# scene check, moves nothing — read the geometry before staging anything
python3 mappo_drive.py --telemetry /tmp/scene.jsonl --calibration ~/go2_front_camera.json \
    --static-prop bin --goal-class chair --goal-height 1.067 \
    --confidence 0.45 --robot-radius 0.25 --no-latch-arm --arrive 0.8 --max-seconds 15

# the run 5 configuration, with the arrival budget it needed
python3 mappo_drive.py --live --telemetry ~/run.jsonl --record ~/run.mp4 \
    --policy-config <config with max_vy_mps 0.10> \
    --calibration ~/go2_front_camera.json \
    --static-prop bin --goal-class chair --goal-height 1.067 \
    --confidence 0.45 --robot-radius 0.25 --no-latch-arm \
    --arrive 0.8 --max-seconds 25 --policy-mode supervised

# after any run, the paired ablated control is what separates avoidance from wandering
python3 replay_mappo.py ~/run.jsonl --config <same config>
```

**Stage from the dry pass, not by eye.** On staging A the first layout put the bin 0.64 m
from the chair, so `--arrive 0.8` fired 6 cm past the bin's front face and the robot would
have stopped *at* the obstacle without ever passing it. Moving the chair back to 1.76 m of
bin-to-goal separation is what made an avoidance manoeuvre possible at all. On staging B
the bin started 1.14 m away against a 0.875 m horizon — just outside it — which produced a
**98% veto rate** in the dry pass, i.e. a planner demo wearing a policy's name.

## Operator notes

- The Jetson's RTC read **January 1970** all day, so `wall_time` in these files is
  meaningless. The policy is immune — it uses `time.monotonic()`.
- `~/robotics-connect-go2` is `install.sh`'s **default** venv and is what the runbook
  correctly documents. This particular robot was installed with `--env-dir` pointed at a
  pre-existing `~/robotics-connect-envs/armwaheed`, so the documented `source` line fails
  *here* and nowhere else. Read `~/.mappo-go2-deploy.manifest` rather than assuming either.
  (An earlier revision of this file called the runbook wrong. It was not; I inferred a
  documentation bug from one non-default deployment without reading the installer.)
- The D1 arm was checked after the third stall and exonerated: sway **1.8° of 3.0°
  allowed**, 0.6 mm off the dorsal centreline, `blocking: None`. Joint temperatures peaked
  at 39 °C. Neither was ever the cause.
- A full-tunnel VPN captures the `192.168.123.0/24` route and makes the robot unreachable.
