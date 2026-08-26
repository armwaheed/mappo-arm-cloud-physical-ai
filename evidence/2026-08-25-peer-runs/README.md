<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-25 — the Go2 Walk cleared a peer robot, and not because it avoided it

Two live runs, same policy, same corridor, same Go2 Wheel parked as the peer. One
cleared it and reached the goal. The other drove into it and pushed it the length of
the corridor. **The variable that changed was where the goal sat relative to the peer,
not anything in the stack.**

This directory is the record of that, because the first run is the most demo-ready
footage we have and it would be easy — and wrong — to caption it *"MAPPO avoids a peer
robot."*

Reproduce every number below with no robot and no dependencies:

```bash
python3 bearings.py
```

![the hero run, six frames](hero-contact-sheet.jpg)

*Top row: the peer boxed at 1.3 m, ranged by height, labelled `horse`. It leaves frame
right as the robot tracks past. Bottom row: the goal chair, arrived at 0.4 m.*

## The two runs

| file | what it shows |
| --- | --- |
| [`hero-clears-peer-on-right.mp4`](hero-clears-peer-on-right.mp4) | Go2 Wheel off the nose to the right. The Go2 Walk tracks past it on the left and finishes nose-on to the goal chair. No contact. 14.1 s. |
| [`contrast-peer-on-the-goal-bearing-collision.mp4`](contrast-peer-on-the-goal-bearing-collision.mp4) | Same policy, same corridor. The peer is on the goal bearing this time. The robot drives through it. |

Both are `--policy-mode raw`. Both are re-timed to real wall-clock pace: the recorder
wrote ~4.1 annotated frames per second and the robot-side files play them at 7 fps,
which compresses the timeline by 1.7x. Robot-side names were `live03-swerve.mp4` and
`live05-left.mp4`; the telemetry headers still carry them.

## Why the first one cleared, measured

| | hero | contrast |
| --- | --- | --- |
| goal bearing (body frame, + is left) | **+11.0 .. +21.4 deg** | −63.9 .. +5.3 deg, mean −18.9 |
| peer bearing while closing | **−19 .. −46 deg** (right) | on the goal bearing |
| commanded lateral | **leftward on 57 of 57 ticks** | leftward on 40 of 89 |
| integrated | 1.98 m forward, **+0.74 m** lateral commanded | 2.74 m forward, −0.28 m |
| lateral actually delivered | **+0.32 m — 43% of what was asked** | +0.01 m |
| corr(lateral cmd, goal distance remaining) | **+0.951** | +0.944 |
| corr(lateral cmd, peer range) | **+0.048** | — |

Three things falsify the avoidance reading, and all three are in `bearings.py`:

1. **The leftward command predates the peer.** It runs +0.154 to +0.166 m/s before the
   peer is tracked at all, and +0.147 to +0.153 m/s while it is tracked. The peer's
   arrival did not increase it. It slightly *decreased* it.
2. **It tracks the goal, not the obstacle.** r = +0.95 against distance still to run;
   r = +0.05 against the peer's range over the eight ticks the peer was held.
3. **The policy was barely given a navigable observation.** `render_observation
   --summary` scores this run at **5 of 59** driven ticks with an open window toward the
   goal. There was nothing to steer with for 54 of them.

So the clearance is a by-product of the goal's bearing. Put the peer on that bearing and
the same policy runs it over — which is the contrast video, and is the only reason it is
kept here.

**Caption the hero run *"clears an off-axis peer"*.** It is a real, clean, on-hardware
pass of another robot and it is worth showing. It is not an avoidance manoeuvre and the
contrast video is the first question anyone will ask.

## Reading the track table

`bearings.py` prints tracks raw and unfiltered, which is the honest thing and needs a
warning:

* **Labels are noise.** The peer is `horse` here, and `motorbike` 613 / `chair` 372 /
  `aeroplane` 200 / `person` 109 times across the 2026-08-24 corpus. That finding is why
  `mappo_bridge.holds_the_robot` routes on box aspect and not on the label (PR #73).
* **The tracker fragments the peer** across short-lived ids as its label flips —
  `_associate` gates on label equality — so one physical robot is several tracks.
* **`track-7` closing to 0.02 m in the hero run is the goal chair being arrived at**,
  not a collision. Near-field tracks whose bearing swings through 180 degrees are junk.

The peer in the hero run is `track-3`, identified from the frame itself and not from
its label.

## What is not here

**No avoidance manoeuvre was demonstrated on hardware on this date.** The supervised
run (`11-live-supervised` in the working set) stops and holds for 32 s rather than
close on the peer — the veto working as designed, but a stop is not a manoeuvre. The
geometric reason a manoeuvre was unavailable at these ranges is
[issue #72](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/72):
below ~0.52 m a peer fills the 85.3 deg field of view and no open bearing exists.

## Numbers established on this date

* **Detector recall against range**, over the 1,903-frame labelled corpus: 0 of 315
  beyond 2.7 m, 80% at 1.5–1.9 m, 91% inside 1.1 m. A cliff, not a slope — and
  scene-dependent enough that five placements inside the "good" band returned 0%.
* **The lateral gait floor is not a floor on a diagonal.** Delivery is proportional at
  roughly 40% from 0.05 m/s upward; the hero run measures 43%. The 0.20 m/s figure in
  `Limits` was measured as a pure strafe from standstill and does not describe a
  diagonal command.
* **Standing clip boundary 0.72 m.** Below it the peer's ground contact leaves the
  frame, ranging falls back to the width prior, and `person_shaped` goes True.
* **Blind radius 0.52 m** = `0.35 / sin(42.6 deg)`. Issue #72.

## A metric that lied, kept on the record

An early reading of the hero run reported 0.067 m of lateral travel and called it
"essentially straight". That is deviation from the start-to-end chord, which is blind to
a swerve held in one direction — the chord rotates instead of bowing. The same run
commands 0.74 m and delivers 0.32 m when integrated tick by tick in the body frame.
**Integrate in the body frame. Do not measure a swerve against its own chord.**
