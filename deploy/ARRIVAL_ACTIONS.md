<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# What the robot does when it reaches the goal — the runbook

Pick one on the dashboard, per run, from **On arrival**. Three choices:

| choice | what it is | who owns it once it starts |
| --- | --- | --- |
| 🕺 **360° dance** (`spin`) | a full turn at the measured 0.8565 rad/s yaw primitive | **this stack** — shapeable, slowable, interruptible |
| 👋 **hello** (`hello`) | vendor `打招呼` `0x21010507`, the front-leg wave | the firmware |
| ⛔⛔ **carpet backflip** (`carpet-backflip`) | vendor `0x2101050C`, travels ~1.5 m BACKWARD | the firmware |

`spin` is the default and is what has always happened on arrival. It is flourish's own
**ARRIVAL_KIND**: a gesture that keeps the robot's centre where it is, which is the
property that makes it safe at the end of a run nobody is steering. `test_flourish.py`
proves it returns the robot to where it started.

> ⛔ **The other two are VENDOR CANNED ACTIONS.** One opcode, twelve bytes, executed by the
> firmware. Nothing in this stack can shape, slow, shorten, steer or interrupt what happens
> next; the only decision this repository makes is whether to put the bytes on the wire.
> The carpet backflip travels ~1.5 m into the one direction the platform cannot sense.

## Where the opcodes come from

`robot-stack/deep_robotics/lite3/locomotion/flourish.py` takes them from the vendor
reference GUI `Lite3_All_control_v20153.py:310-320`, where each is a button. That file was
cross-checked against four opcodes this repository already knew independently — `后空翻`
`0x21010502`, `扭身跳` `0x2101020D`, `向前跳` `0x2101050B` and the end-action `0x21010C0B` —
and it does *not* contain `carpet-backflip` `0x2101050C`, which matches our own finding that
that one appears in no vendor GUI we hold. Four matches and one expected absence is why
`0x21010507` is treated as evidence rather than a guess.

**Never guess a neighbouring opcode.** `0x2101050C` sits one past `0x2101050B`, which is
`向前跳` / jump forward — a guessed neighbour fires the robot into space nobody cleared.

Three more exist in that table and are deliberately **not** wired up: `扭身体` twist-body
`0x21010204`, `翻身` roll-over `0x21010205`, `太空步` moonwalk `0x2101030C`.

## What has to be true, and what says no when it is not

| gate | refusal you will see | what fixes it |
| --- | --- | --- |
| arm motion | the two vendor options are greyed in the dropdown | tick **arm motion**; they are fired by the flourish, which needs `--live` |
| `--operator-triggered` | `--kind … needs --operator-triggered. … Pass it only if you are standing there, looking at the robot, with the emergency stop in your hand.` | choosing a vendor action **is** what passes it |
| `--live` + `--operator-ready` | `--live needs --operator-ready` | `mission.py` pairs them automatically |
| the room's measurements | `MAPPO_ARRIVAL_ACTION=… but MAPPO_FLIP_… is not; the run will NOT do it` | set the three variables below |
| rear clearance | `THIS ROBOT HAS NO REAR SENSING AT ALL` | `MAPPO_FLIP_REAR_CLEARANCE_M`, tape-measured |
| lane width | `check_room` refuses | `MAPPO_FLOURISH_LANE_WIDTH` |
| battery | refuses below the floor | charge, or lower `MAPPO_FLIP_BATTERY_FLOOR_PCT` |

**`--operator-triggered` is the one that surprises people.** Nothing automatic used to pass
it — that was its purpose. What passes it now is your per-run choice of a vendor action in
the dropdown. That means *"a human chose this before this run started"*, which is weaker
than what the flag used to promise. **Choosing one is a safety act, not a preference.**

Each action is held to its **own** floor by `flourish.check_rear`, so a kind that needs less
is not thereby less checked:

- `carpet-backflip` — **1.5 m**, the operator's figure for its backward travel.
- `hello` — **0.90 m**, this platform's own footprint, because its travel has never been
  measured and no number here may be invented to stand in for it. Measure it and this can
  come down.

## Arming it on a robot

The choice comes from the dashboard, per run. The room's **measurements** live in the
robot's profile, because the repository must not guess at them:

```jsonc
// /home/user/mappo-lite3-stage/dc/run-profile.json  →  "env"
"MAPPO_FLIP_REAR_CLEARANCE_M":    "1.5",   // TAPE MEASURED. No default, ever.
"MAPPO_FLIP_BATTERY_FLOOR_PCT":   "21",    // mobility floor is 20 (--battery-abort)
"MAPPO_FLIP_HOLD_SECONDS":        "7.0"    // measured backflip recovery is 5.4-5.9 s
```

`MAPPO_ARRIVAL_ACTION` is sent by the dashboard on **every** run and is authoritative. It is
never set in a profile.

> 🛑 **`MAPPO_FLIP` and `MAPPO_FLIP_KIND` ARE RETIRED AND ARE NOT READ.** A profile still
> carrying `MAPPO_FLIP=1` gets a printed line saying it is being ignored, and the run spins.
> They are refused rather than translated on purpose: a stale variable in a profile written
> weeks ago must not decide what a robot does at the goal.

> 🛑 **Editing the profile does nothing until the driver restarts.**
> `run_control.load_profile` is called once at startup, so an edit without
> `sudo systemctl restart mappo-dc-driver` changes nothing and the next run looks
> byte-identical to the last. This wastes a demo slot every single time.

Deploy code with `deploy/push-to-robot.sh` and **never `scp` into a release tree** — the
`tree_stamp` guard refuses the run. The stamp is computed from your **working tree**, so it
must be clean or verification fails.

## The rear clearance is not the measurement you think it is

The backflip travels **away from the goal**. The robot parks *facing* the marker, so the
space it needs is behind its tail — back down the lane it approached through.

Measuring 1.5 m *in front of the ArUco* is the wrong region, and it is the natural mistake:

```
clear floor in front of marker        1.500 m
robot parks at --arrive                0.550 m
half the body length (0.610 vendor)    0.305 m
                                     ─────────
verified clear floor behind its tail   0.645 m
carpet backflip needs                  1.500 m   ← short by 0.855 m
```

Measured **from the marker**, the lane needs ≈ **2.4 m** clear in one straight line.
`check_rear` compares against **the number you state** and has nothing to verify it with.

## What is still unmeasured

- **Whether ~1.5 m is this robot.** It is the vendor's figure, not a measurement of these
  units on this firmware, payload and floor. The pose channel **freezes for the whole
  manoeuvre** (both Ventures, 2026-09-07: one distinct `pos_world` value across an entire
  flip), so travel cannot be measured from telemetry — it needs a tape and a floor mark.
- **How far `hello` moves, if at all.** A greeting *should* lift a front leg and put it
  back. Nothing here has watched it, so it is treated as one that travels.
- **Body length.** `--body-length 0.610` is vendor spec, never tape-measured.
- **What battery any of these draws.**

## The rear look, and why it is off

`look_behind.py` turns the robot 180°, looks with the run's own camera, turns back, and only
then fires — earning `--operator-triggered` with *evidence* instead of a choice. **It is
disabled**, on the operator's instruction on 2026-09-07, because it turned the robot ~720°
on arrival: its own two 180° turns plus the spin. Its detector finds VOC classes, so it
never covered walls, steps or stage edges anyway. `look_behind.py` is intact and documents
what to restore.

## When nothing happens

The reason is always in the driver journal, in one sentence:

```bash
sudo journalctl -u mappo-dc-driver --since "20 min ago" --no-pager \
  | grep -E "\[mission\]|\[flourish\]|\[venue-run\]"
```

`[mission] the … did not fire (exit 1). The arrival still stands.` means a gate refused —
the `[flourish] REFUSED:` line above it names which. An arrival is never rewritten by a
failed gesture.
