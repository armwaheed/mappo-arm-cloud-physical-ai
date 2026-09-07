<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Flip on arrival, on a Lite3 Venture — the runbook

The robot drives to its goal and fires a vendor backflip when it gets there. First fired
successfully on **2026-09-07**, on both Ventures, from the dashboard.

> ⛔ **It travels ~1.5 m BACKWARD, into the only direction this platform cannot sense.**
> No rear camera, no ultrasonic, no bumper. The person holding the emergency stop is
> usually standing exactly there. Read [The rear clearance is not the measurement you
> think it is](#the-rear-clearance-is-not-the-measurement-you-think-it-is) before arming
> it the first time.

## The short version

```
dashboard → tick "arm motion" → tick "flip on arrival" → Start run
```

That is it, **once the robot's profile carries the four measurements below**. The checkbox
ships unticked, is never auto-ticked, and is disabled unless arm motion is on.

## What has to be true, and what says no when it is not

Five flags reach `flourish.py`, and it refuses on each one separately. Every refusal below
was hit for real getting this working — they are listed in the order you will meet them.

| gate | refusal you will see | what fixes it |
| --- | --- | --- |
| `--flip-on-arrival` + all four `--flip-*` | `mission.py: error: --flip-on-arrival needs …` | set the four `MAPPO_FLIP_*` variables (below) |
| `--flourish` | `--flip-on-arrival needs --flourish` | `MAPPO_FLOURISH=1` and its four settings |
| `--operator-triggered` | `--kind backflip needs --operator-triggered. … Pass it only if you are standing there, looking at the robot, with the emergency stop in your hand.` | tick the **flip** checkbox — that tick is what passes it |
| `--live` + `--operator-ready` | `--live needs --operator-ready` | tick **arm motion**; `mission.py` pairs them automatically |
| `--rear-clearance-metres` ≥ 1.5 | `state --rear-clearance-metres. THIS ROBOT HAS NO REAR SENSING AT ALL` | `MAPPO_FLIP_REAR_CLEARANCE_M`, tape-measured |
| `--lane-width-metres` | `check_room` refuses | `MAPPO_FLOURISH_LANE_WIDTH` |
| battery | refuses below the floor | charge, or lower `MAPPO_FLIP_BATTERY_FLOOR_PCT` |

**`--operator-triggered` is the one that surprises people.** Nothing automatic used to pass
it — that was its entire purpose. What passes it now is the per-run checkbox, and that is
the whole licence: it means *"a human ticked a box before this run started"*, which is
weaker than what the flag used to promise. **The tick is a safety act, not a
convenience.**

## Arming it on a robot

The switch comes from the dashboard. The four **measurements** live in the robot's profile,
next to the flourish's own, because they describe the room and the repository must not
guess at them:

```jsonc
// /home/user/mappo-lite3-stage/dc/run-profile.json  →  "env"
"MAPPO_FLIP":                     "0",          // the standing value; the checkbox overrides per run
"MAPPO_FLIP_KIND":                "backflip",   // or "carpet-backflip"; both travel backward
"MAPPO_FLIP_REAR_CLEARANCE_M":    "1.5",        // TAPE MEASURED. No default, ever.
"MAPPO_FLIP_BATTERY_FLOOR_PCT":   "21",         // mobility floor is 20 (--battery-abort)
"MAPPO_FLIP_HOLD_SECONDS":        "7.0"         // measured recovery is 5.4-5.9 s
```

`MAPPO_FLIP=0` is sent **explicitly** when the box is clear, so a profile carrying
`MAPPO_FLIP=1` cannot license a flip nobody asked for on the day.

> 🛑 **Editing the profile does nothing until the driver restarts.**
> `run_control.load_profile` is called once at startup (`robot_driver.py`), so an edit
> without `sudo systemctl restart mappo-dc-driver` changes nothing and the next run looks
> byte-identical to the last. This wastes a demo slot every single time.

Deploy code with `deploy/push-to-robot.sh` and **never `scp` into a release tree** — the
`tree_stamp` guard refuses the run. The stamp is computed from your **working tree**, so it
must be clean or verification fails.

## The rear clearance is not the measurement you think it is

The flip travels **away from the goal**. The robot parks *facing* the marker, so the space
it needs is behind its tail — back down the lane it approached through.

Measuring 1.5 m *in front of the ArUco* is the wrong region, and it is the natural mistake:

```
clear floor in front of marker        1.500 m
robot parks at --arrive                0.550 m
half the body length (0.610 vendor)    0.305 m
                                     ─────────
verified clear floor behind its tail   0.645 m
backflip needs                         1.500 m   ← short by 0.855 m
```

Measured **from the marker**, the lane needs ≈ **2.4 m** clear in one straight line:
`0.55 (park) + 0.305 (half body) + 1.5 (travel)`.

`check_rear` compares against **the number you state** and has nothing to verify it with.
Stating 1.5 on the strength of a 1.5 m measurement in front of the marker passes the gate
while leaving the robot ~0.85 m short of cleared floor.

## What is still unmeasured

- **Whether ~1.5 m is this robot.** It is the vendor's figure for a Lite3, not a
  measurement of these units, on this firmware, with this payload, on this floor. The pose
  channel **freezes for the whole manoeuvre** (both Ventures, 2026-09-07: one distinct
  `pos_world` value across an entire flip), so the travel cannot be measured from
  telemetry. It needs a tape and a floor mark.
- **Body length.** `--body-length 0.610` is vendor spec, never tape-measured.
- **What battery a flip actually draws.** The floor is an operator's number.

## The rear look, and why it is off

`look_behind.py` turns the robot 180°, looks with the run's own camera and detector, turns
back, and only then flips — earning `--operator-triggered` with *evidence* instead of a
tick. **It is disabled**, on the operator's instruction on 2026-09-07, because it turned the
robot ~720° on arrival: its own two 180° turns plus the pre-existing 360° victory spin
(`play_flourish(command, "spin")`), which fires on every arrival and is invisible in both
telemetry and video because it runs after the drive process exits.

Its detector finds VOC classes, so it never covered walls, steps or stage edges anyway —
the tape number always did that. `look_behind.py` is intact and documents what to restore.

## Evidence of the first successful firing

```
[mission] ARRIVED on attempt 1 after 11s
[mission] backflip
  sends ONE opcode, 0x21010502, as 12 bytes to 127.0.0.1:43893,
  then holds 7.0s while the firmware runs it, and sends NOTHING else.
… returned to force-control 6 unaided
exit_code = 0, elapsed_s = 21.61
```

No end-action opcode is needed: the vendor's own hand controller was captured performing
all three of these and sent none, and the robot returned to force-control 6 unaided each
time.

## When it does not flip

The reason is always in the driver journal, in one sentence:

```bash
sudo journalctl -u mappo-dc-driver --since "20 min ago" --no-pager \
  | grep -E "\[mission\]|\[flourish\]"
```

`[mission] the backflip did not fire (exit 1). The arrival still stands.` means a gate
refused — the `[flourish] REFUSED:` line immediately above it names which. An arrival is
never rewritten by a failed flip.
