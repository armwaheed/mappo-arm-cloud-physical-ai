<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running the MAPPO policy on the Go2 — the runbook

Install, check, and then three rungs in order: simulate, shadow, drive. Each one is a
precondition for the next, and the reason is written next to it rather than assumed.

**[`../robot-stack/SAFETY.md`](../robot-stack/SAFETY.md) governs anything that moves a
leg and is not optional.** `--live` is still the only flag that moves the robot, an
operator stays on the remote, and the Ethernet tether's slack is checked before anything
that turns.

## Install

```bash
git clone <this repo> && cd mappo-arm-cloud-physical-ai
./deploy/install.sh --verify                 # on the Go2's Jetson
./deploy/install.sh --policy-only            # on a workstation
```

It calls the control stack's own installer rather than reimplementing it, adds the policy
package, checks the checkpoint's SHA-256, runs one inference, and runs all 133 offline
tests. `--verify` then probes the robot's DDS read-only — no motion. Every path it creates
goes in `~/.mappo-go2-deploy.manifest`.

```bash
./deploy/uninstall.sh                        # say what would go, remove nothing
./deploy/uninstall.sh --yes
```

It removes **only** paths the manifest records as created by an install run — a venv or an
SDK clone that was already there is left alone — and it never touches the repository, the
telemetry or the recordings.

## ⚠️ Four numbers to know before the robot moves

| | |
| --- | --- |
| **The robot delivers ~0.45 of the velocity it is commanded.** | Fitted against the pose over the recorded run: 2.09 m travelled against 4.32 m commanded. Tethered, with the D1 arm, on the derated envelope. |
| **So `command_scale` is shipped at 0.6, not the delivered 0.3.** | At 0.3 the top speed on the floor is 0.047 m/s — 2.8 m in the entire 60 s run budget, so the robot cannot cross the arena and the simulation read that as a navigation failure. 0.6 is 0.095 m/s and 5.7 m. |
| **`--robot-radius 0.25` is load-bearing.** | The vendored default is 0.40 m. The policy's `meters_per_vmas_unit` of 2.5 is calibrated as 0.25 ÷ the trained 0.10 VMAS radius. `mappo_drive` now **refuses to start** on a mismatch rather than running at a horizon the evidence does not cover — see below. |
| **The policy commands no yaw.** | It is a holonomic 2-D force. Without the heading servo the robot crabs and its 85° forward camera never looks anywhere new. The servo is on by default; `--no-heading-servo` turns it off. |

## Rung 1 — clear the gate, off the robot

Issue #5: nothing drives a leg until the policy has been closed-loop tested. This needs no
hardware and takes about a minute.

```bash
cd integration
python3 closed_loop_sim.py --seeds 30 --scale 1.5 2.5 --command-scale 0.3 0.6 1.0
```

### What it says today

30 seeded scenarios, each paired with its own ablated control, collisions judged at the
0.25 m planning radius. The `planner` row is the incumbent — the controller that walks
today — on the identical scenarios.

| scale | `command_scale` | controller | arrived | collided | worst clearance |
| --- | --- | --- | --- | --- | --- |
| — | — | **planner** (incumbent) | 14/30 | 2 | −0.01 m |
| 1.5 | 0.6 | policy, no veto | 9/30 | **21** | −0.02 m |
| 1.5 | 0.6 | supervised | 11/30 | **0** | +0.01 m |
| 2.5 | 0.6 | policy, no veto | 21/30 | **9** | −0.01 m |
| 2.5 | 0.6 | **supervised** | **18/30** | **0** | +0.00 m |
| 2.5 | 1.0 | supervised | 21/30 | 1 | −0.01 m |

Ablated, every controller reaches the goal 30/30 at `command_scale` 0.6 — so the arrival
rates above are about the obstacle, not about the policy failing to find the goal.

**Read: supervised at scale 2.5 and `command_scale` 0.6.** It is the only configuration
with no collisions that also beats the incumbent on arrivals, and both of those matter.
The raw policy is not a candidate — it collided in every configuration, and worst at the
scale Sagar's package shipped with.

### ⚠️ Near the bin, the planner is driving more often than the policy

**The veto takes over on 61% of the ticks that have an obstacle in the scene.** That is
not the veto being over-strict — shortening its horizon was tried and it costs collisions
without even reducing the rate:

| veto horizon | arrived | collided | vetoed |
| --- | --- | --- | --- |
| 0.3 s | 11/30 | 2 | 60% |
| 0.5 s | 16/30 | 2 | 47% |
| 1.0 s | 16/30 | 1 | 51% |
| **2.5 s (the planner's own, and the default)** | **18/30** | **0** | 61% |

It is the honest measurement: **near the obstacle, this checkpoint's proposal is
infeasible more often than not.** So a supervised run is a genuine policy demo in open
floor and substantially the incumbent planner's manoeuvre at the moment of avoidance.
Know that before the footage is shown to anyone. `mappo_drive.py` prints the counts at
the end of every run — `N/M ticks driven by the policy, K vetoed` — so it can be checked
on the day rather than assumed.

Three more honest caveats:

- **60% is not a good arrival rate.** The other 12 are timeouts, not crashes. The
  incumbent's own rate on the same scenarios is 47%, so this is an improvement on a low
  bar rather than a solved problem.
- **"Collision" is the 0.25 m planning disc overlapping**, and the Go2's real half-width
  is about 0.155 m. The −0.01 m entries above still had roughly 9 cm of real air.
- **The simulation is a planar velocity-commanded body.** It cannot tell you the robot
  will not fall over. Clearing this gate is a licence to try on hardware, not a substitute.

## Rung 2 — shadow, on the robot, legs untouched

The planner drives exactly as it does today. A second process reads the telemetry the run
is writing and records what the policy *would* have done.

```bash
# terminal 1, on the robot
source ~/robotics-connect-go2/bin/activate
cd robot-stack/unitree/go2/visual_nav
python3 visual_nav.py --live --telemetry ~/run.jsonl --record ~/run.mp4 \
    --static-prop bin --goal-class chair --goal-height 1.067 \
    --robot-radius 0.25 --no-latch-arm

# terminal 2
cd integration
python3 mappo_shadow.py ~/run.jsonl --follow --out ~/shadow.jsonl
```

`mappo_shadow.py` holds no locomotion client and opens no DDS channel. It reads a file.
That is the whole safety argument and it is why this rung exists: it puts the policy in
front of real perception, at the real rate, with the real latency, in the actual room,
before anything is riding on it.

**Look for** `STOP_CLOCK_ERROR` or `STOP_STALE_INPUT` in the status tally (either means
the plumbing is wrong, not the policy), the bridge-gap lines at the end, and whether the
policy's intent tracks the goal bearing in open floor. It cannot tell you about avoidance
— the deflection it reports includes this checkpoint's own 6–16° heading bias, and only
`replay_mappo.py`'s paired ablated control separates the two.

## Rung 3 — the policy drives

Only after rungs 1 and 2, with a cleared lane and an operator on the remote.

```bash
cd integration
python3 mappo_drive.py --live --telemetry ~/drive.jsonl --record ~/drive.mp4 \
    --static-prop bin --goal-class chair --goal-height 1.067 \
    --robot-radius 0.25 --no-latch-arm --policy-mode supervised
```

`--policy-scale` and `--policy-command-scale` override `policy/config.json`, which already
carries the 2.5 and the 0.6 above. Pass them to try something else, not to restate the
default — a runbook that repeats a config value is a second place for it to drift.

Every `visual_nav.py` flag still applies — this substitutes the choice of velocity and
nothing else. The camera, detector, tracker, odom map, health monitor, arm gate, telemetry
and teardown are all the shipped stack's, reached through `main(planner_factory=...)`,
which the control stack grew for this and which was merged and re-vendored rather than
patched around.

**The veto**: the policy proposes, and the planner's own rollout must agree the proposed
velocity keeps every obstacle's hard gap over 2.5 s. If it does not, the planner's command
goes out instead. At the end of the run the process prints how many ticks the policy
actually drove, how many were vetoed and how many were stopped. **Expect roughly 60% of
the obstacle-present ticks to be vetoed** — that is what simulation measured, and it means
the manoeuvre around the bin is substantially the planner's. If it comes back much higher
than that, the policy was not driving at all and the footage does not show what it appears
to.

`--policy-mode raw` removes the veto. In simulation the raw policy collided in every
configuration tested. Empty arena only, if at all.

### Start smaller than you want to

1. `--max-seconds 20` and no obstacle at all. Does it walk to the goal in a straight line?
2. Add the bin, well off the lane. Does it deviate at all?
3. Bin on the lane. This is the demo.

Between each, read the tail of the drive log: `N/M ticks driven by the policy`.

## Troubleshooting

| symptom | cause |
| --- | --- |
| every tick `STOP_CLOCK_ERROR` | something is passing a wall clock as `timestamp_s`. The runners pass `time.monotonic()`; a hand-rolled loop is the usual culprit. |
| every tick `STOP_EXTERNAL_HOLD` | a mover is in the lane, or the tracker has a ghost. Check the overlay — a coasting track inflates its radius and blocks from metres away. |
| the robot barely moves | `command_scale`, or `--derate`. At the delivered 0.3 the top speed was 0.047 m/s; the shipped value is 0.6. |
| `veto` on nearly every tick | the policy and the planner disagree about the whole route. Drop to rung 2 and look at the shadow log before turning the veto off. |
| the robot crabs sideways and never turns | `--no-heading-servo` was passed, or the servo's deadband is swallowing the error. |
| `cannot import the policy package` | run from `integration/`, or pass `--package ../policy`. |
| a policy config is refused at load | it disagrees with the checkpoint's own recorded training constants. The message names the field; do not "fix" it by editing the checkpoint. |
| `REFUSING TO RUN — the policy's scale was calibrated for a different robot size` | `--robot-radius` and `meters_per_vmas_unit` disagree. Pass `--robot-radius 0.25`, or pass `--policy-scale` to match the radius you meant and **re-run the simulation** — the numbers above do not transfer. Nothing moved; the robot is still prone. Every refusal is also appended to `~/.mappo-go2-refusals.jsonl` (`--refusal-log`), because a refused run writes no telemetry and would otherwise leave no trace of why the demo did not start. |
| RPC segfault on any DDS call | `setup_env.sh` was not sourced. `LD_LIBRARY_PATH` is load-bearing — see `robot-stack/unitree/go2/install/setup_env.sh`. |

## What is still open

- **Peers are invisible.** Another quadruped is neither a detector class nor a colour
  profile, so a *multi*-agent demo with two robots in one arena has no way for either to
  see the other. Issue #6.
- **The steering response is a cliff, not a ramp** — about 103° once the obstacle is
  inside the horizon and essentially nothing outside it, at every scale from 1.5 to 4.0.
  Only a retrain with a larger `lidar_range` changes that. Issue #4.
- **Episodes were 100 steps in training**, ten seconds at the stack's 10 Hz. A 60 s demo
  run is six times longer than anything the policy saw.
