<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running the MAPPO policy on the Go2 — the runbook

Install, check, and then three rungs in order: simulate, shadow, drive. Each one is a
precondition for the next, and the reason is written next to it rather than assumed.

This installer and the measured numbers below are Go2-specific. For the Lite3 Venture,
use the same simulation/shadow/drive ladder with the platform commissioning runbook in
[`../robot-stack/deep_robotics/lite3/README.md`](../robot-stack/deep_robotics/lite3/README.md);
do not run the Unitree SDK installer on a Lite3 host.

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
package, checks the checkpoint's SHA-256, runs one inference, and runs all 139 policy +
integration tests. `--verify` then probes the robot's DDS read-only — no motion. Every
path it creates goes in `~/.mappo-go2-deploy.manifest`.

```bash
./deploy/uninstall.sh                        # say what would go, remove nothing
./deploy/uninstall.sh --yes
```

It removes **only** paths the manifest records as created by an install run — a venv or an
SDK clone that was already there is left alone — and it never touches the repository, the
telemetry or the recordings.

## 🛑 Python on a robot runs in a virtualenv — and the robot now refuses if it does not

A live run that reaches real hardware from the **system** Python raises `SystemExit` before
it opens a transport. It is a refusal, not a warning, and the message names the venv to
activate and the exact command to build one. The check lives in
[`../robot-stack/preflight/venv_guard.py`](../robot-stack/preflight/venv_guard.py) and is
wired into `visual_nav --live` on the Lite3 binding and into `drive_bridge.py` for the
`go2` and `lite3` platforms. Run it by hand to see this machine's verdict:

```bash
python3 robot-stack/preflight/venv_guard.py
```

### The reason is package isolation, not version isolation — and the difference matters

The tempting argument, *"use the venv so you get the right Python"*, is **false here**, and
a rule with a false premise gets discarded by the first person who checks it. Measured on
the lab Go2 (Orin NX, Ubuntu 20.04.5 LTS, aarch64) on 2026-08-26:

| | |
| --- | --- |
| pythons present | `/usr/bin/python3.8` (3.8.10), `/usr/bin/python3.9` (3.9.5) — **all of them** |
| `~/robotics-connect-envs/armwaheed/pyvenv.cfg` | `home = /usr/bin`, `version = 3.8.10`, `include-system-site-packages = true` |

The venv is **3.8 on a 3.8 system**. It supplies no version isolation whatsoever, and
because it inherits system site-packages it is not even a clean room. What it supplies is
*package* isolation for the one thing that matters, and that half is total:

| module | `/usr/bin/python3` (system) | `~/robotics-connect-envs/armwaheed` (venv) |
| --- | --- | --- |
| `cyclonedds` | `ModuleNotFoundError` | **0.10.2** |
| `unitree_sdk2py` | `ModuleNotFoundError` | **editable install** (egg-link) |
| `numpy` | OK | OK — inherited from the system |
| `cv2` | OK | OK — inherited from the system |

**The vendor DDS stack exists only inside the venv.** So a `pip install` aimed at the system
Python writes to an interpreter that every other user, every vendor tool and every ROS node
on that robot shares, and no `uninstall` puts a shadowed vendor package back the way it was.
A `pip install` inside the venv can, at worst, break one directory that can be deleted and
rebuilt. That asymmetry is the whole argument, and it is why the refusal exists rather than
a sentence somebody might read.

**`--system-site-packages` is not optional and it removes the last objection.** It is what
the deployed venv already uses, it is why `numpy` and `cv2` stay importable, and it means a
machine whose vendor stack genuinely *does* live in the system Python still sees it from
inside a venv. There is therefore always a correct action, which is why the guard ships
with no bypass flag: an escape hatch printed next to a refusal is an instruction to use it.

### 🛑 Why `device-connect-edge` runs OFF the robot — do not spend a demo morning on this

**`device-connect-edge` requires Python >= 3.11. This robot has 3.8.10 and 3.9.5 and no way
to get a third.** Measured on the same Go2, the same day:

```
apt-cache policy python3.11     ->  (no output at all — no such package)
apt-cache policy python3.10     ->  libpython3.10-stdlib  Candidate: (none)
```

**A virtualenv cannot supply a Python the machine does not have.** `python3 -m venv` builds
an environment *from* an interpreter; `python3.8 -m venv` produces a 3.8 environment and
there is no flag that changes that. Nothing about the venv rule above is a route to 3.11,
and reading the two rules together is the mistake this section exists to prevent.

So the split is deliberate, not a workaround waiting to be tidied up:

| half | interpreter | where it runs | why |
| --- | --- | --- | --- |
| the Device Connect driver (`dashboard/robot_driver.py`) | >= 3.11 | **off-robot** — a workstation on the same LAN | `device-connect-edge` requires it and the Jetson cannot provide it |
| `dashboard/drive_bridge.py` | 3.8, stdlib only | **on the robot**, in the SDK venv | `unitree_sdk2py` and CycloneDDS live there and nowhere else |

The driver reaches the robot by running `drive_bridge.py` as a subprocess over the SDK
env's interpreter — that is what `--bridge-python` is, and it is why getting it wrong makes
every command fail with an import error.

**It was proven end to end on the Go2 on 2026-08-26, read-only, with no motion.** From
inside the venv the bridge path returned live odometry — position `(2.797, 2.048, 0.049)`,
yaw `3.071` rad, velocity `(0, 0, 0)` — and the motion-service mode came back as

```
CheckMode -> {'form': '0', 'name': 'mcf'}
```

`mcf` is **not** a sport mode. `Go2Locomotion.connect()` only accepts `SportClient` commands
in `normal` or `ai`, so a `Move` issued in this state would have been **silently ignored** —
no fall, no fault, no error code, exactly the failure shape as the gait floor above. The
two-env bridge is what surfaced it before a demo did; a driver that could not run on the
robot at all would have surfaced nothing.

### What to do when an import fails on a robot

Activate the venv the install recorded, and re-run the same command:

```bash
grep '^env_dir' ~/.mappo-go2-deploy.manifest | tail -1     # what the install actually used
source ~/robotics-connect-envs/$USER/bin/activate          # or whatever that prints
```

If there is genuinely no venv, build one from the interpreter the vendor stack was
installed for — the one that can import it:

```bash
/usr/bin/python3.8 -m venv --system-site-packages ~/robotics-connect-envs/$USER
source ~/robotics-connect-envs/$USER/bin/activate
```

**If an import still fails inside the venv, that is a finding to report, not a dependency
to add.** Do not `pip install` into the system Python, and do not try to install a newer
Python on the robot — the table above is why.

⚠️ **The guard's detection is positive-only, so it can fail to fire and never falsely
fires.** It looks for `/etc/nv_tegra_release` or a Tegra device-tree model — both measured
on the Go2 — and treats CI as never-a-robot. **No marker has been measured on a Lite3**, so
the Lite3 SOP exports `MAPPO_ROBOT_HOST=1` and the guard fires there by declaration rather
than by inference. Every live run prints the verdict it reached, including `venv-guard: not
enforced`, so a gate that is not firing says so instead of being invisible.

### Ask the robot what commit it is running — `deploy/push-to-robot.sh`

**None of the `~/mappo-*` trees on the Go2 is a git checkout** — no `.git`, so no branch and
no `git status`. That much was always true. What this section used to say next was not:
it claimed `~/mappo-run` "corresponds to no single commit". Re-measured 2026-08-26 by
computing **git's own root tree object id from the bytes on the robot's disk**, six of the
ten trees are bit-perfect checkouts:

| tree | root tree id | is |
| --- | --- | --- |
| `~/mappo-run` | `287242e9…` | **`cb42b9a`, exactly — 226/226 files** |
| `~/mappo-probe` | `45fdebf4…` | `15f4f05`, exactly |
| `~/mappo-shape` | `9d2073f5…` | `8f3639a`, exactly |
| `~/mappo-shape2` | `0275ad90…` | `efd0b0b`, exactly |
| `~/mappo-side` | `f0cd2b97…` | `d367fd2`, exactly |
| `~/mappo-swerve` | `76e29fc6…` | `308658b`, exactly |
| `~/mappo-main` | `467c3d16…` | **no commit in this repository** |
| `~/mappo`, `~/mappo-dedup`, `~/mappo-smoke-2b15549` | | no commit |

`~/mappo-run` read as "no single commit" only because it was reconstructed against `main`,
and **`cb42b9a` is not on `main`** — it is the tip of the unmerged branch
`feat/sidestep-when-the-diagonal-cannot-execute`. The tree was never a mixture. The real
finding is worse in a different way: a run there is a run of an unmerged branch, and
nothing on the robot said so.

**`~/mappo-main` is the genuine mixture, and it is the tree the wrappers use.**
`run-smoke.sh`, `run-chair.sh` and `goal-sweep.sh` all `source
/home/unitree/mappo-main/…/setup_env.sh` and `cd /home/unitree/mappo-main/integration`.
Against `main` HEAD: 88 files current, 46 differing, 217 tracked files absent. Those 46
were last current across **21 different commits**, from 8 behind HEAD to 83 behind, and it
carries two files git has never tracked — `integration/mappo_drive.py.bak-preberth` and
`robot-stack/unitree/go2/visual_nav/visual_nav.py.orig`. Dropping those still resolves to
nothing. It is hand-maintained.

#### Deploying

```bash
ROBOT_SSHPASS=… bash deploy/push-to-robot.sh unitree@192.168.123.18 /home/unitree/mappo-run
```

Tracked files only, at a named commit, into a staging directory that is stamped and
verified *before* anything in place is touched; the previous tree is **moved**, never
deleted; then verified again in place, and a disagreement fails the deploy.

#### Asking

```bash
python3 robot-stack/preflight/tree_stamp.py verify /home/unitree/mappo-run
# [tree-stamp] clean: commit 92a0d38817df (feat/…), tree 64a35d3bcd4c
```

That `tree` is a real git object id — `git rev-parse <commit>^{tree}` resolves it, and
`tree_stamp.py id <dir>` prints it for *any* directory, stamped or not, which is how the
table above was built. **No `git` is involved on the robot**: measured on the Go2's Python
3.8.10 under the venv, a `verify` does not even import `subprocess`.

#### It refuses

`mappo_drive.py` calls the guard before it imports the vendored stack, so a run whose tree
does not match its stamp stops with the file named, not with a subtle wrong answer:

```
[tree-stamp] mappo_drive: REFUSING -- /home/unitree/mappo-stamped does not match its stamp.
stamp says   commit 92a0d38817df…  tree 64a35d3bcd4c…
disk gives   tree ca3ca14b8872…
1 modified:
    integration/mappo_policy.py
```

An unstamped tree refuses **on a robot** and is merely reported in a checkout — otherwise
the gate would fire on every clone and every CI job, and gates that always fire get deleted.

#### What it does not claim

It is not tamper-proof, and the module docstring says so. Editing a file is caught. Editing
the file *and* its manifest entry is still caught, because the tree id is derived from the
bytes and the stamp's is not — measured on the robot. Editing the tree id as well **will**
pass on the robot. What that cannot survive is contact with the repository:

```bash
python3 robot-stack/preflight/tree_stamp.py audit /path/to/.mappo-stamp.json .
# MISMATCH: this stamp's tree is not the tree of the commit it names.
```

So a false claim is *falsifiable from the run log*, which is the property a published
measurement needs. Real tamper-proofing needs a signing key, and a key readable by
everything on the robot is a key anything on the robot can sign with.

## 🛑 THE FIRST NUMBER: the Go2 will not walk below ~0.35 m/s

**Commanded top speed must be at or above `MIN_GAIT_COMMAND_M_S` (0.35 m/s), or the robot
does not walk at all.** It stands up, takes a few asymmetric one-or-two-leg steps, and
then stands **perfectly still** while the policy is still commanding it forward. No fall,
no fault, no error code.

| commanded | legs | outcome | runs |
| --- | --- | --- | --- |
| **0.21 m/s** (`command_scale` 0.6) | 10–23° of knee swing in bursts, then **3 s at exactly 0.0°** | travelled 0.34–0.43 m and stopped | **5** |
| **0.35 m/s** (`command_scale` 1.0) | 15–28°, continuous, all four | **2.07 m in 9 s, arrived** | 1 |

Measured 2026-08-14 on carpet with the D1 arm fitted, across BOTH the MAPPO policy and
the incumbent planner — it is a property of the robot, below the controller entirely.

**Why it costs hours.** Every instrument agrees, and all of them point away from the
cause: the joint encoders read 0.0° (the legs really have stopped), the state estimator
correctly reports no motion, and the stall gate then prints *"something is holding the
robot — check the tether"*. Five runs went on tethers, walls and the Go2's built-in
obstacle-avoidance switch before the speed itself was tested. `mappo_drive.py` now prints
a loud warning below the floor; it does not refuse, because the flag is legitimate for a
bench test with the legs off the ground.

**0.35 is the lowest speed observed to work, not a measured threshold.** 0.21–0.35 is
untested. Treat it as a floor and re-measure on any other robot.

## ⚠️ Four more numbers to know before the robot moves

| | |
| --- | --- |
| **The robot delivers 0.45 of what it is commanded when DERATED, and 0.70 at full speed.** | 0.45 was fitted against the pose over the derated recorded run (2.09 m travelled against 4.32 m commanded). At the shipped 0.35 m/s an arriving run measured a mean **0.240 m/s, a ratio of 0.70**, peaking at 0.415. The gain is strongly rate-dependent — 0.70 at full command, 0.45 derated, and **zero** below the gait floor above. Budget `--max-seconds` with the figure for the speed you are actually running, and measure rather than interpolate: the one thing this curve has proven is that it is not linear. |
| **`command_scale` is now shipped at 1.0, not 0.6.** | 0.6 × the 0.35 m/s envelope is exactly 0.21 m/s — the value measured above to stall five times out of five. The simulation's numbers were taken at 0.6 and that configuration **cannot walk on this robot**, so it is not a candidate whatever the sim says. At 1.0 the closed loop gave 21/30 arrived with **1 collision** against 0.6's 18/30 and 0. That extra collision is the honest cost of the change, and it is a cost paid against a baseline that does not move at all. |
| **`--robot-radius 0.25` is load-bearing.** | The vendored default is 0.40 m. The policy's `meters_per_vmas_unit` of 2.5 is calibrated as 0.25 ÷ the trained 0.10 VMAS radius. `mappo_drive` now **refuses to start** on a mismatch rather than running at a horizon the evidence does not cover — see below. |
| **The policy commands no yaw, and the heading servo is now OFF by default.** | It is a holonomic 2-D force, so with no servo the robot crabs and its 85° forward camera never looks anywhere new. Crabbing is the price of not hitting a wall, and **you now pay it by default — you no longer have to remember a flag.** [#16](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/16): the old law steered on the direction of the *command*, so `atan2` swung to ±180° every time the policy's `vx` went negative — 6.0% of moving ticks in simulation — and it **put the robot into a wall three times**. `--heading-servo goal` steers on the goal bearing in odom instead; in the closed loop that cuts the longest continuously-saturated yaw from **16.2 s to 1.7 s** and the peak body rotation from **179.8° to 74.9°** (30 seeds, `--controller policy`, the fitted 0.45 actuator gain). ⚠️ **No robot has been driven with it.** Ask before selecting it, and read the rung-3 note below. `--no-heading-servo` still works and still means off. |

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

Ablated, every controller reaches the goal 30/30 at the measured `command_scale` 0.6 —
so those historical arrival rates are about the obstacle, not goal acquisition.

**Do not select the 0.6 row.** It is the only simulated configuration with no collisions
that also beats the incumbent on arrivals, but it commands 0.21 m/s and failed to sustain
a gait in five hardware runs. The shipped 1.0 setting reached 21/30 with one simulated
collision; it therefore still requires the planner veto and the staged hardware ladder.
The raw policy is not a candidate.

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

`~/robotics-connect-go2` is `install.sh`'s **default** `ENV_DIR`, not a guarantee. An
install run with `--env-dir` (or `ENV_DIR=…`) puts the virtualenv somewhere else, and then
this line fails with `No such file or directory` on a machine that is otherwise correctly
set up. The Go2 in the lab is one of these — it was installed against a pre-existing
per-researcher venv. **The manifest records which was used**, so read it rather than
guessing:

```bash
grep '^env_dir' ~/.mappo-go2-deploy.manifest | tail -1
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
    --calibration ~/go2_front_camera.json \
    --static-prop bin --goal-class chair --goal-height 1.067 \
    --confidence 0.45 --robot-radius 0.25 --no-latch-arm \
    --arrive 0.8 --max-seconds 45 --policy-mode supervised
```

That command names no `--heading-servo`, which since [#16](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/16)
is how you get **no servo at all** — the configuration that has not driven into anything.
It used to be how you got the one that did.

`--policy-scale` and `--policy-command-scale` override `policy/config.json`, which already
carries the 2.5 and the 1.0 above. Pass them to try something else, not to restate the
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

### Stage the scene before you change the software

Two things that look like bugs and are staging, both hit on 2026-08-14. Fix them with
your hands; the goal on the day may well be a potted plant, and code written against a
chair's geometry would not survive that.

- **Swivel the goal so its back faces the robot.** The range is measured to the feature
  the height prior describes — a chair's backrest — and an office chair's five-star base
  sticks out ~0.3 m in FRONT of that. So `--arrive 0.8` stopped 0.78 m from the backrest
  and put the robot's nose in the wheels. Turning the chair round puts the base behind
  the backrest and 0.78 m becomes real clearance. The alternative, a per-object footprint
  model, is a lot of machinery for one prop.
- **Aim the robot at the goal before starting.** The policy beelines at the goal bearing;
  it has no concept of a corridor. Started 13.5° off-axis in a narrow hallway it turned
  to face the chair and walked its shoulder into a cubicle panel — which no part of this
  stack can see. Re-aimed to 4°, it went straight. `mappo_drive.py` with no `--live` is a
  free way to read the bearing off the telemetry first.

### Start smaller than you want to

1. `--max-seconds 20` and no obstacle at all. Does it walk to the goal in a straight line?
2. Add the bin, well off the lane. Does it deviate at all?
3. Bin on the lane. This is the demo.

**Always pass `--calibration`.** Without it the camera model falls back to a nominal 120°
FOV and f=916.7 px against the measured 85.27° and f=1290.2, and **every range reads 29%
short** — a chair at 2.84 m is reported at 2.02 m. Nothing fails; the run simply arrives
early against a goal that was never where it said. The console prints which model it
loaded on every run, and it is worth reading.

Between each, read the tail of the drive log: `N/M ticks driven by the policy`.

## Beside the rungs — the Device Connect dashboard

[`../dashboard/`](../dashboard/README.md) is **not a fourth rung.** The ladder above is about
handing the legs to a learned policy, and the dashboard does not do that: it drives the robot
directly, in bounded nudges, with no perception, no planner and no veto. It is what you use to
check a robot before a run, to nudge it back onto the start line between runs, and to see what
it is carrying — plus a live event stream so a room full of people can watch without an SSH
session each.

```bash
# on the robot, in the Python >= 3.11 env — START HERE, no motion
cd dashboard
python3 robot_driver.py --platform go2 --package ../policy \
        --bridge-python ~/robotics-connect-go2/bin/python

# on a workstation on the same LAN
python3 server.py --port 8080 --host 0.0.0.0        # then open http://<workstation>:8080
```

`--bridge-python` is the **SDK env's** interpreter, not the one running the driver — Device
Connect needs Python ≥ 3.11 and `unitree_sdk2py` lives on the Jetson's 3.8, so the driver
reaches the robot by running `drive_bridge.py` as a subprocess there. Get it wrong and every
command fails with an import error; `get_capabilities` reports the path it will use, so check
it before you need it. On a Lite3, stand the robot and enable high-level navigation mode on the
vendor interface first, then add `--operator-ready`.

Once the robot answers `get_status` and `list_models`, and **with the same conditions §Rung 3
requires — clear area, operator on the controller abort, tether slack checked** — restart the
driver with `--allow-motion`. That flag is the dashboard's `--live`, and
[`../robot-stack/SAFETY.md`](../robot-stack/SAFETY.md) governs it identically.

Two things about it that are easy to misread:

- **A checkpoint swap takes effect on the NEXT run, not the one in progress.**
  `MappoController` loads its weights at construction, so a running `mappo_drive` cannot have
  the network changed under it. Swapping while a run is live is safe and does nothing.
- **Every press reports what the robot measured**, not what it was told — including the
  fraction of the commanded speed actually delivered. On this Go2 that is ~0.45, and a press
  that says `commanded 0.35 for 1.5 s, travelled 0.012 m` is the gait floor at the top of this
  page, diagnosing itself.

Sub-gait-floor speeds are refused from the measured table before the driver even connects.
The Go2's **lateral** floor has never been measured (issue #42), so a Go2 strafe is permitted
and warns on every press that it may produce no gait at all — which would not be a fault. If
you run that experiment, issue #42 is where the number belongs.

⛔ **No robot has moved under the dashboard yet.** Everything above is verified against a bench
double that delivers exactly what it is commanded, which is precisely the number a real robot
does not produce. See `../evidence/2026-08-21-device-connect-dashboard/`.

## Troubleshooting

| symptom | cause |
| --- | --- |
| every tick `STOP_CLOCK_ERROR` | something is passing a wall clock as `timestamp_s`. The runners pass `time.monotonic()`; a hand-rolled loop is the usual culprit. |
| every tick `STOP_EXTERNAL_HOLD` | a mover is in the lane, or the tracker has a ghost. Check the overlay — a coasting track inflates its radius and blocks from metres away. |
| the robot barely moves | `command_scale`, or `--derate`. The shipped value is 1.0; 0.6 commands the 0.21 m/s setting measured to stall. |
| `veto` on nearly every tick | the policy and the planner disagree about the whole route. Drop to rung 2 and look at the shadow log before turning the veto off. |
| the robot crabs sideways and never turns | No `--heading-servo` was named, so it is off — which since [#16](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/16) is the default. ⚠️ This is **expected and correct**. Do not reach for `--heading-servo travel` to stop it: that is the law that hit the wall. |
| the robot yaws hard and keeps turning | `--heading-servo travel` was named. Stop the run. That is [#16](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/16)'s law and it is retained only so the 2026-08-17 evidence stays reproducible; `mappo_drive.py` prints a ⚠️ line at startup when it is selected. |
| `cannot import the policy package` | run from `integration/`, or pass `--package ../policy`. |
| a policy config is refused at load | it disagrees with the checkpoint's own recorded training constants. The message names the field; do not "fix" it by editing the checkpoint. |
| `REFUSING TO RUN — the policy's scale was calibrated for a different robot size` | `--robot-radius` and `meters_per_vmas_unit` disagree. Pass `--robot-radius 0.25`, or pass `--policy-scale` to match the radius you meant and **re-run the simulation** — the numbers above do not transfer. Nothing moved; the robot is still prone. Every refusal is also appended to `~/.mappo-refusals.jsonl` (`--refusal-log`), because a refused run writes no telemetry and would otherwise leave no trace of why the demo did not start. |
| RPC segfault on any DDS call | `setup_env.sh` was not sourced. `LD_LIBRARY_PATH` is load-bearing — see `robot-stack/unitree/go2/install/setup_env.sh`. |
| `[venv-guard] REFUSING TO RUN` | you are on a robot in the system Python. The message names the venv to activate and the command to build one — **do not `pip install` past it**, and do not set `MAPPO_ROBOT_HOST=0` unless the machine really is not a robot. See "Python on a robot" above. |
| `ModuleNotFoundError: cyclonedds` / `unitree_sdk2py` on the robot | the same thing one step earlier — the vendor stack is in the venv and this interpreter is not. `source <env>/bin/activate`. It is never a reason to install anything. |
| `venv-guard: not enforced` on a live run | the guard found no robot-host marker on this machine. Expected on a Lite3 unless the SOP's `MAPPO_ROBOT_HOST=1` is exported; on a Go2 it means something is wrong with the detection and is worth reporting. |
| every dashboard command fails and you are about to install Python 3.11 on the Jetson | **stop.** `device-connect-edge` runs off-robot by design and no venv can supply 3.11 here. Read "Why `device-connect-edge` runs OFF the robot" above. |
| the dashboard shows "no robots found" | the driver is running but not announcing. Check it is on the same LAN segment (D2D is multicast), then that no `@rpc` was added whose name collides with a `DeviceDriver` member — that stops presence silently, with no error in any log. `dashboard/test_robot_driver.py` catches the second case. |
| every dashboard command fails with an import error | `--bridge-python` points at the driver's interpreter instead of the SDK env's. `get_capabilities` reports the path in use. |
| the motion keys are greyed out | the driver was started without `--allow-motion`. That is the default and it is deliberate. |
| a strafe on the Go2 does nothing | expected, and not a fault — that robot's lateral gait floor is unmeasured (issue #42). The result panel reports the distance actually travelled. |

## What is still open

- **Peers are invisible.** Another quadruped is neither a detector class nor a colour
  profile, so a *multi*-agent demo with two robots in one arena has no way for either to
  see the other. Issue #6.
- **The steering response is a cliff, not a ramp** — about 103° once the obstacle is
  inside the horizon and essentially nothing outside it, at every scale from 1.5 to 4.0.
  Only a retrain with a larger `lidar_range` changes that. Issue #4.
- **Episodes were 100 steps in training**, ten seconds at the stack's 10 Hz. A 60 s demo
  run is six times longer than anything the policy saw.
