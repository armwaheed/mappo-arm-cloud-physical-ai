<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The Device Connect dashboard

Watch a quadruped, drive it, and manage the MAPPO checkpoint it is carrying — from a browser,
over [Arm Device Connect](https://github.com/arm/device-connect).

⛔ **[`../robot-stack/SAFETY.md`](../robot-stack/SAFETY.md) governs anything here that moves
a leg, and it is not optional.** Motion is off by default; `--allow-motion` is this
directory's `--live`.

📄 **Bilingual (EN/中文) operator guide: [`OPERATOR-GUIDE.zh-CN.md`](OPERATOR-GUIDE.zh-CN.md).**
It covers the seven workflows below and the Lite3's four surprises, paired paragraph by
paragraph for the Shanghai team. This file stays the authority; that one is the shorter road
in. For a **first Lite3 bring-up**, start instead from
[`../robot-stack/deep_robotics/lite3/LITE3-DASHBOARD-BRINGUP-PROMPT.md`](../robot-stack/deep_robotics/lite3/LITE3-DASHBOARD-BRINGUP-PROMPT.md),
which is a staged, verify-before-proceeding bring-up written to be handed to a coding agent.

```
   browser
      │  HTTP + Server-Sent Events
   server.py                              ┐
      │  device-connect-agent-tools       │
 ═════╪══════ Device Connect mesh ═══════ │ a WORKSTATION, Python >= 3.11.
      │  device-connect-edge              │ D2D by default: no broker, no etcd, no Docker
 robot_driver.py                          ┘
      │  subprocess, one command, JSON, exit
 drive_bridge.py                          ← the SDK env, Python 3.8
      │
 Go2Locomotion / Lite3Locomotion
```

⛔ **The robot cannot host the driver, and that is architecture rather than a packaging
bug.** The Go2's Jetson carries Python 3.8.10 and 3.9.5 and nothing else — measured on
`192.168.123.18`, and `apt-cache policy python3.11` has no candidate on Ubuntu 20.04 /
JetPack 5. `device-connect-edge` requires ≥ 3.11. **A virtualenv cannot close that gap**: a
venv is built *from* an interpreter and cannot supply a version the machine does not have,
which is also why [`../AGENTS.md`](../AGENTS.md) forbids installing a newer Python on a
robot. So `robot_driver.py` runs on a workstation, and everything that must execute *on*
the robot crosses a seam that was built for it:

| what has to be on the robot | how it gets there |
| --- | --- |
| the SDK calls that move or read the robot | `drive_bridge.py`, launched by `--bridge-python` over the robot's own 3.8 |
| the front camera | `go2_frame_server.py`, started by hand on the robot, read by `--camera-url` |
| a MAPPO run | `mappo_drive.py`, launched by `start_run` as **a subprocess over SSH** — the seam is [`run_control.py`](run_control.py), and the shape is written down rather than hidden |

## Start it

One command, from `dashboard/`, on a **workstation** — not on the robot:

```bash
./start-dashboard.sh                              # a bench double. No robot needed.
./start-dashboard.sh --robot 192.168.123.18       # a real Go2's camera, simulated pose
```

It starts the checkpoint server, one driver and the dashboard, waits until the page
answers, prints the URL, and **stops all three together** on Ctrl-C. `--help` lists the
rest; `--dry-run` prints the three commands it would run without running them.

It was three terminals, and each of the three had its own way of failing several steps
after the mistake. So it refuses first, by name:

| it refuses on | rather than |
| --- | --- |
| an interpreter below 3.11, naming one on this machine that works | `ModuleNotFoundError: device_connect_edge`, which reads as a missing package |
| a missing package, printing the exact `pip install` **for that interpreter** | three tracebacks, twenty seconds apart, one per process |
| a policy package with no `config.json` | a driver that starts and answers `list_models` about nothing |
| port 8080 or 8800 already held, naming the pid | `Address already in use` from inside the second process to start |

⚠️ **It never enables motion.** `--allow-motion` is added to the driver's command line only
when a person types it on this one; no environment variable, default or config file can
turn it on, and `test_start_dashboard.py` pins that with the complement test beside it so
the check cannot pass by the flag never working at all.
[`../robot-stack/SAFETY.md`](../robot-stack/SAFETY.md) governs it identically to `--live`.

### What has to be installed first

```bash
python3.11 -m pip install device-connect-edge device-connect-agent-tools aiohttp numpy Pillow
```

⚠️ **`aiohttp` and `numpy` are not optional and nothing else installs them.**
`device-connect-agent-tools` depends only on `device-connect-edge`, which depends on
`eclipse-zenoh`, `nats-py`, `nkeys`, `pydantic` and `pyyaml` — so a line naming the two
Device Connect packages and `eclipse-zenoh` (which is already a dependency) installs
nothing any of these three programs import. Measured in a clean 3.11 venv: `server.py`
dies at `No module named 'aiohttp'`, `robot_driver.py` at `numpy`, and `model_server.py`
at `numpy` through `model_store`. `Pillow` is needed only by the **synthetic** sim camera.

`python3` on macOS is the Command Line Tools 3.9.6 however many Homebrew Pythons are
installed, so `python3.11` above is literal. The launcher picks an interpreter that can
already import `device_connect_edge` before it picks the newest ≥ 3.11, because a machine
with 3.11, 3.12 and 3.13 usually has Device Connect in exactly one of them.

## The seven things, and which of them work

| | | how | today |
| --- | --- | --- | --- |
| 1 | **Open the dashboard** | `./start-dashboard.sh`, then the URL it prints — `http://127.0.0.1:8080` | ✅ verified |
| 2 | **View the fleet, or one real robot** | the fleet table, grouped by platform; `--robot HOST` puts a live Go2 camera on the row | ✅ verified — 1920×1080 frames off `192.168.123.18` at 6 fps, `frame age 0.03 s` |
| 3 | **Connect to a model server** | the launcher starts `model_server.py` and advertises it, so **Load from Cloud AI** opens with the source and address filled in | ✅ verified — `Browse` → `served by mappo-model-server` |
| 4 | **Load a model** | **Download** on a browsed row. The fetch runs **on the robot**, not in the browser | ✅ verified — 268 063 bytes, sha256 `7327f724…ca11` |
| 5 | **Start MAPPO on the robot** | `start_run` / `stop_run`. **Start** with no arguments is the scene check and carries no `--live`, so it *cannot* command a leg; real motion needs `--allow-motion` at driver launch **and** `arm_motion` in the request | ✅ RPCs verified over a real D2D mesh against the bench double; ❌ **no robot has been driven by them** |
| 6 | **Stop / start / take manual control** | per-row stop, `STOP ALL (n)`, and the motion pad — the pad needs `--allow-motion`. `stop` and `STOP ALL` now end a **running policy** as well as a nudge, and `stop_run` is how a person takes the robot back mid-run | ✅ verified on the bench double, including `STOP ALL` ending a live run; ❌ never on a real robot |
| 7 | **Swap models** | **Arm** on a checkpoint row. Takes effect on the **next** run | ✅ verified — `previous` reported, and deleting the armed one refused |

### 5: what Start does, and why it is not one button

**The default press cannot move the robot.** `start_run()` with no arguments builds the
command line `run-smoke.sh scene` builds: the camera, the detector, the policy, the
planner's veto and the telemetry, with **no `--live`**. `--live` is the only flag in
`mappo_drive.py` that commands a leg, so this is an **absent capability, not a checked
permission** — which is what makes it a reasonable thing for a button to do with no
ceremony. It is the right thing to press first, every time.

**Real motion is two deliberate acts, on different surfaces, at different times.**

| gate | who sets it | when | RPC / flag |
| --- | --- | --- | --- |
| 1 | whoever has looked at the room | at driver launch | `--allow-motion`, which `start-dashboard.sh` will never add for you |
| 2 | whoever presses the button | in that request | `start_run(arm_motion: true)` |

Neither is remembered: `--allow-motion` dies with the driver process and the arm dies with
the run. **Missing either is a refusal that says which** — never a quiet dry run. The silent
downgrade is the worse failure of the two: the operator asked for a live run, watched a run
start, and is now watching a robot that is not moving, which is exactly what they would see
from `mode='mcf'`, from a flat battery, and from a sub-gait-floor command. They would spend
the demo telling those apart, and this directory refuses the other two for the same reason.

The gate is checked in the driver **and** in `run_control.build_run_argv`, which is the same
duplication `drive_bridge.py` already has with the driver: two processes with one shared
assumption is one process with an unwritten contract.

`start_run` needs a run profile, and `start-dashboard.sh` reaches it through the `--`
pass-through it already has rather than through a flag of its own:

```bash
./start-dashboard.sh --robot 192.168.123.18 -- --run-profile run-profile.example.json
```

That keeps the launcher's one invariant where it is: **it never adds `--allow-motion`**, and
`--` cannot smuggle one past `start_run` either, because the driver's gate reads the driver's
own flag and not the request's.

### and the seam it needed, which is a subprocess over SSH

`invoke → walk_forward` and `invoke → start a MAPPO run` look like the same shape and are
not. Three things were already load-bearing here, and each one decided part of the answer:

* **A run is 45 seconds and an RPC handler may not block.** `DeviceRuntime._cmd_subscription`
  dispatches one RPC at a time per device. So `start_run` returns as soon as the run is
  *launched* and the outcome arrives as `run_started` / `run_output` / `run_finished`
  events — the same shape the motion RPCs already use, for the same measured reason. The one
  place it does block is a ~1.9 s pre-run status read, and that is argued in `start_run`'s
  own comment: nothing is moving during that window, because a run in flight and a nudge
  holding the lock are both refused before it.
* **`mappo_drive.py` cannot be imported by the driver**, and it does **not** go through
  `drive_bridge.py` either. That file's rule is *one command, one process, exit*, and the
  rule is a safety property: every command runs in a process that exits, so a driver that
  hangs cannot leave a velocity latched. `pose-stream` is exempt because a pose *reader* has
  no velocity to latch. **A run does.** So run control got its own module rather than a
  second exemption from the rule that is protecting exactly this case.
* **And the driver is not on the robot.** So `start_run` starts a process on the Jetson the
  only way it can: over SSH, with the whole command line rendered as one shell line. That is
  written down as what it is — [`run_control.py`](run_control.py)'s module docstring opens
  with it — rather than hidden behind a `launch()` that reads like a local call.

⚠️ **SIGTERM to the local `ssh` client does not stop a remote run.** Measured on the lab Go2
on 2026-08-26, with `/bin/sleep 120` standing in for the policy:

| | |
| --- | --- |
| launch, then read the pidfile and `ps` on the robot | pid `48126`, `/bin/sleep 120` running |
| SIGTERM the local `ssh` client, wait 3 s, `ps` again | **`48126 /bin/sleep 120` — still running** |
| run the stop command `run_control` builds | exit 0 |
| `ps` again | gone |

So the launch records the remote shell's pid before `exec`-ing the run (`exec` keeps the
pid), and the stop is a **second connection** that signals that pid. ⛔ SIGTERM, never
SIGKILL: [`../robot-stack/SAFETY.md`](../robot-stack/SAFETY.md) §0 is written from a robot
that broke a window because a process commanding motors was hard-killed, and
`test_run_control` asserts that no command this module builds contains `-9`.

A stop that does not confirm comes back `ok: false` saying to assume the policy is still
driving and use the physical abort — because the dashboard classifies a stop by `ok`, and an
operator must not be shown a tick for a robot that may still be moving.

### One authority at a time, and it is named

`get_status().control.owner` is `operator` or `policy` — never both, never neither. A live
run moves it to `policy`; `stop_run` and `stop` move it back. **While it is `policy`, the
motion pad is refused**, the refusal names the run and names `stop_run`, and it arrives as a
`motion_refused` like every other turn-down. The page greys the pad and says why on it.

Refusing is the choice, and the two alternatives are what make it one. If both ran, a manual
nudge and a policy tick would each write a velocity on the same bus at 10 Hz and the last
writer would win — an arbitration with no owner, no record, and nothing an operator could
watch. Silently killing the run on a key press is worse still: the operator's press would
end a run they may not have known was happening. So the pad goes inert, visibly, and taking
it back is one deliberate call.

**A run that is not live takes nothing.** A scene check commands no leg, so locking the pad
against it would be a claim about who is driving that is not true.

`stop` ends the policy **first**, then the in-flight nudge, then commands zero. That order is
the point: a run refreshes its velocity every tick, so a zero commanded while it is still
running is overwritten inside one control period and the robot visibly pauses and carries
on — the same defect `stop` already terminated the nudge worker for. `STOP ALL` and every
per-row stop invoke the same `stop` function, so they inherit it rather than growing a second
implementation that could disagree.

### The robot the run happens on, and what cannot be said about it

A run profile — `--run-profile`, worked example in
[`run-profile.example.json`](run-profile.example.json) — fixes the machine, the interpreter,
the working directory, the environment and the deployment's own flags. It is a file on the
machine running the driver, and **none of it can be changed by a request**: the RPC chooses
only `seconds`, `policy_mode`, `heading_servo` and `arm_motion`, each checked against a fixed
set before anything is quoted into a shell. A profile that could spell `--live` would be a
second motion gate nobody would think to look for, so the flags this driver writes are
refused in a profile. A profile that could carry an SSH password would publish it — the
prefix goes into `get_capabilities` and the rendered command onto the event stream — so
`sshpass` and friends are refused at load rather than redacted at render.

The example profile is copied from `/home/unitree/run-smoke.sh`, which is the invocation
known to work on that robot, and a test pins it against that wrapper's flag set.

⚠️ **Nothing here can name the code it launches.** The deployed tree is not a checkout: no
`.git`, so no branch and no commit, and nothing on a robot reports its own staleness.
Measured on 2026-08-26, on the tree `run-smoke.sh` actually runs and whose name says `main`:

| `/home/unitree/mappo-main` | last matched a commit |
| --- | --- |
| `integration/mappo_drive.py` | **67** commits behind `main` |
| `README.md` | **93** behind |
| `policy/config.json` | **95** behind |

Three files, three different commits, one directory. So `start_run` reports the **command**
— in the reply, in the `run_started` event, and in `get_capabilities.run.command_preview`
*before* the press — and says in the same breath that it cannot report the code.
`named_commit` is `null` and is meant to stay that way.

### Every flag is spelled, and one of them has two spellings

`--policy-mode`, the heading servo and `--max-seconds` go out on every run even at their
defaults, because a tree 67 commits behind has whatever defaults it had that day. The servo
is the one that matters: it became opt-in in #106, and on that tree **omitting it selects
issue #16's `travel` law** — the one that saturated the yaw rate and put this robot into a
cubicle panel or a cabinet on three runs out of four on 2026-08-17.

⚠️ And #106 **renamed** the flag. Run against the lab Go2's own parser, 2026-08-26:

```
['--heading-servo', 'off']  -> REJECTED, argparse exit 2
['--no-heading-servo']      -> ACCEPTED end to end
```

So `heading_servo_flag` in the profile says which spelling *that tree* understands.
`"legacy"` writes `--no-heading-servo` and refuses `goal` and `travel` **here** rather than
sending a flag the far end will reject — that rejection arrives as a usage message across an
SSH connection, seconds later, and reads as "the run did not start" with no reason.
`get_capabilities` then advertises `heading_servos: ["off"]`, so the page offers what the
tree can be told instead of offering two refusals.

### A run is bounded, because a nudge is

A motion nudge is capped at 5 s by the worker, so a web button cannot start a walk that
outlives the attention of the person who pressed it. A run needs a much larger number and
still needs one: `seconds` is clamped to 120, sent as `--max-seconds`, and backstopped by a
watchdog in the driver that ends a run which outlived its budget plus 30 s of startup. The
watchdog is a backstop, not the bound — it fires when the run does not honour its own flag.

### What a run costs the rest of the page

`robot_state` is **not** emitted for the whole of a run. Each one is a subprocess that
connects to DDS, reads and exits — 1.94 s of cold SDK import and discovery on the Go2's
Jetson, measured — and a run holds that bus at 10 Hz. So pose and mode go stale on the fleet
row until the run ends. It is not a blackout: `run_output`, `run_finished` and
`control_changed` carry the run, the fleet row shows `POLICY DRIVING`, and `--publish-pose`
is the channel built for a pose needed *while* the robot moves.

⛔ **No robot has been driven by `start_run`.** What is proven against hardware is the launch
and stop mechanism, the environment, and the command line's acceptance by the deployed
parser — all in the pull request. The run itself is not.

### What is real when you ask for a real robot

`--robot 192.168.123.18` expands to `--platform go2 --simulate --camera-url
http://192.168.123.18:8801/`, and **that is a mixture on purpose.** A demo that implies
more than it shows is the failure this repository spends its evidence files avoiding, so:

| on the page | where it comes from | real? |
| --- | --- | --- |
| the camera viewport | the Go2's own `VideoClient`, over HTTP from `go2_frame_server.py` | ✅ **the robot** |
| pose, velocity, `mode` | the bench double — `mode` reads `sim`, pose reads `0, 0, 0` | ❌ simulated |
| the gait floors, and every refusal built on them | the **Go2's measured** table: 0.35 forward, 0.20 lateral, yaw unmeasured | ✅ real numbers |
| a motion command | the bench double. `delivered_fraction` is 1.0, which a real robot never produces | ❌ simulated |
| the checkpoint list, free space, downloads | the **workstation's** `--package` directory | ❌ not the robot's |

Drop `--simulate` (`--platform go2 --bridge-python …`) and pose, mode and motion become the
robot's — at the cost below.

⚠️ **A `status` read costs ~1.95 s, not milliseconds.** Measured on the Go2's Jetson over
three consecutive `drive_bridge.py status` calls: **1.98 s, 1.93 s, 1.95 s**, all of it the
cold SDK import and DDS discovery, paid *per invocation* because the bridge is one command
per process. Against `STATE_INTERVAL_S = 5.0` that is ~39 % of every poll period with a DDS
client on the bus, and it is why the fleet row updates in seconds rather than in frames.

🔴 **The robot reports `mode='mcf'`, not a sport mode.** `Go2Locomotion` warns that `Move`
commands may be silently ignored in that mode, and nothing was commanded on the read-only
run that found it — so it is a reading, not a fault. But an operator who presses
**walk forward** on a robot in `mcf` gets a robot that accepts every command, reports no
error and never steps. A motion run has to call `ensure_sport_mode()` first, and knowing
that beforehand is the difference between a five-second fix and an afternoon.

⚠️ **`~/mappo-run` on that robot matches no single commit.** Most of it was 34–36 commits
behind `main`, its `README.md` older still, its `dashboard/` a different lineage again, and
**two files match nothing at all**. None of the deployed trees is a git checkout — no
`.git`, so no branch and no commit — so **a live run does not tell you which code produced
it.** Copy a fresh tree, record what you copied, and do not report a robot observation as
evidence about `main` without saying what was actually on the robot.

### The camera server, which is started by hand on the robot

Copy `go2_frame_server.py` to the robot, then — this is the recipe that is running as these
words are written, not a reconstruction of one:

```bash
ssh unitree@192.168.123.18
source /home/unitree/mappo-main/robot-stack/unitree/go2/install/setup_env.sh
export PYTHONPATH=/home/unitree/deps:/home/unitree/unitree_sdk2_python
setsid nohup python3 /home/unitree/go2_frame_server.py \
       > /home/unitree/frame_server.log 2>&1 < /dev/null &
```

Read-only: it opens `VideoClient` and nothing else — no lease, no motion, no writes. `/`
returns the latest JPEG; `/status` returns exactly

```json
{"seq": 7635, "age_s": 0.02, "have_frame": true}
```

which is the quickest way to tell "the camera is dark" from "the server is not running":
`have_frame` is a JSON boolean, and `age_s` is `-1.0` when no frame has ever arrived.

**The interface is hard-coded to `eth0`.** A trailing `eth0` on that command line is inert —
`go2_frame_server.py` never reads `sys.argv`, and `ChannelFactoryInitialize(0, "eth0")` is a
literal. Passing another interface name changes nothing, which is worse than being rejected;
edit the file if you need a different one.

⛔ **Drop the `source` line and it does not fail, it SEGFAULTS.** Measured on this robot:
with the environment, `go2_frame_server.py` gets as far as its socket bind; with
`LD_LIBRARY_PATH` unset it dies `Segmentation fault`, rc **139**, at the first
`VideoClient.GetImageSample()`, before printing its banner. The cause is diagnosed in
[`../robot-stack/unitree/go2/install/setup_env.sh`](../robot-stack/unitree/go2/install/setup_env.sh):
the Jetson ships two CycloneDDS 0.10.2 builds, `ldconfig` resolves the wrong one, and only
the RPC serialize path is sensitive — so passive subscribers work either way and every RPC
client crashes. From the workstation it reads as `camera url … unreachable: Connection
refused`, not as a missing variable.

⚠️ **The venv is what makes it work, and it has been reported as the thing to avoid.**
Stated plainly because a wrong reason is a fix nobody can reproduce:
`setup_env.sh` prepends `/home/unitree/robotics-connect-envs/armwaheed/bin` to `PATH`, so
the `python3` in that recipe **is** that venv — the process serving frames right now has
exactly that `PATH`. Measured, four ways, with `LD_LIBRARY_PATH` set:

| interpreter | `import cyclonedds` | `GetImageSample()` |
| --- | --- | --- |
| `robotics-connect-envs/armwaheed/bin/python3` | from the venv's site-packages | **`0`, 188 399 bytes** |
| `/usr/bin/python3` | `ModuleNotFoundError` | never reached |

`cyclonedds` is installed **only** in that venv, and its `bin/python3` is a symlink to
`/usr/bin/python3` with `include-system-site-packages = true` — so "the venv" and "the
system interpreter" are the same binary and differ only in which site-packages they see.
The thing that segfaults is a **missing `LD_LIBRARY_PATH`**, not the venv; running the
venv's interpreter directly without sourcing `setup_env.sh` produces both symptoms at once
and is the likeliest way to reach the wrong conclusion.

⚠️ **To stop it, resolve the pid and kill the pid.**

```bash
ssh unitree@192.168.123.18 "ps -eo pid,cmd | grep '[g]o2_frame_server'"   # note the pid
ssh unitree@192.168.123.18 "kill <pid>"
```

⛔ **Never `pkill -f go2_frame_server` over SSH.** The pattern matches the SSH command line
carrying it, so `pkill` kills your own calling shell, reports nothing, and leaves the target
running — a stop that looks like it worked and did not. The `[g]` in the `grep` above is the
same trap in its milder form: it stops the grep matching itself.

⚠️ **The robot's copy is not a checkout and does not report its own version.** Check `md5sum`
against `dashboard/go2_frame_server.py` on `main` before treating a frame as evidence — the
robot's copy has already been ahead of the repository's once, when it carried a lint fix the
repository did not.

## What it does

| | |
| --- | --- |
| **Run a fleet** | Every robot on the mesh listed at once, each with its own stop, plus one **STOP ALL**. Robots appear as they connect and are tombstoned as GONE when they drop off. |
| **Let robots avoid each other** | `--publish-pose` puts a robot's own pose on the mesh at 10 Hz, and `peer_link.py` on another robot spools it for that robot's control loop. No detector, no marker. See below. |
| **View robot events** | A docked drawer, always on screen. Collapsed it is one bar carrying the newest line and an unread count; open it fills the bottom of the window. Filterable, pausable, and replayed from a ring buffer to a page that opens late. |
| **Basic motion** | Walk forward / back, strafe left / right, turn left / right, lie down. Bounded in time, and every press reports what the robot *measured*. |
| **Front camera** | A live MJPEG viewport beside the pad, default 6 fps, started only while someone is watching. |
| **Start and stop a run** | `start_run` / `stop_run`. The default press is the scene check and *cannot* move the robot; motion is opted into twice. `stop_run` is also how a person takes the robot back mid-run. |
| **Swap checkpoints** | Arm any `.npz` already on the robot for the next run. |
| **Load / unload from Cloud AI** | Pull a checkpoint from an S3 bucket or a direct server address; delete one off the robot. |

## Where the event stream lives, and why it is a drawer

It was a panel at the foot of a scrolling page, and it was missed entirely — an operator had
to already know it existed and scroll to find it. Two pieces of guidance pull in opposite
directions here and the drawer is what satisfies both.

Dashboard practice is consistent that a live log is **secondary**: keep detailed tables and
audit logs below the summaries, hold the initial view to about five or six elements, and give
a high-frequency feed collapsible sections and a pause control rather than letting it compete
with the controls. Equally consistent is that anything critical must not depend on the user
scrolling to find it, and that key controls belong in a fixed position that does not move.

A docked, collapsible drawer is the shape that satisfies both, and it is the one DevTools,
VS Code and PatternFly all converge on: on the same surface as the content, never covering
it, always present. Collapsed it costs 44 px and still carries the newest line and an unread
count, so it is telling you something even shut. `E` toggles it, and the choice is remembered.

`robot_state` is hidden by default. Four robots at 5 s intervals is ~48 lines a minute of
"nothing changed", which buries the lines that matter; one checkbox brings it back.

The page also stopped being a two-column *grid* of panels. A grid puts each panel in a grid
row, so a row begins below the tallest item in the row above — the motion panel — and the
right-hand column grew a block of dead space with the next panel stranded below it. The two
sides are flex columns now and stack independently.

## The camera, and why it is not an RPC

The obvious implementation is a `get_frame()` RPC the page polls. It is the one
implementation that must not be used here, because **the edge runtime dispatches one RPC at a
time per device**. Polling frames at even 6 Hz would occupy the command channel continuously
and put the robot back to being deaf to `stop` — reintroducing the defect the non-blocking
motion work removed, in a form nobody would suspect, because "the camera is on" does not
sound like "the stop button no longer works".

So frames are **emitted as events** (pub/sub, its own subject), drained by the dashboard on
**its own subscription and its own thread**, and served to the browser as
`multipart/x-mixed-replace`. A plain `<img>` renders that natively: no canvas, no per-frame
JavaScript, decode off the main thread, and a dead feed shows as a broken image rather than
as a frozen last frame that still looks live.

**12 fps was asked for; 6 is the default.** A 640×480 JPEG at quality 60 is 25–40 KB and
base64 adds a third, so 6 fps is roughly 200–320 KB/s *per watched robot* and 12 is double
that — for a viewport whose job is "can I see where the robot is pointing". It is a
parameter, so raising it is a decision someone makes with the number in front of them.

⚠️ **On a real Go2 it is four times that, because the frames are 1920×1080.** Measured
through `/api/camera/…?fps=6` against `192.168.123.18`: **48 frames in 8 s, 186 KB each,
1.09 MB/s** from one watcher. The paragraph above describes a 640×480 camera and the Go2 is
not one. `MAX_FRAME_BYTES` (512 KB) still holds with 2.8× of headroom, so nothing refuses —
but two people watching two robots is 4 MB/s off a demo LAN, and `?fps=` is the number to
lower rather than the one to raise.
Nothing streams to an empty room: the feed starts when a viewer asks and stops when that
interest lapses, so closing the tab stops the robot emitting.

**A synthetic or replayed feed starts on its own; a live one waits to be asked.** The black
rectangle was the commonest reaction to the first build — a viewport nobody notices is a
viewport nobody uses. But "start streaming the moment a page opens" is the wrong universal
default: on real hardware that is 200–320 KB/s off a robot, and a camera contended with a
live run, because somebody opened a tab. The driver advertises which kind of feed it has, so
the demo autoplays and a robot does not. An explicit Start/Stop always wins and is
remembered, because a feed you stopped must stay stopped or the button reads as broken.

⚠️ **The Lite3 will disappoint you here.** Its frames come from an OpenCV `VideoCapture`,
which on Linux is typically **exclusive** — so while a `lite3_visual_nav` run holds the
camera, this cannot open it. That is reported as "the camera is in use" rather than as a
black rectangle. The Go2's frames come from the SDK's `VideoClient`, an RPC to the robot's own
video service, so a run and the viewport can both read it. On `sim` the frames are
**synthetic** and marked as such on screen, because a screenshot of a fake camera must not be
mistakable for hardware.

## Letting one robot avoid another, with no perception at all

Another quadruped is not a detector class. A detector fine-tuned on 1,343 real in-domain
frames of the peer reached **53% recall at 38% false positives** on held-out footage, with
no usable operating point at any threshold — the ceiling is the frozen backbone, not the
data. A marker and a colour panel on the peer were both ruled out.

The mesh already carries the answer. A robot knows its own pose to the accuracy of its own
estimator, and both robots are already on it.

```
  peer robot                                    navigator robot
  drive_bridge.py pose-stream  (py3.8, SDK)     peer_link.py       (py>=3.11, DC)
        │ JSON lines, 10 Hz                            │ writes ~/.mappo-peers.json
  robot_driver.py --publish-pose (py>=3.11) ═══════════╡ event(peer_pose)
                                    the mesh           │
                                                integration/peer_source.py  (py3.8)
```

`peer_link.py` is `drive_bridge.py` in the other direction: the driver exists because the
Device Connect side cannot import the robot's SDK; this exists because the SDK side cannot
import Device Connect. Same wall, opposite direction, and the same file-on-disk seam.

**`peer_pose` is a separate event from `robot_state`, not a faster one**, and the reason is
not the rate. `robot_state` is skipped while a motion command holds the lock, and each one
is a subprocess that connects to DDS, reads and exits. Both are disqualifying: a peer pose
is needed *most* while the peer is walking, which is exactly when that lock is held, and
10 Hz of process starts on a Jetson is not a rate a robot can produce. `pose-stream` is
therefore the one bridge command that does not exit after one result — permitted because
the "one command, one process" rule guards against a **latched velocity**, and a command
that only reads the estimator has none to latch. It is also deliberately not wrapped in
`safe_stop_guard`: a pose *reader* must not stop a peer that is walking.

**Publishing is not interest-gated the way the camera is.** A camera feed costs 200–320
KB/s and nobody is harmed by it starting late; a pose is ~200 bytes a sample and is what
stops another robot's legs. Making it wait for a consumer to ask would add "the request
never arrived" as a way for a peer to become silently invisible. So `--publish-pose` starts
at `connect()` and runs for the driver's lifetime: *this robot is on the mesh* and *this
robot's pose is on the mesh* should be the same fact. It needs no `--allow-motion`, because
telling other robots where you are is not permission to walk.

**`--peer` on `peer_link.py` is an allow-list, and required.** The navigator very likely
publishes its own pose too — so that peers can avoid *it* — and a denylist that forgot to
exclude self would put an obstacle on top of the robot and hold the run forever, with a
diagnosis nobody would guess. It is also a safety input, and what a demo LAN with no PKI
gets to put on the obstacle list should be a deliberate act. The cost is that an unlisted
robot is invisible, which is the wrong way to fail, so every unlisted device seen
publishing is logged once by name with the flag to add.

⚠️ **No timestamp taken on the peer is used, anywhere.** Two robots on a demo LAN with no
NTP share no wall clock, and their monotonic clocks count from unrelated boots. What
crosses the mesh is a *duration* — how long ago the peer read its own estimator, measured
on the peer, between two readings of one clock. Every age is then computed from clocks read
on the navigator. The staleness argument, and what happens when a pose stops arriving, is
in [`../integration/peer_source.py`](../integration/peer_source.py); the short version is
that the obstacle is dropped and the robot holds, and those are one decision.

⚠️ **Nothing here can supply the transform between two robots' odom frames.** Each starts
at its own robot's power-on pose. `mappo_drive.py --peer-odom-align` is where that is
declared, and it is the flag that turns peer avoidance on, so there is no state in which it
is missing.

## Alerts, and what is deliberately NOT in them

The bell carries **things that happened** — a refusal, an interruption, a checkpoint change.
Clicking one focuses the robot that raised it, which is the point of an alert: to get you to
the thing, not to tell you about it. They are dismissible individually or all at once.

**The standing capability caveats are not in there.** "No measured lateral gait floor on the
go2" is not an event — it is true on *every* press, forever. A caveat you can dismiss once and
then never see again while you press that control fifty more times is worse than no caveat at
all. Those ride the control instead: a ⚠ marker on the key itself, with the full sentence in
its tooltip. The wall of yellow boxes under the motion pad is gone; one line remains saying
which keys are marked.

## The safety banner collapses, but the state does not

The red "motion is enabled" banner has a dismiss button. The **`MOTION ENABLED` badge in the
top bar does not** — the badge is the *state* and the banner is the *explanation*, and only
the explanation gets to be dismissed. The banner also comes back whenever motion transitions
off→on or the focused robot changes, because "I already read it" is true of a sentence and not
of a different robot, and dismissal is not persisted across a reload.

## A demo host with no robots

`--simulate` presents a platform's identity and **rules** while driving the bench double, so
a simulated Go2 refuses 0.21 m/s with the same measured number a real one does. `--platform`
carries the rules and `--backend` decides what is driven; collapsing the two hands a demo the
bench double's gait floors of zero and the refusal silently stops firing. Simulated robots
are badged `sim` on their fleet row and in their device identity, because a demo fleet
indistinguishable from a real one is a hazard rather than a better demo.

`--camera-replay-dir` serves a directory of JPEGs as the camera, **labelled in the pixels**
rather than by the page — a screenshot keeps the pixels and loses the caption. See
[`../deploy/demo/`](../deploy/demo/README.md).

`--camera-url` is the opposite case and is deliberately **not** labelled: those frames are
live, so stamping them would be the false claim. It is also the one flag that puts a real
robot's pixels beside a simulated pose, which is why
[What is real when you ask for a real robot](#what-is-real-when-you-ask-for-a-real-robot)
spells out which half is which.

## Many robots, of more than one kind

The fleet table is the page's spine and the **Focus** selector is not a fleet control. Focus
decides which robot the motion pad and the checkpoint panels act on; **it never decides which
robot a stop reaches.** Every row carries its own stop, addressed at a named robot, because a
stop that depends on what is selected is a stop the operator has to think about while
something is moving.

Two measurements sit behind that, both taken against live drivers:

| | before | after |
| --- | --- | --- |
| STOP to a **different** robot, fired 1 s into a 5 s walk | 4.23 s | **0.06 s** |
| STOP to the **same** robot, mid-walk | 4.17 s | **0.07 s**, and it interrupts the walk |

The first was the dashboard: one mesh worker, so the stop queued behind the walk. The second
was deeper — **the edge runtime dispatches one RPC at a time per device**, so a motion
handler that blocks for five seconds makes the robot deaf to every command for five seconds.
Motion RPCs therefore return as soon as the nudge is *accepted*, and the measured outcome
arrives as a `motion_completed` event. See the module docstrings for the full argument.

`STOP ALL` uses `invoke_many`, not `broadcast`. Broadcast returns immediately and the replies
arrive separately, which is right for a fan-out whose outcome is advisory. A stop's outcome
is not advisory: the page reports **which robots confirmed and which did not**, because the
ones that did not are the ones you now have to walk over to.

⚠️ **A departed robot takes about 30 seconds to show as GONE**, measured by killing a driver
and watching the fleet: that is the D2D presence TTL deciding a peer has stopped announcing,
not the dashboard's 5 s poll. So the fleet table is a good record of what *has* left and a
poor alarm for what *just* left — if a robot stops responding, its commands fail long before
its row greys out.

**The fleet is grouped by platform — but not because platform is how a fleet is operated.**
Mid-incident what matters is which robot is *moving*, not who made it, and grouping by vendor
buries the one robot that needs attention inside a group of nine that do not. Platform earns
the grouping for a different reason: **the capability differences are per-platform.** Every
Lite3 shares "lie down does not lie it down"; every Go2 shares an unmeasured lateral floor.
Those belong once on a group header instead of repeated on every row. Within a group the sort
is operational — anything not live floats to the top.

⚠️ **The trap that comes with grouping is the scope of a bulk action.** The moment a fleet can
be filtered or grouped, "stop all" becomes ambiguous: an operator looking at a filtered list
reads it as "stop these", and is wrong in whichever direction the implementation chose.
Fleet-management practice is to preview exactly which devices an action will hit. So the
button **names its count** — `■ STOP ALL (4)` — and is never narrowed by grouping or
filtering; a group's own stop is a separate, separately-labelled button
(`Stop go2 (2)`). Both go through one implementation, so they cannot report differently, and
a malformed scope is treated as **everything** rather than nothing: for a stop, the fail-safe
direction is more robots, not fewer.

The fleet table **scrolls** rather than hiding anything, capped near three robot rows with a
toggle that says how many there are (`Show all 6 robots`). Group headers stay sticky while you
scroll, so you always know which platform you are inside. Nothing is removed from the DOM: a
robot is always one scroll from its own stop button, which is why this is a scroll cap and not
a "show first three".

Adding robots costs no extra polling. Pose, mode and armed checkpoint are folded out of the
`robot_state` events already streaming through the page; capabilities are fetched once per
robot and cached, since they cannot change while a driver is up.

## Starting the three by hand

`start-dashboard.sh` runs exactly these, and there is no fourth thing it does. Run them
yourself when you want a driver the launcher does not build — a Lite3, a second robot on
one page, or a real Go2 with no `--simulate`. Every command below is Python **≥ 3.11**, on
a workstation, from `dashboard/`.

```bash
python3.11 -m pip install device-connect-edge device-connect-agent-tools aiohttp numpy Pillow

# 1. the checkpoint source, standing in for the bucket. It WRITES sources.json.
python3.11 model_server.py --models-dir ../policy/models --port 8800 \
        --emit-sources /tmp/sources.json --label "Arm AGI CPU server"

# 2. a robot. Use a COPY of ../policy — arming a checkpoint rewrites its config.json.
python3.11 robot_driver.py --platform go2 --simulate --package /tmp/package \
        --model-sources /tmp/sources.json --camera-url http://192.168.123.18:8801/

# 3. the dashboard
python3.11 server.py --port 8080                  # then open http://127.0.0.1:8080
```

Drop `--camera-url` and you have the same thing with a synthetic camera and no robot at
all; drop `--simulate` too and `--platform sim` is the bench double, which is what to use
on a demo host with no hardware in the room.

**A working screen** has `MESH UP` in the top bar and the robot on a fleet row with a green
`LIVE` badge and a pose; `STOP ALL (1)` counts what it will hit. The checkpoint table lists
what is on the robot with one row marked `ARMED`. Under **Load from Cloud AI** the Source
picker reads *Arm AGI CPU server*, the address field is already filled in, and pressing
**Browse** returns a row saying `served by mappo-model-server` — that round trip is the
proof that the robot, not your browser, can reach the source. Press `E` for the event
drawer; a `walk_forward` should produce a `motion_started` line and then a
`motion_completed` line below it carrying `travelled_m` and `delivered_fraction`.

Add a second, differently-ruled robot to see the platform asymmetry on one page — a
simulated Go2 refuses `strafe_left 0.15` with its own measured 0.200 floor while the bench
double accepts it:

```bash
python3.11 robot_driver.py --platform sim --package /tmp/package \
        --model-sources /tmp/sources.json --allow-motion --device-id mappo-bench
```

### The routes, because prose about them has been wrong

`server.py` serves six, and a runbook written from memory guessed different ones. These are
the names in `create_app`:

| | |
| --- | --- |
| `GET /api/devices` | everything the mesh announces, with each device's function list |
| `GET /api/fleet` | one row per robot: pose, mode, armed checkpoint, capabilities, `age_s` |
| `POST /api/stop-all` | `{stopped: [...], failed: [...], matched: n}` — see the argument for `invoke_many` above |
| `GET /api/camera/{device_id}` | `multipart/x-mixed-replace`, `?fps=` clamped to 1–15 |
| `POST /api/invoke` | `{device_id, function, params}`, against an **allow-list**; anything else is a 403 that never reaches the mesh |
| `GET /api/events` | Server-Sent Events: the ring buffer, then everything new |

Measured against a live driver, so they are runnable and not remembered:

```bash
curl -s localhost:8080/api/fleet | python3 -m json.tool
curl -s -X POST localhost:8080/api/invoke -H 'Content-Type: application/json' \
     -d '{"device_id":"mappo-go2","function":"get_status"}'
curl -s -X POST localhost:8080/api/stop-all -H 'Content-Type: application/json' -d '{}'
```

### On a real robot, without the bench double

```bash
python3.11 robot_driver.py --platform go2 --package ../policy \
        --bridge-python /home/unitree/robotics-connect-envs/$USER/bin/python3 \
        --camera-url http://192.168.123.18:8801/
```

`--bridge-python` is the interpreter that can import `unitree_sdk2py`. It is **not** the one
running the driver, and getting it wrong makes every command fail with an import error —
`get_capabilities` reports the path it will use, so check it before you need it.

⚠️ **That interpreter is on the robot and this process is not**, so `--bridge-python` alone
is not enough: it has to name something on *this* machine that reaches the robot's 3.8 —
today an SSH wrapper, which is a workaround and not a supported deployment. Its limits are
real: `list_models`, `free_bytes` and `download_model` then act on the **workstation's**
package directory, so the checkpoint panel is answering about the wrong machine. Only
`get_status`, `get_capabilities` and the motion gate genuinely reach the robot. See
[`../evidence/2026-08-26-dashboard-local-trial/`](../evidence/2026-08-26-dashboard-local-trial/).

Then, with a clear area and an operator on the controller abort, add `--allow-motion`.
On a Lite3, stand the robot and enable high-level navigation mode on the vendor interface
first, then add `--operator-ready`.

To be able to start a run at all, add `--run-profile`:

```bash
python3.11 robot_driver.py --platform go2 --package ../policy \
        --bridge-python /home/unitree/robotics-connect-envs/$USER/bin/python3 \
        --camera-url http://192.168.123.18:8801/ \
        --run-profile run-profile.example.json
```

**Edit that file first.** Every path in it is a path on the robot; `launch_prefix` must
authenticate without a password (put a key on the robot — a password is refused at load,
because that field is published); and `heading_servo_flag` is a property of the tree it
points at rather than a preference, so run *that* tree's `mappo_drive.py --help` and look.
The driver prints both commands it would run at startup, which is the moment a wrong path is
still cheap. Without `--run-profile`, `start_run` refuses and `get_capabilities` reports
`run.supported: false` — a robot that cannot start a run has to be distinguishable from a
driver too old to have the field, and an absent key looks like both.

The default bind is loopback. Pass `--host 0.0.0.0` to reach it from the demo LAN, and note
what that means: **this dashboard has no login.** Anyone who can reach the port can drive
any robot on the mesh that was started with motion enabled.

⚠️ **`--host 127.0.0.1` on the model server is unreachable from a robot**, and the server
says so at startup. The download runs on the robot, so an address that only the laptop can
resolve gives a field that looks right and fails on fetch. Bind a LAN address for a real
robot.

See [`../evidence/2026-08-26-dashboard-local-trial/`](../evidence/2026-08-26-dashboard-local-trial/)
for a capture of exactly this, and for what it does and does not prove.

### When the bucket arrives, nothing here changes

A checkpoint source is a **base address in a file on the robot**, not a code path. The robot
advertises it and `cloud_models.parse_source` dispatches on its scheme, so today's local
server and tomorrow's bucket are the same feature configured twice:

```jsonc
{"sources": [
  {"label": "Arm AGI CPU server", "location": "the demo LAN",         // now
   "index_url": "http://192.168.123.50:8800/index.json"},
  {"label": "Cloud AI", "location": "eu-west-1",                      // when it exists
   "bucket": "mappo-checkpoints", "prefix": "go2/"}
]}
```

Both are listed at once and the operator picks by name. `list_cloud_models` already branches
on which key is present, `download_model` already routes `s3://` through `fetch_s3` and
`http(s)://` through `fetch_http`, and `ModelStore.install` inspects the bytes either way —
so adopting the bucket is an edit to that file plus `pip install boto3`, and the fallback
does not have to be removed to do it.

Per-model URLs in the index are **relative**, resolved by the client against the address it
fetched the index from, so the server never has to know its own public name. `--base-url`
overrides that only for a reverse proxy or a NAT hop, where the files are reachable at a
different address from the index.

## The four things worth knowing before you use it

**A swap takes effect on the next run, not on the one in progress.** `MappoController` loads
its weights once, at construction, so a live `mappo_drive` cannot have the network pulled out
from under it — not because anything checks, but because no code path exists. What the store
*does* check is that the checkpoint you are arming can actually run under the current config:
`lidar_range_vmas` must match the checkpoint's `training_lidar_range_vmas`, or the next run
dies at load. That refusal happens at the click instead.

**Every button is a measurement.** Commands hold a velocity for a fixed time and then stop,
and the result reports pose before, pose after, distance travelled, and the fraction of the
commanded speed that was delivered. Closed-loop distance control would have hidden exactly
the number this repository has spent the most time on — this Go2 delivers about **0.45** of
what it is commanded. A button that says "commanded 0.35 for 1.5 s, travelled 0.012 m"
diagnoses itself.

**The two platforms are not interchangeable, and the page says so per robot.** `lie_down` on
a Go2 issues `StandDown`; on a Lite3 it only *stops*, because posture there is
operator-controlled through the vendor app.

Speeds below a **measured** gait floor are refused: 0.35 m/s forward and 0.20 m/s lateral on
the **Go2**, both measured on that robot (issue #42's table — vy 0.15 travelled 0.010 m in
1.5 s, vy 0.20 walked 3 of 3). The Go2's **yaw** floor has never been measured, so a Go2 turn
is allowed and carries a warning on every press that it may produce no gait at all, which
would not be a fault.

**Nothing has been measured on a Lite3, on any axis**, and every motion command to one is
therefore refused rather than warned about. Neither Venture has moved under this stack; issue
[#13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)'s measurement boxes
are all still open, and its first instruction is *"do not copy values between units"*. Until
this PR the table gave the Lite3 the Go2's pair verbatim and told the Go2 its own lateral
floor did not exist. `--force` is the way past the refusal, and it says on the result that it
was forced.

**Reverse is open-loop into unobserved space.** Neither platform has rear sensing and the
planner never samples that direction; issue #40 caught the policy commanding it. The button
exists because it was asked for, it is capped at 2 s rather than 5, and it says so.

## Files

| | |
| --- | --- |
| `start-dashboard.sh` | The three processes as one command, with the refusals that stop a demo failing four steps later. `--dry-run`, `--help`. |
| `robot_driver.py` | The Device Connect device. 18 RPCs, 10 events (both counted from the file, and the mesh agrees: `/api/devices` reports `function_count: 18`). Runs on a **workstation**, Python ≥ 3.11 — see the diagram. |
| `drive_bridge.py` | The SDK-env worker: one command, one JSON line, exit — plus `pose-stream`, which does not exit. Python 3.8, stdlib only. |
| `peer_link.py` | The other direction: subscribe to peers' poses on the mesh and spool them for a Python 3.8 control loop. Runs on the robot, Python ≥ 3.11. |
| `model_store.py` | Checkpoints on disk: what is here, what is armed, what may replace it. |
| `cloud_models.py` | S3 and http(s) fetch, with the refusals that make a URL field on a web page safe. |
| `model_server.py` | The other end of that fetch: a directory of checkpoints served as a Cloud AI source, for a demo floor with no bucket. Stdlib only. |
| `camera_source.py` | Front-camera frames per platform — live, synthetic, an HTTP pull, or a labelled replay — with the ceilings and the who-is-watching lifecycle. |
| `go2_frame_server.py` | **Copied to the robot** and run there under its SDK environment: the Go2's `VideoClient` behind a read-only HTTP endpoint, so only the frame crosses the version wall. |
| `server.py` | The dashboard: discovery, an invoke allow-list, the SSE event fan-out, and the MJPEG stream. |
| `templates/`, `static/` | The page. It renders from `get_capabilities()`, so it never hard-codes what a robot can do. |

## Tests

```bash
for t in test_*.py; do python3.11 $t; done    # 3.11, not python3 — see above
ruff check .                                  # must be clean
```

Needs `device-connect-edge`, `device-connect-agent-tools`, `aiohttp` and `numpy`; `boto3`
only for the S3 path, and `Pillow` only for the sim camera. `test_start_dashboard.py` needs
none of them: it drives `start-dashboard.sh` against a **stub interpreter** — a real Python
with `sys.version_info` overwritten, a meta-path finder that refuses named modules, and a
directory of stand-in modules on `sys.path` that satisfy the rest — so both "a 3.9
interpreter is refused" and "a complete one is accepted" are conditions the test supplies
rather than ones the machine happens to have. A test that asserted the first with the local
`python3` would pass on a Mac and assert nothing on a runner whose `python3` is 3.11; and
**the second half was learned the hard way** — without the stand-ins the happy-path tests
passed on a laptop with Device Connect installed and failed on CI, which does not have it.
A stub that can only take modules away still lets the environment decide the answer. Its
one end-to-end test is the teardown: three real children, a real SIGTERM, and the process
table read afterwards.
`test_drive_bridge.py`,
`test_model_store.py`, `test_peer_link.py` and `test_model_server.py` run without the
Device Connect packages — and two of them are end-to-end against the real counterpart
rather than against a second opinion about the format. `test_peer_link.py` runs the real
spooler, the real spool file and the real reader in `../integration/peer_source.py`;
`test_model_server.py` starts a real server and reads it with the real
`cloud_models.list_http_index` and `cloud_models.fetch`, then hands the bytes to
`model_store.inspect_model`, so "the transfer completed" and "what arrived is drivable" are
separate assertions.

⚠️ **`device-connect-edge` and `device-connect-agent-tools` are now on PyPI** — 0.2.5,
installed with a plain `pip install`. `AGENTS.md` and `.github/measure-suites.sh` still say
the edge package is not, and the latter therefore skips `test_robot_driver.py`; its 35 tests
pass and are **not** in the inventory's count for this directory. `arm_dc_robotkit` really
is still absent from PyPI.

**Twenty-four guards are mutation-tested rather than assumed.** Seven of them are this
PR's, on `start-dashboard.sh`, each removed one at a time with the patch verified to have
applied first: the ≥ 3.11 gate, the `aiohttp`/`numpy` half of the package check, an
`ALLOW_MOTION` that reads the environment, the port pre-flight, `--robot` no longer implying
`--simulate`, `cleanup` killing nothing, and the single combined `trap cleanup EXIT INT TERM`
that stops all three correctly and then resumes its own watchdog. One survived on the first
attempt — removing only the `kill -TERM` left the later `kill -KILL` to do the job, so the
children still died and "they all stopped" could not tell the difference. **SIGTERM before
SIGKILL is a safety property, not a tidiness one** (`robot_driver.py`'s motion worker damps
on SIGTERM, and `SportClient.Move` has no dead-man timeout), so the test now uses a child
that records the signal it got, and that mutation is caught. The other seventeen are the ten
in
[`../evidence/2026-08-21-device-connect-dashboard/`](../evidence/2026-08-21-device-connect-dashboard/),
which also records the two defects the bring-up run found, and seven in
[`../evidence/2026-08-26-dashboard-local-trial/`](../evidence/2026-08-26-dashboard-local-trial/),
which records the first hardware contact — **read-only, no robot moved** — and the defect it
exposed: the event stream orders a batch by the emitting device's own clock, and the Go2 it
was run against had no working RTC and reported 1970. **No robot has moved under this yet.**

## Two traps, written down because both cost real time

**"Hidden" must outrank every component rule.** `.hidden { display: none }` and
`.safety { display: flex }` have the same specificity, so whichever is declared later wins —
give a component a `display` and you silently disable hiding on it, everywhere, not just at
the control that looks broken. The `hidden` *attribute* fails from the other side: the
browser's `[hidden] { display: none }` is a user-agent rule and any author rule beats it.
Both are `!important` at the end of the stylesheet, and `test_stylesheet.py` keeps them
there. This cost the safety banner its X *and* its ability to hide for a motion-disabled
robot — the second half went unnoticed because nobody was looking for it.

**Do not name an `@rpc` after anything on `DeviceDriver`.** `capabilities`, `status`,
`identity`, `invoke`, `events`, `functions`, `connect`, `registry`, `router`, `transport`
and about twenty more are live mines. `capabilities` is a property the runtime reads to build
the presence announcement; override it and the device **never appears on the mesh**, with no
exception and no log line, looking exactly like a network fault.
`test_no_rpc_shadows_a_base_class_member` now catches this.

**A blocking RPC handler makes the robot deaf.** `DeviceRuntime._cmd_subscription` awaits
the driver call inline, one message at a time per device. Anything that takes seconds must
return immediately and report its outcome as an event, or `stop` cannot be delivered while it
runs. And a delivered stop is not enough on its own — the worker refreshes velocity at 10 Hz,
so `stop` also terminates the in-flight worker (SIGTERM, so its `SafeStop` damps).

**Events arrive grouped by type, not by time.** A `Subscription` keeps one inbox per subject
and each event name is a subject, so a drained batch must be re-sorted before anyone reads
it — otherwise a completion appears above the start that caused it. `server._in_time_order`
does that, by the device's own `ts`, which has one-second resolution and therefore cannot
order two events inside the same second.

## Why not the Device Connect portal

`device-connect-server` ships a full multi-tenant portal with a devices page, a generic
invoke form and an event stream, and for fleet administration it is the right tool. It is
generic by design — a JSON form per function — which is correct for forty device types and
wrong for a motion pad, where the useful property is that "strafe left" is one key press
away from "stop". This page is demo-specific UI and belongs in the demo repository.

They are not exclusive: `connect()` speaks D2D or a router transparently, so a robot
registered with a full server appears here as readily as one announcing itself by multicast.
