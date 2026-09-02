<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md

Standing rules for any coding agent working in this repository — Codex, Qwen Code, Claude
Code, or otherwise. If your tool does not load this file automatically, paste it into your
first prompt.

Human-facing companion: **[`CODING-AGENT-GUIDELINES.md`](CODING-AGENT-GUIDELINES.md)** —
how we prompt, and why. Read it once.

## What this repo is

A MAPPO policy, trained in simulation, driving real quadrupeds to goals in a shared room.
`robot-stack/` perceives and moves, `integration/` maps its telemetry into the policy's
input, `evidence/` holds the runs that prove it. Start with `README.md`; it is accurate and
current, and it documents three mappings that are *not* the obvious ones.

## Workflow

1. **Work starts from a GitHub issue.** If there isn't one, write one before writing code.
2. **Work ends with a continuation comment on that issue** — outcome numbers, what the run
   found, what is still open, and the issue it continues in. Do this even when stopping
   mid-task; the next session is a fresh context and this comment is the only handover.
3. **A PR carries evidence**: a run log, a test count, a measurement table, or a video.
   "It should work" is not a status. Update `README.md`'s Status table if the PR changes it.
4. **Report failures plainly.** If a test fails, say so with the output. If a step was
   skipped, say that.

## Before you say you are done

Every suite prints one `  ok  <name>` line per test and a `<name>: N/N passed` summary.
Run everything, from the repository root, with the same script CI runs:

```bash
bash .github/measure-suites.sh            # run every suite
bash .github/measure-suites.sh --write    # ...and refresh .github/test-inventory.tsv
bash .github/measure-suites.sh --check    # ...and fail if it disagrees — what CI runs
```

`--write` and `--check` need **Python >= 3.11** with `numpy`, `opencv-python`, `Pillow`,
`pytest` and `aiohttp` importable, and refuse to run otherwise rather than writing an
inventory that is short by however many suites the interpreter could not reach. Plain
`bash .github/measure-suites.sh` runs on 3.8 and still fails on a broken suite; it just
does not touch the counts. Or run one suite at a time — the parentheses are load-bearing,
because the directories are nested and a bare `cd` would run the next line from inside the
last one:

```bash
(cd policy && for t in test_*.py; do python3 "$t"; done)
(cd integration && for t in test_*.py; do python3 "$t"; done)
(cd detector/labels && for t in test_*.py; do python3 "$t"; done)
(cd detector/labels/pipeline && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/unitree/go2/visual_nav && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/unitree/go2/controller && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/unitree/go2/d1_arm && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/unitree/go2/lidar_sight && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/deep_robotics/lite3/locomotion && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/deep_robotics/lite3/visual_nav && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/deep_robotics/lite3/commissioning && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/preflight && for t in test_*.py; do python3 "$t"; done)
(cd dashboard && for t in test_*.py; do python3 "$t"; done)
```

**There are no counts in this file, and putting one back fails the build.** They are in
`.github/test-inventory.tsv`, one line per directory and no total, because the counts were
here until they made *this* file a merge conflict on every change that added a test — two
collided on it in one night and one had to hand its count diff to the other to apply. A
generated file only helps if you can regenerate it, which is what `--write` above is for;
never hand-edit a number into it.

**CI enforces both halves rather than trusting either.**
[`.github/workflows/offline-checks.yml`](.github/workflows/offline-checks.yml) discovers
every `test_*.py` and every `ruff.toml` by globbing, re-measures every count in the
inventory and fails on any disagreement in either direction; and separately checks that the
list of `(cd …)` lines above and the list of directories in the inventory are the same set,
so a suite that runs and is not documented here is an error, and so is a line here that
runs nothing. That list changes when a *directory* appears, not when a test does, which is
why it can live in prose and the numbers cannot.

Then `ruff check .` from inside **every** directory that holds a `ruff.toml`. List them
rather than trusting a number in this sentence — the number that used to be here said
"thirteen" while the tree held sixteen, and the job that forbids counts in this file only
inspects fenced blocks, so a spelled-out one in prose was invisible to it:

```bash
git ls-files '*ruff.toml' | xargs -n1 dirname   # every directory to run it from
```

Running one directory's config against another directory's code is how a PR came to report
"ruff clean" while shipping 13 findings.

**A `ruff.toml` given with `--config` is resolved against the directory you are standing
in, not the directory it lives in.** ruff anchors isort's `src` at the project root, and
under `--config` the project root is your current directory (under plain discovery it is
the config's own directory). So a sibling module is first-party from inside a subdirectory
and third-party from above it, and one config file returns opposite verdicts:
`cd detector/labels/pipeline && ruff check . --config ../ruff.toml` reported eight `I001`
findings that neither documented command produces. `known-third-party` pins the module by
name; CI now runs every config from every directory it governs, and from the repository
root, and fails if the three disagree — because that is a verdict a human gets and CI's own
lint loop, which uses discovery, does not.

Eight directories still hold Python that no `ruff.toml` covers at all: five Go2 directories
beside `visual_nav`, and three `evidence/` run directories. CI names each in a warning and
fails if a ninth appears — the count can shrink, and cannot grow unnoticed. The workflow
comment beside the ratchet prices each of the eight: 41 findings between them, which is why
they are still warnings and not gates, and none of them is vendored, generated or
third-party.

### What is not counted, and why

- `robot-stack/unitree/go2/deploy/test_go2_robot_io.py` imports `arm_dc_robotkit` and
  `dashboard/test_robot_driver.py` imports `device_connect_edge`. Neither package is on
  PyPI before launch, so both die at `ModuleNotFoundError` rather than at a test, and both
  fail that way on `main` today. CI skips them and says so with a `::warning::`. **A
  missing dependency is not a pass and is not a regression — install it or say so.**
- `dashboard/`'s other suites need **Python >= 3.11**, which is what Device Connect requires,
  and that is why `dashboard/drive_bridge.py` is a separate Python 3.8 process rather than
  an import. CI runs a `3.8` leg and a `3.11` leg for exactly this reason: the Go2's Jetson
  is Ubuntu 20.04 / JetPack 5, so 3.8 is what the robot code has to import under, and the
  `3.8` leg skips `dashboard/` apart from `test_drive_bridge.py`.
- Installing `Pillow` is not optional if you want the real number: three tests in
  `dashboard/test_camera_source.py` print `  skip  ` and then `  ok  ` without it, which is
  a missing dependency reading as a pass. `numpy`, `opencv-python` and `pytest` are needed
  the same way — without `cv2` several `visual_nav` files fail at import.
- The block previously read `33 / 189 / 329 / 17 / 39 / 16 / 139` and did not mention
  `detector/labels`, `detector/labels/pipeline` or three of the Go2 directories at all. Its
  Lite3 commissioning line named `test_lite3_state_probe.py` by name, in a directory that
  holds ten test files.

**`ruff --fix` sorts imports and will hoist a `from avoidance import ...` above the
`sys.path` line that makes it importable.** Two test files went from passing to
`ModuleNotFoundError` that way, with nobody touching a test. Put every `sys.path.insert`
in one block before any sibling import, and re-run the suites after a lint fix.

Then do an adversarial pass over your own diff: software engineering best practices,
mistakes, sloppy or brute force algorithms, inconsistent style, incorrect comments. Report
the findings; do not silently fix and move on.

Ask what would make each new test *fail*, and confirm it can. This repo has already shipped
a latch check that proved the arm was held by asserting its joints had stopped moving — an
unpowered arm is perfectly still, so the check could never fail.

## Hardware

- **`robot-stack/SAFETY.md` governs anything that moves a leg.** It is not optional.
- `--live` is the only flag that moves the robot. An operator stays on the remote.
- **Ask the human before any walking.** Sensors and the back-mounted arm do not need
  permission; legs always do.
- The human is your only sensor for the room. Ask where the goal and the obstacles are, and
  ask them to confirm the lane is clear, before a run.
- Return absolute filepaths for any recording you produce, so they can be opened.

⛔ **A network change on a Lite3 can take the operator's remote away, and nothing reports
it.** The remote is served by an access point on the robot's own `p2p0`. Both robots
shipped with `connection.autoconnect: no` on that profile, and
`/etc/netplan/config.yaml` uses `renderer: NetworkManager` — so **`netplan apply`
deactivates the AP and it does not come back**. The robot stays reachable over Ethernet and
WiFi, every service stays `active`, and the only symptom is that the controller finds no
SSID, which reads as a radio fault rather than as a consequence of the address change you
made an hour earlier. It cost an hour on each robot, a day apart.

Before and after any change to `netplan`, `nmcli`, `wlan0` or an address, run this and
compare — `p2p0` must say **`type AP`**:

```bash
iw dev | grep -E "Interface|type|ssid|channel"
```

If the AP is down, restore it with band and channel in **one** `nmcli` call — nmcli
validates the whole connection, so changing one at a time is rejected with a misleading
`'36' is not a valid channel`:

```bash
sudo nmcli con mod myap50G 802-11-wireless.band bg 802-11-wireless.channel 10
sudo nmcli con mod myap50G connection.autoconnect yes connection.autoconnect-priority 60
sudo nmcli con up myap50G
```

The channel must match the venue router's fixed channel: one radio serves both the AP and
the station, and the driver allows `#channels <= 1`. Full detail, including why a
2.4/5 GHz split is refused, is in
[`robot-stack/deep_robotics/lite3/DEMO-NETWORK.md`](robot-stack/deep_robotics/lite3/DEMO-NETWORK.md).

## Never install anything on a robot

You are the reader this rule is written for. An operator following a runbook is *less*
likely to make this mess than an agent improvising against an `ImportError` at 11pm, and
two agents were SSH'd into a live robot on 2026-08-26.

- **Never `pip install` on a robot outside a virtualenv, and never into the system Python.**
  The vendor stack lives inside a venv on these machines; the system interpreter is shared
  by every other user, every vendor tool and every ROS node on the robot, and no
  `uninstall` puts a shadowed vendor package back. A venv can be deleted and rebuilt.
- **If an import fails on a robot, that is a finding to report, not a dependency to add.**
  Activate the venv and re-run the same command. If it still fails inside the venv, say so
  in the issue, with the output. Do not make it go away.
- **Never install a newer Python on a robot, and never reach for a virtualenv to get one.**
  A venv is built *from* an interpreter and cannot supply a version the machine does not
  have. `device-connect-edge` runs off-robot for exactly this reason, and the split is
  deliberate rather than a packaging bug waiting to be fixed.
- **This now refuses rather than warns.** `visual_nav --live` and `dashboard/drive_bridge.py`
  call [`robot-stack/preflight/venv_guard.py`](robot-stack/preflight/venv_guard.py) before
  they open a transport, and it raises rather than printing. Do not route around it, and do
  not set `MAPPO_ROBOT_HOST=0` on a machine that is a robot.
- **Deploy with `deploy/push-to-robot.sh`, and a run will name its own commit.** None of
  the deployed trees is a git checkout — no `.git`, so no branch and no commit — so
  `robot-stack/preflight/tree_stamp.py` records git's own root tree id, recomputed from the
  bytes on disk, and `mappo_drive.py` refuses when the tree stops matching it. Every run
  prints `commit … tree …` as its first line; quote that id, not "deployed from main".
  ⚠️ **An older claim here was wrong and is worth knowing about**: `~/mappo-run` was said to
  match *no single commit*. It matches `cb42b9a` exactly, 226/226 files — the tip of an
  **unmerged branch**, which is why reconstructing it against `main` found nothing. The tree
  that really is a mixture is `~/mappo-main`, spanning 21 commits and 8–83 behind HEAD, and
  it is the one the launch wrappers source. Re-derive rather than trusting either: `python3
  robot-stack/preflight/tree_stamp.py id <dir>` prints a real git tree id for any directory.

The paths, the interpreters, the venv-creation command and the measurements behind all of
this are in **[`deploy/README.md`](deploy/README.md)** (Go2, and the Device Connect split)
and **[`robot-stack/deep_robotics/lite3/DEPLOYMENT-SOP.md`](robot-stack/deep_robotics/lite3/DEPLOYMENT-SOP.md)**
(Lite3, bilingual). They are not repeated here — this section is what you must not do, those
are how to do it.

## Never commit, log, or put in an issue

Robot SSH passwords, WiFi PSKs, API tokens. Reference a local untracked file instead
(`~/.robot-creds`). Scrubbing an issue after the fact does not remove it from the event log.

Do not link `?token=` asset URLs — they expire. Merge first, then link the permanent raw
URL on the default branch.

## Naming

This repository is shared with **Deep Robotics** and with Arm's China teams — and it is
shared *before* the underlying standard launches publicly.

- The product is **Arm Device Connect**, or **Device Connect** where "Arm" is redundant or
  verbose. The abbreviation is **DC**.
- **No earlier internal name for it, and no originating company name, may appear anywhere in
  this repository** — not in code, comments, docs, filenames, branch names, commit messages,
  or issue and PR text. If you find one, remove it and say so. Do not assume it was left
  deliberately, and do not reintroduce one by copying text in from an upstream repository.
- Ask @armwaheed rather than guessing. This rule is absolute; it is not a style preference.

Keep measurement tables self-contained: embedded GIFs and `user-attachments` video links
are slow or unreachable from mainland China, so the argument must survive without them.

## Style

Match the surrounding code and prose. `README.md` sets the register for docs: specific,
measured, with the counter-evidence stated. Prefer a number to an adjective.
