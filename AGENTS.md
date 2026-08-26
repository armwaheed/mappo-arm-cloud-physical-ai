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
The counts below are those `ok` lines, measured on `main` at `526f0b0` on 2026-08-26.
Run each line from the repository root; the parentheses are load-bearing, because the
directories are nested and a bare `cd` would run the next line from inside the last one.

```bash
(cd policy && for t in test_*.py; do python3 "$t"; done)                                         #   33
(cd integration && for t in test_*.py; do python3 "$t"; done)                                    #  196
(cd detector/labels && for t in test_*.py; do python3 "$t"; done)                                #   13
(cd detector/labels/pipeline && for t in test_*.py; do python3 "$t"; done)                       #   10
(cd robot-stack/unitree/go2/visual_nav && for t in test_*.py; do python3 "$t"; done)             #  329
(cd robot-stack/unitree/go2/controller && for t in test_*.py; do python3 "$t"; done)             #    6
(cd robot-stack/unitree/go2/d1_arm && for t in test_*.py; do python3 "$t"; done)                 #   15
(cd robot-stack/unitree/go2/lidar_sight && for t in test_*.py; do python3 "$t"; done)            #    7
(cd robot-stack/deep_robotics/lite3/locomotion && for t in test_*.py; do python3 "$t"; done)     #   60
(cd robot-stack/deep_robotics/lite3/visual_nav && for t in test_*.py; do python3 "$t"; done)     #   58
(cd robot-stack/deep_robotics/lite3/commissioning && for t in test_*.py; do python3 "$t"; done)  #  188
(cd dashboard && for t in test_*.py; do python3 "$t"; done)                                      #  110
#                                                                                          total 1025
```

Then `ruff check .` from inside **every** directory that holds a `ruff.toml` — there are
ten, and running one directory's config against another directory's code is how a PR came
to report "ruff clean" while shipping 13 findings. Twelve directories still hold Python
that no `ruff.toml` covers at all: the five Go2 directories beside `visual_nav`, both of
`deploy/`, and five `evidence/` run directories. CI names each of them in a warning, and
fails if a thirteenth appears — the count can shrink, and cannot grow unnoticed.

**CI enforces this block rather than trusting it.**
[`.github/workflows/offline-checks.yml`](.github/workflows/offline-checks.yml) discovers
every `test_*.py` and every `ruff.toml` in the tree by globbing, re-measures each number
above, and fails if a number here and a number it measured disagree — in either direction,
including a directory of tests that this block does not list. Do not edit a count here to
make CI pass; the count is the measurement.

### What is not counted, and why

- `robot-stack/unitree/go2/deploy/test_go2_robot_io.py` imports `arm_dc_robotkit` and
  `dashboard/test_robot_driver.py` imports `device_connect_edge`. Neither package is on
  PyPI before launch, so both die at `ModuleNotFoundError` rather than at a test, and both
  fail that way on `main` today. CI skips them and says so with a `::warning::`. **A
  missing dependency is not a pass and is not a regression — install it or say so.**
- `dashboard/`'s other 110 need **Python >= 3.11**, which is what Device Connect requires,
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
