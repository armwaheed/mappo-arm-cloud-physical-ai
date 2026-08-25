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

```bash
cd policy      && python3 test_physical_ai_mappo.py                                #  33
cd integration && for t in test_*.py; do python3 $t; done                          # 189
cd robot-stack/unitree/go2/visual_nav && for t in test_*.py; do python3 $t; done   # 265
cd robot-stack/deep_robotics/lite3/locomotion && for t in test_*.py; do python3 $t; done #  17
cd robot-stack/deep_robotics/lite3/visual_nav && for t in test_*.py; do python3 $t; done # 39
cd robot-stack/deep_robotics/lite3/commissioning && python3 test_lite3_state_probe.py # 16
cd dashboard   && for t in test_*.py; do python3 $t; done                          # 139
ruff check .        # must be clean in each code directory above; each has a ruff.toml
```

`policy/` and most of `integration/` need `numpy`. The `visual_nav` suite also needs
`opencv-python`: without `cv2`, several files fail at import and the suite is incomplete.
That is a missing dependency, not a regression — install it or say so explicitly.

`dashboard/` needs `device-connect-edge`, `device-connect-agent-tools` and `aiohttp`, in a
**Python >= 3.11** environment — that is what Device Connect requires, and it is why
`dashboard/drive_bridge.py` is a separate Python 3.8 process rather than an import.
`test_drive_bridge.py`, `test_model_store.py` and `test_peer_link.py` run without any of them.

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
