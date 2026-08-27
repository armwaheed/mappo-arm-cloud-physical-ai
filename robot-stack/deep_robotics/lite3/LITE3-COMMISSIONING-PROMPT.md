<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prompt: commission a Lite3 Venture / 提示词：Lite3 Venture 标定

**EN** — Paste the block below into your coding agent (VS Code Copilot, Codex, Claude Code…).
It drives `commissioning/commission.py` through its four stages and stops at each gate.

**中文** —— 把下面整段粘贴给你的编程助手。它会驱动 `commissioning/commission.py`
走完四个阶段，并在每个关卡停下等你确认。

**Read first / 请先阅读**: [`LIVE-RUN-RUNBOOK.md`](LIVE-RUN-RUNBOOK.md) §0.

---

```text
You are commissioning a Deep Robotics Lite3 Venture so that a live MAPPO run can start.
Nothing on this robot has ever been measured, so a live run currently REFUSES. Your job is
to produce the measurements, not to make the robot walk far.

WORK IN robot-stack/deep_robotics/lite3/commissioning/ AND READ commission.py FIRST.
Its module docstring is the specification. Do not write a new script; this one exists and
its four-stage order is a safety argument, not a convenience.

RULES
R1. Run everything inside a virtualenv on the robot. Do not pip install into the system
    Python. See AGENTS.md.
R2. Do NOT reorder the stages. Read-only probes, then a tape measure, then a camera, and
    only then anything that walks. The gait floor is measured BEFORE the actuator gain,
    because a gain fitted across the floor is dragged down by every sub-floor point.
R3. Before any stage that moves the robot: lane clear, robot standing and in navigation
    mode on the vendor app, a human with a hand on the emergency stop. Ask the operator to
    confirm out loud. If you cannot confirm, stop and say so.
R4. Motor temperatures are an OPEN QUESTION on this platform. Keep every moving run bounded.
    Let the robot cool between runs. Stop if a motor smells hot or the gait changes.
R5. If a command refuses, READ THE REFUSAL. It names what is missing. Do not add --force to
    get past a gate; report the refusal and ask.

STAGE 0 — ask what is missing, before touching the robot
    python3 commission.py --record <artefact>.json --emit-flags
  On an incomplete record this refuses and names the gaps. It moves nothing. Do this first
  and report the list.

STAGE 1 — nothing moves
  Read-only probes, tape measure, marker calibration. Measured values for this robot:
  camera lens height 0.37 m standing, 0.115 m prone; camera pitch fixed at about 11 deg.
  Use the real tape measurements for --front/--back/--left/--right; do not copy the
  example numbers from the docstring.

STAGE 2 — the two that walk
  Gait floor, then actuator gain. Requires --live --operator-ready and a clear lane.
  State --envelope-vx explicitly; unset, the envelope inherits a Go2's numbers.

STAGE 3 — a human signs
    python3 commission.py --record <artefact>.json --review '<your name>'
  --emit-flags refuses a provisional record. A number that has been measured and a number
  that has been believed are different things, and only a person turns one into the other.

STAGE 4 — produce the flags
    python3 commission.py --record <artefact>.json --emit-flags
  This is the only supported path to a live run's --gait-floor / --actuator-gain /
  --robot-radius. Report the flags verbatim.

STAGE 5 — a shadow run before anything walks a goal
  Run the navigator WITHOUT --live and confirm that `decision` and `transport` appear in
  the telemetry. Report the first tick of each.

THEN STOP. Do not attempt --execution-supervisor turn-drive. Report what you measured, what
refused, and anything that surprised you.

ALSO WANTED, and it is a real open question rather than a checkbox:
  motor_temperature_probe.py exists. Use it to find this robot's real thermal limits from
  BOUNDED runs -- start cold, run, record the curve, let it cool, repeat. Report idle
  temperature, rate of rise under load, and whether a thermal channel exists at all. If the
  platform exposes none, THAT IS THE FINDING; say so explicitly rather than reporting
  nothing. Do not find the ceiling by reaching it. For scale, the Go2 aborts at 70 C, warns
  at 55 C and idles near 30 C, and its rear leg motors collapse if left standing.
```

---

## What this unlocks / 这一步解锁了什么

**EN** — `--execution-supervisor turn-drive`, the avoidance path merged in PR #150, **cannot
be exercised on hardware at all** until `measured_m_s` exists. Not "unverified" — unreachable.
Everything else waits on these four numbers.

**中文** —— PR #150 合入的避障路径 `--execution-supervisor turn-drive` 在拿到 `measured_m_s`
之前**根本无法在实机上运行**（不是"未验证"，而是无法到达）。其余工作都在等这几个测量值。

## What is NOT ready / 尚未就绪

The Lite3 detector retraining finished and **recommended nothing**. At 224 px — the size the
peer launcher actually uses — no checkpoint both detects a Lite3 and keeps its people; every
augmentation step traded one for the other. Keep using the shipped detector.
See [`evidence/2026-08-27-lite3-training-set/`](../../../evidence/2026-08-27-lite3-training-set/).

Lite3 检测器再训练已完成，但**没有可推荐的模型**。请继续使用现有检测器。
