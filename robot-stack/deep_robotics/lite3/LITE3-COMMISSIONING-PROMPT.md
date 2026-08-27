<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lite3 commissioning — start here / Lite3 标定 —— 从这里开始

**For: Timo Tang and the Arm Shanghai team. / 致：唐天目与 Arm 上海团队。**

---

## 1. Why this page exists / 本页的目的

**EN** — The Lite3 avoidance work is merged (PR #150, rebased and green). **One thing blocks
it on hardware, and it is yours to unblock**: `axis_primitive_probe.py` has never run on a
Lite3. Without `measured_m_s` a live axis run **refuses**. So
`--execution-supervisor turn-drive` is not "unverified" — it is **unreachable** until those
numbers exist. Everything else waits on them.

**中文** —— Lite3 避障功能已合入（PR #150，已 rebase 且 CI 通过）。**目前只有一件事阻塞实机运行，
需要你来解除**：`axis_primitive_probe.py` 从未在 Lite3 上运行过。没有 `measured_m_s`，
实机 axis 运行会被**直接拒绝**。因此 `--execution-supervisor turn-drive` 不是"未验证"，
而是**根本无法到达**。其他工作都在等这几个数。

## 2. You do not need to type nine flags by hand / 不需要手动输入九个参数

**`commissioning/commission.py` measures them all and prints them.** Start with the command
below — **it moves nothing** and tells you exactly what is missing:

```bash
cd robot-stack/deep_robotics/lite3/commissioning
python3 commission.py --record <artefact>.json --emit-flags
```

On an incomplete record it refuses and names the gaps. Cheapest step in the sequence.
该命令不会让机器人移动，会直接列出缺失项。

## 3. What is NOT ready / 尚未就绪的部分

**The detector retraining from your videos recommended nothing.** At 224 px — the size
`deploy/run-peer-supervised.sh` actually launches — no checkpoint both detects a Lite3 and
keeps its people. Every augmentation step traded one for the other (21 → 7 → 1 people
retained). **Keep using the shipped detector.** Full working:
[`evidence/2026-08-27-lite3-training-set/`](../../../evidence/2026-08-27-lite3-training-set/)

**中文** —— 基于你视频的检测器再训练**没有产出可推荐的模型**：在生产实际使用的 224 px 下，
没有任何检查点能同时检测到 Lite3 又不丢失人的检测。**请继续使用现有检测器。**

## 4. Your recordings, and one thing that would improve them / 关于你的录像

**They were excellent** — raw pixels, six scenes across subject and lighting, telemetry
attached, and `--record-raw` used correctly. That is exactly what was needed, and it is a
large step up from the earlier clip.

**One improvement**: all six are **tripod shots** — 0.0–1.0 px median camera displacement.
The *subject* varies in distance and angle; the *camera* never moves. So 5,854 frames
contain **456 distinct views**. Moving the camera between takes would multiply that at no
extra cost in time.

**中文** —— 录像质量很好：原始像素、六个场景涵盖不同对象与光照、附带遥测，且正确使用了
`--record-raw`。**一点改进建议**：六段都是三脚架固定拍摄，主体在变、相机不动，
因此 5,854 帧只包含 **456 个不同视角**。在拍摄之间移动相机可以成倍增加视角数量，且不增加时间成本。

## 5. One small ask / 一个小请求

If the telemetry `.jsonl` for the earlier 60-second clip
(`lite3-pov-20260827T024720Z-60s.mp4`) is still on the robot, please send it — it may let us
label frames we already hold, without you recording anything.
如果那段 60 秒视频对应的遥测 `.jsonl` 还在机器人上，请一并发送。

---

## 6. The prompt — paste this into your coding agent / 提示词：粘贴给你的编程助手

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

# Reference / 参考资料

*Everything below is for when something refuses. You should not need it to start.
以下内容仅在出现拒绝时查阅，开始时不需要。*

## 7. Before anything walks / 运行前

1. **Shadow first.** Confirm `decision` and `transport` appear in telemetry before a live run.
   先跑 shadow，确认遥测中出现 `decision` 与 `transport`。
2. **Cordon the lane.** A colleague walking past has not accepted this risk; you have.
   请隔离通道 —— 路过的同事并未同意承担这个风险。
3. **Emergency stop in hand.** Stop if a motor smells hot or the gait changes.
   手持急停；若电机有异味或步态改变，立即停止。
4. `--operator-ready` is typed **last**, after STANDING and navigation mode.

## 8. The nine inputs a live run requires / 实机运行所需的九项输入

Produced by `commission.py --emit-flags`. This table is for reading a refusal, not for typing.

| flag | where the value comes from / 数值来源 |
|---|---|
| `--calibration` | This Lite3's camera. ⚠️ See §11 — the file in circulation is not self-consistent. |
| `--gait-floor` | **Measured**, `axis_primitive_probe.py`. Not the Go2's 0.35. |
| `--actuator-gain` | **Measured at this envelope** — fit against pose, not the velocity estimate. |
| `--robot-radius` | **0.40** — the robot. ⚠️ Not 0.33: that is the *obstacle box*. Must satisfy `--policy-scale = radius / 0.10`, so 0.40 pairs with 4.0. |
| `--max-vx` | Stated. Unset, it silently inherits **the Go2's**. |
| `--max-vy` | Stated. This Lite3's lateral primitive is unmeasured, so `0` is honest. |
| `--max-wz` | Stated. Unset, inherits the Go2's. |
| `--operator-ready` | Typed **after** STANDING and navigation mode on the vendor app. |
| `--axis-profile` | Only with `--locomotion-transport axis`. Needs evidenced primitives. |

Also required and not a flag: **a virtualenv**. 另需在 **venv** 中运行。

**Why the envelope cannot be defaulted**: it is the right-hand side of a safety gate.
`_validate_axis_profile_speeds` refuses a primitive whose `measured_m_s` exceeds
`--max-vx x --derate`. Against a borrowed right-hand side that comparison is arithmetic,
not a gate. 包络值是安全门限的右侧；借用别的机器人的数值，这个比较就只是算术。

## 9. Refusal decoder / 拒绝信息对照表

| what you see | what it means | what to do |
|---|---|---|
| `REFUSING TO WALK: missing ...` | One or more of the nine | Read the list — it names each and why |
| `REFUSING TO WALK: --accept-no-motor-temperatures needs --max-seconds set to 120s or less` | No thermal feed **and** an unbounded run | Add `--max-seconds 120` or lower |
| `TOP SPEED x IS BELOW ... GAIT FLOOR y` | ⚠️ **Warning, not a refusal.** The run continues and the robot may not move, reporting no fault | Raise the envelope or command scale |
| exits ~3 s, no banner, **rc 139** | `Segmentation fault` — the SDK env was not sourced | `source .../install/setup_env.sh` first |
| `[tree-stamp] no stamp` | Off-robot a finding; **on a robot a refusal** | Deploy with `deploy/push-to-robot.sh` |
| `ModuleNotFoundError: ros2_twist_locomotion` | On the ROS 2 path | `--locomotion-transport axis` (or `udp`) |
| `axis profile lacks evidenced primitives for lateral_*` | Lateral was never measured | Expected — use `--max-vy 0` |

**A refusal is the system working.** Every one exists because of a measured failure.
每一条拒绝都源自一次实测到的故障。

## 10. Known-good command shape / 已验证的命令形状

⚠️ **Values are placeholders — use what `--emit-flags` gives you.** This shows which flags
must be present. 数值为占位符，请使用 `--emit-flags` 的输出。

```bash
source /path/to/robot-stack/deep_robotics/lite3/install/setup_env.sh
# in a venv -- see AGENTS.md

python3 mappo_drive.py --live \
  --policy-mode supervised --policy-scale 4.0 \
  --calibration  lite3_front_camera.json \
  --gait-floor   <measured> \
  --actuator-gain <measured at this envelope> \
  --robot-radius 0.40 \
  --max-vx <stated> --max-vy 0 --max-wz <stated> \
  --max-seconds 120 --accept-no-motor-temperatures \
  --operator-ready \
  --telemetry run.jsonl --record run.mp4 --record-raw run-raw.mp4
```

**`--record-raw` is the one people forget.** `--record` burns the HUD and every detection box
into the pixels — readable, and **useless as training data**. `--record-raw` writes the same
frame before anything is drawn, and both share a frame index with `perception.video_frame`,
so your detections become labels for clean pixels.

Recording costs control-loop rate: **246.4 ms** per new-result tick with `--record` against
**100.6 ms** without. A run recorded for training is not a run whose timing numbers mean
anything. 用于训练的录制不适合用来做时序测量。

## 11. ⚠️ The calibration in circulation is not self-consistent / 现有标定文件自相矛盾

The camera block embedded in the 2026-08-27 recordings:

```json
{"focal_px": 469.63, "height_m": 0.4, "hfov_deg": 156.16, "width": 1280, "height": 720}
```

- `height_m` **0.40** contradicts the measured **0.37** standing / **0.115** prone;
- **no pitch**, where the mount is a fixed **~11°**;
- `focal_px` implies a **107.46°** field, `hfov_deg` says **156.16°** — **48.7° apart** — with
  none of the `method`/`samples`/`residual_deg_rms` provenance the Go2's block carries.

**So every `range_m` derived from it is not a measurement.** Box pixel coordinates are fine.
The fix is a spin calibration on a Lite3 — Stage 1 above. See
[`robot-stack/CAMERA-GEOMETRY.md`](../../CAMERA-GEOMETRY.md).

**因此由它推导的所有 `range_m` 都不是测量值**（像素框坐标可用）。
