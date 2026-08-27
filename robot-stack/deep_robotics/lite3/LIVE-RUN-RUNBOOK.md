<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lite3 live run: every flag, every refusal / Lite3 实机运行：所有参数与所有拒绝原因

**EN** — A live Lite3 run refuses unless **nine** things are stated. Each refusal exits in
about three seconds, before the robot stands, and looks from across the room like nothing
happened. This page exists so that no refusal costs you a day.

**中文** —— Lite3 实机运行需要**九项**参数，缺一即拒绝。拒绝会在约 3 秒内退出，
机器人尚未站起，从远处看就像"什么都没发生"。本页的目的是让任何一次拒绝都不会浪费你一天。

---

## 1. The nine required inputs / 九项必需输入

| flag | where the value comes from / 数值来源 |
|---|---|
| `--calibration` | This Lite3's camera. ⚠️ See §4 — the one in circulation is not self-consistent. |
| `--gait-floor` | **Measured**, `axis_primitive_probe.py`. Not the Go2's 0.35. |
| `--actuator-gain` | **Measured at this envelope** — fit against pose, not against the velocity estimate. |
| `--robot-radius` | **0.33** — measured by Timo 2026-08-27. The 0.20 nominal is 65% low. |
| `--max-vx` | Stated. Unset, it silently inherits **the Go2's**. |
| `--max-vy` | Stated. This Lite3's lateral primitive is unmeasured, so `0` is honest. |
| `--max-wz` | Stated. Unset, inherits the Go2's. |
| `--operator-ready` | Typed **after** the robot is STANDING and in navigation mode on the vendor app. |
| `--axis-profile` | Only when `--locomotion-transport axis`. Needs physically evidenced primitives. |

**Why the envelope cannot be defaulted / 为什么包络值不能用默认值**: it is the right-hand
side of a safety gate. `_validate_axis_profile_speeds` refuses a primitive whose
`measured_m_s` exceeds `--max-vx x --derate`. Against a borrowed right-hand side that
comparison is arithmetic, not a gate. 包络值是安全门限的右侧；借用别的机器人的数值，
这个比较就只是算术，而不是门限。

Also required, and not a flag: **a virtualenv**. `require_virtualenv(reaching_hardware=True)`
prints on every live run, including when it decides not to enforce. 另需在 **venv** 中运行。

---

## 2. The refusal decoder / 拒绝信息对照表

| what you see | what it means | what to do |
|---|---|---|
| `REFUSING TO WALK: missing ...` | One or more of the nine above | Read the list — it names each one and why |
| `REFUSING TO WALK: --accept-no-motor-temperatures needs --max-seconds set to 120s or less` | Thermal feed absent **and** the run is unbounded | Add `--max-seconds 120` or lower. §3. |
| `REFUSING TO WALK: <D1 arm not stowed>` | Go2 only — arm out of stow | Hand-pose the arm flat along the spine |
| `TOP SPEED x IS BELOW THIS ROBOT'S MEASURED GAIT FLOOR y` | **Warning, not a refusal.** The run continues and the robot may not move | Raise the envelope or command scale |
| `[tree-stamp] no stamp -- this tree cannot name its commit` | Off-robot: a finding. **On a robot: a refusal** | Deploy with `deploy/push-to-robot.sh` |
| exits ~3 s with no banner, rc 139 | `Segmentation fault` — the SDK env was not sourced | `source .../install/setup_env.sh` first |
| `ModuleNotFoundError: ros2_twist_locomotion` | Dashboard is on the ROS 2 path | `--locomotion-transport axis` (or `udp`) |

**A refusal is the system working.** Every one of them exists because of a measured
failure. 每一条拒绝都源自一次实测到的故障，拒绝本身是系统在正常工作。

---

## 3. Motor temperature — measure it, we do not have a number / 电机温度：我们还没有数值

**EN** — The thermal question is open with Deep Robotics. Until it is answered,
`--accept-no-motor-temperatures` runs **bounded to 120 s** and nothing bounds the *next*
run: heat builds across back-to-back runs and no software here can see it.

**We have a comparable number for the other robot and it is sobering.** The Go2 aborts at
**70 °C** and warns at **55 °C**, idles near **30 °C**, and its rear leg motors overheat and
collapse if it is left standing — which is why that robot lives lying down.

**中文** —— 电机温度问题仍在与云深处确认。在此之前，`--accept-no-motor-temperatures`
每次运行限时 **120 秒**，且**无法约束下一次运行** —— 连续运行时热量累积，软件无法感知。
参考：Go2 在 **70 °C** 中止、**55 °C** 警告、空载约 **30 °C**，且长时间站立会导致后腿电机
过热瘫倒。

### A task for your coding agent / 给你的编程助手的任务

> Observe whatever motor-temperature telemetry this Lite3 exposes and find its real limits.
> Log temperature across several **bounded** runs — start cold, run 120 s, record the curve,
> let it cool, repeat — and report: idle temperature, the rate of rise under load, and
> whether any channel exists at all. If the Lite3 exposes **no** thermal channel, that is
> the finding: say so explicitly rather than reporting nothing. Do not run back-to-back to
> find the ceiling by reaching it.
>
> 请观察本台 Lite3 能提供的电机温度遥测，找出真实上限。在多次**限时**运行中记录温度：
> 从冷机开始，运行 120 秒，记录曲线，冷却后重复。报告：空载温度、负载升温速率、
> 以及是否存在该通道。如果 Lite3 **完全没有**温度通道，这本身就是结论，请明确说明。
> 不要用连续运行去"试出"上限。

---

## 4. ⚠️ The calibration you have is not self-consistent / 现有标定文件自相矛盾

The camera block embedded in the 2026-08-27 recordings:

```json
{"focal_px": 469.63, "height_m": 0.4, "hfov_deg": 156.16, "width": 1280, "height": 720}
```

- `height_m` **0.40** contradicts the measured **0.37** standing / **0.115** prone;
- **no pitch**, where the mount is a fixed **~11°**;
- `focal_px` implies a **107.46°** field, `hfov_deg` says **156.16°** — **48.7° apart** — with
  none of the `method` / `samples` / `residual_deg_rms` provenance the Go2's block carries.

**So every `range_m` derived from it is not a measurement.** Box pixel coordinates are fine.
See `robot-stack/CAMERA-GEOMETRY.md`. The fix is a spin calibration on a Lite3.

**因此由它推导的所有 `range_m` 都不是测量值**（像素框坐标可用）。需要在 Lite3 上做一次
spin 标定。

---

## 5. Before you press anything / 运行前

1. **Shadow first.** Confirm `decision` and `transport` appear in the telemetry before any
   live run. 先跑 shadow，确认遥测中出现 `decision` 与 `transport`。
2. **Cordon the lane.** A colleague walking past has not accepted this risk; you have.
   请隔离通道 —— 路过的同事并未同意承担这个风险。
3. **Emergency stop in hand**, and stop if a motor smells hot or the gait changes.
   手持急停；若电机有异味或步态改变，立即停止。
4. `--operator-ready` is typed **last**, after STANDING and navigation mode.

## 6. Known-good shape / 已验证的命令形状

⚠️ **Values are placeholders — substitute your measured numbers.** This shows which flags
must be present, not what they should equal. 数值为占位符，请替换为你的实测值。

```bash
source /path/to/robot-stack/deep_robotics/lite3/install/setup_env.sh   # or the SDK env
# in a venv -- see AGENTS.md

python3 mappo_drive.py --live \
  --policy-mode supervised --policy-scale 4.0 \
  --calibration  lite3_front_camera.json \
  --gait-floor   <measured> \
  --actuator-gain <measured at this envelope> \
  --robot-radius 0.33 \
  --max-vx <stated> --max-vy 0 --max-wz <stated> \
  --max-seconds 120 --accept-no-motor-temperatures \
  --operator-ready \
  --telemetry run.jsonl --record run.mp4 --record-raw run-raw.mp4
```

**`--record-raw` is the one people forget.** `--record` burns the HUD and every detection
box into the pixels, which makes the file readable and **useless as training data**.
`--record-raw` writes the same frame before anything is drawn on it, and both share a frame
index with `perception.video_frame` in the telemetry — so your detections become labels for
clean pixels. 别忘了 `--record-raw`：`--record` 会把 HUD 和检测框烧进画面，无法用作训练数据。

Recording costs control-loop rate: measured **246.4 ms** per new-result tick with `--record`
against **100.6 ms** without (#18). A run recorded for training is not a run whose timing
numbers mean anything. 录制会拖慢控制循环，用于训练的录制不适合用来做时序测量。
