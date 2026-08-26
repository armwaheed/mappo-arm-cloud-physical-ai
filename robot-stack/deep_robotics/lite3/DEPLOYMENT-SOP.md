<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lite3 deployment SOP — from a new robot to the live MAPPO demo
# Lite3 部署标准流程 —— 从一台新机到 MAPPO 实机演示

**EN** — This is the end-to-end procedure for bringing a **new** Lite3 Venture from
first power-on to a recorded live MAPPO visual-navigation run. It exists because the
first deployment (LITE3-A, 2026-08-26) took one full day, and most of that day was spent
on six avoidable traps that are written down here. Two further robots need the same
deployment; each one is a full independent pass through this document.

**中文** —— 本文档是把一台**新的** Lite3 Venture 从首次上电到跑出留有录像的 MAPPO
视觉导航实机演示的完整流程。它存在的理由：第一台（LITE3-A，2026-08-26）部署花了一整天，
而其中大部分时间耗在六个本可避免的坑里，这些坑已全部写进本文。还有两台机器人要做同样的部署；
**每台都要独立完整走一遍本流程**。

> **EN** — This SOP does **not** repeat two sibling documents:
>
> - [`commissioning/RUNBOOK.md`](commissioning/RUNBOOK.md) — the per-robot measurement
>   session (state capture, loaded radius, camera fit, gait floor, actuator gain, signing
>   the record). Phase 1 below runs that document, verbatim.
> - [`README.md`](README.md) — what the platform binding is, which vendor interfaces it
>   uses, and why. Read it once before your first deployment.
>
> **中文** —— 本 SOP **不重复**两份姊妹文档的内容：
>
> - [`commissioning/RUNBOOK.md`](commissioning/RUNBOOK.md) —— 每台机器人的测量流程
>   （状态采集、负载半径、相机拟合、步态下限、执行器增益、签署记录）。下面的阶段 1
>   就是原样执行那份手册。
> - [`README.md`](README.md) —— 平台绑定的构成、使用了哪些厂商接口、以及为什么。
>   首次部署前通读一遍。

**EN** — Everything is tracked in
[issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13). Every
session ends with a continuation comment there — that comment is the only handover the
next session gets.

**中文** —— 所有工作都挂在
[issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)
上。每次工作都以一条交接评论收尾 —— 那是下一次工作唯一的交接依据。

---

## 0. Safety and naming — read first
## 0. 安全与命名 —— 最先读

**EN**

- [`../../SAFETY.md`](../../SAFETY.md) governs anything that moves a leg. It is not
  optional. `--live` is the only flag that moves the robot; an operator holds the vendor
  remote with the emergency stop for the whole run, and the human in the room is asked
  before **any** walking.
- Credentials live in `~/.robot-creds` on your own machine and **nowhere else** — never
  in the repo, an issue, a PR, or a log. The robot's address is `192.168.1.120` and may
  be written down; its password may not.
- The product is **Arm Device Connect** (**DC**). No earlier internal name, and no
  originating company name, may appear in any file, comment, commit message, issue or PR
  text. See `AGENTS.md` at the repository root.

**中文**

- [`../../SAFETY.md`](../../SAFETY.md) 管辖一切会让腿动起来的操作，不可跳过。
  `--live` 是唯一会让机器人动起来的开关；运行全程必须有一名操作员手持厂商遥控器和急停；
  任何行走之前必须先征得现场人员的同意。
- 凭证只放在你自己机器的 `~/.robot-creds` 里，**其它任何地方都不行** —— 不进仓库、
  不进 issue、不进 PR、不进日志。机器人地址 `192.168.1.120` 可以记录，密码不可以。
- 产品名称是 **Arm Device Connect**（**DC**）。任何早期内部名称和原公司名称都不得出现在
  任何文件、注释、提交信息、issue 或 PR 文字里。见仓库根目录的 `AGENTS.md`。

---

## 1. Prerequisites
## 1. 前提条件

| # | item / 项目 | requirement / 要求 |
| --- | --- | --- |
| 1 | repo / 代码 | A tree that contains the `battery_level` combinator delegation (PR #98 or later). Earlier trees walk straight into `REFUSING TO MOVE: no battery on '/battery_state'`. / 必须包含 `battery_level` 组合器委托修复（PR #98 或更新）。更早的代码会直接撞上 `REFUSING TO MOVE: no battery on '/battery_state'`。 |
| 2 | laptop Python / 笔记本环境 | Commissioning tools need Python ≥ 3.8 and nothing else; the camera fitter also needs `opencv-python`. Repo test suites run in the `py312` conda env. / 标定工具只需 Python ≥ 3.8；相机拟合另需 `opencv-python`。仓库测试套件在 conda 的 `py312` 环境里跑。 |
| 3 | network / 网络 | Laptop Ethernet on the **single static address** the robot streams to (see Phase 0), netmask `255.255.255.0`, gateway **empty**. `ping 192.168.1.120` must reply. / 笔记本以太网设为机器人状态流指向的**唯一静态地址**（见阶段 0），掩码 `255.255.255.0`，网关**留空**。必须能 ping 通 `192.168.1.120`。 |
| 4 | battery / 电量 | ≥ 40% before any session. The abort line is 20%. On 2026-08-26 four short live runs pulled the pack from 50% to 28% — charge early, not late. / 开工前 ≥ 40%。中止线 20%。2026-08-26 当天四次短时实机运行就把电量从 50% 拉到 28% —— 充电要趁早。 |
| 5 | room / 场地 | A measured lane, clear on **both sides** — the robot has no lateral sensing. Second robot parked outside the lane, powered off. / 一条**实测过**的通道，两侧同样清空 —— 机器人没有侧向感知。第二台机器人停在通道外并关机。 |

---

## 2. Phase 0 — robot-side one-time setup
## 2. 阶段 0 —— 机器人侧一次性部署

**EN** — Done once per robot, over SSH: `ssh user@192.168.1.120`. Do not install
anything system-wide and do not touch the vendor services; everything lives under one
directory:

**中文** —— 每台机器人做一次，通过 SSH：`ssh user@192.168.1.120`。不做任何系统级安装，
不改动厂商服务；所有内容都放在一个目录下：

```
/home/user/mappo-lite3-stage/
├── releases/     # one directory per deployed repo snapshot, never edited in place
│                 # 每次部署一个目录，绝不原地修改
├── python/       # AArch64 site-packages incl. cv2; PYTHONPATH points here
│                 # AArch64 的 Python 包（含 cv2）；PYTHONPATH 指向这里
├── models/       # mobilenet-ssd detector files / MobileNet-SSD 检测器模型文件
├── calibration/  # per-robot calibration and axis profiles / 本机的标定与轴配置文件
└── evidence/     # run recordings before they are pulled back / 拉取回传前的运行录像
```

**EN** — Release discipline: unpack each new repo snapshot into a **fresh**
`releases/<tag>/` directory. To iterate, `cp -r` the previous release into a new
directory and change the copy. A release directory that was edited in place can no
longer be tied to a commit, and evidence recorded from it becomes unattributable.

**中文** —— 版本纪律：每一个新的代码快照都解包到**新的** `releases/<tag>/` 目录。
需要迭代时，把上一个版本 `cp -r` 成新目录，然后改副本。被原地改过的版本目录无法再对应到
某个提交，从它录出的证据也就无法溯源。

**EN** — The four robot-side checks, in order. Do not proceed past a failed check.

**中文** —— 机器人侧的四项检查，按顺序做。任何一项不过，不要往下走。

1. **`~/jy_exe/conf/network.toml`** must read:

   ```toml
   ip = '127.0.0.1'
   target_port = 43897   # motion host streams RobotState here
   local_port  = 43893   # motion host accepts commands here
   ```

   **EN** — If `ip` has regressed to a LAN address, the host streams state to itself and
   every tool reports a silent link. This was the first trap of the first deployment.
   **中文** —— 如果 `ip` 回退成了局域网地址，状态流会发给它自己，所有工具都会报"链路沉默"。
   这是首次部署踩的第一个坑。

2. **State stream** — run task 0 of the RUNBOOK (`lite3_state_probe.py --seconds 30`).
   Hundreds of frames, several `kind`s, a valid `battery_level`.
   **中文** —— **状态流** —— 跑 RUNBOOK 的任务 0。应有几百帧、多种 `kind`、
   有效的 `battery_level`。

3. **Camera** — the vendor GStreamer publisher owns `/dev/video0`; consume its existing
   RTSP output instead of competing for the device:

   ```bash
   release=$HOME/mappo-lite3-stage/releases/<TAG>
   export PYTHONPATH=$HOME/mappo-lite3-stage/python
   python3 "$release/robot-stack/deep_robotics/lite3/visual_nav/lite3_vision_shadow.py" \
       --camera-source rtsp://127.0.0.1:8554/test \
       --model-dir "$HOME/mappo-lite3-stage/models/mobilenet-ssd" \
       --classes person,chair --seconds 20 \
       --output "$HOME/mappo-lite3-stage/evidence/vision-shadow-check.jsonl"
   ```

   **EN** — Pass: frames arriving at 1280x720, zero camera read errors. A detection or
   no detection is a scene observation, not a failure.
   **中文** —— 通过标准：有 1280x720 的帧进来、没有相机读取错误。有没有检测到目标是
   场景问题，不是故障。

4. **Mode state** — if any tool reports `robot_basic_state=98` (or anything that is not
   documented force-control `basic_state=6`), that is not a code bug. Per the Deep
   Robotics engineer: in the vendor app, **re-zero (回零) the robot on the controller,
   then stand it**. Live mode refuses any state other than `basic_state=6` rather than
   switching it.
   **中文** —— **模式状态** —— 如果工具报 `robot_basic_state=98`（或任何不是文档中
   力控 `basic_state=6` 的值），这不是代码 bug。按云深处工程师的指导：在厂商 app 里
   **先在控制器上回零，再执行站立**。实机模式会拒绝 `basic_state=6` 以外的任何状态，
   而不会替你切换。

---

## 3. Phase 1 — commissioning (run the RUNBOOK)
## 3. 阶段 1 —— 标定取证（执行 RUNBOOK）

**EN** — Work through [`commissioning/RUNBOOK.md`](commissioning/RUNBOOK.md) top to
bottom, **once per robot**: tasks 0–5 produce the artefacts, task 6 merges them into one
record, a human reviews it (`--review 'Your Name'`), and `--emit-flags` prints the
`--gait-floor` / `--actuator-gain` / `--robot-radius` / `--policy-scale` values the live
run needs. Nothing transfers between robots — `commission.py` refuses to merge a result
measured on another unit, and that refusal is the premise of issue #13, not an
inconvenience.

**中文** —— 从头到尾执行 [`commissioning/RUNBOOK.md`](commissioning/RUNBOOK.md)，
**每台机器人一遍**：任务 0–5 产出各测量文件，任务 6 合并成一份记录，由人审阅签名
（`--review '你的名字'`），然后 `--emit-flags` 打印实机运行所需的
`--gait-floor` / `--actuator-gain` / `--robot-radius` / `--policy-scale`。
两台机器人之间不得互通任何数字 —— `commission.py` 会拒绝合并另一台机器人的测量结果，
这个拒绝正是 issue #13 的前提，而不是麻烦。

**EN** — One extra artefact this phase must also produce, because the RUNBOOK predates
it: the **axis profile** (`--axis-profile`) for the sign-only simple-axis transport
described in [`README.md`](README.md). Measure each enabled direction's raw primitive on
this robot, give every nonzero primitive an evidence reference, and record the file's
SHA-256. A direction with no evidence stays `null` — that is a designed outcome, not a
gap. Reference values from LITE3-A (2026-08-26): forward `+32767`, yaw `±16000`,
lateral and reverse `null`; profile `lite3_axis_profile_LITE3-A.json`, 6/6 profile
checks PASS.

**中文** —— 本阶段还要额外出具一份 RUNBOOK 成文时还没有的文件：**轴配置文件**
（`--axis-profile`），用于 [`README.md`](README.md) 里描述的"只取符号"的简易轴传输。
在本机上实测每个启用方向的原始轴值，给每个非零值附上证据来源，并记录文件的 SHA-256。
没有证据的方向保持 `null` —— 这是设计如此，不是缺口。LITE3-A（2026-08-26）的参考值：
前进 `+32767`，偏航 `±16000`，侧移与倒退为 `null`；配置文件
`lite3_axis_profile_LITE3-A.json`，6/6 项检查通过。

---

## 4. Phase 2 — camera calibration and the goal profile
## 4. 阶段 2 —— 相机标定与目标配置

**EN — the calibration scene.** Robot standing in demo posture with its event payload
(loaded plan-view footprint on LITE3-A: 55 cm × 40 cm). Place the marker/panel so that:

**中文 —— 标定场景。** 机器人以演示姿态站立并装好负载（LITE3-A 负载俯视外形：55 cm ×
40 cm）。标记板的摆放要求：

| dimension / 尺寸 | value / 数值 | measured on / 测量来源 |
| --- | --- | --- |
| lens-to-marker distance / 镜头到标记板距离 | 0.50 m, tape-measured from the lens centre / 用卷尺从镜头中心量 | LITE3-A, 2026-08-26 |
| panel centre height / 标记板中心离地高度 | 0.26 m above the carpet / 距地毯 0.26 m | LITE3-A, 2026-08-26 |

**EN** — The panel must be **fully inside the frame** at that distance. The 2026-08-26
session lost an hour here: the marker sat on top of a **vertical** bin, and at 0.50 m
the close camera no longer saw the top of it. Either lay the bin on its side with the
panel on top facing the camera, or mount the panel lower, square-on to the lens. Moving
the marker does **not** require re-running the whole commissioning — only the camera fit
and its validation observation are distance-dependent.

**中文** —— 在这个距离上，标记板必须**完整落在视野内**。2026-08-26 当天在这里耗了一小时：
标记板放在**竖直的**箱子顶端，距离拉近到 0.50 m 后相机拍不到箱顶。两种摆法都行：
把箱子横放、标记板置于顶部正对相机；或者把标记板放低、正对镜头。挪动标记板**不需要**
重走整个标定流程 —— 只有相机拟合及其验证观测与距离有关。

**EN — provisional vs validated.** A static fit writes a **provisional** calibration and
`--live` refuses it. Validate focal length, optical-centre height and mount pitch
against a second known-distance observation, then write `calibration_status=validated`
and record the SHA-256. One known trap: at close range the shared fitter can report a
**width-displacement FAIL that is a detector min-area morphology artifact**, not a real
geometry change. When the stand-1 width is within a few percent and the pitch fit is
small (LITE3-A: width +2.4%, pitch −0.57°), that is the artifact being observed, and
the validation stands. The LITE3-A reference artefact is
`calibration/chair-bin-20260825T172500Z/lite3_front_camera_green_panel_validated.json`
(SHA-256 `e7a77143…`).

**中文 —— 待审与已验证。** 静态拟合写出的是**待审**标定，`--live` 会拒绝它。需要用第二个
已知距离的观测独立验证焦距、光心高度和安装俯仰角，然后写入 `calibration_status=validated`
并记录 SHA-256。一个已知的坑：近距离下共用拟合程序可能报出**宽度位移 FAIL，而那只是
检测器最小面积形态学的伪影**，不是真实的几何变化。当站 1 宽度偏差在百分之几以内、
俯仰拟合很小（LITE3-A：宽度 +2.4%、俯仰 −0.57°）时，看到的就是这个伪影，验证成立。
LITE3-A 的参考文件是
`calibration/chair-bin-20260825T172500Z/lite3_front_camera_green_panel_validated.json`
（SHA-256 `e7a77143…`）。

**EN — the goal and obstacle profiles.** The demo goal is a chair; measure the actual
chair (LITE3-A event chair: 1.00 m tall, 0.71 m wide). The static obstacle uses a
`colour-profile/v1` static profile whose radius must cover the **whole** physical
obstacle: the event box measured 0.28–0.33 m against a 0.20 m nominal, and the
shortfall cost a stalled run. Measure the obstacle, not its spec sheet.

**中文 —— 目标与障碍物配置。** 演示目标是一把椅子，要实测（LITE3-A 的活动用椅：
高 1.00 m、宽 0.71 m）。静态障碍物使用 `colour-profile/v1` 静态配置，其半径必须覆盖
障碍物的**全部**实际外形：活动用的箱子实测半径 0.28–0.33 m，而标称只有 0.20 m，
这个差额直接废掉了一次运行。量实物，不要量说明书。

---

## 5. Phase 3 — the demo scene
## 5. 阶段 3 —— 演示场地布置

**EN** — Three measured placements; use a tape, not an estimate.

**中文** —— 三处实测摆放；用卷尺，不要估算。

1. **Goal chair**: 2.0–2.5 m ahead of the robot, **on the camera axis**, fully visible,
   not half-hidden behind a table. `goal never sighted in 20 s` is a placement problem,
   not a detector problem — re-place the chair before touching any threshold. The
   fallback is `--goal-confidence 0.40`, used only after placement is confirmed good.
   **目标椅**：在机器人正前方 2.0–2.5 m、**位于相机中轴线上**、完整露出、不被桌子半遮。
   `goal never sighted in 20 s` 是摆放问题，不是检测器问题 —— 先重新摆椅子，再考虑动阈值。
   备用手段是 `--goal-confidence 0.40`，仅在确认摆放无误后使用。
2. **Obstacle box**: centre at least **0.9 m** off the robot→chair line
   (0.40 m robot radius + 0.28–0.33 m measured box radius + margin). The earlier
   "≥ 0.5 m" rule was computed from the 0.20 m nominal radius and is not enough — a run
   held in corridor for 58 s and timed out on it.
   **障碍箱**：箱心距"机器人→椅子"连线至少 **0.9 m**（机器人半径 0.40 m + 箱子实测半径
   0.28–0.33 m + 余量）。之前的"≥ 0.5 m"规则是按 0.20 m 标称半径算的，不够 ——
   有一次运行因此在走廊判定里卡了 58 秒直到超时。
3. **Everything else**: the lane stays clear on both sides; the second robot stays
   outside the lane, powered off; people step out before `--live`.
   **其它所有东西**：通道两侧保持清空；第二台机器人停在通道外并关机；
   `--live` 之前人员离场。

---

## 6. Phase 4 — shadow, then live
## 6. 阶段 4 —— 先影子运行，再实机

**EN** — Always shadow first (Phase 0, check 3 is the shadow command): perception
running, goal detected at the placement you measured, ranges plausible. Only then live.
The live command, with every robot-specific value in `<angle brackets>`:

**中文** —— 永远先跑影子模式（阶段 0 第 3 项检查就是影子命令）：感知在跑、目标在你实测的
位置上被检测到、测距合理。然后才实机。实机命令模板如下，所有因机而异的值放在
`<尖括号>` 里：

```bash
cd $HOME/mappo-lite3-stage/releases/<TAG>/robot-stack/deep_robotics/lite3/visual_nav
PYTHONPATH=$HOME/mappo-lite3-stage/python python3 lite3_visual_nav.py \
    --camera-source rtsp://127.0.0.1:8554/test \
    --model-dir $HOME/mappo-lite3-stage/models/mobilenet-ssd \
    --calibration <VALIDATED_CALIBRATION_JSON> \
    --static-profile <BOX_PROFILE_JSON> \
    --goal-class chair --goal-height 1.00 --goal-width 0.71 \
    --goal-input-size 300 --goal-crop 0.5 --goal-confidence 0.50 \
    --locomotion-transport axis --axis-profile <AXIS_PROFILE_JSON> \
    --state-bind 127.0.0.1 \
    --live --operator-ready \
    --gait-floor 0.30 --actuator-gain 1.07 --robot-radius 0.40 \
    --max-vx 0.55 --max-vy 0 --max-wz 0.90 \
    --accept-no-motor-temperatures --max-seconds 60 \
    --record $HOME/mappo-lite3-stage/evidence/<RUN_ID>.mp4 \
    --telemetry $HOME/mappo-lite3-stage/evidence/<RUN_ID>.jsonl
```

**EN** — `--gait-floor` / `--actuator-gain` / `--robot-radius` come from this robot's
**reviewed** commissioning record (`--emit-flags`), not from this document; the values
shown are LITE3-A's and are placeholders. `--accept-no-motor-temperatures` is an
explicit operator decision that bounds **one** run: `--max-seconds` is capped at 120 s,
battery gates stay enforced, and heat is invisible to software — let the robot cool
between runs.

**中文** —— `--gait-floor` / `--actuator-gain` / `--robot-radius` 必须来自本机**已审**
标定记录（`--emit-flags`），不能抄本文档；上面展示的是 LITE3-A 的值，仅作占位。
`--accept-no-motor-temperatures` 是操作员的明确决定，它只约束**一次**运行：
`--max-seconds` 上限 120 秒，电池门照常生效，而热量对软件完全不可见 —— 两次运行之间
让机器人散热。

---

## 7. Phase 5 — after every run
## 7. 阶段 5 —— 每次运行之后

**EN**

1. **Pull the evidence back immediately.** `scp` the `.mp4` and `.jsonl` to the laptop
   and verify the SHA-256 on both sides. Evidence left on the robot is evidence at
   risk: on 2026-08-26 the SSH link dropped after run 4, and run 4's recording was
   still on the robot at handover. If the link is down, recovering it comes before any
   further run.
2. **Write the continuation comment** on issue #13: outcome numbers, the room state,
   what the run found, what is still open. Tables must be self-contained — no
   `user-attachments` links and no `?token=` asset URLs, which are slow or unreachable
   from mainland China and expire.

**中文**

1. **立刻把证据拉回来。** 用 `scp` 把 `.mp4` 和 `.jsonl` 取回笔记本，两端核对 SHA-256。
   留在机器人上的证据就是有风险的证据：2026-08-26 第四次运行后 SSH 链路中断，
   交接时第四次运行的录像还留在机器人上。链路断了，先恢复链路，再谈继续跑。
2. **写交接评论**到 issue #13：结果数字、现场状态、本次发现了什么、还有什么没完成。
   表格必须自包含 —— 不放 `user-attachments` 链接，不放会过期的 `?token=` 资源链接，
   它们在中国大陆很慢或无法访问。

---

## 8. Troubleshooting — the traps, in the order they cost time
## 8. 故障排查 —— 按当天耗时排序的坑

| symptom / 现象 | cause / 原因 | action / 处理 |
| --- | --- | --- |
| `NO FRAMES RECEIVED` / state stream silent / 状态流沉默 | `network.toml` `ip` regressed to a LAN address; the host streams to itself. / `network.toml` 的 `ip` 回退成局域网地址，状态流发给了它自己。 | Phase 0, check 1. Set `ip='127.0.0.1'`, `target_port=43897`, `local_port=43893`. / 见阶段 0 第 1 项，按上文改回。 |
| `REFUSING TO MOVE: no battery on '/battery_state'` | The locomotion combinator broke `battery_level` delegation; fixed in PR #98. / 运动组合器断掉了 `battery_level` 委托；PR #98 已修复。 | Deploy a tree that contains the fix. Do not patch around it on the robot. / 部署包含该修复的代码。不要在机器人上临时绕。 |
| `robot_basic_state=98` (or any non-`6` basic state) | Mode state left by previous handling; not a code bug. / 之前的操作留下的模式状态；不是代码 bug。 | Per the Deep Robotics engineer: re-zero on the controller in the vendor app, then stand. Live needs `basic_state=6`. / 按云深处工程师指导：在厂商 app 里先回零再站立。实机需要 `basic_state=6`。 |
| `goal never sighted in 20 s` | Chair off the camera axis, half-occluded, or too far. / 椅子偏出相机中轴、被半遮、或太远。 | Re-place the chair (Phase 3), then retry. Only then consider `--goal-confidence 0.40`. / 先按阶段 3 重摆椅子再重试，之后才考虑 `--goal-confidence 0.40`。 |
| run holds in corridor / stall, no motion / 运行卡在走廊判定、不动 | Obstacle too close to the robot→goal line. / 障碍物离"机器人→目标"连线太近。 | Box centre ≥ 0.9 m off the line, computed from **measured** radii (robot 0.40 m, box 0.28–0.33 m). / 箱心离连线 ≥ 0.9 m，按**实测**半径计算（机器人 0.40 m、箱子 0.28–0.33 m）。 |
| calibration width-displacement FAIL at close range / 近距离标定报宽度位移 FAIL | Detector min-area morphology artifact, not a geometry change. / 检测器最小面积形态学伪影，不是几何变化。 | If stand-1 width is within a few percent and pitch fit is small (LITE3-A: +2.4%, −0.57°), accept and write `calibration_status=validated`. / 若站 1 宽度偏差在百分之几内且俯仰拟合很小（LITE3-A：+2.4%、−0.57°），接受并写入 `calibration_status=validated`。 |
| battery falls faster than expected / 电量下降比预期快 | Live runs are short but draw hard: 50→28% across four runs on 2026-08-26. / 实机运行虽短但耗电大：2026-08-26 四次运行从 50% 掉到 28%。 | Start ≥ 40%, abort 20%, charge between sessions, cool between runs. / 开工 ≥ 40%、20% 中止、场次之间充电、运行之间散热。 |
| SSH times out mid-session / SSH 中途超时 | Wired link or connector disturbed. / 有线链路或接头被动过。 | Check the cable and link first; recover any evidence still on the robot before further runs. / 先查网线和链路；继续跑之前先把还留在机器人上的证据取回。 |

**EN** — A refusal is a result, not a failure. If a refusal blocks you and neither this
table nor the RUNBOOK catalogue resolves it, stop and write it into issue #13 with the
full terminal output. Do not improvise a value.

**中文** —— "拒绝执行"是一种结果，不是故障。如果某个拒绝挡住了你，而本表和 RUNBOOK 的
对照表都没有答案，就停下来，把完整的终端输出写进 issue #13。不要临场编一个值。

---

## 9. Per-robot retention checklist
## 9. 每台机器人必须留存的物件

**EN** — One set per robot, kept with its robot ID. When any of these is missing, the
robot is not deployed, whatever the last run looked like.

**中文** —— 每台机器人一套，与其机器人编号一起保存。缺了任何一样，这台机器人就不算
完成部署 —— 不管上一次运行看起来多顺利。

| artefact / 物件 | where / 位置 | LITE3-A reference (2026-08-26) / LITE3-A 参考值 |
| --- | --- | --- |
| commissioning record, **reviewed** / 已审的标定记录 | laptop + issue #13 | `lite3-commissioning-LITE3-A.json` |
| axis profile + SHA-256 / 轴配置文件及哈希 | robot `calibration/` | `lite3_axis_profile_LITE3-A.json`, `e41949cc…`, 6/6 PASS |
| validated camera calibration + SHA-256 / 已验证相机标定及哈希 | robot `calibration/` | `chair-bin-20260825T172500Z/lite3_front_camera_green_panel_validated.json`, `e7a77143…` |
| static obstacle profile + SHA-256 / 静态障碍物配置及哈希 | robot `calibration/` | box profile, measured radius 0.28–0.33 m / 箱子配置，实测半径 0.28–0.33 m |
| measured goal dimensions / 目标实测尺寸 | issue #13 | chair 1.00 m × 0.71 m / 椅子 1.00 m × 0.71 m |
| `network.toml` snippet / 网络配置摘录 | issue #13 | `127.0.0.1`, `43897`, `43893` |
| firmware version / 固件版本 | issue #13 | V1.0.8 |
| run evidence, pulled back / 已取回的运行证据 | laptop `evidence/` | `evidence/2026-08-26-lite3-live-demo/` (runs 1–3; run 4 pending link recovery / 第 4 次待链路恢复后取回) |
| continuation comment / 交接评论 | issue #13 | comment `5425166162` |

---

## 10. If you are stuck
## 10. 如果卡住了

**EN** — Same rule as the RUNBOOK: write it into issue #13 with the exact terminal
output, the state of the room, and what you had already tried. A session that ends with
a good continuation comment costs the next person nothing; a session that ends in
someone's scrollback costs them a day.

**中文** —— 与 RUNBOOK 同一条规则：把完整的终端输出、现场情况、以及你已经尝试过什么
写进 issue #13。一次以良好交接评论收尾的工作，对下一个人来说成本为零；一次只留在
某人终端回滚记录里的工作，会让下一个人损失一整天。
