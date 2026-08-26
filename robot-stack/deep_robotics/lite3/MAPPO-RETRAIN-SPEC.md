<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MAPPO retrain specification — Deep Robotics Lite3 Venture
# MAPPO 重训练规格书 —— 深度机器人 Lite3 Venture

# ⛔ FILED FOR LATER. NOT TO BE STARTED. / ⛔ 已归档待办，暂不启动。

**EN** — This is an **executable ask addressed to whoever owns the VMAS training
repository**, written so it can be picked up cold. It is **not** a plan for work in this
repository and **not** authorization to start training. Read §1 (status), then §5 (the
transport constraint) before anything else — §5 changes what a retrained policy should
even produce.

**中文** —— 这是一份**写给 VMAS 训练仓库负责人的、可直接执行的需求书**，写成即使完全没有
跟进过这个项目也能接手的形式。它**不是**本仓库内部的工作计划，**也不是**开始训练的授权。
请先读第 1 节（状态），然后**直接读第 5 节（传输链路限制）**——第 5 节决定了重训练出来的
策略"应该输出什么"。

> **EN** — Commands, flags, filenames, symbols and every numeric value are left in
> English/ASCII throughout and must be matched **exactly**. Only the prose and the
> reasoning are translated. Where a Chinese technical rendering is uncertain it is written
> plainly and marked **请确认** rather than guessed at; please correct it **in place** and
> say so on the issue.
>
> **中文** —— 全文的命令、参数、文件名、符号和所有数值一律保留英文 / ASCII 原文，必须
> **逐字**比对。只有说明文字和推理过程是中文。凡是中文技术译法没有把握的地方，本文一律
> 采用**直白说法**并标注 **请确认**，而不是硬猜术语；请**就地更正**，并在 issue 上说明。

**EN** — Every number below is traceable to a file, a PR or an issue comment in this
repository, and is cited where it is used. Numbers that are attributed but **cannot** be
re-derived from a clone are marked ⚠️. Three places where this document disagrees with an
earlier published statement are listed in §10.

**中文** —— 下文每一个数字都能追溯到本仓库的某个文件、PR 或 issue 评论，并在使用处标注了
出处。凡是"引用得来、但**无法**从一份干净 clone 重新算出"的数字，都标了 ⚠️。本文与此前
已发布说法不一致的三处，集中列在第 10 节。

---

## 1. Status, and the direction this comes from
## 1. 状态，以及这份文档的来源

**EN** —

| | |
| --- | --- |
| Status | **Filed. Not commissioned. No training is authorized by this document.** |
| Addressed to | the owner of the VMAS training repository — the checkpoint's author |
| Platform | Deep Robotics Lite3 Venture. **Not** the Go2 |
| Supersedes for the Lite3 | nothing — [issue #29](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/29) stays open and remains the discussion thread |
| Prerequisites | §8. Several are unmeasured platform constants; one is a decision, not a measurement |

Direction from the project owner, in force from 2026-08-25: **no more Go2 retraining. All
new retraining is for the Lite3 robots.** Everything in §3 was nevertheless measured on Go2
hardware, because that is the only hardware corpus that exists — §3.4 is the one Lite3
measurement, and §7 lists what a Lite3 corpus would have to add.

**中文** ——

| | |
| --- | --- |
| 状态 | **已归档。未立项。本文档不构成任何训练授权。** |
| 交付对象 | VMAS 训练仓库的负责人，也就是当前 checkpoint 的作者 |
| 目标平台 | 深度机器人 Lite3 Venture。**不是** Go2 |
| 与 #29 的关系 | 不替代。[issue #29](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/29) 保持 open，仍是讨论主线 |
| 前置条件 | 见第 8 节。其中若干项是尚未测量的平台常数；有一项是决策，不是测量 |

项目负责人 2026-08-25 起的方向：**Go2 不再重训练，后续所有重训练都面向 Lite3。**
但第 3 节里的数据仍然全部来自 Go2 硬件，因为目前只有这一份硬件数据。第 3.4 节是唯一一项
Lite3 上的测量；第 7 节列出了一份 Lite3 数据集还需要补什么。

### There is no training code in this repository
### 本仓库内没有任何训练代码

**EN** — This matters more than it sounds. `policy/PROVENANCE.md` states it plainly:
*"It is his deliverable; this repository holds a copy, not the original."* The shipped
weights were trained from `3_agent_m9g48xl_checkpoint_1910000.pt` and vendored as
`policy/models/mappo_actor_3agent_1910000.npz` (268 063 bytes, SHA-256
`7327f724…2ca11`). **Exactly one commit has ever touched `policy/models/`** — `a333e1c`,
2026-08-14, *"Vendor the MAPPO policy and its checkpoint, and correct five silent
defects"*. Nobody on this side can run, restart or tune a training job. This document is
therefore a specification handed across a repository boundary, and it has to be complete
enough to be executed without a conversation.

**中文** —— 这一点比字面上更重要。`policy/PROVENANCE.md` 写得很直接：*"这是他的交付物；
本仓库保存的是副本，不是原件。"* 现有权重训练自 `3_agent_m9g48xl_checkpoint_1910000.pt`，
以 `policy/models/mappo_actor_3agent_1910000.npz` 的形式收录（268 063 字节，SHA-256
`7327f724…2ca11`）。**`policy/models/` 目录历史上只被一次提交动过**——`a333e1c`，
2026-08-14。我们这边没有任何人能运行、重启或调整训练。所以这份文档是一份**跨仓库交付的
规格书**，必须完整到"不需要再开会讨论就能执行"。

---

## 2. Terminology — please confirm these renderings
## 2. 术语表 —— 这些译法请确认

**EN** — Written plainly rather than in polished domain jargon, deliberately. If the
Shanghai team has a house term for any of these, **correct it here in place** and say so on
issue #29; do not let a wrong term propagate into a training config.

**中文** —— 这里刻意用**直白说法**，不用行业习惯用语。如果上海团队已有习惯译法，请**就地
更正本表**，并在 issue #29 说明；不要让错误术语传进训练配置。

| English (use verbatim) | 中文（本文用法） | 需确认 |
| --- | --- | --- |
| ray fan / `n_rays` | 射线扇：从机器人向外投射的一组测距射线 | ✅ 请确认 |
| aperture | 通过间隙：两个障碍物之间可以穿过去的那个开口 | ✅ 请确认 |
| sensing horizon | 感知视野：射线能报出障碍物的最远距离（到**表面**，不是到中心） | ✅ 请确认（issue #13 中文已这样用） |
| proximity (the lidar value) | 接近度：`lidar` 通道的取值，**越大表示越近**，`0.000` 表示该方向没有东西 | ✅ 请确认 |
| observation / observation vector | 观测向量：每个控制周期送进网络的那 18 个数 | ✅ 请确认 |
| driven tick | 由策略驱动的控制周期（区别于被否决 / 保持的周期） | ✅ 请确认 |
| run-local frame | 运行局部坐标系：在 `reset_run` 那一刻冻结的坐标系，之后不再随机身转动 | ✅ 请确认 |
| goal-seeking | 趋目标：只朝目标方向走，与障碍物无关 | ✅ 请确认 |
| ablated control | 对照回放：同一段数据、把障碍物删掉后再跑一遍，用来判断障碍物到底有没有起作用 | ✅ 请确认 |
| falsifiable | 可证伪：能明确说出"出现什么结果就算没通过" | ✅ 请确认 |
| sign-only mapping | 只取符号的映射：只用指令的正负号，幅值被丢弃 | ✅ 请确认（issue #13 中文已这样用） |
| octant snapping | 八方向吸附：把线速度指令的方位角吸附到最近的八个方向之一 | ✅ 请确认 |
| axis primitive | 轴向原语：厂商接口里"一个方向一个固定原始值"的那一档指令 | ✅ 请确认（issue #13 中文已这样用） |
| graduated command | 连续可变的速度指令（与"只有全速或停"相对） | ✅ 请确认（issue #13 中文已这样用） |
| checkpoint | 权重文件（`.npz`） | ✅ 请确认 |
| acceptance criterion | 验收标准 | — |
| gait floor / `--gait-floor` | 步态下限：能让机器人真正走起来的**最低**速度指令 | — （沿用 `commissioning/RUNBOOK.md`） |
| actuator gain / `--actuator-gain` | 执行器增益：实际速度 ÷ 指令速度 | — （沿用 `commissioning/RUNBOOK.md`） |
| loaded planning radius / `--robot-radius` | 负载状态下的规划半径（俯视外形的外接圆半径） | — （沿用 `commissioning/RUNBOOK.md`） |

---

## 3. Why — what the evidence actually says
## 3. 为什么要重训练 —— 证据究竟说了什么

**EN** — Four findings. A reader who has never followed this project can check every one of
them from a clean clone; the commands are given. Two of them are **not** the finding this
work started from, and one of them retires an earlier conclusion.

**中文** —— 四条结论。完全没有跟进过这个项目的人，也可以从一份干净 clone 逐条核对，命令
都给了。其中两条**不是**这项工作最初出发的那条结论，还有一条推翻了早先的判断。

### 3.1 The observation is 18 values, and none of them is an obstacle's velocity
### 3.1 观测只有 18 个数，其中没有任何一个是障碍物的速度

**EN** — The layout, from `integration/mappo_bridge.py:121-124` and the checkpoint's own
`metadata_json` (`actor_input_dim: 18`, recorded in `policy/PROVENANCE.md`):

```text
[x, y, vx, vy, x-gx, y-gy, lidar0 ... lidar11]
```

`vx, vy` are the **robot's own** velocity. There is no channel for an obstacle's velocity,
so a moving obstacle enters the network as an instantaneous disc wherever it happened to
be. `policy/physical_ai_mappo.py:40-41` pins `N_RAYS = 12` and `OBS_DIM = 18`, and the
adapter refuses a checkpoint that disagrees (`:314-322`).

Three conventions that a retrain must keep or must explicitly renegotiate:

* **`lidar` is proximity, not distance** — `lidar_range_vmas - range/scale`
  (`policy/physical_ai_mappo.py:492-493`). Bigger means closer; `0.000` means clear.
* **The rays are at fixed angles `2πi/N` in the run-local frame** frozen at the first
  `reset_run` tick (`:483-484`). The trained agent is holonomic and never rotates, so the
  fan does **not** turn with the robot's nose. Ray 0 is the heading the run started in.
* **Everything is divided by `meters_per_vmas_unit`** before it reaches the network.

**中文** —— 观测的排布见 `integration/mappo_bridge.py:121-124` 以及 checkpoint 自带的
`metadata_json`（`actor_input_dim: 18`，记录在 `policy/PROVENANCE.md`）。

`vx, vy` 是**机器人自己**的速度。**没有任何通道承载障碍物的速度**，所以一个运动的障碍物
只能以"它此刻恰好在哪"的瞬时圆盘形式进入网络。`policy/physical_ai_mappo.py:40-41` 把
`N_RAYS = 12`、`OBS_DIM = 18` 写死，适配层会拒绝与之不符的 checkpoint（`:314-322`）。

有三条约定，重训练要么保持不变，要么必须明确提出更改：

* **`lidar` 是接近度，不是距离**——`lidar_range_vmas - range/scale`
  （`policy/physical_ai_mappo.py:492-493`）。**值越大表示越近**，`0.000` 表示该方向没有
  东西。
* **射线固定在运行局部坐标系的 `2πi/N` 角度上**，该坐标系在第一个 `reset_run` 时刻冻结
  （`:483-484`）。训练出来的智能体是全向的、从不转身，所以**射线扇不随机身朝向转动**。
  ray 0 就是这次运行起步时的朝向。
* 所有量在进入网络前都要**除以 `meters_per_vmas_unit`**。

### 3.2 The 12-ray / 360° fan cannot resolve an aperture — this is issue #29's core
### 3.2 12 条射线 / 360° 的射线扇分辨不出通过间隙 —— 这是 issue #29 的核心

**EN** — Four hardware runs of 2026-08-18 drove the Go2 between two bins. Replayed through
the real checkpoint, no robot needed:

```bash
python3 evidence/2026-08-19-what-the-policy-sees/radius_latch.py
```

Its header states the geometry: **policy horizon 0.875 m, bin separation measured
1.27–1.43 m centre to centre.** The horizon is shorter than the aperture is wide, so:

| run | outcome | ticks with **both** bins inside the horizon |
| --- | --- | ---: |
| **11** | **threaded the gap** | **33 of 79** |
| 10 | stalled, pure strafe | **0** of 46 |
| 14 | stalled, retreating | **0** of 50 |
| 15 | walked into a flank | **0** of 42 |

(Run 11's 33 is the script's *converged (corrected)* column; its *latched (delivered)*
column reads 39. The three failing runs read **0** in **both** columns, which is why the
correction in §6.E does not rescue them.)

**0 of 138 driven ticks across the three failing runs.** Off the midline only the near bin
exists in the observation, so *"drive between two things"* is not a manoeuvre the policy can
choose. `evidence/2026-08-19-what-the-policy-sees/` holds a committed frame where both bins
are plainly visible in the camera at 1.7 m and 2.7 m and the observation handed to the
network is **twelve zeros**.

⚠️ `README.md:244` and `evidence/2026-08-18-threading-two-bins/README.md:205` both state
**0 of 137**; the script regenerates 46 + 50 + 42 = **138** driven ticks today. The
numerator — **0** — is the same either way and is what the acceptance criterion in §6 is
built on. See §10.

**The ray count is not what caused this**, and that is the finding worth carrying: no ray
count fixes an obstacle that is outside the horizon entirely.

**中文** —— 2026-08-18 有四次硬件实验，让 Go2 从两个垃圾桶中间穿过去。用真实 checkpoint
回放（不需要机器人）：命令见上。

脚本自己的表头给出几何关系：**策略感知视野 0.875 m，两桶中心距实测 1.27–1.43 m。**
**视野比间隙还短**，于是：三次失败的实验里，**两个桶同时进入视野的周期数是 0 / 46、
0 / 50、0 / 42——合计 138 个周期里 0 个**；唯一成功的第 11 次是 33 / 79
（第 11 次的 33 取自脚本的*半径已收敛（已修正）*那一列，*半径被锁死（交付版）*那一列是 39；
三次失败的实验在**两列里都是 0**，这也是第 6.E 节那项修正救不了它们的原因）。

偏离中线之后，观测里只存在离得近的那一个桶，所以**"从两个东西中间穿过去"根本不是这个
策略能选的动作**。`evidence/2026-08-19-what-the-policy-sees/` 里收录了一帧图：相机里
1.7 m 和 2.7 m 处两个桶清清楚楚，而送进网络的观测是**十二个零**。

⚠️ `README.md:244` 和 `evidence/2026-08-18-threading-two-bins/README.md:205` 写的是
**0 of 137**；今天脚本重新算出来的驱动周期数是 46 + 50 + 42 = **138**。分子都是 **0**，
第 6 节的验收标准建立在分子上。详见第 10 节。

**造成这个问题的不是射线条数**——这一条最值得记住：障碍物完全在视野之外时，射线加得再密
也没用。

### 3.3 The policy does not avoid. It goal-seeks and stops.
### 3.3 这个策略不会绕行。它只会朝目标走，然后停下。

**EN** — This is the finding that most changes what a retrain is for, and it is measured
three independent ways.

**In simulation, with the peer's exact position handed over every tick** — better
information than any sensor could supply — the policy sees it (ray proximity 0.176, well
inside the 0.875 m horizon) and **responds by stopping**: forward command to zero, closest
approach **0.194 m** at the worst crossing speed. `integration/mappo_bridge.py:125-127`,
pinned by `integration/test_mappo_bridge.py:191`.

**On hardware**, over the 2026-08-25 peer runs
(`evidence/2026-08-25-peer-runs/README.md:49-50`, regenerate with `python3 bearings.py`):

| | value |
| --- | ---: |
| corr(lateral command, **goal** distance remaining) | **+0.951** |
| corr(lateral command, **peer** range) | **+0.048** |

The leftward command **predates the peer**: +0.154…+0.166 m/s before the peer is tracked at
all, +0.147…+0.153 m/s while it is tracked. The peer's arrival did not increase it; it
slightly *decreased* it. The run that "cleared a peer robot" cleared it because the goal
happened to sit off the peer's bearing — put the peer **on** the goal bearing and the same
policy drives through it, which is the contrast video kept in the same directory.

**In the steering response itself** (`policy/README.md`): **0.1°** mean deflection while
the obstacle is outside the horizon, **103°** once inside — *"a cliff, not a ramp"* — and
that magnitude is **saturated at every scale from 1.5 to 4.0 m/unit**. Raising the scale
buys *warning*, never proportionality.

So the thing to retrain for is not "see the gap better". It is **produce a graduated,
obstacle-attributed manoeuvre at all**. Hold that thought until §5.

**中文** —— 这一条最能改变"重训练到底为了什么"，而且是三条互相独立的证据。

**在仿真里，把同伴机器人的精确位置每个周期都直接喂给策略**——比任何传感器都强——策略
确实"看见"了（射线接近度 0.176，远在 0.875 m 视野之内），但**它的反应是停车**：前向指令
归零，最恶劣穿越速度下最近距离 **0.194 m**。见 `integration/mappo_bridge.py:125-127`，
由 `integration/test_mappo_bridge.py:191` 固化。

**在硬件上**，2026-08-25 的同伴机器人实验（`evidence/2026-08-25-peer-runs/README.md:49-50`，
用 `python3 bearings.py` 复现）：横向指令与**目标剩余距离**的相关系数是 **+0.951**；与
**同伴机器人距离**的相关系数是 **+0.048**。

而且那个向左的指令**早于同伴机器人出现**：还没跟踪到同伴时是 +0.154…+0.166 m/s，跟踪到
之后是 +0.147…+0.153 m/s。同伴出现没有让它变大，反而略微变小。那次"避开了同伴机器人"的
实验之所以避开了，是因为目标恰好不在同伴的方位上——把同伴**放到目标方位上**，同一个策略
就直接撞上去，同目录下的对比视频就是这个。

**从转向响应本身看**（`policy/README.md`）：障碍物在视野之外时平均偏转 **0.1°**，一进入
视野就是 **103°**——*"是悬崖，不是斜坡"*——而且这个幅度在 1.5 到 4.0 m/unit 的**每一个
尺度上都是饱和的**。加大尺度只能买到**提前预警**，永远买不到**比例响应**。

所以重训练要解决的不是"把间隙看得更清楚"，而是**让它产生一个真正由障碍物引起的、连续可变
的机动动作**。这个念头请一直带到第 5 节。

### 3.4 The one Lite3 measurement: the obstacle changed the command by nothing
### 3.4 唯一一项 Lite3 上的测量：障碍物对指令的影响是零

**EN** — Replayed at the Lite3's own **run 2** geometry — robot 0.54 m in, box at 1.18 m
ahead and −0.10 m lateral — with the box present, then with it deleted (issue #13):

| scale | horizon | box | vx | vy | rays > 0 |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2.0 | 0.700 m | removed | +0.331 | +0.031 | 0 |
| 2.0 | 0.700 m | **present** | **+0.331** | **+0.031** | **0** |
| 2.5 | 0.875 m | removed | +0.337 | +0.038 | 0 |
| 2.5 | 0.875 m | **present** | **+0.337** | **+0.038** | 1 (1.1%) |

**Identical to three decimal places.** The box's *surface* sat at 1.184 − 0.28 =
**0.904 m**, outside the 0.700–0.875 m sensing horizon
(`lidar_range_vmas × meters_per_vmas_unit`), and every one of the twelve lidar values was
exactly `0.0`. **The robot was not walking toward the box — it was walking toward the
chair, and the box was on that bearing.**

The +0.031 m/s of *leftward* `vy`, away from a box that was 0.10 m to the *right*, is the
known systematic lateral bias, and it is present identically with **no box in the scene at
all**.

At scale 4.0 the policy does see the box, and its measured response is to **slow by 18%**
(0.342 → 0.281) — not to turn. Walking the box closer: vx +0.338 → +0.021 at 0.476 m →
**−0.231 at 0.377 m**, a reverse that `mappo_drive` clamps to 0. **It stops and backs off.
It never goes around.** That is §3.3 again, on this robot's own scene.

**中文** —— 按 Lite3 **第 2 次实验**的几何回放——机器人已前进 0.54 m，箱子在正前方
1.18 m、横向 −0.10 m——分别在"有箱子"和"把箱子删掉"两种情况下跑（见 issue #13）：结果见
上表。

**小数点后三位完全一致。** 箱子**表面**在 1.184 − 0.28 = **0.904 m** 处，落在
0.700–0.875 m 的感知视野之外（视野 = `lidar_range_vmas × meters_per_vmas_unit`），
十二个 lidar 值全部恰好是 `0.0`。**机器人不是朝箱子走，而是朝椅子走，箱子正好在那个方位
上。**

那 +0.031 m/s 的**向左** `vy`——方向恰好背离位于**右侧** 0.10 m 的箱子——是已知的系统性
横向偏置，在**场景里完全没有箱子**时也一模一样地存在。

把尺度加到 4.0，策略确实看见箱子了，实测反应是**减速 18%**（0.342 → 0.281），**不是
转向**。把箱子往近处挪：vx 从 +0.338 → 0.476 m 处的 +0.021 → **0.377 m 处的 −0.231**
（倒车，被 `mappo_drive` 钳到 0）。**它会停下、会后退，但从不绕行。** 这就是第 3.3 节的
结论，在这台机器人自己的场景上再现了一次。

---

## 4. What to change: the observation and the ray fan
## 4. 要改什么：观测与射线扇

**EN** — Two changes, in priority order. Both are stated in the units the training config
uses; §4.3 explains why the metre value cannot be pinned yet on this robot.

**中文** —— 两项改动，按优先级排列。都用训练配置本身的单位表述；第 4.3 节解释为什么在这台
机器人上还无法把它换算成米。

### 4.1 Item 1 — the horizon. `training_lidar_range_vmas` 0.35 → 0.80
### 4.1 第一项 —— 感知视野。`training_lidar_range_vmas` 0.35 → 0.80

**EN** — This is the top ask, and §3.2 is the whole argument for it: an obstacle outside
the horizon is indistinguishable from clear floor, and no ray count changes that. ⚠️ At the
Go2's calibrated 2.5 m/unit, issue #29 records that 0.80 puts both bins in range on
**35/49, 38/41 and 43/45** of the three failing runs' ticks against 0 at 0.35, and that
1.00 covers everything if it is free. Those three denominators are not the 46/50/42 the
replay script prints for driven ticks; they are a different tick selection and are quoted
here as attributed rather than re-derived. See §10.

**⛔ Do not raise `meters_per_vmas_unit` instead.** It is not a free knob: it is calibrated
as the planner's loaded robot radius ÷ the trained `training_agent_radius_vmas` of 0.10
(`policy/PROVENANCE.md`; `robot-stack/deep_robotics/lite3/README.md` states the same rule
as *"The MAPPO scale is `robot_radius_m / 0.10`"*). Moving it de-calibrates the agent's own
size — the policy starts believing it is a different width than the planner does.

**中文** —— 这是**第一优先**的需求，第 3.2 节就是它的全部论据：视野之外的障碍物与空地在
观测上完全无法区分，射线加多少条都改变不了这一点。⚠️ 在 Go2 已标定的 2.5 m/unit 下，
issue #29 记录：0.80 让三次失败实验中"两个桶同时在视野内"的周期数分别达到 **35/49、
38/41、43/45**（0.35 时是 0）；如果 1.00 不额外增加成本，1.00 能覆盖全部情况。这三个分母
与回放脚本给出的驱动周期数 46/50/42 不一致，是另一种周期筛选口径，此处按**引用**列出，
未重新推导。详见第 10 节。

**⛔ 不要改用提高 `meters_per_vmas_unit` 的办法。** 它不是一个自由参数：它是按"规划器使用
的负载状态机器人半径 ÷ 训练时的 `training_agent_radius_vmas`（0.10）"标定出来的
（`policy/PROVENANCE.md`；`robot-stack/deep_robotics/lite3/README.md` 也写了同一条规则：
*"MAPPO 尺度 = `robot_radius_m / 0.10`"*）。动它就会让智能体自身尺寸失准——策略会认为自己
的宽度和规划器认为的不一样。

### 4.2 Item 2 — 24 rays / 360°, and drop 16 from the fallback
### 4.2 第二项 —— 24 条射线 / 360°，并且把 16 条这个备选去掉

**EN** — 24 rays over a full circle is 15° spacing against the delivered 30°. The reasoning
is not "more is better"; it is a measured table. Regenerated on `77d1d28` with
`radius_latch.py`, **converged (corrected) radii**, counting ticks where any ray of an
N-ray fan lands inside the clear angular window toward the goal:

| fan | run 10 | run 14 | run 15 | run 11 |
| --- | ---: | ---: | ---: | ---: |
| **12 rays** (delivered) | 34/46 | **3/50** | 41/42 | 79/79 |
| **16 rays** | 37/46 | **33/50** | 41/42 | 79/79 |
| **24 rays** | 46/46 | **48/50** | 41/42 | 79/79 |

Run 14 is the discriminating run. Its clear window sat at **[−29.3°, −0.7°]** — between ray
11 at −30° and ray 0 at 0°, missing each of them by under a degree. Run 11, the one that
worked, had the 12-ray fan sampling its window on **every** tick: the fan was never its
problem, its approach line happened to put the aperture on a ray.

**Why 16 is not worth buying:** with the **latched** (uncorrected) radii — which is what
every hardware run actually flew — a 16-ray fan finds run 14's window on **1 of 27** open
ticks, exactly as the 12-ray fan does. It is measurably no better than 12 on the run that
failed. 24 rays find it on 15 of 27. If 24 is expensive, take the **horizon first and the
rays second**.

⚠️ `radius_latch.py` on `77d1d28` prints **41/42** for run 15 at 24 rays; issue #29's
comment table prints 42/42. See §10.

**中文** —— 24 条射线覆盖 360°，相邻间隔 15°，而交付版本是 30°。理由不是"越多越好"，
而是一张实测表。在 `77d1d28` 上用 `radius_latch.py` 重新生成，取**半径已收敛（已修正）**
那一列，统计"N 条射线中是否有任意一条落进朝向目标的空旷角窗"的周期数——见上表。

**第 14 次实验是关键。** 它的空旷角窗落在 **[−29.3°, −0.7°]**——正好卡在 −30° 的 ray 11
和 0° 的 ray 0 之间，两边都差不到 1°。而唯一成功的第 11 次实验，12 条射线在**每一个周期**
都采到了角窗：射线扇从来不是它的问题，只是它的接近路线恰好让间隙落在了某条射线上。

**为什么不值得买 16 条：** 在**半径被锁死（未修正）**的情况下——也就是所有硬件实验实际
飞的那个状态——16 条射线在第 14 次实验里 27 个开放周期中只命中 **1 个**，和 12 条一模一样。
在真正失败的那次实验上，它**可测量地没有任何改善**。24 条能命中 15/27。如果 24 条成本太
高，请**先做视野，再做射线**。

⚠️ `radius_latch.py` 在 `77d1d28` 上给出的第 15 次实验 24 射线结果是 **41/42**，而
issue #29 评论里的表格写的是 42/42。详见第 10 节。

### 4.3 What is *not* being asked for, and why the metre value is open
### 4.3 **没有**提出的要求，以及为什么"多少米"这个数还悬着

**EN** — Not asked for: an obstacle-velocity channel. It would be the right fix for §3.3's
crossing-peer case, but it changes `OBS_DIM` twice over and this side has a working
workaround — the bridge grows a moving obstacle's disc by `speed × horizon` so the ray cast
reports where the peer *will* be, measured to lift clearance at 0.20 m/s from 0.194 m to
0.505 m (`integration/mappo_bridge.py`). **If it is cheap in training, say so** and it goes
to the top of a future revision of this document.

**Why the horizon cannot yet be stated in metres for this robot:** the horizon in metres is
`training_lidar_range_vmas × meters_per_vmas_unit`, and `meters_per_vmas_unit` is
`loaded_radius_m / 0.10`. **The Lite3's loaded planning radius is unmeasured** (§7). So
0.80 buys a 2.0 m horizon at the Go2's 2.5 m/unit, and something else on the Lite3 — the
same VMAS number is a different distance on a robot of a different width. Whoever executes
this should train the VMAS value **and record it**, and the metre value gets computed on
this side once the radius exists.

**中文** —— **没有**提出的要求：给障碍物速度加一个通道。对第 3.3 节的"横穿同伴"场景来说
它才是正解，但它会两次改变 `OBS_DIM`，而且我们这边已有可用的规避手段——桥接层把运动障碍物
的圆盘按 `speed × horizon` 放大，使射线报出同伴**将要**在的位置，实测把 0.20 m/s 下的
最近距离从 0.194 m 提高到 0.505 m（`integration/mappo_bridge.py`）。**如果在训练侧代价很
低，请告诉我们**，它会进入本文档下一版的首位。

**为什么现在还不能用"米"来表述这台机器人的视野：** 以米计的视野 =
`training_lidar_range_vmas × meters_per_vmas_unit`，而 `meters_per_vmas_unit` =
`loaded_radius_m / 0.10`。**Lite3 负载状态下的规划半径尚未测量**（第 7 节）。所以 0.80 在
Go2 的 2.5 m/unit 下是 2.0 m 视野，在 Lite3 上则是另一个数——同一个 VMAS 数值，在不同宽度
的机器人上对应不同的实际距离。执行方请按 VMAS 单位训练**并记录该数值**；等半径测出来之后，
米制数值由我们这边换算。

---

## 5. ⚠️ The transport constraint — read this before specifying anything
## 5. ⚠️ 传输链路的限制 —— 在确定任何规格之前先读这一节

**EN** — **This is the single most important thing in this document.** A retraining spec
that ignores it is specifying a policy this platform cannot execute.

**中文** —— **这是本文档最重要的一条。** 忽略它的重训练规格，等于在规定一个这个平台根本
执行不了的策略。

### 5.1 What the Lite3 can actually execute today
### 5.1 Lite3 今天到底能执行什么

**EN** — The transport both Ventures have actually walked on is
`--locomotion-transport axis`. `lite3_axis_locomotion.py:252` says it in the code:

> *"**This mapping is sign-only: the commanded magnitude is discarded.** A profile holds
> one evidenced raw value per direction, so every command past the deadband leaves at
> whatever speed that one primitive was measured to produce. Nothing here scales with
> `vx`…"*

And `_linear_direction` gates the linear pair on `hypot(vx, vy)` and then **snaps its
bearing to the nearest of the eight** `(forward, lateral)` sign pairs in
`_LINEAR_DIRECTIONS` (`:60-61`). What that does to three real commands (issue #13):

| command | bearing | what reaches the legs |
| --- | ---: | --- |
| policy at run 2 `(+0.332, +0.030)` | 5.2° | octant 0 — **pure forward, full primitive speed** |
| a planner swerve `(+0.30, +0.20)` | 33.7° | octant 1 — 45° diagonal, full speed |
| a planner creep after a hold `(+0.05, 0)` | 0.0° | octant 0 — **full speed, not a creep** |

`robot-stack/deep_robotics/lite3/README.md` states the same thing from the other end:
`--derate` **does not reach the wire** on this transport —

| `--derate` | commanded `vx` | forward axis emitted |
| ---: | ---: | ---: |
| 1.0 | 0.300 m/s | `+32767` |
| 0.6 | 0.180 m/s | `+32767` |
| 0.3 | 0.090 m/s | `+32767` |
| 0.2 | 0.060 m/s | `+32767` |

**The only command this transport delivers faithfully is the stop.** This is also why
issue #13's *"gait floor"* and *"actuator gain"* have no answer here: a gait floor is the
lowest *commanded* speed that still walks, and there is one command per direction; an
actuator gain is delivered ÷ commanded, and the denominator never left the laptop.
`gait_floor_probe.py` and `actuator_gain_probe.py` now refuse `--locomotion-transport axis`
**by name**, because a descending ladder pointed at it does not fail — every rung fires the
same primitive, every rung walks at the same speed, and the probe reports the bottom rung
as the floor.

**中文** —— 两台 Venture 实际走起来用的链路是 `--locomotion-transport axis`。
`lite3_axis_locomotion.py:252` 在代码里就写明了：*"**该映射只取符号，指令幅值被丢弃。**
配置里每个方向只有一个有实测证据的原始值，所以任何超过死区的指令，出去的都是那一个原语
被实测出来的速度。这里没有任何东西随 `vx` 缩放……"*

而且 `_linear_direction` 先用 `hypot(vx, vy)` 做门限，再把方位角**吸附到
`_LINEAR_DIRECTIONS`（`:60-61`）里八个 `(forward, lateral)` 符号对中最近的一个**。三条
真实指令的结果见上表：规划器的"小步慢挪"到腿上会变成**全速前进**。

`robot-stack/deep_robotics/lite3/README.md` 从另一端说了同一件事：在这条链路上
`--derate` **根本到不了总线**——1.0 / 0.6 / 0.3 / 0.2 四档发出去的都是同一个 `+32767`。

**这条链路唯一能忠实传达的指令是"停"。** 这也正是 issue #13 里的"步态下限"和"执行器增益"
在这里没有答案的原因：步态下限是"仍能走起来的**最低指令速度**"，而这里每个方向只有一个
指令；执行器增益是"实际 ÷ 指令"，而分母从来没离开过笔记本电脑。`gait_floor_probe.py` 和
`actuator_gain_probe.py` 现在会**指名拒绝** `--locomotion-transport axis`——因为拿一个
递降阶梯去测它**不会失败**：死区以上每一档都触发同一个原语、每一档速度都一样、每一项检查
都通过，工具会把最低档误报成步态下限。

### 5.2 Therefore: what a retrained policy must produce to be useful
### 5.2 因此：重训练出来的策略要有用，必须输出什么

**EN** — Stated plainly, because the question was asked plainly:

**Is a finer ray fan worth anything before the transport can express a graduated command?**

**Partly, and less than it looks — and the headline benefit is unreachable.**

* **What survives the transport:** *direction* (one of eight) and *stop*. A finer fan
  improves **which octant** the policy picks and **when it decides to stop**. Those two
  things do reach the legs, so a 24-ray retrain is not worthless on the axis transport.
* **What does not survive:** *magnitude*. The behaviour a retrain is otherwise being bought
  for — replacing the measured **cliff** (0.1° outside the horizon, 103° inside, saturated
  at every scale; §3.3) with a **ramp** — is expressed entirely in graduated speeds, and
  graduated speeds are discarded at `lite3_axis_locomotion.py:252`. **A proportional
  avoidance policy cannot be executed by this robot on this transport.** Training one and
  running it on `axis` would deliver, at best, a better-chosen full-speed octant.
* **The corollary nobody should skip:** the planner's whole design is *swerve early, stop
  late*, and it expresses that in graduated speeds too. Its `(+0.05, 0)` creep after a hold
  leaves as full speed. Retraining the policy does not fix that; it is a property of the
  wire.

**So the retrain must first be told which transport it is for**, and the three answers are
genuinely different specifications:

| transport | magnitude reaches the wire? | what to train |
| --- | --- | --- |
| `axis` (both Ventures have walked on this) | **no** | an action space **quantised to the eight octants plus stop**. Make the policy's output the thing the wire can carry, and let the reward see that a "slow approach" is not available |
| `udp` (legacy complex-velocity, codes 320/325/321) | yes | the graduated policy this document otherwise describes |
| `ros2` (`Lite3_ROS` `/cmd_vel` + `/leg_odom2`) | yes | same as `udp`; needs a ROS 2 Foxy perception host these two Ventures may not have |

**Recommendation:** answer the transport question **before** commissioning any retrain, and
measure the **(1,1) diagonal primitive's speed and bearing** first — issue #13 §6 —
*"Everything about whether this robot can ever sidestep turns on that pair, and it is
currently unmeasured."* `commissioning/axis_primitive_probe.py` is the tool, and it refuses
a primitive that moves the robot the wrong way.

**中文** —— 直接回答被直接问到的那个问题：

**在传输链路还无法表达连续可变指令之前，把射线扇加密有没有价值？**

**部分有，但比看上去少得多——而且最主要的那份收益拿不到。**

* **能通过链路的：** *方向*（八选一）和*停车*。加密射线扇能改善策略**选哪个方向**、以及
  **什么时候决定停**。这两件事确实能到腿上，所以在 axis 链路上，24 射线的重训练**不是
  完全没有价值**。
* **通不过链路的：** *幅值*。重训练本来要买的核心收益——把实测的**悬崖**（视野外 0.1°、
  视野内 103°、每个尺度都饱和；见第 3.3 节）换成一条**斜坡**——完全是靠连续可变的速度来
  表达的，而这些速度在 `lite3_axis_locomotion.py:252` 处被丢弃。**一个比例式避障策略，
  在这条链路上、在这台机器人上无法被执行。** 训练出来放到 `axis` 上跑，最好的情况也只是
  "全速冲进一个选得更好的方向"。
* **不能跳过的推论：** 规划器的整体设计是"早绕行、晚停车"，同样靠连续可变的速度表达。它在
  保持之后发出的 `(+0.05, 0)` 慢挪，到腿上是全速。**重训练策略修不好这一点**——这是总线的
  属性。

**所以重训练之前必须先确定它面向哪条链路**，三个答案对应的是三份实质不同的规格：见上表。

**建议：** 在为任何重训练立项**之前**先回答链路问题，并**先测 (1,1) 对角原语的实测速度与
实测方位角**——issue #13 第 6 节：*"这台机器人到底能不能侧移，全取决于这一对数值，而目前
尚未测量。"* 工具是 `commissioning/axis_primitive_probe.py`，它会拒绝"把机器人带向错误
方向"的原语。

---

## 6. Acceptance criteria — falsifiable, and none of them needs a robot
## 6. 验收标准 —— 可证伪，而且没有一条需要动用机器人

**EN** — A spec without a test it must pass is a wish. Every criterion below is a desk
check against telemetry already committed to this repository. **A candidate checkpoint is
replayed and scored before anyone books robot time.**

**中文** —— 没有"必须通过的测试"的规格书只是愿望。下面每一条都是针对本仓库已收录的遥测
数据做的**桌面检查**。**候选 checkpoint 先回放打分，然后才谈预约机器人时间。**

### A. Both obstacles in range on more than 0 ticks — the primary gate
### A. 两个障碍物同时在视野内的周期数 > 0 —— 主门限

```bash
python3 evidence/2026-08-19-what-the-policy-sees/radius_latch.py
```

**EN** — The script reads the horizon from `policy/config.json` — `lidar_range_m` is
`lidar_range_vmas × meters_per_vmas_unit` (`policy/physical_ai_mappo.py:227`) — so score a
candidate by setting `lidar_range_vmas` to its `training_lidar_range_vmas` and
`meters_per_vmas_unit` to the **measured Lite3** value (§7), then re-running. **This
criterion is pure geometry** and does **not** need the candidate weights to load, so it can
be scored before the adapter is parameterised (§9).

**PASS requires strictly more than 0** of the driven ticks on **each** of runs 10 (46
ticks), 14 (50) and 15 (42). Today's value is 0, 0, 0. This is the gate §3.2 exists for, it
is the one issue #29 is actually about, and it fails loudly if the horizon change is
dropped.

**中文** —— 脚本从 `policy/config.json` 读取视野——`lidar_range_m` 就是
`lidar_range_vmas × meters_per_vmas_unit`（`policy/physical_ai_mappo.py:227`）——所以评分
时把 `lidar_range_vmas` 设为候选权重的 `training_lidar_range_vmas`，把
`meters_per_vmas_unit` 设为 **Lite3 实测值**（第 7 节），然后重跑。**这一条是纯几何
判定**，**不需要**候选权重能被加载，因此在适配层参数化（第 9 节）之前就可以评分。

**通过的条件是：第 10 次（46 个周期）、第 14 次（50 个）、第 15 次（42 个）实验中，
每一次都必须严格大于 0。** 现在是 0、0、0。这就是第 3.2 节存在的意义，也是 issue #29
真正要解决的问题；一旦视野这项改动被砍掉，它会**明确地失败**。

### B. A ray lands in run 14's clear window on ≥ 48 of 50 driven ticks
### B. 第 14 次实验中，有射线落进空旷角窗的周期数 ≥ 48 / 50

**EN** — Same script, same geometry-only property: it counts, per tick, whether **any ray
of an N-ray fan** lands inside the clear angular window toward the goal, for N = 12, 16 and
24. Run 14's window is **[−29.3°, −0.7°]** and it is the discriminating geometry. The
delivered 12-ray fan scores **3/50**; a 24-ray fan scores **48/50**. A candidate whose
`n_rays` does not reach 48/50 has not bought what §4.2 asked for. Report runs 10, 15 and 11
alongside it, and **do not regress run 11 below 79/79.** If the candidate's `n_rays` is not
one of 12/16/24, add it to the script's fan list rather than interpolating.

**中文** —— 同一个脚本，同样是纯几何判定：它逐周期统计"**N 条射线中是否有任意一条**落进
朝向目标的空旷角窗"，N 取 12、16、24。第 14 次实验的角窗是 **[−29.3°, −0.7°]**，是最有
区分度的几何。交付版 12 射线得 **3/50**；24 射线得 **48/50**。候选权重的 `n_rays` 达不到
48/50，就没有买到第 4.2 节要的东西。同时报出第 10、15、11 次的数值，并且**第 11 次不得从
79/79 退步**。如果候选的 `n_rays` 不是 12/16/24 之一，请把它加进脚本的射线数列表，**不要
插值**。

### C. The ablated control must change the command — the avoidance gate
### C. 对照回放必须改变输出指令 —— 避障门限

```bash
cd integration && python3 replay_mappo.py <telemetry.jsonl> --scale <the run's real scale>
```

**EN** — Replay a run twice: once as recorded, once with the obstacle deleted. On the Lite3
run-2 geometry the delivered checkpoint produces `(+0.331, +0.031)` **both times, identical
to three decimal places** (§3.4). **PASS requires the two commands to differ**, and the
difference to be in the direction of clearance. This is the cleanest falsifier for *"does
it avoid, or does it goal-seek?"* — and it is the criterion that the delivered checkpoint
most clearly fails.

⚠️ **This one does need the weights to load**, so it is blocked until `N_RAYS`/`OBS_DIM`
are read from the checkpoint's metadata (§9). A 24-ray `.npz` against today's adapter fails
at load by design (`policy/physical_ai_mappo.py:314-322`) — that is the gate working, not a
delivery defect. Run it at the run's **real** `meters_per_vmas_unit`; a different scale is a
different horizon and describes nothing.

**中文** —— 把同一段数据回放两次：一次照原样，一次把障碍物删掉。在 Lite3 第 2 次实验的
几何下，交付版 checkpoint 两次都输出 `(+0.331, +0.031)`，**小数点后三位完全一致**
（第 3.4 节）。**通过的条件是：两次指令必须不同**，而且差异方向要是"让开"。这是对
*"到底是避障还是趋目标"* 最干净的证伪测试——也是交付版最明确不通过的一条。

⚠️ **这一条确实需要权重能被加载**，因此在 `N_RAYS` / `OBS_DIM` 改为从 checkpoint metadata
读取（第 9 节）之前无法执行。24 射线的 `.npz` 配今天的适配层会在加载时失败，这是**设计如此**
（`policy/physical_ai_mappo.py:314-322`）——是门限在起作用，不是交付物有问题。运行时必须用
该次实验**真实的** `meters_per_vmas_unit`；尺度不同就是视野不同，结果没有意义。

### D. Report both correlations, computed the same way
### D. 用同样的方法报出两个相关系数

```bash
python3 evidence/2026-08-25-peer-runs/bearings.py
```

**EN** — corr(lateral command, obstacle range) is **+0.048** today, against **+0.951** for
corr(lateral command, goal distance remaining). **No threshold is set here on purpose** —
inventing one would be inventing a number. The requirement is directional and must be
stated in the result: **the obstacle term must stop being indistinguishable from zero.**
Report both figures for every candidate.

**中文** —— 目前"横向指令 vs 障碍物距离"是 **+0.048**，"横向指令 vs 目标剩余距离"是
**+0.951**。**这里刻意不设阈值**——凭空定一个阈值就是凭空造数。要求是方向性的，并且必须
写进结果里：**障碍物那一项必须不再"与零无法区分"。** 每个候选都要同时报出这两个数。

### E. ⚠️ The criterion in issue #29's body has been overtaken — do not use it alone
### E. ⚠️ issue #29 正文里的那条验收标准已经过时 —— 不要单独使用

**EN** — Issue #29's body says: *"Replaying run 14's telemetry through the new checkpoint …
the commanded `vx` should be positive rather than negative."* **That test now passes
without any retrain.** `CORRECTION 6` in `policy/PROVENANCE.md` stopped the retained
obstacle radius being a high-water mark, and on the same recorded poses run 14's mean
commanded `vx` moved **−0.007 → +0.198 m/s** (runs 10 and 15: −0.037 → +0.265,
−0.039 → +0.271). A criterion the current code already satisfies cannot discriminate a new
checkpoint. Keep it as a **regression guard** — it must not go negative again — but A, B, C
and D are the gates.

**中文** —— issue #29 正文写的是：*"用新 checkpoint 回放第 14 次实验的遥测……指令 `vx`
应该为正而不是为负。"* **这条测试现在不用重训练就已经通过了。**
`policy/PROVENANCE.md` 里的 `CORRECTION 6` 修正了"保留的障碍物半径只增不减"的问题，在同一
批记录位姿上，第 14 次实验的平均指令 `vx` 从 **−0.007 变成 +0.198 m/s**（第 10 次：
−0.037 → +0.265；第 15 次：−0.039 → +0.271）。一条**现有代码已经满足**的标准，无法区分
新旧 checkpoint。把它保留为**回归看护**（不得再次变负）即可；真正的门限是 A、B、C、D。

### F. ⚠️ No hardware acceptance criterion is defined, deliberately
### F. ⚠️ 刻意不定义任何硬件验收标准

**EN** — Because of §5. Until the transport question is answered, a hardware run of a
graduated policy on the `axis` transport measures the transport, not the policy. Adding a
hardware gate now would produce a number that describes nothing.

**中文** —— 原因见第 5 节。在链路问题有答案之前，把一个"连续可变"的策略放到 `axis` 链路上
跑硬件，测到的是链路而不是策略。现在加硬件门限，只会得到一个什么都不说明的数字。

---

## 7. What is unmeasured, and must be supplied before training can be calibrated
## 7. 尚未测量、且必须在训练标定之前补齐的量

**EN** — From `robot-stack/deep_robotics/lite3/README.md`'s own platform table and
issue #13. **None of the Go2's numbers are defaults here**, and the Lite3 path currently
reaches `max_vx = 0.35` / `max_vy = 0.2` with no flag — numerically the Go2's
`MIN_GAIT_COMMAND_M_S`, whose own comment reads *"MEASURED 2026-08-14, on carpet, with the
3.15 kg D1 arm fitted"*, stamped into telemetry under
`platform.name: deep-robotics-lite3-venture` with no attribution (issue #13).

| quantity | status on the Lite3 | why the retrain needs it |
| --- | --- | --- |
| **loaded planning radius** (`--robot-radius`) | **not measured** | it *is* `meters_per_vmas_unit` — the scale is `robot_radius_m / 0.10`. Without it, `training_lidar_range_vmas` cannot be turned into a horizon in metres (§4.3), and the trained agent's own size is uncalibrated |
| **`meters_per_vmas_unit`** | **derived, therefore also unknown** | see above. The Go2's 2.5 must not be inherited |
| **gait floor** (`--gait-floor`) | **not measured — and not measurable on the `axis` transport** | a reward that shapes toward slow, careful approach is shaping toward commands that may not walk |
| **actuator gain** (`--actuator-gain`) | **not measured — same reason**; the denominator never reaches the wire | decides how much of a commanded manoeuvre is real, and therefore `--max-seconds` |
| **axis primitive speeds** (`measured_m_s`), incl. the **(1,1) diagonal** | **not measured** | §5. Decides whether a sidestep exists at all on this robot |
| **camera focal length / HFOV / mount pitch** | **provisional only** — static green-panel fit, `focal_px = 469.6297`, `pitch_rad = -0.2625193` (−15.04°), `height_m = 0.40`, `calibration_status = provisional`, explicitly refused by `--live` | sets detection range, which sets whether an obstacle is mapped in time to be inside any horizon |
| **camera near limit** | **not computed** | on the Go2, `object_fit_range = camera_height / tan(half_vfov) = 0.32 / tan(0.4185) = 0.719 m` — pure mounting geometry. The Lite3's camera sits at a different height, so its near limit is a different number nobody has computed, and the symptom will be *"the robot will not approach closer than X"* (issue #29) |
| **motor temperatures** | **absent from the high-level interface** | bounds run length; no software here can see accumulated heat |
| **forward/lateral gait floors are two numbers, not one** | measured on the **Go2** only: forward **0.35 m/s**, lateral in **(0.15, 0.25]**, lateral delivery ~27% of commanded against ~74% forward ([issue #42](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/42)) | the Lite3 has neither measured, and the calibration interface has one field |

**One Lite3 number that *is* measured**, and it is the only one: a bounded full-forward
`+32767` axis pulse of ≤ 1 s produced **0.401080 m** of world displacement at a peak body-x
velocity of **0.728896 m/s**, `error_state = 0`
(`lite3-chair-bin-live-readiness-20260825.md`). Separately, **0.54 m in 1.2 s ≈ 0.45 m/s**
is a measurement of `forward_positive` — not evidence that a 0.55 m/s command was tracked,
because on a sign-only wire that ramp never happened (issue #13).

**中文** —— 出处是 `robot-stack/deep_robotics/lite3/README.md` 自己的平台表和 issue #13。
**Go2 的数字在这里一个都不能当默认值**；而目前 Lite3 路径在不加任何参数的情况下就会拿到
`max_vx = 0.35` / `max_vy = 0.2`——这两个数在数值上就是 Go2 的 `MIN_GAIT_COMMAND_M_S`，
其注释写的是*"2026-08-14 实测，地毯上，装有 3.15 kg 的 D1 机械臂"*，却被原样写进遥测的
`platform.name: deep-robotics-lite3-venture` 字段里，没有任何来源标注（issue #13）。

上表逐项对应：**负载规划半径**未测（它就是 `meters_per_vmas_unit`，尺度 =
`robot_radius_m / 0.10`，缺了它就无法把 `training_lidar_range_vmas` 换算成米，见第 4.3
节）；**`meters_per_vmas_unit`** 因此同样未知，Go2 的 2.5 不能继承；**步态下限**和
**执行器增益**未测，而且**在 `axis` 链路上根本无法测**；**各轴向原语的实测速度**（尤其是
**(1,1) 对角原语**）未测，它决定这台机器人到底存不存在"侧移"这个动作；**相机焦距 / 视场角
/ 安装俯仰**只有**待审**的静态标定值，`--live` 会明确拒绝；**相机近距极限**未计算——Go2 上
是 `0.32 / tan(0.4185) = 0.719 m`，纯粹由安装几何决定，Lite3 相机高度不同，这个数还没人
算过，表现出来就是*"机器人不肯靠得比 X 更近"*；**电机温度**在高层接口里根本没有；
**前向与横向步态下限是两个数不是一个**，而且只在 **Go2** 上测过。

**Lite3 上唯一测到的数**：一次有界的全速前向 `+32767` 轴向脉冲（≤ 1 s）产生世界坐标位移
**0.401080 m**，机身 x 向峰值速度 **0.728896 m/s**，`error_state = 0`
（`lite3-chair-bin-live-readiness-20260825.md`）。另外，**1.2 秒走 0.54 m ≈ 0.45 m/s**
是对 `forward_positive` 的一次测量——而不是"0.55 m/s 的指令被跟踪了"的证据，因为在只取符号
的总线上那段加速过程根本没发生（issue #13）。

---

## 8. ⛔ Not to be started yet — what has to be true first
## 8. ⛔ 暂不启动 —— 需要先满足什么

**EN** — In order. Items 1 and 2 are decisions; 3 to 5 are measurements; 6 is an agreement
with the training repository's owner.

1. **This document is filed, not commissioned.** Direction from the project owner: no more
   Go2 retraining; all new retraining is for the Lite3. Nothing here authorizes a training
   run.
2. **The transport decision (§5).** `axis`, `udp` or `ros2`. Three genuinely different
   specifications. Answer this first, because it decides what the action space should be.
3. **The (1,1) diagonal primitive's measured speed and bearing.**
   `commissioning/axis_primitive_probe.py`. Whether this robot can sidestep at all turns on
   this one pair.
4. **The loaded planning radius**, which is `meters_per_vmas_unit`, which is the only thing
   that turns a VMAS horizon into metres.
5. **A reviewed, non-provisional camera calibration** (`calibration_status=validated`),
   which needs a second independently measured static observation.
6. **The training repository owner's acceptance of the ask** — including whether an
   obstacle-velocity channel is cheap (§4.3), whether `training_max_steps` can rise above
   100, and which of the three transports the checkpoint is being trained for.

`training_max_steps` is worth its own line: it is **100**, which at the stack's 10 Hz
control rate is a **ten-second** episode (`policy/PROVENANCE.md`). Demo runs are 40–60 s —
four to six times anything the policy saw in training. Longer episodes would help
independently of the fan and of the horizon.

**中文** —— 按顺序。第 1、2 项是决策；第 3–5 项是测量；第 6 项是与训练仓库负责人的约定。

1. **本文档是归档，不是立项。** 项目负责人的方向：Go2 不再重训练，后续全部面向 Lite3。
   本文不构成任何训练授权。
2. **链路决策（第 5 节）。** `axis` / `udp` / `ros2`，对应三份实质不同的规格。**必须先回答
   这一条**，因为它决定动作空间应该长什么样。
3. **(1,1) 对角原语的实测速度与方位角。** 工具是 `commissioning/axis_primitive_probe.py`。
   这台机器人到底能不能侧移，全押在这一对数值上。
4. **负载状态下的规划半径**——它就是 `meters_per_vmas_unit`，也是唯一能把 VMAS 视野换算成
   米的东西。
5. **一份已审、非"待审"的相机标定**（`calibration_status=validated`），需要在另一个独立
   测量的距离上再拍一次静态观测。
6. **训练仓库负责人对本需求的确认**——包括：加一个障碍物速度通道代价是否很低（第 4.3 节）、
   `training_max_steps` 能否高于 100、以及这个 checkpoint 是为三条链路中的哪一条训练的。

`training_max_steps` 值得单独一行：它是 **100**，在本栈 10 Hz 的控制频率下就是**十秒**的
回合（`policy/PROVENANCE.md`）。而演示实验是 40–60 秒——是策略在训练中见过的任何时长的
四到六倍。**把回合训得更长，其收益与射线扇、与视野都无关，是独立的一份。**

---

## 9. Deliverable shape — unchanged from issue #29
## 9. 交付物格式 —— 与 issue #29 一致，不变

**EN** — A `.npz` for `policy/models/`, loadable with `allow_pickle=False`:

| key | shape | notes |
| --- | --- | --- |
| `W1`, `b1` | `(H, OBS_DIM)`, `(H,)` | `OBS_DIM = 6 + n_rays` |
| `W2`, `b2` | `(H, H)`, `(H,)` | |
| `W3`, `b3` | `(4, H)`, `(4,)` | `tanh(loc)` on the first 2 is the deterministic action, as today |
| `metadata_json` | JSON string | **keep it** — it is the only in-band statement of what the network was trained with, and the adapter validates the config against it at load |

```json
{
  "actor_input_dim": 30,
  "observation_layout": ["x","y","vx","vy","x-gx","y-gy","lidar0", "...", "lidar23"],
  "training_lidar_range_vmas": 0.80,
  "training_agent_radius_vmas": 0.10,
  "training_max_steps": 100
}
```

**On this side, and not the training repository's problem:** `N_RAYS` and `OBS_DIM` in
`policy/physical_ai_mappo.py:40-41` are hardcoded to 12 and 18, and the actor hard-checks
the input width. They will be parameterised from the checkpoint's own metadata. The
mismatch is currently gated **loudly** — a new-fan checkpoint against the old adapter fails
at load with a clear message rather than feeding the network a wrong-width observation
(`:314-322`).

**中文** —— 交付一个可用 `allow_pickle=False` 加载的 `.npz`，放进 `policy/models/`，键名
与形状见上表。`metadata_json` **务必保留**——它是"这个网络到底用什么训出来的"唯一带内声明，
适配层在加载时会拿配置与它校验。

**我们这边负责、不属于训练仓库的部分：** `policy/physical_ai_mappo.py:40-41` 里的
`N_RAYS` 和 `OBS_DIM` 现在写死为 12 和 18，actor 还会硬校验输入宽度。这两处会改成从
checkpoint 自带的 metadata 读取。目前不匹配是**高声报错**的：新射线扇的 checkpoint 配旧
适配层，会在加载时明确失败，而不会把宽度错误的观测喂进网络（`:314-322`）。

---

## 10. Where this document disagrees with an earlier published statement
## 10. 本文与此前已发布说法不一致之处

**EN** — Three, all small, all stated so nobody has to rediscover them. The repository's
own regenerable output is treated as authoritative.

| | published | regenerated on `77d1d28` | which to use |
| --- | --- | --- | --- |
| both bins in range, three failing runs | **0 of 137** (`README.md:244`, `evidence/2026-08-18-threading-two-bins/README.md:205`) | 0 of 46 + 0 of 50 + 0 of 42 = **0 of 138** driven ticks (`radius_latch.py`) | the numerator **0** is identical; use **0 of 138** for the §6 gate and cite the script |
| run 15, 24-ray fan, converged radii | **42/42** (issue #29 comment) | **41/42** (`radius_latch.py`) | **41/42** |
| ticks in range at `training_lidar_range_vmas` 0.80 | **35/49, 38/41, 43/45** (issue #29 comment) | not re-derivable — those denominators are not the 46/50/42 the script prints for driven ticks | ⚠️ **attributed, not verified.** Quoted in §4.1 as attributed |

**EN** — None of the three changes a conclusion. They are recorded because this project has
already shipped one measurement that lied, and the habit of writing the disagreement down
is cheaper than the habit of rediscovering it.

**中文** —— 三处，都很小，都写出来免得别人再发现一遍。**以本仓库能重新生成的输出为准。**

三处分别是：三次失败实验中"两个桶同时在视野内"的周期数，已发布为 **0 of 137**，脚本今天
重新算出来是 **0 of 138**（分子都是 0，第 6 节的门限用 138 这个口径并注明脚本）；第 15 次
实验 24 射线、半径已收敛那一列，issue #29 评论里是 **42/42**，脚本给出的是 **41/42**，
以 41/42 为准；`training_lidar_range_vmas` 取 0.80 时的在视野周期数 **35/49、38/41、
43/45**，无法从 clone 重新推导（这三个分母与脚本给出的驱动周期数 46/50/42 不是同一口径），
⚠️ 在第 4.1 节按**引用**列出，未经核实。

三处都不改变任何结论。之所以记录，是因为这个项目已经出过一次"会说谎的度量"；**把不一致写
下来，比日后重新发现它便宜。**

---

## Sources / 出处

| claim | source |
| --- | --- |
| 18-value layout, no obstacle-velocity channel; 0.194 m closest approach | `integration/mappo_bridge.py:121-127`; `integration/test_mappo_bridge.py:191` |
| `N_RAYS`/`OBS_DIM`, proximity convention, run-local fan, horizon | `policy/physical_ai_mappo.py:40-41, 227, 483-484, 492-493, 314-322`; `policy/config.json` |
| provenance, `CORRECTION 6`, `training_max_steps` 100, scale calibration | `policy/PROVENANCE.md` |
| the steering cliff: 0.1° vs 103°, saturated 1.5–4.0 m/unit | `policy/README.md` |
| 0 of 137/138 ticks; per-fan window sampling; latched-vs-converged radii | `evidence/2026-08-19-what-the-policy-sees/radius_latch.py`; `README.md:244` |
| ray-count table, `training_lidar_range_vmas` 0.35 → 0.80 ask, deliverable shape | [issue #29](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/29) and its comments |
| r = +0.951 / +0.048; leftward command predates the peer | `evidence/2026-08-25-peer-runs/README.md:49-50` |
| Lite3 run-2 replay, 0.904 m, octant table, the (1,1) diagonal ask | [issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13) |
| sign-only mapping, octant snapping, `--derate` does not reach the wire | `robot-stack/deep_robotics/lite3/locomotion/lite3_axis_locomotion.py:60-61, 252`, `_linear_direction`; `robot-stack/deep_robotics/lite3/README.md` |
| unmeasured platform constants; provisional calibration; 0.401080 m proof | `robot-stack/deep_robotics/lite3/README.md`; `lite3-chair-bin-live-readiness-20260825.md` |
| two gait floors, forward 0.35 / lateral (0.15, 0.25] | [issue #42](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/42) |
| bilingual pairing style this document follows | `robot-stack/deep_robotics/lite3/commissioning/RUNBOOK.md` |
