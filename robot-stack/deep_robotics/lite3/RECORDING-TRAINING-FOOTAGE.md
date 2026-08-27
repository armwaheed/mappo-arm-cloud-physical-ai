<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recording Lite3 training footage — what the 2026-08-27 session measured
# 录制 Lite3 训练素材 —— 2026-08-27 那次拍摄测出了什么

**EN** — Six clips were recorded in Shanghai on the morning of 2026-08-27 and turned into
a training set. They are a real improvement on what came before, and this document exists
because **the next session can be worth several times as much for the same hour of work**.
Everything below is a measurement from those six clips, not a preference.

**中文** —— 2026-08-27 上午在上海录制了六段素材，并已做成训练集。相比之前的素材，这次有实质性
进步。写这份文档，是因为**下一次拍摄，只要方法调整，同样一小时可以产出好几倍的价值**。
下面每一条都来自对这六段素材的实测，不是主观偏好。

> **EN** — Commands, flags and filenames are left in English throughout and must be typed
> exactly. Only the explanation is translated.
>
> **中文** —— 全文中的命令、参数和文件名一律保留英文原文，必须逐字输入。只有说明文字是中文。

---

## 1. What went right, and please keep doing it
## 1. 这次做对了的，请继续保持

**EN** — Two things in this session fixed problems that had blocked the previous clip
entirely.

**中文** —— 这次有两件事，直接解决了上一段素材完全卡住的问题。

| | **EN** | **中文** |
| --- | --- | --- |
| `--record-raw` | The video carries no HUD, no radar inset and no burned-in boxes. The previous clip had an overlay painted into 48.1% of its frames and half the clip had to be discarded. | 视频里没有 HUD、没有雷达小窗、没有烧录进画面的检测框。上一段素材有 48.1% 的帧被叠加层污染，一半素材只能丢弃。 |
| the `.jsonl` telemetry | Sent alongside every `.mp4`. It is what establishes which detector configuration was running. | 每个 `.mp4` 都附带发来了。正是靠它才能确认当时跑的是哪一套检测配置。 |

---

## 2. ⛔ The single change worth more than everything else: MOVE THE CAMERA
## 2. ⛔ 比其他所有改进都更重要的一条：让相机动起来

**EN** — All six clips are **tripod shots**. Measured by ORB+RANSAC homography against each
clip's own first frame, the median camera displacement across a whole clip is **0.0–1.0
pixels**, and the steadiest clip never moves more than 0.18 px in 90 seconds. "In different
distance and angle" in the folder names describes **the subject** moving. The camera did not.

**中文** —— 六段素材全部是**三脚架固定机位**。用 ORB+RANSAC 单应性变换与各段首帧比对，整段素材
的相机位移中位数是 **0.0–1.0 像素**，最稳的一段在 90 秒里位移从未超过 0.18 像素。文件夹名里的
"in different distance and angle"（不同距离和角度）指的是**被拍摄对象**在动，相机没有动。

**EN** — The cost of that is exact. Sampling every 5th frame and keeping a frame only when
it differs from the last kept frame by more than 3.0 mean grey levels on a 160x90
thumbnail, **5,854 frames contain 456 distinct views — 7.8%**. The rest are copies.

**中文** —— 代价可以精确算出来。每 5 帧取样一次，缩到 160x90 灰度图，只有与"上一张保留的帧"
平均灰度差超过 3.0 才算新视角 —— 结果是 **5,854 帧里只有 456 个不同视角，占 7.8%**。
其余都是重复。

| **EN** — what to do instead | **中文** —— 应该怎么做 |
| --- | --- |
| Carry the camera robot, or drive it slowly, while recording. A hand-carried camera covers the space far faster than a walking policy. | 录制时把带相机的机器人抱着走，或者让它慢速行进。手持移动覆盖空间的速度远快于让策略自己走。 |
| Stop and restart the recording from **5–10 different standing positions** in the room. Ten short clips from ten positions beat one long clip from one. | 在房间里换 **5–10 个不同站位**，每换一次就停一次、重录一段。十个位置各录一小段，远好过一个位置录一长段。 |
| Change the camera **height** too — the Lite3 is 0.37 m standing and 0.115 m prone, and those two are different pictures of the same room. | 相机**高度**也要变 —— Lite3 站立 0.37 m、趴下 0.115 m，这两个高度拍出来是同一房间的两张不同画面。 |

⚠️ **EN** — No amount of augmentation replaces this. Shear, crop, colour filtering and
occlusion multiply *examples*; they cannot manufacture a second viewpoint, a second room or
a second day. This is a limit on the data, not on the processing.

⚠️ **中文** —— 任何数据增强都替代不了这一条。倾斜、裁剪、色彩滤镜、遮挡只能增加**样本数量**，
造不出第二个机位、第二个房间、第二天。这是素材本身的上限，不是处理方法的问题。

---

## 3. ⛔ Record on a second day, in a second room
## 3. ⛔ 请换一天、换一个房间再录一次

**EN** — All six clips were recorded within **13 minutes of one morning in one room**. This
project has already measured what that is worth: one model scored **0 of 705** on
same-session frames against **60 of 159** across a day boundary. A score measured on the
same session as its training data does not predict tomorrow.

**中文** —— 六段素材全部录制于**同一个上午、同一个房间、前后 13 分钟之内**。本项目已经实测过
这意味着什么：同一个模型在**同场次**素材上是 **0/705**，跨天素材上是 **60/159**。
用与训练数据同场次的素材评分，无法预测明天的表现。

**EN** — Two sessions a week apart, in two different rooms, are worth more than ten sessions
in one morning. If only one extra session is possible, make it a **different day** rather
than a different room.

**中文** —— 相隔一周、在两个不同房间录的两场，价值高于同一个上午录的十场。如果只能补录一场，
优先选**不同的一天**，而不是不同的房间。

---

## 4. Lighting: say which one you are recording
## 4. 灯光：请明确说明你在录哪一种

**EN** — The three `dim-*` clips are not simply darker. Their mean frame luminance sweeps
**28 to 83** within a single clip, with frame-to-frame steps as large as **37.9 grey
levels**, while the steadiest `light-*` clip holds a standard deviation of **0.44**. Someone
was changing the room lights, or the camera's auto-exposure was hunting, during the take.

**中文** —— 三段 `dim-*` 素材不只是"更暗"。单段素材内部的平均亮度就从 **28 扫到 83**，
相邻帧之间最大跳变达 **37.9 个灰阶**；而最稳的 `light-*` 素材标准差只有 **0.44**。
说明拍摄过程中有人在调房间灯光，或者相机自动曝光在来回搜索。

**EN** — Unstable lighting is not useless — it is genuinely hard data. But it has to be
deliberate, because a background-subtraction labeller silently fails on it: on the dim
quadruped clip, twelve of twelve boxes it produced landed on the office chairs rather than
the robot. Either **hold the lighting still** for a take, or **sweep it on purpose** and say
so in the handover note.

**中文** —— 亮度不稳的素材并非没有价值 —— 它确实是困难样本。但必须是**有意为之**，因为背景相减
类的自动标注在这种素材上会无声失效：在那段暗光四足素材上，自动标注产出的框十二个里有十二个
落在办公椅上，而不是机器人身上。要么**整段保持灯光不变**，要么**有意做一次亮度扫变**并在交接
说明里写清楚。

---

## 5. Two things to fix in the recording tool
## 5. 录制工具需要修的两个问题

**EN** — Neither is the operator's fault, and both cost real work downstream.

**中文** —— 这两条都不是操作人员的问题，但都在后续处理中造成了实际工作量。

| # | **EN** | **中文** |
| --- | --- | --- |
| 1 | **`perception.video_frame` is `null` in all 3,896 ticks of all six files.** It is documented as the join between telemetry and video, and it does not exist. The join had to be recovered as `round((t - frame_age_s) * 15)` and validated by hand on 17 frames. | **六个文件全部 3,896 个 tick 里，`perception.video_frame` 都是 `null`。** 文档说它是遥测与视频的对应关系，实际上没有写入。只能用 `round((t - frame_age_s) * 15)` 反推，并人工核对 17 帧来验证。 |
| 2 | **The detector ran with `classes: ['person']`.** So across 2,701 frames of quadruped footage the telemetry carries **3 boxes**. The network was not blind — the class filter discarded what it saw. | **检测器是以 `classes: ['person']` 运行的。** 因此 2,701 帧四足机器人素材里，遥测只留下 **3 个框**。不是网络看不见，是类别过滤把它看到的丢掉了。 |

---

## 6. ⛔ Still not done: calibrate the camera
## 6. ⛔ 仍未完成：相机标定

**EN** — `detector/COLLECTING-LITE3-DATA.md` Step 0 asks for `lite3_front_camera.json` and
the camera's optical-centre height above the floor, **before** any recording. It has still
not been produced, and the camera block embedded in these six recordings is wrong three
ways — see [`robot-stack/CAMERA-GEOMETRY.md`](../../CAMERA-GEOMETRY.md).

**中文** —— `detector/COLLECTING-LITE3-DATA.md` 的 Step 0 要求在**任何录制之前**先产出
`lite3_front_camera.json` 以及相机光心离地高度。这一步至今没有做，而这六段素材里内嵌的相机
参数有三处错误 —— 见 [`robot-stack/CAMERA-GEOMETRY.md`](../../CAMERA-GEOMETRY.md)。

**EN** — The detector work does not need it: a detector outputs boxes in pixels and transfers
across cameras, and nothing in this training set reads a focal length. **Ranging does need
it**, and ranging is what turns a box into an obstacle the planner can avoid.

**中文** —— 检测器本身不需要它：检测器输出的是像素坐标的框，可以跨相机迁移，本训练集也没有读取
任何焦距。**但测距需要它**，而正是测距才能把一个框变成规划器可以绕开的障碍物。

---

## 7. What to send back
## 7. 需要回传的内容

**EN** — Same as before, plus one line that was missing this time.

**中文** —— 与之前相同，另外补充这次缺少的一行。

1. **EN** — The `.mp4` / `.jsonl` pairs, recorded with `--record-raw`.
   **中文** —— 用 `--record-raw` 录制的 `.mp4` / `.jsonl` 成对文件。
2. **EN** — `lite3_front_camera.json` and the measured camera height in metres.
   **中文** —— `lite3_front_camera.json` 以及实测的相机离地高度（米）。
3. **EN** — For each clip: the building, the room, **the date**, and whether the lighting was
   held still or swept on purpose.
   **中文** —— 每段素材注明：楼、房间、**日期**，以及灯光是保持不变还是有意扫变。
4. **EN** — For each clip, roughly how many times the camera was moved to a new standing
   position. If the answer is "none", the clip is one viewpoint however long it is.
   **中文** —— 每段素材注明相机大致换了几个站位。如果答案是"没换"，那么无论录多长，这段素材
   都只有一个视角。
