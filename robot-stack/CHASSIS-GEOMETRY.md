<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Chassis and scene geometry, per robot / 各机型底盘与场景几何参数

The sibling of [`CAMERA-GEOMETRY.md`](CAMERA-GEOMETRY.md), and it exists for the same
reason: the planner divides by these numbers, and a radius nobody measured makes every
clearance downstream a guess wearing two decimal places. Same rule — **measured** or
**nominal**, said out loud, per field.

[`CAMERA-GEOMETRY.md`](CAMERA-GEOMETRY.md) 的姊妹文件，理由相同：规划器用这些数字做除法，
没人量过的半径会让下游每一个安全间隙都变成带两位小数的猜测。规则一致 —— 每个字段都标明
是**实测**还是**标称**。

## Unitree Go2

| field | value | provenance |
|---|---|---|
| `robot_radius_m` | 0.25 | **measured** — `deploy/README.md`, and the value `deploy/run-peer-supervised.sh` passes as `--robot-radius` |
| `PlannerConfig.robot_radius_m` default | 0.40 | **nominal** — `avoidance.py`'s "Go2 is ~0.70 x 0.31 m; half-diagonal, rounded up". A default, not this robot's number; the deploy script overrides it |
| peer `--obstacle-radius` | 0.20 | nominal — the radius a *second* Go2 is inflated to when it appears as an obstacle |

## Deep Robotics Lite3 — LITE3-A, measured 2026-08-26/27, Arm Shanghai

| field | value | provenance |
|---|---|---|
| `robot_radius_m`, loaded | **0.40** or **0.33** — ⛔ **UNRESOLVED, see below** | two documents disagree, and the number is the numerator of the policy scale |
| demo obstacle box, plan-view radius | **0.28 – 0.33** | **measured** 2026-08-26 against a **0.20 m nominal** — a 65% error on the value the planner was using |
| green marker `min_area_px` | **150** | **measured** — the marker's contour area is 165.5 – 185.5 px, so 150 passes **111/111** frames and the **400 nominal** passes **0/111** |
| forward primitive `measured_m_s` | — | ⛔ **NOT MEASURED.** `commissioning/axis_primitive_probe.py` has never run on a Lite3, and since #145 a live run **refuses** without it |
| yaw primitive `measured_rad_s` | — | ⛔ **NOT MEASURED**, and deliberately so — the probe declines to time yaw while `Segment.yaw_change_deg` can report a turn through pi backwards |

### ⛔ 0.33 vs 0.40: two documents disagree, and this needs a human answer

| document | says |
|---|---|
| [`deep_robotics/lite3/DEPLOYMENT-SOP.md`](deep_robotics/lite3/DEPLOYMENT-SOP.md) | robot **0.40 m** loaded, box **0.28 – 0.33 m**, `--policy-scale 4.0` |
| [`deep_robotics/lite3/LIVE-RUN-RUNBOOK.md`](deep_robotics/lite3/LIVE-RUN-RUNBOOK.md) (#152) | `--robot-radius` **0.33**, "the 0.20 nominal is 65% low", `--policy-scale 4.0` |

The SOP's 0.28–0.33 is the **box**; the runbook reads the same number as the **robot**.
They cannot both be right, and the pair in the runbook is not self-consistent either:
`MappoPlanner._check_radius_calibration` computes the implied scale as
`robot_radius_m / 0.10`, so **0.33 implies 3.30 and 0.40 implies 4.00**. Run the
runbook's own command verbatim and the drive path refuses to start —

```
planner robot radius   0.330 m   (--robot-radius)
implied scale          3.30 m/unit
```

— or, with `--policy-scale 4.0` forced, starts under the DELIBERATE SCALE OVERRIDE
banner, which says in as many words that the policy is being told the robot is a
different size from the one the planner plans with.

**Do not resolve this by picking the number that makes the command start.** The two
readings put the robot 0.07 m apart and the policy scale 21% apart, and they move the
veto's clearance and the detour's tangent corner with them. It wants the measurement,
from whoever took it. This is the `--robot-radius` double meaning that **issue #146** is
about — one flag that is simultaneously a chassis dimension and the numerator of a
calibration — and a 65% error on the clearance half makes splitting the two urgent
rather than tidy.

**⛔ 0.33 与 0.40：两份文档互相矛盾，需要人来裁决。** SOP 说机器人 0.40 m、箱子
0.28–0.33 m；#152 的 RUNBOOK 把同一个数字当成机器人半径。两者不可能都对，而且
RUNBOOK 自身也不自洽：implied scale = 半径 / 0.10，0.33 得 3.30，不是 4.0，照抄该命令
会被拒绝启动。**不要为了让命令跑起来而随便挑一个数字** —— 这正是 issue #146 所说的
`--robot-radius` 一个参数承担两种含义的问题。

### ⚠️ If 0.40 is the robot, it collides with the vendored default by coincidence

On the SOP's reading the two are added, not chosen between: it derives its passing width
as **0.40 m robot + 0.28–0.33 m box + 0.14 m clearance ≈ 0.9 m**.

`PlannerConfig.robot_radius_m`
is already 0.40, derived from the **Go2's** half-diagonal. The Lite3's measured loaded
radius independently lands on the same digits. Anything concluding "the default is already
right for the Lite3" would be right by accident, and would stop being right the moment
either number is corrected.

按 SOP 的读法，约 0.9 m 的通过宽度是机器人半径与箱子半径相加得来的。另外，0.40 与
`PlannerConfig` 里源自 **Go2** 半对角线的默认值数字相同，纯属巧合，不能据此认为默认值
对 Lite3 是对的。

### ⚠️ `min_area_px` is a detector gate, not a chassis property — and the code still says 400

It is a pixel-area threshold applied at detection scale in
`robot-stack/unitree/go2/visual_nav/colour_detector.py`, so it depends on range, sensor
resolution and the detector's own downscale — none of which are facts about the robot. It
is recorded here anyway because it was measured in the same set of runs, against the same
nominal-versus-measured failure, and because its measured value currently lives nowhere in
code: **`colour_detector.py`'s default is still `min_area_px = 400`**, the value that
passes 0 of 111 frames on this marker.

Changing that default is not this record's call to make — the same class serves the Go2's
markers at other ranges, and no Go2 evidence was re-scored against 150. What this row
does is stop the measurement being lost.

**`min_area_px` 属于检测器阈值，不是底盘参数**，但它与上面的半径出自同一批实测、同一类
"标称 vs 实测"错误，因此一并记录。代码里的默认值目前仍是 400（在本标记上 111 帧全不通过）。
是否修改该默认值不由本文件决定，因为同一个类也服务于 Go2 的标记。

### What is still needed

`commissioning/axis_primitive_probe.py`, on a Lite3, producing `measured_m_s` for every
direction the envelope enables. Until it runs, `--execution-supervisor turn-drive` cannot
be exercised live at all: the axis transport refuses a live run without the field, and the
planner would otherwise be rolling out a speed nobody has timed.

在 Lite3 上运行 `commissioning/axis_primitive_probe.py`，为包线启用的每个方向产出
`measured_m_s`。在此之前，`--execution-supervisor turn-drive` 无法进行实机验证。
