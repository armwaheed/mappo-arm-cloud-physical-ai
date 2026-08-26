<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lite3 Venture commissioning runbook — Shanghai
# Lite3 Venture 现场标定操作手册 —— 上海

**EN** — This is the operator's document for
[issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13). It assumes
you did not write the code and cannot ask the person who did, because they are nine time
zones away and asleep. Everything you need to run, judge and report a commissioning session
is here. Work through it top to bottom, once per robot.

**中文** —— 这是
[issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)
的现场操作文档。本文假设你没有参与写这些代码，而且无法当场找到作者——他们在九个时区以外，
现在是深夜。运行、判断和上报一次标定所需要的全部内容都在这里。请从上到下依次执行，
**每台机器人各做一遍**。

> **EN** — Commands, flags, filenames and the text the programs print are left in English
> throughout, and must be typed and matched **exactly**. Only the explanation is translated.
>
> **中文** —— 全文中的命令、参数、文件名以及程序打印出来的文字一律保留英文原文，
> 必须**逐字**输入和比对。只有说明文字是中文。

---

## 0. Terminology — please confirm these renderings
## 0. 术语表 —— 这些译法请确认

**EN** — These are the terms this runbook needs and the Chinese renderings used below. They
are written plainly rather than in polished domain jargon. If the team has a house term for
any of them, correct it here and say so in issue #13; do not let a wrong term propagate into
the measurement report.

**中文** —— 下表是本手册用到的术语和所采用的中文译法。为避免误译，这里用的是**直白说法**，
不是行业习惯用语。如果团队已有习惯译法，请在这里改正，并在 issue #13 里说明；
不要让错误的术语传到测量报告里。

| English (use verbatim in commands) | 中文（本手册用法） | 需确认 |
| --- | --- | --- |
| gait floor / `--gait-floor` | 步态下限：能让机器人真正走起来的**最低**速度指令 | ✅ 请确认 |
| actuator gain / `--actuator-gain` | 执行器增益：实际速度 ÷ 指令速度 | ✅ 请确认 |
| loaded planning radius / `--robot-radius` | 负载状态下的规划半径（俯视外形的外接圆半径） | ✅ 请确认 |
| policy scale / `--policy-scale` | 策略缩放系数 = 半径 ÷ 0.10 m | ✅ 请确认 |
| anchor segment | 锚段：用**已知能走**的速度跑的一段，用来证明腿真的在动 | ✅ 请确认 |
| control segment | 对照段：指令为零（或被测轴为零）的一段，用来量测漂移 | ✅ 请确认 |
| lane | 试验通道：机器人前方清空出来的一条直线区域 | — |
| provisional / reviewed | 待审 / 已审：标定结果的状态 | — |
| refuse / refusal | 拒绝执行：程序主动停下并说明原因，不是报错崩溃 | — |

---

## 1. Safety — read this before anything is powered
## 1. 安全须知 —— 上电之前必须先读

**EN** — `robot-stack/SAFETY.md` governs anything that moves a leg and is not optional.
Two of the six tools in this directory walk the robot: `gait_floor_probe.py` and
`actuator_gain_probe.py`. Both refuse to move without **both** `--live` and
`--operator-ready`. Neither flag is a formality.

**中文** —— `robot-stack/SAFETY.md` 管辖一切会让腿动起来的操作，不可跳过。
本目录下六个工具中有两个会让机器人行走：`gait_floor_probe.py` 和 `actuator_gain_probe.py`。
两者都必须**同时**给出 `--live` 和 `--operator-ready` 才会动。这两个参数都不是走形式。

| | **EN** | **中文** |
| --- | --- | --- |
| Who holds the stop | One named person holds the vendor remote with the emergency stop, for the whole run, and does nothing else. | 指定一个人全程手持厂商遥控器和急停，全程只做这一件事。 |
| The other robot | Park the second Venture **outside the lane and powered off**. It is not a peer sensor, it is an obstacle this robot cannot see. | 第二台 Venture 停在**通道之外并关机**。它不是传感器，只是一个本机看不见的障碍物。 |
| Sideways | The robot has **no lateral sensing at all**. Clear both sides of the lane, not just the ends. | 机器人**完全没有侧向感知**。通道两侧也必须清空，不能只清两端。 |
| Motor heat | Motor temperatures are not readable on this interface (task 2 proves it). Let the robot cool between runs; no software can see accumulated heat. | 本接口读不到电机温度（任务 2 会给出证据）。两次运行之间让机器人冷却；累积热量没有任何软件能看到。 |
| Stopping | Stop the run yourself the moment the gait changes, a motor smells hot, or the robot goes anywhere you did not expect. | 一旦步态变化、闻到电机发热气味、或机器人走到预期之外的位置，立刻自行停止。 |

---

## 2. Set up the laptop — do this once per session
## 2. 笔记本电脑的准备 —— 每次开工做一次

**EN** — The motion host at `192.168.1.120` streams its state to **exactly one** address,
configured in `~/jy_exe/conf/network.toml` on the robot. If your laptop does not hold that
address, the robot is silent and every tool below looks broken for the wrong reason. Read
the setting without changing anything:

**中文** —— 运动主机 `192.168.1.120` 只向**一个**地址发送状态数据，该地址配置在机器人上的
`~/jy_exe/conf/network.toml` 里。如果笔记本没有占用这个地址，机器人就是"沉默"的，
下面所有工具都会因为一个完全无关的原因看起来像是坏了。先只读、不修改地查看这个配置：

```bash
ssh <user>@192.168.1.120
cat ~/jy_exe/conf/network.toml
```

**EN** — Then set the laptop's Ethernet interface to that static address, netmask
`255.255.255.0`, and **leave the Router / gateway field completely empty**. A gateway here
installs a default route through the robot and black-holes the laptop's normal internet.

**中文** —— 然后把笔记本的以太网口设为该静态地址，子网掩码 `255.255.255.0`，
并且**"路由器 / 网关"一栏必须完全留空**。这一栏填了值会通过机器人安装一条默认路由，
导致笔记本无法上网。

```bash
ping 192.168.1.120      # must reply before you go on / 必须能 ping 通才继续
cd robot-stack/deep_robotics/lite3/commissioning
```

**EN** — Everything in this directory needs Python 3.8 or newer and nothing else. No ROS, no
`pip install`, no internet. The one exception is task 3, the camera, which needs
`opencv-python` because it drives the shared camera fitter.

**中文** —— 本目录下的工具只需要 Python 3.8 或更新版本，别无依赖：不需要 ROS，不需要
`pip install`，不需要联网。唯一的例外是任务 3（相机），它需要 `opencv-python`，
因为它调用的是共用的相机标定程序。

---

## 3. The order, and why it is this order
## 3. 执行顺序，以及为什么是这个顺序

**EN** — Read-only first, tape measure second, camera third, and only then the two that
walk. The gait floor comes before the actuator gain because the gain is only claimed above
the floor. `commission.py` runs them in exactly this order and stops at the first refusal
rather than writing a record with a hole in it.

**中文** —— 先做只读的，再做卷尺测量，然后是相机，最后才是两个会让机器人走路的。
步态下限必须在执行器增益之前测，因为增益只在下限以上才成立。`commission.py` 就是按这个顺序跑的，
并且**一旦有任何一步被拒绝就立即停止**，而不是写出一份有缺口的记录。

### ⚠️ First decide which transport, because it changes which tasks exist
### ⚠️ 先确定用哪条通道，因为它决定了哪些任务成立

**EN** — Tasks 4 and 5 measure a *commanded* speed. Task 5b measures a *primitive*. They
are not alternatives you may pick between — the transport decides, and each tool refuses
the other transport by name.

- **`--locomotion-transport axis`** is what both Ventures have actually walked on. Its
  mapping is **sign-only**: every command above the profile's deadband sends the same
  full-scale value, so a commanded speed is a *direction*. There is no gait floor and no
  actuator gain here. **Do task 5b, and skip 4 and 5.**
- **`--locomotion-transport udp`** carries the commanded number to the wire, so 4 and 5
  are defined — but **no Venture has ever moved on it.** On 2026-08-24 a `vx=0.10` m/s
  pulse arrived correctly at the motion host and the robot did not move. If you run a
  walking probe on it, expect `0.000 m/s` on every segment; that is the interface, not
  the robot, and not a floor.

**中文** —— 任务 4 和 5 测的是**下发速度**；任务 5b 测的是**动作原语**。
这不是可以随便二选一的：由通道决定，而且每个工具都会**指名拒绝**另一条通道。

- **`--locomotion-transport axis`**：两台 Venture 实际走起来用的就是它。它的映射是
  **只看符号**的——只要超过配置文件的死区，发出去的就是同一个满量程值，所以"下发速度"
  在这里其实是"方向"。这里不存在步态下限，也不存在执行器增益。**请做任务 5b，跳过 4 和 5。**
- **`--locomotion-transport udp`**：下发的数值会真正到达链路，所以任务 4、5 在数学上成立——
  **但没有任何一台 Venture 在它上面动过。** 2026-08-24 曾有一次 `vx=0.10` m/s 的指令被
  确认正确到达运动主机，机器人没有动。如果在它上面跑行走探针，每一段都会是 `0.000 m/s`；
  那是接口的问题，不是机器人的问题，更不是"步态下限"。

| # | task / 任务 | moves? / 会动吗 | issue #13 |
| --- | --- | --- | --- |
| 0 | `lite3_state_probe.py` — link, battery, mode, remote-driven speeds | no / 否 | mode transition, angular-velocity unit |
| 1 | `motor_temperature_probe.py` — 12 channels or proof of absence | no / 否 | health bridge (vendor blocker) |
| 2 | `loaded_radius_probe.py` — tape measure | no / 否 | loaded radius, `--policy-scale` |
| 3 | `camera_calibration.py` — focal length, HFOV, lens height | no (with `--marker`) / 否 | camera calibration |
| 4 | `gait_floor_probe.py` — **WALKS**, `udp` transport only | **YES / 是** | `--gait-floor` |
| 5 | `actuator_gain_probe.py` — **WALKS**, `udp` transport only | **YES / 是** | `--actuator-gain` |
| 5b | `axis_primitive_probe.py` — **WALKS**, `axis` transport only | **YES / 是** | `measured_m_s` in the axis profile |
| 6 | `commission.py --review` then `--emit-flags` | no / 否 | signs the record |

---

## Task 0 — the passive capture
## 任务 0 —— 被动采集

**EN** — Run this twice. It transmits nothing and cannot move the robot; the module has no
send path at all and the test suite asserts that structurally.

**中文** —— 这一步要做**两次**。它不发送任何数据，也不可能让机器人动；
该模块根本没有发送通道，测试套件用代码结构断言了这一点。

```bash
python3 lite3_state_probe.py --seconds 30 --robot-id LITE3-A --record lite3-a-1.jsonl
```

**EN** — **Capture 1**: the robot is prone and untouched. This confirms the link is alive
and gives you `battery_level` and the resting mode fields.
**Capture 2**: keep the probe running. The operator stands the robot on the **vendor
remote**, puts it into high-level navigation mode, walks it slowly forward, then turns it in
place. Our software sends nothing during this; the vendor's own controller is the only thing
commanding the legs.

**中文** —— **第 1 次采集**：机器人趴着、不要触碰。用于确认链路正常，并读到 `battery_level`
和静止状态下的模式字段。
**第 2 次采集**：保持程序运行。操作员用**厂商遥控器**让机器人站起、切到高层导航模式、
缓慢向前行走，然后原地转向。这期间我们的软件不发送任何数据，指挥腿的只有厂商自己的控制器。

**EN — what a good capture looks like.** Hundreds of frames across several `kind`s, a
`battery_level` above 40%, at least two mode transitions timestamped in capture 2, and a
"Remote-commanded vs measured forward speed" table with several rows in it. **Write down the
lowest and highest commanded speeds in that table** — you need them as `--ladder-top` and
`--lateral-top` in task 4.

**中文 —— 什么算是一次好的采集。** 有几百帧、覆盖多个 `kind`；`battery_level` 高于 40%；
第 2 次采集里至少出现两次带时间戳的模式切换；并且
"Remote-commanded vs measured forward speed" 表里有若干行数据。
**请记下该表中最低和最高的指令速度** —— 任务 4 需要它们作为 `--ladder-top` 和 `--lateral-top`。

**EN — what a bad capture looks like.** `NO FRAMES RECEIVED` means the address is wrong;
go back to section 2. "no forward command seen" means the operator did not actually drive
the robot during capture 2, or the remote was released between frames — repeat it while
holding the stick forward.

**中文 —— 什么算是一次失败的采集。** `NO FRAMES RECEIVED` 表示地址不对，回到第 2 节重做。
"no forward command seen" 表示第 2 次采集期间操作员并没有真的驱动机器人，
或者摇杆在两帧之间被松开了 —— 请保持摇杆前推重做一次。

---

## Task 1 — motor temperatures
## 任务 1 —— 电机温度

```bash
python3 motor_temperature_probe.py --robot-id LITE3-A --firmware V1.0.8 \
    --payload 'stock, no payload' --seconds 20
```

**EN** — Set up: robot powered on, prone, untouched. Nothing moves and nothing is
transmitted.

**中文** —— 现场准备：机器人上电、趴下、不要触碰。全程没有动作，也不发送任何数据。

**EN — a good result, either way.** Twelve numbers under `motor 0 … motor 11` closes the
issue #13 health-bridge item outright. `NO MOTOR TEMPERATURE CHANNEL on the high-level
interface` is **also a good result**: it is the evidence the vendor question needs attached
to it, and it is only valid because the tool watched the stream flowing while nothing
arrived. Paste whichever one you get.

**中文 —— 两种结果都算成功。** 如果打印出 `motor 0 … motor 11` 十二个数值，
issue #13 的健康接口一项就直接完成了。如果打印
`NO MOTOR TEMPERATURE CHANNEL on the high-level interface`，**这同样是一个有效结果**：
这正是我们要附在厂商问题后面的证据，而且它成立的前提正是工具确实看到了数据流在跑、
只是里面没有温度字段。无论得到哪一种，都请原样贴到 issue 里。

**EN — a bad result.** A refusal saying *"no state frames arrived at all"*. That is not a
finding; it means the laptop is not receiving. A disconnected laptop must never be allowed
to "prove" the vendor question for us. Go back to section 2.

**中文 —— 失败的结果。** 出现 *"no state frames arrived at all"* 的拒绝提示。
这不是结论，只说明笔记本根本没在收数据。绝不能让一台没连上的笔记本替我们"证明"厂商问题。
回到第 2 节重做。

---

## Task 2 — the loaded planning radius
## 任务 2 —— 负载状态下的规划半径

**EN — set up.** Stand the robot in its normal demo posture **with the event payload
fitted**. Drop a plumb line from the point the robot turns about — not the centre of the
chassis casting if those differ — and mark that point on the floor. Measure from the mark to
the furthest part of the robot in each of four directions, at floor level, **including the
legs at their widest and anything bolted on**.

**中文 —— 现场准备。** 让机器人以演示时的正常姿态站立，**并装上活动当天要带的负载**。
从机器人的**旋转中心**（若与机身铸件几何中心不一致，以旋转中心为准）垂下铅垂线，
在地面上标出该点。以该标记点为原点，向四个方向量到机器人最外缘，
**贴地测量，必须包含腿张开到最大时的位置和一切外挂件**。

```bash
python3 loaded_radius_probe.py --robot-id LITE3-A --firmware V1.0.8 \
    --payload 'stock + 0.6 kg camera mast' \
    --front 0.42 --back 0.38 --left 0.24 --right 0.24 --stance-confirmed
```

**EN — what the number means.** The planner treats the robot as a disc, so the radius
reported is the **corner** distance `sqrt(max(front,back)² + max(left,right)²)`, not the
largest single extent. The corner is the part that clips the door frame. `--policy-scale` is
that radius divided by the 0.10 m agent the policy was trained with.

**中文 —— 这个数字的含义。** 规划器把机器人当作一个圆盘，所以输出的半径是**对角**距离
`sqrt(max(front,back)² + max(left,right)²)`，而不是四个方向里最大的那一个。
撞到门框的正是这个对角。`--policy-scale` 就是该半径除以策略训练时使用的 0.10 m 智能体半径。

**EN — good / bad.** Good: front and back are within roughly 2× of each other, and so are
left and right; the printed radius is a few centimetres larger than the largest extent. Bad:
the tool refuses with *"the front/back extents differ by 3.0×"* — that almost always means
the tape was referenced to an edge of the body instead of the turning point, which halves
one side and doubles the other. Re-drop the plumb line.

**中文 —— 好结果 / 坏结果。** 好：front 与 back 之比、left 与 right 之比大致都在 2 倍以内；
打印出来的半径比最大单边测量值大几厘米。坏：工具拒绝并提示
*"the front/back extents differ by 3.0×"* —— 这几乎总是因为卷尺的原点取在了机身边缘而不是旋转中心，
结果一边少了一半、另一边多了一倍。请重新垂铅垂线。

---

## Task 3 — the camera
## 任务 3 —— 相机标定

**EN — set up.** Print an ArUco marker, mount it square-on to the camera, and tape-measure
the camera-to-marker distance. Measure it at the **longest distance where the marker is
still comfortably detected**: placement error is roughly constant in centimetres, so its
relative cost falls as the distance grows. Separately, measure the **lens height** — floor to
the centre of the lens, with the robot **standing**.

**中文 —— 现场准备。** 打印一张 ArUco 标记板，正对相机安装，用卷尺量出相机到标记板的距离。
请在**标记板仍能被稳定识别的最远距离**上测量：摆放误差以厘米计基本是固定的，
距离越远，它占比越小。另外单独测量**镜头高度** —— 机器人**站立**状态下，从地面到镜头中心。

```bash
python3 camera_calibration.py --robot-id LITE3-A --firmware V1.0.8 --payload none \
    --camera-source 0 --marker 1.50 --marker-size 0.15 \
    --lens-height 0.31 --lens-height-source 'tape, standing, floor to lens centre' --run
```

**EN — why `--marker` and not `--object` or `--spin`.** `--marker` uses ArUco corner
refinement and no neural detector at all, so it cannot disagree with what the production
navigator runs the detector at — and it does not move the robot. `--object` and
`--spin --spin-target object` both go through the detector, and the tool asserts their
configuration matches production's before it will fit anything. `--spin` also turns the
robot and needs a `--spin-rate` measured above **this** robot's yaw deadband; the Go2's
0.8 rad/s is not a Lite3 measurement.

**中文 —— 为什么用 `--marker` 而不是 `--object` 或 `--spin`。** `--marker` 用的是 ArUco
角点精化，完全不经过神经网络检测器，因此不可能与实际运行时的检测器配置不一致 ——
而且它不会让机器人动。`--object` 和 `--spin --spin-target object` 都要过检测器，
工具会在拟合之前先断言两边配置一致。`--spin` 还会让机器人原地旋转，
并且需要一个**在本机偏航死区之上实测得到的** `--spin-rate`；Go2 的 0.8 rad/s 不是 Lite3 的实测值。

**EN — one thing to know about the output.** The shared fitter writes `height_m: 0.32` into
every calibration it produces, because 0.32 m is the height of the **Go2's** camera when the
Go2 stands, and nothing on the fitting path has ever asked for a Lite3 value. This tool
overwrites it with your measurement and prints what it replaced. If the line says
`replaced the fitter's 0.32`, that is the bug being fixed in front of you, not an error.

**中文 —— 关于输出需要知道的一件事。** 共用的标定程序会在每一份标定文件里写入
`height_m: 0.32`，因为 0.32 m 是 **Go2** 站立时相机的高度，而标定流程从来没有向使用者
索取过 Lite3 的对应数值。本工具会用你实测的数值覆盖它，并打印出被替换掉的旧值。
如果看到 `replaced the fitter's 0.32`，那是这个缺陷正在被当场修正，**不是报错**。

**EN — good / bad.** Good: a focal length in pixels, an HFOV in degrees, and a lens height
equal to what you measured. Bad: *"only N sightings"* means the marker id, the lighting or
the framing is wrong. *"is tagged platform='unitree-go2'"* means you pointed it at a Go2
calibration file; do not stamp it as a Lite3 one.

**中文 —— 好结果 / 坏结果。** 好：输出一个以像素为单位的焦距、一个以度为单位的 HFOV，
以及和你实测一致的镜头高度。坏：*"only N sightings"* 说明标记板编号、光照或取景有问题。
*"is tagged platform='unitree-go2'"* 说明指到了一份 Go2 的标定文件上，不要把它标成 Lite3 的。

---

## Task 4 — the gait floor (THE ROBOT WALKS)
## 任务 4 —— 步态下限（机器人会行走）

**EN — set up.** Clear a lane at least **3 m long and 2 m wide**, clear on both sides.
Measure it; do not estimate it, because the tool refuses a run that would not fit and the
refusal is only as good as the number you gave it. Park the second Venture outside the lane
and powered off. One person on the vendor remote with the emergency stop, doing nothing
else. Stand the robot on the remote, put it into high-level navigation mode, and hand it
over.

**中文 —— 现场准备。** 清出一条至少**长 3 m、宽 2 m** 的通道，两侧同样清空。
请**实测**通道尺寸，不要估计——工具会拒绝跑不下的方案，而这个拒绝的可靠性完全取决于你给的数字。
第二台 Venture 停在通道外并关机。一人手持厂商遥控器和急停，全程只做这件事。
用遥控器让机器人站起、切到高层导航模式，然后移交给电脑。

```bash
# First, without --live: it prints the plan and the lane it needs, and exits.
# 先不加 --live：它会打印执行计划和所需通道长度，然后退出。
python3 gait_floor_probe.py --robot-id LITE3-A --firmware V1.0.8 --payload none \
    --ladder-top 0.50 --lateral-top 0.30 --lane-metres 6.0 --lane-width-metres 2.0

# Then, when the plan fits the room and the operator is ready:
# 计划与场地匹配、操作员就位之后，再加：
python3 gait_floor_probe.py --robot-id LITE3-A --firmware V1.0.8 --payload none \
    --ladder-top 0.50 --lateral-top 0.30 --lane-metres 6.0 --lane-width-metres 2.0 \
    --live --operator-ready
```

**EN** — `--ladder-top` and `--lateral-top` are **not guesses**. They come from the task 0
capture: a speed you have already watched this robot walk at on the vendor remote. There is
no default and there will not be one.

**中文** —— `--ladder-top` 和 `--lateral-top` **不是估计值**，而是来自任务 0 的采集结果：
你已经亲眼看到这台机器人用厂商遥控器以该速度走过。这里没有默认值，将来也不会有。

**EN — what the run does.** It walks a descending ladder of speeds, one second each, with a
zero-command **control** between every rung and **anchor** segments at the top of the ladder.
Three phases: `forward`, `strafe` (sideways from a standstill) and `diagonal` (sideways
while already walking forward). It pauses between phases so you can walk the robot back to
the start of the lane.

**中文 —— 这次运行做了什么。** 它按由高到低的速度阶梯逐级行走，每级 1 秒，
每级之前插入一个零指令**对照段**，并在阶梯顶端速度上安排若干**锚段**。
共三个阶段：`forward`（前进）、`strafe`（原地起步的纯侧移）和
`diagonal`（已经在前进的同时侧移）。阶段之间会暂停，方便你把机器人带回通道起点。

**EN — a good result.** The forward table shows `yes` for the high rungs and `NO` for the
low ones, with the switch happening once. The tool then names a `conservative --gait-floor`
one ladder step above the lowest command that walked. The two lateral tables each end in a
verdict line reading `LINE` or `STEP`.

**中文 —— 好结果。** forward 表里高速档显示 `yes`、低速档显示 `NO`，并且只切换一次。
随后工具会给出一个 `conservative --gait-floor`，即比"最低仍能走"的档位再高一级。
两张侧移表各自以一行结论收尾，写着 `LINE` 或 `STEP`。

**EN — the single most important thing in this document.** The `strafe` number and the
`diagonal` number are **two different measurements and neither describes the other**. On the
Go2, sideways motion from a standstill had a hard floor, while sideways motion during a
forward walk was delivered proportionally from a quarter of that floor upward. Report both.
Never substitute one for the other, and never let anyone downstream do it either.

**中文 —— 本文档中最重要的一点。** `strafe` 的数值和 `diagonal` 的数值是
**两个不同的测量结果，任何一个都不能代表另一个**。在 Go2 上，
从静止起步的侧移存在一个明确的下限，而在前进过程中的侧移则从该下限的四分之一处起就已按比例输出。
**两个都要上报。** 绝不可以互相替代，也不要让下游的人这样做。

**EN — a bad result, and it is designed to look like a good one.** A table of `0.000` on
every axis at every setting is what a robot that **never stood up** produces, and it reads
exactly like "the floor is real and total". The tool refuses that run outright rather than
tabulating it — see the refusal catalogue below.

**中文 —— 坏结果，而且它天生就长得像好结果。** 所有轴、所有档位全是 `0.000` 的表格，
正是一台**根本没有站起来**的机器人会产生的输出，而它看起来恰恰就像"下限是真实存在且覆盖全部档位的"。
工具会直接拒绝这种运行，而不是把它做成表格 —— 见下面的拒绝提示对照表。

---

## Task 5 — the actuator gain (THE ROBOT WALKS)
## 任务 5 —— 执行器增益（机器人会行走）

**EN — set up.** Same lane, same operator, same stop. `--gait-floor` is the number task 4
just produced **for this robot**. `--envelope-vx` is the top forward speed the demo will
actually command.

**中文 —— 现场准备。** 同一条通道、同一名操作员、同一个急停。
`--gait-floor` 就是任务 4 刚刚为**这台机器人**测出的数值。
`--envelope-vx` 是演示时实际会下发的最高前进速度。

```bash
python3 actuator_gain_probe.py --robot-id LITE3-A --firmware V1.0.8 --payload none \
    --gait-floor 0.30 --envelope-vx 0.40 --lane-metres 6.0 --live --operator-ready
```

**EN — what the number means.** The gain multiplies every distance and duration budget
downstream: at gain 0.6 a 2 m waypoint takes 1.7× as long as the command implies. It is
fitted through the origin from the robot's **pose**, never from the robot's own velocity
estimate — on another platform a real 0.45 gain was read as 0.17 m/s of estimator noise, and
a parked robot then drifted toward its goal because the controller believed the estimate.

**中文 —— 这个数字的含义。** 增益会乘进下游所有的距离和时间预算：增益为 0.6 时，
一个 2 m 的航点实际耗时是指令值推算的 1.7 倍。它是用机器人的**位姿**过原点拟合出来的，
**不是**用机器人自己报告的速度估计 —— 在另一个平台上，真实的 0.45 增益曾被当成
0.17 m/s 的估计噪声，结果一台停着不动的机器人因为控制器相信了那个估计而慢慢漂向目标。

**EN — good / bad.** Good: a residual well under 0.01 m/s RMS, per-segment ratios clustered
inside about ±0.05 of the fitted gain, and a closing line saying the two instruments agree.
Bad, and worth reporting rather than retrying: a line beginning `⚠️ the platform's own
velocity estimate fits a gain of …`. That means this robot's velocity estimate cannot be
trusted for control. Ship the pose number and put the warning in issue #13.

**中文 —— 好结果 / 坏结果。** 好：残差远低于 0.01 m/s RMS；各段比值集中在拟合增益的
±0.05 左右；末尾一行说明两个测量手段结果一致。坏——但这一种要上报而不是重试：
出现以 `⚠️ the platform's own velocity estimate fits a gain of …` 开头的一行。
这说明这台机器人的速度估计不能用于控制。请以位姿测得的数值为准，并把这条警告写进 issue #13。

---

## Task 5b — the axis primitive speeds (THE ROBOT WALKS)
## 任务 5b —— 轴动作原语的实际速度（机器人会行走）

**EN — when to do this.** Instead of tasks 4 and 5, whenever the demo will run on
`--locomotion-transport axis` — which today is the only transport either Venture has
walked on.

**中文 —— 什么时候做。** 当演示将使用 `--locomotion-transport axis` 时，用本任务**代替**
任务 4 和 5。目前两台 Venture 唯一真正走起来过的就是这条通道。

**EN — set up.** Clear a lane long enough **in front of and behind** the robot, and wide
enough on **both sides**, for every primitive your profile carries. Measure it. The tool
checks the lane against `--assume-up-to`, your honest upper bound on how fast this robot
might turn out to walk — nobody knows the real number yet, which is the entire point of
the task. One person on the vendor remote with the emergency stop, doing nothing else.

**中文 —— 现场准备。** 按配置文件里实际存在的每个动作原语，清出**前后**足够长、**两侧**
足够宽的通道，并**实测**尺寸。工具会用 `--assume-up-to`（你对这台机器人可能达到的最快速度
给出的诚实上界）来校验通道是否够用——真实数值目前没有人知道，这正是本任务的目的。
一人手持厂商遥控器和急停，全程只做这件事。

```bash
# First, without --live: it prints the plan and the lane each primitive needs, and exits.
# 先不加 --live：它会打印执行计划以及每个原语所需的通道尺寸，然后退出。
python3 axis_primitive_probe.py --robot-id LITE3-A --firmware V1.0.8 --payload none \
    --locomotion-transport axis --axis-profile lite3-axis-LITE3-A.json \
    --assume-up-to 0.80 --lane-metres 8.0 --lane-width-metres 3.0

# Then, when the plan fits the room and the operator is ready:
# 计划与场地匹配、操作员就位之后，再加：
python3 axis_primitive_probe.py --robot-id LITE3-A --firmware V1.0.8 --payload none \
    --locomotion-transport axis --axis-profile lite3-axis-LITE3-A.json \
    --assume-up-to 0.80 --lane-metres 8.0 --lane-width-metres 3.0 \
    --live --operator-ready
```

**EN — what it produces.** One speed per primitive, and a `measured_m_s` block to paste
into that same profile. The number is the **maximum** across the repeats, not the mean:
it exists to be compared against a safety ceiling, and a mean hides the fast sample the
ceiling has to survive. Until it is there, the live run prints `AXIS SPEEDS ARE NOT
VERIFIED AGAINST THE ENVELOPE` and walks anyway.

**中文 —— 产出什么。** 每个原语一个速度，外加一段可直接粘回同一份配置文件的
`measured_m_s`。取的是多次重复中的**最大值**，不是平均值——这个数字是用来和安全上限做比较的，
而平均值会把"上限必须扛住的那一次最快"藏起来。在它填上之前，实机运行只会打印
`AXIS SPEEDS ARE NOT VERIFIED AGAINST THE ENVELOPE`，然后照走不误。

**EN — the refusal that matters most.** If a primitive moves the robot the **wrong way**,
the run stops and records nothing. The lateral pair inverts between this stack's frame
(+y is left) and the vendor's raw axis (positive is right), and a profile with those two
swapped strafes into the side of the lane you cleared least. A swapped profile produces a
perfectly plausible speed, so this is checked before the number is written down.

**中文 —— 最要紧的一条拒绝。** 如果某个原语把机器人带向**相反方向**，运行会立即停止且不记录
任何数值。横向的一对在本代码栈的坐标系（+y 为左）和厂商原始轴（正值为右）之间是反的，
配置文件里这两个写反了，机器人就会横向撞进你清理得最不干净的那一侧。
写反的配置照样会给出一个非常像样的速度，所以必须在记录数值之前先查这一项。

**EN — ⚠️ this is only half the gate.** The measured speed is compared against
`--max-vx × --derate`, and the Lite3's `--max-vx` still defaults to **0.35 m/s — the
Go2's** measured envelope. Measuring this side does not make the comparison a Lite3 one.
State `--max-vx` / `--max-vy` / `--max-wz` explicitly on the live run.

**中文 —— ⚠️ 这只是这道闸门的一半。** 实测速度会与 `--max-vx × --derate` 比较，
而 Lite3 的 `--max-vx` 目前仍默认为 **0.35 m/s —— 那是 Go2 的**实测包线。
把这一侧测准了，并不能让这个比较变成 Lite3 的比较。实机运行时请显式写出
`--max-vx` / `--max-vy` / `--max-wz`。

---

## Task 6 — sign the record, then get the flags
## 任务 6 —— 签署记录，然后取得运行参数

**EN** — Each task above wrote its own artefact (`lite3-loaded-radius-LITE3-A.json` and
so on). Fold them into one record for this robot, look at what it closes, sign it, and only
then ask for the flags.

**中文** —— 上面每个任务都写出了自己的结果文件（例如 `lite3-loaded-radius-LITE3-A.json`）。
先把它们合并成这台机器人的一份记录，看看它完成了哪些条目，签名，然后才去取运行参数。

```bash
python3 commission.py --robot-id LITE3-A --firmware V1.0.8 --payload none \
    --merge lite3-motor-temperatures-LITE3-A.json lite3-loaded-radius-LITE3-A.json \
            lite3-camera-LITE3-A.json lite3-gait-floor-LITE3-A.json \
            lite3-actuator-gain-LITE3-A.json

python3 commission.py --record lite3-commissioning-LITE3-A.json --status
python3 commission.py --record lite3-commissioning-LITE3-A.json --review 'Your Name'
python3 commission.py --record lite3-commissioning-LITE3-A.json --emit-flags
```

**EN** — Every artefact is written `"provenance": "provisional"`. A human reads the numbers,
agrees they describe this robot, and signs. Only then will `--emit-flags` print the
`--gait-floor` / `--actuator-gain` / `--robot-radius` / `--policy-scale` / `--calibration`
line a live run needs. This is not paperwork: a number that has been *measured* and a number
that has been *believed* are different things, and only a person turns one into the other.

**中文** —— 每一份结果文件写出时都是 `"provenance": "provisional"`（待审）。
必须由人读过这些数字、确认它们确实描述这台机器人，然后签名。只有签名之后，
`--emit-flags` 才会打印实机运行所需的
`--gait-floor` / `--actuator-gain` / `--robot-radius` / `--policy-scale` / `--calibration`
参数行。这不是走流程：**已测得**的数字和**被相信**的数字是两回事，
而只有人能把前者变成后者。

**EN** — Then paste the `PASTE THIS INTO ISSUE #13` block from each tool into issue #13, one
comment per robot. Repeat everything above for the second Venture. **Do not copy a single
number between the two robots** — that is the entire premise of the issue, and
`commission.py` refuses to merge a result measured on the other robot.

**中文** —— 然后把每个工具打印的 `PASTE THIS INTO ISSUE #13` 段落贴到 issue #13 里，
每台机器人一条评论。第二台 Venture 从头再做一遍全部步骤。
**两台机器人之间不得复制任何一个数字** —— 这正是该 issue 的全部前提，
而且 `commission.py` 会拒绝合并在另一台机器人上测得的结果。

---

## 4. When a gate refuses — the catalogue
## 4. 当程序拒绝执行 —— 对照表

**EN** — **A refusal is a result, not a failure.** These tools stop when they have decided
that no number they could print would mean anything. Working around a refusal — inventing a
value, passing a flag to silence it, retrying until it passes — produces a number that
looks exactly like a measurement and is not one. If a refusal blocks you and this table does
not resolve it, **stop and write it into issue #13**; do not improvise.

**中文** —— **"拒绝执行"是一种结果，不是故障。** 这些工具在判断出"无论打印什么数字都没有意义"时会主动停下。
绕开一次拒绝——自己编一个值、加个参数把它压下去、反复重试直到它通过——
产出的东西看起来和真实测量一模一样，但它不是。
如果某个拒绝挡住了你、而下表没有解答，**请停下来把它写进 issue #13**，不要临场发挥。

| the message says / 提示内容 | what it means / 含义 | what to do / 怎么办 |
| --- | --- | --- |
| `these values must be MEASURED on this robot` | A value has no default here and you did not supply it. / 该值在这里没有默认值，而你没有提供。 | Go and measure it. Do not borrow it from the other robot or from the Go2. / 去实测。不要从另一台机器人或 Go2 借用。 |
| `the operator has not confirmed STANDING + vendor high-level navigation mode` | `--operator-ready` was not given. A robot in manual mode ignores every command silently. / 没有给 `--operator-ready`。手动模式下的机器人会静默忽略所有指令。 | Stand it on the remote, select navigation mode, hold the stop, then pass the flag. / 用遥控器站起、切导航模式、握住急停，再加该参数。 |
| `the Lite3 state stream is silent` / `… stale` | The laptop is not receiving, or is receiving too slowly to time-stamp a measurement. / 笔记本没有收到数据，或数据太慢，无法为测量打上有效时间戳。 | Section 2. Check `ip` in `network.toml` against this laptop's address. / 见第 2 节，核对 `network.toml` 中的 `ip` 与本机地址。 |
| `battery is N%, at or below the … abort limit` | A gait floor measured on a flat battery is a measurement of the battery. / 电量耗尽时测出的步态下限，测的是电池。 | Charge it. Do not lower `--battery-abort`. / 去充电。不要调低 `--battery-abort`。 |
| `the Lite3 reports error_state=N` | The robot is already reporting a fault. / 机器人本身已经在报故障。 | Clear it on the vendor remote first. / 先用厂商遥控器清除故障。 |
| `N of M anchor segments travelled less than 0.05 m` | **The legs were not walking.** Every zero below it is a dead robot, not a floor. / **腿根本没有在走。** 下面所有的零都是"机器停着"，不是"下限"。 | Check the robot actually stood and is in navigation mode, then repeat. Do **not** report the table. / 确认机器人确实站起并处于导航模式，然后重做。**不要**上报那张表。 |
| `N of M control segments drifted …` | The odometry drifts as far as a real walk travels, so this run cannot tell motion from drift. / 里程计的漂移量已经和一次真实行走的位移相当，本次运行无法区分"在动"和"漂移"。 | Increase `--segment` so a walking segment travels further, or report the odometry problem. / 增大 `--segment`，让行走段位移更大；或把里程计问题上报。 |
| `N of M segments travelled less than 0.05 m forward … real and total` | A phase where every segment should have walked contains one that did not. / 在一个"每段都该走"的阶段里，出现了没走的段。 | Repeat. If it recurs at the same segment, report it — something changed mid-run. / 重做。若同一段反复出现，请上报——运行中途有东西变了。 |
| `non-monotonic ladder` | A lower rung walked while a higher one did not. A robot cannot do that. / 低档位走了、而更高档位没走。机器人不可能这样。 | It is a confound, not a floor. Find it — battery, temperature, floor surface — before believing any number in that run. / 这是干扰因素，不是下限。先找出原因（电量、温度、地面），再谈那次运行里的任何数字。 |
| `every rung walked, including the smallest tested` | The floor is **below** the lowest speed you tested, so this run bounded it but did not find it. / 下限**低于**你测试的最低速度，本次运行只给出了上界，没有找到它。 | Rerun with a lower `--ladder-top`. / 用更低的 `--ladder-top` 重跑。 |
| `--envelope-vx … is below the measured gait floor` | The demo would command a speed this robot does not walk at. / 演示会下发一个这台机器人走不动的速度。 | Raise the demo envelope, or accept the floor as the envelope. / 提高演示速度上限，或直接以下限作为上限。 |
| `the … phase needs up to N m of lane` / `swings up to N m to each side` | The run does not fit the room you described. / 该运行放不进你描述的场地。 | Lower `--rungs` or `--segment`, or find a bigger room. Do **not** overstate the lane. / 降低 `--rungs` 或 `--segment`，或换更大的场地。**不要**虚报通道尺寸。 |
| `no Lite3 Venture has been seen to walk on --locomotion-transport udp` | You are about to spend robot time on the interface that has never actuated one. / 你正准备把机器人时间花在一条从未真正驱动过它的接口上。 | Use `--locomotion-transport axis`. Pass `--accept-unwalked-transport` **only** if finding out whether `udp` actuates is itself the point of the run. / 改用 `--locomotion-transport axis`。只有当"验证 udp 到底能不能驱动"本身就是本次运行的目的时，才加 `--accept-unwalked-transport`。 |
| `--locomotion-transport axis discards the commanded magnitude` | A gait floor or an actuator gain is a measurement *of* the commanded speed, and this transport throws it away. Nothing would have crashed: every rung would walk at the same speed and the probe would report the lowest rung as the floor. / 步态下限和执行器增益测的都是"下发速度"本身，而这条通道会把它丢掉。它并不会报错：每一档都会以同样的速度走，探针会把最低的一档当成下限报出来。 | Run `axis_primitive_probe.py` (task 5b) instead. Do **not** pass a transport flag to force it through. / 改跑 `axis_primitive_probe.py`（任务 5b）。**不要**靠改通道参数硬把它跑过去。 |
| `carries the commanded magnitude to the wire, so it has no fixed primitives` | `axis_primitive_probe.py` was pointed at the velocity transport, where there are no primitives to measure. / `axis_primitive_probe.py` 被指向了速度通道，那里没有可测的动作原语。 | Add `--locomotion-transport axis --axis-profile …`. / 加上 `--locomotion-transport axis --axis-profile …`。 |
| `N treatment(s) moved the robot the WRONG WAY` | **The profile's directions do not match this robot.** A swapped lateral pair strafes into unsensed space. / **配置文件里的方向和这台机器人对不上。** 横向写反会让机器人横着撞进没有传感器覆盖的一侧。 | Fix the profile's raw axis values. Do **not** record a speed for it. Report it in issue #13 — it may be a firmware convention difference. / 修正配置文件里的原始轴数值。**不要**为它记录速度。请写进 issue #13——这可能是固件约定的差异。 |
| `there is no low rung in this probe that could legitimately produce nothing` | A primitive commanded at full scale did not move the robot: the raw value is under the firmware dead zone, or the legs were not running. / 一个满量程下发的原语没有让机器人动：要么原始值低于固件死区，要么腿根本没在跑。 | Check the robot stood and is in moving/AI state; then check the raw value against the vendor dead zone for that axis. / 先确认机器人已站立并处于 moving/AI 状态；再对照厂商文档核对该轴的死区与原始值。 |
| `--axis-profile was given but --locomotion-transport is 'udp'` | The profile would be ignored, which looks exactly like a profile that took effect. / 这份配置文件会被忽略，而那看起来和"配置已生效"一模一样。 | Add `--locomotion-transport axis`, or drop the profile. / 加上 `--locomotion-transport axis`，或者去掉该配置文件。 |
| `pass --stance-confirmed` | The outline may have been measured prone or unloaded. / 外形可能是在趴姿或空载状态下量的。 | Re-measure standing and loaded, then confirm. / 在站立且带负载状态下重测，然后确认。 |
| `the front/back extents differ by N×` | The tape was probably referenced to a body edge, not the turning point. / 卷尺原点大概取在了机身边缘，而不是旋转中心。 | Re-drop the plumb line. Use `--asymmetric-confirmed` only if the payload really does hang off one side. / 重新垂铅垂线。只有当负载确实偏向一侧时才使用 `--asymmetric-confirmed`。 |
| `no state frames arrived at all` (temperatures) | Nothing was learned. A silent link and an absent channel look identical from here. / 本次没有得出任何结论。从这里看，"链路沉默"和"通道不存在"完全一样。 | Section 2, then repeat. This must not be reported as an absent channel. / 见第 2 节后重做。**不得**把它当作"通道不存在"上报。 |
| `the temperature field carried N values, not 12` | A partial thermal set is a health gate with a hole in it. / 不完整的温度集合等于健康检查上有个洞。 | Report the partial reading verbatim in issue #13 as a vendor question. / 把这个不完整读数原样写进 issue #13，作为厂商问题。 |
| `the calibration and the production navigator would run the detector differently` | The fit would be made through one detector configuration and used under another. / 标定所用的检测器配置和实际运行时的不一致。 | Use `--marker`, which uses no detector at all. / 改用 `--marker`，它完全不经过检测器。 |
| `--lens-height-source is required` | A height with no method attached is a number somebody will later assume was measured. / 没有说明测量方法的高度，日后会被人默认为"实测值"。 | Say how you measured it, in plain words. / 用普通话把测量方法写清楚。 |
| `is marked 'provisional'` | Nobody has signed for these numbers yet. / 还没有人为这些数字签名。 | Read them, then `--review 'Your Name'`. / 读一遍，然后 `--review 'Your Name'`。 |
| `was measured on 'LITE3-B' but this record is for 'LITE3-A'` | Two robots' numbers were about to be mixed. / 两台机器人的数据差点被混在一起。 | Keep one record per robot. Nothing transfers between them. / 每台机器人一份记录。两者之间没有任何数据可以互通。 |

---

## 5. Prompts for VS Code Copilot or your coding agent
## 5. 给 VS Code Copilot 或其他编程助手的提示词

**EN** — These are the handover mechanism. They are written to be pasted **without editing**
except for the values in `<angle brackets>`. Before pasting any of them, point the session
at `AGENTS.md` at the repository root: it is the standing instruction file for coding agents
in this family of repositories, it carries the absolute naming rule, and an agent that has
not read it will get the conventions wrong.

**中文** —— 这些提示词就是交接机制。它们的设计目标是**不需要修改**即可直接粘贴，
只有 `<尖括号>` 里的值需要替换。粘贴之前，请先让该会话读取仓库根目录的 `AGENTS.md`：
它是本系列仓库对编程助手的常设说明文件，其中包含**绝对不可违反**的命名规则；
没有读过它的助手会把约定写错。

```
Read AGENTS.md and CODING-AGENT-GUIDELINES.md at the repository root before doing
anything else, and follow them. They are not optional.
```

### Prompt A — interpreting a commissioning run that refused
### 提示词 A —— 解读一次被拒绝的标定运行

```
Read AGENTS.md and CODING-AGENT-GUIDELINES.md at the repository root first and follow
them.

Continue from GitHub issue:
https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13

I ran a Lite3 commissioning probe in
robot-stack/deep_robotics/lite3/commissioning/ and it refused instead of producing a
number. The exact output is below, verbatim, including the banner.

<PASTE THE FULL TERMINAL OUTPUT HERE, INCLUDING THE COMMAND LINE YOU RAN>

The room was: <lane length and width in metres, what was in it, where the second robot
was>. The robot was: <standing or prone; which control mode; which firmware; what
payload>. The battery read <N>%.

What I need from you:

1. Find the refusal in the source and quote the code that raised it, with the file and
   line. Explain in one paragraph what condition it was testing and why that condition
   exists — the docstrings in that directory carry the history, use them.
2. Tell me whether this refusal means (a) the room or the setup was wrong, (b) the robot
   was not in the state I thought it was, or (c) the measurement is genuinely telling me
   something about this robot. These need different responses and I need to know which
   one I have.
3. If it is (c), tell me what to report in issue #13 and what the next measurement is.

Constraints, and these are absolute:

- Do NOT propose a workaround that makes the refusal stop firing. The guards are the
  deliverable. If you think a guard is wrong, say so and argue it, but do not weaken it
  as a side effect of helping me.
- Do NOT invent, estimate, or interpolate any measured value. Specifically, do not
  suggest a gait floor, actuator gain, planning radius, lens height or spin rate that
  was not measured on THIS robot. The Go2's 0.35 m/s and 0.20 m/s are a different
  robot's numbers and must not appear as Lite3 values.
- If the answer depends on something only a person in the room can see, say so and tell
  me exactly what to go and look at.

Once you are done, do a pass on your own answer for mistakes, unsupported claims, and
anything you asserted that the code does not actually say.
```

### Prompt B — adding a new measurement script in the same idiom
### 提示词 B —— 按同样的写法新增一个测量脚本

```
Read AGENTS.md and CODING-AGENT-GUIDELINES.md at the repository root first and follow
them.

Continue from GitHub issue:
https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13

Add one new measurement to robot-stack/deep_robotics/lite3/commissioning/ that
measures <WHAT, in one sentence — e.g. "the end-to-end age of a frame from the installed
RGB endpoint">.

Match the existing idiom exactly. Read measurement.py, robot_link.py and
gait_floor_probe.py before writing anything; they define the shape.

The new script must:

1. Produce ONE number, ONE machine-readable artefact via measurement.new_record /
   write_record, and ONE "PASTE THIS INTO ISSUE #13" block via paste_block.
2. Refuse when its preconditions are unmet, by raising measurement.Refusal with a
   message that tells the operator what to go and do. Refusals are the deliverable, not
   an edge case.
3. Carry NO measured default. Where a value must come from the robot, the sentinel is
   None and the script refuses via refuse_unmeasured. Do not add a plausible fallback
   for anything, and do not copy a value from the Go2.
4. Require --robot-id, --firmware and --payload through
   robot_link.add_context_arguments, because issue #13 requires them beside every number.
5. Print a brief() before doing anything: what it does, what it needs from the operator
   in the room, and what the number means. The reader is an operator in Shanghai who did
   not write the code.
6. If it moves the robot: gate on --live AND --operator-ready, go through
   robot_link.connect / robot_link.preflight, measure displacement from POSE and never
   from the platform's velocity estimate, and interleave contemporaneous zero controls
   with anchor segments at a speed already known to walk this robot.

Evidence the change must carry:

- a test_<name>.py in the same directory, in the same plain-assert style with a
  __main__ runner that prints a pass count;
- a test for EVERY refusal branch, and each one must be mutation-checked: break the
  guard, confirm the named test goes red, restore it. Report which mutations you ran and
  the result of each. A guard nobody can make fire is not a guard;
- a dry-run path exercised by a test, because this cannot be run on hardware from here;
- `ruff check . --config robot-stack/deep_robotics/lite3/commissioning/ruff.toml` clean,
  and every test file in that directory still passing. Report the real numbers, not
  "tests pass".
- README.md and RUNBOOK.md in that directory updated: the tool table, and a bilingual
  task section plus any new refusal rows in the catalogue.

Do not modify anything outside
robot-stack/deep_robotics/lite3/commissioning/ without saying so and why.

Once you are done, do a pass on the code for software engineering best practices,
mistakes, sloppy or brute force algorithms, inconsistent style, incorrect comments, etc.
Ask what would make each new test FAIL, and confirm it can.
```

### Prompt C — writing the results up in issue #13
### 提示词 C —— 把结果写进 issue #13

```
Read AGENTS.md and CODING-AGENT-GUIDELINES.md at the repository root first and follow
them, especially the naming rule and the section on writing a continuation comment.

Continue from GitHub issue:
https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13

I have finished a commissioning session on <LITE3-A / LITE3-B>. The artefacts and the
terminal output are below.

<PASTE EACH "PASTE THIS INTO ISSUE #13" BLOCK, AND THE CONTENTS OF
lite3-commissioning-<ROBOT-ID>.json>

Write the continuation comment for issue #13. Four parts, in this order:

1. An outcome table: numbers, not adjectives, with robot ID, firmware, payload and
   command envelope beside every result — the issue requires all four.
2. An answer to each open per-robot checkbox in the issue body, one by one, saying
   explicitly which are now closed, which are not, and which came back NEGATIVE with
   evidence. A measured absence is a result: the motor-temperature channel not existing
   on the high-level interface is the evidence the vendor question needs, not a gap.
3. What the run found that we did not expect, with the observation that showed it.
4. What is still open and which issue it continues in.

Constraints:

- Every number must come from the pasted output. Do not round, do not average across the
  two robots, and do not fill a gap with a value from the other robot or from the Go2 —
  the whole premise of this issue is that nothing transfers between units.
- Keep the tables self-contained: no images and no user-attachments links. They are slow
  or unreachable from mainland China and the argument has to survive without them.
- State the two lateral numbers separately and say in one sentence why they are not
  interchangeable.
- If the artefact is still marked "provisional", say so and say who needs to review it.
- Follow the naming rule in AGENTS.md exactly.

Draft it as a comment for me to review; do not post it.
```

---

## 6. If you are stuck
## 6. 如果卡住了

**EN** — Write it into issue #13 with the exact terminal output, the state of the room, and
what you had already tried. A session that ends with a good continuation comment costs the
next person nothing; a session that ends in someone's scrollback costs them a day. That is
the whole working method in `CODING-AGENT-GUIDELINES.md`, and it applies to people as much
as to agents.

**中文** —— 把它写进 issue #13：附上完整的终端输出、现场情况，以及你已经尝试过什么。
一次以良好交接评论收尾的工作，对下一个人来说成本为零；
一次只留在某人终端回滚记录里的工作，会让下一个人损失一整天。
这就是 `CODING-AGENT-GUIDELINES.md` 里的整套工作方法，它对人和对助手同样适用。
