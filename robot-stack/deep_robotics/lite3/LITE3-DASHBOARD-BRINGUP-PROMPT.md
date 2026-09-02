<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lite3 × Device Connect dashboard — bring-up prompt for a coding agent
# Lite3 × Device Connect dashboard —— 交给编码 agent 的联调提示词

**EN** — **How to use this file.** Paste everything below the `─── PROMPT STARTS ───` line
into VS Code Copilot, or whichever coding agent you use, as your first message. It is written
to be read by an agent and supervised by you. It is staged: **each stage ends with a check,
and the agent is told to stop rather than continue past a failed one.**

**中文** —— **本文件怎么用。** 把 `─── PROMPT STARTS ───` 那一行以下的全部内容，
作为第一条消息粘贴进 VS Code Copilot 或你使用的任何编码 agent。
本文是写给 agent 读、由你监督执行的。它是**分阶段**的：
**每个阶段结束都有一个检查点，并且明确要求 agent 在检查不通过时停下，而不是继续往下走。**

**EN** — The operator's reference for the dashboard itself is
[`../../../dashboard/OPERATOR-GUIDE_CN.md`](../../../dashboard/OPERATOR-GUIDE_CN.md)
(bilingual) and [`../../../dashboard/README.md`](../../../dashboard/README.md) (English, 933
lines, authoritative). Keep the first one open beside you.

**中文** —— dashboard 本身的操作参考是
[`../../../dashboard/OPERATOR-GUIDE_CN.md`](../../../dashboard/OPERATOR-GUIDE_CN.md)
（中英对照）和 [`../../../dashboard/README.md`](../../../dashboard/README.md)
（英文，933 行，**以它为准**）。请把前者开在旁边随时对照。

> **EN** — Commands, flags, filenames and the text the programs print are left in English
> throughout, and must be typed and matched **exactly**. Only the explanation is translated.
>
> **中文** —— 全文中的命令、参数、文件名以及程序打印出来的文字一律保留英文原文，
> 必须**逐字**输入和比对。只有说明文字是中文。

---

## Before you paste it — three things only a human can decide
## 粘贴之前 —— 有三件事只能由人来决定

| | **EN** | **中文** |
| --- | --- | --- |
| 1 | **A named person holds the vendor remote with the emergency stop**, for the whole of any stage that can move a leg, and does nothing else. | **指定一个人全程手持厂商遥控器和急停**，凡是可能让腿动起来的阶段全程在位，且只做这一件事。 |
| 2 | **You stand the robot up and put it into high-level navigation mode on the vendor app.** No software in this repository can do it for you — §5.3. | **由你在厂商 App 上让机器人站起来并切到高层导航模式。** 本仓库没有任何软件能替你做——见 §5.3。 |
| 3 | **Clear both sides of the lane, not just the ends.** The robot has no lateral and no rear sensing. | **通道两侧也要清空，不能只清两端。** 本机没有侧向感知，也没有后向感知。 |

---

─── PROMPT STARTS ─── 提示词从这里开始 ───

You are helping an operator in Shanghai bring up the **Arm Device Connect dashboard** against
a **real Deep Robotics Lite3 Venture** quadruped, for the first time that anyone has done it.
Work through the stages below **in order**. Each stage has a **CHECK**. **If a CHECK fails,
stop, report the exact output, and do not start the next stage.** Do not work around a
refusal, do not invent a value to get past one, and do not retry until something passes.

你正在协助一位上海的操作员，把 **Arm Device Connect dashboard** 接到一台**真实的
Deep Robotics Lite3 Venture** 四足机器人上——**这是所有人中的第一次**。
请**按顺序**执行下面各阶段。每个阶段都有一个 **CHECK（检查点）**。
**若 CHECK 不通过，请停下、原样报告输出，并且不要开始下一阶段。**
不要绕开任何"拒绝执行"，不要为了越过它而自己编造数值，也不要反复重试直到它通过。

## Rules that override anything else you might infer
## 优先于你其他任何推断的硬性规则

**R1. Never install anything on the robot.** The driver you will run does **not** run on the
robot; it runs on the MacBook. If an import fails on the robot, **that is a finding to report,
not a dependency to add**. Never `pip install` on a robot outside a virtualenv, never into its
system Python, and never install a newer Python on it. See `AGENTS.md`.

**R1. 绝不在机器人上安装任何东西。** 你要运行的驱动进程**不在**机器人上跑，而是在
MacBook 上跑。如果机器人上有 import 失败，**那是一个需要上报的发现，不是一个需要装上的依赖**。
绝不在机器人上于虚拟环境之外执行 `pip install`，绝不装进它的系统 Python，
也绝不在它上面装更新版本的 Python。见 `AGENTS.md`。

**R2. Do not enable motion in an early stage.** `--allow-motion` appears for the first time in
**Stage 6**, and only after a scene run has succeeded in Stage 5. If you find yourself adding
it earlier, you have misread the stage — stop and re-read.

**R2. 不要在靠前的阶段开启运动权限。** `--allow-motion` 第一次出现是在**第 6 阶段**，
而且必须在第 5 阶段的场景运行成功之后。如果你发现自己提前加上了它，
说明你读错了阶段——请停下重读。

**R3. A refusal is a result, not a failure.** These tools stop when they have decided that no
number they could print would mean anything. Report the refusal text verbatim.

**R3. "拒绝执行"是一种结果，不是故障。** 这些工具在判断出"无论打印什么数字都没有意义"时
会主动停下。请**原样**报告拒绝信息的文字。

**R4. Never put a robot SSH password, WiFi PSK or API token into a file, a log, a commit or an
issue.** Reference an untracked local file instead.

**R4. 绝不把机器人的 SSH 密码、WiFi 密码或 API token 写进文件、日志、提交或 issue。**
改为引用一个未纳入版本管理的本地文件。

---

## What is already true, so you do not discover it as a bug
## 这些是既成事实，请不要把它们当成 bug 去发现

**Every item below was verified against the code at this file's commit.** Tell the operator
these **before** they hit them.

**下列每一条都对照本文件提交时的代码核实过。** 请在操作员撞上它们**之前**就告诉他。

| # | **EN** | **中文** |
| --- | --- | --- |
| 1 | **Nothing has been measured on a Lite3, on any axis** (`dashboard/README.md:811`), and every motion command to one is unverified. The Go2 path was exercised on hardware; **the Lite3 path never has. This operator is the first.** | **Lite3 上没有任何一个轴被测量过**（`dashboard/README.md:811`），发给它的每一条运动指令都是未经验证的。Go2 那条路径在真机上跑过；**Lite3 这条从来没有。这位操作员是第一个。** |
| 2 | **The camera is exclusive** (`dashboard/README.md:468`, `dashboard/camera_source.py:247`). Lite3 frames come from an OpenCV `VideoCapture`, typically exclusive on Linux, so **while a `lite3_visual_nav` run holds the camera the dashboard cannot have it.** He **cannot** watch the live feed and run the policy at the same time. It reports `the camera is in use`. **Not a bug.** ⚠️ **But see B2 — on a Lite3 the panel is blank anyway, for a different and worse reason.** | **相机是独占的**（`dashboard/README.md:468`、`dashboard/camera_source.py:247`）。Lite3 的图像来自 OpenCV `VideoCapture`，在 Linux 上通常独占，所以**当 `lite3_visual_nav` 占用相机时，dashboard 就拿不到它**。**不能**一边看实时画面一边跑策略。它会提示 `the camera is in use`。**这不是缺陷。** ⚠️ **但请看 B2——在 Lite3 上画面无论如何都是空的，原因不同且更严重。** |
| 3 | **Posture is operator-controlled** (`dashboard/robot_driver.py:1526`). Lite3 posture goes through the vendor app; the dashboard's stop only **stops** — it does not lie the robot down as it does on a Go2. The driver reports `lie_down_changes_posture: false` itself. **He must stand the robot and enable high-level navigation mode on the vendor interface himself** (`dashboard/README.md:726`). | **姿态由操作员控制**（`dashboard/robot_driver.py:1526`）。Lite3 的姿态通过厂商 App 控制；dashboard 的 stop **只是停住**，不会像在 Go2 上那样让它趴下。驱动进程自己会报告 `lie_down_changes_posture: false`。**必须由他本人在厂商界面上让机器人站起来并切到高层导航模式**（`dashboard/README.md:726`）。 |
| 4 | **Steering will not be proportional** (`robot-stack/deep_robotics/lite3/locomotion/lite3_axis_locomotion.py:252`). The axis mapping is **sign-only: the commanded magnitude is discarded**, and line 309 snaps direction to one of 8 octants. **The motion pad will feel like an 8-way d-pad, not a joystick. That is the transport, not a fault.** Nothing here scales with `vx`, so **nothing here honours `--derate` either**. | **转向不是连续的**（`robot-stack/deep_robotics/lite3/locomotion/lite3_axis_locomotion.py:252`）。轴映射**只取符号：指令的大小被丢弃**；第 309 行还会把方向归一到 8 个八分方向之一。**运动面板用起来会像八向十字键，不是摇杆。这是传输层设计，不是故障。** 这里没有任何东西随 `vx` 缩放，所以**这里也不遵守 `--derate`**。 |
| 5 | **Two gait floors differing by 1.75x, and both are Go2 numbers.** Issue #42 measured **0.35 m/s forward** and **0.20 m/s lateral** — its title says "2x", its body says "nearly 2x", the ratio is 1.75. **Its table is Go2 data; the issue says so itself.** `--gait-floor` is **one** field applied to the forward axis. **There is no measured floor on any Lite3 axis, so the gait-floor guard cannot protect him the way it protects a Go2.** | **两个步态下限相差 1.75 倍，而且都是 Go2 的数据。** issue #42 实测**前进 0.35 m/s**、**横移 0.20 m/s**——标题写的是 "2x"，正文写的是"接近 2 倍"，实际比值是 1.75。**它的表格是 Go2 数据，issue 正文自己就是这么写的。** 而 `--gait-floor` 只有**一个**字段，且作用在前进轴上。**Lite3 三个轴都没有实测下限，所以步态下限保护在这里起不到它在 Go2 上的作用。** |

---

## ⛔ STOP — read this before Stage 3. The motion path is BLOCKED, not merely unmeasured
## ⛔ 停 —— 第 3 阶段之前必须读。运动路径是**被堵死的**，不只是"没测过"

**EN** — This was found by running the code, not by reading it, on a workstation in the same
state as this operator's MacBook. **Tell him now.** Everything above is about a path that is
*unproven*; this is a path that is *closed*.

**中文** —— 下面这些是**跑代码跑出来的**，不是读代码读出来的，
运行环境与这位操作员的 MacBook 状态相同。**请立刻告诉他。**
上面那些讲的是一条**尚未验证**的路径；这里讲的是一条**已经堵死**的路径。

### B1. The dashboard can only drive a Lite3 over ROS 2 — a transport these robots may not have
### B1. dashboard 只能通过 ROS 2 驱动 Lite3 —— 而这些机器人可能没有这条通道

**EN** — `drive_bridge.py`'s `_load_lite3` is nine lines and **takes no transport argument**.
It constructs `Lite3Locomotion`, whose default `implementation_factory` is `_ros2_locomotion`,
which imports `arm_dc_robotkit.ros2_twist_locomotion` (or a flat `ros2_twist_locomotion`).
**There is no `--locomotion-transport` anywhere in `dashboard/`.** The `udp` and `axis`
transports exist only behind `visual_nav`/`mappo_drive`.

**中文** —— `drive_bridge.py` 里的 `_load_lite3` 只有九行，**不接受任何通道参数**。
它构造 `Lite3Locomotion`，而后者默认的 `implementation_factory` 是 `_ros2_locomotion`，
它 import 的是 `arm_dc_robotkit.ros2_twist_locomotion`（或平铺的 `ros2_twist_locomotion`）。
**`dashboard/` 目录下根本没有 `--locomotion-transport`。**
`udp` 和 `axis` 两条通道只存在于 `visual_nav` / `mappo_drive` 背后。

**EN** — This is the wrong transport for these robots. The Lite3 README says `jetson2motion`
"runs on a *perception* host … so it needs a ROS 2 Foxy runtime on a computer **these two
Ventures may not have**." **The only transport a Venture has ever actually walked on is the
profile-gated axis UDP one — and the dashboard cannot reach it at all.**

**中文** —— 这对这些机器人来说是**错的通道**。Lite3 的 README 写明 `jetson2motion`
"运行在**感知主机**上……因此需要一个 ROS 2 Foxy 运行时，而**这两台 Venture 可能并没有**"。
**Venture 唯一真正走起来过的通道，是受配置文件约束的 axis UDP 那条——而 dashboard 完全够不到它。**

**EN** — **This is what the operator will actually see**, reproduced on a clean workstation:

**中文** —— **操作员实际会看到的就是下面这个**，在干净的工作站上复现：

```
$ python3.11 drive_bridge.py status --platform lite3
{"ok": false, "refused": false, "error": "ModuleNotFoundError: No module named 'ros2_twist_locomotion'"}
```

⛔ **EN** — **Read `"refused": false` carefully. The dashboard colours this as a FAULT, not as
a refusal** — so it sends the operator off to diagnose the robot, when nothing is wrong with
the robot. The message names neither ROS 2, nor the Lite3, nor which host, nor what to install.
**Identical output for `stop`, `stand`, `stand-down` and `pose-stream`** — so **the STOP
button's velocity-zero backstop fails on a Lite3 too.** (`stop` still SIGTERMs a running
policy, so a run does stop; but the result reads red for an unrelated reason.)

⛔ **中文** —— **仔细看 `"refused": false`。dashboard 会把它显示为"故障"，而不是"拒绝执行"**——
于是操作员会被引去排查机器人，而机器人其实没有任何问题。
这条信息既没提 ROS 2，也没提 Lite3，没说是哪台主机，更没说该装什么。
**`stop`、`stand`、`stand-down`、`pose-stream` 的输出完全相同**——
也就是说 **STOP 按钮的"速度归零"兜底在 Lite3 上同样失效**。
（`stop` 仍然会向正在运行的策略发 SIGTERM，所以运行确实会停；但返回结果会因为一个无关的原因显示为红色。）

⛔ **EN** — **R1 still applies, and applies hardest here.** Do **not** `pip install` anything
on the robot to make this import succeed. **Report it and stop.**

⛔ **中文** —— **R1 在这里最为适用。** **不要**为了让这个 import 成功而在机器人上
`pip install` 任何东西。**报告它，然后停下。**

### B2. The Lite3 camera panel can never produce a frame
### B2. Lite3 的相机面板永远出不了图

⛔ **EN** — This is **separate from, and worse than, caveat 2.** `Lite3CameraSource.read()`
(`dashboard/camera_source.py:269-273`) returns `getattr(frame, "jpeg", None)`. The Lite3
`Frame` (`lite3/visual_nav/camera.py:28-35`) is a dataclass with fields `image`,
`capture_time`, `seq`, `stamp` — **there is no `jpeg` field.** So `read()` returns `None`
forever, the encoder emits nothing, and there is **no error, no `camera_error` and no log
line**: a permanently blank feed that looks exactly like a camera problem. There is **no Lite3
equivalent of `go2_frame_server.py`** in the tree.

⛔ **中文** —— 这一条**与第 2 条既成事实无关，而且比它更严重。**
`Lite3CameraSource.read()`（`dashboard/camera_source.py:269-273`）返回的是
`getattr(frame, "jpeg", None)`。而 Lite3 的 `Frame`（`lite3/visual_nav/camera.py:28-35`）
是一个 dataclass，字段为 `image`、`capture_time`、`seq`、`stamp`——**根本没有 `jpeg` 字段**。
所以 `read()` 永远返回 `None`，编码器什么也不输出，而且**没有报错、没有 `camera_error`、
没有任何日志行**：呈现为一个永久空白的画面，看起来和相机故障一模一样。
本仓库中**没有 `go2_frame_server.py` 的 Lite3 对应物**。

**EN** — So on a Lite3 the viewport is blank **whether or not** anything else holds the camera.
Caveat 2 is still true and still worth knowing; it is simply not the reason the panel is empty.

**中文** —— 所以在 Lite3 上，**无论**有没有别的程序占着相机，画面窗口都是空的。
第 2 条依然成立、依然值得知道；它只是**并不是**画面为空的原因。

### B3. Turning is permanently unreachable; reverse is the only ungated motion
### B3. 转向永久不可达；倒退是唯一没有闸门的运动

⚠️ **EN** — `walk_forward`, `strafe_left` and `strafe_right` pass `force=force` through to the
gait-floor check, so `force` can get past their refusal. **`turn_left` and `turn_right` take
only `(seconds, rate_rad_s)` — there is no `force` parameter to pass** (`robot_driver.py:783`,
`:796`), and yaw is floor-checked like everything else. **On a Lite3 they can never succeed.**

⚠️ **中文** —— `walk_forward`、`strafe_left`、`strafe_right` 会把 `force=force`
一路传给步态下限检查，所以 `force` 能越过它们的拒绝。
**而 `turn_left` 和 `turn_right` 只接受 `(seconds, rate_rad_s)`——根本没有 `force` 参数可传**
（`robot_driver.py:783`、`:796`），而偏航同样要过下限检查。
**在 Lite3 上它们永远不可能成功。**

⛔ **EN** — Meanwhile **`walk_back` is never floor-checked at all**, so with default parameters
**the only Lite3 motion the dashboard will let through is reverse** — open-loop, into space
with **no rear sensing**, on a robot with no measured gait. That is a defensible pair of local
decisions with a bad joint outcome. **Do not use it as your "does motion work?" probe.**

⛔ **中文** —— 与此同时，**`walk_back` 根本不做下限检查**，
所以在默认参数下，**dashboard 唯一放行的 Lite3 运动就是倒退**——
开环、进入**完全没有后向感知**的空间、而且这台机器人没有任何实测步态数据。
这是两个各自站得住脚的局部决定叠加出的坏结果。
**不要拿它当作"运动能不能用"的试探手段。**

### What this means for the stages below
### 这对下面各阶段意味着什么

| **EN** | **中文** |
| --- | --- |
| Stages 1, 2 and 4 (with `--simulate`) are worth doing and will work. | 第 1、2 阶段，以及用 `--simulate` 的第 4 阶段，值得做，而且能跑通。 |
| **Stage 3 will end at B1's `ModuleNotFoundError` unless this robot really does ship a ROS 2 Foxy host with the robotkit.** Check that first — it is one SSH command — and if it does not, **stop at Stage 3 and report.** | **除非这台机器人确实带了装好 robotkit 的 ROS 2 Foxy 主机，否则第 3 阶段会终止在 B1 的 `ModuleNotFoundError`。** 请先确认这一点——一条 SSH 命令即可——如果没有，**在第 3 阶段停下并上报**。 |
| **Stages 5 and 6 against real hardware are blocked by B1.** Do not attempt dashboard-driven Lite3 motion. | **第 5、6 阶段在真机上被 B1 堵死。** 不要尝试用 dashboard 驱动 Lite3 运动。 |
| **What does work today** is the read-only commissioning path — `commissioning/lite3_state_probe.py` is stdlib, read-only, and cannot command a leg. It produces the very numbers everything else refuses without. Its bilingual runbook is `commissioning/RUNBOOK.md`. | **今天真正能用的**是只读的标定路径——`commissioning/lite3_state_probe.py` 只用标准库、只读、不可能给腿下指令。它产出的正是其他一切在缺失时都会拒绝执行的那些数值。它的中英对照手册是 `commissioning/RUNBOOK.md`。 |
| For a visual demo, `--platform lite3 --simulate` presents the Lite3's rules and refusals against the bench double, and is badged as simulated. | 若只是要视觉演示，`--platform lite3 --simulate` 会在台架替身上呈现 Lite3 的各项规则与拒绝行为，并明确标注为"模拟"。 |

**EN** — **The one change that would unblock this** is a `--locomotion-transport` flag on
`drive_bridge.py` mirroring the one in `robot_bindings.py`, so the Lite3 backend can reach the
axis transport these robots have actually walked on. **That is a real piece of work, not a
config fix**, and it still needs a commissioned axis profile behind it. Do not attempt it as
part of a bring-up.

**中文** —— **能解开这个死结的唯一改动**，是在 `drive_bridge.py` 上加一个
与 `robot_bindings.py` 中同名的 `--locomotion-transport` 参数，
让 Lite3 后端能够到这些机器人真正走起来过的 axis 通道。
**那是一项实打实的开发工作，不是改配置**，而且它背后仍然需要一份已完成标定的 axis 配置文件。
**不要把它当作联调的一部分顺手去做。**

---

## Stage 1 — the workstation, and only the workstation
## 第 1 阶段 —— 只准备工作站

**Goal / 目标** — Get the three programs importable on the **MacBook**. Nothing touches the
robot in this stage. / 让三个程序在 **MacBook** 上能正常 import。本阶段完全不碰机器人。

⛔ **The robot cannot host the driver, and that is architecture rather than a packaging bug.**
`device-connect-edge` requires Python ≥ 3.11 and a virtualenv **cannot** supply a version the
machine does not have. Do not try to move the driver onto the robot.

⛔ **机器人上跑不了这个驱动进程，这是架构决定的，不是打包问题。**
`device-connect-edge` 需要 Python ≥ 3.11，而虚拟环境**无法**提供一台机器本身没有的版本。
不要试图把驱动进程挪到机器人上。

```bash
python3.11 -m pip install device-connect-edge device-connect-agent-tools aiohttp numpy Pillow
```

⚠️ `aiohttp` and `numpy` are **not optional and nothing else installs them**. On macOS
`python3` is the Command Line Tools 3.9.6 however many Homebrew Pythons are installed —
**`python3.11` is literal.**

⚠️ `aiohttp` 和 `numpy` **不是可选项，也没有别的包会顺带装上它们**。在 macOS 上，
无论装了多少个 Homebrew Python，`python3` 都是系统自带的 3.9.6——**`python3.11` 是字面意思。**

**CHECK 1** — from `dashboard/`, `./start-dashboard.sh --dry-run` prints three commands and
exits without starting anything. **If it refuses, it refuses by name** (interpreter too old, a
missing package with the exact `pip install` for that interpreter, a port held with the pid).
**Fix what it names. Do not proceed on a refusal.**

**CHECK 1** —— 在 `dashboard/` 目录下执行 `./start-dashboard.sh --dry-run`，
它应打印三条命令并直接退出，不启动任何东西。**如果它拒绝，它会指名道姓地说明原因**
（解释器版本过低、缺哪个包并给出针对该解释器的确切 `pip install`、端口被哪个 pid 占用）。
**它指出什么就修什么。拒绝状态下不要继续。**

---

## Stage 2 — serve the checkpoints from a local folder on the MacBook
## 第 2 阶段 —— 从 MacBook 上的本地目录提供模型文件

**Goal / 目标** — `model_server.py` serves `.npz` checkpoints over HTTP, and writes the
sources file so the address is typed once instead of twice. /
让 `model_server.py` 通过 HTTP 提供 `.npz` 权重文件，并写出 sources 文件，
使地址**只写一次**而不是在两处各写一遍。

```bash
# Replace <LAN> with the MacBook's address on the demo LAN, e.g. 192.168.1.50
python3.11 model_server.py --models-dir ../policy/models --port 8800 \
        --host <LAN> \
        --emit-sources /tmp/sources.json --label "Shanghai model server"
```

⛔ **`--host 127.0.0.1` is unreachable from a robot**, and the server says so at startup:
`bound to loopback: a ROBOT cannot fetch from this address.` **The robot does the fetching**,
not the browser — a loopback address gives a field that looks right and fails on fetch.

⛔ **绑定 `127.0.0.1` 时机器人访问不到**，服务启动时会明确提示：
`bound to loopback: a ROBOT cannot fetch from this address.`
**下载动作是机器人执行的**，不是浏览器——填回环地址会出现"看起来填对了、一下载就失败"。

⚠️ There is **no authentication of any kind** on this server or on the dashboard. Anyone who
can reach the port can use it. Bind a demo-LAN address, not a routable public one.

⚠️ 这个服务和 dashboard **都没有任何身份认证**。任何能访问该端口的人都能使用它。
请绑定演示局域网地址，不要绑定可公网路由的地址。

**CHECK 2** — from the **MacBook**: `curl -s http://<LAN>:8800/healthz` returns `{"ok": true}`,
and `curl -s http://<LAN>:8800/index.json` lists the checkpoints. Then **from the robot**,
over SSH, `curl -s http://<LAN>:8800/healthz` returns the same. **The second curl is the one
that matters** — it is the only proof the robot can reach the source. **If it fails, stop:
Stage 6's Download cannot work, and every later stage will mislead you about why.**

**CHECK 2** —— 在 **MacBook** 上：`curl -s http://<LAN>:8800/healthz` 返回 `{"ok": true}`，
`curl -s http://<LAN>:8800/index.json` 能列出权重文件。然后**在机器人上**通过 SSH 执行
`curl -s http://<LAN>:8800/healthz`，应返回同样结果。**第二条 curl 才是关键的那条**——
它是"机器人能访问到模型源"的唯一证据。
**如果它失败，请停下：第 6 阶段的 Download 不可能成功，而且后面每个阶段都会让你误判原因。**

---

## Stage 3 — bring up the driver against the real robot, read-only
## 第 3 阶段 —— 让驱动进程连上真机，只读

**Goal / 目标** — `robot_driver.py --platform lite3` appears on the mesh and reads the robot.
**No motion flag in this stage.** /
让 `robot_driver.py --platform lite3` 出现在设备网上并能读到机器人。
**本阶段不加任何运动相关参数。**

```bash
python3.11 robot_driver.py --platform lite3 --package /tmp/package \
        --bridge-python <interpreter ON THE ROBOT that imports the Lite3 SDK> \
        --model-sources /tmp/sources.json
```

⚠️ **Use a COPY of `../policy` as `--package`** (`/tmp/package` above): arming a checkpoint
**rewrites its `config.json`**.

⚠️ **`--package` 请指向 `../policy` 的副本**（上面的 `/tmp/package`）：
挂载模型会**改写其中的 `config.json`**。

⚠️ **`--bridge-python` is the interpreter that can import the robot's SDK. It is NOT the one
running the driver**, and getting it wrong makes every command fail with an import error.
`get_capabilities` reports the path it will use — **check it before you need it.**

⚠️ **`--bridge-python` 是那个能 import 机器人 SDK 的解释器，不是正在跑驱动进程的那个**；
填错会让每一条命令都以 import 错误失败。`get_capabilities` 会报告它将要使用的路径——
**在你真正需要它之前先核对一遍。**

⛔ **That interpreter is on the robot and this process is not.** `--bridge-python` alone is not
enough: it has to name something on the **MacBook** that reaches the robot's Python — today an
SSH wrapper, which is **a workaround and not a supported deployment**. Its limits are real and
you must tell the operator now: **`list_models`, `free_bytes` and `download_model` then act on
the MacBook's `--package` directory**, so the checkpoint panel is answering about the wrong
machine. Only `get_status`, `get_capabilities` and the motion gate genuinely reach the robot.

⛔ **那个解释器在机器人上，而这个进程不在。** 光有 `--bridge-python` 是不够的：
它必须指向 **MacBook 上**某个能够到达机器人 Python 的东西——目前是一个 SSH wrapper，
**那是权宜之计，不是受支持的部署方式**。它的限制是实实在在的，现在就要告诉操作员：
**`list_models`、`free_bytes`、`download_model` 操作的是 MacBook 上的 `--package` 目录**，
所以模型面板回答的是另一台机器的情况。只有 `get_status`、`get_capabilities`
和运动闸门是真正到达机器人的。

**EN** — **Before you run it, do the one-command check that decides whether this stage can
pass at all** (B1). On the machine `--bridge-python` points at:

**中文** —— **在运行它之前，先做那条决定本阶段能否通过的单条命令检查**（B1）。
在 `--bridge-python` 所指向的那台机器上执行：

```bash
python3 -c "import arm_dc_robotkit.ros2_twist_locomotion; print('ros2 transport present')"
```

**CHECK 3** — **Two outcomes, and only one of them continues.**

**CHECK 3** —— **两种结果，其中只有一种可以继续。**

| outcome / 结果 | **EN** | **中文** |
| --- | --- | --- |
| the import **fails** | **This is the expected outcome** (B1). `get_status` will return `{"ok": false, "refused": false, "error": "ModuleNotFoundError: No module named 'ros2_twist_locomotion'"}`. **STOP HERE.** Report it, note that Stages 5 and 6 are blocked, and move the operator to the read-only commissioning path instead. **Do not install anything on the robot (R1).** | **这是预期结果**（B1）。`get_status` 会返回 `{"ok": false, "refused": false, "error": "ModuleNotFoundError: No module named 'ros2_twist_locomotion'"}`。**就此停下。** 上报此事，说明第 5、6 阶段已被堵死，并让操作员转向只读的标定路径。**不要在机器人上安装任何东西（R1）。** |
| the import **succeeds** | This unit really does ship the ROS 2 host. Continue: `get_capabilities` returns and its `bridge_python` field is the path you intended; `get_status` returns real state. **Expect it to be slow** — a `status` read costs ~1.95 s, measured, nearly all of it cold SDK import and discovery paid per invocation. **Slow is expected; an import error is not.** | 说明这台机器确实带了 ROS 2 主机。继续：`get_capabilities` 能返回且其 `bridge_python` 字段是你想要的路径；`get_status` 返回真实状态。**慢是正常的**——一次 `status` 读取实测约 1.95 s，其中绝大部分是每次调用都要重付的 SDK 冷启动和发现过程。**慢是预期内的；import 错误不是。** |

**CHECK 3** —— `get_capabilities` 能返回，且其中的 `bridge_python` 字段是你想要的那个路径。
`get_status` 返回机器人的真实状态而不是报错。**慢是正常的**：
一次 `status` 读取实测约 1.95 s，其中绝大部分是每次调用都要重付的 SDK 冷启动和发现过程。
**慢是预期内的；import 错误不是。若出现 import 错误，请报告路径并停下——
不要为了让它消失而在机器人上安装任何东西（R1）。**

---

## Stage 4 — the dashboard, the fleet, the model server, loading and swapping
## 第 4 阶段 —— dashboard、机器人列表、模型服务、下载与切换

**Goal / 目标** — The page shows the real robot and the model round trip works. Still no
motion. / 页面上能看到真机，且模型这一圈往返是通的。**仍然不涉及任何运动。**

```bash
python3.11 server.py --port 8080     # then open http://127.0.0.1:8080
```

⚠️ **This dashboard has no login.** The default bind is loopback; `--host 0.0.0.0` reaches it
from the demo LAN and means anyone who can reach the port can drive any robot on the mesh that
was started with motion enabled.

⚠️ **本 dashboard 没有任何登录认证。** 默认只绑定回环地址；`--host 0.0.0.0`
可以从演示局域网访问，这意味着**任何能访问该端口的人**都能操控设备网上所有已开启运动权限的机器人。

**In order / 按顺序：**

1. **View the fleet / 查看列表** — `MESH UP` in the top bar; the Lite3 on a row with a green
   `LIVE` badge and a pose. `STOP ALL (n)` counts what it will hit.
2. **Connect to the model server / 连接模型服务** — open **Load from Cloud AI**. The Source
   picker should already read `Shanghai model server` and the address should already be filled
   in, because the **robot** advertised it from `--model-sources`. Press **Browse**.
3. **Load a model / 下载模型** — **Download** on a browsed row. Read caveat §Stage 3 again
   before believing the free-space and list numbers.
4. **Swap models / 切换模型** — **Arm** on a checkpoint row. **It takes effect on the NEXT
   run, not on one in progress** — `MappoController` loads its weights once, at construction,
   so no code path exists to change them mid-run.

⚠️ **The camera viewport will be empty or will say `the camera is in use` if anything else has
the camera.** That is caveat 2 and it is expected. Do not debug it.

⚠️ **如果有别的程序占着相机，画面窗口会是空的，或者提示 `the camera is in use`。**
那是第 2 条既成事实，属于预期行为。**不要去排查它。**

**CHECK 4** — **Browse** returns a row saying `served by mappo-model-server`. **That round
trip is the proof that the robot, not the browser, can reach the source.** If Browse returns
nothing, go back to CHECK 2 — do not continue.

**CHECK 4** —— **Browse** 返回带有 `served by mappo-model-server` 字样的条目。
**这一次往返就是"机器人（而不是浏览器）能访问到模型源"的证据。**
如果 Browse 什么也没返回，请回到 CHECK 2——**不要继续**。

---

## Stage 5 — the scene run. This one cannot move the robot
## 第 5 阶段 —— 场景运行。这一次不可能让机器人动

⛔ **EN** — **Do not start this stage on real hardware if CHECK 3 took the "import fails"
branch.** B1 blocks it. Everything below is correct and worth reading, and it is what to do
once the transport question is answered — on a `--simulate` driver in the meantime.

⛔ **中文** —— **如果 CHECK 3 走的是"import 失败"那一支，就不要在真机上开始本阶段。**
B1 把它堵死了。下面的内容都是正确的、值得读，它是**通道问题解决之后**该做的事——
在此之前请在 `--simulate` 的驱动进程上做。

### Two flags that `start_run` will NOT add for you, and one that exits 2
### `start_run` 不会替你加的两个参数，以及一个会让进程以 2 退出的参数

⛔ **EN** — `run_control.build_run_argv` spells `--package`, the profile's `extra_args`,
`--policy-mode`, the servo flags, `--max-seconds`, the output paths and `--live` — **and
nothing else.** In particular it **does not spell `--operator-ready`**, while `mappo_drive.py`
on a Lite3 **refuses `--live` without it**: `[lite3] REFUSING TO WALK: missing --operator-ready
after STANDING + navigation mode`. **So `--operator-ready` has to go in the run profile's
`extra_args`** — which turns a per-session operator confirmation into a **static constant in a
JSON file**. Say that out loud to the operator; it is a real weakening of gate 2 and he should
know it is there.

⛔ **中文** —— `run_control.build_run_argv` 只会写出 `--package`、配置里的 `extra_args`、
`--policy-mode`、伺服相关参数、`--max-seconds`、输出路径和 `--live`——**再没有别的了**。
特别是它**不会写出 `--operator-ready`**，而 Lite3 上的 `mappo_drive.py`
**没有它就拒绝 `--live`**：`[lite3] REFUSING TO WALK: missing --operator-ready after
STANDING + navigation mode`。**所以 `--operator-ready` 必须写进运行配置的 `extra_args` 里**——
这就把**每次由操作员当场确认**的动作，变成了**JSON 文件里的一个静态常量**。
请明确告诉操作员；这是对第 2 道闸门的实质削弱，他应该知道它的存在。

⛔ **EN** — `--camera-source` is `required=True` on the Lite3 (`robot_bindings.py:88`). **Omit
it from `extra_args` and argparse exits 2 on the far end of an SSH connection** — the run
simply never starts, and the reason is on a stream you are not watching.

⛔ **中文** —— 在 Lite3 上 `--camera-source` 是 `required=True`（`robot_bindings.py:88`）。
**如果 `extra_args` 里漏了它，argparse 会在 SSH 连接的另一端以 2 退出**——
运行根本不会开始，而原因出现在一个你没有在看的输出流上。

**Goal / 目标** — A full MAPPO run with the camera, the detector, the policy, the planner's
veto and the telemetry — **and no `--live`, so it cannot command a leg.** /
一次完整的 MAPPO 运行：相机、检测器、策略、规划器否决逻辑、遥测全部在内——
**并且不带 `--live`，所以它不可能给腿下指令。**

**EN** — To be able to start a run at all, add `--run-profile` to the Stage 3 command line.
Without it `start_run` refuses and `get_capabilities` reports `run.supported: false`.

**中文** —— 要能启动运行，需要在第 3 阶段的命令行上加 `--run-profile`。
没有它，`start_run` 会拒绝执行，且 `get_capabilities` 会报告 `run.supported: false`。

⛔ **Edit `run-profile.example.json` first. Every path in it is a path ON THE ROBOT**;
`launch_prefix` must authenticate **without a password** (put a key on the robot — a password
is **refused at load, because that field is published**); and `heading_servo_flag` is a
property of the tree it points at rather than a preference, so run **that tree's**
`mappo_drive.py --help` and look. **The driver prints both commands it would run at startup —
that is the moment a wrong path is still cheap.**

⛔ **先改 `run-profile.example.json`。里面每一个路径都是机器人上的路径**；
`launch_prefix` 必须能**免密**认证（在机器人上放公钥——**写密码会在加载时被拒绝，
因为这个字段会被广播出去**）；`heading_servo_flag` 取决于它所指向的那份代码树，不是个人偏好，
所以请去跑**那份代码树的** `mappo_drive.py --help` 看一眼。
**驱动进程启动时会把它将要执行的两条命令都打印出来——那是发现路径写错代价最小的时刻。**

**EN** — Press **Start** with no arguments. `start_run()` with no arguments builds the command
line with **no `--live`**, and `--live` is the only flag that commands a leg. This is an
**absent capability, not a checked permission** — which is what makes it the right thing to
press first, every time.

**中文** —— 直接按 **Start**，不带任何参数。不带参数的 `start_run()` 生成的命令行里
**没有 `--live`**，而 `--live` 是唯一会给腿下指令的参数。这**不是"权限被拒绝"，
而是"这个能力根本不存在"**——正因如此，**每次都应该先按它**。

### If you record this run, record it twice
### 如果要录制这一次运行，请录两份

⛔ **`--record` alone produces a video that CANNOT be used as training data.** It writes the
**annotated** frame — the HUD, the plan-view inset and **a box around every detection** — and
that is what makes it readable and **what makes it useless as training data: the label is
burned into the pixels the model would have to learn from.** `--record-raw` writes the same
frame **before** any of that is drawn on it. (`visual_nav.py:1167`, `:1627` — the Lite3 uses
this same shared navigator; `lite3_visual_nav.py` is a shim that calls into it.)

⛔ **只用 `--record` 录出来的视频，不能用作训练数据。** 它写出的是**带标注**的画面：
HUD、俯视小地图，以及**每个检测目标外面的框**——这让画面便于人看，
但也**使它完全不能用作训练数据：标签被烧进了模型本该去学习的那些像素里**。
`--record-raw` 写出的是**在画任何标注之前**的同一帧。
（`visual_nav.py:1167`、`:1627`——Lite3 用的就是这同一个共享导航模块，
`lite3_visual_nav.py` 只是一层调用它的薄封装。）

```bash
--record run.mp4  --record-raw run-raw.mp4  --telemetry run.jsonl
```

**EN** — **Keep the telemetry `.jsonl` from the same run.** Raw frames, annotated frames and
telemetry all share one frame index via `perception.video_frame`, which is what turns the
detector's own boxes into **labels for the clean pixels**. Both recorders are advanced by one
gate, so frame *n* of one file is frame *n* of the other.

**中文** —— **保留同一次运行产生的遥测 `.jsonl`。** 原始帧、标注帧和遥测三者通过
`perception.video_frame` 共用同一个帧号，正是它让检测器自己输出的框成为**干净像素的标签**。
两个录制器由同一个节拍推进，所以一个文件的第 *n* 帧就是另一个文件的第 *n* 帧。

⚠️ **Recording costs control-loop rate.** Measured on a Go2: **246.4 ms** per new-result tick
with `--record` against **100.6 ms** without (issue #18); the recorder is still synchronous on
the perception cycle (`visual_nav.py:1216`). **A run recorded for training is not a run whose
timing numbers mean anything.** If you want both, record a separate pass.

⚠️ **录制会拖慢控制环频率。** 在 Go2 上实测：开 `--record` 时每个新结果周期 **246.4 ms**，
不开时 **100.6 ms**（issue #18）；录制目前仍与感知周期同步执行（`visual_nav.py:1216`）。
**为采训练数据而录的那次运行，其时间性能数字没有参考价值。** 两者都要的话，请分两次跑。

**CHECK 5** — The run starts, the event drawer shows it, telemetry is being written, and **the
robot does not move**. Then press **stop** and confirm the run ends. **If the robot moves
during this stage, something is badly wrong: stop everything, hit the vendor emergency stop,
and report it — a run with no `--live` has no code path that commands a leg.**

**CHECK 5** —— 运行能启动，事件抽屉里能看到，遥测在写入，**并且机器人不动**。
然后按 **stop**，确认运行结束。
**如果机器人在本阶段动了，说明出了严重问题：立即停掉一切、按下厂商急停并上报——
一次不带 `--live` 的运行，在代码里根本没有能给腿下指令的路径。**

---

## Stage 6 — motion. Only now, and only with a person on the abort
## 第 6 阶段 —— 运动。到这里才开始，且必须有人守着急停

⛔ **EN** — **BLOCKED on real hardware by B1 unless CHECK 3 took the "import succeeds"
branch.** Do not attempt dashboard-driven Lite3 motion otherwise. Read this stage anyway — the
gates it describes are the ones that will apply when the transport question is answered.

⛔ **中文** —— **除非 CHECK 3 走的是"import 成功"那一支，否则本阶段在真机上被 B1 堵死。**
其余情况下不要尝试用 dashboard 驱动 Lite3 运动。但请照样把本阶段读完——
它描述的那些闸门，正是通道问题解决之后会生效的那些。

⛔ **Do not begin this stage until CHECK 5 has passed.** `robot-stack/SAFETY.md` governs
everything in it and is not optional.

⛔ **CHECK 5 通过之前不要开始本阶段。** `robot-stack/SAFETY.md` 管辖本阶段的一切，不可跳过。

**EN** — **The human, not you, does these two first:**
1. **Stand the robot and enable high-level navigation mode on the vendor interface.** No
   software here can do it — caveat 3.
2. **Take up position on the vendor remote with the emergency stop, and do nothing else.**

**中文** —— **下面这两件事由人来做，不是由你来做：**
1. **在厂商界面上让机器人站起来，并切到高层导航模式。** 这里没有任何软件能代劳——见第 3 条。
2. **手持厂商遥控器守住急停，全程只做这一件事。**

**EN** — Then, and only then, restart the driver with the two motion flags. **Motion is opted
into twice, on two different surfaces, at two different times, by design:**

**中文** —— 然后——也只有到这时——才带上那两个运动参数重启驱动进程。
**让机器人动起来需要两次独立授权，在两个不同的位置、两个不同的时刻给出，这是设计如此：**

```bash
python3.11 robot_driver.py --platform lite3 --package /tmp/package \
        --bridge-python <interpreter on the robot> \
        --model-sources /tmp/sources.json \
        --run-profile run-profile.example.json \
        --allow-motion --operator-ready
```

| gate | **EN** | **中文** |
| --- | --- | --- |
| 1 | `--allow-motion` at driver launch, typed by whoever has looked at the room | 启动驱动进程时的 `--allow-motion`，由**看过现场环境的人**亲手输入 |
| 2 | `arm_motion: true` in the request, by whoever presses the button | 请求里的 `arm_motion: true`，由**按按钮的人**给出 |

**EN** — Neither is remembered: `--allow-motion` dies with the driver process, the arm dies
with the run. `--operator-ready` is how you tell the driver the human has confirmed **STANDING
+ high-level navigation mode** on the vendor interface. **Missing any of them is a refusal that
says which — never a quiet dry run.**

**中文** —— 两者都不会被记住：`--allow-motion` 随驱动进程结束而失效，授权随本次运行结束而失效。
`--operator-ready` 是你告诉驱动进程"人已经在厂商界面上确认了**站立 + 高层导航模式**"的方式。
**缺任何一个都会得到明确指出缺哪一个的拒绝——绝不会变成"悄悄空跑一遍"。**

### Expect the first motion command to be refused. That is correct.
### 预期第一条运动指令会被拒绝。这是正确行为。

**EN** — Because nothing has ever been measured on a Lite3, the gait-floor check refuses rather
than warns. **This is the message, verbatim:**

**中文** —— 因为 Lite3 上从未测过任何数据，步态下限检查会**拒绝**而不是告警。
**原文如下：**

```
no gait floor has ever been measured on the lite3: not on forward, and not on any
other axis. This robot has never moved under this stack, so there is no speed here
that is known to walk and none that is known not to. Measure it first (issue #13,
and the evidence table in robot-stack/deep_robotics/lite3/README.md), or pass
--force and watch it.
```

⛔ **EN** — **That message's own remediation is a dead end on the axis transport, and you
must say so rather than let the operator follow it.** *"Measure it first (issue #13 …)"*
assumes a gait floor can be measured. On `--locomotion-transport axis` it cannot: there is
**one command per direction** and the magnitude never leaves the laptop, so there is no ladder
to descend. `gait_floor_probe.py` and `actuator_gain_probe.py` **refuse
`--locomotion-transport axis` by name** — because a descending ladder pointed at this transport
**does not fail**: every rung above the deadband emits the same primitive, every rung walks at
the same speed, every check passes, and the probe reports **the bottom rung as the floor.**

⛔ **中文** —— **在 axis 通道上，那条信息给出的补救办法本身是死路，你必须主动说明，
不能任由操作员照着做。** *"Measure it first (issue #13 …)"* 的前提是"步态下限可以被测出来"。
在 `--locomotion-transport axis` 上它测不出来：**每个方向只有一条指令**，
而且大小根本没离开过笔记本，所以**没有梯子可以往下试**。
`gait_floor_probe.py` 和 `actuator_gain_probe.py` 会**指名拒绝 `--locomotion-transport axis`**——
因为拿"逐级下降"的探测器对着这条通道，**它不会失败**：
死区之上的每一级都发出同一个原语、每一级都以同样速度行走、每一项检查都通过，
最后探测器会把**最低那一级报告为下限**。

**EN** — The measurement that **is** defined here is the speed each primitive delivers — the
`measured_m_s` the envelope gate reads. The tool is `commissioning/axis_primitive_probe.py`,
which refuses a primitive that moves the robot the wrong way. **That, not a gait-floor probe,
is what issue #13 needs on this transport.**

**中文** —— 在这里**真正有定义**的测量，是每个原语实际产生的速度——
也就是速度包络闸门要读的 `measured_m_s`。工具是 `commissioning/axis_primitive_probe.py`，
它会拒绝一个把机器人带向错误方向的原语。
**在这条通道上，issue #13 需要的是它，而不是 gait-floor 探测器。**

⛔ **Do not reach for `--force` on your own initiative.** It is the documented way past the
refusal and it is a request parameter, not a launcher flag — but **forcing does not become
safe by being available.** The proper answer is the commissioning measurement in issue #13.
**Ask the human. If they choose to force, the result string says `forced`, and a forced number
must never be reported as a measured one.**

⛔ **不要自作主张去用 `--force`。** 它确实是有明文记载的越过方式，而且它是**请求参数**、
不是启动脚本的参数——但**"可用"不等于"安全"**。正确的做法是按 issue #13 做标定测量。
**请询问操作员。如果他决定强制执行，返回结果里会带 `forced` 字样；
强制得到的数字绝不可以当作实测数字上报。**

### What manual control will feel like
### 人工操控的实际手感

⚠️ **The motion pad is an 8-way d-pad, not a joystick** — caveat 4. The commanded **magnitude
is discarded** and the direction is **snapped to one of 8 octants**. Tell the operator before
he presses, or he will report it as a bug.

⚠️ **运动面板是八向十字键，不是摇杆**——见第 4 条。指令的**大小被丢弃**，
方向被**归一到 8 个八分方向之一**。请在他按下之前就告诉他，否则他会把它当缺陷上报。

⛔ **`stop` on a Lite3 only stops. It does not lay the robot down** — caveat 3. Bringing the
robot back down is done by the human on the vendor app.

⛔ **Lite3 上的 `stop` 只是停住，不会让机器人趴下**——见第 3 条。
让机器人趴下是由人在厂商 App 上完成的。

**CHECK 6** — a single, short, deliberate command; the robot does what the direction (not the
magnitude) asked; **stop** and `STOP ALL` both end it. **If the robot accepts every command,
reports no error and never steps, do not raise the speed and do not retry — check the vendor
interface mode first.** That failure mode looks identical to a flat battery and to a
sub-gait-floor command, and telling them apart by trial is how an afternoon disappears.

**CHECK 6** —— 一条**单独的、短促的、有意为之的**指令；机器人按**方向**（而不是按大小）动作；
**stop** 和 `STOP ALL` 都能结束它。
**如果机器人接受了每一条指令、不报任何错、却始终不迈步，不要加大速度，也不要反复重试——
请先检查厂商界面上的模式。** 这种失效表现和"电量耗尽"以及"指令低于步态下限"
**看起来一模一样**，靠试错去区分它们，一个下午就没了。

---

## When you finish, or when you stop
## 完成时，或中途停下时

**EN** — Write a continuation comment on the GitHub issue: what ran, the outcome numbers, what
failed with its exact output, and what is still open. **Do this even when stopping mid-task —
the next session is a fresh context and this comment is the only handover.** Report failures
plainly; "it should work" is not a status. Return **absolute filepaths** for any recording
produced, so they can be opened. Quote the `commit … tree …` line a run prints as its first
line rather than saying "deployed from main".

**中文** —— 在对应的 GitHub issue 上写一条接续评论：跑了什么、结果数字是多少、
哪里失败了并附上**确切输出**、还有什么没做完。
**即使是中途停下也要写——下一次工作是全新的上下文，这条评论是唯一的交接材料。**
如实上报失败，"应该没问题"不算状态。产出的任何录制文件都要给出**绝对路径**，以便打开。
引用运行打印出的第一行 `commit … tree …`，而不要笼统地说"部署自 main"。

─── PROMPT ENDS ─── 提示词到此结束 ───
