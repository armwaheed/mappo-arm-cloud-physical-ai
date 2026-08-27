<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Device Connect dashboard — operator guide
# Device Connect dashboard —— 操作手册

**EN** — This is the operator's document for the dashboard in this directory. It assumes you
did not write the code and cannot ask the person who did, because they are eight time zones
away and asleep. It covers **seven things**, in the order you will do them, and it says for
each one whether it has been proven on hardware or not.

**中文** —— 这是本目录下 dashboard 的操作文档。本文假设你没有参与写这些代码，而且无法当场
找到作者——他们在八个时区以外，现在是深夜。全文覆盖**七件事**，按你实际操作的顺序排列；
每一件都会说明它**是否已经在真机上验证过**。

**EN** — The full English reference is [`README.md`](README.md) beside this file: 933 lines,
and it is the authority. This guide is shorter on purpose. Where the two disagree, the
README is right and this file is a bug — say so in an issue.

**中文** —— 完整的英文参考文档是与本文件同目录的 [`README.md`](README.md)，共 933 行，
**以它为准**。本手册是有意精简的。若两者不一致，以 README 为准，并请提 issue 说明本文件的问题。

> **EN** — Commands, flags, filenames and the text the programs print are left in English
> throughout, and must be typed and matched **exactly**. Only the explanation is translated.
>
> **中文** —— 全文中的命令、参数、文件名以及程序打印出来的文字一律保留英文原文，
> 必须**逐字**输入和比对。只有说明文字是中文。

**EN** — If you are bringing up a **Lite3 for the first time**, do not start here. Start with
[`../robot-stack/deep_robotics/lite3/LITE3-DASHBOARD-BRINGUP-PROMPT.md`](../robot-stack/deep_robotics/lite3/LITE3-DASHBOARD-BRINGUP-PROMPT.md),
which is a staged bring-up written to be handed to a coding agent, and which stops at each
stage if the check fails. Come back here for the reference.

**中文** —— 如果你是**第一次给 Lite3 做上电联调**，请不要从这里开始。请先看
[`../robot-stack/deep_robotics/lite3/LITE3-DASHBOARD-BRINGUP-PROMPT.md`](../robot-stack/deep_robotics/lite3/LITE3-DASHBOARD-BRINGUP-PROMPT.md)。
那份文档是分阶段的联调流程，写给编码 agent 使用，**每一阶段验证不通过就停下**。
本手册作为参考随时回来查。

---

## 0. Terminology — please confirm these renderings
## 0. 术语表 —— 这些译法请确认

**EN** — Written plainly rather than in polished domain jargon. If the team has a house term
for any of them, correct it here and say so in an issue; do not let a wrong term propagate.

**中文** —— 为避免误译，这里用的是**直白说法**，不是行业习惯用语。如果团队已有习惯译法，
请在这里改正并提 issue 说明；不要让错误的术语传播出去。

| English (use verbatim in commands) | 中文（本手册用法） | 需确认 |
| --- | --- | --- |
| the mesh / `MESH UP` | 设备网：Device Connect 的设备发现网络；`MESH UP` 表示已连上 | ✅ 请确认 |
| driver / `robot_driver.py` | 驱动进程：代表一台机器人接入设备网的那个进程（**跑在工作站上，不在机器人上**） | ✅ 请确认 |
| bench double / `--platform sim` | 台架替身：完全不涉及真机的模拟后端，用于无硬件演示 | ✅ 请确认 |
| checkpoint / model | 模型权重文件（`.npz`） | — |
| arm (a checkpoint) | 挂载：把某个权重指定为**下一次**运行要用的那个 | ✅ 请确认 |
| scene run | 场景运行：只跑感知与策略、**不发任何腿部指令**的一次运行 | ✅ 请确认 |
| motion run | 运动运行：真的会让腿动起来的一次运行 | ✅ 请确认 |
| gait floor | 步态下限：能让机器人真正走起来的**最低**速度指令 | ✅ 请确认 |
| refuse / refusal | 拒绝执行：程序主动停下并说明原因，不是报错崩溃 | — |

---

## 1. Safety — read this before anything is powered
## 1. 安全须知 —— 上电之前必须先读

⛔ **EN** — [`../robot-stack/SAFETY.md`](../robot-stack/SAFETY.md) governs anything here that
moves a leg, and it is not optional. In this directory `--allow-motion` is what `--live` is
elsewhere.

⛔ **中文** —— [`../robot-stack/SAFETY.md`](../robot-stack/SAFETY.md) 管辖这里一切会让腿动
起来的操作，不可跳过。在本目录中，`--allow-motion` 就相当于别处的 `--live`。

**EN** — **Motion is opted into twice, on two different surfaces, at two different times.**
This is the design, not an obstacle to route around:

**中文** —— **让机器人动起来需要两次独立授权**，在两个不同的位置、两个不同的时刻给出。
这是设计如此，**不是可以绕开的障碍**：

| gate | **EN** | **中文** |
| --- | --- | --- |
| 1 | `--allow-motion` on the **driver command line**, typed by whoever has looked at the room. `start-dashboard.sh` will never add it for you. | 在**驱动进程的命令行**上加 `--allow-motion`，由**看过现场环境的人**亲手输入。`start-dashboard.sh` 永远不会替你加上。 |
| 2 | `arm_motion: true` in the **request**, by whoever presses the button. | 在**这一次请求**里给出 `arm_motion: true`，由按按钮的人给出。 |

**EN** — Neither is remembered: `--allow-motion` dies with the driver process and the arm
dies with the run. Missing either is **a refusal that says which** — never a quiet dry run.
The default **Start** press builds a command line with no `--live` in it at all, so it is an
**absent capability, not a checked permission**. That is what makes it safe to press first,
every time.

**中文** —— 两者都不会被记住：`--allow-motion` 随驱动进程结束而失效，授权随本次运行结束而
失效。缺任何一个都会得到**明确指出缺哪一个的拒绝**，而**绝不会**变成"悄悄空跑一遍"。
默认的 **Start** 按下去时，生成的命令行里**根本没有 `--live`**——也就是说它**不是"权限被
拒绝"，而是"这个能力根本不存在"**。正因如此，**每次都应该先按它**。

⛔ **EN** — **Always do a scene run before any motion run**, with a named person holding the
vendor remote and its emergency stop, doing nothing else for the whole run.

⛔ **中文** —— **任何运动运行之前，必须先做一次场景运行**；并且指定一个人全程手持厂商遥控器
和急停，**全程只做这一件事**。

---

## 2. Before you start — what has to be installed, and where
## 2. 开始之前 —— 需要装什么，装在哪里

⛔ **EN** — **The robot cannot host the driver, and that is architecture rather than a
packaging bug.** `robot_driver.py` runs on a **workstation** (your laptop), Python **≥ 3.11**.
Do not try to make it run on the robot, and **never install packages on a robot outside a
virtualenv** — see [`../AGENTS.md`](../AGENTS.md).

⛔ **中文** —— **机器人上跑不了这个驱动进程，这是架构决定的，不是打包问题。**
`robot_driver.py` 跑在**工作站**（你的笔记本）上，需要 Python **≥ 3.11**。
不要试图把它挪到机器人上跑；**更不要在机器人上于虚拟环境之外安装任何 Python 包**——
见 [`../AGENTS.md`](../AGENTS.md)。

```bash
python3.11 -m pip install device-connect-edge device-connect-agent-tools aiohttp numpy Pillow
```

⚠️ **EN** — `aiohttp` and `numpy` are **not optional and nothing else installs them**. In a
clean 3.11 venv: `server.py` dies at `No module named 'aiohttp'`, and both `robot_driver.py`
and `model_server.py` die at `numpy`. `Pillow` is needed only by the synthetic sim camera.

⚠️ **中文** —— `aiohttp` 和 `numpy` **不是可选项，也没有别的包会顺带装上它们**。
在干净的 3.11 虚拟环境中实测：`server.py` 会以 `No module named 'aiohttp'` 退出，
`robot_driver.py` 和 `model_server.py` 都会以 `numpy` 缺失退出。
`Pillow` 只有模拟相机才需要。

⚠️ **EN** — On macOS, `python3` is the Command Line Tools 3.9.6 however many Homebrew Pythons
are installed. **`python3.11` above is literal**, not a placeholder.

⚠️ **中文** —— 在 macOS 上，无论你装了多少个 Homebrew Python，`python3` 都是系统自带的
3.9.6。上面的 **`python3.11` 就是字面意思**，不是占位符。

---

## 3. The seven things, and which of them are proven
## 3. 七件事，以及各自的验证状态

**EN** — Read this table before anything else. The right-hand column is the honest state as
of this file's commit, and the ❌ entries are not pessimism — they are things nobody has done
yet on a real Lite3.

**中文** —— 请先读这张表。最右列是本文件提交时的**真实状态**；❌ 不是保守估计，
而是**在真机 Lite3 上确实还没有人做过**。

| | **EN** | **中文** | proven / 验证状态 |
| --- | --- | --- | --- |
| 1 | **Open the dashboard** | **打开 dashboard** | ✅ verified |
| 2 | **View the fleet** | **查看机器人列表** | ✅ verified (Go2 camera, 1920×1080 @ 6 fps) |
| 3 | **Connect to a model server** | **连接模型服务** | ✅ verified |
| 4 | **Load a model** | **下载模型到机器人** | ✅ verified on Go2 — ⚠️ see §7 for what it really touches |
| 5 | **Start MAPPO on the robot** | **在机器人上启动 MAPPO** | ✅ RPCs verified against the bench double — ❌ **no robot has been driven by them**; ⛔ **on a Lite3, blocked — §10.5** |
| 6 | **Stop / start / manual control** | **停止／启动／人工接管** | ✅ verified on the bench double — ❌ **never on a real robot**; ⛔ **on a Lite3, blocked — §10.5** |
| 7 | **Swap models** | **切换模型** | ✅ verified — takes effect on the **next** run |

---

## 4. Thing 1 and 2 — open the dashboard, and view the fleet
## 4. 第 1、2 件 —— 打开 dashboard，查看机器人列表

**EN** — One command, from `dashboard/`, **on a workstation — not on the robot**:

**中文** —— 一条命令，在 `dashboard/` 目录下、**在工作站上执行——不是在机器人上**：

```bash
./start-dashboard.sh                              # a bench double. No robot needed.
./start-dashboard.sh --robot 192.168.123.18       # a real Go2's camera, simulated pose
```

**EN** — It starts the checkpoint server, one driver and the dashboard, waits until the page
answers, prints the URL, and **stops all three together** on Ctrl-C. `--dry-run` prints the
three commands it would run without running them — use it the first time.

**中文** —— 它会启动模型服务、一个驱动进程和 dashboard 三个进程，等页面能响应后打印出网址；
按 Ctrl-C 时**三个一起停**。`--dry-run` 只打印将要执行的三条命令而不真的执行——**第一次请先用它**。

⚠️ **EN** — **`--platform lite3` works on the launcher, but the flags a Lite3 needs are not
launcher flags.** `--robot HOST` is a Go2 shorthand — it expands to
`--platform go2 --simulate --camera-url http://HOST:8801/` — so do not use it for a Lite3.
`--bridge-python`, `--run-profile` and `--operator-ready` belong to `robot_driver.py` and
reach it only through the `--` pass-through:

⚠️ **中文** —— **`--platform lite3` 在启动脚本上是支持的，但 Lite3 需要的那几个参数不是启动
脚本自己的参数。** `--robot HOST` 是 Go2 的简写——它展开后是
`--platform go2 --simulate --camera-url http://HOST:8801/`——所以 **Lite3 不要用它**。
`--bridge-python`、`--run-profile` 和 `--operator-ready` 属于 `robot_driver.py`，
只能通过 `--` 透传过去：

```bash
./start-dashboard.sh --platform lite3 --host <LAN address> \
        -- --bridge-python <interpreter on the robot> \
           --run-profile run-profile.example.json
```

**EN** — Everything after `--` goes to `robot_driver.py` unchanged. **`--` cannot smuggle an
`--allow-motion` past the launcher's one invariant either** — the driver's gate reads the
driver's own flag, and `--allow-motion` is added *only* when a person types it literally on
the launcher's own command line. Starting the three by hand (§5) is the other way, and it is
the clearer one the first time.

**中文** —— `--` 之后的内容会原样传给 `robot_driver.py`。**`--` 也无法借道绕开启动脚本
唯一的那条铁律**——驱动进程的闸门读的是它自己的参数，而 `--allow-motion` **只有**在有人
把它**字面地**敲在启动脚本自己的命令行上时才会被加上。另一种方式是**手工分别启动三个进程**
（见 §5）；第一次做的时候，那种方式更清楚。

**EN** — **A working screen** has `MESH UP` in the top bar, and the robot on a fleet row with
a green `LIVE` badge and a pose. `STOP ALL (n)` counts what it will hit. Press `E` for the
event drawer.

**中文** —— **页面正常的标志**：顶栏显示 `MESH UP`；机器人出现在列表中，带绿色 `LIVE` 标记
并显示位姿。`STOP ALL (n)` 中的 n 是它将要停掉的对象数。按 `E` 打开事件抽屉。

⚠️ **EN** — **This dashboard has no login.** The default bind is loopback. `--host 0.0.0.0`
reaches it from the demo LAN, and means anyone who can reach the port can drive any robot on
the mesh that was started with motion enabled.

⚠️ **中文** —— **本 dashboard 没有任何登录认证。** 默认只绑定本机回环地址。
用 `--host 0.0.0.0` 可以从演示局域网访问，但这意味着**任何能访问该端口的人**都能操控
设备网上所有已开启运动权限的机器人。

---

## 5. Starting the three by hand — the only way to get a Lite3
## 5. 手工启动三个进程 —— Lite3 只能这样起

**EN** — `start-dashboard.sh` runs exactly these three and there is no fourth thing it does.
Every command is Python **≥ 3.11**, on a workstation, from `dashboard/`.

**中文** —— `start-dashboard.sh` 做的就是下面这三件事，没有第四件。
每条命令都用 Python **≥ 3.11**，在工作站上、在 `dashboard/` 目录下执行。

```bash
# 1. the checkpoint source. It WRITES sources.json.
python3.11 model_server.py --models-dir ../policy/models --port 8800 \
        --host 192.168.1.50 \
        --emit-sources /tmp/sources.json --label "Shanghai model server"

# 2. the robot driver — a real Lite3
python3.11 robot_driver.py --platform lite3 --package ../policy \
        --bridge-python <interpreter on the robot that imports the Lite3 SDK> \
        --model-sources /tmp/sources.json \
        --run-profile run-profile.example.json

# 3. the dashboard
python3.11 server.py --port 8080                  # then open http://127.0.0.1:8080
```

⚠️ **EN** — **`--host 127.0.0.1` on the model server is unreachable from a robot**, and the
server says so at startup. **The robot does the fetching**, not your browser, so a loopback
address gives a field that looks right and fails on fetch. Bind a LAN address.

⚠️ **中文** —— **模型服务绑定 `127.0.0.1` 时机器人访问不到**，服务启动时会明确提示。
**下载动作是机器人执行的**，不是浏览器，所以填回环地址会出现"看起来填对了、一下载就失败"。
请绑定一个局域网地址。

---

## 6. Thing 3 — connect to a model server
## 6. 第 3 件 —— 连接模型服务

**EN** — You do not type an address into the browser. `model_server.py --emit-sources` writes
a file; `robot_driver.py --model-sources` reads it; the **robot** advertises it; the dashboard
fills the picker in from what the robot advertised. The address is written **once**.

**中文** —— 你不需要在浏览器里输入地址。`model_server.py --emit-sources` 写出一个文件，
`robot_driver.py --model-sources` 读取它，由**机器人**把它广播出来，dashboard 再据此填好
下拉框。地址**只写一次**。

**EN** — Open **Load from Cloud AI**. The Source picker should already read your `--label`
and the address field should already be filled in. Press **Browse**.

**中文** —— 打开 **Load from Cloud AI**。Source 下拉框里应该已经是你传的 `--label`，
地址栏应该已经填好。点 **Browse**。

✅ **EN** — A row saying `served by mappo-model-server` is the proof that **the robot**, not
your browser, can reach the source. That round trip is the whole check.

✅ **中文** —— 出现 `served by mappo-model-server` 的条目，就证明**机器人**（而不是你的
浏览器）能访问到模型服务。这一次往返就是全部的验证。

---

## 7. Thing 4 and 7 — load a model, and swap models
## 7. 第 4、7 件 —— 下载模型，切换模型

**EN** — **Download** on a browsed row fetches the checkpoint. **Arm** on a checkpoint row
selects it for the next run.

**中文** —— 在浏览到的条目上点 **Download** 下载权重文件。在权重条目上点 **Arm**
把它指定为下一次运行要用的模型。

**EN** — **A swap takes effect on the next run, not on the one in progress.**
`MappoController` loads its weights once, at construction, so a live run cannot have the
network pulled out from under it — not because anything checks, but because **no code path
exists** to do it.

**中文** —— **切换模型在下一次运行时生效，不会影响正在进行的运行。**
`MappoController` 只在构造时加载一次权重，所以正在跑的运行不可能被换掉模型——
这不是因为有什么检查拦着，而是因为**根本没有这条代码路径**。

⚠️ **EN** — **On a real robot reached through an SSH wrapper, the checkpoint panel is
answering about the wrong machine.** `list_models`, `free_bytes` and `download_model` act on
the **workstation's** `--package` directory, not the robot's. Only `get_status`,
`get_capabilities` and the motion gate genuinely reach the robot. This is a known limitation
of the SSH-wrapper workaround, not a supported deployment —
[`README.md`](README.md) §"On a real robot, without the bench double".

⚠️ **中文** —— **在通过 SSH wrapper 连接真机时，模型面板显示的是另一台机器的情况。**
`list_models`、`free_bytes` 和 `download_model` 操作的是**工作站上**的 `--package` 目录，
不是机器人上的。只有 `get_status`、`get_capabilities` 和运动闸门是真正到达机器人的。
这是 SSH wrapper 这个临时方案的已知限制，**它并不是受支持的部署方式**——
详见 [`README.md`](README.md) 的 "On a real robot, without the bench double" 一节。

⚠️ **EN** — Use a **copy** of `../policy` as `--package`: arming a checkpoint **rewrites its
`config.json`**.

⚠️ **中文** —— `--package` 请指向 `../policy` 的**副本**：挂载模型会**改写其中的
`config.json`**。

---

## 8. Thing 5 — start MAPPO on the robot
## 8. 第 5 件 —— 在机器人上启动 MAPPO

**EN** — **The default press cannot move the robot.** `start_run()` with no arguments builds
the camera, the detector, the policy, the planner's veto and the telemetry, with **no
`--live`**. It is the right thing to press first, every time.

**中文** —— **默认按下去不会让机器人动。** 不带参数的 `start_run()` 会启动相机、检测器、
策略、规划器的否决逻辑和遥测，但**不带 `--live`**。**每次都应该先按它。**

**EN** — To be able to start a run at all, the driver needs `--run-profile`. Without it,
`start_run` refuses and `get_capabilities` reports `run.supported: false`.

**中文** —— 要能启动运行，驱动进程必须带 `--run-profile`。没有它，`start_run` 会拒绝执行，
并且 `get_capabilities` 会报告 `run.supported: false`。

⚠️ **EN** — **Edit `run-profile.example.json` first. Every path in it is a path on the
robot**, `launch_prefix` must authenticate **without a password** (put a key on the robot — a
password is refused at load, because that field is published), and `heading_servo_flag` is a
property of the tree it points at rather than a preference. The driver prints both commands
it would run at startup, which is the moment a wrong path is still cheap.

⚠️ **中文** —— **先改 `run-profile.example.json`。里面每一个路径都是机器人上的路径**；
`launch_prefix` 必须能**免密**认证（在机器人上放公钥——**写密码会在加载时被拒绝**，
因为这个字段会被广播出去）；`heading_servo_flag` 取决于它所指向的那份代码树，不是个人偏好。
驱动进程启动时会把它将要执行的两条命令都打印出来——**那是发现路径写错代价最小的时刻**。

### Recording a run — and why `--record` alone is useless as training data
### 录制运行 —— 以及为什么只用 `--record` 录的视频不能当训练数据

⛔ **EN** — `--record` writes the **annotated** frame: the HUD, the plan-view inset and **a
box around every detection**. That is what makes it readable, and **what makes it useless as
training data — the label is burned into the pixels the model would have to learn from.**
`--record-raw` writes the same frame **before** any of that is drawn on it.

⛔ **中文** —— `--record` 写出的是**带标注**的画面：HUD、俯视小地图，以及**每个检测目标外面
的框**。这让画面便于人看，但也**使它完全不能用作训练数据——标签被烧进了模型本该去学习的那些
像素里**。`--record-raw` 写出的是**在画任何标注之前**的同一帧。

**EN** — So **always pass both**, and keep the telemetry `.jsonl` from the same run:

**中文** —— 所以**两个都要给**，并且保留同一次运行产生的遥测 `.jsonl` 文件：

```bash
--record run.mp4  --record-raw run-raw.mp4  --telemetry run.jsonl
```

**EN** — Both recorders are advanced by **one** gate, so frame *n* of one file is frame *n*
of the other, and both join to the same `perception.video_frame` in the telemetry. That join
is what turns the detector's own boxes into **labels for the clean pixels**.

**中文** —— 两个录制器由**同一个**节拍推进，所以一个文件的第 *n* 帧就是另一个文件的第 *n* 帧，
并且都通过遥测里的 `perception.video_frame` 对应起来。正是这个对应关系，
让检测器自己输出的框成为**干净像素的标签**。

⚠️ **EN** — **Recording costs control-loop rate.** Measured on a Go2: **246.4 ms** per
new-result tick with `--record`, against **100.6 ms** without — issue #18, and the recorder
is still synchronous on the perception cycle. **A run recorded for training is not a run
whose timing numbers mean anything.** If you want both, record a separate pass.

⚠️ **中文** —— **录制会拖慢控制环频率。** 在 Go2 上实测：开 `--record` 时每个新结果周期
**246.4 ms**，不开时 **100.6 ms**——见 issue #18，而且录制目前仍与感知周期同步执行。
**为采训练数据而录的那次运行，其时间性能数字没有参考价值。** 两者都要的话，请分两次跑。

---

## 9. Thing 6 — stop, start, and take manual control
## 9. 第 6 件 —— 停止、启动、人工接管

| **EN** | **中文** |
| --- | --- |
| per-row **stop** | 每一行上的 **stop** 按钮 |
| **`STOP ALL (n)`** | 全部停止；n 是它将要停掉的对象数 |
| the **motion pad** — needs `--allow-motion` | **运动控制面板**——需要 `--allow-motion` |

**EN** — `stop` and `STOP ALL` end a **running policy** as well as a nudge, and `stop_run` is
how a person takes the robot back mid-run. A delivered stop is not enough on its own — the
worker refreshes velocity at 10 Hz, so `stop` also **terminates the in-flight worker**
(SIGTERM, so its `SafeStop` damps).

**中文** —— `stop` 和 `STOP ALL` 不只停单次点动，也会**结束正在运行的策略**；
`stop_run` 就是人在运行中途把机器人接管回来的方式。
仅仅把 stop 送达是不够的——执行体以 10 Hz 刷新速度指令，所以 `stop`
还会**终止正在执行的工作进程**（用 SIGTERM，让它的 `SafeStop` 完成阻尼停止）。

⛔ **EN** — **On a Lite3, stop only stops. It does not lay the robot down.** `lie_down` on a
Go2 issues `StandDown`; on a Lite3 posture is **operator-controlled through the vendor app**.
The driver reports this itself as `lie_down_changes_posture: false`. **Standing the robot up
and putting it into high-level navigation mode is your job, on the vendor interface**, and
`--operator-ready` is how you tell the driver you have done it.

⛔ **中文** —— **在 Lite3 上，stop 只是停住，不会让机器人趴下。**
`lie_down` 在 Go2 上会下发 `StandDown`；在 Lite3 上，**姿态由操作员通过厂商 App 控制**。
驱动进程自己会报告 `lie_down_changes_posture: false`。
**让机器人站起来、并切到高层导航模式，是你在厂商界面上要做的事**；
`--operator-ready` 就是你告诉驱动进程"我已经做完了"的方式。

---

## 10. What the Lite3 will not do the way you expect
## 10. Lite3 上与预期不符的几件事

**EN** — Every item here is **current behaviour verified against the code**, not a rumour and
not a bug report. Read them before you run, or you will spend the afternoon reporting them.

**中文** —— 下面每一条都是**对照代码核实过的当前行为**，既不是传闻也不是缺陷报告。
**请在开跑之前读完**，否则你会花一下午去上报它们。

### 10.1 Nothing has been measured on a Lite3, on any axis
### 10.1 Lite3 上没有任何一个轴被测量过

⛔ **EN** — Neither Venture has moved under this stack. **Every motion command to a Lite3 is
refused** rather than warned about, because there is no measured floor to check it against.
**You are the first person to run this path.** This is the message you will get, verbatim:

⛔ **中文** —— 两台 Venture 都还没有在本套软件下动过。
**发给 Lite3 的每一条运动指令都会被拒绝**（而不是仅仅告警），因为没有任何实测下限可供比对。
**你是第一个跑这条路径的人。** 你会看到的原文如下：

```
no gait floor has ever been measured on the lite3: not on forward, and not on any
other axis. This robot has never moved under this stack, so there is no speed here
that is known to walk and none that is known not to. Measure it first (issue #13,
and the evidence table in robot-stack/deep_robotics/lite3/README.md), or pass
--force and watch it.
```

**EN** — **That is the expected result, not a fault.** `force` is the documented way past it:
it is a parameter on the motion request (`--force` on `drive_bridge.py`), **not** a flag you
put on the launcher. Forcing does not become safe by being available — the result string says
`forced` precisely so that a forced number can never be mistaken for a measured one.

**中文** —— **这是预期结果，不是故障。** `force` 是有明文记载的越过方式：
它是**运动请求上的一个参数**（在 `drive_bridge.py` 上是 `--force`），
**不是**你加在启动脚本上的参数。可用不等于安全——
返回结果里会带上 `forced` 字样，**就是为了让强制得到的数字永远不会被当成实测数字**。

⚠️ **EN** — The Lite3's own navigator already fails closed the same way:
`lite3/visual_nav/robot_bindings.py` answers a live run with no `--gait-floor` with
**`REFUSING TO WALK: missing`**. A dashboard button is not a weaker authority than that, so
it does not get a weaker rule.

⚠️ **中文** —— Lite3 自己的导航程序本来就是同样的"失败即停"逻辑：
`lite3/visual_nav/robot_bindings.py` 对一次没有给 `--gait-floor` 的实机运行，
回答的是 **`REFUSING TO WALK: missing`**。
dashboard 上的按钮并不比它更有权威，所以也不会得到更宽松的规则。

### 10.2 The gait-floor guard cannot protect you the way it protects a Go2
### 10.2 步态下限保护在 Lite3 上起不到 Go2 那样的作用

**EN** — The Go2 has **two** measured floors and they differ by a factor of **1.75**:
**0.35 m/s forward** and **0.20 m/s lateral** (issue #42's table; the issue itself says
"nearly 2x"). Those are **Go2** numbers — issue #42 says so in its own body. Meanwhile
`--gait-floor` is **one** field, applied to the forward axis. On the Go2 the conflation
happens to be safe by coincidence; on a Lite3 there is nothing to conflate yet, because the
table is empty on all three axes.

**中文** —— Go2 有**两个**实测下限，相差 **1.75 倍**：**前进 0.35 m/s**、**横移 0.20 m/s**
（见 issue #42 的表格；该 issue 自己的说法是"接近 2 倍"）。这些是 **Go2** 的数据——
issue #42 正文里明确这样写。而 `--gait-floor` 只有**一个**字段，且作用在前进轴上。
在 Go2 上，把两者混为一谈**恰好**是安全的；在 Lite3 上则连可混淆的数据都还没有，
三个轴的表格都是空的。

### 10.2b The refusal says "measure it first" — on the axis transport you cannot
### 10.2b 拒绝信息叫你"先去测" —— 在 axis 通道上你测不了

⛔ **EN** — The refusal above ends with *"Measure it first (issue #13 …)"*. **On the
`--locomotion-transport axis` path that instruction is a dead end, and following it will cost
you a day.** A gait floor is the lowest *commanded* speed that still walks — but here there is
**one command per direction** and the magnitude never leaves the laptop, so there is no ladder
to descend. `gait_floor_probe.py` and `actuator_gain_probe.py` **refuse
`--locomotion-transport axis` by name**, and they are right to: pointing a descending ladder at
this transport **does not fail** — every rung above the deadband emits the same primitive,
every rung walks at the same speed, every check passes, and the probe reports **the bottom rung
as the floor**.

⛔ **中文** —— 上面那段拒绝信息的结尾是 *"Measure it first (issue #13 …)"*。
**在 `--locomotion-transport axis` 这条通道上，这条指示是死路，照做会浪费你一整天。**
步态下限指的是"仍能走起来的**最低指令速度**"——但在这里**每个方向只有一条指令**，
而且大小根本没离开过笔记本，所以**没有梯子可以往下试**。
`gait_floor_probe.py` 和 `actuator_gain_probe.py` 会**指名拒绝 `--locomotion-transport axis`**，
而且这样做是对的：拿一个"逐级下降"的探测器对着这条通道，**它不会失败**——
死区之上的每一级都发出同一个原语、每一级都以同样速度行走、每一项检查都通过，
最后探测器会把**最低那一级报告为下限**。

**EN** — What *is* measurable here is **the speed each primitive delivers** — the
`measured_m_s` the envelope gate reads. The tool is
[`../robot-stack/deep_robotics/lite3/commissioning/axis_primitive_probe.py`](../robot-stack/deep_robotics/lite3/commissioning/axis_primitive_probe.py),
and it refuses a primitive that moves the robot the wrong way.

**中文** —— 在这里**真正可测的**是**每个原语实际产生的速度**——
也就是速度包络闸门要读的 `measured_m_s`。工具是
[`../robot-stack/deep_robotics/lite3/commissioning/axis_primitive_probe.py`](../robot-stack/deep_robotics/lite3/commissioning/axis_primitive_probe.py)，
并且它会拒绝一个把机器人带向错误方向的原语。

### 10.3 Steering will not feel proportional — it is an 8-way d-pad
### 10.3 转向不是连续的——它更像八向十字键

⚠️ **EN** — The Lite3 axis mapping is **sign-only: the commanded magnitude is discarded.** A
profile holds one evidenced raw value per direction, so every command past the deadband
leaves at whatever speed that one primitive was measured to produce. The linear pair is then
**snapped to the nearest of eight directions**. **The motion pad will feel like an 8-way
d-pad, not a joystick. That is the transport, not a fault.**

⚠️ **中文** —— Lite3 的轴映射**只取符号：指令的大小被丢弃**。
每个方向的配置里只有一个有实测依据的原始值，所以只要超过死区，
机器人就以那一个原语被实测出来的速度运动。
随后，平移的两个分量会被**归一到最近的八个方向之一**。
**运动面板用起来会像八向十字键，而不是摇杆。这是传输层的设计，不是故障。**

⚠️ **EN** — A consequence worth knowing: nothing here scales with `vx`, which means
**`--derate` and `--max-vx` do not reach the wire on this transport.** Every setting emits the
same raw axis value:

⚠️ **中文** —— 一个值得知道的后果：这里没有任何东西随 `vx` 缩放，也就意味着
**在这条通道上 `--derate` 和 `--max-vx` 根本到不了总线。** 每一种设置发出的原始轴值都相同：

| `--derate` | commanded `vx` | forward axis emitted |
| ---: | ---: | ---: |
| 1.0 | 0.300 m/s | `+32767` |
| 0.6 | 0.180 m/s | `+32767` |
| 0.3 | 0.090 m/s | `+32767` |
| 0.2 | 0.060 m/s | `+32767` |

**EN** — That is the **deliberate** half of the design: this transport will not invent a raw
value it has no physical evidence for, and interpolating between an evidenced `+32767` and an
unevidenced zero is inventing one. The envelope is enforced at preflight instead — `--live`
**refuses the run** when a declared `measured_m_s` exceeds `--max-vx × --derate`.

**中文** —— 这是设计中**有意为之**的一半：这条通道不会去编造一个没有物理依据的原始值，
而在"有依据的 `+32767`"和"没有依据的 0"之间做插值，就是在编造。
速度包络改为在预检阶段强制执行——当声明的 `measured_m_s` 超过 `--max-vx × --derate` 时，
`--live` 会**拒绝这次运行**。

### 10.4 The camera is exclusive — you cannot watch the feed during a run
### 10.4 相机是独占的 —— 运行期间看不到实时画面

⛔ **EN** — The Lite3's frames come from an OpenCV `VideoCapture`, which on Linux is typically
**exclusive**. So **while a `lite3_visual_nav` run holds the camera, the dashboard cannot open
it.** You will see **"the camera is in use"**, not a black rectangle. **You cannot watch the
live feed and run the policy at the same time.** This is not a bug — do not report it as one.

⛔ **中文** —— Lite3 的图像来自 OpenCV 的 `VideoCapture`，它在 Linux 上通常是**独占**的。
所以**当 `lite3_visual_nav` 正在运行并占用相机时，dashboard 就打不开它**。
你会看到 **"the camera is in use"**，而不是一个黑框。
**你不能一边看实时画面一边跑策略。这不是缺陷，请不要作为缺陷上报。**

**EN** — The Go2 does not have this problem: its frames come from the SDK's `VideoClient`, an
RPC to the robot's own video service, so a run and the viewport can both read it.

**中文** —— Go2 没有这个问题：它的图像来自 SDK 的 `VideoClient`，
那是对机器人自身视频服务的一次 RPC 调用，因此运行和画面窗口可以同时读取。

### 10.5 On a Lite3 the dashboard can only speak ROS 2 — and that is probably not this robot
### 10.5 dashboard 对 Lite3 只会说 ROS 2 —— 而这台机器人多半不是

⛔ **EN** — `drive_bridge.py`'s `_load_lite3` takes **no transport argument**. It constructs
`Lite3Locomotion`, whose default implementation imports
`arm_dc_robotkit.ros2_twist_locomotion`. **There is no `--locomotion-transport` anywhere in
`dashboard/`** — the `udp` and `axis` transports exist only behind `visual_nav`/`mappo_drive`.
The Lite3 README notes that the ROS 2 bridge "runs on a *perception* host … so it needs a
ROS 2 Foxy runtime on a computer **these two Ventures may not have**." **The one transport a
Venture has actually walked on — profile-gated axis UDP — the dashboard cannot reach at all.**

⛔ **中文** —— `drive_bridge.py` 的 `_load_lite3` **不接受任何通道参数**。
它构造 `Lite3Locomotion`，其默认实现 import 的是 `arm_dc_robotkit.ros2_twist_locomotion`。
**`dashboard/` 目录下根本没有 `--locomotion-transport`**——
`udp` 和 `axis` 两条通道只存在于 `visual_nav` / `mappo_drive` 背后。
Lite3 的 README 指出，ROS 2 桥"运行在**感知主机**上……因此需要一个 ROS 2 Foxy 运行时，
而**这两台 Venture 可能并没有**"。
**Venture 真正走起来过的那条通道——受配置约束的 axis UDP——dashboard 完全够不到。**

**EN** — Reproduced on a clean workstation. Note `"refused": false`: **the dashboard colours
this as a FAULT, not a refusal**, and it will send you to diagnose a robot that is fine.

**中文** —— 在干净的工作站上复现如下。注意 `"refused": false`：
**dashboard 会把它显示为"故障"而不是"拒绝执行"**，于是把你引去排查一台其实没问题的机器人。

```
$ python3.11 drive_bridge.py status --platform lite3
{"ok": false, "refused": false, "error": "ModuleNotFoundError: No module named 'ros2_twist_locomotion'"}
```

⚠️ **EN** — The same output comes back for `stop`, `stand`, `stand-down` and `pose-stream`, so
**the STOP button's velocity-zero backstop fails on a Lite3 too.** `stop` still SIGTERMs a
running policy, so a run does end — but the result reads red for an unrelated reason.

⚠️ **中文** —— `stop`、`stand`、`stand-down`、`pose-stream` 返回的是同样的输出，
所以 **STOP 按钮的"速度归零"兜底在 Lite3 上同样失效**。
`stop` 仍会向正在运行的策略发 SIGTERM，所以运行确实会结束——
但返回结果会因为一个无关的原因显示为红色。

### 10.6 The Lite3 camera panel can never produce a frame
### 10.6 Lite3 的相机面板永远出不了图

⛔ **EN** — Separate from §10.4, and worse. `Lite3CameraSource.read()`
(`camera_source.py:269-273`) returns `getattr(frame, "jpeg", None)`, and the Lite3 `Frame`
(`lite3/visual_nav/camera.py:28-35`) has fields `image`, `capture_time`, `seq`, `stamp` —
**no `jpeg`**. So `read()` returns `None` forever: **no error, no `camera_error`, no log
line**, just a permanently blank feed that looks like a camera fault. There is **no Lite3
equivalent of `go2_frame_server.py`** in the tree. **The viewport is blank whether or not
anything else holds the camera** — §10.4 is true, but it is not the reason.

⛔ **中文** —— 这与 §10.4 无关，而且更严重。`Lite3CameraSource.read()`
（`camera_source.py:269-273`）返回 `getattr(frame, "jpeg", None)`，
而 Lite3 的 `Frame`（`lite3/visual_nav/camera.py:28-35`）字段是
`image`、`capture_time`、`seq`、`stamp`——**没有 `jpeg`**。
所以 `read()` 永远返回 `None`：**没有报错、没有 `camera_error`、没有日志行**，
只有一个永久空白、看起来像相机故障的画面。
本仓库中**没有 `go2_frame_server.py` 的 Lite3 对应物**。
**无论有没有别的程序占着相机，画面都是空的**——§10.4 成立，但它不是原因。

### 10.7 Turning can never succeed; reverse is the only ungated motion
### 10.7 转向永远不会成功；倒退是唯一没有闸门的运动

⚠️ **EN** — `walk_forward` and both strafes pass `force` through to the gait-floor check.
**`turn_left` and `turn_right` take only `(seconds, rate_rad_s)` — there is no `force`
parameter** (`robot_driver.py:783`, `:796`), and yaw is floor-checked like everything else, so
**on a Lite3 they can never succeed.** Meanwhile **`walk_back` is never floor-checked at
all**, which makes reverse — open-loop, with **no rear sensing** — the only Lite3 motion the
dashboard lets through by default. **Do not use it as a "does motion work?" probe.**

⚠️ **中文** —— `walk_forward` 和两个横移都会把 `force` 传给步态下限检查。
**而 `turn_left` 和 `turn_right` 只接受 `(seconds, rate_rad_s)`——没有 `force` 参数**
（`robot_driver.py:783`、`:796`），且偏航同样要过下限检查，
所以**在 Lite3 上它们永远不可能成功**。
与此同时，**`walk_back` 根本不做下限检查**，
这使得倒退——开环、**没有任何后向感知**——成为默认情况下 dashboard 唯一放行的 Lite3 运动。
**不要拿它当作"运动能不能用"的试探手段。**

---

## 11. Where to look when something is wrong
## 11. 出问题时该看哪里

| symptom / 现象 | **EN** | **中文** |
| --- | --- | --- |
| no `MESH UP` | the driver is not running, or is on another interface | 驱动进程没起来，或者绑在了另一张网卡上 |
| robot accepts every command and never steps | check the vendor interface mode first, not the code | **先查厂商界面上的模式**，不要先查代码 |
| `run.supported: false` | the driver has no `--run-profile` | 驱动进程没有带 `--run-profile` |
| motion refused | missing `--allow-motion`, or missing `arm_motion`, or no measured gait floor — **the refusal says which** | 缺 `--allow-motion`、缺 `arm_motion`，或者没有实测步态下限——**拒绝信息里会说明是哪一个** |
| `the camera is in use` | expected on a Lite3 during a run — §10.4 | Lite3 运行期间属于**正常现象**——见 §10.4 |
| a permanently blank Lite3 viewport | **not a camera fault** — §10.6 | **不是相机故障**——见 §10.6 |
| `No module named 'ros2_twist_locomotion'` | **the transport, not the robot** — §10.5. Do **not** install anything on the robot | **是通道问题，不是机器人问题**——见 §10.5。**不要**在机器人上安装任何东西 |
| checkpoint panel shows the wrong disk | expected through an SSH wrapper — §7 | 通过 SSH wrapper 时属于**已知限制**——见 §7 |

**EN** — **Report failures plainly.** If a step fails, say so with the output. If you skipped
a step, say that. "It should work" is not a status. Every session ends with a continuation
comment on its GitHub issue — that comment is the only handover the next session gets.

**中文** —— **如实上报失败。** 某一步失败了，就把输出贴出来说明；跳过了某一步，也要说明。
"应该没问题"不算状态。每次工作结束都要在对应的 GitHub issue 上留一条接续评论——
**那条评论是下一次工作唯一的交接材料。**
