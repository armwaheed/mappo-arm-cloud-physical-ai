<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lite3 实机运行：所有参数与所有拒绝原因

[English](LIVE-RUN-RUNBOOK.md)

> 英文版 [LIVE-RUN-RUNBOOK.md](LIVE-RUN-RUNBOOK.md) 为权威版本；如两版有出入，以英文版为准。

Lite3 实机运行需要**九项**参数，缺一即拒绝。拒绝会在约 3 秒内退出，机器人尚未站起，
从远处看就像"什么都没发生"。本页的目的是让任何一次拒绝都不会浪费你一天。

---

## 0. 先别手动输入这些参数

`commissioning/commission.py` 会测量全部参数并**直接输出命令行参数**。请从这里开始；
下面第 1 节是排查拒绝原因时的参考。

```bash
cd robot-stack/deep_robotics/lite3/commissioning

# 1. Nothing moves: read-only probes, tape measure, marker calibration.
python3 commission.py --robot-id LITE3-A --firmware V1.0.8 --payload none \
    --front 0.42 --back 0.38 --left 0.24 --right 0.24 --stance-confirmed \
    --camera-source 0 --marker 1.50 --marker-size 0.15 \
    --lens-height 0.37 --lens-height-source 'tape, standing, floor to lens centre'

# 2. The two that walk. Lane clear, robot standing, hand on the stop.
python3 commission.py ... --live --operator-ready \
    --ladder-top 0.50 --lateral-top 0.30 --lane-metres 6.0 --lane-width-metres 2.0 \
    --envelope-vx 0.35

# 3. A human reads the numbers and signs for them.
python3 commission.py --record lite3-commissioning-LITE3-A.json --review 'Timo Tang'

# 4. Now the live-run flags exist.
python3 commission.py --record lite3-commissioning-LITE3-A.json --emit-flags
```

**在接近机器人之前，先问它缺什么**：

```bash
python3 commission.py --record <artefact>.json --emit-flags
```

对一份不完整的记录，它会拒绝并**逐项点名缺失内容** —— *"this record cannot produce
live-run flags; it is missing …. Run those stages before asking for the flags."*
这项检查零成本，也不会让任何一条腿动起来。

**顺序本身就是安全论证**，不是为了方便：先只读，再卷尺，再相机，最后才是任何会走路的东西 ——
而且步态下限要在执行器增益**之前**测量，因为一条跨越步态下限拟合出来的增益，
会被它吞进去的每一个低于下限的点拖低。请勿调换顺序。

⚠️ `--emit-flags` **拒绝 `provisional` 记录。** 第 3 步不是走流程：一个已经被测量过的数字
和一个被人*相信*的数字是两回事，而只有人能把后者变成前者。

电机温度有自己的探测工具 —— `motor_temperature_probe.py` —— 所以下面第 3 节问的是限值，
而不是怎么造一件仪器。

---

## 0b. 改用仪表板运行

整个演示可以在 Device Connect 仪表板上启动和停止，完全不用终端。让它成立的是什么，
以及在信任它之前要检查什么：

| 组件 | 位置 | 为什么它不是可选项 |
| --- | --- | --- |
| 驱动跑**在机器人上** | `mappo-dc-driver.service` | 工作站上的驱动读不到 `jy_exe` 的状态（状态发往 `127.0.0.1`），而且 `download_model` 会把模型下载到工作站上 |
| `script` 必须是 **`venue_run.py`** | `~/mappo-lite3-stage/dc/run-profile.json` | 写成 `mission.py` 会失败 —— `run_control` 拼的是 `--package`，而 supervisor 从没听说过这个参数 |
| 配置文件 `env` 里的 `MAPPO_VOICE_DIR` | 同一个文件 | 不设它，这次运行就是静音的，而且启动时会这么说 |
| unit 里的 `XDG_RUNTIME_DIR` + `PULSE_SERVER`，以及 `loginctl enable-linger` | `mappo-dc-driver.service.d/audio.conf` | 服务没有登录会话，所以 `aplay` 够不到 PulseAudio，于是运行成功而每一条语音提示都失败 |
| 机器人上的**帧服务器** | `lite3-frame-server.service`，由 `--camera-url http://<robot>:8801/` 读取 | 没有它，相机窗格就是空的；而且 `go2_frame_server.py` 顶替不了 —— 它 import 的是 `unitree_sdk2py`，而这台机器人发布的是 RTSP。两台机器人上都已启用；实测每次 GET 返回 HTTP 200 与一张 127–135 KB 的 JPEG |
| unit 里**没有 `--allow-motion`** | 同上 | 一个会自行启动的服务，绝不能一起来就具备移动机器人的能力 |

⚠️ **实机运行需要 `--allow-motion`，而那是一个由人在控制台前做出的决定。**
`build_run_argv` 会拒绝没有它的 `--live`，而不是降级成一次空跑，因为一次"启动了但动不了"的运行
与一次"根本不会动"的运行无法区分，而你会把这个差别花在诊断机器人上。
要武装它，请在通道清空、手放在停止键上的前提下：

```bash
sudo systemctl edit mappo-dc-driver     # add:  [Service] / ExecStart= / ExecStart=... --allow-motion
sudo systemctl restart mappo-dc-driver
```

演示结束后用同样的方式解除武装。在信任一次仪表板运行之前的检查：

```bash
# the robot answers, with a pose that is not a simulated zero
curl -s -X POST http://<laptop>:8080/api/invoke -H 'Content-Type: application/json' \
  -d '{"device_id":"mappo-lite3-robot1","function":"get_status","params":{}}'

# and the run narrates: if the panel shows three lines for a 25-second run, something
# in the chain is buffered -- both mission.py and mappo_drive.py must be started with -u
```

## 1. 九项必需输入

| 参数 | 数值来源 |
|---|---|
| `--calibration` | 这台 Lite3 的相机。⚠️ 见 §4 —— 目前在流通的那份文件自相矛盾。 |
| `--gait-floor` | **实测**，用 `axis_primitive_probe.py`。不是 Go2 的 0.35。 |
| `--actuator-gain` | **在这个包络下实测** —— 对位姿拟合，不要对速度估计值拟合。 |
| `--robot-radius` | **0.40** —— 机器人本身。⚠️ 不是 0.33：那是*障碍物纸箱*（`DEPLOYMENT-SOP.md:383`）。必须满足 `--policy-scale = radius / 0.10`，所以 0.40 配 4.0。 |
| `--max-vx` | 必须显式声明。不设它，会静默继承 **Go2 的**值。 |
| `--max-vy` | 必须显式声明。这台 Lite3 的横向原语未经实测，所以填 `0` 才是诚实的。 |
| `--max-wz` | 必须显式声明。不设它，继承 Go2 的值。 |
| `--operator-ready` | 在机器人已经**站立**、并已在厂商 App 上进入导航模式**之后**才输入。 |
| `--axis-profile` | 仅当 `--locomotion-transport axis` 时需要。需要有物理证据支撑的原语。 |

> ⚠️ **0.40 是机器人半径，0.33 是障碍物纸箱半径。** 本页早期版本混淆了两者，
> 导致命令直接被拒绝：`--robot-radius` 同时是策略缩放门限的分子（`scale = radius / 0.10`），
> 0.33 对应 **3.30**，与 `--policy-scale 4.0` 不匹配，机器人尚未站起就会退出。
> 实测的 **0.28–0.33 m** 是活动用的纸箱尺寸，相对标称的 0.20 m —— 见
> `DEPLOYMENT-SOP.md:383` 与 issue #146，那个 issue 讲的就是这个参数承载了两个互不相关的含义。

**为什么包络值不能用默认值**：它是一道安全门限的右侧。`_validate_axis_profile_speeds`
会拒绝一个 `measured_m_s` 超过 `--max-vx x --derate` 的原语。若右侧是借来的，
这个比较就只是算术，而不是门限。

另有一项要求，且不是命令行参数：**必须在虚拟环境里运行**。
`require_virtualenv(reaching_hardware=True)` 在每次实机运行时都会打印信息，
包括它决定不强制执行的时候。

---

## 2. 拒绝信息对照表

| 你看到的信息 | 它的含义 | 该怎么做 |
|---|---|---|
| `REFUSING TO WALK: missing ...` | 上面九项中的一项或多项缺失 | 读那份清单 —— 它会逐项点名并说明原因 |
| `REFUSING TO WALK: --accept-no-motor-temperatures needs --max-seconds set to 120s or less` | 温度数据缺失**并且**这次运行没有时长上限 | 加上 `--max-seconds 120` 或更小。见 §3。 |
| `REFUSING TO WALK: <D1 arm not stowed>` | 仅 Go2 —— 机械臂未收纳 | 用手把机械臂摆平贴在背脊线上 |
| `TOP SPEED x IS BELOW THIS ROBOT'S MEASURED GAIT FLOOR y` | **这是警告，不是拒绝。** 运行会继续，而机器人可能不动 | 提高包络值或指令 scale |
| `[tree-stamp] no stamp -- this tree cannot name its commit` | 在机器人之外：这是一个发现。**在机器人上：这是一次拒绝** | 用 `deploy/push-to-robot.sh` 部署 |
| 约 3 秒退出、没有启动横幅、返回码 139 | `Segmentation fault` —— SDK 环境没有 source | 先执行 `source .../install/setup_env.sh` |
| `ModuleNotFoundError: ros2_twist_locomotion` | 仪表板走的是 ROS 2 路径 | 改用 `--locomotion-transport axis`（或 `udp`） |

**一次拒绝就是系统在正常工作。** 每一条拒绝都源自一次实测到的故障。

---

## 3. 电机温度：我们还没有数值

温度问题仍在与云深处确认。在得到答案之前，`--accept-no-motor-temperatures`
每次运行限时 **120 秒**，而且没有任何东西约束*下一次*运行：热量会在背靠背的多次运行中累积，
这里没有任何软件看得见它。

**我们手上有另一台机器人的可比数值，而它令人警醒。** Go2 在 **70 °C** 中止、
在 **55 °C** 告警，空载约 **30 °C**；如果让它一直站着，它的后腿电机会过热并瘫倒 ——
这也正是那台机器人平时趴着的原因。

### 给你的编程助手的任务

> 请观察这台 Lite3 能提供的任何电机温度遥测，找出它真实的限值。在多次**限时**运行中记录温度 ——
> 从冷机开始，运行 120 秒，记录曲线，让它冷却，再重复 —— 并报告：空载温度、负载下的升温速率，
> 以及这个通道究竟存不存在。如果 Lite3 **完全没有**温度通道，那这本身就是结论：
> 请明确说出来，而不是什么都不报。不要用背靠背连续运行去"够到"上限来找它。

---

## 4. ⚠️ 现有标定文件自相矛盾

2026-08-27 那批录制里内嵌的相机数据块：

```json
{"focal_px": 469.63, "height_m": 0.4, "hfov_deg": 156.16, "width": 1280, "height": 720}
```

- `height_m` **0.40** 与实测的站立 **0.37** / 趴下 **0.115** 矛盾；
- **没有 pitch**，而这个安装位置是固定的**约 11°**；
- `focal_px` 推出的视场是 **107.46°**，`hfov_deg` 却说 **156.16°** —— **相差 48.7°** ——
  而且完全没有 Go2 的数据块所带的 `method` / `samples` / `residual_deg_rms` 溯源信息。

**因此由它推导出来的每一个 `range_m` 都不是测量值。** 像素框坐标是可用的。
见 `robot-stack/CAMERA-GEOMETRY.md`。修复办法是在一台 Lite3 上做一次 spin 标定。

---

## 5. 运行前

1. **先跑 shadow。** 在任何实机运行之前，确认遥测中出现了 `decision` 与 `transport`。
2. **隔离通道。** 一位路过的同事并没有接受这个风险；接受它的人是你。
3. **手持急停**，若电机有异味或步态发生变化，立即停止。
4. `--operator-ready` **最后**输入，在机器人站立并进入导航模式之后。

## 6. 已验证的命令形状

⚠️ **其中的数值是占位符 —— 请替换为你自己的实测值。** 这里展示的是哪些参数必须出现，
而不是它们应该等于多少。

```bash
source /path/to/robot-stack/deep_robotics/lite3/install/setup_env.sh   # or the SDK env
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

**`--record-raw` 是人们最常忘记的那一个。** `--record` 会把 HUD 和每一个检测框烧进像素里，
这让文件便于阅读，同时**让它作为训练数据毫无用处**。`--record-raw` 写的是同一帧在被画上任何东西
之前的样子，而且两者与遥测中的 `perception.video_frame` 共用同一个帧索引 ——
于是你的检测结果就成了干净像素的标注。

录制会牺牲控制循环速率：带 `--record` 时实测每个出新结果的 tick 为 **246.4 ms**，
不带时为 **100.6 ms**（#18）。一次为了训练而录制的运行，其时序数字不具备任何意义。
