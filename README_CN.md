<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 使用 Arm Cloud AI 与 Arm Physical AI 的多智能体近端策略优化（MAPPO）

[English](README.md)

> 英文版 [README.md](README.md) 为权威版本；如两版有出入，以英文版为准。

一个在仿真中训练出来的 MAPPO 策略，驱动真实的四足机器人在同一个房间里走向各自的目标点。
本仓库是它们的交汇点：负责感知与行走的**机器人端控制栈**、它与策略之间的**遥测契约**，
以及把前者转换成后者的**适配层**。

构建于 **[Arm Device Connect](https://deviceconnect.dev/)** 之上 —— 用软件描述并驱动物理硬件的
开放标准（[github.com/arm/device-connect](https://github.com/arm/device-connect)）。

| 谁 | 做什么 |
| --- | --- |
| **Waheed Brown** ([@armwaheed](https://github.com/armwaheed)) | 机器人栈 —— 感知、规划、安全、遥测 |
| **Sagar Surendran** ([@spsagar13](https://github.com/spsagar13)) | MAPPO 策略、训练环境、检查点 |
| **Deep Robotics（云深处）** | Lite3 Venture 平台支持 —— 见下文*移植*一节 |

## 今天真正能跑的东西

一台 Go2，**只有 RGB** —— 没有 LiDAR、没有深度、没有动作捕捉。下面两段都是真机上的实机运行。

### 绕开一个静态障碍物，走向被检测出来的目标

![Go2 把回收箱建图并绕行，走向一把椅子](go2-static-obstacle-run.gif)

实时录制。机器人根本不知道"回收箱"是什么 —— 没有任何检测器是针对它训练的 —— 所以它靠颜色找到它、
校验形状，并**把它建图在 odom 坐标系里**，这样绕行动作把它移出画面之后它依然存在。目标是那把**椅子**，
由检测器在中心裁剪区上运行找到。检测框上带着单目测距结果和产生这个结果的先验；画中画是规划器自己的
信念，含已建图的回收箱和它选择的弧线。

`cmd avoid v=(+0.30,−0.20,+0.12)` 是机器人决定从右侧通过。

它走了 1.89 m，与回收箱齐平后停在那里 —— 办公室的通道到头了。那一刻规划器是满意的
（需要 0.60 m 间距，实际有 0.70 m）；只是地面没有了。这是走廊问题，不是规划问题。
`evidence/live_run.{mp4,log,jsonl}`。

### 给行人让路

![Go2 走向目标点并给行人让路](robot-stack/unitree/go2/visual_nav/images/go2-visual-nav-run.gif)

走向 2.0 m 外的一个航位推算航点，`arrived (0.96 m from goal)`，期间反复给一位横穿它路径的行人让路。
145 个感知周期，0 次错误，电机温度 31 → 32 °C。107 个控制 tick 中：63 次 `goal`、24 次 `avoid`、
20 次 `hold` —— 而且**间距变为负值的那 12 个 tick，每一个都下达了完全停止。**
这就是 Sagar 复核过的那次运行；`evidence/go2_nav_run.{mp4,log}`。

## 接口是 `--telemetry`，不是控制台日志

**不要解析控制台日志。** 它是给人读的散文，而且并不携带它看上去携带的东西。
按上面那次运行的 107 个控制 tick 统计：

| 策略需要什么 | 控制台日志里有什么 |
| --- | --- |
| 运动指令 | ✅ 每个 tick 都有 |
| 目标点 | ⚠️ 只有一个标量*距离* —— 没有位置 |
| 里程计 / 位姿 | ❌ **只有一次**，在启动横幅里 |
| 相机数据 | ❌ 没有（`lat=235ms` 是一帧的*年龄*） |

它还会为了可读性而被改写 —— `people=0` 在一周之内变成了 `obst=[binx1,personx1]`，因为一个裸计数
已经无法区分"已建图的回收箱"和"正在惯性滑行的幽灵目标"。这对散文是对的，对解析器是致命的。

所以这套栈改为写 JSONL：每个控制 tick 一个对象，带版本号：

```bash
python3 visual_nav.py ... --record run.mp4 --telemetry run.jsonl
```

```json
{"type": "tick", "t": 6.46, "pose": {"x": -2.464, "y": 2.112, "yaw": -1.613},
 "goal": {"x": -2.332, "y": -0.800, "distance_m": 2.915},
 "obstacles": [{"label": "bin", "kind": "static", "id": "landmark-1",
                "x": -2.276, "y": -0.081, "vx": 0.0, "vy": 0.0, "radius_m": 0.23}],
 "command": {"vx": 0.35, "vy": -0.135, "wz": 0.09, "reason": "goal", "gap_m": 0.752},
 "measured": {"vx": 0.331, "vy": -0.128, "wz": 0.084},
 "perception": {"seq": 83, "frame_age_s": 0.564, "video_frame": null, "stale": false},
 "posture": "standing", "live": false}
```

文件头一行声明**每个向量位于哪个坐标系** —— `pose`、`goal` 和 `obstacles` 是 odom 坐标系；
`command` 和 `measured` 是机体坐标系。这是消费方唯一无法从数据本身还原出来的东西：
只要机器人朝着它的起始航向，两个坐标系就*完全*重合，只有在它转向之后才会分开。
建立在错误假设上的集成会通过每一项台架测试，然后在第一个转弯处失败。

`kind` 取 `"static"` 或 `"tracked"`，而 `label` **不能**替代它：`label` 是类别名，
它之所以能区分这两者，只是因为当时场景里恰好只有一个已建图的道具和一个检测器类别。
一个*停下来*的人拥有回收箱的速度和行人对通道的占用权。`id` 才是稳定的身份标识，
消费方可以据此跨 tick 跟踪同一个物体，而不必按位置重新关联并合并邻近目标。

每一个 tick 都会写出来 —— 包括 hold、感知过期导致的跳过，以及目标搜索过程，因为
"它站着不动了 1.4 秒"是一个信号，而不是一段空白。`perception.video_frame` 是 `--record` 中
对应帧的索引，这就是回到录像的连接键。`evidence/sample_telemetry.jsonl` 就是上面那次实机运行。
每个 tick 还在下达速度旁边携带**实测**速度 —— 没有它，"下达了 0.12 m/s 却纹丝不动"
和"正在行走"就无法区分，而这个失败模式花了三次运行才被看见。

## 用一个 tick 驱动 MAPPO 策略

**策略包和它的检查点就在代码树里**，位于 [`policy/`](policy/) —— 262 KiB 的权重，
所以从一份干净的克隆就能跑起演示，issue 里引用的每个数字任何人都能复现。
[`policy/PROVENANCE.md`](policy/PROVENANCE.md) 列出了它入库时被修正的五处问题；
五处全都是静默失败，而交付时的冒烟测试在这五处都存在的情况下依然通过。

策略包自己做射线投射，所以这次集成是一次*映射*，而不是一个适配器：
`integration/mappo_bridge.py` 把一个遥测 tick 变成一个 `RobotInput`。其中三处映射不是想当然的那种，
每一处都由一个说明理由的测试钉住。

```python
from mappo_bridge import robot_input
from physical_ai_mappo import MappoController, RobotInput, StationaryObject

for tick in read_run("run.jsonl").ticks:
    mapped = robot_input(tick, reset_run=first)     # None while searching for the goal
    if mapped is None:
        continue
    mapped["stationary_objects"] = [StationaryObject(**o)
                                    for o in mapped["stationary_objects"]]
    out = controller.step(RobotInput(**mapped))
```

| 映射 | 想当然的答案 | 为什么它是错的 |
| --- | --- | --- |
| `velocity_frame` | `"odom"` | `measured` 是状态估计器给出的**机体**坐标系速度 |
| `external_hold` | `reason == "hold"` | 规划器也会为*回收箱*而 hold —— 把它转发过去，会在这套系统唯一存在意义的那个场景里把策略清零 |
| `timestamp_s` | `wall_time` | 它是拿来和 `time.monotonic()` 比较的；用纪元时间会让"年龄"变成约 −1.8e9 秒，于是过期判定门限永远不会触发 |

### 回放是验证映射正确与否的测试

逐字段对照的表格抓不出坐标系错误、单位错误，也抓不出一个"存在但含义不同"的字段。
把一次录制下来的运行通过真实检查点回放，可以：

```bash
cd integration && python3 replay_mappo.py ../evidence/sample_telemetry.jsonl
```

每次运行都配一次自己的对照实验 —— 同样的 tick，走第二个控制器，但把障碍物移除。
没有它，"策略把方向偏离目标方位角 36°"根本不构成任何证据：这个检查点在附近没有任何障碍物时，
本身就带有 6–16° 的航向偏差。

### 在策略驱动任何东西之前，先闭环

`replay_mappo.py` 是开环的：路径是随包交付的规划器走出来的，所以策略从未遇到过它自己的动作
所产生的状态。`integration/closed_loop_sim.py` 把环闭上 —— 动作 → 执行器 → 位姿 →
相机此刻能看见什么 → 下一次观测，全部经过同一个 bridge —— 并让策略**在完全相同的场景上**
与随包交付的规划器比较，因为"策略在 30 次里到达了 18 次"在不知道现任方案在同样这些运行上
表现如何之前，并不是一个结论。

```bash
cd integration && python3 closed_loop_sim.py --seeds 30 --scale 1.5 2.5 \
    --command-scale 0.3 0.6 1.0
```

它的判决，也是 [`deploy/README.md`](deploy/README.md) 推荐的配置，是策略**只有在规划器否决之下**
才可以驱动机器人：不加否决时，它在测试过的每一种配置下都发生了碰撞 —— 在策略包出厂自带的
那个 scale 下是 30 次中撞了 21 次。

### ⚠️ 对这个检查点，起约束作用的是*感知视距*，而不是射线扇

一般性的警告依然成立 —— 射线是采样，不是积分，所以只有当一个物体张角超过射线间隔的一半时
才保证被击中，`observation.reliable_range_m()` 会针对任意半径与射线扇计算出这个距离。
但真正限制这个交付检查点的**不是**它的 12 射线 360° 扇：

| 看见回收箱（实机建图半径 r = 0.42 m）的限制 | 距离 |
| --- | --- |
| 12 射线 360° 扇的几何限制 | 1.62 m |
| 策略感知视距 —— 0.35 VMAS × 2.5 m/单位 | **0.875 m** |

先起约束作用的是感知视距。决定它的是 `meters_per_vmas_unit`，而它是一个**标定参数**，
这一点已由 @spsagar13 确认：交付值 1.5 是把*房间*对齐到训练时的出生区域，而 2.5 是把*机器人*
对齐到训练时的智能体（实机运行的 0.25 m 规划半径 ÷ 训练时的 0.10 VMAS 智能体半径）。
用 `replay_mappo.py --scale` 扫描它，可以看出它买到了什么、又没买到什么：

| m/单位 | 感知视距 | 在多少个 tick 上看见障碍物 | 视距内的平均转向响应 |
| --- | --- | --- | --- |
| 1.5 | 0.525 m | 59/121 tick | 96.6° |
| **2.5** | **0.875 m** | **77/121** | **103.4°** |
| 4.0 | 1.400 m | 97/121 | 96.6° |

**这个响应是一道悬崖，不是一段斜坡，而且没有任何 scale 能修好它** —— 它在各处都饱和在
约 100°，而视距之外是 0.1°。提高 scale 买到的是*提前预警*，永远买不到比例性。
要把悬崖磨成斜坡，需要用更大的 `lidar_range` 重新训练；见 issue #4。

## 这套系统给不了策略什么

之所以写在这里，是因为一个测距向量*看起来*像一次 LiDAR 扫描，而它不是：

- **自由空间的意思是"没有识别到东西"，不是"那里没有东西"。** 这套栈只看得见被跟踪的行人
  和一个有名字的彩色道具。墙壁、桌腿和门框是隐形的。
- **没有后视。** 相机约 85°，而且机器人从不倒退。其余方向一律读作畅通 —— 也就是乐观的那个方向。
  一个学会了从死胡同倒着退出来的策略，会相信它身后的空间是空的。
- **同伴机器人没有一个相机能给它的*名字*。** 另一台四足机器人既不是随包交付的检测器类别，
  也不是一个颜色配置。库存网络给它打的标签是胡说八道（`motorbike` 613 次、`chair` 372 次、
  `aeroplane` 200 次），但位置是正确的 —— 而位置正是规划器需要的，所以部署栈按检测框形状路由，
  而不是按名字（PR #73）。`horse` 并不是它看上去的那个"正面视角例外"：在 1,903 张有标注的帧上
  读原始 softmax，它在 **0** 帧上超过 0.25，在 **0** 帧上进入前三。
  **要不要微调，这个问题重新打开了，而且当初是基于一次错误的比较被关掉的。** 这一条原本写的是
  *"推断同伴不值得做"*，依据是库存权重在**1,903 张有标注的同伴帧上取得 64% 召回、在 897 张
  无同伴帧上 18% 误报**，而最好的微调模型是 **53% 对 38%** —— 但两者用的都是微调模型自己的
  训练日数据，库存模型在留出日上根本没有被评过分。在留出日上、对所有模型用同一条判定规则评分：
  库存权重读出 **68% 召回、57% 误报**，每一个微调模型在同伴上都胜过它，
  但每一个微调模型都会漏掉随包交付的网络能看见的 15 个人中的 2 到 11 个。今天这两条路线都不可部署 ——
  见 [`detector/FROZEN-FEATURE-CEILING.md`](detector/FROZEN-FEATURE-CEILING.md)。
  今天的同伴机器人是**通过 Device Connect 网格发布它自己的位姿**到达的 —— 见下文*网格上的同伴机器人* ——
  这也正是被训练的策略所描述的方式，因为它学习时面对的 VMAS 智能体是互相观测真实位置的，
  而不是互相跑检测器。
- **感知比现实滞后几百毫秒**（中位数 309 ms，p90 436 ms）。这套栈会外推轨迹并膨胀它们的半径
  来覆盖这段滞后；策略看到的是这个结果，而不是原始传感器数据。

## 状态

| | |
| --- | --- |
| ✅ 走向目标点、给行人让路 | 已在硬件上验证（Go2 栈 PR #10） |
| ✅ 从一份干净的克隆就能运行 | Go2 栈 PR #11 |
| ✅ 已部署的代码树能说出自己的 commit | `deploy/push-to-robot.sh` 打上 git 的根 tree id，由磁盘上的字节重新计算得出；一旦代码树与之不再匹配，`mappo_drive.py` 就拒绝运行 —— 已在 Go2 上、Python 3.8.10 下演示，全程从未调用 `git`。十个 `~/mappo-*` 代码树中有六个原来是精确的检出；而启动脚本 source 的那一个 `~/mappo-main` 横跨 21 个 commit（[#128](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/128)） |
| ✅ 把静态障碍物建图、绕开它、走向被检测出来的目标 | 实机；走了 1.89 m，因通道宽度不足而停止 |
| ✅ 离线回归测试 | 每一个数量都记录在 [`.github/test-inventory.tsv`](.github/test-inventory.tsv) 里，由 `bash .github/measure-suites.sh --write` **自动生成**，并且 **CI 在每个 pull request 上都会重新测量**；只要其中任何数字与实测不一致——不论偏大还是偏小——CI 就会失败。⚠️ **这里刻意不重述任何合计值。** 这一行已经错过四次：先写 526，然后 772，然后 1,027（而强制值是 1,038），然后 1,046（而强制值是 1,337）—— 每一次的原因都相同：CI 不会读的数字，除了靠人记住，没有任何东西能让它保持正确 |
| ✅ 用一次录制的运行驱动 MAPPO 策略 | 回放了全部 122 个 tick；除了物体 id 之外映射干净，而日志现在已经携带 id |
| ✅ 策略包 + 检查点在代码树里 | `policy/`，262 KiB；修正了六处静默缺陷，每一处都由一个测试钉住 |
| ✅ 闭环仿真 | 30 个带随机种子的场景 × 3 个控制器 × 2 个 scale × 3 个指令 scale，每一组都配一次消融对照 |
| ⚠️ 策略感知视距 | 在重新标定后的 scale 下，到障碍物表面 0.875 m —— 它在 121 个 tick 中的 77 个上看得见回收箱，而在那个距离上的响应是一道悬崖而不是斜坡，**任何** scale 下都如此 |
| ⛔ 策略驱动机器人从**两个**障碍物**之间**通过 | 感知视距比通过口的宽度还短：在三次失败运行的 137 个 tick 中，两个回收箱同时在视距内的有 **0** 个；而在那次成功的运行里是 79 个中的 33 个。需要重新训练 —— issue #29，证据日期 2026-08-19 |
| ⚠️ 策略驱动腿，**受监督** | 在可行走的 1.0 指令 scale 下：仿真中 30 次到达 21 次，**1 次碰撞**；面对障碍物必须有规划器否决 |
| ⛔ 策略驱动腿，**无监督** | 在每一种仿真配置下都发生碰撞 —— 在策略包出厂自带的 scale 下是 30 次中 21 次。不是候选方案。 |
| ✅ 策略在 Go2 硬件上、空通道 | 走了 2.78 m 后在距椅子 0.77 m 处到达；策略驱动了 53/53 个 tick，0 次被否决，0 次被停止；带障碍物的运行仍未解决 |
| ⚠️ 越过回收箱抵达椅子 | 需要的通道比这条走廊长约 0.3 m |
| ⚠️ D1 机械臂锁存 | 它的舵机总线不上电（在上游 Go2 栈中跟踪）；运行时使用 `--no-latch-arm`，而机械臂每次运行都会偏离背脊线几度 |
| ⚠️ 把同伴机器人当作一个训练类别、从相机识别 | **2026-08-26 重新打开 —— 在同伴上更好，但在行人上不可部署。** 留出的 8 月 20 日划分（47 帧有同伴、136 帧无同伴），对所有模型用同一条判定规则：**库存**模型读出 68% 同伴召回、56% 误报，而每一个微调模型在同伴上都胜过它。早先的结论（"64% 对 18%，相对 53% 对 38%"）是在微调模型自己的训练日上评的分。**随后对已有的 640 个检查点中的 64 个评了分，而最好的那个从来没有人测量过** —— `k_full_pseudo03`，89% 召回、12% 误报，来自唯一一次 `--pseudo-labels 0.3` 的训练。64 个里有 13 个在两项同伴指标上都胜过随包交付的权重；全部 64 个在误报上都胜过它。⛔ **没有一个通过闸门 —— 随包交付的网络在 0.25 阈值下能看见的 22 个人，一个都不能丢。** 64 个里最好的那个保住了 20 个，而且同伴召回低于库存模型；同伴指标最好的那一行漏了五个人。标记物与颜色面板此前已被排除。[`evidence/2026-08-26-checkpoint-sweep/`](evidence/2026-08-26-checkpoint-sweep/) |
| ⏳ 与类别无关地从相机识别同伴机器人 | 检测框是存在的，位置也正确；但 `PersonDetector` 被配置成 `classes=("person",)`，在下游任何环节之前就把它们丢掉了。**测距那一半现在已经接通** —— `--static-detect-ground` 从物体接触地面的位置来测距，于是不再需要 `--static-detect-height` 和 `--static-detect-width`，一个谁也没测量过的物体在 0.81 m 到（由声明的俯仰误差决定的）一个上限之间是可测距的。仍未解决的是：这些误报中哪些应该由静态地图保留、而不是由跟踪器当成移动目标带走 —— 在布置好的空走廊上占 18% 的帧，在有家具的场景中跨日达 **56%**（PR #91 修好推导出的划分之后是 76/136；此前读作 134 中的 57%） |
| ✅ 无法测距的画面会让机器人停下 | issue [#72](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/72)：当物体与地面的接触点离开画面之后 —— **在画面角落是 0.806 m**，而不是通常引用的 0.719 m —— 这台机器人上没有任何估计器能说出它在哪里，而规划器会把一个空的障碍物列表读成一片**开阔场地**。现在，当"测距无法定位的东西"让锥形视野里剩下的可通行角度小于机器人本可转向的范围时，导航器会带着明确理由 hold。在两次已提交的实机运行的 16 个无法测距的 tick 中触发了 4 次，并让主打运行的 59 个 tick 中损失了 1 个。⚠️ **在真实场地里从未触发过** —— 它真正针对的那次运行 `live05` 并不在本仓库里。[`evidence/2026-08-26-no-open-bearing/`](evidence/2026-08-26-no-open-bearing/) |
| ✅ 通过 Device Connect 网格获得同伴机器人 | 同伴机器人以 10 Hz 发布自己的位姿，导航器把它当作一个普通障碍物消费 —— 没有检测器、没有标记物、不需要训练。66 个离线测试，其中 11 个做了变异检验；**尚未做过双机硬件运行** |
| ✅ 仪表板驱动一支机队 | 所有机器人同时列出、各自带一个停止按钮，另加 STOP ALL；跨机停止 4.23 s → 0.06 s，同机停止 4.17 s → 0.07 s，而且现在能打断行走 |
| ✅ Device Connect 仪表板，脱离机器人 | [#43](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/43)：事件、运动、检查点切换以及 Cloud AI 的加载/卸载，全链路跑通，走的是真实的 D2D 网格、对接台架替身 —— 见 `evidence/2026-08-21-device-connect-dashboard/` |
| ⏳ Device Connect 仪表板，在硬件上 | 尚未在机器人上运行。台架替身会精确交付它被下达的 1.00 倍，而这恰恰是真机绝不会给出的数字；那里没有任何东西测试了步态、DDS、ROS 桥或 SDK 导入 |
| ✅ Lite3 Venture 离线移植 | 高层 UDP/轴传输、RGB 相机、故障即关闭的健康闸门、标定与 MAPPO 入口点；132 个平台测试 |
| ✅ Lite3 高层解锁前基线 | 10 秒内 2,000 个有效的 220 字节 `RobotState` 帧，200.0 Hz，电量 99%，error 为 0；AI Motion Control Mode 报告为锁定 |
| ⛔ Lite3 传统自主速度接口 | 在厂商解锁 AI Motion 并由官方 App 确认可行走之后，`vx=0.10` 对应的 20 字节 320/325/321 数据包以 10 Hz 到达 UDP 43893，并带完整清零；但没有可见的前进动作、位姿变化量为零、error 为 0。厂商指南说这些指令需要自主模式，而 AI 状态无法进入自主模式 |
| ✅ Lite3 厂商 moving 模式轴指令验证 | 先后选择 manual（`0x21010C02`）与 moving（`0x21010D06`）模式后，一路源端口 20001 的心跳加轴指令流，在一次有界的 `+32767`、1 秒试验中让机器人移动了 0.401 m；实测机体 x 向峰值速度为 0.729 m/s，error 保持为 0，操作员确认了稳定停止。这**不构成**对视觉导航或重复运行的批准。 |
| ✅ Lite3 运动主机上的 MAPPO shadow | AArch64 上完成 30 秒的椅子/回收箱 MAPPO shadow，158 个感知周期，0 次感知错误，无任何执行动作；在 RTSP 读取故障之后，相机的关闭流程已被回归测试覆盖。 |
| ✅ Lite3 不行走的生命周期 | [#13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)：厂商 ARM 控制器完成了 Idle 获取 -> 3 秒阻尼 -> 一次 `ControlGet(1)` 释放；进程正常退出，厂商控制恢复 |
| ⛔ Lite3 短距离前进 RL 行走 | 控制器完成了 Idle -> StandUp -> RL -> 阻尼/释放，但 250 ms、1 s 和 5 s 的有界 `w` 输入都没有观察到前进动作；历史试验缺少 tick/载荷追踪，所以既不能证明推理真正被暴露出来，也不能排除时长因素；生产配置已恢复为源码测试过的 250 ms 死人开关 |
| ✅ Lite3 硬件联调 | [#13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)：两站位相机验证（站位 1 宽度 +2.4%，两站位仰角拟合 −0.57°；宽度-位移那一项 FAIL 是检测器最小面积形态学处理的假象，已记录在通过验证的 JSON 里）、两个方向的偏航原语 ±16000（±0.857 rad/s，固件回显）、步态下限 0.30 m/s、执行器增益 1.07、负载半径 0.40 m、由操作员确立的站立/移动状态（basic=6）。高层数据流中不含电机温度；实机运行接受有界的 `--accept-no-motor-temperatures` 豁免（≤120 s）。 |
| ⚠️ Lite3 实机导航，最初几次行走 | [#13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)：在 release v8 上的四次 `--live` 运行让机器人走了起来 —— 0.05→0.55 m/s 的爬升，目标距离从 2.29 m 收敛到 1.75 m（走了 0.54 m），随后对纸箱做出了正确的 58 秒走廊 hold；第四次运行**到达**了，距椅子 0.99 m。电量闸门被强制执行（50→47→35→28%），失速中止与安全驻车都被触发过。**但遥测显示这几次运行跑的是普通视觉导航的目标跟随，而不是 MAPPO** —— 控制原因只有 `goal`/`hold`，从未出现 `policy`/`veto-*` —— 而且没有一次运行绕开了纸箱：完整运动策略的横移意图被 `--max-vy 0` 钳位删掉了，因为这台 Lite3 的轴配置没有实测过的横向原语。[`evidence/2026-08-27-lite3-executable-avoidance/`](evidence/2026-08-27-lite3-executable-avoidance/) |
| ⏳ Lite3 可执行的避障（转-走监督器） | [#13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)：`mappo_drive.py --execution-supervisor turn-drive` —— 当一个已建图的静态障碍物挡住通往目标的直线时，避障被重新表达为纯转向与纯直行，也就是这台 Lite3 实测轴配置唯一能执行的动作；同时规划器的否决依然会审判监督器发出的指令，所以有人走进绕行路径时机器人仍会 hold。绕行段是按切线规划的，在静态硬间距之上留 2 cm 执行余量，而第二段只有在通往目标的直线畅通时才开始，因此监督器不会与自己的否决逻辑死锁。现在每一个规划器 tick 都记录完整的决策链（策略原始值 → 包络 → 步态下限 → 监督器 → 最终值 → 轴指令预览）以及原始传输轴值。**仅离线验证；尚未在硬件上运行。** |

## 移植到云深处 Lite3 Venture

活动用的两台 Lite3 Venture **有一个 RGB 相机、没有 LiDAR**。离线移植位于
[`robot-stack/deep_robotics/lite3/`](robot-stack/deep_robotics/lite3/README_CN.md)。
它没有让任何一台机器人动过；它的作业手册逐项列出联调时仍需补齐的每一项测量和每一路厂商数据，
而不是用 Go2 的数值把它们填满。

**在这些机器人上，网络是一个安全面，不只是管线。**
[`DEMO-NETWORK_CN.md`](robot-stack/deep_robotics/lite3/DEMO-NETWORK_CN.md) 是网络标准流程：
隔离的场地局域网、两台机器人的地址分配，以及最咬人的那一部分 —— 每台机器人用**同一路射频**
为自己的手柄提供服务。它在一路 PHY 上既托管接入点又接入场地路由器，所以 `#channels <= 1` 这条限制
起了作用：手柄 AP 在 5 GHz、路由器链路在 2.4 GHz 会被直接拒绝，然后机器人干脆什么都不广播。
一次 WiFi 变更就是这样静默地关掉了手动控制路径
（[A22](docs/WHITEPAPER.md#a22-we-configured-the-wifi-and-switched-off-the-robots-manual-control-path)）。

**原样可移植的部分。** 从像素到计划的每一步都是与机器人无关的 numpy 与 OpenCV：
`camera_model`（鱼眼像素 ↔ 方位角、角尺寸测距）、`person_detector`、`colour_detector`、
`tracker`、`static_map`、`avoidance`、`goal`、`overlay`、`telemetry`、`replay`。
它们没有一个 import 机器人。它们是模块的绝大部分，也是集成面的全部。

**厂商接缝已经实现。** 三个窄接口包住共同的主循环：

| 接缝 | Go2 | Lite3 Venture |
| --- | --- | --- |
| RGB 相机 | 来自 `VideoClient` 的 JPEG | 显式的 V4L2、RTSP 或 GStreamer BGR 源，带本地到达时间与位姿时间戳 |
| 行走 | CycloneDDS 上的 `SportClient` | 高层 `Lite3_ROS` 的 `/cmd_vel` + `/leg_odom2`；底层 MotionSDK 被刻意不用作步态控制器 |
| 安全 | 电机温度、电量、D1 机械臂收纳 | 标准的电量与电机温度 ROS 数据流；数据缺失或过期即拒绝实机运行；**不存在任何机械臂相关参数** |

**在相信任何一个距离值之前，先重新标定。** `go2_front_camera.json` 是*这一台*机器的镜头 ——
焦距 1290.2 px，HFOV 85.27°。系统里的每一个距离都与它成正比。Lite3 的包装层使用同样的
"绕自身旋转、对位姿偏航拟合"的方法，并在生成的 JSON 上标注平台名。Lite3 的实机运行会拒绝
一个属于 Go2 的、缺失的或格式错误的标定文件。

**在这台机器上，这个尺度现在已经测出来了，它是 1.0。** issue
[#35](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/35) 里对同一个回收箱
有四个互相相差 ±25% 的估计，而距离尺度从未做过端到端验证。用一个**打印出来的 10 cm ArUco 标记物**
（尺寸由制造工艺已知）在 2.2–2.6 m 的接近过程中对机器人自身里程计做拟合，得到
`evidence/2026-08-25-peer-runs/` 两次运行的 **k = 1.09 与 1.02**，在 43 个和 52 个定位点上残差
4–5 cm。*"地图读数偏长 25%"*（k ≈ 1.25）和*"地图读数偏短 24%"*（k ≈ 0.81）都无法在这个结果下成立，
所以**下文引用的米数是在一个实测为 1.0 的尺度下引用的，不需要修正。**
⚠️ k 是相机尺度与里程计尺度的*比值*；要排除"两者以同一个系数同时错了"，唯一的手段仍然是一把卷尺。
[`evidence/2026-08-26-range-scale-audit/`](evidence/2026-08-26-range-scale-audit/)

**没有机械臂是一种简化，不是一次绕过。** D1 让 Go2 栈付出了很大代价 —— 它是机器人在两次移动之间
趴着休息的原因、是速度包络被降额到 0.35 m/s 的原因，也是一次运行可能被直接拒绝的原因
（机械臂每次运行都会偏离背脊线几度，而这个闸门是绝对的）。Lite3 的参数解析器没有
`--no-require-arm` 也没有 `--no-latch-arm`：机械臂子系统在那个平台绑定里根本不存在。

**尚未解决的一项是硬件证据。** 一次实机运行需要实测的步态下限、执行器增益、负载半径以及
Lite3 的相机 JSON。它还需要公开的高层厂商桥并不发布的电量与电机温度话题。在有一个受支持的
配套数据源提供它们之前，这个绑定保持故障即关闭。同伴机器人已经不再是一个*检测*上的缺口 ——
网格上的一台 Lite3 通过与 Go2 相同的驱动发布自己的位姿 —— 但两个平台上都还没有做过双机运行。

## 网格上的同伴机器人

**第二台机器人是在完全没有感知的情况下被避开的。** 每台机器人把自己的位姿作为一个 `peer_pose`
事件发布到 Device Connect 网格上；导航器消费它，并把它当作又一个障碍物圆盘交给策略。
没有检测器、没有标记物、没有颜色面板，也没有任何东西需要训练。

```
  peer robot                                    navigator robot
  drive_bridge.py pose-stream  (py3.8, SDK)     peer_link.py       (py>=3.11, DC)
        │ JSON lines, 10 Hz                            │ writes ~/.mappo-peers.json
  robot_driver.py --publish-pose (py>=3.11) ═══════════╡ event(peer_pose)
                                    the mesh           │
                                                 peer_source.py    (py3.8, stdlib)
                                                       │ an odom-frame Obstacle
                                                 visual_nav's ONE obstacle list
                                                       ├── the policy   (a disc to path around)
                                                       ├── the veto     (rolled forward on its velocity)
                                                       ├── telemetry, the overlay, the log
```

在使用它之前有三件事值得知道。

**两台机器人的 odom 坐标系之间没有任何关系，直到有人去测量它。** 每一个都从那台机器人自己上电时的
位姿开始，所以一台同伴机器人报告 `(0, 0)` 并不说明那是哪里。因此
`--peer-odom-align DX,DY,DYAW_DEG` 是*启用*开关，而不是启用之后的一个选项：不存在任何一条
"同伴避障已开启但坐标系关系未声明"的代码路径。一台起点在前方 2 m、左侧 1 m 且朝后的同伴机器人
写作 `2.0,1.0,180`。这需要一把卷尺和一个地面标记，而这两样东西本来就是布置目标点所必需的。
它还会衰减：两套里程计各自漂移，而没有任何东西在观测这件事。

**同伴位姿不再到达，并不等于同伴机器人站着不动。** 那是一台位置不再已知的机器人，所以超过 0.6 秒后
这个障碍物会被**丢弃，同时机器人 hold** —— 而这是一个决定，不是两个。丢弃那个圆盘之所以安全，
仅仅因为腿停住了。0.6 秒就是 `perception_timeout_s`，也就是这套栈本来就为"相机落后了"分配的预算，
因为那是同一种失明。两次采样之间，同伴机器人按它最后的速度外推，半径按 `0.5·σ·t²` 增长，
而这个增长被限制在 0.18 m —— 正是因为超时会终结它。

**同伴机器人是一个 0.40 m 的圆盘，而这是半对角线。** 一台 Go2 是 0.70 × 0.31 m，
所以半长是 0.35，半对角线是 0.383。`avoidance.NavConfig.robot_radius_m` 对这台机器人自己的机体
做了同样的判断并向上取整；这里没有任何东西控制同伴机器人朝向哪一边，所以必须按长轴来算。
作为对比，彩色道具那条路径的默认值是 0.15 m，那是一个回收箱。

网格提供而任何检测器都提供不了的，是同伴机器人的**速度** —— 不过这个速度并不是给网络的，
网络没有障碍物速度通道。它到达的是规划器的推演，用来对照同伴机器人*将会*出现的位置来给策略的指令打分，
以及那个决定"同伴机器人究竟是交给策略处理、还是干脆为它停下"的速度闸门。

```bash
# on the peer
python3 dashboard/robot_driver.py --platform go2 --device-id mappo-go2-peer --publish-pose
# on the navigator
python3 dashboard/peer_link.py --peer mappo-go2-peer
python3 integration/mappo_drive.py --live --peer-odom-align 2.0,1.0,180 ...
```

## 从浏览器观看并驱动它

[`dashboard/`](dashboard/README.md) 把机器人作为一个设备接入 Device Connect 网格，并提供一个
能发现它的网页 —— 实时事件流、有界的运动、检查点切换，以及从 S3 存储桶或局域网内的服务器加载
检查点。没有 broker、没有 etcd、没有 Docker：D2D 模式通过组播在演示本来就在跑的那个局域网上找到机器人。

完全不用机器人也能试：

```bash
pip install device-connect-edge device-connect-agent-tools aiohttp    # Python >= 3.11

cd dashboard
python3 robot_driver.py --platform sim --package ../policy --allow-motion   # terminal 1
python3 server.py --port 8080                                              # terminal 2
```

然后打开 <http://127.0.0.1:8080>。`--platform sim` 是一个台架替身，它把下达的速度积分成位姿；
它能在房间里没有机器人的情况下走通网格、schema、事件流和每一条拒绝逻辑。

在真机上，启动时**不要**带 `--allow-motion` —— 此时设备只提供状态与检查点功能 —— 并用
`--bridge-python` 指向那个能 import 机器人 SDK 的解释器：

```bash
python3 robot_driver.py --platform go2 --package ../policy \
        --bridge-python /home/unitree/robotics-connect-go2/bin/python
```

Device Connect 需要 Python ≥ 3.11，而 Go2 的 Jetson 用 3.8 跑它的 SDK，所以驱动通过在那第二个
环境里把 `drive_bridge.py` 作为子进程运行来触达机器人。这个拆分也正是"驱动挂起或被杀掉"不会
留下一个锁存速度的原因。

**`robot-stack/SAFETY.md` 管辖这些运动按钮，正如它管辖 `--live` 一样。**
`--allow-motion` 就是这个目录下的 `--live`：它需要一片清空的区域、一名守在手柄中止键上的操作员，
以及足够的电量。这个页面**没有登录**，所以 `--host 0.0.0.0` 意味着任何能连到这个端口的人，
都可以驱动网格上任何一台启用了运动的机器人。

有两件事它刻意不去粉饰。每一次运动按压都是有时长上限且开环的，并会报告机器人*实测*到了什么 ——
包括下达速度中实际交付出去的比例，这台 Go2 上大约是 0.45，也正是这个数字解释了
"机器人看上去没在动"。另外，两个平台并不可以互换：Lite3 上的 `lie_down` 只是让它*停下*，
因为那台机器人的姿态是操作员通过厂商 App 控制的；而 Go2 的横移带有警告，因为那台机器人的
横向步态下限从未被测量过（[#42](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/42)）。
这个页面是从 `get_capabilities()` 学到这两件事的，而不是把任何一件写死在代码里。

## 目录结构

```
robot-stack/     Go2 control stack plus Lite3 platform bindings — see PROVENANCE.md
policy/          the MAPPO adapter and checkpoint, vendored — see policy/PROVENANCE.md
integration/     the bridge, the replay, the closed-loop sim, and the two live runners
deploy/          install.sh, uninstall.sh, and the runbook for a day at the robot
dashboard/       the robot as a Device Connect device, and a browser page that drives it
evidence/        the approved run, the static-obstacle dry run, a sample telemetry file
```

| `integration/` 中 | |
| --- | --- |
| `mappo_bridge.py` | 一个遥测 tick → 一个 `RobotInput`。那三处不想当然的映射。 |
| `mappo_policy.py` | 共享的主循环：bridge → 策略 → 指令，外加航向伺服 |
| `replay_mappo.py` | 把一次录制的运行送进检查点，并与一次消融对照比较 |
| `render_observation.py` | 相机画面、射线扇与观测向量，按 tick 并排绘制 —— 策略看到了什么、为什么 |
| `closed_loop_sim.py` | 用策略自己的动作驱动一台仿真机器人 —— issue #5 的闸门 |
| `mappo_shadow.py` | 一次**实机**运行，策略与规划器并排记录。不能移动任何一条腿。 |
| `mappo_drive.py` | 一次实机运行，策略在规划器否决之下驱动机器人，通过一个受支持的上游接缝 |

## 运行测试

全部测试，从仓库根目录，用 CI 所用的同一个脚本：

```bash
bash .github/measure-suites.sh            # every suite; one `  ok  ` line per test
bash .github/measure-suites.sh --check    # ...and fail if any count has drifted
```

`--check` 需要 Python ≥ 3.11，并且能 import `numpy`、`opencv-python`、`Pillow`、`pytest` 和
`aiohttp`；否则它宁可拒绝运行，也不会报出一个因解释器够不到而短缺的合计值。不带参数的
`bash .github/measure-suites.sh` 可以在 3.8 下运行，也就是机器人 Jetson 上的版本，并且遇到坏掉的
测试文件依然会失败。

一次一个目录 —— **括号是有作用的**，因为这些目录是嵌套的，一个裸 `cd` 会让下一行在上一行的
目录里执行：

```bash
(cd policy && for t in test_*.py; do python3 "$t"; done)
```

这些目录列在 [`AGENTS_CN.md`](AGENTS_CN.md) 里；每个目录的数量在
[`.github/test-inventory.tsv`](.github/test-inventory.tsv) 里，该文件由
`bash .github/measure-suites.sh --write` 重新生成，并且 CI 在每个 pull request 上都会重新测量，
只要有任何一个方向的不一致就失败。**两者都不在这里重复**，因为这一段以前自带过它们各自的副本，
而等到有人来读的时候，它们已经在两个方向上同时错了：声称的测试数大约只有代码树实际运行的一半，
列出了七个目录而实际存在十二个，Lite3 联调那一行还指名了一个装着十个文件的目录里的单个文件。
一个数量的第二份副本就是第二个会漂移的东西，而其中只有一份是被校验的。

`policy/` 以及 `integration/` 中接触策略的部分需要 `numpy`；机器人栈的测试还需要
`opencv-python`；`dashboard/` 需要在 Python ≥ 3.11 下的 `device-connect-edge`、
`device-connect-agent-tools` 和 `aiohttp`。`deploy/install.sh` 会在安装过程中运行 policy 与
integration 的测试，因为一份被截断的检出应该在那里失败，而不是在场地上失败。有十三个目录带
`ruff.toml`；在它所管辖的目录内部运行 `ruff check .`，每一个都是干净的。

## 安全

`robot-stack/SAFETY.md` 管辖任何会让腿动起来的东西，而且它不是可选的。简而言之：`--live` 是
唯一会让机器人移动的参数，操作员始终守在遥控器上，通道里除了道具之外一律清空，而且机器人由
网线牵着 —— 任何转向动作之前先检查线缆余量。

对于策略驱动的运行，[`deploy/README.md`](deploy/README.md) 补上了那道阶梯 —— 先仿真、再 shadow、
再驱动 —— 以及那些用来判断"这次运行是不是在做它看上去在做的事"的实测数值。执行器增益与速率有关：
这台 Go2 在降额时实测约 0.45，在满指令下 0.70，而在步态下限以下为零。
