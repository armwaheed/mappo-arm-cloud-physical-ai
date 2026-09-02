<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 云深处 Lite3 Venture：纯 RGB 演示移植

[English](README.md)

> 英文版 [README.md](README.md) 为权威版本；如两版有出入，以英文版为准。

这是把 Go2 上用的同一套视觉导航器与 MAPPO 集成绑定到 Lite3 Venture 上的实现。
它假定活动用的两台机器人各有一个前向 RGB 相机、**没有 LiDAR**。这条路径上没有任何东西
会启动 LiDAR 节点或消费点云。

📄 **第一次让这台机器人对接 Device Connect 仪表板？** 请用
[`LITE3-DASHBOARD-BRINGUP-PROMPT.md`](LITE3-DASHBOARD-BRINGUP-PROMPT.md) —— 一份双语
（EN/中文）的分阶段联调流程，写来直接粘贴给编码 agent 使用，每一阶段的检查不通过就停下。
它面向操作员的配套文档是
[`../../../dashboard/OPERATOR-GUIDE_CN.md`](../../../dashboard/OPERATOR-GUIDE_CN.md)。

视觉导航与 MAPPO 这条路径只做过离线测试。另有一次独立的、有界的厂商高层行走验证，
在 2026-08-24 让一台活动机器人动了起来；它**不构成**对视觉导航或 MAPPO 路径移动机器人的授权。
硬件联调在 [issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)
中跟踪，仍须补齐下表中的四项测量以及两路健康数据；其中任何一项缺失时，实机模式都会故障即关闭。

| 平台项 | 实现方式 | 硬件证据 |
| --- | --- | --- |
| 高层行走 | 有界的厂商 moving 模式轴 UDP、传统复杂 UDP，或 `Lite3_ROS` 的 `/cmd_vel` + `/leg_odom2` | 一次有界的前向轴验证移动了 0.401 m 并干净停止；通用速度映射仍未验证 |
| RGB 采集 | 显式的 V4L2 索引、RTSP URI 或 GStreamer 管线 | 端点尚未提供 |
| 步态下限 | 必须作为 `--gait-floor` 提供 | 未实测 —— **而且在轴传输上根本无法测量**，因为那里的映射只有符号。见下文 |
| 执行器增益 | 必须作为 `--actuator-gain` 提供 | 未实测 —— 同上；这个比值的分母（下达的幅值）根本没有上线 |
| 轴原语速度 | 轴配置文件中的 `measured_m_s`；包络闸门会读它，而自 #145 起规划器的可行性推演也会读它 —— 没有它，实机运行会被拒绝 | 未实测。工具是 `commissioning/axis_primitive_probe.py` |
| 轴原语偏航角速度 | `measured_rad_s`；**所有配置文件都未声明它**，因此规划器不会把转向与迈步组合起来 | 未实测，而且只要 `Segment.yaw_change_deg` 的角度解缠 bug 还在，原语探测工具就刻意不去测它 |
| 负载规划半径 | 必须作为 `--robot-radius` 提供 | 已实测，但 ⛔ **是 0.40 还是 0.33 —— SOP 与 `LIVE-RUN-RUNBOOK.md` 说法不一致，而 0.33 推出的策略 scale 是 3.30 而不是 4.0**。见 [`robot-stack/CHASSIS-GEOMETRY.md`](../../CHASSIS-GEOMETRY.md) |
| 焦距 / HFOV | 实机运行要求带 Lite3 标签的标定 JSON | 未实测 |
| 电量 | 有文档记载的传统 `RobotState` UDP 字段 | 2026-08-21 厂商服务重启后为 21% |
| 电机温度 | 高层接口中不存在 | 厂商问题，仍未解决 |

## 这套移植用了哪些厂商接口，以及为什么

云深处以两种方式暴露同一个高层步态控制器，另有第三个更底层的接口，本仓库不使用它。

**传统复杂速度 UDP 接口 —— 离线绑定的默认选项。** 运动主机在 43893 端口接收一个 20 字节的
`{int32 cmd_code, int32 size, int32 type, double data}` 帧，其中代码 `320` 携带前向速度、
`325` 横向、`321` 偏航。它把位姿、机体速度、IMU、关节、手柄状态和电量回传到
`~/jy_exe/conf/network.toml` 中配置的那个唯一地址，端口 43897。
`robot-stack/deep_robotics/lite3/locomotion/lite3_udp_locomotion.py` 说的正是这套协议，
`--locomotion-transport udp` 选择它。

**`Lite3_ROS` 桥 —— 同一个东西，包了一层。** 官方
[Lite3_ROS](https://github.com/DeepRoboticsLab/Lite3_ROS) 的 `transfer` 包把那个控制器
通过一个 `geometry_msgs/msg/Twist` 指令暴露出来，并以 `nav_msgs/msg/Odometry` 发布位姿
和实测机体速度。它文档中的约定与本套栈一致：+x 向前、+y 向左、+yaw 向左。
读 `Jetson2Motion.cpp` 可以看到，它整条指令路径就是上面那三次 `sendto`；它没有增加心跳、
没有定时器、也没有周期性发送。它跑在一台*感知*主机上 —— 可执行文件叫 `jetson2motion`，
它的 `target_ip` 默认指向运动主机 —— 所以它需要一个 ROS 2 Foxy 运行时，
而这两台 Venture 上可能并没有这样一台计算机。对确实带它的机器，`--locomotion-transport ros2`
选择它。

官方 [Lite3_MotionSDK](https://github.com/DeepRoboticsLab/Lite3_MotionSDK) 是另一个层级。
它的示例会复位全部 12 个关节并取得底层控制权；调用方必须自己提供关节目标、增益、平衡和步态。
本仓库不包含 Lite3 的平衡控制器，所以把那个 SDK 当作速度行走客户端来用，
就等于移除掉让机器人保持直立的厂商控制器。因此本移植刻意不使用它。

公开的 ROS 桥要求操作员先让机器人站起来，并选择它的高层导航/AUTO 模式。它的 README 描述的是
在 App 上做这个切换，但 [Lite3_ROS issue #2](https://github.com/DeepRoboticsLab/Lite3_ROS/issues/2)
报告说，一些低配机型没有那个 App 按钮，需要与固件版本相关的 `/simple_cmd` 代码外加一个心跳。
那些数字代码不是稳定的公开 API，所以这个绑定不发送它们。联调时必须为每台活动机器人取得
被认可的操作序列；`--operator-ready` 记录的是"外部切换已成功"这件事。
没有任何有文档记载的高层"趴下"指令，所以清理阶段是重复发布零速度，
并由操作员通过被认可的厂商接口把机器人恢复到手动/趴下状态。

### 有界的直接速度验证

[`locomotion/lite3_velocity_udp.py`](locomotion/lite3_velocity_udp.py) 是对 `Lite3_ROS`
`Jetson2Motion.cpp` 中那三个出站帧的手工、有界复现；它既不 import ROS，也不解码状态。
在活动用的运动主机上，厂商那个打包过的 C++ 结构体实测为 20 字节，与它显式的小端 Python
编码一致：`<int32 code, int32 size=8, int32 type=1, double data>`。它只接受代码 320、325
和 321，并且像那个桥一样把偏航字段取负后发送。

第一条实机指令刻意比厂商示例更窄：只有前向 `0.00..0.10 m/s`，没有横向或偏航分量，
最长一秒，而且每一条退出路径都会连续发送两秒的零速度三元组。即使是只发零值的实机检查，
也需要 `--operator-ready`；一条非零指令还需要操作员确认那个有文档记载的外部控制模式、
一条清空的通道，以及手持急停。

2026-08-21，在 `192.168.1.120:43893` 被动抓取到一路 10 Hz 的纯零值流：来自开发主机的 12 个包，
全部是 20 字节的 320/325/321 零三元组。机器人保持静止。当前的 Venture App 暴露的是
**AI Motion Mode**，而公开的 `Lite3_ROS` 桥文档写的是 **Auto Mode**。没有任何公开资料把这两个
标签对应起来，所以在厂商确认适用的外部控制切换方式之前，这个工具绝不能发送非零速度。
当前的运动主机镜像也缺少官方 Venture SDK Mode 路径所需的 ROS 2 与 SDK 服务包，
所以 SDK Mode 目前也提供不了另一条控制切换途径。

#### AI Motion Control Mode 闸门

高层发送端已经就绪，但必须停留在空跑或零速度状态，直到一位云深处工程师在这台特定的
Venture 上以正当方式启用了 **AI Motion Control Mode**。不要绕过、修改或逆向那个激活过程。
在发出一条非零程序指令之前：

1. 在厂商解锁之后重新采集一段 10 秒的 `RobotState` 基线。只把有文档记载的字段与解锁前的采集
   做对比；某个未文档化的模式比特位不是必须条件。
2. 让操作员通过官方 App/手柄证明机器人能小步前进并停下。如果在那里都走不了，就停下来：
   本仓库不是应该首先排查的故障点。
3. 在机器人由自身控制器支撑站立时，抓取一路到达 UDP `43893` 的 10 Hz 零三元组流，
   并确认 `RobotState` 是新鲜的、电量有效、`error_state == 0`。
4. 为一次 `vx=0.10`、`vy=0`、`wz=0`、10 Hz、持续 1 秒的运行取得单独的、即时的操作员授权。
   它必须以至少两秒的零三元组收尾。记录下达指令与实测机体速度、错误状态，以及操作员目视到的结果。

不能假定 AI Motion 这个标签等价于传统的 Auto Mode。2026-08-24，在厂商确认 AI Motion 解锁、
操作员确认官方 App 可行走之后，一次 `vx=0.10`、10 Hz、持续 1 秒的高层脉冲被抓取到，
它以有文档记载的 20 字节 320/325/321 序列到达 `43893`，并包含零速度清理。机器人保持稳定，
但没有可见的移动；六秒有效的 `RobotState` 显示世界位姿变化量为零、`error_state == 0`。
没有尝试更高速度，也没有重复。

厂商的运动主机通信指南解释了这个 A/B 结果：它的复杂浮点速度指令
（`0x0140`、`0x0145`、`0x0141`，等价于 320/325/321）必须在**自主模式**下发送，
而处于 AI 状态的机器人无法切进自主模式。在 moving 模式或 AI 状态下，
该指南另外规定了一套独立的简单轴指令接口：至少 20 Hz、250 ms 指令超时，以及用零轴值停止。

### 已验证的 manual-moving 轴路径

2026-08-24，使用厂商的参考控制脚本与通信指南，做了一次有界的、单独授权的硬件验证：

1. 严格的主机抓包确认了一条小端 12 字节的 manual 模式指令 `0x21010C02`，随后一条 moving
   模式指令 `0x21010D06`；这两条指令都不含轴值或速度值。
2. 一次不产生动作的健康检查（源自 `192.168.1.103:20001`）确认了有文档记载的 `0x21040001`
   心跳与重复的 `0x21010130 = 0` 轴数据包到达运动主机的 UDP 43893 端口。机器人保持健康：
   10 秒内 2,000 个 `RobotState` 帧，200.1 Hz，`error_state = 0`。
3. 发送了一次标称 20 Hz 的 `+32767` 前向轴脉冲，最长一秒，带心跳与零轴清理。严格抓包中包含
   起始心跳、19 个间隔约 50--55 ms 的 `+32767` 轴包，以及最初两个零轴清理包。
   它固定的抓包数量是有意的，因此没有覆盖完整的清理区间。
4. 在同步的被动遥测抓取过程中，固件的 `goal_vel_forward` 从 0 上升到最高 0.5012，
   实测机体 x 向速度峰值为 0.7289 m/s，在录制边界处的世界平面位移为 0.4011 m。
   在全部 3,999 个抓取到的 `RobotState` 帧中 `error_state` 始终为 0；操作员观察到了前进、
   停止和一台稳定的机器人。随后 10 秒的被动抓取显示没有前进指令，`robot_motion_state = 0`，
   且 2,000/2,000 帧中 `error_state = 0`。

`+32767` 是厂商参考脚本的满量程轴值，不是一个有文档记载的"米每秒"设定。上面那些遥测数值是
一次被观察到的响应，不是给通用导航器用的标定。同步抓取在它的手柄状态遥测报告零值清理之前就结束了，
所以它无法确立一个精确的固件停止延迟；它只确立了发送端已开始清理，以及在后续抓取中机器人是停住的。

[`locomotion/lite3_control_mode_udp.py`](locomotion/lite3_control_mode_udp.py) 把模式选择限制在
有文档记载的 12 字节指令上。[`locomotion/lite3_axis_udp.py`](locomotion/lite3_axis_udp.py)
把轴发送端限制在有文档记载的前向满量程值或零值上，使用厂商的本地端口 20001，默认空跑，
实机使用时要求 `--operator-ready`，并在 `finally` 中发送零轴值。它的周期调度器在这次验证之后
做了修正，使发送开销不会累积进标称的 20 Hz 周期；那次修正有离线测试，但没有仅仅为了测量节奏
而在物理上重做一次。同一批离线测试还证明：一个失败的零值包不会中断清理区间 ——
后续的零值包仍会被尝试发送，失败会被暴露出来，套接字也会被关闭。

这次验证不允许重复、不允许更高数值、不允许横向/偏航指令，也不允许一次自主的视觉导航运行。
电机温度仍然不可获得；今后每一次会让腿动起来的测试，都需要新的、即时的操作员授权、
一条清空的通道、手持遥控/急停，以及一次新的健康数据抓取。

### 受配置文件门控的简单轴导航传输

厂商 V1.0.8 的《运动主机通信接口》定义了 moving/AI 的轴契约：

| 轴 | 代码 | 厂商定义的正方向 | 死区 |
| --- | ---: | --- | ---: |
| 前进/后退 | `0x21010130` | 向前 | `[-6553, +6553]` |
| 横向 | `0x21010131` | 向右 | `[-12553, +12553]` |
| 偏航 | `0x21010135` | 右转 | `[-9553, +9553]` |

它还规定了至少 20 Hz 的轴发送节奏、250 ms 的轴超时，以及以零轴值作为停止指令。
厂商提供的参考 GUI 以 20 Hz 重复被按住的轴、以 2 Hz 发送心跳，并在输入松开时发送零值。

[`locomotion/lite3_axis_locomotion.py`](locomotion/lite3_axis_locomotion.py) 是本仓库中针对
该接口、经过离线测试的传输实现。它刻意不是一个通用的原始轴命令行工具：一次实机运行需要一个
显式的本地轴配置文件。配置文件中每一个非零原语都必须携带证据引用、必须落在对应厂商死区之外，
并且必须覆盖每一个已启用的导航方向。默认不随包提供任何非零原语；见
[`locomotion/lite3_axis_profile.example.json`](locomotion/lite3_axis_profile.example.json)。

共享导航器的约定是横向/偏航以左为正，而厂商的原始轴是以右为正。传输层在边界处做一次这个反号。
它只有在拿到新鲜的 `RobotState` 和一条非零导航指令之后，才启动一路受配置文件门控的独立 20 Hz
轴数据流；它以 4 Hz 发送心跳，在 150 ms 指令 TTL 之后把所有轴清零，并在停止、失败和关闭时
持续发送零值。它既不改变控制/moving 模式，也不回退到传统的 320/325/321 速度指令。

#### ⚠️ 这个映射只有符号：下达的幅值被丢弃

**这就是为什么 issue #13 里的"步态下限"和"执行器增益"在这套传输上没有答案，
也是为什么联调工具自己长出了一个 `--locomotion-transport` 参数。** 步态下限是仍能走起来的
最低*下达*速度；而这里每个方向只有一条指令。执行器增益是交付值 ÷ 下达值；而这里分母根本
没有离开过笔记本。`gait_floor_probe.py` 和 `actuator_gain_probe.py` 现在会按名字拒绝
`--locomotion-transport axis`，因为拿一把递减的阶梯去测它并不会失败 —— 死区以上的每一级
都发出同一个原语、每一级都以同样的速度行走、每一项检查都通过，然后探测工具会把最低那一级
报告为步态下限。

这里被定义出来的，是每个原语实际交付的速度，也正是下面那道包络闸门要读的 `measured_m_s`。
[`commissioning/axis_primitive_probe.py`](commissioning/axis_primitive_probe.py) 测量它，
并会拒绝一个把机器人往错误方向推的原语。

一个配置文件为每个方向保存一个有证据支撑的原始值，于是 `map_velocity` 读取指令的*符号*，
并以满幅值发出那个原语。它从不缩放。后果是 `--derate` 和 `--max-vx` 在这套传输上到不了线上 ——
下面每一种设置发出的都是同一个原始轴值：

| `--derate` | 下达的 `vx` | 发出的前向轴值 |
| ---: | ---: | ---: |
| 1.0 | 0.300 m/s | `+32767` |
| 0.6 | 0.180 m/s | `+32767` |
| 0.3 | 0.090 m/s | `+32767` |
| 0.2 | 0.060 m/s | `+32767` |

这是设计中刻意的那一半：这套传输不会去发明一个它没有物理证据支撑的原始值，
而在一个有证据的 `+32767` 和一个无证据的零之间做插值，就是在发明它。速度包络改在别处强制执行。
一个配置文件可以声明每个原语被实测出来的速度：

```json
"measured_m_s":   { "forward_positive": 0.729, "lateral_negative": 0.31 },
"measured_rad_s": { "yaw_positive": 0.55 }
```

于是 `--live` 预检会在声明的速度超过 `--max-vx × --derate`（或对应的 `--max-vy` / `--max-wz`）
时**拒绝这次运行**，所以拿 `--derate 0.2` 去配一个实测 0.729 m/s 的原语，会停在闸门上，
而不是以安全否决所预设速度的 3.6 倍走起来。`measured_m_s` 对每一个已启用的线性方向都是
**必需的**，缺了它 `--live` 运行会被拒绝（issue #145）。自从规划器开始读它之后，
它就不再只是一个可选的包络交叉校验了：在这套传输上，腿实际产生的速度*就是*那个数字，
所以这个字段缺失时，传输层以上没有任何环节能说出机器人会走多快，规划器只能拒绝每一条指令，
而不是去猜。`measured_rad_s` 目前仍然只打印一条警告 —— 见下文。这两个字段都会与配置文件的
SHA-256 一起落进本次运行的遥测里。

⚠️ **`--max-vx` / `--max-vy` / `--max-wz` 在这个平台上没有默认值，缺了它们 `--live` 运行会被拒绝。**
它们是上面那道闸门的右侧，而在 2026-08-26 之前，它们默认取共享导航器的 `Limits` ——
前向 0.35 m/s、横移 0.20 m/s、偏航 0.70 rad/s，那是 **Unitree Go2** 装了机械臂之后的配置，
其中两个还是那台机器人实测的步态下限。这里面没有任何一项是在 Lite3 上测出来的。
现在它们在 `Lite3Bindings` 里与 `--robot-radius` 一起被置空，于是：

* 三个都没给的 `--live` 运行会被拒绝，而且拒绝信息会引用它拒绝继承的那些 Go2 数值；
* 空跑仍然会规划，用的还是那些 Go2 数值，但会先打印 `ENVELOPE NOT STATED`，
  点名这些数值并指向 issue #13；
* 遥测头记录 `platform.envelope_provenance`，这是唯一一个能告诉复核者
  "一次盖着 `deep-robotics-lite3-venture` 戳的运行，用的是有人为这台机器人选的数值、
  还是它继承来的数值"的字段。

`0` 是一个合法取值，表示禁用该轴 —— 部署 SOP 用的就是 `--max-vy 0`。

⚠️ **在策略驱动路径（`mappo_drive.py`）上还有第二条通道，声明上面那些参数并不能把它关掉。**
`--max-vx` 是一个*钳位*；真正被下达的是来自 [`policy/config.json`](../../../policy/config.json)
的 `max_vx_mps × command_scale`，而那个文件里的 `max_vx_mps` 0.35 / `max_vy_mps` 0.20
就是同一对 Go2 数值，被带进来时没有任何来源注释，而它旁边的每一个字段都有。
这里没有逐字段覆盖的办法 —— `--policy-config` 替换的是整个策略包配置，
而 `--policy-command-scale` 会同时缩放两个轴 —— 所以 `mappo_drive.py` 是告警而不是拒绝：
拒绝会逼着某个人去填一个看起来像样的 Lite3 数字，而那正是缺陷本身，不是修复。
这项测量归 issue #13。

**线性死区作用于向量，而不是每个轴。** `input_deadband.linear_m_s` 门控的是
`hypot(vx, vy)`，然后方位角会被吸附到这个映射能表达的八个 `(forward, lateral)` 符号组合中
最近的那一个。而按轴分别设死区，会丢掉较小的那个分量并把指令旋转掉：在随包交付的 0.05 m/s
死区下，`(0.049, 0.051)` m/s 是一条 46° 的指令，它只通过了横向那道门，于是前向被清零，
机器人就以满量程 90° 横移离开 —— 这与 #70 里 Go2 步态下限那一类失败是同一类。
吸附并不会让一条斜线变得精确；一条斜线被执行出来的方位角，是由两个原语各自的实测速度决定的，
而不是由指令决定的。它做到的是：不再让方向取决于某个分量恰好落在两个独立阈值中的哪一个之下。
偏航保留它自己的标量死区 —— 厂商给了它自己的死区，而且它不属于那个线性向量。

反面证据，明说出来：门控向量而不是门控每个轴，会让*更多*指令被执行，而不是更少。
`(0.040, 0.040)` m/s —— 45° 方向上的 0.057 m/s —— 过去在两个按轴的门下都通不过、什么也不发；
现在它会以满量程发出两个原语。因为映射只有符号，`input_deadband.linear_m_s` 并不是一个
"小指令过滤器"：它是**满速原语被点火时所对应的下达幅值**。请据此来设定它，
而不是按照"看起来可以忽略不计的速度"来设。

#### ⚠️ 规划器验证的是腿将会收到什么，而不是它自己写下了什么（issue #145）

上面那些段落描述的是传输层。这里讲的是它上面一层发生的变化，因为一个只有符号的映射
打破了共享规划器从未明说过的一个前提：它采样一组速度，把每一个向前推演，
并拒绝那些会停在某个东西里面的；而所有这些推演都假定腿收到的就是被采样的那个速度。
这里它们并没有。在 2026-08-27 Go2 那次僵住时的距离上回放 —— 距回收箱 0.72 m ——
一个以 0.05 m/s 慢挪的策略，`is_feasible` 验证的是 0.125 m 的行程，而腿实际走了 0.75 m。

`avoidance.Limits.transport` 是一个平台现在用来声明"我的腿拿一条指令会做什么"的地方；
`Lite3Bindings.transport_model` 在这套传输上回答 `SignOnlyAxisTransport`，在其它任何地方
回答 `PROPORTIONAL`，所以一次 Go2 运行和一次 Lite3 UDP 运行是逐位不变的。
它下游的一切 —— 间距、可行性检验、刹车距离上限、代价函数以及步态下限守卫 ——
问的都是**执行出来的**速度。没有任何东西去钳位一个幅值，因为根本没有幅值可钳：
GO 和 STOP 就是全部词汇，而这也正是 #26 的守卫从规划器一侧得出的同一个结论。

运行前值得知道的四个后果：

* **机器人每迈一步就承诺了"原语速度 × 预测时域"的距离** —— 0.30 m/s 配默认的 2.5 s
  就是 0.75 m。它在那么远的地方就做决定，而在同样的位置 Go2 会减速并保留选择余地，
  所以它会更早、更远地就 hold 住。这是正确的，而不是过度保守：它只有一个挡位。
* **由传输层造成的 hold 会说明自己。** `Command.transport_refusal` 携带一句同时包含两个数字的话 ——
  被请求的是什么，以及腿本来会收到什么 —— 而且它会结束这次运行，
  而不是把机器人晾在那里站着，这正是步态下限停止逻辑已经必须解决过的那个陷阱。
* **步态下限守卫现在判定的是执行速度。** 一台原语速度处于或高于其 `--gait-floor` 的机器人
  永远产生不了一个低于下限的 tick，所以守卫是沉默的 —— 这是诚实的答案，而不是守卫被关掉了。
  而一台原语*低于*声明下限的机器人，会在两秒钟寸步未行之后触发它。
* **偏航是同一个缺陷，而且没有被修好。** `input_deadband.yaw_rad_s` 是偏航原语以满量程点火时
  对应的角速度，而没有任何东西测量过那个角速度：只要 `Segment.yaw_change_deg` 还可能把一次
  超过 π 的转向报成反方向，原语探测工具就刻意不给偏航计时。所以规划器不会把转向与迈步组合起来 ——
  一段角速度未知的圆弧会终结在没人说得出的地方 —— 而是原地转向。一旦有了实测的
  `measured_rad_s`，这个限制会自动解除。

⚠️ **没有任何一台 Lite3 跑过上述内容。** 它是从传输层自己的代码以及对它的一次探测中读出来的。
一台真实机器人产生的数字取决于它自己的 `measured_m_s`；但论证的形状不会变。

非零轴还要求 `error_state=0`、有文档记载的力控 `basic_state=6`、`policy_state=0`、
一个配置文件允许的、有文档记载的步态状态，以及 `motion_state` 为 0 或 1。
manual/moving 状态由操作员确立；MAPPO 进程遇到非预期状态时会拒绝运行，而不是去切换它。

这套传输在能够支撑现有 MAPPO/规划器实现之前，需要经过验证的、主机本地的 `RobotState`。
只靠相机的 shadow 检测是有用的证据，但它不能替代里程计：共享的 RGB 路径会把检测结果投影到
世界坐标、在那里锁定目标与静态地图、在 MAPPO 输入中消费实测机体速度，
并用实际位移来做失速闸门。

### 有证据支撑的自定义静态配置

`--static-profile PATH.json` 选择一个自定义的静态颜色配置，且与随包交付的
`--static-prop bin` 配置互斥。一个自定义配置必须包含 schema `colour-profile/v1`、
有限的 HSV/形状阈值、用于单目测距的已知面板尺寸、覆盖整个障碍物的保守半径，
以及非空的证据引用。它的文件哈希与证据会被写进遥测；它的本地路径不会。
几何覆盖会被拒绝，所以一份经过复核的配置不会被命令行参数悄悄改掉。

这适用于在一个本来难以分割的障碍物上贴一块已知面板的场景。那块面板必须够大、够饱和、够稳定，
并且在真实的 Lite3 RTSP 画面中与所有背景物体明显可区分。它的视觉尺寸驱动测距，
而它的配置半径必须覆盖完整的物理障碍物加上有文档记载的测量不确定度。

## 在往机器人上装任何东西之前，先读懂这台机器人

把一台**新**机器人端到端地带起来 —— 机器人端预置、联调、标定、场景布置、实机运行、证据回收 ——
遵循 [`DEPLOYMENT-SOP.md`](DEPLOYMENT-SOP.md)，每台机器人走一遍。

把机器人和操作笔记本放到同一个隔离的 WiFi 局域网上 —— 让机器人无需拖线行走，
也让 Device Connect 仪表板能够发现它们 —— 遵循
[`DEMO-NETWORK_CN.md`](DEMO-NETWORK_CN.md)。在断定某个接入点在隔离客户端之前，
先读它那一节关于路由表的陷阱；2026-09-01 那次看起来正是那样，而事实并非如此。

`192.168.1.120` 是**运动主机**，而它本来就不应该有 ROS 2、`Lite3_ROS` 的检出或对外 DNS。
请从 [`commissioning/`](commissioning/README.md) 开始：它用标准库 Python 解码机器人本来就在
发送的状态，而且无法命令任何一条腿。一次由操作员用厂商遥控驱动的抓取，就能提供步态下限、
执行器增益、模式切换过程和角速度单位。

## 先把只读的那一半带起来

任何硬件作业之前先读 [`../../SAFETY_CN.md`](../../SAFETY_CN.md)。本节中的命令只检查话题
或命令行接线，不会让任何一条腿动起来。

在默认的 UDP 传输上没有话题要确认、也没有机器人核心要部署 —— 联调探测工具就是那个只读检查，
而且它会报告这套栈所依赖的帧率。对 `--locomotion-transport ros2`，请在感知计算机上使用官方的
`ros2-foxy` Lite3_ROS 分支，部署共享的 Device Connect 机器人核心以便
`arm_dc_robotkit.ros2_twist_locomotion` 可被 import，并确认那些有文档记载的话题：

```bash
ros2 topic info /cmd_vel
ros2 topic echo --once /leg_odom2
ros2 topic hz /leg_odom2

cd robot-stack/deep_robotics/lite3/visual_nav
python3 lite3_visual_nav.py --help
python3 calibrate_camera.py --help
python3 mappo_drive.py --help
```

Lite3 的命令行工具刻意不包含任何 D1 机械臂、锁存或 Go2 运动模式相关的参数。

相机来源必须显式给出，因为公开的桥和测试版感知手册都没有为每一台 Venture 的图像定义唯一端点。
例如 V4L2 用 `--camera-source 0`，RTSP 用 `--camera-source rtsp://HOST/PATH`，
或者给一条管线再加上 `--camera-gstreamer`。OpenCV 给一帧网络画面打的时间戳是它被解码的时刻，
而不是快门触发的时刻。请在实际安装的端点上测量端到端的画面年龄；否则一个 RTSP 解码队列
会让看上去很新鲜的画面其实很旧。

## 健康数据

**只有在数据流符合有文档记载的契约时，电量才是可用的。** 公开的 `Lite3_ROS` 把 `code=2305`
标识为 `RobotState`，其中包含 `battery_level`。在 2026-08-21 厂商服务重启之前，
这台 Venture 用一个不兼容的 212 字节包发出这个代码，解码器正确地拒绝了它。
重启后的服务发出的是有文档记载的 220 字节布局，报告电量 21%。解码器今后必须继续拒绝任何
布局不匹配的情况，而不是去猜偏移量；实机运行前请给机器人充电，
因为 21% 只比 20% 的中止阈值高一个百分点。

**电机温度是真的不存在。** 高层接口以任何形式都不携带它。底层的 `Lite3_MotionSDK` 会报告它，
但仅仅为了读一个温度就去取得底层控制权，会移除掉让机器人保持直立的厂商控制器，
所以本移植不这么做。受支持的发布方是一个厂商问题，仍未解决。

在它被回答之前，有两种运行方式：

| | 电机温度 | 能跑什么 |
| --- | --- | --- |
| 默认 | 必需 | 只能做空跑导航；`--live` 会被拒绝 |
| `--accept-no-motor-temperatures` | 不受监控 | 需要经过验证的新鲜电量；一次充满电的运行是有时长上限且被录制的 |

这个覆盖开关是一个明确的操作员决定，不是一种让闸门通过的手段：

- 电量和两秒的过期闸门仍然强制执行；
- `--max-seconds` 被限制在 120 秒；
- 预检会打印一条横幅，而 `warning_reason()` 每个 tick 都会重复它；
- 遥测记录 `motor_temperatures_monitored: false`，这样一份录制以后就不会被误认为是一次
  受监控的运行。

**它只约束一次运行，仅此而已。** 热量会在背靠背的多次运行中累积，而这里没有任何软件看得见。
请让机器人在两次运行之间冷却。不要用常量替换上述任何一项。

对 `--locomotion-transport ros2`，那两个配套话题仍然是必经之路：
`/battery_state` 上的 `sensor_msgs/msg/BatteryState`，其 `percentage` 采用 ROS 标准的 0..1
范围；以及 `/motor_temperatures` 上的 `std_msgs/msg/Float64MultiArray`，恰好 12 个摄氏度数值。
两个话题名都可配置。

## 测量并标定这台机器人

对两台活动机器人各自独立地做一遍，并把结果与它的机器人 ID 一起保存。
Go2 的数值在这里没有一个可以作为默认值。

1. 测量带负载的俯视规划半径，包括腿的活动范围以及为活动加装的任何东西。MAPPO 的 scale 是
   `robot_radius_m / 0.10`，其中 0.10 是检查点训练时的 VMAS 智能体半径。
2. 在通道清空、手持 App 急停的前提下，找出能产生持续步态而不是原地蹭动的最低前向下达速度。
   把那个保守可用的数值记录为 `--gait-floor`。
3. 在演示打算使用的那个确切指令包络下，用 `/leg_odom2` 得到的平均实测前向速度除以平均下达速度。
   把这个比值记录为 `--actuator-gain`；在选择 `--max-seconds` 时使用它。
   不要跨越步态下限做插值。
4. 标定安装好的 RGB 相机。静止标记物标定不会让机器人移动：

   ```bash
   python3 calibrate_camera.py --camera-source 0 \
       --marker MEASURED_CAMERA_TO_MARKER_M --out lite3_front_camera.json
   ```

   静态拟合写出的是一份 **provisional** 的 Lite3 标定。它可以用于 shadow/建图证据，
   但 `--live` 会拒绝它。在产出一份经复核的 `calibration_status=validated` 产物之前，
   请用第二个已知距离的观测独立校验焦距、光心高度和安装俯仰角。

   基于里程计的旋转拟合避免了距离测量，但它会让机器人动起来。只有在 Lite3 的偏航死区、
   健康数据、清空区域以及系绳/遥控方案都已明确之后，才运行它：

   ```bash
   python3 calibrate_camera.py --camera-source 0 --spin --spin-target marker \
       --spin-rate MEASURED_WORKING_YAW_RAD_S --live --operator-ready \
       --record lite3-camera-calibration.mp4 --out lite3_front_camera.json
   ```

旋转拟合使用的是里程计位姿偏航角的变化，而不是桥发布的瞬时角速度字段。联调期间，
仍然要把那个字段与位姿偏航角随时间的变化做对比：公开的桥把厂商的 `rpy_vel` 值原样拷进一个
ROS 字段而没有做单位换算，而对应的底层 SDK 文档写的角速度单位是度/秒。
在安装好的固件上确认这个单位之前，不要在下游使用实测偏航角速度。

## 运行共同的那套栈

### 在不移动的情况下录制实时感知

[`visual_nav/lite3_vision_shadow.py`](visual_nav/lite3_vision_shadow.py) 是机器人端部署的
第一级台阶。它只打开你给出的相机来源并运行现有的 MobileNet-SSD 检测器；
它不 import 任何行走、UDP、ROS 或厂商控制模块，也没有 `--live` 选项。
它的 JSONL 对凭据是安全的：它只记录相机来源的类型、帧元数据、像素空间的检测框和推理耗时。

在已预置的运动主机上，`/dev/video0` 已经被厂商的 GStreamer 发布进程占用。请消费它现成的
本地 RTSP 输出，不要去争抢那个 V4L2 设备：

```bash
release=$HOME/mappo-lite3-stage/releases/mappo-arm-cloud-physical-ai-lite3-20260825
export PYTHONPATH=$HOME/mappo-lite3-stage/python
python3 "$release/robot-stack/deep_robotics/lite3/visual_nav/lite3_vision_shadow.py" \
    --camera-source rtsp://127.0.0.1:8554/test \
    --model-dir "$HOME/mappo-lite3-stage/models/mobilenet-ssd" \
    --classes person,chair --seconds 60 \
    --output "$HOME/mappo-lite3-stage/evidence/vision-shadow.jsonl"
```

这只确认相机/模型/目标类别的感知。它不提供经过标定的距离、里程计、规划器输出、避障，
也不构成行走授权。

2026-08-25，这条命令在活动用的运动主机上对着现成的本地 RTSP 发布进程运行：10 帧 1280x720、
零次相机读取错误、MobileNet-SSD 平均推理 77.5 ms。在那个短样本中它没有检测到 `person` 或
`chair`；这是一次场景观察，不是相机或模型失败的证据。部署在 AArch64 上的源码通过了 284 项
不产生动作的测试：policy 33、integration 144、Lite3 locomotion 45、visual navigation 44、
commissioning 18。

先在不带 `--live` 的情况下运行感知与规划；它发不出任何非零速度：

```bash
python3 lite3_visual_nav.py --camera-source 0 \
    --calibration lite3_front_camera.json --waypoint 2.0 0.0 \
    --record lite3-dry.mp4 --telemetry lite3-dry.jsonl
```

只有在那些测量以及本仓库的仿真/shadow 阶梯全部完成之后，实机视觉导航器才会运行，
而且所有平台数值都必须显式给出：

```bash
python3 lite3_visual_nav.py --camera-source 0 --live --operator-ready \
    --calibration lite3_front_camera.json --gait-floor MEASURED_GAIT_M_S \
    --actuator-gain MEASURED_GAIN --robot-radius MEASURED_RADIUS_M \
    --max-vx CHOSEN_VX_M_S --max-vy CHOSEN_VY_M_S --max-wz CHOSEN_WZ_RAD_S \
    --waypoint 2.0 0.0 --record lite3-live.mp4 --telemetry lite3-live.jsonl
```

`CHOSEN_*` 是占位符，而且它们不叫 `MEASURED_*` 并非偶然：一个上限是一个决定，不是一次测量。
拒绝逻辑坚持的是：有人*为这台机器人*做出了那个决定。把 Go2 的 0.35/0.20/0.70 抄回来，
恰恰就是这道闸门存在的意义所在 —— 让它显形。

策略驱动使用同样的参数，再加上由半径推导出的策略 scale。保持规划器否决开启；
raw 模式仍然不合适：

```bash
python3 mappo_drive.py --camera-source 0 --live --operator-ready \
    --calibration lite3_front_camera.json --gait-floor MEASURED_GAIT_M_S \
    --actuator-gain MEASURED_GAIN --robot-radius MEASURED_RADIUS_M \
    --max-vx CHOSEN_VX_M_S --max-vy CHOSEN_VY_M_S --max-wz CHOSEN_WZ_RAD_S \
    --policy-scale RADIUS_DIVIDED_BY_0_10 --policy-mode supervised \
    --waypoint 2.0 0.0 --record lite3-drive.mp4 --telemetry lite3-drive.jsonl
```

这条路径会打印上文描述的 `policy/config.json` 告警。钳位是声明过的；
但它下面被下达的速度仍然是 Go2 的 `max_vx_mps`。

策略、遥测 schema、感知、跟踪、静态地图、规划器否决和闭环仿真器都是共享的。
同伴检测不是：除非给每台机器人一个显式的视觉标记物或颜色配置，
否则第二台 Lite3 仍然是不可见的。

## 离线检查

```bash
cd robot-stack/deep_robotics/lite3/commissioning
python3 test_lite3_state_probe.py && ruff check .

cd ../locomotion
for test in test_*.py; do python3 "$test"; done
ruff check .

cd ../visual_nav
for test in test_*.py; do python3 "$test"; done
ruff check .
```

随附的厂商测试版手册仍然通过 GitHub issue #12 链接，而不是拷贝到这里；
它的 LiDAR 和深度相机章节描述的不是这两台 Venture 机器人。
