<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS_CN.md

[English](AGENTS.md)

> 英文版 [AGENTS.md](AGENTS.md) 为权威版本；如两版有出入，以英文版为准。

给任何在本仓库中工作的编码代理的长期规则 —— Codex、Qwen Code、Claude Code 或其它。
如果你的工具不会自动加载本文件，请把它粘贴进你的第一条提示词。

面向人的配套文档：**[`CODING-AGENT-GUIDELINES.md`](CODING-AGENT-GUIDELINES.md)** ——
我们如何写提示词，以及为什么。读一遍即可。

## 这个仓库是什么

一个在仿真中训练出来的 MAPPO 策略，驱动真实的四足机器人在同一个房间里走向各自的目标点。
`robot-stack/` 负责感知与行走，`integration/` 把它的遥测映射成策略的输入，
`evidence/` 保存证明这一切的运行记录。从 `README.md` 开始；它是准确且最新的，
而且它记录了三处*并非*想当然的映射。

## 工作流程

1. **工作从一个 GitHub issue 开始。** 如果还没有，先写一个再写代码。
2. **工作以在那个 issue 上留下一条延续性评论结束** —— 结果数字、这次运行发现了什么、
   还有什么未解决，以及它在哪个 issue 中继续。即使是中途停下也要写；下一次会话是全新的上下文，
   而这条评论是唯一的交接。
3. **一个 PR 要携带证据**：一份运行日志、一个测试数量、一张测量表格，或者一段视频。
   "它应该能工作"不是一种状态。如果这个 PR 改变了 `README.md` 的状态表，请更新它。
4. **如实报告失败。** 如果一个测试失败了，就带着输出说出来。如果某一步被跳过了，也说出来。

## 在你说自己做完之前

每个测试文件会为每个测试打印一行 `  ok  <name>`，并给出一行 `<name>: N/N passed` 汇总。
请从仓库根目录、用 CI 所用的同一个脚本把全部测试跑一遍：

```bash
bash .github/measure-suites.sh            # run every suite
bash .github/measure-suites.sh --write    # ...and refresh .github/test-inventory.tsv
bash .github/measure-suites.sh --check    # ...and fail if it disagrees — what CI runs
```

`--write` 和 `--check` 需要 **Python >= 3.11**，并且能 import `numpy`、`opencv-python`、
`Pillow`、`pytest` 和 `aiohttp`；否则它们宁可拒绝运行，也不会写出一份因解释器够不到多少个
测试文件而短缺的测试清单。不带参数的 `bash .github/measure-suites.sh` 可以在 3.8 下运行，
遇到坏掉的测试文件同样会失败；它只是不去动那些数量。或者一次跑一个测试目录 ——
括号是有作用的，因为这些目录是嵌套的，一个裸 `cd` 会让下一行在上一行的目录里执行：

```bash
(cd policy && for t in test_*.py; do python3 "$t"; done)
(cd integration && for t in test_*.py; do python3 "$t"; done)
(cd detector/labels && for t in test_*.py; do python3 "$t"; done)
(cd detector/labels/pipeline && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/unitree/go2/visual_nav && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/unitree/go2/controller && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/unitree/go2/d1_arm && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/unitree/go2/lidar_sight && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/deep_robotics/lite3/locomotion && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/deep_robotics/lite3/visual_nav && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/deep_robotics/lite3/commissioning && for t in test_*.py; do python3 "$t"; done)
(cd robot-stack/preflight && for t in test_*.py; do python3 "$t"; done)
(cd dashboard && for t in test_*.py; do python3 "$t"; done)
```

**本文件里没有任何测试数量，把一个数量放回来会让构建失败。** 它们在
`.github/test-inventory.tsv` 里，每个目录一行、没有合计行 —— 因为这些数量曾经就写在这里，
直到它们让*本文件*在每一次新增测试的改动上都变成合并冲突：有一个晚上两个改动在这里撞上了，
其中一个不得不把自己的数量 diff 交给另一个去应用。一个自动生成的文件只有在你能重新生成它时
才有帮助，而上面的 `--write` 就是干这个的；永远不要手工往里面填数字。

**CI 会强制校验两半，而不是信任其中任何一半。**
[`.github/workflows/offline-checks.yml`](.github/workflows/offline-checks.yml)
用通配符发现每一个 `test_*.py` 和每一个 `ruff.toml`，重新测量测试清单里的每一个数量，
并在任意方向的不一致时失败；此外它还单独检查上面那些 `(cd …)` 行与测试清单中的目录列表
是同一个集合，所以一个会运行但没有记录在本文件里的测试目录是一个错误，
而本文件里一条什么都跑不出来的行同样是错误。那份列表只在有*目录*出现时才变，
而不是有测试出现时才变，这正是它可以写在正文里而那些数字不可以的原因。

然后在**每一个**带 `ruff.toml` 的目录内部执行 `ruff check .`。请把它们列出来，
不要相信这句话里的某个数字 —— 曾经写在这里的那个数字说"十三个"，而代码树里有十六个；
而那个禁止在本文件中出现数量的检查任务只检查代码块，所以一个用文字拼出来的数字它看不见：

```bash
git ls-files '*ruff.toml' | xargs -n1 dirname   # every directory to run it from
```

拿一个目录的配置去检查另一个目录的代码，正是一个 PR 一边声称"ruff 干净"、
一边交付了 13 处问题的原因。

**用 `--config` 给出的 `ruff.toml` 是相对于你所站的目录解析的，而不是它自己所在的目录。**
ruff 把 isort 的 `src` 锚定在项目根上，而在 `--config` 之下，项目根就是你的当前目录
（在普通的自动发现之下，项目根是配置文件自己所在的目录）。于是同一个模块，
从子目录里看是第一方的，从上层看是第三方的，同一个配置文件给出相反的判定：
`cd detector/labels/pipeline && ruff check . --config ../ruff.toml` 报出了八处 `I001`，
而任何一条有文档记载的命令都不会产生它们。`known-third-party` 按名字钉住那个模块；
CI 现在会从每一个受它管辖的目录、以及从仓库根目录运行每一份配置，并在这三者判定不一致时失败 ——
因为那正是"人会得到、而使用自动发现的 CI 自身 lint 流程不会得到"的一种判定结果。

还有八个目录里的 Python 完全没有任何 `ruff.toml` 覆盖：`visual_nav` 旁边的五个 Go2 目录，
以及三个 `evidence/` 运行目录。CI 会在一条警告里逐个点名，并在出现第九个时失败 ——
这个数量可以缩小，但不能在无人察觉的情况下增长。工作流里那道棘轮旁边的注释给出了这八个目录
各自的代价：它们之间一共 41 处问题，这也是它们目前仍是警告而不是闸门的原因，
而且它们没有一个是第三方引入、自动生成或第三方代码。

### 什么不被计入，以及为什么

- `robot-stack/unitree/go2/deploy/test_go2_robot_io.py` 会 import `arm_dc_robotkit`，
  而 `dashboard/test_robot_driver.py` 会 import `device_connect_edge`。这两个包在发布之前
  都不在 PyPI 上，所以它们都死在 `ModuleNotFoundError` 而不是死在某个测试上，
  而且今天在 `main` 上就是这么失败的。CI 会跳过它们，并用一条 `::warning::` 说明。
  **缺依赖不是通过，也不是回归 —— 要么把它装上，要么把它说出来。**
- `dashboard/` 的其它测试需要 **Python >= 3.11**，这也是 Device Connect 的要求，
  而这正是 `dashboard/drive_bridge.py` 是一个独立的 Python 3.8 进程而不是一次 import 的原因。
  CI 之所以同时跑一条 `3.8` 和一条 `3.11` 分支，正是因为这个：Go2 的 Jetson 是
  Ubuntu 20.04 / JetPack 5，所以机器人代码必须能在 3.8 下 import，
  而 `3.8` 那条分支除了 `test_drive_bridge.py` 之外会跳过 `dashboard/`。
- 如果你想要真实的数字，安装 `Pillow` 就不是可选项：没有它，
  `dashboard/test_camera_source.py` 里有三个测试会先打印 `  skip  ` 再打印 `  ok  `，
  这是一次缺依赖被读成了通过。`numpy`、`opencv-python` 和 `pytest` 也是同样的道理 ——
  没有 `cv2`，好几个 `visual_nav` 文件在 import 阶段就会失败。
- 这一段以前写的是 `33 / 189 / 329 / 17 / 39 / 16 / 139`，而且完全没有提到
  `detector/labels`、`detector/labels/pipeline` 或三个 Go2 目录。它那行 Lite3 联调按名字点了
  `test_lite3_state_probe.py`，而那个目录里有十个测试文件。

**`ruff --fix` 会排序 import，并把一句 `from avoidance import ...` 提到让它可被 import 的
那行 `sys.path` 之上。** 有两个测试文件就这样从通过变成了 `ModuleNotFoundError`，
而没有任何人动过一个测试。请把每一个 `sys.path.insert` 放在一个块里、置于任何同级 import 之前，
并在一次 lint 修复之后重新跑一遍测试。

然后对你自己的 diff 做一遍对抗性审查：软件工程最佳实践、错误、马虎或蛮力的算法、
不一致的风格、不正确的注释。把发现报告出来；不要默默改掉然后继续往下走。

问一问怎样才能让你新写的每一个测试*失败*，并确认它确实能失败。本仓库已经交付过一个锁存检查，
它通过断言机械臂的关节停止运动来"证明"机械臂被锁住了 —— 而一条没通电的机械臂本来就纹丝不动，
所以那个检查永远不可能失败。

## 硬件

- **`robot-stack/SAFETY.md` 管辖任何会让腿动起来的东西。** 它不是可选的。
- `--live` 是唯一会让机器人移动的参数。操作员始终守在遥控器上。
- **任何行走之前先问人。** 传感器和背装机械臂不需要许可；腿永远需要。
- 人是你了解这个房间的唯一传感器。运行之前，问清楚目标和障碍物在哪里，
  并请他们确认通道已清空。
- 对你产生的任何录制文件，请返回绝对路径，以便打开。

⛔ **在 Lite3 上一次网络改动可以夺走操作员的遥控手柄，而且没有任何东西会报告它。**
手柄是由机器人自己 `p2p0` 上的一个接入点提供服务的。两台机器人出厂时那个配置上都带着
`connection.autoconnect: no`，而 `/etc/netplan/config.yaml` 使用 `renderer: NetworkManager` ——
所以 **`netplan apply` 会让 AP 失效，而且它不会自己回来**。机器人通过以太网和 WiFi 仍然可达，
每一个服务都仍然是 `active`，唯一的症状是手柄找不到 SSID，而这看起来像射频故障，
而不像你一小时前那次地址改动的后果。它在两台机器人上各花掉一小时，相隔一天。

在对 `netplan`、`nmcli`、`wlan0` 或某个地址做任何改动之前和之后，都运行这条命令并做对比 ——
`p2p0` 必须显示 **`type AP`**：

```bash
iw dev | grep -E "Interface|type|ssid|channel"
```

如果 AP 掉了，请用**一条** `nmcli` 命令同时设置频段和信道来恢复它 —— nmcli 会校验整条连接，
所以一次只改一项会被拒绝，并给出一条有误导性的 `'36' is not a valid channel`：

```bash
sudo nmcli con mod myap50G 802-11-wireless.band bg 802-11-wireless.channel 10
sudo nmcli con mod myap50G connection.autoconnect yes connection.autoconnect-priority 60
sudo nmcli con up myap50G
```

信道必须与场地路由器的固定信道一致：一路射频同时服务 AP 和 station，而驱动只允许
`#channels <= 1`。完整细节，包括为什么 2.4/5 GHz 的分离方案会被拒绝，见
[`robot-stack/deep_robotics/lite3/DEMO-NETWORK_CN.md`](robot-stack/deep_robotics/lite3/DEMO-NETWORK_CN.md)。

## 永远不要在机器人上安装任何东西

这条规则就是写给你这个读者的。一个照着作业手册操作的操作员，比一个在晚上 11 点对着
`ImportError` 即兴发挥的代理*更不容易*把事情搞成一团糟 —— 而 2026-08-26 有两个代理同时
SSH 在一台带电的机器人上。

- **永远不要在机器人上于虚拟环境之外 `pip install`，永远不要装进系统 Python。**
  厂商栈在这些机器上是装在一个 venv 里的；系统解释器是所有其他用户、所有厂商工具和机器人上
  每一个 ROS 节点共用的，而且没有任何 `uninstall` 能把一个被遮蔽掉的厂商包放回去。
  一个 venv 可以删掉重建。
- **如果一个 import 在机器人上失败了，那是一个需要报告的发现，不是一个需要补装的依赖。**
  激活 venv 再跑同一条命令。如果它在 venv 里仍然失败，就带着输出把它写进 issue。
  不要把它弄消失。
- **永远不要在机器人上装一个更新的 Python，也不要指望用虚拟环境搞到一个。**
  venv 是*从*一个解释器构建出来的，它给不了这台机器上没有的版本。
  `device-connect-edge` 在机器人之外运行正是因为这个，而这个拆分是刻意的，
  不是一个等着被修的打包 bug。
- **现在它是拒绝，而不是警告。** `visual_nav --live` 和 `dashboard/drive_bridge.py` 在打开
  任何传输之前都会调用 [`robot-stack/preflight/venv_guard.py`](robot-stack/preflight/venv_guard.py)，
  而它是抛异常而不是打印。不要绕过它，也不要在一台本身就是机器人的机器上设
  `MAPPO_ROBOT_HOST=0`。
- **用 `deploy/push-to-robot.sh` 部署，这样一次运行就能说出自己的 commit。**
  部署出去的那些代码树没有一个是 git 检出 —— 没有 `.git`，所以既没有分支也没有 commit ——
  因此 `robot-stack/preflight/tree_stamp.py` 记录的是 git 自己的根 tree id，
  由磁盘上的字节重新计算得出，而 `mappo_drive.py` 会在代码树与之不再匹配时拒绝运行。
  每一次运行都会把 `commit … tree …` 作为第一行打印出来；请引用那个 id，
  而不是"从 main 部署的"。
  ⚠️ **这里以前的一个说法是错的，而且值得知道**：曾经说 `~/mappo-run` 与*任何单个 commit
  都不匹配*。它其实与 `cb42b9a` 精确匹配，226/226 个文件 —— 那是一个**未合并分支**的顶端，
  这也正是拿它去对着 `main` 重建什么都找不到的原因。真正是混合体的那棵树是 `~/mappo-main`，
  它横跨 21 个 commit、落后 HEAD 8 到 83 个提交，而它正是启动脚本 source 的那一个。
  不要相信其中任何一个说法，自己重新推导：`python3
  robot-stack/preflight/tree_stamp.py id <dir>` 会为任意目录打印出一个真实的 git tree id。

以上这一切背后的路径、解释器、venv 创建命令和测量结果，都在
**[`deploy/README.md`](deploy/README.md)**（Go2，以及 Device Connect 的拆分）和
**[`robot-stack/deep_robotics/lite3/DEPLOYMENT-SOP.md`](robot-stack/deep_robotics/lite3/DEPLOYMENT-SOP.md)**
（Lite3，双语）里。它们不在这里重复 —— 本节讲的是你不能做什么，那两份讲的是该怎么做。

## 永远不要提交、记录或写进 issue

机器人 SSH 密码、WiFi PSK、API 令牌。改为引用一个本地的、未被跟踪的文件
（`~/.robot-creds`）。事后去清洗一个 issue 并不能把它从事件日志里删掉。

不要链接带 `?token=` 的附件 URL —— 它们会过期。先合并，再链接默认分支上那个永久的 raw URL。

## 命名

本仓库与**云深处（Deep Robotics）**以及 Arm 的中国团队共享 —— 而且是在底层标准公开发布*之前*
就共享的。

- 产品名是 **Arm Device Connect**，在"Arm"显得多余或冗长时用 **Device Connect**。
  缩写是 **DC**。
- **它此前的任何内部名称、以及任何原始公司名称，都不得出现在本仓库的任何地方** ——
  不得出现在代码、注释、文档、文件名、分支名、提交信息，或 issue 与 PR 的文字里。
  如果你发现了一个，请删掉它并说明。不要假定它是被故意留下的，
  也不要因为从上游仓库拷贝文字而重新引入一个。
- 拿不准就问 @armwaheed，不要猜。这条规则是绝对的；它不是风格偏好。

保持测量表格自成一体：内嵌的 GIF 和 `user-attachments` 视频链接在中国大陆很慢或根本打不开，
所以论证必须在没有它们的情况下依然成立。

## 风格

与周围的代码和文字保持一致。`README.md` 为文档定下基调：具体、有实测、并把反面证据写出来。
宁可给一个数字，也不要给一个形容词。
