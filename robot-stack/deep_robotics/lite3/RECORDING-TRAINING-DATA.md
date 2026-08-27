<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recording Lite3 training footage — what to pass, and what makes a clip worthless
# 录制 Lite3 训练素材 —— 该加哪些参数，以及什么会让素材作废

**EN** — This is for whoever holds the Lite3s. It exists because the first clip we
received (2026-08-27, Shanghai, 60 s) could not be trained on, for two reasons that are
both one flag or one instruction away from being fixed. The full measurement is in
[`evidence/2026-08-27-lite3-pov-clip-audit/`](../../../evidence/2026-08-27-lite3-pov-clip-audit/);
this page is only what to do differently.

**中文** —— 本文面向持有 Lite3 的团队。之所以写它，是因为我们收到的第一段素材
（2026-08-27，上海，60 秒）无法用于训练，原因有两个，而两个都只差一个参数或一句嘱咐就能解决。
完整测量记录见
[`evidence/2026-08-27-lite3-pov-clip-audit/`](../../../evidence/2026-08-27-lite3-pov-clip-audit/)；
本文只讲下次要怎么改。

---

## 1. Pass `--record-raw`. This is the single most important line on this page
## 1. 必须加 `--record-raw` —— 这是本文最重要的一句

**EN** — `--record` writes the **annotated** canvas: the plan-view radar inset, the
bottom-left status plate, and an orange box around every detection the network made. That
orange box is the label burned into the pixels the model would have to learn from. A
network trained on those frames learns "a peer comes with an orange rectangle."
`--record-raw` writes the same frames before anything is drawn on them. Pass **both** —
`--record` stays useful for humans to watch.

**中文** —— `--record` 写出的是**带标注**的画面：右上角的雷达图、左下角的状态条，以及网络每次
检测都会画上的橙色框。那个橙色框相当于把标签直接烧进了模型本该去学习的像素里。用这种画面训练，
网络学到的是"同伴机器人总是自带一个橙色方框"。`--record-raw` 写的是**任何标注绘制之前**的同一帧。
请**两个都加** —— `--record` 仍然方便人工回看。

```sh
python3 lite3_visual_nav.py \
    --calibration lite3_front_camera.json \
    --record      peer_NN_annotated.mp4 \
    --record-raw  peer_NN_raw.mp4 \
    --telemetry   peer_NN.jsonl
```

> **EN** — Both files are advanced by the same gate, so frame *n* of one is frame *n* of
> the other, and both join to the same `perception.video_frame` in the telemetry. **Send
> the `.jsonl` too, always.** It carries the `sightings` that turn a frame into a labelled
> frame, and it is not recoverable from the video afterwards.
>
> **中文** —— 两个文件由同一个门控推进，所以一个文件的第 *n* 帧就是另一个文件的第 *n* 帧，
> 并且都能通过 `perception.video_frame` 与遥测对齐。**`.jsonl` 也请务必一并发回。**
> 它携带的 `sightings` 才能把一帧画面变成带标注的一帧，事后无法从视频里还原。

⛔ **EN** — If a clip was already recorded with only `--record`, do not delete it — but do
check whether its `.jsonl` still exists on the robot. The telemetry alone can label frames
we already have.

⛔ **中文** —— 如果某段素材当时只用了 `--record`，不要删除；但请检查机器人上对应的 `.jsonl`
是否还在。仅凭遥测就能给我们手上已有的画面打标签。

---

## 2. The robot must MOVE. A parked camera is one photograph
## 2. 机器人必须**走动**。相机不动，等于只拍了一张照片

**EN** — The 2026-08-27 clip is 324 frames over 60 s, and the camera's total movement
across all of it is a median of **0.19 px**, maximum 7.14 px. Not one frame moved more than
10 px. The status line said `DRY RUN STANDING(sim)` throughout: the legs were never
enabled. 324 frames of a fixed camera are **one viewpoint**, not 324 samples — and no
amount of cropping, shearing or colour filtering in the training pipeline creates a second
one. Augmentation multiplies examples; it does not add information.

**中文** —— 2026-08-27 那段素材是 60 秒 324 帧，而相机在整段里的位移中位数只有 **0.19 像素**，
最大 7.14 像素，没有任何一帧超过 10 像素。状态条全程显示 `DRY RUN STANDING(sim)`：腿从未使能。
固定机位的 324 帧只是**一个视角**，不是 324 个样本 —— 训练流程里再多的裁剪、错切、调色也变不出
第二个视角。数据增广只能把样本数变多，不会增加信息量。

| do / 要做 | why / 原因 |
| --- | --- |
| Carry or drive the camera robot around the peer, in an arc and in and out. / 抱着或驱动带相机的机器人绕着同伴走，走弧线，也要有远近变化。 | Bearing and range are the two axes the detector has to generalise over. / 方位角和距离是检测器必须泛化的两个维度。 |
| Cover the peer's own heading through all 360°. / 让同伴机器人的**朝向**覆盖 360°。 | A quadruped is 0.61 m long and 0.37 m wide — head-on and broadside differ by 1.6x in apparent size. / 四足机器人长 0.61 m、宽 0.37 m —— 正面与侧面的视觉尺寸相差 1.6 倍。 |
| Record on **more than one day**, and in more than one room. / 在**不止一天**、不止一个房间录制。 | Same-session data proves nothing; this project has measured 0/705 same-session against 60/159 cross-day. / 同场次数据什么也证明不了；本项目实测同场次 0/705，跨天 60/159。 |
| Record the same spaces with **no peer present**. / 在**没有同伴机器人**的同一场地各录一段。 | This is what kills false positives, and it is the half people skip. We currently have **zero** in-domain negatives. / 这是消除误报的关键，也是最常被跳过的一半。我们目前**一个**同场景负样本都没有。 |

---

## 3. Calibrate the camera FIRST — nothing can be rendered without it
## 3. 先做相机标定 —— 没有它就无法生成合成数据

**EN** — This is Step 0 of [`detector/COLLECTING-LITE3-DATA.md`](../../../detector/COLLECTING-LITE3-DATA.md)
and it has still not been done. The synthetic half of the dataset is rendered by
`render_lite3.py`, whose `--focal-px` **defaults to a Go2's 1290.2 px**. A Lite3 Venture is
a different lens, and every apparent size scales linearly on that number: render at the
wrong focal length and the detector learns a scale prior wrong by exactly that ratio.

**中文** —— 这是 [`detector/COLLECTING-LITE3-DATA.md`](../../../detector/COLLECTING-LITE3-DATA.md)
的第 0 步，至今仍未完成。数据集的合成部分由 `render_lite3.py` 生成，而它的 `--focal-px`
**默认值是 Go2 的 1290.2 px**。Lite3 Venture 用的是另一颗镜头，所有视觉尺寸都与该数值成正比：
焦距填错，检测器学到的尺度先验就会按同样的比例错下去。

```sh
python3 calibrate_camera.py --spin --live --object-class person \
        --spin-rate 0.8 --spin-max-yaw 35 --start-delay 20 \
        --record calib.mp4 --out lite3_front_camera.json
```

**EN** — Send back `lite3_front_camera.json` **and** the camera's optical-centre height
above the floor with the robot standing, measured with a tape. The height cannot be
recovered from the video and is the second render parameter.

**中文** —— 请把 `lite3_front_camera.json` **以及**机器人站立时相机光心距地面的高度
（用卷尺实测）一并发回。这个高度无法从视频反推，它是第二个渲染参数。

⚠️ **EN** — Calibrate in the posture you deploy in, and check the achieved yaw: at
0.30 rad/s the robot barely turns and the first Go2 sweep produced 6.7° of yaw and an
unconstrained fit.

⚠️ **中文** —— 用实际部署时的姿态做标定，并检查实际转过的偏航角：在 0.30 rad/s 下机器人几乎不转，
Go2 第一次扫描只转了 6.7°，拟合完全不收敛。

---

## 4. What a good set looks like
## 4. 一份合格的素材应该是什么样

| dimension / 维度 | cover / 覆盖范围 | why / 原因 |
| --- | --- | --- |
| range / 距离 | 0.4 – 4 m, densest under 1.5 m / 0.4 – 4 m，1.5 m 以内最密 | Where avoidance happens, and where synthetic data is weakest. / 避障就发生在这个范围，也是合成数据最薄弱的地方。 |
| bearing / 方位 | all 360° of the peer's heading / 同伴朝向的完整 360° | Head-on and broadside are a 1.6x difference in apparent size. / 正面与侧面视觉尺寸相差 1.6 倍。 |
| truncation / 截断 | plenty under 1 m / 1 m 以内要多拍 | At avoidance range the peer is usually clipped by the frame edge. / 在避障距离上，同伴通常会被画面边缘切掉。 |
| lighting / 光照 | window backlight, shadow, overhead / 窗边逆光、阴影、顶灯 | The blown-out window end is the hardest condition we have. / 过曝的窗边是我们遇到过最难的场景。 |
| negatives / 负样本 | the same spaces with no peer / 同一场地、无同伴机器人 | Kills false positives. We have zero. / 用于消除误报。我们目前一个都没有。 |
| sessions / 场次 | at least two different days / 至少两个不同的日子 | A test sharing conditions with what it tests proves nothing. / 与被测对象共享条件的测试，什么也证明不了。 |

**EN** — Volume: a few hundred frames **containing genuine viewpoint change** is worth more
than ten thousand from a tripod. Awkward, half-occluded, badly-lit frames are the valuable
ones; perfect framing is not needed.

**中文** —— 数量方面：几百帧**真正有视角变化**的画面，价值远高于固定机位拍的一万帧。构图别扭、
被部分遮挡、光线差的画面才是有价值的；不需要追求完美构图。

---

## 5. What to send back
## 5. 需要回传的内容

**EN** — 1. `lite3_front_camera.json` and the measured camera height in metres.
2. The `_raw.mp4` / `.jsonl` pairs — **both**, for every clip. 3. A note saying which
building and lighting each run was, and on which date.

**中文** —— 1. `lite3_front_camera.json` 以及实测的相机高度（米）。
2. 每段素材的 `_raw.mp4` / `.jsonl` **配对文件，两个都要**。
3. 一份说明：每段分别是哪栋楼、什么光照、哪一天录的。

**EN** — Labelling is on us. You do not need to draw a single box.

**中文** —— 打标签由我们负责，你们不需要画任何标注框。
