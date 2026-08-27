<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera geometry, per robot / 各机型相机几何参数

Everything the ranging chain needs, and — as important — what is **measured** versus what is
**assumed**. A range is a division by a focal length; a focal length nobody measured makes
every range downstream a guess wearing three decimal places.

一切测距链路所需的参数，以及同样重要的：哪些是**实测值**，哪些是**假设值**。

## Unitree Go2 — `robot-stack/unitree/go2/visual_nav/go2_front_camera.json`

| field | value | provenance |
|---|---|---|
| `focal_px` | 1290.16 | **measured** — `method: "spin"`, 53 samples, `residual_deg_rms` 3.13, `yaw_span_deg` 80.05 |
| `width` x `height` | 1920 x 1080 | measured |
| `height_m` | 0.32 | measured, standing |
| `pitch_rad` | 0.0 | measured — the Go2's front camera is level |
| `hfov_deg` | 85.27 | **nominal.** Note it does not follow from `focal_px`: 1290.16 at 1920 px implies **73.31°**. Treat `focal_px` as the measurement and `hfov_deg` as the spec sheet. |

## Deep Robotics Lite3 — measured 2026-08-27, Arm Shanghai

| field | value | provenance |
|---|---|---|
| `height_m` | **0.37** standing · **0.115** prone | **measured** by Timo, 2026-08-27 |
| `pitch_deg` | **~11°**, fixed | **measured** — the mount is not adjustable, so this is a constant, not a per-run reading |
| `focal_px` | — | ⛔ **NOT MEASURED.** No spin calibration has been run on a Lite3. |
| `width` x `height` | 1280 x 720 | from the recordings |

**The prone height is not a footnote.** At 0.115 m the camera is roughly a third of its
standing height, so the ground-plane geometry — and every range read off a floor contact
row — changes completely between postures. Any ranging that assumes one height is wrong in
the other.

**两个高度都要记录。**趴下时相机高度约为站立时的三分之一，地平面几何完全不同，
因此任何假设单一高度的测距在另一种姿态下都是错的。

### ⚠️ What the 2026-08-27 recordings actually carried

The six `lite3-raw-shadow-*` recordings embed this camera block:

```json
{"focal_px": 469.63, "height_m": 0.4, "hfov_deg": 156.16, "width": 1280, "height": 720}
```

Three problems, and they compound:

1. **`height_m` 0.40 contradicts the measurement above (0.37).**
2. **No `pitch`** at all, where the real mount sits at ~11°.
3. **`focal_px` and `hfov_deg` imply different lenses** — 469.63 at 1280 px is a 107.46°
   field; 156.16° would need a focal of 135.1 px. They disagree by **48.7° / 335 px**.
   Unlike the Go2's block, it carries no `method`, `samples` or `residual_deg_rms`, so
   there is no way to tell which — if either — was measured.

**Therefore every `range_m` in those six telemetry files is derived from a calibration that
does not describe this camera.** The `box` fields are pixel measurements and are fine; the
ranges are not. Do not fit, train, or threshold against them.

**因此这六个遥测文件中的 `range_m` 均不可信**（`box` 像素坐标可用）。

### What is still needed

A spin calibration on a Lite3, producing `focal_px` with `method`, `samples` and
`residual_deg_rms` the way the Go2's did. Until then `render_lite3.py --focal-px` falls
back to a **Go2's** 1290.16 — a different lens on a different sensor at a different
resolution — and synthetic data built on it places objects at the wrong scale.

在 Lite3 上完成一次 spin 标定，得到带 `method`/`samples`/`residual_deg_rms` 的
`focal_px`。在此之前，合成数据的物体尺度是错的。
