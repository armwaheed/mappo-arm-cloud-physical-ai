<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Collecting Lite3 data — handoff

**This is for whoever has the Lite3s.** The demo ships on Lite3-avoiding-Lite3, and the
synthetic half of the dataset is already built. What is missing is footage of a real Lite3
through a real Lite3's camera, and only a team with two Lite3s can produce it.

Everything below is unblocked by anything happening in Seattle.

---

## Step 0 — calibrate the camera FIRST, before recording anything

Not optional and not reorderable. **Every range this stack produces is proportional to one
focal length**, and the synthetic training data has to be rendered at *your* value, not
ours. The numbers in this repository — 1290.2 px, 85.27° HFOV — belong to a Go2's front
camera. A Lite3 Venture is a different lens.

```sh
python3 calibrate_camera.py --spin --live --object-class person \
        --spin-rate 0.8 --spin-max-yaw 35 --start-delay 20 \
        --record calib.mp4 --out lite3_front_camera.json
```

The robot turns and uses its own yaw odometry as the angular ruler — no tape measure, no
size prior, no fiducial. Someone stands ~2.5 m in front and **stays still**; `--start-delay`
exists because the operator is usually the target and needs time to walk into place.

Four things that cost us real time on the Go2 and will cost you the same:

* **`--spin-rate 0.8`.** At 0.30 rad/s the robot barely turns — 7–14% of commanded — and
  the first sweep produced 6.7° of yaw and an unconstrained fit. Check the achieved yaw.
* **Calibrate in the posture you deploy in.** Standing vs prone moves the camera enough to
  change what is even visible: a 0.25 m object at 0.5 m filled the frame prone and fell
  *below* it standing.
* **A high residual can be the TARGET, not the lens.** A person's box centre wobbles ~70 px,
  which is ~3° on its own. `spin_fit_quality()` decomposes it — check that before
  concluding the model is wrong.
* **Calibration displaces the robot** ~0.2 m per sweep. Re-stage between runs.

**Also measure and write down the camera's optical centre height above the floor when the
robot is standing.** It is the second render parameter and there is no way to recover it
from the video.

Send back `lite3_front_camera.json` and that height.

---

## Step 1 — record the peer

One Lite3 **parked**, the other carrying the camera. Park it: the policy has no
obstacle-velocity channel, so the first runs are a static peer, and moving footage is worth
much less until that changes.

```sh
python3 visual_nav.py --calibration lite3_front_camera.json \
        --record peer_NN.mp4 --telemetry peer_NN.jsonl
```

No `--live` needed — the robot does not have to walk to record useful frames, and a
hand-carried or slowly-driven camera covers the space faster.

### What the set has to span

The synthetic generator already covers scale and truncation, so what recorded footage adds
is **real appearance** — real specular highlights on brushed metal, real corridor lighting,
real motion blur. Bias toward the conditions the renderer cannot fake.

| dimension | cover | why |
| --- | --- | --- |
| **range** | 0.4 – 4 m, densest under 1.5 m | where avoidance happens and where the synthetic set is weakest |
| **bearing** | all 360° of the PEER's heading | a quadruped is 0.61 m long and 0.37 m wide — head-on and broadside are a 1.6x difference in apparent size, and that is the single largest ranging error |
| **truncation** | plenty under 1 m | at avoidance range the peer is usually clipped by the frame edge; a set of whole robots omits the case that matters |
| **lighting** | window backlight, shadow, overhead | the corridor's blown-out window end is the hardest condition we have |
| **negatives** | the same spaces with **no** peer | this is what kills false positives, and it is the half people skip |

Record the negatives. Our existing detector confidently calls a corridor cabinet a `train`
at 0.97; without in-domain negatives a new class will do the same thing.

### What NOT to bother with

* Poses the robot cannot hold — no need to lift or tilt it.
* Distances beyond ~5 m. Recall thins and the policy's horizon is 0.875 m anyway.
* Perfect framing. Awkward, half-occluded, badly-lit frames are the valuable ones.

**Volume:** a few hundred frames containing the peer is enough given the synthetic half and
a frozen backbone. This is one rigid object in known buildings, not an open-world class.

---

## Step 2 — send back

1. `lite3_front_camera.json` + the measured camera height (metres).
2. The `.mp4` / `.jsonl` pairs. **Both** — the telemetry carries the pose that makes a
   frame interpretable, and the console log carried pose once in 107 ticks, so the jsonl is
   not redundant.
3. A note on which building/lighting each run was.

Labelling is on us: one frame per clip by hand, propagated with a tracker.

---

## What happens to it

Merged with the synthetic Lite3 set, re-rendered against **your** focal length and camera
height:

```sh
python3 render_lite3.py --mjcf .../Lite3/mjcf/Lite3.xml --backgrounds your_negatives/ \
        --out lite3_ds --count 2000 \
        --focal-px <yours> --camera-height-m <yours>
```

Then `add_class.py` grows the shipped MobileNet-SSD from 21 to 23 classes and the fine-tune
runs with the backbone frozen. `person` is preserved by construction — its features and
logits are untouched, only the softmax denominator moves (measured: 0.928999 → 0.928962).

**You get weights back, not a calibration.** A detector outputs boxes and transfers across
cameras; ranging does not. Each unit keeps its own `lite3_front_camera.json`.

---

## Two things to know before you plan around this

**A detector is not yet peer avoidance.** Until the routing change lands, a detected peer
is a track, and every track was dropped before the policy saw it — arriving as one boolean
meaning *stop*. Detection and routing have to arrive together or nothing changes.

**The policy has no obstacle-velocity channel.** An 18-value observation carries the
robot's own state, the goal offset and 12 proximity rays. A moving peer enters as an
instantaneous disc. That is why the first runs are a parked peer, and why anything above
0.25 m/s still stops the robot rather than being handed to the policy.
