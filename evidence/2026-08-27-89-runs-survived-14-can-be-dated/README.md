<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 2026-08-27 — 89 runs came off the Go2, and 14 of them can be dated

The lab Go2 was going offline. What came off it is **89 unique runs, 9,117 control ticks
and 4,624 detections**, plus 84 videos — 544 MB, recorded over an unknown fortnight in
Arm office corridors. This directory is the inventory, the measurement of what that
corpus can still prove, and the record of what was published and what was not.

The headline is a limit, not a result. **Only 14 of the 89 runs carry a clock that can be
converted to a date.** The other 75 are not undated by oversight; they are undatable from
the bytes that survive, and the reason is worth knowing before anyone builds a split.

## Reproduce

```bash
cd evidence/2026-08-27-89-runs-survived-14-can-be-dated

# the inventory and the recall tables. Standard library only, runs on the 3.8 leg.
python3 inventory.py --corpus /path/to/robot-pull      # -> inventory.tsv, failure-modes.json

# the walk-3 numbers, at both input sizes. Needs cv2 and the stock weights.
python3 rescore.py --frames /path/to/walk3-frames --model-dir /path/to/models
python3 rescore.py --frames ... --model-dir ... --subfloor   # prototxt floor 0.25 -> 0.01
```

Committed here and small: `inventory.tsv` (89 rows, 7 KB), `visual-truth.tsv` (250
hand-marked frames), `failure-modes.json` and `walk3-rescore.json`. The corpus itself is
not committed, for the reason PR #138 gives — bulk robot data belongs in the dataset
store, keyed back to a manifest. Where it went is at the bottom of this file.

## Why 75 runs cannot be dated

`wall_time` is **two different clocks wearing one name**.

| | runs | `wall_time` is | can be dated |
|---|---:|---|---|
| tonight's runs, and six from 2026-08-11 | **14** | epoch seconds | **yes** |
| everything else with a stamp | **69** | the robot's monotonic uptime | no |
| six joint-telemetry logs | **6** | absent | no |

A run stamped `327832.73` is 3.79 days after a boot whose date nothing in the corpus
records. The obvious fallback fails too: **the rsync that pulled the corpus did not
preserve mtimes**, so all 179 pulled files carry the pull time instead — 2026-08-26
20:30:24 to 20:36:42 local, a 378-second window that is when rsync ran, not when anything
was recorded. The second fallback also fails: `ffprobe` reports **no `creation_time`** on
any of the 84 MP4s. And `logs/` contains no date string of any form — checked with
`grep -rhoE '20[0-9]{2}-[01][0-9]-[0-3][0-9]'`, which matches nothing.

What survives is **order, not date**. All 69 uptime stamps fall in a single monotonic run
from 3.78 d to 14.94 d with no backward step, so the robot did not reboot across them and
they span **268.0 hours — 11.2 days — of continuous uptime**. That is enough to separate
sessions by many hours, which is what a cross-day split actually needs; it is not enough
to put a calendar date on any of them.

⚠️ **So this corpus cannot supply a labelled cross-day split by date.** It can supply a
cross-*session* split by uptime gap, and that is a weaker claim which must be written as
the weaker claim. The corpus that already carries a genuine cross-day holdout is
`aug20_crossday`, and it stays the one to evaluate against.

## Six code generations, and the one PR #138 describes

Runs differ in which telemetry fields the writing code emitted, which sorts them into six
generations by feature accretion. This is the only provenance most of them have.

| gen | runs | emits | 
|---|---:|---|
| G0 | 11 | no `obstacles` array at all |
| G1 | 6 | `obstacles`, without `id`/`kind` |
| G2 | 32 | `id`/`kind`, still no `sightings` |
| **G3** | **16** | **`sightings` — the tree PR #138 preserved** |
| G4 | 9 | + `person_shaped` on each obstacle |
| G5 | 8 | + a `profile` on each tick |
| — | 7 | not nav telemetry (6 joint logs, 1 arm-latch log) |

G3 is pinned to the manifest by exclusion, not by assertion: the preserved tree emits
`sightings` (`robot-stack/unitree/go2/visual_nav/telemetry.py:176`) and contains the
string `person_shaped` **nowhere in its 163 files**, which is exactly G3 and no other row.

The consequence is the part to carry forward:

- **16 runs (G3)** can be keyed to
  [`evidence/2026-08-27-what-the-robot-was-running/manifest.tsv`](../2026-08-27-what-the-robot-was-running/manifest.tsv)
  — 134 files, 15 commits, 2026-08-11 → 2026-08-19. That manifest names a *tree*, not a
  run, so this is the identity of the code that was on disk, not proof that a given run
  executed it.
- **49 runs (G0–G2)** predate that tree. Nothing in this corpus or the repository
  identifies the code that produced them.
- **17 runs (G4, G5)** postdate it — they emit fields the manifest's tree cannot produce.
  **The manifest does not describe the code that produced tonight's runs.**

Only the 6 tonight runs under `telemetry/` carry a stamped tree id in their filename
(`20260827T010144Z-00319b6`). Every other run in the corpus, all 83, has no commit
identity beyond the generation above.

## The detector drops three tracked people in ten

Ground truth for "a person was there" is the hard part, and hand-labelling 9,117 ticks was
not on. **Track continuity** supplies a denominator without it: the tracker carries a
`person` obstacle across ticks, so a tick that holds a person track and produces no
`person` *sighting* is one the detector dropped between two frames where it found them.

> **474 of 678 person-track ticks kept the detection — recall 69.9%.**

That number has a trap under it, and the trap is worth more than the number. Scored
naively over every run, recall reads **43.0%**. The difference is 7 runs of generation
G0–G2, whose telemetry **has no `sightings` field to write into**: all 424 of their
person-track ticks score as drops, and the metric measures the schema rather than the
detector. `inventory.py` gates on generation and prints both figures so the gap stays
visible.

### Recall collapses close, and off-centre

Restricted to the 14 runs whose schema can express a sighting, and to ranges that are
measurements (see below):

| range | hit | miss | recall |
|---|---:|---:|---:|
| 0–1 m | 8 | 34 | **19.0%** |
| 1–2 m | 233 | 2 | 99.1% |
| 2–3 m | 53 | 70 | 43.1% |
| 3–4 m | 24 | 30 | 44.4% |
| 4–5 m | 159 | 20 | 88.8% |
| 5–10 m | 19 | 48 | 28.4% |

**It is not a slope, and the worst band is the nearest one.** Recall at 0–1 m is 19.0%,
against 99.1% one metre further out. A person close enough to be a hazard is the person
this detector is least likely to report — the box clips the frame edge, which is the same
condition that triggers the fabricated ranges below.

| bearing | hit | miss | recall |
|---|---:|---:|---:|
| −30° | 5 | 25 | **16.7%** |
| −15° | 236 | 11 | 95.5% |
| 0° | 197 | 65 | 75.2% |
| +15° | 33 | 53 | 38.4% |
| +30° | 31 | 50 | 38.3% |

Off-centre recall falls to roughly a third. The −30° column is 30 samples and the
asymmetry against +30° should not be read as a lens property on that count.

### 417 of 4,624 ranges are constants, not measurements

`person_detector.py` names four sources whose range is substituted rather than measured.
The deployed tree predates the `GroundRanger` that emits two of them, so only two occur:

| source | rows | value |
|---|---:|---|
| `width-capped` | 322 | **0.7194 m**, identical to four decimals on all 23 of walk 5's |
| `frame-fill` | 95 | **0.800 m** exactly, all 15 of walk 5's |

Both are `FILLS_FRAME_RANGE_M` and the object-fit cap — constants the code substitutes
when the geometry cannot be recovered. **Do not train or evaluate range against these
417 rows.** They are flagged per box in the published labels, not silently dropped.

## Against human eyes: 69% of visible people were missed

Track continuity above measures only ticks where the tracker *already had* a person, which
biases it towards frames where detection was working. To get the number the retraining
question actually needs — **how often was a person who was plainly there not reported** —
250 frames were sampled (every 10th, all five walks) and marked by eye for the presence of
a person, a brown cardboard carton, a doorframe and the blue bin.

Of those, **175 frames contain a visible person.** Scored against the shipped weights at
the 0.45 operating threshold the wrappers passed:

| | found | **missed** |
|---|---:|---:|
| **300 px — what these runs used** | 55 / 175 | **120 / 175 = 68.6%** |
| 224 px — the peer launcher's size | 22 / 175 | 153 / 175 = 87.4% |

Dropping the threshold to 0.25 recovers a lot — 130/175 at 300 px — but 0.45 is what the
wrappers passed, so 68.6% is the number that describes what the robot did.

**The two denominators disagree by design and both are reported**: 69.9% *kept* on track
continuity against 31.4% *found* on human presence. The first asks "having found someone,
did it keep them?"; the second asks "was someone there, and did it ever notice?". Quoting
either without its denominator would be a different claim, and `person_shaped` — a
box-aspect test, not a label — is a third gate again.

Walk 3 is the extreme: **21 of the sampled frames contain a clearly visible person and
the detector reported none of them, at either input size.**

### Every sampled frame contained something the class list cannot name

**250 of 250** sampled frames contain a brown cardboard carton; it stood in that corridor
all session. It has no VOC class and no colour profile. The detector's own output says the
same thing from the other side: given all 20 classes to choose from, **11.5% of its
detections (76 of 661) name something that cannot be in an office** — `aeroplane` 58,
`horse` 10, `train` 4, `car` 2, `cow` 1, `motorbike` 1.

⚠️ **How confident this is, stated plainly.** These are presence flags read off contact
sheets by eye, not boxes: they support "a person was visible in this frame" and nothing
finer. The sample is **250 of 2,473 frames (10%)**. Two columns need care before anyone
trains on them:

- **`cardboard` is `yes` on all 250 and means two visually unrelated things** — a distant,
  partly occluded carton seen through a glass door in walks 1/2/4/5, and the same carton
  filling half the frame from centimetres away in walk 3 from frame 209. One label, two
  objects.
- **`doorframe` is the weak column: 218 `yes`, 32 `unsure`, no `no`.** This office is a
  corridor of pod partitions, so some vertical edge bounding an opening is in nearly every
  unobstructed frame, and "genuine door" could not be separated from "partition edge"
  without inventing a rule. Re-cut it or drop it.

`bin` is the clean human column — 212 `yes`, 38 `no` — and it is corroborated by 948
independent colour-detector boxes. **No box was invented for any of these classes.**

## Three walks that show three different failures

| | walk 2 | walk 3 | walk 5 |
|---|---|---|---|
| telemetry | `20260827T010144Z-00319b6` | `20260827T010614Z-0041b2a` | `20260827T012702Z-00652ea` |
| ticks | 161 | 139 | 86 |
| person sightings | **157** | **0** | 18 |
| outcome | timeout after 31 s | stalled, 0.26 m of an expected 1.39 m | stalled, 0.05 m of an expected 0.46 m |

**Walk 2 fires on nearly every tick. Walk 3 fires on none.** Same robot, same weights,
same corridor, four minutes apart.

**Walk 5's avoidance ran on invented geometry.** 37 ticks commanded `veto-avoid`; **21 of
those 37** carried at least one sighting whose range was one of the two constants above
(12 `width-capped`, 9 `frame-fill`). The robot swerved for a distance no sensor measured.

**Walk 3 ended wedged against a cardboard box.** The obstacle is a large upright brown
corrugated carton, roughly human height with a white shipping label near the top, leaning
against a white office phone-booth pod; the robot drove into the narrow corner between the
carton and the pod's frosted-glass door. Bisected to single frames: the person is last
visible at **208**, the view is first fully blocked at **225**, a sliver of room reopens
**non-monotonically over 268-285**, and it closes for good at **286**. Neither a cardboard
carton nor a door has a VOC class. The detector was not wrong about them — it had no way
to be right.

### Re-scored against the shipped weights, at both input sizes

Issue #129 says three different things answer "what input size does production use", and
the telemetry does not record it — confirmed here, the header has no `input_size` field.
For **these** runs it is nonetheless recoverable, from the launcher rather than the log:

- The wrappers pulled off the robot (`run-smoke.sh`, `run-berth.sh`, `run-chair.sh`)
  **pass no `--input-size` at all**. `run-chair.sh` passes `--goal-input-size 300`, which
  is the goal-crop pass and a different detector instance.
- The deployed `visual_nav.py` defaults `--input-size` to **300** (line 948).

**So tonight's runs ran the detector at 300 px.** The 224 px figure belongs to
`deploy/run-peer-supervised.sh`, a different launcher for peer runs, and does not describe
this corpus. Both are reported below because #129 is still open and the next run may use
either.

| | frames firing `person` ≥ 0.45 | best `person` score in the whole run |
|---|---:|---:|
| **300 px — what these runs used** | **2 / 535** | **0.6155** |
| 224 px — the peer launcher's size | 0 / 535 | 0.4274 |

Across the whole of walk 3 the detector fired twice, and at 224 px its best moment in the
entire run was **0.4274 against a 0.45 threshold** — it would never have fired at all.

⚠️ **Do not read the sub-threshold scores as near-misses.** The prototxt's `detection_out`
layer carries `confidence_threshold: 0.25`, so nothing below that leaves the network at
all; patching it to 0.01 makes 224 px report a `person` in **535 of 535** walk-3 frames —
including the ~300 where the camera sees only cardboard. Below ~0.25 the score is noise,
which is why the shipped floor is where it is.

## What this does not cover

- **No retraining, and nothing here targets the Go2 detector.** Retraining targets the
  Lite3 and is owned elsewhere; this is the corpus and the measurement only.
- **The weights are assumed, not proven.** The local `MobileNetSSD_deploy.caffemodel`
  (sha256 `761c86fb…`) is the stock file the repo's own scorers expect, but the robot is
  offline and no checksum of what it ran is recorded anywhere, so byte-identity is
  unverified.
- **Track continuity cannot see a person who was never detected once.** It gives walk 3 a
  denominator of zero. Walk 3 is measured against visual inspection instead, separately.
- **The 84 MP4s carry no usable frame-to-tick mapping.** All are 7 fps while ticks ran at
  6–10 Hz, and the frames-to-ticks ratio across the 71 paired runs spans **0.14 to 0.90**
  — the recorder dropped frames unpredictably, so a frame cannot be indexed to a tick.
  This is why the video was not labelled.

## Where the corpus went

Published to the existing private dataset
[`armwaheed/go2-peer-detection`](https://huggingface.co/datasets/armwaheed/go2-peer-detection)
— extending it rather than starting a rival, because it is the same robot, the same
camera and the same detector, and it already measures people in shot. It is private and
must stay private: Arm office footage with an identifiable person in it.

The 84 MP4s went with it. They cannot carry labels, for the reason above, but they are
the only surviving imagery for 60 of the runs and the robot is gone — and PR #138's own
rule is that bulk robot data belongs in the dataset store rather than in git.

Left behind deliberately: the **six `legs_*.jsonl` joint logs and one arm-latch log,
72 MB** — ~100 Hz joint angles, torques and foot forces, with no perception field in them
and no bearing on detection. All seven are listed in `inventory.tsv` with their sizes and
hashes, so a locomotion question can still find them.
