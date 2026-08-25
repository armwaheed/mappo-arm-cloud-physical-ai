<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The Aug-20 held-out set, labelled frame by frame

`peer_crossday_20260820.json` — 284 frames, every frame of seven clips recorded on
2026-08-20, each marked `present: true` / `false` / `null` for a Go2 Wheel, six of them
carrying a box.

**Nine of the fifteen stills this repository has been calling "held-out peer frames" contain
no peer.** That is the reason this file exists. `FROZEN-FEATURE-CEILING.md` reports 53%
recall from "fires on 8 of 15 held-out peer frames"; six of those fifteen hold a robot, so
the denominator was wrong and at least two of the eight fires were on frames with nothing to
detect. The same model's false-positive rate over the same session was 38%, which is where
most of that "recall" came from.

## Provenance

Every frame is in this repository already, as video. Nothing new was recorded.

| clip | frames | peer present | split |
| --- | --- | --- | --- |
| `peer_cross1` | 43 | 30–42 (13) | select |
| `chair1` | 28 | none | select |
| `gs-0.6-300-0.35` | 13 | none | select |
| `gs-1.0-300-0.50` | 15 | none | select |
| `peer_cross5` | 82 | 23–69 (47) | test |
| `peer_baseline` | 45 | none | test |
| `smoke1` | 58 | none | test |

Three frames are `null` and scored as neither: `peer_cross1` 29, `peer_cross5` 22 and 70.
In each the only visible part of the robot is its lit LED bar against a shadowed pillar,
one or two frames before it enters or after it leaves. Calling them positives would punish
a detector for missing something a person can only identify by knowing what came next;
calling them negatives would punish it for a correct detection.

**The fifteen stills are frames of these clips**, and matching them back confirms the
labelling from both directions: `arrive_01..09` are `peer_cross5` frames 0, 10, 20 … 80 and
`cross_01..06` are `peer_cross1` frames 0, 8, 16 … 40. `arrive_08` is `peer_cross5` 070 —
independently flagged as the LED-only frame when reading the stills and again when reading
the clip.

**The negatives are the same 159 frames the earlier evidence used**, now addressed by clip
and index rather than by a directory listing. Four more gait-sweep clips exist in
`_raw/` (26 frames) and are deliberately NOT included, so the false-positive denominator
stays comparable with what is already on the record.

## Splits, and why they are by clip

`select` is for choosing a checkpoint; `test` is for reporting. Splitting by CLIP rather
than at random matters more than usual here: these are 7 fps video, so neighbouring frames
are near-duplicates and a random split would put the same moment on both sides. Early
stopping on the cross-day set is a real risk of tuning until it looks good — the split is
what keeps the reported number honest, and both halves are the held-out day.

## Method, and what to distrust

Presence was read at 1440 px with CLAHE, on a labelled pixel grid, plus 2–3x crops on every
frame where the answer was not immediate. The peer is unmistakable when present at all: a
metallic wheeled quadruped in a corridor of beige cabinetry. The two things that were
genuinely hard:

1. **The office chair is not the robot.** A chair with an ArUco marker is parked mid-corridor
   in every clip at almost exactly the apparent size of a mid-range peer, and in
   `peer_cross5` 015–021 it is the only object in the frame. `LABELLING.md` already names it
   as the corpus's persistent hard negative; here it is also the thing most likely to be
   mislabelled as a positive, and was checked at zoom on every frame of the run-up.
2. **Entry and exit frames.** The transitions were read frame by frame rather than sampled,
   which is what produced the three `null`s.

**Only six frames carry a box**, so `recall (IoU>=0.5)` is measured over six and not over
sixty, and the evaluator reports both denominators separately. Presence over a whole clip is
cheap; a box is not. The boxes were drawn on a labelled grid and are worth about ±25 px,
which on objects 250–1000 px wide is comfortably inside an IoU-0.5 decision but is not
tight enough to compare two detectors that both localise well.

## Reproducing

```
python3 - <<'EOF'
import cv2
for clip in ("peer_cross1", "peer_cross5", "chair1", "gs-0.6-300-0.35",
             "gs-1.0-300-0.50", "peer_baseline", "smoke1"):
    cap = cv2.VideoCapture(f"evidence/2026-08-20-peer-avoidance/scene-captures/_raw/{clip}.mp4")
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(f"xday/{clip}_{index:03d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        index += 1
EOF

eval_detector.py --proto M.prototxt --model M.caffemodel \
                 --manifest detector/labels/peer_crossday_20260820.json \
                 --frames-dir xday --split test
```

⚠️ **Re-encoding moves the numbers by a few points.** The same model scored 56% and 60%
false-positive rate on the same frames extracted twice at different JPEG qualities. Quote a
false-positive rate to the nearest few points, not to the percent, and compare models only
against frames extracted the same way.
