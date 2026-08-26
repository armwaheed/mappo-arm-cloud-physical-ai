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

## Provenance — where the video actually is

Nothing new was recorded; every frame below is a frame of one of seven clips shot on
2026-08-20. **The clips are not on the default branch.** They were committed to
`evidence/2026-08-24-peer-capture-and-gait-sweeps` and that branch was never merged, so
`evidence/2026-08-20-peer-avoidance/` does not exist in a fresh checkout of `main`. This
file said "every frame is in this repository already, as video" for two days, and against
`main` that was false.

Recover them by commit, not by path:

```
git show f7b158f3bf18ba9868a40305985f75dc42374a7b:evidence/2026-08-20-peer-avoidance/scene-captures/_raw/<clip>.mp4 > <clip>.mp4
```

Two other copies are reported to exist and **neither was verifiable from here**: a working
copy at `arm-seattle-spark-02:~/ssdft/eval/xday/`, and a Hugging Face dataset
`armwaheed/go2-peer-detection`. Both come from the correction comment on issue #77. The
dataset is not public — an unauthenticated request cannot tell it apart from one that does
not exist — so it is not the location to check first. The commit above is, because anyone
with this repository can check it in one command.

| clip | frames | peer present | split |
| --- | --- | --- | --- |
| `peer_cross1` | 43 | 30–42 (13) | select |
| `chair1` | 28 | none | select |
| `gs-0.6-300-0.35` | 13 | none | select |
| `gs-1.0-300-0.50` | 15 | none | select |
| `peer_cross5` | 82 | 23–69 (47) | test |
| `peer_baseline` | 45 | none | test |
| `smoke1` | 58 | none | test |

### Checked on 2026-08-26

All seven clips were recovered from that commit, decoded frame by frame, and the result
compared against this manifest with `check_manifest.py`:

```
whole manifest: 284 rows — present 60, absent 221, null 3, boxes 6
xday: 0 named-but-absent, 0 present-but-unnamed
OK
```

**Every frame this manifest names exists, and nothing else is in the directory.** The
second half matters as much as the first: `eval_class_agnostic.py` builds its negative set
as every JPEG the manifest does *not* name, so one stray file would silently join the
false-alarm denominator.

⚠️ **The indices are 0-based and the last one is not the count.** `peer_baseline` is 45
frames, `peer_baseline_000.jpg` … `peer_baseline_044.jpg`; `smoke1` is 58 frames, `_000`
… `_057`. `peer_baseline_045.jpg` and `smoke1_058.jpg` do not exist and are not named
here. A check that compares a count against the highest index reports two missing
negatives and lowers the test-split denominator from 136 to 134; that reading is wrong
about *this* manifest and was made once already. The denominators are **47 present /
136 absent / 2 null** on `test`, and 13 / 85 / 1 on `select`.

⚠️ **A SECOND MANIFEST EXISTS, AND IT REALLY DID HAVE THE OFF-BY-ONE.** The scoring runs
do not read this file. They read a derived split, `eval/remote_xday_test.json`, which
lives on the training host and was generated from it — and that file named
`peer_baseline_045.jpg` and `smoke1_058.jpg` while omitting `peer_baseline_000.jpg` and
`smoke1_000.jpg`. Two phantoms in, two real negatives out: the clip indices were shifted
by one. So the 134 reading was **right about the file the evaluator actually opens** and
wrong about this one, and "that reading is wrong" on its own would have sent the next
person to check the correct manifest and find nothing.

**What it cost, measured rather than assumed.** The derived split was repaired by mapping
each phantom onto the real frame its clip omitted, and every model rescored: **all
numerators are identical**. Neither recovered frame fires on any model. Only the
denominator moved — the shipped weights' cross-day false-alarm rate goes from 76/134
(57%) to 76/136 (56%), and every other rate by at most one point. No conclusion in any
document changes.

**Why the blast radius was that small, and do not count on it next time.** Both affected
clips — `peer_baseline` and `smoke1` — are entirely peer-free, so a one-frame shift could
not put a positive's label on a negative's pixels. On `peer_cross1` or `peer_cross5`, both
of which contain a peer for part of their length, the same shift would have mislabelled
the frames at each boundary and nothing here would have caught it. `check_manifest.py`
validates this file; the derived split had no such check, which is why it drifted.

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
CLIPS=f7b158f3bf18ba9868a40305985f75dc42374a7b:evidence/2026-08-20-peer-avoidance/scene-captures/_raw
mkdir -p _raw xday
for clip in peer_cross1 peer_cross5 chair1 gs-0.6-300-0.35 \
            gs-1.0-300-0.50 peer_baseline smoke1; do
  git show "$CLIPS/$clip.mp4" > "_raw/$clip.mp4"
done

python3 - <<'EOF'
import cv2
for clip in ("peer_cross1", "peer_cross5", "chair1", "gs-0.6-300-0.35",
             "gs-1.0-300-0.50", "peer_baseline", "smoke1"):
    cap = cv2.VideoCapture(f"_raw/{clip}.mp4")
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(f"xday/{clip}_{index:03d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        index += 1
EOF

# Check the extraction BEFORE scoring against it. Non-zero exit means do not score.
python3 detector/labels/check_manifest.py \
        detector/labels/peer_crossday_20260820.json --frames-dir xday --split test

eval_detector.py --proto M.prototxt --model M.caffemodel \
                 --manifest detector/labels/peer_crossday_20260820.json \
                 --frames-dir xday --split test
```

⚠️ **Re-encoding moves the numbers by a few points.** The same model scored 56% and 60%
false-positive rate on the same frames extracted twice at different JPEG qualities. Quote a
false-positive rate to the nearest few points, not to the percent, and compare models only
against frames extracted the same way.
