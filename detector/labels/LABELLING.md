# peercap — Unitree Go2 "Wheel" peer-robot boxes

`annotations.json` — 1903 labelled frames, one class `go2wheel`, one box per frame.

```
{"label": "go2wheel",
 "source_dir": ".../scratchpad/peercap/",
 "count": 1903,
 "records": [{"image": "p1_close_broadside_0000.jpg", "label": "go2wheel",
              "box": [232, 0, 1783, 1042]}, ...]}
```

`box` is `[x0, y0, x1, y1]` in pixels of the original 1920x1080 frame, corners inclusive.
Images were not moved or modified; reference them by name inside `source_dir`.

**Box convention: visible extent, clipped to the frame.** Nothing is extrapolated behind a
frame edge or behind the human operator. 744 of 1903 boxes (39%) touch a frame border
(bottom 647, right 272, left 140, top 97). Boxes cover 1.3%–78% of the frame by area.

## Method

The brief suggested seeding one frame per segment and propagating with CSRT. Two things
changed that plan:

1. **There is no CSRT in this OpenCV.** `caffevenv` has cv2 4.10 but was built without
   `cv2.legacy`, so CSRT / KCF / MOSSE / Boosting / TLD do not exist. What is left is
   `TrackerMIL` plus GOTURN / DaSiamRPN / Nano / Vit, all of which need weight files that
   are not on this box and cannot be fetched offline.
2. **Nothing needed tracking.** Ten of the eleven segments are rigid-static — see below.

So propagation used two mechanisms instead:

* **Anchor-template NCC** (`track.py`) — `cv2.matchTemplate(TM_CCOEFF_NORMED)` of the
  *anchor* patch (never the previous frame's patch, so drift cannot accumulate) in a ±70 px
  window. Used on the nine static segments. It also doubles as the check that the peer is
  static: it reports the peak correlation and the shift.
* **Background-plate difference** (`blobs.py`, `p4boxes.py`) — for p4, and for seeding the
  truncated segments where reading an edge-clipped robot by eye is hardest.
  - Corridor plate (`buildplate.py`): the peer is parked somewhere different in every
    corridor segment, so a per-pixel median over one frame from each — all ORB+RANSAC
    registered to a common reference — leaves an empty corridor. ORB/RANSAC rather than ECC,
    because ECC gets dragged off by a peer that fills half the frame.
  - p4 plate (`p4plate.py`): built from p4's own frames — left half from the late frames
    (peer parked right), right half from the early frames (peer still at the left). Same
    camera pose in both, so registration error is exactly zero.

Seeds were drawn by eye on renders with a labelled pixel grid at 1–4.5x with CLAHE
(`view.py`), cross-checked against plate-difference heat maps, then re-drawn and re-viewed
until the edges sat on the robot.

**Verification.** 6–9 frames spread across every segment were rendered with the final box
and inspected (`verify.py`), plus targeted 2–4.5x zooms on every box edge I was unsure of
(p1 left/right/bottom, p2 all four, p3 top/left/right, p4 frames 0/40/73/104, p5_1 and
p5_23 all four, p6_1/p6_2/p6_3). Nothing needed re-seeding after the first refinement pass;
p4 was recomputed twice, once to add proximity/area blob merging and once to reject the
floor shadow.

## Per-segment result

| segment | frames | kept | dropped | box | propagation | evidence | trust |
|---|---|---|---|---|---|---|---|
| p1_close_broadside | 97 | 97 | 0 | `232,0,1783,1042` const | hand seed + NCC | peak ≥0.993, shift 0 px | high |
| p2_close_headon_stand | 98 | 98 | 0 | `655,268,1665,1080` const | hand seed + NCC | peak ≥0.982, shift ≤1 px | high |
| p3_close_rearon_stand | 133 | 133 | 0 | `595,308,1790,1080` const | hand seed + NCC | peak ≥0.988, shift ≤2 px | medium-high |
| p4_mid_sweep_stand | 208 | 208 | 0 | per-frame, x0 157→1308 | plate difference | smooth trajectory, no empty frame | medium-high |
| p5_1_far_left_stand | 109 | 109 | 0 | `618,527,792,682` const | hand seed + NCC | peak ≥0.968, shift ≤2 px | **low-medium** |
| p5_23 frames 0000–0182 | 183 | 183 | 0 | `898,548,1078,706` const | hand seed + NCC | peak ≥0.98 until f182 | **low-medium** |
| p5_23 frames 0183–0205 | 23 | 23 | 0 | per-frame, walks right | gated blob centroid | visual check at f185/192/198/205 | **low** |
| p6_1_trunc_left_stand | 140 | 140 | 0 | `0,311,249,1080` const | plate-diff seed + NCC | single clean blob, peak ≥0.986 | high |
| p6_2_trunc_right_stand | 136 | 136 | 0 | `1577,495,1920,1080` const | plate-diff seed + NCC | single clean blob, peak ≥0.986 | high |
| p6_3_trunc_half_left_stand | 136 | 136 | 0 | `1046,297,1920,1080` const | plate-diff seed + NCC | single clean blob, peak ≥0.984 | high |
| peer01 | 640 | 640 | 0 | `852,403,1332,778` const | hand seed + NCC | peak ≥0.991, shift 0 px | high |
| smoke | 58 | **0** | 58 | — | — | — | — |

**Dropped: 58 frames, all of `smoke_*`, dropped as instructed** (throwaway smoke test — and
it is a byte-for-byte duplicate *viewpoint* of peer01, see below). **Zero frames were dropped
for an absent or unrecognisable peer.** The peer is present and identifiable in all 1903
remaining frames, including the 15%-visible truncations: p6_2 shows a shoulder, one leg and
one wheel, which is still unambiguously this robot.

On the truncated segments the box is the visible extent **snapped to the frame border** — a
1–2 px NCC shift would otherwise have left a sliver of background between the box and the
edge on 165 of those 412 frames.

## What to distrust, worst first

1. **p5_23 frames 0183–0205 (23 frames).** The only far-and-moving frames in the set, and
   the weakest labels. NCC cannot follow a 180x158 px backlit robot — the background inside
   the box outvotes it — so the centre comes from plate-difference blobs (gated above the
   floor glare streak) and the size is copied from the anchor. Expect ±15–20 px, and the
   robot is *turning* through this span so its true aspect ratio changes while the box's
   does not.
2. **p5_1 (109 frames).** ~4 m, backlit through a glass door, the robot is a dark silhouette
   against a dark bench. Edges were read off a 109-frame temporal average at 4x (averaging
   was necessary — a single frame is too noisy). The left edge is the least certain: there is
   a dark patch at x≈600–618 that is either the rear wheel in shadow or floor shadow.
   Call it ±10–15 px on a 174 px box — roughly 8% of the object.
3. **p5_23 frames 0000–0182 (183 frames).** Same failure mode, slightly better contrast.
   ±8–10 px on a 180 px box.
4. **p4 (208 frames).** Tight and automatic through the middle of the crossing, but two
   systematic biases. The polished floor reflects the robot directly beneath it, so the box
   can run up to ~40 px **low** on the close frames (worst around f0–f45, where the reflection
   is brightest); a colour-ratio shadow test removes the soft shadow but not the specular
   reflection. On the closest frames the left edge can also run ~20 px wide for the same
   reason. Only 4 p4 boxes reach y=1080, and in those the wheels really are cut by the frame.
5. **p3 (133 frames).** The box top (y=308) rests on my reading that the dark rounded object
   at x 1235–1330, y 305–325 is the robot's rear grab handle and not the office chair's
   backrest directly behind it. I checked this at 4.5x and I believe it is the robot, but if
   it is not, every p3 box is ~70 px too tall (9% of its height).
6. **p2 (98 frames).** Solid, one caveat: x0=655 is set by ~45 rows of tyre in the
   bottom-left corner (y>1035). If your loader ever crops the bottom of the frame, that
   box becomes 250 px too wide.
7. **p1, peer01, p6_1, p6_2, p6_3.** The strongest labels. Large or high-contrast objects,
   crisp edges, boxes verified edge-by-edge at zoom. I would train on these without
   hesitation.

## Things about the data worth knowing before you train

* **There is no settle drift, anywhere.** The brief expected the camera to settle for a
  second after standing. It does not: worst-case global phase correlation against frame 0 is
  1.5 px (p3), and it is *exactly* 0.0 px for p1, peer01 and smoke. Peer motion inside the
  box is 0.6–4% of pixels changing between consecutive frames, which is sensor noise plus
  balance micro-sway, not translation. Anchor NCC shift over a 640-frame segment: 0 px.
* **peer01 is 640 frames of one picture.** Identical box in all 640, NCC peak ≥0.991, zero
  shift. It is 34% of the dataset and roughly 1 viewpoint. The two bursts of frame-to-frame
  change (f179–209, f267–291) that look like the robot moving are the **human operator
  shifting his feet** — the peer does not move at all. Subsample or down-weight it, or
  p1+peer01+smoke (one camera pose, 795 frames) will dominate training.
* **smoke is not just a throwaway — it is a duplicate of peer01.** smoke, peer01 and p1 share
  one camera pose exactly (phase correlation 0.0 px between them); the camera robot stayed
  prone from t=9766 through t=10228 and only the peer moved. smoke_* and peer01_* even show
  the peer in the same place. Dropping smoke costs nothing.
* **A persistent hard negative sits in every corridor frame:** an office chair with an ArUco
  marker parked mid-corridor, at almost exactly the apparent size of the far peer in
  p5_1/p5_23. It is *not* labelled (it is not a robot), and it is the thing most likely to
  generate false positives. If you want explicit negatives, that is the crop to mine.
* **The floor is a mirror.** Every corridor segment has a specular reflection of the robot
  under it, plus a fixed bright vertical glare streak at x≈935, y 720–920. The streak is what
  made the first naive plate-difference boxes 200 px too tall on p5_23. If you train with
  heavy vertical jitter, the reflection is a plausible source of doubled detections.
* **Two segment names lie.** p6_3 is `trunc_half_left` but the peer is cut by the **right**
  frame edge (box x0=1046, x1=1920). And p5_23's own name is the honest one: it is
  `far_centre_then_right`, and the peer really does walk right at the end — the brief
  described the whole segment as static.
* **peer01.jsonl is truncated.** It has 585 manifest lines for 640 images; the capture was
  aborted before the file was flushed, so peer01_0585..0639 have no manifest entry. The
  images themselves are fine and are labelled. All 1961 jpgs are 1920x1080 and every file is
  byte-unique (no duplicated files — though most segments are near-duplicate *content*).
* **p1's peer is cut off at the top of the frame** (box y0=0) — it is the only segment with
  top truncation, so it carries all 97 of the top-edge-touching boxes.

## Reproducing

Scripts are in [`pipeline/`](pipeline/), lint-clean under `cd detector/labels/pipeline &&
ruff check . --config ../ruff.toml`. They take no path arguments: every one resolves the
capture from `PEERCAP_FRAMES` and **refuses** with the locations the corpus is reported to
be in when it is unset (`pipeline/peercap.py`). Nothing here hard-codes a directory, and
`test_peercap.py` walks the AST of every module here to keep it that way.

```
export PEERCAP_FRAMES=~/go2-peer-dataset-20260824      # the 2,800 jpg
python3 buildplate.py                                  # then the pipeline below
```

⚠️ This block used to say the scripts were in `../peercap_work/` and ran under
`../caffevenv/bin/python`. Neither path has ever existed in a checkout — `ls
detector/peercap_work detector/caffevenv` fails on `main` — and this is the block a reader
pastes. Same defect class as the manifest that named a scratchpad and had 2,800 frames
declared lost over it (#86, #92, issue #77).

```
buildplate.py                                   # corridor background plate
p4plate.py                                      # p4-native composite plate
blobs.py <tag> <thr> <min_area>                 # per-frame plate-difference blobs
p4boxes.py                                      # p4 per-frame boxes
track.py <tag> <anchor_idx> x0 y0 x1 y1         # anchor-NCC propagation, one per segment
build.py                                        # -> peercap_labelled/annotations.json
verify.py <tag> <out.jpg> <n>                   # verification sheet
checks.py [drift|change|roi]                    # the static-segment evidence above
view.py <frame> <out> x0 y0 x1 y1 [step] [scale] [enh] [box...]   # seeding tool
```

The anchor seeds are the `track.py` argument lists recorded in `checks.py:BOXES`.

## Auto-labelling a recorded run

The above is a *hand*-labelling pipeline for one staged capture. Every live run of
`visual_nav.py` also carries labels already — the pixel box, the label, the score, the
range, the bearing and the ranging prior are in each telemetry tick, keyed to the recorded
video by `perception.video_frame`. [`autolabel_run.py`](autolabel_run.py) is the join, and
it writes a manifest in the `records` shape this directory already uses, so
`check_manifest.py` and `eval_class_agnostic.py` read it unchanged:

```
python3 autolabel_run.py RUN.jsonl --frames-dir OUT --manifest OUT/labels.json \
    --unlabelled-dir MISSED --classes person --label go2wheel
python3 check_manifest.py OUT/labels.json --frames-dir OUT        # both directions
```

⛔ **These are detector boxes.** They inherit the shipped network's recall — 64%
class-agnostic at the deployed 0.25 — so this cannot benchmark that network, and a frame
it found nothing in is an *unlabelled* frame, not a peer-free one. Good for range and
aspect statistics over real runs, and for producing frames a human corrects rather than
draws. Not a test set. Issue #77.

**One record per box, not per frame.** Two peers in one frame are two records naming one
image; `eval_class_agnostic.load_frames` accumulates a list per image and scores each
frame against the best match in it. Over the two committed runs that carry `sightings`,
21% of the frames with a box hold more than one, and a top-box-per-frame manifest drops
23 of their 118 boxes.

Three things it refuses, and each of them has cost this project something before:

* **The `--record` video, by name.** That file has an orange box drawn around every
  detection, at exactly the place the label goes. Use `--record-raw` (#84).
  `--allow-annotated` exists for range statistics, where a drawn box does not matter, and
  it is stamped into the manifest.
* **A video that is not this run's own recording.** Joining
  `evidence/2026-08-25-peer-runs/hero-run-telemetry.jsonl` to the committed
  `hero-clears-peer-on-right.mp4` yields auto-labels that pass `check_manifest.py` in both
  directions and are nonsense — 58 frames recorded, 423 in that file, because the committed
  clips are re-timed edits. The recorder writes exactly one frame per recorded index, so the
  count is checked: against the container's own header before a single JPEG is written, and
  against the decoded count afterwards.
* **Putting the frames the detector missed beside the ones it found.**
  `eval_class_agnostic.py` scores every JPEG a manifest does not name as peer-free, so
  those would be filed as the network's own false alarms. `--unlabelled-dir` keeps the
  pixels somewhere else — they are the frames a human should label next — and the two
  directories may not be the same one.
