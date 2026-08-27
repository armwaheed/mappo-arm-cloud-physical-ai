<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Teaching the perception stack to see a peer robot

No off-the-shelf detector has a **name** for another quadruped. That much is measured,
not assumed — though the first row used to say something stronger and false, and why it
was wrong turned out to be the whole story:

| detector | best response to a Go2 Wheel filling half the frame |
| --- | --- |
| MobileNet-SSD / VOC-21 (**what the robot runs**) | no *correct* class; the box comes back as `bicycle` **0.984** — see the correction below |
| YOLO11n / COCO-80 | `motorcycle` 0.138, `bicycle` 0.112 |
| YOLO-World, prompted "robot dog" / "quadruped robot" | `robotic dog` **0.039** |

The open-vocabulary row is the one that settles it. That model, on the same frames, with
the same code path, scores `person` **0.932** — so the pipeline is verified and the null
result is real. There is no off-the-shelf model that knows what a Lite3 is.

> ⚠️ **Correction, 2026-08-25 — the floor, and the sweep it hid.** The first row of that
> table used to read "nothing at any score down to 0.02", and it was wrong twice over.
> `MobileNetSSD_deploy.prototxt` carries `confidence_threshold: 0.25` inside its
> `DetectionOutput` layer, so **0.02 was never reachable** — the network cannot emit a box
> below 0.25 whatever a caller passes. And the stock net does not return "nothing": read
> class-agnostically it puts a box on a close-range peer at **0.984**, and on **81.2%** of a
> 1,903-frame peer corpus at 0.15, against 64.4% at the shipped 0.25. What is genuinely
> absent is the *label*, not the box — which is the premise this directory rests on, and it
> is unchanged. The measured sweep is in
> [`evidence/2026-08-25-peer-detector-threshold-and-tracks/`](../evidence/2026-08-25-peer-detector-threshold-and-tracks/).
>
> That sweep also killed the **18% false-alarm rate** this directory had been quoting: all
> 159 of those "alarms" are correct boxes on a peer in a frame the label file forgot, and on
> the 705 genuinely peer-free frames of that capture the rate is **0/705** at every
> threshold ≥ 0.14.
>
> ⛔ **Do not read that 0/705 as a licence to drop the floor.** Those 705 frames are staged
> negatives — one corridor, cleared and shot for the purpose, same session as the positives.
> Scored **cross-day** on a furnished room the same class-agnostic read fires on **57%** of
> peer-free frames, which is the number the section below is built on. The recall figure is
> a property of the network and stands; the false-alarm figure is a property of the room and
> does not.

### ⚠️ Two of the sentences above were wrong, and finding out ended the fine-tune

**"Nothing at any score down to 0.02" was never measured, and the claim it stands for is
false.** The deployed prototxt carries `confidence_threshold: 0.25` in its
`detection_output_param`, and `DetectionOutput` applies it before `forward()` returns — so
0.02 was 0.25 wearing another number, and no sub-0.25 box has ever left this network. Asked
the same question over 1,903 labelled peer frames rather than one, the shipped 21-class model
puts a box **on** the peer in **64%** of them, at 0.25, under names like `motorbike` and
`chair`. It sees the robot. It has no word for it, and it never needed one.

**"The telemetry contains `bin` and `person` and nothing else" is a fact about the
configuration, not about the model.** `PersonDetector` is constructed with
`classes=DYNAMIC_CLASSES`, which is `("person",)`, at confidence 0.45; `bin` comes from
`colour_detector.py`. Those 148 ticks could not have contained a third label whatever the
network emitted. Four instruments agreeing here was one filter, counted four times.

Both corrections point the same way, and
[`FROZEN-FEATURE-CEILING.md`](FROZEN-FEATURE-CEILING.md) now ends with the measurement rather
than with a training plan: **do not fine-tune this detector — read the stock one
class-agnostically.** [`eval_class_agnostic.py`](eval_class_agnostic.py) is the check.
Both fine-tunes were taken to the end first — the frozen-feature head, then the backbone
unfrozen over twelve 40-epoch runs, which does lift the first's ceiling and still loses to
the stock model on the same frames ([`UNFROZEN-FINE-TUNE.md`](UNFROZEN-FINE-TUNE.md)).

### ⛔ That recommendation is wrong on the peer, right on `person` — corrected 2026-08-26

The paragraph above is left standing because it is what this repository acted on, and because
the way it was reached is the lesson. **It compares two models that were never scored on the
same day.** "64% at 18%" and the fine-tune's "53% at 38%" are both the Aug-24 capture, which is
the fine-tune's own training day; its honest held-out number is the Aug-20 split, and **the stock
model had no Aug-20 number at all.** `UNFROZEN-FINE-TUNE.md` says in its own words that "47
positives and 136 negatives could not rank anything" and the recommendation was issued anyway.

Scored on the **Aug-20 held-out split — 47 peer-present frames, 134 peer-free** — with one rule
applied identically to every model: **a frame counts as a fire if any detection at >= 0.25 has box
aspect h/w < 2.0.** That is "not person-shaped", which is what the deployed stack routes to the
policy as an obstacle rather than holding for (`person_detector.PERSON_ASPECT_MIN`, PR #73).
`people kept` counts, of the 15 frames where the **shipped** network sees a person at >= 0.45 —
the confidence the navigator runs at — how many the candidate still sees.

| model | peer recall | false alarms | precision | people kept |
| --- | ---: | ---: | ---: | ---: |
| **stock 21-class (shipped)** | **68%** (32/47) | **57%** (76/134) | 30% | **15/15** |
| `l_full_bb02` ep015 | 70% | 19% | 56% | 13/15 |
| `j_full_distil03` ep015 | 66% | 6% | 79% | 11/15 |
| `i_full_pseudo` ep020 | 74% | 8% | 76% | 8/15 |
| `f_full_distil01` ep020 | 74% | 1% | 95% | 4/15 |

Two things follow, and both of them are the finding:

1. **On the peer, every fine-tune beats the shipped weights cross-day.** Three of the four are
   above its 68% recall and the fourth is two points under, at 1-19% false alarms against its
   **57%**. The 57% is the number that moved: the class-agnostic stock read fires on 57% of
   peer-free corridor frames, not 18%. That 18% was measured against *staged* empty-corridor
   negatives — 705 of those 897 Aug-24 peer-free frames are the `neg_prone` and `neg_standing`
   sets, corridor cleared and shot for the purpose — and a furnished room is not an empty one.
   So "do not fine-tune" does not survive on the axis it was argued.
2. **Every fine-tune buys that with people.** `f_full_distil01`, the best peer precision on the
   table, keeps **4 of 15**. `FROZEN-FEATURE-CEILING.md` already warns that one-class fine-tuning
   teaches the network that people are background; that warning was right and understated at 2 of
   15. **Nothing here is deployable.** The open objective is to hold every person the shipped
   network sees while keeping the peer gains — a sweep testing exactly that was running as this
   was written, and has since reported. It did not clear the gate. See below.

⚠️ **47 and 134 are small, and this is one corridor on one day.** Three evidence sets in this work
have already been too small and too like themselves — the 0-of-705 refusal gate, the fifteen
stills of which nine held no peer, and the 47-positive ranking — and each flattered the thing being
tested. The table above is worth what it compares, not what it measures: every row scored the same
way on the same frames.

⚠️ *The 134 is now explained, and it was not the manifest.*
[#91](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/pull/91) found the *derived* split
on the training host shifted by one clip index — two named frames that do not exist, two real ones
omitted. Repaired and all thirteen rows rescored: every numerator identical, only the denominator
moving, and the shipped weights going from 76/134 = 57% to **76/136 = 56%**. The table above is
left at 134 and 57% because that is what it was measured on; the 2026-08-26 sweep is scored on the
repaired 136.

### ⚠️ Those four checkpoints were not the best four — swept 2026-08-26

**They were four of the thirteen model/epoch rows anybody had scored, out of 640 checkpoints that
existed.** Sixteen training runs at 40 epochs each left 640 checkpoints on the training host;
thirteen rows had ever been evaluated, and a fifth training wave was launched before the rest were
measured. 64 have now been scored the same way, on the same split, and **the best detector in the
whole set was already on disk in a run no document in this repository had ever named.**

`git grep` finds four of the sixteen run names on `main` before 2026-08-26 — `f_full_distil01`,
`i_full_pseudo`, `j_full_distil03`, `l_full_bb02`. The winner is one of the other twelve.

Same Aug-20 split, same one rule, on the **136** peer-free frames the manifest actually holds
(see the correction to the note above). `people` counts, of the **22** frames where the shipped
network sees a person at **0.25** — the value `deploy/run-peer-supervised.sh` launches the peer
runs with, not the 0.45 the earlier table uses — how many the candidate still sees. **ep022 and
ep017 come from a finer pass over the winning run and are not among the 64**; they are recorded
as attributed rather than verified, and so is the 22-frame denominator, which needs the corpus
pixels this repository does not hold:

| model | peer recall | false alarms | people at 0.25 |
| --- | ---: | ---: | ---: |
| **stock 21-class (shipped)** | 68% (32/47) | **56%** (76/136) | **22/22** |
| `k_full_pseudo03` ep022 | **89%** | 12% | 17/22 |
| `k_full_pseudo03` ep020 | 85% (40/47) | 10% (14/136) | 17/22 |
| `k_full_pseudo03` ep017 | 83% | 12% | **19/22** |
| `p_bb02_d01_aug` ep020 | 70% (33/47) | 18% (25/136) | **19/22** |
| `f_full_distil01` ep020 | 74% (35/47) | **1%** (2/136) | 5/22 |

**13 of the 64 beat the shipped weights on both peer axes. All 64 beat them on false alarms.**

Three things follow, and the first is the one worth carrying to another project:

1. **The order of work was the defect.** The best detector was on disk, unscored, while a fifth
   wave was queued against the same host. Measuring 640 existing checkpoints costs an evaluation
   pass; four more 40-epoch runs cost a day of GB10 time and produced nothing better than what
   was already there. **Measure what you have before you make more of it.**
2. **The lever is `--pseudo-labels`.** `k_full_pseudo03` is the only run of the sixteen at 0.3;
   every other used 0.5 or none. A lower threshold carries more of the starting network's own
   detections into training as old-class ground truth — nearly all of them `person` — which is
   the axis every other run failed on. `UNFROZEN-FINE-TUNE.md` had already called that knob *"the
   single most effective knob in this work"* and it was never turned again. A wave sweeping below
   0.3 is running as this is written; **no conclusion about its outcome is recorded here.**
3. ⛔ **Nothing is deployable, and the gate is unchanged.** No checkpoint of the 64 keeps all 22
   people. The best keeps 20 and is *below* the shipped weights on peer recall; the set that
   beats them on both peer axes and keeps at least 20 is empty. `k_full_pseudo03` ep022, the best
   peer row here, loses five people. **Gives way to people** is this robot's shipped safety
   property, and a checkpoint that finds more peers and drops a person is not an improvement to
   it.

The full record — the 64-row grid, the raw sweep data, both figures, the augmentation control and
what a clone cannot re-derive — is in
[`evidence/2026-08-26-checkpoint-sweep/`](../evidence/2026-08-26-checkpoint-sweep/).

### ⛔ Every row of that table is a 300 px number, and the robot runs 224 — measured 2026-08-26

The sweep above, the wave-6 sweep beside it, and every scorer in this directory squash the
frame into **300 x 300**. `deploy/run-peer-supervised.sh` — the script the sweep cites by line
number for its `--confidence 0.25` — passes **`--input-size 224`** seven lines earlier, so the
peer runs this project reports were not executed at the size any of them were scored at.

Measured on the shipped weights, same frames, same rule, one process, changing only the square:

| shipped weights, whole Aug-20 day | peer recall | false alarms | people held |
| --- | ---: | ---: | ---: |
| at **300** — every table above | 41/60 = 68% | 108/221 = 49% | 32 |
| at **224** — what the runs launch | 30/60 = **50%** | 57/221 = **26%** | **25** |

So the baseline the candidates were ranked against is not the incumbent's production
behaviour, and **no candidate checkpoint has been scored at 224 at all**. The launch script's
own comment already warned that this axis is *"non-monotonic … a marginal-detection smell"* —
it measured 224 as **6x better** than 300 on one close-range clip, where the cross-day day
measures it 18 points **worse**. Both are real; neither transfers.

Rankings *among* candidates survive this, because a shared preprocessing error cancels in a
comparison scored the same way. The comparison against the shipped weights does not, and that
is the only one that decides whether anything ships. `--pseudo-labels 0.2` still looks like the
best configuration found.

⚠️ **The `people` denominator in the wave-6 table is also the wrong one.** It counts frames
carrying a box *labelled* `person`; the robot stops on `person_shaped` — aspect h/w ≥ 2.0 —
and this file already explains at length that the label cannot be trusted for it. Filtered by
the gate the stack applies, the candidate loses **5 of 31 (16%)**, not 4 of 54 (7%).

The measurement, the reproduction script, and the two published denominators it re-derives from
a clone are in
[`evidence/2026-08-26-detector-input-size/`](../evidence/2026-08-26-detector-input-size/).

#### ✅ Fixed, and then measured — and the section above has its premise wrong

**The robot does not run 224. It runs 224 under ONE launcher and 300 under the other
three**, and the sweep ran neither:

| launcher | `--input-size` | `--confidence` | `--classes` |
| --- | ---: | ---: | ---: |
| `deploy/run-peer-supervised.sh` | **224** | 0.25 | all **20** VOC labels |
| `run-smoke.sh` / `run-berth.sh` / `run-chair.sh` (on the robot) | none → **300** | 0.45 | none → **`person` only** |
| a bare `visual_nav.py` | default → **300** | 0.4 | default → **`person` only** |
| **the 2026-08-26 checkpoint sweep** | 300 | 0.25 | all 20 |

So this directory's 300 was right for three launchers and wrong for the fourth, and the
sweep's *pair* — the square from a scorer constant, the floor from the peer launcher — is
run by nothing. `evidence/2026-08-27-89-runs-survived-14-can-be-dated/` settles which is
which: the 89 recorded runs ran at 300 px, established by reading the launchers because the
telemetry did not record it.

**No file in this directory declares a network input size any more.** `eval_detector.py`,
`eval_class_agnostic.py`, `peer_recall.py`, `train_new_class.py`, `ssd_torch.py` and
`score_crossday.py` all take it from
[`robot-stack/unitree/go2/visual_nav/inference_profile.py`](../robot-stack/unitree/go2/visual_nav/inference_profile.py),
which declares each configuration beside the launcher that produces it. **A scorer's
`--preprocessing` is required and has no default** — with four real configurations there is
no safe one to fall back on — and a configuration no launcher runs is refused unless a
reason is given, which is then recorded in the output.

The same weights on the same 284 cross-day frames, once per configuration:

| the shipped weights, as | peer recall | false alarms | hold |
| --- | ---: | ---: | ---: |
| `go2-peer-supervised` 224 px, 0.25, 20 labels | 30/60 = 50% | 57/221 = 26% | 27 |
| `go2-run-smoke` 300 px, 0.45, `person` only | **8/60 = 13%** | 5/221 = 2% | 20 |
| `go2-navigator-default` 300 px, 0.4, `person` only | **8/60 = 13%** | 10/221 = 5% | 24 |
| **the sweep's 300 px, 0.25, 20 labels — run by nothing** | **41/60 = 68%** | 108/221 = 49% | 40 |

**Every "the shipped network gets 68% recall" on this page is the last row**, and the
configuration behind all 89 logged runs sees the peer on **13%** of the same frames — a 5x
gap, most of it the class list rather than the square. `PersonDetector` drops a detection
whose label is not in `--classes` before anything downstream sees it, and this page already
records the peer coming back as `horse 0.28` head-on at 1.3 m: on a smoke run that box is
not an obstacle at all. All 800 checkpoints have since been scored at all four
configurations; how many beat the incumbent has a different answer at each. See
[`evidence/2026-08-27-one-robot-four-detectors/`](../evidence/2026-08-27-one-robot-four-detectors/README.md).

### ⚠️ `horse` is not a viable label for a quadruped, and counting emitted labels could not tell you

A Go2 Wheel head-on at 1.3 m is labelled `horse 0.28` in the hero run of 2026-08-25, and the
proposal that followed — use `horse` as the VOC label that means "quadruped robot" — was reasonable
on that evidence. **Counting emitted labels could not settle it.** `horse` never appears in the
1,294 detections that landed on the peer, but per the confidence-floor correction above **nothing
below 0.25 has ever left this network**, so a `horse` logit sitting at 0.20 on every head-on frame
would look identical to no `horse` logit at all. The other objection was that the Aug-24 corpus is
broadside-dominated — which is why `motorbike` wins it — and therefore holds no head-on peer to
test the idea against.

Read properly — prototxt threshold patched **0.25 → 0.01** and the full 21-way softmax taken over
all **1,903 labelled frames** — it is not there:

| | `horse` |
| --- | --- |
| frames where it clears 0.25 | **0** |
| frames where it reaches the top three | **0** |

In every staged group, and the second objection does not hold either: **`p2_close_headon_stand` is
98 frames of the peer square to the camera**, which is exactly the view the proposal was about, and
`horse` scores zero there too. Across the four live runs of 2026-08-25 — 412 sightings — `horse`
appears six times, and those six are six consecutive ticks of one approach in one run, not six
observations.

What the label was tracking is box shape, which is measured directly and scale-free by
`person_detector.person_shaped`. Routing on `horse` would put a noisy proxy in front of a clean
measurement, and re-introduce the label routing PR #73 removed. Closes
[#76](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/76).

Everything below this line is still accurate about *how* to add a class, and remains the
route if one ever has to be added for a reason a stock label cannot serve.

## What binds, and why it is not the training

The Jetson runs Python 3.8 and **OpenCV 4.2**, with no CUDA, no torch and no onnxruntime.
`cv2.dnn` is the only inference engine on the robot, so whatever is trained has to come
back as the `.caffemodel` the stack already loads. Training is the easy half; **deployment
is the constraint**, and there is a working demo not to regress.

`add_class.py` is the answer: a `.caffemodel` is protobuf, and protobuf does not need
Caffe. Train in PyTorch anywhere, write the weights back, change nothing on the robot —
same prototxt, same preprocessing, same 131 ms, same ~7 Hz.

### Verified end-to-end, offline

| check | result |
| --- | --- |
| Parse `.caffemodel` in pure Python | 194 layers, 117 with weights, 5,821,468 params |
| Re-serialise, compare `cv2` output | **bit-identical** (max abs diff `0.000e+00`) |
| Perturb a weight, confirm `cv2` sees it | yes (halving `conv0`, max diff 15.0) |
| Grow 21 -> 23 classes, reload | loads; 6 conf heads resized |
| **`person` after the surgery** | 0.928999 -> **0.928962** (Δ `3.6e-5`), over three frames |

`person` stays on the stop path, so it is the safety-critical class. Adding a class rather
than retraining the model means its features and logits are untouched and only the softmax
denominator moves — a guarantee by construction, not a validation run.

### ⚠️ The trap that would have cost a training run

Initialising a new class the obvious way — zero its weights, give it a large negative bias
— is wrong. With zeroed weights the new logit is that constant *everywhere*, and a
constant is a **floor**: on any prior whose trained logits all fall below it, the new class
wins. Measured at bias -50: **eight detections at confidence 1.000000**, on degenerate
slivers at the frame edge.

Seed from an existing class instead. `dog` is the default — the nearest thing VOC knows to
a quadruped, so it is also a warmer start than noise.

## Training data

Two sources, because neither is sufficient alone.

**`render_lite3.py` — synthetic Lite3.** Geometry from
[`DeepRoboticsLab/deep_robotics_model`](https://github.com/DeepRoboticsLab/deep_robotics_model)
(**BSD-3-Clause**) — the vendor's own MJCF and meshes, redistributable with attribution.
That is a strictly better source than their marketing photographs: a mesh can be posed and
ranged, a press shot cannot, and the licence is settled.

Only the *robot* is synthetic. It is composited into frames from the robot's own camera,
so the background carries the real lens geometry, the real blown-out corridor window, the
real sensor noise and the real JPEG artefacts — the domain gap is confined to one object.
Labels come from the segmentation buffer, so they are exact and free.

Three things it gets right that a naive generator does not:

* **Scale is derived.** A sprite rendered at a known distance through a known focal length
  fixes a physical extent; re-projecting that through the deployment camera gives the
  pixel size the real sensor would see. Nothing consults a published dimension, so nothing
  can disagree with one.
* **The robot stands on the floor.** Range and image row are not independent for an object
  resting on the ground. Sampling the row freely produces robots near the ceiling, and
  teaches the detector that apparent size carries no positional information.
* **35% of samples are truncated.** A peer close enough to avoid is usually clipped by the
  frame edge, and a set of whole robots omits the case that matters.

**Recorded frames.** The only route for the Go2 Wheel: it is a wheeled variant, Unitree
publishes no matching description, and the upstream Go2 driver repository ships no URDF
at all. About 60-90 usable frames exist today, all from one corridor.

### ⚠️ A background is a picture PLUS the camera that took it

The sprite is stood on the floor of the background frame, which needs that frame's focal
length, camera height and camera pitch. Those are properties of the FRAME. Get them wrong
and nothing looks broken — the composite is still a photograph of a corridor, the box is
still tight on the robot, and the robot is simply standing at the wrong depth.

So geometry travels with the frames. `--backgrounds` is repeatable, and each directory may
carry a `geometry.json`. That file is deliberately the format `calibrate_camera.py` already
writes, so a new environment is calibrated and copied in rather than retyped:

```sh
cp robot-stack/unitree/go2/visual_nav/go2_front_camera.json bg_lab/geometry.json
```

A directory without one falls back to the command line **and says so loudly**. A frame
whose pixel size disagrees with its declared geometry is refused rather than rescaled:
`focal_px` is pixels per radian at a particular capture size, and from the pixels alone a
downscale, a crop and a different lens are indistinguishable. Declaring that directory's
own `width`/`height`/`focal_px` makes its frames legal — what is refused is the assumption,
not the size.

Generated backgrounds, if they are ever supplied, are just files in a directory and arrive
the same way; nothing here knows where a background came from. But a generated frame has no
camera, so somebody must decide what geometry to claim for it, and `geometry.json` is where
that claim becomes reviewable instead of implicit.

**Recorded frames carry the debug overlay.** `visual_nav` writes its MP4 from the annotated
canvas — detection rectangles, the plan-view inset, the status plate. Composite onto those
and the detector learns that a peer comes with an orange rectangle. Use raw camera frames.

### ⚠️ Posture is part of the geometry

The Go2 rests **prone** and stands only to walk: it initialises prone, acquires its goal
prone, and lies down again whenever the path stays blocked for `--rest-after`. A dry run
never enables the legs at all, so it is prone start to finish whatever its status line
says. Prone is a recurring run state and prone frames are legitimate training data — from
a different camera:

| posture | `height_m` | `pitch_rad` | source |
| --- | --- | --- | --- |
| standing | 0.32 | 0.0 | `go2_front_camera.json` |
| prone | **0.1540** | **-0.0227** (1.3° nose-**up**) | tape, 2026-08-24 |

Neither prone number was recorded anywhere before. Applying the standing pair to a prone
frame puts the ground line **320 px high at 0.5 m** — a third of the frame, at the range
where avoidance happens.

**Sign convention:** `pitch_rad` is `camera_model.py`'s — tilt below the body's forward
axis, **positive = nose-down**. A nose-up lens is therefore negative. `--posture prone`
supplies the measured value already signed, which is the safe way to get it.

### ⚠️ `--focal-px` is a deployment parameter

It defaults to the Go2 Walk's measured 1290.2 px because that is the camera the backgrounds
came from. **A Lite3 Venture is a different lens at a different height**, and every apparent
size scales linearly on the focal length. Render at the wrong one and the detector learns a
scale prior wrong by exactly that ratio.

The same split applies to the model itself: a detector outputs boxes and transfers across
cameras, but **ranging does not**. Ship weights, never a calibration — each unit runs
`calibrate_camera.py` for itself.

### What the synthetic set does not model

The sprite is rendered through a pinhole camera at a narrow field of view; the real camera
is an equidistant fisheye. Over a small sprite that is slight, but a robot filling the
frame near the edge is genuinely distorted in a way this does not reproduce. **Treat the
close-range end as the weakest part of the set and keep recorded frames for it.** There
are also no contact shadows. The backgrounds are unaffected — they were imaged by the real
lens.

## Where this still has to reach

Detection is necessary and nowhere near sufficient. `mappo_bridge.stationary_objects`
filters to `kind == "static"`, and a detected peer becomes a track — `kind == "tracked"` —
so it is **dropped** before the policy sees it, arriving only as the `external_hold`
boolean that means *stop*. A perfect detector changes nothing until that path does.

And the policy's 18-dim observation has no obstacle-velocity channel, so a moving peer
enters as an instantaneous disc. Park the peer for the first runs.

## Usage

```sh
# 1. one protoc command, from weiliu89/caffe branch ssd (the SSD messages upstream lacks)
protoc --python_out=. caffe.proto

# 2. synthetic Lite3, composited onto peer-free frames from the deployment camera
python3 render_lite3.py --mjcf .../Lite3/mjcf/Lite3.xml \
        --backgrounds frames/ --out lite3_ds --count 2000 \
        --focal-px 1290.2 --posture standing

# 2b. several environments at once; each directory states its own camera in
#     geometry.json, so a prone clip and a standing clip can share one run
python3 render_lite3.py --mjcf .../Lite3/mjcf/Lite3.xml \
        --backgrounds corridor/ --backgrounds lab/ --backgrounds atrium/ \
        --out lite3_ds --count 4000

# 3. grow the shipped weights; person is preserved by construction
python3 add_class.py --in-proto MobileNetSSD_deploy.prototxt \
        --in-model MobileNetSSD_deploy.caffemodel \
        --out-proto mnssd23.prototxt --out-model mnssd23.caffemodel \
        --classes lite3 go2wheel

# 4. score checkpoints the way a LAUNCHER would see them: peer recall, false alarms and
#    both people denominators, once per configuration. --preprocessing is required and has
#    no default; profiles sharing a square come from one forward pass. `mobilenet-ssd-
#    trained` is run by no launcher and REFUSES without the reason flag, which is then
#    written into the output beside the numbers.
python3 inference_profile.py            # every configuration, and what runs it
python3 score_crossday.py --frames-dir XDAY --proto mnssd22.prototxt \
        --models 'runs/*/epoch*.caffemodel' --inventory-glob 'runs/*/*.caffemodel' \
        --preprocessing go2-peer-supervised --preprocessing go2-run-smoke \
        --preprocessing go2-navigator-default --preprocessing mobilenet-ssd-trained \
        --allow-preprocessing-mismatch 'the pair the 2026-08-26 sweep scored' \
        --out sweep.json
```

Pin `opencv-python` to **4.x** for any of this: OpenCV 5 has removed `readNetFromCaffe`.
The robot's 4.2 is unaffected.
