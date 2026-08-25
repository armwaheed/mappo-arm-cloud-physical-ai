<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Teaching the perception stack to see a peer robot

No off-the-shelf detector has a **name** for another quadruped. That much is measured,
not assumed — though the first row is wrong, and why it is wrong turned out to be the
whole story:

| detector | best response to a Go2 Wheel filling half the frame |
| --- | --- |
| MobileNet-SSD / VOC-21 (**what the robot runs**) | **nothing at any score down to 0.02** |
| YOLO11n / COCO-80 | `motorcycle` 0.138, `bicycle` 0.112 |
| YOLO-World, prompted "robot dog" / "quadruped robot" | `robotic dog` **0.039** |

The open-vocabulary row is the one that settles it. That model, on the same frames, with
the same code path, scores `person` **0.932** — so the pipeline is verified and the null
result is real. There is no off-the-shelf model that knows what a Lite3 is.

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
publishes no matching description, and `arm-mhs-unitree-go2` ships no URDF at all. About
60-90 usable frames exist today, all from one corridor.

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
```

Pin `opencv-python` to **4.x** for any of this: OpenCV 5 has removed `readNetFromCaffe`.
The robot's 4.2 is unaffected.
