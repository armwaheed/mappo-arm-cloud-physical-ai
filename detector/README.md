<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Teaching the perception stack to see a peer robot

Nothing in this stack can see another quadruped. That is measured, not assumed:

| detector | best response to a Go2 Wheel filling half the frame |
| --- | --- |
| MobileNet-SSD / VOC-21 (**what the robot runs**) | **nothing at any score down to 0.02** |
| YOLO11n / COCO-80 | `motorcycle` 0.138, `bicycle` 0.112 |
| YOLO-World, prompted "robot dog" / "quadruped robot" | `robotic dog` **0.039** |

The open-vocabulary row is the one that settles it. That model, on the same frames, with
the same code path, scores `person` **0.932** — so the pipeline is verified and the null
result is real. There is no off-the-shelf model that knows what a Lite3 is. Across 148
ticks of two runs staged specifically to record a crossing peer, the telemetry contains
`bin` and `person` and nothing else.

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

### ⚠️ `--focal-px` and `--camera-height-m` are deployment parameters

They default to the Go2 Walk's measured 1290.2 px and 0.32 m because that is the camera
the backgrounds came from. **A Lite3 Venture is a different lens at a different height**,
and every apparent size scales linearly on the focal length. Render at the wrong one and
the detector learns a scale prior wrong by exactly that ratio.

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
        --focal-px 1290.2 --camera-height-m 0.32

# 3. grow the shipped weights; person is preserved by construction
python3 add_class.py --in-proto MobileNetSSD_deploy.prototxt \
        --in-model MobileNetSSD_deploy.caffemodel \
        --out-proto mnssd23.prototxt --out-model mnssd23.caffemodel \
        --classes lite3 go2wheel
```

Pin `opencv-python` to **4.x** for any of this: OpenCV 5 has removed `readNetFromCaffe`.
The robot's 4.2 is unaffected.
