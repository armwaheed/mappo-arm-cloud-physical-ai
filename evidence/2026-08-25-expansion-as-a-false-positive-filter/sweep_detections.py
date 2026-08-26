#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dump every MobileNet-SSD detection on the peer dataset, class-agnostically.

Output: one JSON entry per frame name -> list of [label, score, x1, y1, x2, y2], so a
threshold sweep costs no inference.

⚠️ THE CONFIDENCE FLOOR BELOW IS NOT THE ONE THAT BINDS. `MobileNetSSD_deploy.prototxt`
sets `confidence_threshold: 0.25` in its `detection_out` layer, so the network never
emits anything under 0.25 and every "floor" passed here that is lower is inert. To see
the real floor, copy the model directory and lower the prototxt first:

    mkdir -p models_low && cp $PEER_DATASET/artifacts/models_robot/*.caffemodel models_low/
    sed 's/confidence_threshold: 0.25/confidence_threshold: 0.01/' \
        $PEER_DATASET/artifacts/models_robot/MobileNetSSD_deploy.prototxt \
        > models_low/MobileNetSSD_deploy.prototxt
    MODELS=models_low DETECTIONS=detections_low.json python3 sweep_detections.py

That is the difference between "zero false positives down to 0.05" (which measures
nothing) and the real answer, which is zero down to 0.15 and a 0.132 ceiling.
"""
import glob
import json
import os
import sys
import time

#: Where the modules under test live. Overridable because these scripts run on the DGX
#: Spark, where the repo is not necessarily checked out next to the data.
VISUAL_NAV = os.environ.get(
    "VISUAL_NAV",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "robot-stack", "unitree", "go2", "visual_nav"))
sys.path.insert(0, VISUAL_NAV)
from person_detector import VOC_CLASSES, PersonDetector

D = os.environ.get("PEER_DATASET",
                   os.path.expanduser("~/go2-peer-dataset-20260824"))
MODELS = os.environ.get("MODELS", os.path.join(D, "artifacts", "models_robot"))
OUT = os.environ.get("DETECTIONS", "detections.json")

import cv2

det = PersonDetector(MODELS, input_size=300,
                     confidence=float(os.environ.get("FLOOR", 0.011)),
                     classes=tuple(c for c in VOC_CLASSES if c != "background"))

frames = sorted(os.path.basename(p) for p in glob.glob(D + "/*.jpg"))
print(f"{len(frames)} frames", flush=True)
out = {}
t0 = time.time()
for i, name in enumerate(frames):
    img = cv2.imread(os.path.join(D, name))
    if img is None:
        print("UNREADABLE", name, flush=True)
        continue
    out[name] = [[d.label, round(d.score, 5), round(d.x1, 1), round(d.y1, 1),
                  round(d.x2, 1), round(d.y2, 1)] for d in det.detect(img)]
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(frames)}  {time.time()-t0:.0f}s", flush=True)
with open(OUT, "w") as handle:
    json.dump(out, handle)
print("WROTE", OUT, time.time() - t0, flush=True)
