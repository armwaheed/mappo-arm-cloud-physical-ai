#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One-sided exceedance of the drop statistic on the parked null.

The robot is parked in every clip, so the true d(ln R)/dt is zero and every window
slope measured here is noise. The gate drops a track when the OBSERVED slope sits
REJECT_SIGMAS above the slope ego-motion predicts; with the robot parked that is
exactly "observed slope > REJECT_SIGMAS * sigma". So the numbers below are the
per-window false-drop rate the gate would pay, measured rather than assumed Gaussian.
"""
import collections
import json
import math
import os
import re
import sys

import numpy as np

#: Where the modules under test live. Overridable because these scripts run on the DGX
#: Spark, where the repo is not necessarily checked out next to the data.
VISUAL_NAV = os.environ.get(
    "VISUAL_NAV",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "robot-stack", "unitree", "go2", "visual_nav"))
sys.path.insert(0, VISUAL_NAV)
from camera_model import FisheyeCamera
from person_detector import PERSON_PRIOR, Detection, estimate_range


def _load_json(path):
    """Read a JSON file. A helper so these one-shot scripts still close their files."""
    with open(path) as handle:
        return json.load(handle)

D = os.environ.get("PEER_DATASET",
                   os.path.expanduser("~/go2-peer-dataset-20260824"))
CAM = FisheyeCamera.load(os.path.join(D, "artifacts", "models_robot",
                                      "go2_front_camera.json"))
DETS = _load_json(os.environ.get("DETECTIONS", "detections.json"))


def pref(n):
    return re.sub(r"_?\d{4}\.jpg$", "", n)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2 - ix1), max(0., iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.


stamps = {}
for p in os.listdir(D):
    if p.endswith(".jsonl"):
        with open(os.path.join(D, p)) as manifest:
            for line in manifest:
                record = json.loads(line)
                stamps[record["image"]] = record["t"]

clips = collections.defaultdict(list)
for f in sorted(DETS):
    clips[pref(f)].append(f)


def chain(frames):
    tracks, live = [], []
    for f in frames:
        t = stamps.get(f)
        if t is None:
            continue
        obs = []
        for label, score, x1, y1, x2, y2 in DETS[f]:
            det = Detection(x1, y1, x2, y2, score, label)
            rng, src = estimate_range(det, CAM, PERSON_PRIOR)
            if math.isfinite(rng):
                obs.append((t, rng, src, (x1, y1, x2, y2)))
        used, still = set(), []
        for tr in live:
            best, bi = 0.0, None
            for i, o in enumerate(obs):
                if i in used:
                    continue
                v = iou(tr[-1][3], o[3])
                if v > best:
                    best, bi = v, i
            if bi is not None and best >= 0.30:
                used.add(bi)
                tr.append(obs[bi])
                still.append(tr)
            else:
                tracks.append(tr)
        for i, o in enumerate(obs):
            if i not in used:
                still.append([o])
        live = still
    tracks.extend(live)
    return [tr for tr in tracks if len(tr) >= 5]


def runs(tr):
    out, cur = [], [tr[0]]
    for s in tr[1:]:
        if s[2] == cur[-1][2]:
            cur.append(s)
        else:
            out.append(cur)
            cur = [s]
    out.append(cur)
    return out


all_tracks = [tr for _, fr in sorted(clips.items()) for tr in chain(fr)]
homo = [r for tr in all_tracks for r in runs(tr) if len(r) >= 5]
steps = [math.log(b[1] / a[1]) for r in homo for a, b in zip(r, r[1:])]
SIG = float(np.std(steps)) / math.sqrt(2.0)
print(f"per-sample sigma(ln R), source held: {SIG:.4f} ({SIG*100:.2f}%) "
      f"from {len(steps)} frame pairs in {len(homo)} runs")
print()
print(f"{'n':>3} {'span s':>7} {'windows':>8} {'sd/s':>8} {'sigma_a/s':>10} "
      f"  one-sided exceedance of +k*sigma_a")
for n in (8, 12, 16, 20, 30):
    sl, sg = [], []
    for r in homo:
        for i in range(len(r) - n + 1):
            w = r[i:i + n]
            t = np.array([s[0] for s in w], dtype=float)
            t -= t.mean()
            sxx = float((t * t).sum())
            if sxx <= 0:
                continue
            y = np.log(np.array([s[1] for s in w], dtype=float))
            sl.append(float((t * (y - y.mean())).sum() / sxx))
            sg.append(SIG / math.sqrt(sxx))
    if not sl:
        continue
    sl, sg = np.array(sl), np.array(sg)
    z = sl / sg
    span = float(np.median([r[n - 1][0] - r[0][0] for r in homo
                            if len(r) >= n] or [0]))
    ex = "  ".join(f"{k}s:{100*np.mean(z > k):5.2f}%" for k in (2, 3, 4, 5))
    print(f"{n:>3} {span:>7.2f} {len(sl):>8} {np.std(sl):>8.4f} "
          f"{np.median(sg):>10.4f}   {ex}")

# The reach that follows from this sigma is NOT derived here. A hand-rolled version of
# that table was wrong: it assumed the drop condition was the ego-motion comparison
# alone, and the contact-horizon clause moves the answer by a factor. `reach.py` asks
# the real gate instead.
