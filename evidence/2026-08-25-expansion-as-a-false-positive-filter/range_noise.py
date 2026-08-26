#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Same measurement as jitter.py, but split by whether the RANGING SOURCE held.

jitter.py found the range series has a 1.17% median step and a 104% p95. The tail is
not the detector. It is estimate_range switching prior between frames.
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
                obs.append((t, rng, src, (x1, y1, x2, y2), y2 - y1))
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


all_tracks = [tr for tag, fr in sorted(clips.items()) for tr in chain(fr)]
print(f"{len(all_tracks)} tracks, {sum(len(t) for t in all_tracks)} samples")

same, switch = [], []
hstep = []
for tr in all_tracks:
    for a, b in zip(tr, tr[1:]):
        step = math.log(b[1] / a[1])
        (same if a[2] == b[2] else switch).append(step)
        if a[2] == b[2]:
            hstep.append(math.log(b[4] / a[4]))
for name, arr in (("SOURCE HELD", same), ("SOURCE SWITCHED", switch)):
    v = np.abs(np.array(arr))
    print(f"{name:18s} n={len(v):5d}  |dlnR| median={np.median(v)*100:7.2f}%  "
          f"p95={np.percentile(v,95)*100:8.2f}%  p99={np.percentile(v,99)*100:8.2f}%  "
          f"sd={np.std(arr)*100:8.2f}%")
h = np.abs(np.array(hstep))
print(f"{'BOX HEIGHT (source held)':18s} n={len(h):5d}  |dln h_px| median="
      f"{np.median(h)*100:.2f}%  p95={np.percentile(h,95)*100:.2f}%  "
      f"sd={np.std(hstep)*100:.2f}%")


def runs(tr):
    """Split a track at every source change: only source-homogeneous runs."""
    out, cur = [], [tr[0]]
    for s in tr[1:]:
        if s[2] == cur[-1][2]:
            cur.append(s)
        else:
            out.append(cur)
            cur = [s]
    out.append(cur)
    return out


homo = [r for tr in all_tracks for r in runs(tr) if len(r) >= 5]
print(f"\nsource-homogeneous runs of >=5: {len(homo)}, "
      f"{sum(len(r) for r in homo)} samples "
      f"({collections.Counter(r[0][2] for r in homo)})")
dts = [b[0] - a[0] for r in homo for a, b in zip(r, r[1:])]
dt = float(np.median(dts))
sd_step = float(np.std([math.log(b[1] / a[1]) for r in homo
                        for a, b in zip(r, r[1:])]))
sd_ln = sd_step / math.sqrt(2.0)
print(f"median frame interval {dt*1000:.0f} ms; per-sample sd(ln R) = {sd_ln*100:.2f}%")
print(f"\n{'n':>3} {'window s':>9} {'slope sd /s':>12} {'p99|slope| /s':>14} "
      f"{'iid pred /s':>12} {'ratio':>6}")
for n in (5, 8, 12, 20, 30):
    sl = []
    for r in homo:
        for i in range(len(r) - n + 1):
            w = r[i:i + n]
            t = np.array([s[0] for s in w])
            t = t - t[0]
            if t[-1] <= 0:
                continue
            sl.append(float(np.polyfit(t, np.log([s[1] for s in w]), 1)[0]))
    if not sl:
        print(f"{n:>3} {'-':>9} (no run this long)")
        continue
    sl = np.array(sl)
    iid = sd_ln * math.sqrt(12.0 / (n ** 3 - n)) / dt
    print(f"{n:>3} {n*dt:>9.2f} {np.std(sl):>12.4f} "
          f"{np.percentile(np.abs(sl),99):>14.4f} {iid:>12.4f} "
          f"{np.std(sl)/iid:>6.2f}x   (n_win={len(sl)})")
