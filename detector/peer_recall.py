#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Score the stock 21-class detector on a peer robot, per FRAME and per TRACK.

Two questions, and the second is the one the planner actually asks.

1. THE BAKED-IN FLOOR. ``MobileNetSSD_deploy.prototxt`` carries
   ``confidence_threshold: 0.25`` inside its ``DetectionOutput`` layer, so the network
   cannot emit a box below 0.25 however low a caller sets its own threshold. Every
   "confidence 0.15" figure this project has produced is therefore really 0.25 -- the
   two rows are the same forward pass. ``verify`` below proves that rather than asserting
   it. Lowering that one line to 0.05 costs no retraining and no GPU, and ``score``
   measures what appears underneath.

2. FRAMES ARE NOT WHAT THE PLANNER SEES. It consumes ``ObstacleTracker`` tracks, which
   have a confirm/coast lifecycle: ``CONFIRM_HITS`` detections before a track is planned
   against, ``MAX_MISSES``/``COAST_TIMEOUT_S`` before one is dropped. That should turn a
   flickering per-frame recall into a near-continuous per-track one, and should suppress
   an UNCORRELATED false alarm -- a different scrap of background each frame, which never
   accumulates two associated hits -- while leaving a CORRELATED one untouched. The two
   have opposite consequences: the first is noise the lifecycle absorbs, the second is a
   phantom obstacle the planner steers around for as long as it is in shot. Only
   track-level scoring tells them apart, and this corpus contains a textbook example of
   the second (an office chair, parked mid-corridor, wearing an ArUco marker).

WHAT THIS CORPUS CAN AND CANNOT SUPPORT. The frames are STAGED STILLS grouped by
position, not walking footage. Ten of the eleven labelled segments are rigid-static --
the peer does not move and neither does the camera robot (the corpus's own
``LABELLING.md``: worst-case global phase correlation against frame 0 is 1.5 px, and
exactly 0.0 px for three segments). The tracker is therefore run at a FIXED pose, which
is measured rather than assumed. The consequence is a real limit on what the track-level
numbers mean: they exercise the confirm/coast lifecycle and the association gate ONLY.
They do NOT exercise the constant-velocity motion model, the odom-frame ego-motion
cancellation that is the module's stated reason for existing, or ``is_visible``'s
out-of-shot coasting -- none of which can move when neither robot does.
``p4_mid_sweep_stand`` (208 frames, ground-truth x0 sweeping 157 -> 1308) is the one
segment with real target motion, and is reported separately for exactly that reason.

Usage::

    # one forward pass per prototxt, cached so every threshold scores the same passes
    peer_recall.py cache  --prototxt PROTO --weights W --frames DIR --out cache.json
    peer_recall.py verify --low cache_t005.json --high cache_t025.json
    peer_recall.py score  --cache cache_t005.json --labels L.json --root DIR \\
                          --visual-nav robot-stack/unitree/go2/visual_nav --camera C.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import sys
import time

# ── The corpus's own bookkeeping ────────────────────────────────────────────
# Segments whose frames carry NO peer label but DO contain the peer. Both are named in
# the corpus's own LABELLING.md and both were confirmed by looking at the pixels:
#   smoke   -- "a byte-for-byte duplicate viewpoint of peer01", peer in the same place;
#              58 frames dropped from the label file "as instructed".
#   p1b_..  -- not described in LABELLING.md at all; the peer fills over half the frame.
# Counting either as peer-free scores a CORRECT detection as a false alarm, and that is
# not a rounding error: it is the whole of the previously reported false-alarm rate.
CONTAMINATED_NEGATIVES = ("smoke", "p1b_close_broadside_STANDING")

#: Genuinely empty corridor, confirmed by eye. The honest false-alarm set.
CLEAN_NEGATIVES = ("neg_prone", "neg_standing")

#: Median inter-frame gap over every capture manifest in the corpus, seconds (13.9 Hz).
#: Used only where a manifest is missing or truncated.
DEFAULT_DT_S = 0.072

# Size prior for the peer. Both numbers are stated rather than derived, and the metric
# scale of every range below is proportional to them.
#
# WIDTH IS GIVEN EXPLICITLY, NOT VIA SizePrior.of_height. That helper fills the width in
# from a PERSON's aspect ratio (0.50/1.70), which for a 0.40 m quadruped yields 0.118 m
# -- against a real body width nearer 0.31 m, and a broadside LENGTH nearer 0.70 m. The
# width prior is what ranges a vertically-clipped box, and 39% of this corpus's boxes
# touch a frame border, so the person aspect ratio puts every close approach at roughly
# a third of its true distance.
PEER_HEIGHT_M = 0.40
PEER_WIDTH_M = 0.31

VOC = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
       "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
       "pottedplant", "sheep", "sofa", "train", "tvmonitor")

SWEEP = (0.05, 0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20, 0.25,
         0.40, 0.50)
TRACK_SWEEP = (0.05, 0.10, 0.15, 0.25)


def segment_of(filename: str) -> str:
    """The staged position a frame belongs to: ``p1_close_broadside_0007`` -> prefix."""
    return filename[:-4].rsplit("_", 1)[0]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


# ── Inference ───────────────────────────────────────────────────────────────
def cmd_cache(args) -> None:
    """One forward pass per frame; cache every detection the prototxt lets out.

    Caching rather than re-running per threshold is not just a speed trick: it
    guarantees every row of the sweep is scored against the SAME forward passes, so a
    difference between two rows cannot be an inference difference.
    """
    import cv2

    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
    frames = sorted(f for f in os.listdir(args.frames) if f.endswith(".jpg"))
    print(f"{len(frames)} frames, prototxt={args.prototxt}", flush=True)

    cache = {}
    started = time.time()
    for index, name in enumerate(frames):
        image = cv2.imread(os.path.join(args.frames, name))
        if image is None:
            continue
        height, width = image.shape[:2]
        # Exactly the robot's own preprocessing: person_detector.PersonDetector.detect
        # hands the full frame to blobFromImage and lets it do the resize.
        blob = cv2.dnn.blobFromImage(image, 1.0 / 127.5, (300, 300), 127.5)
        net.setInput(blob)
        raw = net.forward()
        rows = []
        for _, class_id, score, x1, y1, x2, y2 in raw[0, 0]:
            cid = int(class_id)
            if cid <= 0:
                continue
            rows.append([VOC[cid] if cid < len(VOC) else f"?{cid}",
                         round(float(score), 6),
                         round(float(x1) * width, 2), round(float(y1) * height, 2),
                         round(float(x2) * width, 2), round(float(y2) * height, 2)])
        cache[name] = rows
        if index % 400 == 0:
            print(f"  {index}/{len(frames)}", flush=True)

    with open(args.out, "w") as handle:
        json.dump({"prototxt": args.prototxt, "frames": len(cache), "dets": cache},
                  handle)
    print(f"wrote {args.out} in {time.time() - started:.0f}s", flush=True)


def cmd_verify(args) -> None:
    """Prove the low-threshold prototxt only ADDS boxes, and changes none that existed.

    This is the guard against the one way lowering ``confidence_threshold`` could have
    reduced recall and produced a non-monotonic sweep: ``keep_top_k: 100`` caps the
    detections kept across all classes, so in principle a flood of new low-scoring boxes
    could evict a high-scoring one. NMS and keep_top_k both rank by score, so it should
    not happen -- but "should not" is not a measurement, and a non-monotonic result would
    otherwise look like a finding instead of the bug it is.
    """
    low = load_json(args.low)["dets"]
    high = load_json(args.high)["dets"]

    def key(row):
        return (row[0], round(row[1], 4), *(round(v, 1) for v in row[2:6]))

    differing = lost = extra = 0
    for name, rows in low.items():
        below = sorted(key(r) for r in rows if r[1] >= args.at)
        above = sorted(key(r) for r in high.get(name, []))
        if below != above:
            differing += 1
            lost += len(set(above) - set(below))
            extra += len(set(below) - set(above))
    print(f"frames compared            : {len(low)}")
    print(f"frames differing at >={args.at:.2f}  : {differing}")
    print(f"boxes in HIGH not in LOW   : {lost}   "
          "(a non-zero here is keep_top_k eviction)")
    print(f"boxes in LOW not in HIGH   : {extra}")
    print(f"total detections, LOW      : {sum(len(v) for v in low.values())}")
    print(f"total detections, HIGH     : {sum(len(v) for v in high.values())}")
    if differing == 0:
        print("\nIDENTICAL above the high prototxt's own floor. The edit is purely "
              "additive,\nso the sweep is monotonic by construction -- and the old "
              "0.15 and 0.25 rows\nwere the same forward pass.")


# ── Frame level ─────────────────────────────────────────────────────────────
def frame_sweep(dets, gt, positives, negatives, thresholds, iou_min):
    """Recall on labelled frames, false-alarm rate on peer-free frames."""
    table = []
    for threshold in thresholds:
        on_peer = any_det = 0
        classes = collections.Counter()
        for name in positives:
            keep = [r for r in dets[name] if r[1] >= threshold]
            if keep:
                any_det += 1
            landed = [r for r in keep if iou(r[2:6], gt[name]) >= iou_min]
            if landed:
                on_peer += 1
                for row in landed:
                    classes[row[0]] += 1
        alarms = sum(1 for n in negatives if any(r[1] >= threshold for r in dets[n]))
        boxes = sum(len([r for r in dets[n] if r[1] >= threshold]) for n in negatives)
        table.append({"conf": threshold, "on_peer": on_peer, "any_det": any_det,
                      "n_pos": len(positives), "fa": alarms, "fa_boxes": boxes,
                      "n_neg": len(negatives),
                      "classes": dict(classes.most_common(6))})
    return table


def per_segment(dets, gt, positives, thresholds, iou_min):
    """Recall broken out by staged position. The far segments are the whole question."""
    segments = collections.defaultdict(list)
    for name in positives:
        segments[segment_of(name)].append(name)
    rows = []
    for segment in sorted(segments):
        frames = segments[segment]
        row = {"segment": segment, "n": len(frames)}
        for threshold in thresholds:
            hit = sum(1 for f in frames
                      if any(r[1] >= threshold and iou(r[2:6], gt[f]) >= iou_min
                             for r in dets[f]))
            row[f"r{threshold:.2f}"] = hit / len(frames)
        row["best_on_peer_conf"] = max(
            max((r[1] for r in dets[f] if iou(r[2:6], gt[f]) >= iou_min), default=0.0)
            for f in frames)
        rows.append(row)
    return rows


# ── Track level ─────────────────────────────────────────────────────────────
def gt_bearing_span(box, camera):
    """Angular extent of a ground-truth box: ``(left, right)`` bearings in radians.

    Matching a track to the peer on BEARING rather than on a reprojected pixel box is
    deliberate. The tracker's own noise model calls bearing the accurate axis and a
    size-prior range the weak one, and 39% of these boxes touch a frame border -- for
    those, range comes from ``width-capped`` or ``frame-fill``, which are CONSTANTS
    rather than measurements. Reprojecting a track into pixels would fold the size
    prior's scale error straight into the scoring criterion; bearing does not depend on
    the prior at all.

    The cost is that this criterion is LOOSER than IoU >= 0.30, so it is also applied
    per frame (:func:`frame_bearing_hits`). Without that, comparing frames to tracks
    would change the lifecycle and the criterion at once, and credit the tracker with
    the difference.
    """
    x1, _, x2, _ = box
    left, _ = camera.bearing_elevation(float(x1), float(camera.height / 2))
    right, _ = camera.bearing_elevation(float(x2), float(camera.height / 2))
    return min(float(left), float(right)), max(float(left), float(right))


def run_tracker(frames, times, dets, camera, prior, conf, collapse_labels,
                max_range_m, tracker_mod, detector_mod):
    """Feed one staged segment through the real ObstacleTracker, in capture order.

    ``collapse_labels`` rewrites every detection's class to one name. The tracker's
    association step refuses to match an observation to a track of a DIFFERENT label,
    and a class-agnostic policy -- the whole premise here -- hands it a stream whose
    class flips between frames and between boxes on one object. The switch measures what
    that costs, which turns out to be track COUNT rather than track recall.
    """
    tracker = tracker_mod.ObstacleTracker(max_range_m=max_range_m)
    pose = (0.0, 0.0, 0.0)          # measured static; see the module docstring
    out = []
    previous = None
    beyond_range = 0
    sources = collections.Counter()
    for name, stamp in zip(frames, times):
        tracker.predict(0.0 if previous is None else max(0.0, stamp - previous))
        previous = stamp
        raw = [r for r in dets[name] if r[1] >= conf]
        detections = [detector_mod.Detection(
            x1=min(max(r[2], 0.0), camera.width - 1.0),
            y1=min(max(r[3], 0.0), camera.height - 1.0),
            x2=min(max(r[4], 0.0), camera.width - 1.0),
            y2=min(max(r[5], 0.0), camera.height - 1.0),
            score=r[1], label="obstacle" if collapse_labels else r[0]) for r in raw]
        observations = []
        for ranged in detector_mod.range_detections(detections, camera, prior):
            sources[ranged.source] += 1
            if ranged.range_m > max_range_m:
                beyond_range += 1
            observations.append(tracker_mod.observation_from(
                ranged.bearing_rad, ranged.range_m, ranged.source, ranged.label, pose))
        tracker.update(observations, now=stamp, robot_x=0.0, robot_y=0.0, robot_yaw=0.0)
        out.append({
            "frame": name, "n_det": len(raw), "n_live": len(tracker.tracks),
            "confirmed": [(t.track_id, float(math.atan2(t.state[1], t.state[0])),
                           float(math.hypot(t.state[0], t.state[1])))
                          for t in tracker.confirmed_tracks()],
        })
    return out, {"beyond_range": beyond_range, "sources": dict(sources)}


def score_segment_tracks(per_frame, gt, camera, margin_rad):
    """Frames with a confirmed track on the peer, and how MANY sat on it at once.

    The multiplicity matters as much as the hit rate. One physical peer arriving as four
    simultaneous confirmed tracks is four obstacles to the planner, which is not a
    detection success however good the recall column looks.
    """
    hit = 0
    multiplicity = []
    for row in per_frame:
        box = gt.get(row["frame"])
        if box is None:
            continue
        low, high = gt_bearing_span(box, camera)
        on_peer = [t for t in row["confirmed"]
                   if low - margin_rad <= t[1] <= high + margin_rad]
        if on_peer:
            hit += 1
        multiplicity.append(len(on_peer))
    return hit, multiplicity


def frame_bearing_hits(frames, dets, gt, camera, conf, margin_rad):
    """The bearing criterion applied per FRAME, so frame vs track is like-for-like."""
    hit = 0
    for name in frames:
        box = gt.get(name)
        if box is None:
            continue
        low, high = gt_bearing_span(box, camera)
        for row in dets[name]:
            if row[1] < conf:
                continue
            bearing, _ = camera.bearing_elevation(float((row[2] + row[4]) / 2.0),
                                                  float(camera.height / 2))
            if low - margin_rad <= float(bearing) <= high + margin_rad:
                hit += 1
                break
    return hit


def track_lifetimes(per_frame):
    """How long each confirmed track survived, and at what bearing.

    A phantom lasting one or two frames is the uncorrelated false alarm the lifecycle is
    meant to absorb. One holding a stable bearing across a whole segment is a CORRELATED
    false alarm -- a real object read as the wrong class -- and no lifecycle tuning
    removes it, because every premise the lifecycle relies on is satisfied.
    """
    seen = collections.defaultdict(list)
    for row in per_frame:
        for track_id, bearing, range_m in row["confirmed"]:
            seen[track_id].append((bearing, range_m))
    return {track_id: {
        "frames": len(samples),
        "bearing_deg": math.degrees(statistics.median(b for b, _ in samples)),
        "bearing_spread_deg": math.degrees(max(b for b, _ in samples)
                                           - min(b for b, _ in samples)),
        "range_m": statistics.median(r for _, r in samples)}
        for track_id, samples in seen.items()}


def load_times(root, frames_by_segment):
    """Per-frame capture times from the capture manifests, falling back to a constant.

    The manifests are the only record of the real frame rate: filesystem mtimes have 1 s
    resolution here because the corpus was copied in bulk. ``peer01.jsonl`` is truncated
    (585 lines for 640 images) and the negative segments have no manifest at all, so
    those frames get :data:`DEFAULT_DT_S`, the median gap over every manifest that does
    exist.
    """
    times = {}
    for segment, frames in frames_by_segment.items():
        stamps = {}
        manifest = os.path.join(root, segment + ".jsonl")
        if os.path.isfile(manifest):
            with open(manifest) as handle:
                for line in handle:
                    row = json.loads(line)
                    stamps[row["image"]] = row["t"]
        base = min(stamps.values()) if stamps else 0.0
        clock = 0.0
        for index, name in enumerate(sorted(frames)):
            if name in stamps:
                clock = stamps[name] - base
            else:
                clock = clock + DEFAULT_DT_S if index else 0.0
            times[name] = clock
    return times


def cmd_score(args) -> None:
    sys.path.insert(0, args.visual_nav)
    import camera_model
    import person_detector
    import tracker as tracker_mod

    camera = camera_model.FisheyeCamera.load(args.camera)
    prior = person_detector.SizePrior(height_m=args.peer_height,
                                      width_m=args.peer_width)

    dets = load_json(args.cache)["dets"]
    gt = {r["image"]: r["box"] for r in load_json(args.labels)["records"]}

    all_frames = sorted(dets)
    positives = [f for f in all_frames if f in gt]
    unlabelled = [f for f in all_frames if f not in gt]
    clean_neg = [f for f in unlabelled if segment_of(f) in CLEAN_NEGATIVES]
    dirty_neg = [f for f in unlabelled if segment_of(f) in CONTAMINATED_NEGATIVES]

    report = {"iou_min": args.iou, "peer_height_m": args.peer_height,
              "peer_width_m": args.peer_width, "n_frames": len(all_frames),
              "n_positive": len(positives), "n_unlabelled": len(unlabelled),
              "n_clean_negative": len(clean_neg),
              "n_contaminated_negative": len(dirty_neg)}

    print("=" * 79)
    print("CORPUS")
    print("=" * 79)
    print(f"  {len(all_frames)} frames: {len(positives)} labelled peer, "
          f"{len(unlabelled)} unlabelled")
    print(f"  of the unlabelled: {len(clean_neg)} genuinely peer-free  "
          f"{list(CLEAN_NEGATIVES)}")
    print(f"                     {len(dirty_neg)} CONTAIN THE PEER, unlabelled  "
          f"{list(CONTAMINATED_NEGATIVES)}")
    print(f"  -> honest false-alarm denominator is {len(clean_neg)}, "
          f"not {len(unlabelled)}")

    # The separability ceiling: the best score the detector ever gives to something that
    # is NOT the peer. No global threshold admits a peer scoring below this without also
    # admitting that object, so it bounds what lowering the floor can buy.
    ceiling = max(((r[1], f, r[0], [round(v) for v in r[2:6]])
                   for f in clean_neg for r in dets[f]), default=(0.0, "", "", []))
    report["clean_negative_ceiling"] = {"conf": ceiling[0], "frame": ceiling[1],
                                        "label": ceiling[2], "box": ceiling[3]}

    print()
    print("=" * 79)
    print(f"MEASUREMENT 1 -- FRAME LEVEL, class-agnostic, IoU >= {args.iou:.2f}")
    print("=" * 79)
    sweep = frame_sweep(dets, gt, positives, clean_neg, SWEEP, args.iou)
    report["frame_sweep_clean"] = sweep
    report["frame_sweep_all_unlabelled"] = frame_sweep(dets, gt, positives, unlabelled,
                                                       SWEEP, args.iou)
    header = ("conf", "recall (box ON peer)", "any det, peer frame",
              "FALSE ALARM (clean)", "FA boxes")
    print(f"{header[0]:<6} {header[1]:<21} {header[2]:<21} {header[3]:<21} "
          f"{header[4]:>8}")
    for row in sweep:
        recall = (f"{row['on_peer']:>4}/{row['n_pos']} = "
                  f"{100.0 * row['on_peer'] / row['n_pos']:>5.1f}%")
        anydet = (f"{row['any_det']:>4}/{row['n_pos']} = "
                  f"{100.0 * row['any_det'] / row['n_pos']:>5.1f}%")
        alarms = (f"{row['fa']:>3}/{row['n_neg']} = "
                  f"{100.0 * row['fa'] / row['n_neg']:>5.1f}%")
        print(f"{row['conf']:<6.2f} {recall:<21} {anydet:<21} {alarms:<21} "
              f"{row['fa_boxes']:>8}")
    print()
    print(f"  classes landing on the peer @0.05: {sweep[0]['classes']}")
    print("  SEPARABILITY CEILING -- best score on anything that is NOT the peer,")
    print(f"  across all {len(clean_neg)} genuinely peer-free frames: "
          f"{ceiling[0]:.4f}  ({ceiling[2]} {ceiling[3]} in {ceiling[1]})")
    print("  No global threshold admits a peer scoring below that without also")
    print("  admitting this object.")
    print()
    print(f"  the SAME sweep scored the old way, all {len(unlabelled)} unlabelled "
          "counted peer-free:")
    for row in report["frame_sweep_all_unlabelled"]:
        if row["conf"] in (0.15, 0.25, 0.40):
            rate = 100.0 * row["fa"] / row["n_neg"]
            print(f"     conf {row['conf']:.2f}  false alarm {row['fa']:>3}/"
                  f"{row['n_neg']} = {rate:>5.1f}%   <-- every one of these is a "
                  "CORRECT detection")

    print()
    print("=" * 79)
    print("MEASUREMENT 1b -- WHERE THE RECALL IS, by staged position")
    print("=" * 79)
    seg_rows = per_segment(dets, gt, positives, (0.05, 0.15, 0.25, 0.40), args.iou)
    report["per_segment"] = seg_rows
    print(f"{'segment':<36} {'n':>5} {'@0.05':>7} {'@0.15':>7} {'@0.25':>7} "
          f"{'@0.40':>7} {'best conf':>10}")
    for row in seg_rows:
        flag = "" if row["best_on_peer_conf"] > ceiling[0] else "  <-- BELOW CEILING"
        print(f"{row['segment']:<36} {row['n']:>5} "
              f"{100 * row['r0.05']:>6.0f}% {100 * row['r0.15']:>6.0f}% "
              f"{100 * row['r0.25']:>6.0f}% {100 * row['r0.40']:>6.0f}% "
              f"{row['best_on_peer_conf']:>10.3f}{flag}")

    print()
    print("=" * 79)
    print("MEASUREMENT 2 -- TRACK LEVEL, through the real ObstacleTracker")
    print("=" * 79)
    print(f"  CONFIRM_HITS={tracker_mod.CONFIRM_HITS}  "
          f"MAX_MISSES={tracker_mod.MAX_MISSES}  "
          f"COAST_TIMEOUT_S={tracker_mod.COAST_TIMEOUT_S:.1f}  "
          f"max_range={args.max_range:.1f} m")
    print("  fixed robot pose (both robots measured static); peer prior "
          f"{args.peer_height:.2f} x {args.peer_width:.2f} m")

    by_segment = collections.defaultdict(list)
    for name in all_frames:
        by_segment[segment_of(name)].append(name)
    times = load_times(args.root, by_segment)
    margin = tracker_mod.BEARING_SIGMA_RAD

    report["tracks"] = {}
    for conf in TRACK_SWEEP:
        rows = []
        for segment in sorted(by_segment):
            frames = sorted(by_segment[segment])
            stamps = [times[f] for f in frames]
            labelled = segment not in (*CLEAN_NEGATIVES, *CONTAMINATED_NEGATIVES)
            row = {"segment": segment, "n": len(frames), "labelled": labelled}
            for collapse in (False, True):
                per_frame, stats = run_tracker(frames, stamps, dets, camera, prior, conf,
                                               collapse, args.max_range, tracker_mod,
                                               person_detector)
                lifetimes = track_lifetimes(per_frame)
                entry = {"frames_with_confirmed": sum(1 for r in per_frame
                                                      if r["confirmed"]),
                         "n_confirmed_ids": len(lifetimes),
                         "beyond_max_range": stats["beyond_range"],
                         "sources": stats["sources"], "lifetimes": lifetimes}
                if labelled:
                    hit, mult = score_segment_tracks(per_frame, gt, camera, margin)
                    entry["on_peer_frames"] = hit
                    entry["max_simultaneous"] = max(mult, default=0)
                    entry["median_simultaneous"] = (statistics.median(mult) if mult
                                                    else 0)
                row["collapsed" if collapse else "asis"] = entry
            if labelled:
                row["frame_bearing_hits"] = frame_bearing_hits(frames, dets, gt, camera,
                                                               conf, margin)
            rows.append(row)
        report["tracks"][f"{conf:.2f}"] = rows

    print()
    print("  RECALL, held to ONE criterion (bearing) so the lifecycle is the only thing")
    print("  changing between columns. 'mult' = median simultaneous confirmed tracks")
    print("  sitting on the one physical peer.")
    print(f"  {'conf':<6} {'per-frame':>10} {'track as-is':>12} {'collapsed':>12} "
          f"{'mult':>6} {'maxm':>6} {'conf ids':>8}")
    for conf in TRACK_SWEEP:
        totals = collections.Counter()
        mults = []
        worst = 0
        ids = 0
        for row in report["tracks"][f"{conf:.2f}"]:
            if not row["labelled"]:
                continue
            totals["n"] += row["n"]
            totals["frame"] += row["frame_bearing_hits"]
            totals["asis"] += row["asis"]["on_peer_frames"]
            totals["coll"] += row["collapsed"]["on_peer_frames"]
            mults.append(row["asis"]["median_simultaneous"])
            worst = max(worst, row["asis"]["max_simultaneous"])
            ids += row["asis"]["n_confirmed_ids"]
        print(f"  {conf:<6.2f} {100.0 * totals['frame'] / totals['n']:>9.1f}% "
              f"{100.0 * totals['asis'] / totals['n']:>11.1f}% "
              f"{100.0 * totals['coll'] / totals['n']:>11.1f}% "
              f"{statistics.median(mults):>6.1f} {worst:>6} {ids:>8}")

    print()
    print("  p4_mid_sweep_stand ONLY -- the single segment with real target motion, and")
    print("  so the only one where the lifecycle does more than hold still:")
    print(f"  {'conf':<6} {'per-frame':>10} {'track as-is':>12} {'collapsed':>12}")
    for conf in TRACK_SWEEP:
        for row in report["tracks"][f"{conf:.2f}"]:
            if row["segment"] != "p4_mid_sweep_stand":
                continue
            print(f"  {conf:<6.2f} "
                  f"{100.0 * row['frame_bearing_hits'] / row['n']:>9.0f}% "
                  f"{100.0 * row['asis']['on_peer_frames'] / row['n']:>11.0f}% "
                  f"{100.0 * row['collapsed']['on_peer_frames'] / row['n']:>11.0f}%")

    print()
    print("  PEER-FREE SEGMENTS -- phantom obstacles handed to the planner")
    print(f"  {'segment':<16} {'conf':<6} {'n':>5} {'fr w/ det':>9} "
          f"{'fr w/ conf':>10} {'conf ids':>9} {'longest':>9}")
    for conf in TRACK_SWEEP:
        for row in report["tracks"][f"{conf:.2f}"]:
            if row["segment"] not in CLEAN_NEGATIVES:
                continue
            entry = row["collapsed"]
            lifetimes = entry["lifetimes"]
            longest = max((v["frames"] for v in lifetimes.values()), default=0)
            det_frames = sum(1 for f in sorted(by_segment[row["segment"]])
                             if any(r[1] >= conf for r in dets[f]))
            print(f"  {row['segment']:<16} {conf:<6.2f} {row['n']:>5} "
                  f"{100.0 * det_frames / row['n']:>8.0f}% "
                  f"{100.0 * entry['frames_with_confirmed'] / row['n']:>9.0f}% "
                  f"{entry['n_confirmed_ids']:>9} "
                  f"{100.0 * longest / row['n']:>8.0f}%")
            for track_id, value in sorted(lifetimes.items(),
                                          key=lambda kv: -kv[1]["frames"])[:2]:
                share = 100.0 * value["frames"] / row["n"]
                print(f"       track {track_id:<4} {value['frames']:>4} frames "
                      f"({share:>3.0f}%)  bearing {value['bearing_deg']:>+6.1f} deg "
                      f"(spread {value['bearing_spread_deg']:>4.1f})  "
                      f"range {value['range_m']:.1f} m")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nwrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache", help="one forward pass per frame, cached")
    cache.add_argument("--prototxt", required=True)
    cache.add_argument("--weights", required=True)
    cache.add_argument("--frames", required=True)
    cache.add_argument("--out", required=True)
    cache.set_defaults(func=cmd_cache)

    verify = sub.add_parser("verify", help="prove the low prototxt only ADDS boxes")
    verify.add_argument("--low", required=True)
    verify.add_argument("--high", required=True)
    verify.add_argument("--at", type=float, default=0.25)
    verify.set_defaults(func=cmd_verify)

    score = sub.add_parser("score", help="frame- and track-level scoring")
    score.add_argument("--cache", required=True)
    score.add_argument("--labels", required=True)
    score.add_argument("--root", required=True, help="frame + manifest directory")
    score.add_argument("--visual-nav", required=True,
                       help="robot-stack/unitree/go2/visual_nav -- the REAL tracker")
    score.add_argument("--camera", required=True, help="go2_front_camera.json")
    score.add_argument("--iou", type=float, default=0.30)
    score.add_argument("--peer-height", type=float, default=PEER_HEIGHT_M)
    score.add_argument("--peer-width", type=float, default=PEER_WIDTH_M)
    score.add_argument("--max-range", type=float, default=6.0)
    score.add_argument("--out", default=None)
    score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
