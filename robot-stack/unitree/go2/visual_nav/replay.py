#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the detector and tracker over a recorded video — no robot, no volunteer.

    python3 replay.py walkers.avi --out annotated.mp4

The moving-obstacle half of this pipeline is the hard half to test: it needs people
who actually move, which on a robot means a person, a clear floor, and a willingness
to be walked at. This replays the SAME :class:`PersonDetector` and
:class:`ObstacleTracker` the navigator uses over any video file, so detection
thresholds, the association gate and the confirm/coast lifecycle can be tuned and
regression-checked from a desk.

It is also how a run gets reviewed after the fact: point it at a recording made by
``visual_nav.py --record`` and watch what the tracker made of it.

WHAT THIS DOES AND DOES NOT PROVE. Detection, association, track lifetime and the
SHAPE of the velocity estimates are all genuine. The metric scale is only as good as
the camera model you give it: for third-party footage the field of view and mounting
height are guesses, so the metres are indicative and the m/s with them. Pass
``--calibration`` with a real model — or replay the Go2's own footage — for numbers
worth quoting.

The robot is assumed STATIONARY at the odom origin, which is true of fixed-camera
footage and of a ``--record`` clip only while the robot was held. Ego-motion
compensation is exercised by ``test_tracker.py``, not here.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))       # sibling modules

import overlay
from avoidance import Obstacle, PlannerConfig
from camera_model import FisheyeCamera
from colour_detector import PROFILES, ColourBlobDetector
from person_detector import (
    DEFAULT_CONFIDENCE,
    DYNAMIC_CLASSES,
    PersonDetector,
    range_detections,
)
from static_map import StaticObstacleMap
from tracker import ObstacleTracker, observation_from

STATIONARY_ROBOT = (0.0, 0.0, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay the detectors + tracker + static map over a video file.")
    ap.add_argument("video", help="input video")
    ap.add_argument("--out", default=None, help="write an annotated video here")
    ap.add_argument("--model-dir", default=str(Path.home() / "go2_models"),
                    help="directory holding the MobileNet-SSD files")
    ap.add_argument("--calibration", default=None,
                    help="camera model JSON; without it the nominal FOV is assumed "
                         "and the metric scale is indicative only")
    ap.add_argument("--hfov", type=float, default=None,
                    help="horizontal field of view of the FOOTAGE, if known")
    ap.add_argument("--input-size", type=int, default=300)
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument("--classes", nargs="+", default=list(DYNAMIC_CLASSES),
                    metavar="VOC_CLASS",
                    help="which VOC classes to detect. Matches visual_nav.py's flag of "
                         "the same name, so a replay can reproduce a run's settings "
                         "rather than silently tracking only people")
    ap.add_argument("--static-prop", default=None, choices=sorted(PROFILES),
                    help="also segment a known-coloured static prop and map it. This "
                         "is the offline harness for tuning the colour gates: iterate "
                         "against a recorded clip with no robot involved")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (emulates a slower detector)")
    args = ap.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    # The tracker is driven by the VIDEO's clock, not the wall clock, so a replay
    # produces the same velocities however long inference takes.
    dt = args.stride / source_fps

    if args.calibration:
        camera_model = FisheyeCamera.load(args.calibration)
        if (camera_model.width, camera_model.height) != (width, height):
            camera_model = camera_model.scaled(width, height)
        scale_note = f"calibrated ({args.calibration})"
    else:
        camera_model = FisheyeCamera.from_hfov(width, height, args.hfov or 120.0)
        scale_note = "ASSUMED FOV — metres and m/s are indicative only"

    detector = PersonDetector(args.model_dir, input_size=args.input_size,
                              confidence=args.confidence,
                              classes=tuple(args.classes))
    tracker = ObstacleTracker(fov_rad=math.radians(camera_model.hfov_deg))

    colour_detector = static_map = None
    if args.static_prop:
        profile = PROFILES[args.static_prop]
        colour_detector = ColourBlobDetector(profile)
        static_map = StaticObstacleMap(radii={profile.label: profile.radius_m},
                                       fov_rad=math.radians(camera_model.hfov_deg))

    writer = None
    if args.out:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 source_fps / args.stride, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"[replay] cannot open {args.out} for writing (mp4v)")

    print(f"[replay] {args.video}  {width}x{height} @ {source_fps:.1f}fps")
    print(f"[replay] camera model: HFOV {camera_model.hfov_deg:.1f}deg — {scale_note}")

    frame_index = processed = total_detections = 0
    peak_tracks = 0
    track_ids: set = set()
    landmarks_seen: set = set()
    detect_ms_total = 0.0
    now = 0.0
    mover_radius_m = PlannerConfig().obstacle_radius_m

    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame_index += 1
        if (frame_index - 1) % args.stride:
            continue
        if args.max_frames and processed >= args.max_frames:
            break

        started = time.monotonic()
        detections = detector.detect(image)
        detect_ms_total += (time.monotonic() - started) * 1000.0

        ranged = range_detections(detections, camera_model)
        observations = [
            observation_from(item.bearing_rad, item.range_m, item.source, item.label,
                             STATIONARY_ROBOT)
            for item in ranged
        ]

        static_ranged = []
        if colour_detector is not None:
            static_ranged = range_detections(colour_detector.detect(image),
                                             camera_model,
                                             colour_detector.profile.prior)

        now += dt
        tracker.predict(dt)
        occluders = ()
        if static_map is not None:
            static_map.observe(
                [observation_from(item.bearing_rad, item.range_m, item.source,
                                  item.label, STATIONARY_ROBOT)
                 for item in static_ranged],
                now, *STATIONARY_ROBOT)
            occluders = tuple(static_map.occluders(*STATIONARY_ROBOT))
            landmarks_seen.update(lm.landmark_id for lm in static_map.confirmed())
        tracker.update(observations, now, *STATIONARY_ROBOT, occluders=occluders)

        confirmed = tracker.confirmed_tracks()
        track_ids.update(t.track_id for t in confirmed)
        peak_tracks = max(peak_tracks, len(confirmed))
        total_detections += len(detections)
        processed += 1

        if writer is not None:
            # Obstacles are built here rather than shared with visual_nav's version
            # because the two genuinely differ: the navigator extrapolates tracks over
            # the perception latency, and a replay has none — the tracker is driven off
            # the VIDEO's clock, so its estimate is already current for the frame being
            # drawn. Reusing that code would mean passing a latency of zero and
            # pretending the concepts matched.
            obstacles = [
                Obstacle(x=float(t.state[0]), y=float(t.state[1]),
                         vx=float(t.state[2]), vy=float(t.state[3]),
                         radius_m=mover_radius_m + t.position_sigma, label=t.label)
                for t in confirmed
            ]
            if static_map is not None:
                obstacles.extend(
                    Obstacle(x=lm.x, y=lm.y, vx=0.0, vy=0.0,
                             radius_m=lm.planning_radius_m, label=lm.label)
                    for lm in static_map.confirmed())
            canvas = image.copy()
            overlay.draw_detections(canvas, ranged + static_ranged)
            # A quarter of the frame width — on small footage a fixed 300 px panel
            # covers the very people being tracked.
            overlay.draw_plan_view(canvas, STATIONARY_ROBOT, obstacles, None, None,
                                   size_px=min(300, width // 4))
            speeds = " ".join(f"#{t.track_id}:{t.speed:.1f}" for t in confirmed[:6])
            overlay.draw_status(canvas, [
                f"frame {frame_index}  det {len(detections)}  tracks {len(confirmed)}",
                f"speeds m/s {speeds}" if speeds else "speeds -",
            ])
            writer.write(canvas)

    capture.release()
    if writer is not None:
        writer.release()

    if processed == 0:
        raise SystemExit("[replay] no frames processed")

    moving = [t for t in tracker.tracks if t.speed > 0.2]
    # Guard the rate: a fast enough machine (or a stubbed detector) can total zero
    # milliseconds and turn the summary line into a ZeroDivisionError at the very end,
    # after all the work is done.
    rate = f"{processed / (detect_ms_total / 1000.0):.1f} fps" if detect_ms_total else "-"
    print(f"[replay] {processed} frames, {detect_ms_total / processed:.0f} ms/frame "
          f"inference ({rate})")
    print(f"[replay] {total_detections} detections "
          f"({total_detections / processed:.1f}/frame)")
    print(f"[replay] {len(track_ids)} distinct confirmed tracks, "
          f"peak {peak_tracks} at once, {len(moving)} still moving at the end")
    if static_map is not None:
        # Distinct landmark IDs over the whole clip against how many survive: a colour
        # profile that keeps re-spawning the same prop shows up here as a large first
        # number and a small second one, which is the failure the shape gates exist to
        # prevent and is invisible in any per-frame count.
        print(f"[replay] {len(landmarks_seen)} distinct confirmed landmarks over the "
              f"clip, {len(static_map.confirmed())} held at the end")
        for landmark in static_map.confirmed():
            print(f"[replay]   #{landmark.landmark_id} {landmark.label} at "
                  f"({landmark.x:+.2f}, {landmark.y:+.2f}) m, "
                  f"{math.hypot(landmark.x, landmark.y):.2f} m out, "
                  f"sigma {landmark.position_sigma:.3f} m, "
                  f"{landmark.sightings} sightings")
    if args.out:
        print(f"[replay] wrote {args.out}")


if __name__ == "__main__":
    main()
