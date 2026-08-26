#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Is the peer placed somewhere a run can actually learn something? Check before walking.

    python3 peer_scene_check.py --stand      # stands the robot, checks, lies back down

⚠️ IT STANDS THE ROBOT, AND THAT IS THE ENTIRE POINT. On 2026-08-25 the peer was checked
while the robot was PRONE — 10 of 10 detections, box unclipped, aspect 1.03, range
0.74 m, every gate green — and then the run failed on all of them at once. The camera
sits at **0.154 m prone and 0.32 m standing**. Standing lifts it 0.166 m, which pushes
the peer's ground-contact point down the frame by ``focal * atan(0.166 / 0.74)`` = about
**284 px**, straight off the bottom edge. A check that does not stand measures a geometry
the run will never have.

WHAT GOES WRONG WHEN THE BOX CLIPS, and it is three failures from one cause:

* ``estimate_range`` switches from the height prior to the WIDTH prior, because a
  vertically clipped box is shorter than the object and its height would read FAR. On
  that run 176 of 177 sightings came back ``width-capped`` and the peer was reported at
  0.72 m when the tape said 0.74 — then at 0.2 m once the robot was beside it.
* ``RangedDetection.person_shaped`` returns True for a vertically clipped box, because a
  person with their head out of frame is indistinguishable from a quadruped. So the peer
  HOLDS the robot instead of reaching the policy: 195 of 195 obstacle records.
* The robot then never moves, its tracks coast unseen, and the radius inflates from the
  configured 0.20 m to **1.08 m median and 7.09 m worst** — which vetoes everything.

THE BOUNDARY IS COMPUTABLE, so this reports it rather than guessing. Standing, the
contact point stays in frame only while ``focal * atan(camera_height / range) < cy``:

    range > camera_height / tan(cy / focal) = 0.32 / tan(540 / 1290.16) = 0.72 m

0.74 m was two centimetres inside that. It is not a margin.

THE OTHER END. Detection falls off a cliff, not a slope: measured over 1,903 labelled
frames, this peer is found in **0 of 315** frames beyond 2.7 m — never — against 80% at
1.5-1.9 m and 91% inside 1.1 m. And below about **0.52 m** a 0.35 m peer subtends the
whole 85.3 degree field of view, so no open bearing exists at all and no policy can
steer around it (issue #72).

So the usable band is roughly **0.8 m to 2.0 m**, and this script says which side of it
you are on and by how much.
"""

from __future__ import annotations

import argparse
import collections
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Range sources that mean the height prior was NOT used. `estimate_range` falls back to
#: these when the box is clipped, and each one is a signal that the geometry is wrong
#: rather than a merely noisier number.
DEGRADED_SOURCES = ("width", "width-capped", "frame-fill")

#: Detection rate below which the run is not worth the arm re-pose.
MIN_DETECTION_RATE = 0.8

#: Usable range band. Lower bound is the standing clip boundary (0.72 m) with margin;
#: upper bound is where detection has fallen away.
RANGE_MIN_M, RANGE_MAX_M = 0.85, 2.00

#: Minimum bearing off the nose. A peer dead ahead cannot tell you which way the policy
#: chose to go, because either way clears it — see the left/right test of 2026-08-25.
MIN_BEARING_DEG = 5.0


def _gate(name: str, ok: bool, detail: str) -> bool:
    print("  %-22s %-4s %s" % (name, "PASS" if ok else "FAIL", detail))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stand", action="store_true",
                    help="DANGER: stands the robot. Without it this refuses to run, "
                         "because a prone check measures a geometry the run will not have")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--model-dir", default=str(Path.home() / "go2_models"))
    ap.add_argument("--calibration", default=str(Path.home() / "go2_front_camera.json"))
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument("--peer-height", type=float, default=0.514)
    ap.add_argument("--peer-width", type=float, default=0.31)
    args = ap.parse_args(argv)

    if not args.stand:
        print("[scene] REFUSING: pass --stand. A prone check measures the wrong geometry —\n"
              "        the camera rises 0.166 m when standing, which moved the peer's\n"
              "        contact point 284 px down the frame and off the bottom edge on\n"
              "        2026-08-25, after a prone check had passed every gate.")
        return 2

    import cv2
    import numpy as np

    from camera_model import FisheyeCamera
    from person_detector import (
        PERSON_ASPECT_MIN,
        Detection,
        PersonDetector,
        RangedDetection,
        SizePrior,
        estimate_range,
    )
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient

    from locomotion.go2_locomotion import Go2Locomotion
    from safety import lie_down, stand_up

    ChannelFactoryInitialize(0, args.iface)
    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()

    loco = Go2Locomotion(iface=args.iface)
    loco.connect()

    detector = PersonDetector(args.model_dir, input_size=args.input_size,
                              confidence=args.confidence)
    prior = SizePrior.of_height(args.peer_height, args.peer_width)

    rows = []
    try:
        print("[scene] standing — this is the geometry the run will have")
        stand_up(loco)
        camera = None
        for _ in range(args.frames):
            code, data = video.GetImageSample()
            image = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            if camera is None:
                camera = FisheyeCamera.load(args.calibration).scaled(width, height)
            best = None
            for detection in detector.detect(image):
                if best is None or detection.score > best.score:
                    best = detection
            if best is None:
                rows.append(None)
            else:
                range_m, source = estimate_range(best, camera, prior)
                ranged = RangedDetection(detection=best, range_m=range_m,
                                         bearing_rad=0.0, source=source)
                vertical, _ = best.clipped(width, height)
                bearing, _ = camera.bearing_elevation(*best.centre)
                rows.append({
                    "label": best.label, "score": best.score, "source": source,
                    "range_m": range_m, "vertical_clip": vertical,
                    "aspect": best.height_px / max(best.width_px, 1.0),
                    "bearing_deg": math.degrees(float(bearing)),
                    "person_shaped": ranged.person_shaped(width, height),
                })
            time.sleep(0.1)
    finally:
        lie_down(loco)

    seen = [r for r in rows if r]
    print()
    print("PEER SCENE CHECK — %d frames, robot STANDING" % len(rows))
    ok = True
    ok &= _gate("detected", len(seen) >= MIN_DETECTION_RATE * len(rows),
                "%d/%d  labels %s" % (len(seen), len(rows),
                                      dict(collections.Counter(r["label"] for r in seen))))
    if not seen:
        print("\n  Nothing to measure. Beyond ~2.7 m this peer is found in 0 of 315 "
              "frames — move it closer.")
        return 1

    sources = collections.Counter(r["source"] for r in seen)
    degraded = sum(sources[s] for s in DEGRADED_SOURCES)
    ok &= _gate("ranged by height", degraded == 0,
                "%s%s" % (dict(sources),
                          "   <- clipped: range and person_shaped are both wrong"
                          if degraded else ""))
    clipped = sum(1 for r in seen if r["vertical_clip"])
    ok &= _gate("not vertically clipped", clipped == 0,
                "%d/%d clipped   (standing clip boundary is 0.72 m)"
                % (clipped, len(seen)))
    held = sum(1 for r in seen if r["person_shaped"])
    ok &= _gate("reaches the policy", held == 0,
                "%d/%d would HOLD the robot instead" % (held, len(seen)))
    rng = statistics.median(r["range_m"] for r in seen)
    ok &= _gate("range in band", RANGE_MIN_M <= rng <= RANGE_MAX_M,
                "%.2f m   (band %.2f-%.2f)" % (rng, RANGE_MIN_M, RANGE_MAX_M))
    bearing = statistics.median(r["bearing_deg"] for r in seen)
    ok &= _gate("offset from the nose", abs(bearing) >= MIN_BEARING_DEG,
                "%+.0f deg (%s)   dead ahead cannot tell you which way it chose"
                % (bearing, "LEFT" if bearing > 0 else "RIGHT"))
    aspect = statistics.median(r["aspect"] for r in seen)
    ok &= _gate("aspect below threshold", aspect < PERSON_ASPECT_MIN,
                "%.2f  (holds at >= %.1f)" % (aspect, PERSON_ASPECT_MIN))

    half = math.degrees(math.asin(min(1.0, 0.35 / max(rng, 0.36))))
    window = camera.hfov_deg / 2.0 - half
    ok &= _gate("open window exists", window > 0,
                "peer subtends %.0f deg, leaving %.0f deg each side (issue #72)"
                % (2 * half, window))

    print()
    print("VERDICT: %s" % ("READY — run it" if ok else "NOT READY — see the FAILs above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
