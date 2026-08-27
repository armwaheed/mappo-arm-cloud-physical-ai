#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recompute every number in this directory's README, from the committed JSON.

THE CLAIM UNDER TEST. Four claims, in the order they decide what to do next.

  1. The clip is ONE viewpoint. Not "324 frames that are near-duplicates" -- one camera
     pose. Augmentation cannot manufacture a second one.
  2. The frames the burned-in overlay leaves alone are ONE contiguous block in which the
     peer does not move, so the usable half is one image, not 168 samples.
  3. ``detector/eval_detector.py``'s ``--mask-overlay`` regions were fitted at 1920x1080
     and are wrong on this 1280x720 footage -- they leave most of the radar panel behind.
  4. At the input size production launches, ``--input-size 224``, neither the shipped
     network nor anything trained here fires on the Lite3 at all.

READING THE OUTPUT. Every line prints a numerator, a denominator and the claim it
supports. A line that disagrees with the README is a bug in one of them.

WHAT IT NEEDS. numpy, and the four JSON files beside it. No video, no model, no network,
no GPU. The video that produced ``clip_measurements.json`` is 20 MB and is not in git;
``measure_clip.py`` is the script that read it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
#: Fractions of the frame, copied from detector/eval_detector.py. They are applied to
#: whatever frame they are handed, which is the bug this directory reports.
RADAR_REGION = (0.834, 0.0, 1.0, 0.287)
PLATE_REGION = (0.0, 0.930, 0.215, 1.0)


def rule(text: str) -> None:
    print(f"\n{text}\n" + "=" * len(text))


def main() -> None:
    m = json.loads((HERE / "clip_measurements.json").read_text())
    W, H, n = m["width"], m["height"], m["opencv_frames"]

    rule("1. What the clip is")
    ff = m["ffprobe"]
    print(f"  md5                       {m['md5']}")
    print(f"  container nb_frames       {ff['container_nb_frames']}")
    print(f"  decoded frame count       {ff['decoded_frames']}")
    print(f"  packet count              {ff['packets']}")
    print(f"  OpenCV read               {n}")
    print(f"  -> all four agree: {n} frames. A count of 421 does not describe this file.")
    print(f"  container r_frame_rate    {ff['r_frame_rate']}   (the container disagrees")
    print(f"  container avg_frame_rate  {ff['avg_frame_rate']}    with itself)")
    print(f"  measured rate             {m['fps_measured']:.2f} fps over "
          f"{float(ff['duration_s']):.0f} s  <- the real one")
    print(f"  frame size                {W}x{H}")

    rule("2. How many independent viewpoints (camera poses) the clip holds")
    d = np.array([x for x in m["camera_displacement_px"] if x is not None])
    print(f"  homography recovered on   {len(d)}/{n} frames")
    print("  camera displacement vs frame 1, median of the four frame corners, in pixels:")
    print(f"    min {d.min():.2f}   median {np.median(d):.2f}   "
          f"p90 {np.percentile(d, 90):.2f}   max {d.max():.2f}")
    for t in (10, 25, 50):
        print(f"    frames displaced by more than {t:>2} px: {(d > t).sum()}")
    print("  -> the camera never moves. The clip is ONE viewpoint, and the robot's own")
    print("     status line says DRY RUN STANDING(sim) throughout: it never walked.")

    rule("3. The burned-in overlay")
    out = set(m["outline_frames"])
    print(f"  frames carrying a detection outline on the scene   {len(out)}/{n} "
          f"= {len(out)/n:.1%}")
    print(f"  frames free of one                                 {n-len(out)}/{n} "
          f"= {(n-len(out))/n:.1%}")
    clean = m["clean_block"]
    runs, cur = [], [clean[0]]
    for a in clean[1:]:
        if a == cur[-1] + 1:
            cur.append(a)
        else:
            runs.append((cur[0], cur[-1]))
            cur = [a]
    runs.append((cur[0], cur[-1]))
    print(f"  contiguous runs of outline-free frames             {len(runs)}")
    for a, b in runs:
        # 1-based, to match the f%04d.jpg names ffmpeg writes; the JSON stores 0-based.
        print(f"    frames {a+1}..{b+1}  ({b-a+1} frames, "
              f"{(b-a+1)/m['fps_measured']:.1f} s)")
    print("  -> the usable frames are one unbroken block, not a sample of the clip.")

    rule("4. Does the peer move inside that block?")
    dev = m["region_deviation"]
    for k, v in sorted(dev.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<22} {v:5.2f} grey levels from the block median")
    print(f"  -> the peer ({dev['peer']:.2f}) sits between inert carpet "
          f"({dev['carpet_inert']:.2f}) and inert wall ({dev['glass_wall_inert']:.2f}).")
    print(f"     The synthetic radar panel ({dev['radar_synthetic']:.2f}) is the control: "
          f"pixels that")
    print("     truly cannot change read ~0. The peer did not move; that number is noise.")

    rule("5. detector/eval_detector.py --mask-overlay on 1280x720 footage")
    rx0, ry0, rx1, ry1 = (int(RADAR_REGION[0]*W), int(RADAR_REGION[1]*H),
                          int(RADAR_REGION[2]*W), int(RADAR_REGION[3]*H))
    tx, ty, tw, th = m["radar_from_overlay_source"]
    print(f"  the panel overlay.py actually draws   x {tx}..{tx+tw-1}, y {ty}..{ty+th-1}")
    print(f"  what RADAR_REGION masks here          x {rx0}..{rx1-1}, y {ry0}..{ry1-1}")
    iw = max(0, min(rx1, tx+tw) - max(rx0, tx))
    ih = max(0, min(ry1, ty+th) - max(ry0, ty))
    print(f"  panel area {tw*th} px; masked {iw*ih} px = {iw*ih/(tw*th):.1%}")
    print(f"  LEFT UNMASKED                         {tw*th - iw*ih} px = "
          f"{1 - iw*ih/(tw*th):.1%} of the panel")
    px0, py0, px1, py1 = (int(PLATE_REGION[0]*W), int(PLATE_REGION[1]*H),
                          int(PLATE_REGION[2]*W), int(PLATE_REGION[3]*H))
    ax, ay, aw, ah = m["plate_measured"]
    print(f"  status plate, measured                x {ax}..{ax+aw-1}, y {ay}..{ay+ah-1}")
    print(f"  what PLATE_REGION masks here          x {px0}..{px1-1}, y {py0}..{py1-1}")
    print("  -> both regions are fractions fitted at 1920x1080. overlay.py places the")
    print("     panel at (width - 300 - 16, 16) in PIXELS, so the fraction moves with")
    print("     the frame size and the mask lands in the wrong place.")

    rule("6. The incumbent on the Lite3, at both input sizes")
    inc = json.loads((HERE / "incumbent_lite3.json").read_text())
    VOC = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    for size in ("224", "300"):
        frames = inc[size]
        tag = "  <- PRODUCTION (deploy/run-peer-supervised.sh)" if size == "224" else ""
        print(f"  input size {size}{tag}")
        for t in (0.25, 0.40, 0.50):
            lands = sum(any(x["score"] >= t and x["iou"] >= 0.30 for x in f) for f in frames)
            fires = sum(any(x["score"] >= t for x in f) for f in frames)
            names = {}
            for f in frames:
                on = [x for x in f if x["score"] >= t and x["iou"] >= 0.30]
                if on:
                    b = VOC[max(on, key=lambda x: x["score"])["cls"]]
                    names[b] = names.get(b, 0) + 1
            top = ", ".join(f"{k} {v}" for k, v in
                            sorted(names.items(), key=lambda kv: -kv[1])[:3]) or "-"
            print(f"    @{t:.2f}  lands on peer {lands:>3}/{len(frames)} "
                  f"= {lands/len(frames):>4.0%}   fires anywhere {fires:>3}/{len(frames)}"
                  f"   as: {top}")
        best = [max((x["iou"] for x in f), default=0.0) for f in frames]
        print(f"    best IoU with the peer box, any class, any score: "
              f"median {np.median(best):.3f}  max {max(best):.3f}")
    print("  -> at 224 the shipped network never puts a box on the robot.")

    rule("7. The two fine-tunes, scored at both sizes")
    for tag, path in (("augmented (wave-6 recipe)", "scored_a_lite3_aug.json"),
                      ("paired control, no aug   ", "scored_b_lite3_ctl.json")):
        s = json.loads((HERE / path).read_text())
        nf = s["lite3_frames"]
        print(f"  {tag}   (Lite3 column is SAME-SESSION: train == eval)")
        for r in s["rows"]:
            print(f"    {r['model']:<20} lite3@224 {r['lite3@224']:>3}/{nf}   "
                  f"lite3@300 {r['lite3@300']:>3}/{nf}   "
                  f"person kept @224 {r['person_kept@224']}/{r['person_base@224']}   "
                  f"@300 {r['person_kept@300']}/{r['person_base@300']}")
    print("  -> every checkpoint of both runs scores 0 at 224, including the control that")
    print("     scores 168/168 at 300 on the frames it was trained on. The trainer resizes")
    print("     to 300 (finetune_ssd.py) and the deployment launches 224.")
    print("  -> person retention fails the Go2 gate ('lose zero people') in both runs.")


if __name__ == "__main__":
    main()
