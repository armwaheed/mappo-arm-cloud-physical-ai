#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record Lite3 camera detections without connecting to or commanding locomotion.

This is a camera/model evidence tool, not a navigation launcher. It intentionally imports no
Lite3 locomotion, UDP, ROS, or vendor-control module. Each output record contains only a
credential-free camera kind, frame metadata, pixel-space detections, and detector timing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_ROBOT_STACK = Path(__file__).resolve().parents[3]
_COMMON = _ROBOT_STACK / "unitree" / "go2" / "visual_nav"
sys.path.insert(0, str(_ROBOT_STACK))
sys.path.insert(0, str(_COMMON))

from deep_robotics.lite3.visual_nav.camera import (  # noqa: E402
    Lite3Camera,
    camera_source_kind,
    parse_camera_source,
)
from person_detector import DEFAULT_CONFIDENCE, VOC_CLASSES, PersonDetector  # noqa: E402

SCHEMA = "lite3-vision-shadow/v1"
DEFAULT_SECONDS = 60.0
MAX_SECONDS = 3600.0
DEFAULT_CLASSES = ("person", "chair")


def parse_voc_classes(value: str) -> tuple[str, ...]:
    """Parse a non-empty, duplicate-free list of MobileNet-SSD VOC labels."""
    classes = tuple(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))
    if not classes:
        raise argparse.ArgumentTypeError("--classes must contain at least one VOC label")
    unknown = sorted(set(classes).difference(VOC_CLASSES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown VOC classes: {', '.join(unknown)}")
    return classes


def _write_record(handle, record: dict) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def run(*, camera_source: int | str, camera_gstreamer: bool, model_dir: Path,
        classes: tuple[str, ...], confidence: float, input_size: int, seconds: float,
        max_frames: int | None, output: Path, camera_factory=Lite3Camera,
        detector_factory=PersonDetector, clock=time.monotonic) -> dict:
    """Write bounded, non-actuating camera detections and return their summary."""
    if not 0.0 < seconds <= MAX_SECONDS:
        raise ValueError(f"--seconds must be within 0..{MAX_SECONDS:.0f}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if not 0.0 < confidence <= 1.0:
        raise ValueError("--confidence must be within 0..1")
    if input_size <= 0:
        raise ValueError("--input-size must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    detector = detector_factory(
        model_dir,
        input_size=input_size,
        confidence=confidence,
        classes=classes,
    )
    camera = camera_factory(camera_source, gstreamer=camera_gstreamer)
    frames = 0
    detection_counts: Counter[str] = Counter()
    inference_total_s = 0.0

    camera.start()
    try:
        started = clock()
        with output.open("x", encoding="utf-8") as handle:
            _write_record(handle, {
                "kind": "header",
                "schema": SCHEMA,
                "started_monotonic_s": started,
                "camera_kind": camera_source_kind(camera_source, gstreamer=camera_gstreamer),
                "classes": list(classes),
                "confidence": confidence,
                "input_size": input_size,
            })
            last_sequence = 0
            while max_frames is None or frames < max_frames:
                remaining = seconds - (clock() - started)
                if remaining <= 0.0:
                    break
                frame = camera.wait_for_new(last_sequence, min(1.0, remaining))
                if frame is None:
                    continue
                last_sequence = frame.seq
                inference_started = clock()
                detections = detector.detect(frame.image)
                inference_s = clock() - inference_started
                inference_total_s += inference_s
                frames += 1
                detection_counts.update(detection.label for detection in detections)
                _write_record(handle, {
                    "kind": "frame",
                    "monotonic_s": clock(),
                    "sequence": frame.seq,
                    "frame_age_s": max(0.0, clock() - frame.capture_time),
                    "shape": list(frame.image.shape),
                    "inference_s": inference_s,
                    "detections": [
                        {
                            "label": detection.label,
                            "score": detection.score,
                            "box_px": [
                                detection.x1,
                                detection.y1,
                                detection.x2,
                                detection.y2,
                            ],
                        }
                        for detection in detections
                    ],
                })

            if frames == 0:
                raise RuntimeError("no camera frames were recorded")
            elapsed_s = clock() - started
            summary = {
                "frames": frames,
                "detections": dict(sorted(detection_counts.items())),
                "mean_inference_s": inference_total_s / frames,
                "camera_errors": camera.error_count,
                "elapsed_s": elapsed_s,
            }
            _write_record(handle, {"kind": "outcome", **summary})
    finally:
        camera.stop()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record non-actuating Lite3 RTSP/V4L2 MobileNet-SSD detections.",
    )
    parser.add_argument(
        "--camera-source",
        required=True,
        help="V4L2 index, RTSP URI, or GStreamer pipeline; it is not written to evidence",
    )
    parser.add_argument("--camera-gstreamer", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--classes",
        type=parse_voc_classes,
        default=DEFAULT_CLASSES,
        metavar="VOC_LABELS",
        help="comma-separated MobileNet-SSD VOC labels (default: person,chair)",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--input-size", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            camera_source=parse_camera_source(args.camera_source),
            camera_gstreamer=args.camera_gstreamer,
            model_dir=args.model_dir,
            classes=args.classes,
            confidence=args.confidence,
            input_size=args.input_size,
            seconds=args.seconds,
            max_frames=args.max_frames,
            output=args.output,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"[lite3_vision_shadow] FAILED: {error}", file=sys.stderr)
        return 2
    print(
        "[lite3_vision_shadow] "
        f"frames={summary['frames']} detections={summary['detections']} "
        f"mean_inference_s={summary['mean_inference_s']:.4f} "
        f"camera_errors={summary['camera_errors']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
