#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remap this Lite3's front camera into the projection the shared stack assumes.

WHY THIS EXISTS, and it is not a tuning problem. ``camera_model.FisheyeCamera`` is an
EQUIDISTANT model, ``radius_px = focal_px * theta``, carrying one intrinsic and no
distortion terms. Measured on 2026-09-01 against 26 ArUco grid-board views, this lens is
not that shape at all -- it is very nearly RECTILINEAR:

    cv2.fisheye.calibrate  (equidistant)      RMS 52.9 px
    cv2.calibrateCamera    (pinhole + dist)   RMS  2.9 px

Because ``tan(theta) ~ theta`` near the axis, the two agree on the optical axis and part
company with angle. What an equidistant model reports for a true off-axis bearing:

    true  10 deg -> 10.1    20 -> 20.9    30 -> 33.1    40 -> 48.1    50 -> 68.3

That is the whole of the 2026-08-31 failure. In ``live-goal-avoid-20260831T110648Z`` the
robot swept 63 deg of yaw while translating 0.08 m, and its range to a STATIONARY box
collapsed from 1.27 m to 0.33 m -- a 281% swing on an obstacle that had not moved. The
planner veto-held, correctly, on a number that was wrong; the stall detector then blamed
the tether. Three focal lengths were fitted that day (469.63, 660, 392.4) and each was
right at the bearing it was fitted at, because a single scalar cannot repair a model
whose EQUATION is wrong.

WHAT THIS DOES. Rather than fight the vendored model, it hands the model an image the
model is true for: every frame is remapped from the real lens into a synthetic
equidistant projection of known focal length. ``--calibration`` then states that focal,
and the stack's geometry is correct BY CONSTRUCTION across the whole frame.

Measured end to end, comparing the stack's pose against an accurate pinhole pose over the
same 26 views spanning -51 to +28 degrees of bearing:

    range error   mean +0.0%   sd 0.2%   worst 0.9%     (was a 281% swing)
    bearing error mean +0.01d  sd 0.05d  worst 0.18d    (was +8.1d at 40 deg)

WHAT IT DOES NOT DO. It converts the measured pinhole model faithfully; it cannot be
better than that model. That fit carried RMS 2.94 px with fx/fy 1.028 and no coverage of
the frame's top third, so ABSOLUTE scale inherits those limits even though STABILITY
across bearing does not. Re-measure with fuller coverage before quoting a range as
better than a few percent.

``robot-stack/unitree/`` is the vendored tree and PROVENANCE.md forbids editing it in
place; ``deep_robotics/`` is this repository's own code, so the correction lives here and
no upstream sync can revert it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


def equidistant_focal_for(width: int, height: int, camera_matrix, distortion) -> float:
    """The largest equidistant focal that loses NO field of view.

    A sensor is 16:9 and an equidistant mapping is radially symmetric, so the two axes
    disagree about how much focal they can afford. Taking the smaller keeps every ray the
    lens delivers and pays for it with unused corners; taking the larger would fill the
    frame by CROPPING the real field, and the cropped part is the periphery -- which is
    where an obstacle appears before the robot is committed to a path.
    """
    probes = np.array([[[0.0, height / 2.0]], [[width - 1.0, height / 2.0]],
                       [[width / 2.0, 0.0]], [[width / 2.0, height - 1.0]]])
    undistorted = cv2.undistortPoints(probes, camera_matrix, distortion).reshape(-1, 2)
    angles = [math.atan(math.hypot(x, y)) for x, y in undistorted]
    return float(min((width / 2.0) / max(angles[0], angles[1]),
                     (height / 2.0) / max(angles[2], angles[3])))


def build_maps(camera_matrix, distortion, width: int, height: int,
               focal_out: float) -> tuple:
    """``cv2.remap`` maps for real-lens -> synthetic-equidistant.

    Built once and reused: the maps are 1280x720 float32 pairs and regenerating them per
    frame would cost more than the remap itself.
    """
    grid_u, grid_v = np.meshgrid(np.arange(width, dtype=np.float64),
                                 np.arange(height, dtype=np.float64))
    offset_u, offset_v = grid_u - width / 2.0, grid_v - height / 2.0
    radius = np.hypot(offset_u, offset_v)
    theta = radius / focal_out                       # equidistant, by construction
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_phi = np.where(radius > 0, offset_u / radius, 0.0)
        sin_phi = np.where(radius > 0, offset_v / radius, 0.0)
    # Clipped below 90 deg because tan() diverges there: a ray at the pole has no
    # rectilinear image, and without the clip those pixels become inf and then garbage
    # rather than the black they should be.
    tangent = np.tan(np.clip(theta, 0.0, math.radians(88.0)))
    x, y = tangent * cos_phi, tangent * sin_phi
    k1, k2, p1, p2, k3 = (list(np.asarray(distortion).ravel()) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return ((camera_matrix[0, 0] * x_d + camera_matrix[0, 2]).astype(np.float32),
            (camera_matrix[1, 1] * y_d + camera_matrix[1, 2]).astype(np.float32))


class Rectifier:
    """Holds the maps and applies them. Also the record of what it was built from."""

    def __init__(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        if data.get("schema") != "lite3-rectify/v1":
            raise SystemExit(
                f"[lite3] REFUSING TO USE CAMERA RECTIFICATION: {path} is not a "
                f"lite3-rectify/v1 file (schema={data.get('schema')!r})")
        self.width, self.height = int(data["width"]), int(data["height"])
        self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        self.distortion = np.array(data["distortion"], dtype=np.float64)
        self.focal_out = float(data["focal_out_px"])
        self.source = str(path)
        self.provenance = data.get("provenance", {})
        self._map_x, self._map_y = build_maps(self.camera_matrix, self.distortion,
                                              self.width, self.height, self.focal_out)

    def describe(self) -> str:
        return (f"rectify: real lens fx={self.camera_matrix[0, 0]:.1f} "
                f"fy={self.camera_matrix[1, 1]:.1f} -> equidistant focal "
                f"{self.focal_out:.1f} px "
                f"(HFOV {math.degrees(2.0 * (self.width / 2.0) / self.focal_out):.1f} deg)")

    def apply(self, image: np.ndarray) -> np.ndarray:
        # A frame that is not the size the maps were built for is a configuration error,
        # not something to silently resize: the maps encode absolute pixel coordinates.
        if image.shape[1] != self.width or image.shape[0] != self.height:
            raise SystemExit(
                f"[lite3] REFUSING TO RECTIFY: camera gives "
                f"{image.shape[1]}x{image.shape[0]} but {self.source} was measured at "
                f"{self.width}x{self.height}")
        return cv2.remap(image, self._map_x, self._map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def rectified_camera(camera, rectifier: Rectifier):
    """Wrap a camera so every frame it hands out is already rectified.

    A wrapper rather than a change to ``Lite3Camera`` for the same reason
    ``mappo_drive.peer_navigator`` is a subclass: the correction is composable, it is
    visible at the call site, and a run without ``--camera-rectify`` reads exactly as
    every run recorded before this existed.
    """

    original_latest = camera.latest

    def latest():
        frame = original_latest()
        if frame is None:
            return None
        # Rectify in place on a copy: the vendored loop hands the SAME frame object to
        # the planner, the recorder and the telemetry writer, so rectifying once here is
        # what keeps all four looking at the same pixels.
        object.__setattr__(frame, "image", rectifier.apply(frame.image))
        return frame

    camera.latest = latest
    return camera
