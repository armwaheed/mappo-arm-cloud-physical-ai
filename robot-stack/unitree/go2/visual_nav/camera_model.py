# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fisheye camera model for the Go2's front RGB camera — pixels to metric bearings.

The Go2's forward camera is a wide fisheye; a pinhole model is wrong across most of
the frame. This module models it as **equidistant** (``r = f·theta``, the standard
fisheye projection), which needs exactly one intrinsic — the focal length in pixels —
and gives two things the navigator needs:

  * :meth:`FisheyeCamera.bearing_elevation` — where a pixel is, as an angle off the
    nose. This is what steers the robot.
  * :meth:`FisheyeCamera.range_from_span` — how far away something is, from the angle
    its known physical size subtends. This is the ONLY range source on this robot:
    the unit has no depth camera (no RealSense, no ``/dev/video``), so a monocular
    size prior is all there is.

WHY ANGULAR SIZE AND NOT THE GROUND PLANE. The obvious monocular ranger is to
intersect the ray through a person's feet with the floor. It is unusable here: the
camera sits ~0.32 m off the ground, so a person at 3 m is only 6° below the horizon
and the range goes as ``h/tan(elevation)`` — a 2° trunk-pitch wobble (a trotting Go2
does more) swings the estimate from 2.3 m to 4.4 m. Angular size instead gives
``d = L / (2·tan(dtheta/2))``, whose relative error just equals the relative error in
the measured pixel span — a 4% range error for a 20 px error on a 520 px box. It
degrades gracefully and does not care about pitch at all.

That last property is why the camera's pitch barely matters here, and why this module
does NOT bother folding live IMU pitch into the model. Tilting the camera does shift a
pixel's azimuth, but only through the cosine of the tilt: a 10° pitch moves a bearing
by under 0.5° anywhere in this frame, and a trotting Go2's few degrees of trunk wobble
by well under 0.2°. That is inside the bearing noise the tracker already assumes, so
pitch is carried as a static mounting parameter rather than compensated per-frame.

CALIBRATION. ``focal_px`` is the one number everything scales on, and this robot ships
no intrinsics for the front camera (its VIO calibration is for the RealSense that is
not fitted). :data:`DEFAULT_HFOV_DEG` is a NOMINAL value from the published sensor
spec — good enough to walk, not good enough to quote. Measure it with
``calibrate_camera.py`` and the model is exact; :meth:`load` reads the result back.

Frames: body ``+x`` forward / ``+y`` left / ``+z`` up (the repo-wide convention).
Image ``u`` right / ``v`` down, origin top-left. Pure numpy — no robot needed, so the
whole module is unit-testable off-robot (``python3 test_camera_model.py``).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# ── Nominal intrinsics ──────────────────────────────────────────────────────
# Horizontal field of view of the Go2's front fisheye, from the published sensor
# spec. NOMINAL — the robot ships no calibration file for this camera, so this is a
# starting point, not a measurement. `calibrate_camera.py` replaces it.
DEFAULT_HFOV_DEG = 120.0

# ── Size priors for monocular ranging ───────────────────────────────────────
# Standing adult, floor to crown. The detector's box tracks the visible extent, so
# this is the right prior for an untruncated full-body box.
PERSON_HEIGHT_M = 1.70
# Shoulder-to-shoulder, used only when the box is cut off by the frame edge and the
# height is therefore a lower bound. Much weaker (width varies with body yaw and arm
# swing), so it is a fallback, not a peer.
PERSON_WIDTH_M = 0.50


@dataclass(frozen=True)
class FisheyeCamera:
    """Equidistant fisheye model: ``radius_px = focal_px · theta``.

    Args:
        width/height: image size in pixels the model is expressed in. Detections
            given to this model must be in the SAME pixel coordinates — scale boxes
            back to full resolution before calling, or build a model for the
            downscaled size with :meth:`scaled`.
        focal_px: the single intrinsic. Pixels per radian off the optical axis.
        cx/cy: principal point. Defaults to the image centre.
        pitch_rad: camera tilt below the body's forward axis, positive = nose-down.
        height_m: optical centre above the floor when the robot stands. It is NOT the
            ranger — see the module docstring for why the ground-plane intersection is
            unusable here — but it is not overlay-only either: :meth:`ground_point`
            uses it for the debug overlay, and ``person_detector.object_fit_range``
            uses it to work out the distance below which a person no longer fits in
            frame, which caps a width-derived range. So it does feed a range bound, and
            a wrong value moves that bound.
    """

    width: int
    height: int
    focal_px: float
    cx: float
    cy: float
    pitch_rad: float = 0.0
    height_m: float = 0.32

    # ── Constructors ────────────────────────────────────────────────────────
    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float = DEFAULT_HFOV_DEG,
                  **kwargs) -> FisheyeCamera:
        """Build from a horizontal field of view.

        Under the equidistant model the image half-width spans half the HFOV, so
        ``focal_px = (width/2) / (hfov/2)``.
        """
        if not 0.0 < hfov_deg < 360.0:
            raise ValueError(f"hfov_deg must be in (0, 360), got {hfov_deg}")
        focal = (width / 2.0) / math.radians(hfov_deg / 2.0)
        kwargs.setdefault("cx", width / 2.0)
        kwargs.setdefault("cy", height / 2.0)
        return cls(width=width, height=height, focal_px=focal, **kwargs)

    @classmethod
    def load(cls, path: str | Path) -> FisheyeCamera:
        """Read a model written by :meth:`save` (i.e. by ``calibrate_camera.py``)."""
        data = json.loads(Path(path).read_text())
        fields = {f: data[f] for f in ("width", "height", "focal_px", "cx", "cy")}
        for optional in ("pitch_rad", "height_m"):
            if optional in data:
                fields[optional] = data[optional]
        return cls(**fields)

    def save(self, path: str | Path, **provenance) -> None:
        """Write the model as JSON, plus any provenance keys (how it was measured).

        Provenance is not read back by :meth:`load` — it is there so a calibration
        file can never be mistaken for a nominal one by whoever reads it next.
        """
        payload = asdict(self)
        payload.update(provenance)
        payload["hfov_deg"] = self.hfov_deg
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def scaled(self, width: int, height: int) -> FisheyeCamera:
        """The same optics expressed in a resized image's pixel coordinates.

        Only valid for a uniform rescale of the same field of view (``cv2.resize`` of
        the whole frame), not for a crop.
        """
        sx, sy = width / self.width, height / self.height
        if abs(sx - sy) > 1e-6:
            raise ValueError(
                f"non-uniform rescale {sx:.4f}x{sy:.4f} would need two focal lengths; "
                f"resize {self.width}x{self.height} by a single factor")
        return FisheyeCamera(width=width, height=height, focal_px=self.focal_px * sx,
                             cx=self.cx * sx, cy=self.cy * sy,
                             pitch_rad=self.pitch_rad, height_m=self.height_m)

    # ── Properties ──────────────────────────────────────────────────────────
    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view implied by ``focal_px``."""
        return math.degrees(2.0 * (self.width / 2.0) / self.focal_px)

    @property
    def pixel_angle_rad(self) -> float:
        """Angle one pixel subtends — the smallest span this model can resolve.

        Exact and uniform under the equidistant model: ``r = f·theta`` gives
        ``dtheta = dr/f``. It is the honest floor for :meth:`range_from_span`, which
        cannot mean anything for a span the sensor could not have separated.
        """
        return 1.0 / self.focal_px

    # ── Projection ──────────────────────────────────────────────────────────
    def unit_vector(self, u, v) -> np.ndarray:
        """Body-frame unit direction(s) through pixel(s) ``(u, v)``.

        Accepts scalars or equal-shaped arrays; returns shape ``(..., 3)`` as
        ``(x forward, y left, z up)``.
        """
        du = np.asarray(u, dtype=float) - self.cx
        dv = np.asarray(v, dtype=float) - self.cy
        radius = np.hypot(du, dv)
        theta = np.clip(radius / self.focal_px, 0.0, math.pi)

        # Direction in camera-optical axes (X right, Y down, Z along the axis). At
        # radius 0 the direction is the optical axis itself and the in-image angle is
        # undefined, so guard the division rather than letting it produce a NaN.
        safe = np.where(radius > 1e-9, radius, 1.0)
        sin_t = np.sin(theta)
        cam = np.stack([sin_t * du / safe, sin_t * dv / safe, np.cos(theta)], axis=-1)
        cam = np.where(radius[..., None] > 1e-9, cam, np.array([0.0, 0.0, 1.0]))

        # Optical axes -> body axes: forward = +Z, left = -X, up = -Y.
        body = np.stack([cam[..., 2], -cam[..., 0], -cam[..., 1]], axis=-1)

        if self.pitch_rad:
            # Rotate about body +y (left). Positive angle tips +x downward, which is
            # what a nose-down camera mount does.
            c, s = math.cos(self.pitch_rad), math.sin(self.pitch_rad)
            x, y, z = body[..., 0], body[..., 1], body[..., 2]
            body = np.stack([c * x + s * z, y, -s * x + c * z], axis=-1)
        return body

    def project(self, direction) -> tuple[float, float]:
        """Pixel ``(u, v)`` a body-frame direction lands on — the inverse of
        :meth:`unit_vector`.

        ``direction`` need not be normalised. Points behind the camera project onto
        the far side of the image circle rather than raising, which is what the
        equidistant model says happens; callers that care must range-check the result
        against the frame.
        """
        vector = np.asarray(direction, dtype=float)
        if self.pitch_rad:
            c, s = math.cos(-self.pitch_rad), math.sin(-self.pitch_rad)
            x, y, z = vector
            vector = np.array([c * x + s * z, y, -s * x + c * z])

        # Body axes -> optical axes: right = -y, down = -z, forward = +x.
        optical = np.array([-vector[1], -vector[2], vector[0]])
        planar = math.hypot(optical[0], optical[1])
        theta = math.atan2(planar, optical[2])
        radius = self.focal_px * theta
        if planar <= 1e-12:
            return float(self.cx), float(self.cy)
        return (float(self.cx + radius * optical[0] / planar),
                float(self.cy + radius * optical[1] / planar))

    def bearing_elevation(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        """``(azimuth, elevation)`` in radians for pixel(s) ``(u, v)``.

        Azimuth is positive to the robot's LEFT (right-handed about ``+z``, matching
        the sign of a positive yaw command). Elevation is positive up.
        """
        d = self.unit_vector(u, v)
        azimuth = np.arctan2(d[..., 1], d[..., 0])
        elevation = np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1]))
        return azimuth, elevation

    def angle_between(self, p1, p2) -> np.ndarray:
        """Angle in radians subtended at the camera by two pixel points.

        Exact under the model — it is the angle between the two rays, so it stays
        correct out at the frame edge where the fisheye compresses hardest and a
        naive ``pixels / focal`` would read low.
        """
        d1 = self.unit_vector(p1[0], p1[1])
        d2 = self.unit_vector(p2[0], p2[1])
        dot = np.clip(np.sum(d1 * d2, axis=-1), -1.0, 1.0)
        return np.arccos(dot)

    def range_from_span(self, p1, p2, physical_m: float) -> float:
        """Distance to an object whose known extent ``physical_m`` spans ``p1`` to ``p2``.

        ``d = L / (2·tan(dtheta/2))`` — exact for an extent perpendicular to, and
        bisected by, the line of sight. A standing person viewed from a low camera is
        close enough to both. Returns ``inf`` for a span below the sensor's own
        resolution, which callers must treat as "no usable range".

        The floor is :attr:`pixel_angle_rad`, not a float epsilon. ``angle_between``
        goes through ``arccos`` of two nearly-parallel rays, which loses half its
        mantissa: a span of exactly zero measures ~2e-8 rad rather than 0, so an
        epsilon gate lets it through and reports a FINITE range of thousands of
        kilometres — past every ``isfinite`` check downstream. One pixel is both a
        real bound and one this arithmetic can actually see.
        """
        if physical_m <= 0.0:
            raise ValueError(f"physical_m must be positive, got {physical_m}")
        dtheta = float(self.angle_between(p1, p2))
        if dtheta <= self.pixel_angle_rad:
            return math.inf
        return physical_m / (2.0 * math.tan(dtheta / 2.0))

    def ground_point(self, u, v) -> tuple[float, float] | None:
        """Where the ray through ``(u, v)`` meets the floor, in body-frame metres.

        Debug/overlay only — see the module docstring for why this is NOT the ranger.
        Returns ``None`` when the ray points at or above the horizon.
        """
        d = self.unit_vector(u, v)
        if d[2] >= -1e-6:
            return None
        t = self.height_m / -d[2]
        return float(t * d[0]), float(t * d[1])


def solve_focal_px(residual_fn, width: int, lo_hfov_deg: float = 40.0,
                   hi_hfov_deg: float = 220.0, iterations: int = 60) -> float:
    """Find the ``focal_px`` minimising ``residual_fn(focal_px)``, by golden section.

    Both calibration routines in ``calibrate_camera.py`` reduce to a 1-D search over
    this single intrinsic, so they share the search. The residual need only be
    unimodal in the bracket, which holds for both (sum-of-squares in a monotone
    reparametrisation of the projection).

    The bracket is expressed in HFOV rather than pixels so the same bounds are
    meaningful at any image size.
    """
    lo = (width / 2.0) / math.radians(hi_hfov_deg / 2.0)  # widest FOV -> smallest f
    hi = (width / 2.0) / math.radians(lo_hfov_deg / 2.0)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = residual_fn(c), residual_fn(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = residual_fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = residual_fn(d)
    return (a + b) / 2.0
