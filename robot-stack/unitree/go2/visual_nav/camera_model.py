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

WHY ANGULAR SIZE FOR A PERSON, AND THE FLOOR CONTACT POINT FOR EVERYTHING ELSE. Angular
size gives ``d = L / (2·tan(dtheta/2))``, whose relative error just equals the relative
error in the measured pixel span — a 4% range error for a 20 px error on a 520 px box.
It degrades gracefully and does not care about pitch at all. That is why it ranges the
one object this robot has a measured size for, and it is the right default.

**It needs ``L``, and for an object nobody measured there is no ``L``.** That is not a
tuning problem, it is the whole of it: substituting another object's prior scales the
answer by the ratio of the two sizes, and the error is unbounded in the direction that
matters. Measured, in this repository's own telemetry — the two live runs of
2026-08-25 (``evidence/2026-08-25-peer-runs/``) ranged EVERY detection with the peer
robot's 0.514 m prior, including the goal chair. Over the eight sightings of that chair a
1.6°/18% ground ceiling keeps, the size prior reported it at 0.69-0.85 m where the floor
contact of the same boxes puts it at 0.88-1.69 m — and those boxes imply an object 0.85 m
tall, which is an office chair and is 1.7x the prior it was ranged with. A prior too small
reads NEAR, which is merely useless; a prior too large reads FAR, which is the direction
that walks into things.

:meth:`ground_range` is the estimator with no ``L`` in it at all. Intersect the ray
through the object's lowest pixel with the floor and read the range off the geometry:
``d = h / tan(elevation)``. It costs one ray, it needs no class and no prior, and it is
what makes an object a stranger dropped in the arena rangeable at all.

ITS ERROR IS ENTIRELY THE PITCH, AND IT IS A FUNCTION OF RANGE — WHICH IS WHY IT IS
GATED RATHER THAN TRUSTED. ``|Δd/d| = δ·(d/h + h/d)`` for a pitch error ``δ``, so on this
0.32 m mount a 2° trunk wobble puts the FAR bound +10% out at 0.72 m, +14% at 1.0 m,
+20% at 1.5 m and +49% at 3 m — far being the side that walks into things. The 3 m
figure is where this docstring used to stop, and *at 3 m the conclusion was right*:
2.25 m to 4.49 m, which is not a measurement. What it missed is that the same
expression is inside the 18% the tracker already budgets for its BEST source
(``tracker.RANGE_SIGMA_FRACTION``) out to about 1.33 m, and that this detector's recall
is 91% inside 1.1 m and 0 of 315 beyond 2.7 m — so the band in which the ground plane is
unusable is a band no detection arrives from. :meth:`ground_range_limit` turns
that into a number the caller chooses rather than a rule this module asserts.

Neither estimator escapes the NEAR wall, and it is the same wall for both:
``h / tan(half_vfov)`` = 0.719 m on this unit, below which the object's floor contact has
left the bottom of the frame. There is nothing to intersect and nothing to measure, and
a landmark inside it is memory rather than perception. That figure is the BEST case and
is quoted around this repository as though it were the only one: it is the centre column,
and the bottom-corner ray is shallower, so an object at the frame edge loses its contact
point at 0.806 m instead. :meth:`ground_range` at ``(u, height - 1)`` is the per-column
answer.

The camera's static pitch is carried as a mounting parameter and applied by
:meth:`unit_vector`; this module does NOT fold live IMU pitch in. For AZIMUTH that is
free — tilting shifts a bearing only through the cosine of the tilt, so a 10° pitch moves
one by under 0.5° anywhere in this frame, well inside the bearing noise the tracker
assumes. For :meth:`ground_range` it is not free, and ``pitch_error_rad`` is where the
caller states how much wobble it is willing to be wrong by. **Nobody has recorded this
robot's trunk pitch while walking, so there is no default here.** The nearest thing to a
measurement is an upper bound: over 71 unclipped sightings in six tracks of the two
2026-08-25 runs, the frame-to-frame disagreement between the size-prior range and the
ground range implies a combined angular error of 1.6° median, 4.5° p90 — combined,
because the size prior's own box-height noise is in that number too.

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

# ── Where the lens is, per platform ─────────────────────────────────────────
# Floor to the optical centre with the robot STANDING. Measured on the Go2, and a
# statement about the Go2 only. This module is not Go2-only: the Lite3 stack imports it
# through `robot-stack/deep_robotics/lite3/`, and the Lite3's lens height has never been
# measured — issue #13 still lists it as outstanding.
#
# That is why `FisheyeCamera.height_m` has NO default. It used to default to this number,
# so any Lite3 calibration that did not explicitly pass one inherited the Go2's, silently,
# in a field that bounds a width-derived range estimate in `person_detector`. Name the
# platform's value at the call site, or the model has no floor and says so.
GO2_CAMERA_HEIGHT_M = 0.32

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
        height_m: optical centre above the floor when the robot stands, in metres.
            NO DEFAULT, deliberately — see :data:`GO2_CAMERA_HEIGHT_M`. It is NOT the
            ranger — see the module docstring for why the ground-plane intersection is
            unusable here — but it is not overlay-only either: :meth:`ground_point`
            uses it for the debug overlay, and ``person_detector.object_fit_range``
            uses it to work out the distance below which a person no longer fits in
            frame, which caps a width-derived range. So it does feed a range bound, and
            a wrong value moves that bound. ``None`` means "not measured on this
            platform"; both of those callers go through :meth:`require_height_m` and
            refuse rather than substitute another robot's number.
    """

    width: int
    height: int
    focal_px: float
    cx: float
    cy: float
    pitch_rad: float = 0.0
    height_m: float | None = None

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

        An unset :attr:`height_m` is written as ``null`` rather than omitted, and that is
        the honest record: the fitter considered the lens height and had nothing to put
        there. The Lite3 commissioning wrapper reads this field straight back out of the
        file (``stamp_lens_height``) and reports what it replaced, so a ``null`` is what
        tells its operator the number is theirs to measure.
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

    def require_height_m(self, consequence: str) -> float:
        """:attr:`height_m`, or a refusal naming what it was needed for.

        One accessor rather than an ``is None`` check at each call site, so the two
        callers cannot drift into telling an operator two different things to do. It is
        public because one of them is in another module (``person_detector``).
        """
        if self.height_m is None:
            raise ValueError(
                f"this camera model states no lens height, so {consequence}. height_m is "
                f"a per-platform MEASUREMENT — floor to the optical centre, robot "
                f"standing — and there is no default that is right for two robots: "
                f"{GO2_CAMERA_HEIGHT_M} m is the Go2's (camera_model.GO2_CAMERA_HEIGHT_M) "
                f"and a Lite3 that inherited it would move a real range bound. Pass "
                f"height_m= explicitly, or load a calibration file that states one — "
                f"deep_robotics/lite3/commissioning/camera_calibration.py --lens-height "
                f"writes one for a Lite3."
            )
        return self.height_m

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

        Ray-plane intersection under the EQUIDISTANT model, not a homography. The two
        are not interchangeable: a homography between the image and the floor is a
        pinhole construct, and on an 85.3° measured fisheye it is wrong by more than the
        thing it is estimating out at the frame edge — which is exactly where an obstacle
        the robot is about to swerve past sits. Every pixel gets its own ray instead.

        Returns ``None`` when the ray points at or above the horizon, and refuses when
        :attr:`height_m` is unset: with no lens height there is no floor to intersect,
        and ``None`` already means something else here.

        Scalar only. ``unit_vector`` vectorises and this deliberately does not — the
        horizon test and the intersection would each need a masked form, and the one
        caller ranges at most a handful of boxes per frame.
        """
        height_m = self.require_height_m("there is no floor to intersect")
        d = self.unit_vector(u, v)
        if d[2] >= -1e-6:
            return None
        t = height_m / -d[2]
        return float(t * d[0]), float(t * d[1])

    def ground_range(self, u, v) -> float | None:
        """Plan-view range to the floor point under ``(u, v)``, in metres.

        THE ONE RANGE SOURCE ON THIS ROBOT THAT NEEDS NO SIZE PRIOR, which is what makes
        an unrecognised object rangeable at all. ``None`` for a ray at or above the
        horizon, as :meth:`ground_point`.

        Plan view, i.e. ``hypot`` of the body-frame intersection, because that is the
        quantity every consumer downstream means by "range": ``tracker.Observation``
        places an obstacle at ``robot + range·(cos, sin)`` of a bearing, in the floor
        plane. The slant range would be longer by ``h``-over-``d``, ~5% at 1 m.

        ⚠️ IT IS ONLY A RANGE TO THE OBJECT IF THE PIXEL IS THE OBJECT'S FLOOR CONTACT.
        Give it the bottom edge of a box that was cut off by the frame, or of something
        standing on a table, and the elevation is shallower than the contact point's, so
        the range reads FAR — the direction that walks into things. The caller owns that
        test; ``person_detector.GroundRanger`` is the one that makes it.
        """
        point = self.ground_point(u, v)
        if point is None:
            return None
        return math.hypot(*point)

    def ground_range_bounds(self, range_m: float,
                            pitch_error_rad: float) -> tuple[float, float]:
        """``(nearest, farthest)`` a :meth:`ground_range` of ``range_m`` could really be.

        The whole error budget of the contact-point ranger, exactly rather than
        linearised, for a camera whose pitch is wrong by up to ``pitch_error_rad``:

            ``d = h / tan(e)``, ``e = atan(h/d)`` -> ``h / tan(e ± δ)``

        The two sides are NOT symmetric and the asymmetry is the safety-relevant half.
        Nose-down error shortens the estimate, which is harmless. Nose-up error lengthens
        it, and once ``δ`` reaches ``e`` the far bound is ``inf`` — the ray no longer
        meets the floor at all. On this 0.32 m mount that happens at 9.2 m for 2°, and
        the growth on the way there is what :meth:`ground_range_limit` gates.

        First-order the same statement is ``|Δd/d| = δ·(d/h + h/d)``, which is the form
        worth carrying in your head: the error grows LINEARLY in range for a fixed
        wobble, so this estimator is good exactly where it is needed and bad where the
        detector has stopped producing detections anyway.
        """
        # `isfinite` FIRST, and not as tidiness: every comparison below is `<= 0.0`, and
        # NaN fails all of them, so a NaN would sail through and come back out as a NaN
        # bound that no downstream `>` test can ever be true against. That is a gate
        # failing open, which is the direction this repository has been bitten by before.
        if not math.isfinite(range_m) or range_m <= 0.0:
            raise ValueError(f"range_m must be finite and positive, got {range_m}")
        if not math.isfinite(pitch_error_rad) or pitch_error_rad < 0.0:
            raise ValueError(
                f"pitch_error_rad must be finite and not negative, got {pitch_error_rad}")
        height_m = self.require_height_m("the ground-range error cannot be bounded")
        elevation = math.atan2(height_m, range_m)
        nearest = height_m / math.tan(elevation + pitch_error_rad)
        if pitch_error_rad >= elevation:
            return nearest, math.inf
        return nearest, height_m / math.tan(elevation - pitch_error_rad)

    def ground_range_limit(self, pitch_error_rad: float,
                           max_error_frac: float) -> float:
        """Farthest range whose :meth:`ground_range_bounds` stay inside ``max_error_frac``.

        The ceiling on contact-point ranging, expressed as the two things a caller
        actually knows rather than as a constant this module asserts:

          * ``pitch_error_rad`` — how far the camera's pitch may be from the mounting
            value while the robot walks. **Unmeasured on this robot**; see the module
            docstring for the 1.6°-median / 4.5°-p90 upper bound the 2026-08-25 telemetry
            supports, and for why it is an upper bound rather than the number.
          * ``max_error_frac`` — the relative range error the consumer can absorb.
            ``tracker.RANGE_SIGMA_FRACTION`` (0.18) is the natural reference point: it is
            what the filter already budgets for the size-prior source it trusts most.

        Solved in closed form off the FAR bound, since far is the unsafe side. With
        ``T = tan(e) = h/d``, ``D = tan(δ)`` and ``eps = max_error_frac``, requiring
        ``h/tan(e-δ) <= (1+eps)·d`` reduces to ``D·T² - eps·T + (1+eps)·D = 0``; the
        smaller root is the larger range.

        Returns ``0.0`` when no range satisfies it — a real answer, not a failure. Below
        roughly ``2·tan(δ)`` of tolerance the wobble alone exceeds the budget at every
        range, and a caller asking for 5% accuracy from a 2° wobble is asking for
        something the geometry does not contain. ``inf`` for a stated pitch error of
        exactly zero, which is a claim about the mount, not a measurement.
        """
        if not math.isfinite(pitch_error_rad) or pitch_error_rad < 0.0:
            raise ValueError(
                f"pitch_error_rad must be finite and not negative, got {pitch_error_rad}")
        if not math.isfinite(max_error_frac) or max_error_frac <= 0.0:
            raise ValueError(
                f"max_error_frac must be finite and positive, got {max_error_frac}")
        height_m = self.require_height_m("the ground-range ceiling cannot be computed")
        if pitch_error_rad >= math.pi / 2.0:
            return 0.0
        tan_delta = math.tan(pitch_error_rad)
        if tan_delta <= 0.0:
            return math.inf
        discriminant = (max_error_frac ** 2
                        - 4.0 * tan_delta ** 2 * (1.0 + max_error_frac))
        if discriminant < 0.0:
            return 0.0
        tan_elevation = (max_error_frac - math.sqrt(discriminant)) / (2.0 * tan_delta)
        if tan_elevation <= 0.0:
            return math.inf
        return height_m / tan_elevation


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
