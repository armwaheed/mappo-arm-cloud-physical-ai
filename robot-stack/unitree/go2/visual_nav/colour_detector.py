# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Find a known-coloured object by segmentation, for things no detector was trained on.

MobileNet-SSD knows twenty PASCAL VOC classes. A recycling bin is not one of them and
never will be, so no confidence threshold reaches it — measured on this robot's own
footage, nothing bin-like fires at any score. What the bin *does* have is a property the
office does not share: it is a large, saturated blue solid. That is enough to segment.

This is deliberately the cheap half of the problem. Measured on this robot's Jetson,
segmentation of a 1920x1080 frame at half resolution costs **10.2 ms** against the
person detector's 114 ms, so a colour target rides along with the person pass for under
a tenth of its cost, and the whole ranging path downstream — :class:`Detection`,
``range_detections``, the tracker — is reused unchanged.

IT IS ALSO THE FRAGILE HALF, and the gates below are what make it usable rather than a
demo. Colour alone finds every blue thing in the room: on the staged scene the naive
hue+saturation mask returns the bin, a second bin further down the corridor, a blue tag
on a cubicle, and — the one that matters — a tall strip of the glazed wall on the left.
Three shape gates separate a bin from a wall, and none of them is about colour:

  * **fill** — contour area over bounding-box area. A bin is convex and solid (measured
    0.85); the wall strip is a ragged sliver (0.35). This is the discriminating gate.
  * **aspect** — a bin is roughly as wide as it is tall (measured 164x183 px at 2.15 m).
  * **area** — below a few hundred pixels the box is too coarse to range anyway.

Everything surviving those is returned, largest first, because a second bin down the
corridor is a real obstacle and not a false positive. What this module must never do is
*guess*: an unlit or part-occluded bin should drop out and let the static map coast on
what it already knows, rather than be reported at a wrong size and so a wrong range.

Nothing here needs the robot — pass any BGR image. ``python3 test_colour_detector.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from person_detector import Detection, SizePrior

#: OpenCV packs hue into 0..179, half the usual degrees. Blue sits near 110.
HUE_MAX = 179
COLOUR_PROFILE_SCHEMA = "colour-profile/v1"


@dataclass(frozen=True)
class ColourProfile:
    """A coloured object worth finding: how it looks, how big it is, and its shape gates.

    The HSV window is deliberately wider than the object measured under one lighting
    condition. Saturation is what actually separates a coloured prop from an office —
    carpet, cubicle panels and paintwork are all near-grey — so the hue window can
    afford to be generous while ``sat_min`` does the work.

    Args:
        label: what to call it downstream. Becomes ``Detection.label``, and the tracker
            and static map key on it, so it must be distinct from any VOC class.
        hue_lo/hue_hi: inclusive hue window in OpenCV's 0..179 scale. ``hue_lo`` may
            exceed ``hue_hi`` to wrap through red at 0.
        sat_min/val_min: floors that reject grey (low saturation) and shadow (low value).
        height_m/width_m: the real object, for the size-prior ranging downstream. These
            are measurements, not guesses — every range scales linearly on them.
        radius_m: plan-view half-width the planner must keep clear of. Not derived from
            ``width_m`` because the two answer different questions: ``width_m`` is the
            extent the camera sees across the line of sight, ``radius_m`` is the
            footprint in any direction.
        min_area_px/min_fill/min_aspect/max_aspect: the shape gates. See the module
            docstring for what each one rejects.
    """

    label: str
    hue_lo: int
    hue_hi: int
    sat_min: int
    val_min: int
    height_m: float
    width_m: float
    radius_m: float
    min_area_px: int = 400
    #: Contour area over bounding-box area. LOWERED 0.55 -> 0.35 on 2026-08-19, measured
    #: rather than felt. 0.55 came from this unit's own footage at 2.15 m, square-on and
    #: well lit, where a bin fills 0.85 of its box. Live runs are not that: a bin at an
    #: angle, in the shadow of a cabinet, clipped by the frame edge, or with the recycling
    #: logo breaking the mask comes in at 0.36-0.48.
    #:
    #: Swept over the 63 recorded frames of the run that ended by driving over a bin, with
    #: two bins staged throughout:
    #:
    #:   ==========  ==================  ===================
    #:   min_fill    frames w/ >=1 bin   frames w/ BOTH bins
    #:   ==========  ==================  ===================
    #:   0.55            19 of 63             2 of 63
    #:   0.45            25                   8
    #:   0.35            28                  19
    #:   0.30            29                  21
    #:   0.25            29                  22
    #:   ==========  ==================  ===================
    #:
    #: At the shipped gate the stack saw both bins on 3% of frames, which is why landmarks
    #: went unobserved, accrued misses and were pruned mid-approach while the robot was
    #: still walking toward them. 0.35 is the knee: below it the curve is flat and only
    #: admits more irregular blue.
    #:
    #: This gate is what separates a bin from a same-coloured wall, so loosening it is not
    #: free — spurious blue on a humanoid's panels was observed passing at low fill during
    #: the same session. ``score`` IS this fill, so a marginal blob arrives labelled as
    #: marginal and a consumer can weight it.
    min_fill: float = 0.35
    min_aspect: float = 0.35
    max_aspect: float = 2.60
    evidence: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("hue_lo", "hue_hi", "sat_min", "val_min"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            maximum = HUE_MAX if name.startswith("hue") else 255
            if not 0 <= value <= maximum:
                raise ValueError(f"{name} must be within 0..{maximum}")
        for name in ("height_m", "width_m", "radius_m", "min_fill", "min_aspect", "max_aspect"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.min_area_px, bool) or not isinstance(self.min_area_px, int) \
                or self.min_area_px <= 0:
            raise ValueError("min_area_px must be a positive integer")
        if self.min_aspect > self.max_aspect:
            raise ValueError("min_aspect must not exceed max_aspect")
        for key, value in self.evidence:
            if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
                raise ValueError("profile evidence must contain non-empty string pairs")

    @property
    def prior(self) -> SizePrior:
        """Size prior for ``person_detector.estimate_range``.

        Given explicitly rather than through ``SizePrior.of_height``, which infers width
        from a PERSON's aspect ratio — for a bin that would claim 0.09 m across instead
        of the measured 0.27 m, and the width prior is what ranges the object once its
        base drops out of frame at close quarters.
        """
        return SizePrior(height_m=self.height_m, width_m=self.width_m)

    def _band(self, hue_lo: int, hue_hi: int) -> tuple:
        """``(lower, upper)`` HSV bounds for one contiguous hue range."""
        return (np.array([hue_lo, self.sat_min, self.val_min], np.uint8),
                np.array([hue_hi, 255, 255], np.uint8))

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """Binary mask of pixels inside the colour window, for an HSV image."""
        if self.hue_lo <= self.hue_hi:
            return cv2.inRange(hsv, *self._band(self.hue_lo, self.hue_hi))
        # Wrapped window (e.g. red spans 170..179 and 0..10): two bands, OR-ed.
        return cv2.bitwise_or(cv2.inRange(hsv, *self._band(self.hue_lo, HUE_MAX)),
                              cv2.inRange(hsv, *self._band(0, self.hue_hi)))


#: The staged prop. Measured on this unit's own footage at 2.15 m: 164x183 px, fill
#: 0.85, hue ~108. Height is the operator's tape measure (1 ft); width is what that
#: height implies from the same box, and matches a standard office recycling bin.
#: ``radius_m`` is half the width, i.e. the bin's own footprint with nothing added —
#: the planner's clearances are added on top and must not be double-counted here.
BLUE_BIN = ColourProfile(
    label="bin",
    hue_lo=95, hue_hi=135, sat_min=90, val_min=40,
    height_m=0.3048, width_m=0.27, radius_m=0.15,
)

#: Named profiles the CLI can select. Extend rather than teaching callers HSV.
PROFILES = {"bin": BLUE_BIN}


def load_colour_profile(path: str | Path) -> ColourProfile:
    """Load an evidence-backed custom static-colour profile from versioned JSON."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read colour profile {path}: {error}") from None
    if not isinstance(data, dict):
        raise ValueError("colour profile must be a JSON object")
    if data.get("schema") != COLOUR_PROFILE_SCHEMA:
        raise ValueError(
            f"colour profile schema must be {COLOUR_PROFILE_SCHEMA!r}, "
            f"got {data.get('schema')!r}"
        )
    evidence = data.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("custom colour profile requires a non-empty evidence object")
    label = data.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("custom colour profile label must be a non-empty string")
    for name in ("hue_lo", "hue_hi", "sat_min", "val_min", "min_area_px"):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"custom colour profile {name} must be an integer")
    for name in ("height_m", "width_m", "radius_m", "min_fill", "min_aspect", "max_aspect"):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise ValueError(f"custom colour profile {name} must be finite")
    fields = {
        "label": label,
        "hue_lo": data.get("hue_lo"),
        "hue_hi": data.get("hue_hi"),
        "sat_min": data.get("sat_min"),
        "val_min": data.get("val_min"),
        "height_m": data.get("height_m"),
        "width_m": data.get("width_m"),
        "radius_m": data.get("radius_m"),
        "min_area_px": data.get("min_area_px", 400),
        "min_fill": data.get("min_fill", 0.55),
        "min_aspect": data.get("min_aspect", 0.35),
        "max_aspect": data.get("max_aspect", 2.60),
        "evidence": tuple(evidence.items()),
    }
    try:
        return ColourProfile(**fields)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid colour profile {path}: {error}") from None


class ColourBlobDetector:
    """Segment one :class:`ColourProfile` out of a frame and return boxes.

    Args:
        profile: what to look for.
        detect_scale: segment at this fraction of the input's size. Colour blobs are
            large and low-frequency, so half resolution costs nothing in accuracy and
            quarters the pixel work; boxes are scaled back to the input's coordinates
            so callers never see the downscale.
        max_blobs: cap on how many boxes are returned, largest first. A cap rather than
            "the largest" because a second bin down the corridor is a real obstacle.
        blur_px: odd-sized median blur applied to the mask before contouring, to knock
            out the speckle a compressed JPEG leaves at colour edges. 0 disables it.
    """

    def __init__(self, profile: ColourProfile = BLUE_BIN, detect_scale: float = 0.5,
                 max_blobs: int = 3, blur_px: int = 5) -> None:
        if not 0.0 < detect_scale <= 1.0:
            raise ValueError(f"detect_scale must be in (0, 1], got {detect_scale}")
        if blur_px and blur_px % 2 == 0:
            raise ValueError(f"blur_px must be odd (or 0), got {blur_px}")
        self._profile = profile
        self._detect_scale = detect_scale
        self._max_blobs = max_blobs
        self._blur_px = blur_px
        # Reused across frames — building it per call showed up in the 10 Hz budget.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    @property
    def profile(self) -> ColourProfile:
        return self._profile

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Boxes for every blob passing the profile's gates, largest area first.

        Boxes are in ``image``'s own pixel coordinates, matching
        ``PersonDetector.detect``, so the two are interchangeable to callers.
        """
        scale = self._detect_scale
        small = image if scale == 1.0 else cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = self._profile.mask(hsv)
        if self._blur_px:
            mask = cv2.medianBlur(mask, self._blur_px)
        # Open then close: opening removes speckle that would inflate a box, closing
        # seals the recycling logo printed across the bin's face, which otherwise
        # splits one object into two half-height blobs and doubles its reported range.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # findContours returns (image, contours, hierarchy) on OpenCV 3 and
        # (contours, hierarchy) on 4.x. This robot is on 4.2; index from the END so
        # the module does not break on either.
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # Gates are applied at DETECTION scale, then boxes are scaled back, so
        # min_area_px is a threshold on the same pixels the mask was built from.
        found = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._profile.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            aspect = w / float(h)
            if not self._profile.min_aspect <= aspect <= self._profile.max_aspect:
                continue
            fill = area / float(w * h)
            if fill < self._profile.min_fill:
                continue
            found.append((area, x, y, w, h, fill))

        found.sort(key=lambda item: item[0], reverse=True)
        inverse = 1.0 / scale
        height, width = image.shape[:2]
        detections = []
        for _, x, y, w, h, fill in found[:self._max_blobs]:
            detections.append(Detection(
                x1=min(max(x * inverse, 0.0), width - 1.0),
                y1=min(max(y * inverse, 0.0), height - 1.0),
                x2=min(max((x + w) * inverse, 0.0), width - 1.0),
                y2=min(max((y + h) * inverse, 0.0), height - 1.0),
                # Fill is the honest confidence here: it is the gate that separates a
                # bin from a same-coloured wall, so reporting it lets the overlay and
                # the logs show WHY a blob was believed.
                score=float(fill),
                label=self._profile.label))
        return detections


def collapse_stacked_blobs(detections: list[Detection], *,
                           overlap_frac: float = 0.5,
                           refused: list | None = None) -> list[Detection]:
    """Collapse blobs sharing a bearing to the LOWEST one: one object, one obstacle.

    ``detect``'s closing kernel seals a logo printed across a face. It cannot seal a
    traffic cone, whose white stripe between the two red bands is wider than any kernel
    that would still preserve the bands — so segmentation returns BOTH red bands, as two
    boxes stacked in the same image column.

    That is not two obstacles, and the second copy is not merely redundant, it is WRONG
    AND FAR. ``range_detections`` sizes a blob against ``ColourProfile.prior``, which is
    calibrated on the lower band; the upper band is smaller in frame, so the same prior
    reads it as the lower band seen from further away. Measured on robot 1 against a
    single cone at 0.74 m:

        lower band   (1070, 488)-(1152, 592)   82 x 104 px   0.74 m  @ -48.6 deg
        upper band   (1084, 366)-(1132, 426)   48 x  60 px   1.29 m  @ -49.5 deg

    One cone, one bearing, two obstacles, the far one 1.75x out. A phantom standing
    behind a real cone is usually harmless — it is occluded by the thing that made it —
    but it moves with the robot, and on 2026-09-02 one landed at ``x=+3.40 y=-0.01``,
    dead on robot 1's goal line and 0.46 m PAST the goal, which ended the run.

    Keeping the LOWEST box is the right choice twice over: it is the one whose bottom
    edge is nearest the object's floor contact, so it is the one the prior was
    calibrated on AND the one ``ground_range`` could check. Two genuinely separate
    objects at one bearing collapse too, and that is safe in the only direction that
    matters: what survives is the NEARER of the two, and the one discarded stood behind
    it, already occluded and already unreachable without passing the first.

    ``refused`` collects ``(detection, source)`` exactly as ``range_detections`` does,
    because a silently dropped box is indistinguishable from an open world downstream.
    """
    kept: list[Detection] = []
    for candidate in detections:
        for index, existing in enumerate(kept):
            overlap = (min(candidate.x2, existing.x2)
                       - max(candidate.x1, existing.x1))
            narrower = min(candidate.x2 - candidate.x1, existing.x2 - existing.x1)
            if narrower <= 0.0 or overlap / narrower < overlap_frac:
                continue
            # Same column. The lower box wins; `y2` grows DOWNWARD in image
            # coordinates, so the larger `y2` is the one nearer the floor.
            if candidate.y2 > existing.y2:
                if refused is not None:
                    refused.append((existing, "stacked-blob"))
                kept[index] = candidate
            elif refused is not None:
                refused.append((candidate, "stacked-blob"))
            break
        else:
            kept.append(candidate)
    return kept
