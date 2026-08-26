# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Detect people (and other movers) in a frame, and range them monocularly.

Detector: MobileNet-SSD (Caffe, 21-class PASCAL VOC) through ``cv2.dnn``. Chosen for
what this robot actually has — OpenCV 4.2 with **no CUDA**, no torch, no onnxruntime,
and TensorRT with no usable Python buffer bindings. So inference is on four Cortex-A78
cores, measured on this unit at **131 ms @ 300x300** and **76 ms @ 224x224**. The
detector therefore runs in its own thread at ~7 Hz and the controller consumes its
latest output at 10 Hz; see ``visual_nav.py``.

RANGING — and the frame-edge problem that shapes it. Range comes from the angle a
known physical size subtends (:meth:`FisheyeCamera.range_from_span`), because there is
no depth sensor on this unit. A standing adult's height is the good prior. But work
the geometry through and the camera bites: it sits ~0.32 m up and has a ~67° vertical
field, so **a person closer than ~2.1 m has their head out of frame** — precisely the
band where avoidance matters. A truncated box is shorter than the person, so the
height estimate reads FAR, in the one direction that is dangerous.

:func:`estimate_range` handles that head-on. Truncation in a dimension turns that
dimension's estimate into an *upper bound* on range, so the code picks by which
dimension is intact:

  * box fully inside the frame  -> height prior. Accurate; ~4% per 20 px of box error.
  * top/bottom clipped          -> shoulder-width prior, capped at the distance where
    a person stops fitting (computed from the model, not hard-coded). Width is noisier
    — a person in profile reads ~0.3 m not ~0.5 m — but that error makes them seem
    NEARER, which is the safe direction.
  * clipped both ways           -> they fill the frame; report a fixed close range.

TWO TIERS, ONE FORWARD PASS (opt-in; see :meth:`PersonDetector.detect_tiered`). The
mover tier is the one described above and is unchanged. The STATIC tier exists because
of a floor rather than a model: the published prototxt bakes
``confidence_threshold: 0.25`` into its ``DetectionOutput`` layer, which applies it
INSIDE ``forward()``, so no Python-side threshold can reach below it. On the one Lite3
frame containing a cardboard box the network scores that box **0.1221** and then deletes
it — the box is suppressed, not missed. :func:`prototxt_with_floor` lowers that layer's
floor for a copy of the file, and the sub-threshold rows are then read off the SAME pass
and offered to ``static_map`` rather than to the tracker. Nothing about the mover tier
changes, and no second inference is run.

Off unless constructed with ``static_confidence``. What it costs when on is measured in
:data:`DEFAULT_STATIC_CONFIDENCE`, :data:`STATIC_MIN_AREA_FRAC` and :func:`static_shaped`;
what it BUYS is not measured and cannot be from one frame, which those docstrings say
plainly.

Nothing here needs the robot: pass any BGR image. ``python3 test_person_detector.py``
covers the ranging and truncation logic against a synthetic model.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_model import PERSON_HEIGHT_M, PERSON_WIDTH_M, FisheyeCamera

# MobileNet-SSD is trained on PASCAL VOC; index 0 is background.
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
# Classes that can get up and walk into the robot's path. Static furniture is
# deliberately excluded: this pipeline models MOVING obstacles, and a chair detected
# as an obstacle-with-velocity would be noise. Static geometry is out of scope here —
# see the README.
DYNAMIC_CLASSES = ("person",)

#: VOC labels a SUB-THRESHOLD detection may carry to be offered to the static map. This
#: is not "the classes a cardboard box comes back as" — it cannot be, on the evidence
#: below — it is the set of blocky floor-standing furniture whose members are, whatever
#: the label says, things a robot should not walk into.
#:
#: THE LABELS ARE NOISE AND THIS FILTER DOES NOT PRETEND OTHERWISE. Rendered and
#: inspected on the 2026-08-24 cross-day frames, the detections this tier keeps land on a
#: real blue recycling bin under the label ``tvmonitor`` at 0.123 in one frame and
#: ``chair`` at 0.146 in another. Both are correct as OBSTACLES and meaningless as
#: classifications. What the filter buys is measured and modest: on 216 peer-absent
#: corridor frames it cuts the per-frame fire rate from 74.5% class-agnostic to 46.3%,
#: mostly by discarding the whole-wall ``aeroplane`` and ``train`` slabs that a lowered
#: score floor lets through. ``person`` is deliberately absent — a person belongs to the
#: tracker, and :func:`static_shaped` refuses person-shaped boxes a second time.
STATIC_CLASSES = ("chair", "sofa", "diningtable", "pottedplant", "tvmonitor")

#: Score floor for the STATIC tier. Below :data:`DEFAULT_CONFIDENCE`, and below the 0.25
#: the published prototxt bakes into its own ``DetectionOutput`` layer — which is why
#: :func:`prototxt_with_floor` exists at all.
#:
#: ⚠️ THIS NUMBER IS NOT VALIDATED FOR RECALL AND CANNOT BE. Exactly ONE frame containing
#: a cardboard box exists (a Lite3 dry run in the Shanghai office, 1280x720). In it the
#: box scores **0.1221** as ``chair`` at input 300. One observation cannot set a
#: threshold, so this floor is not fitted to it: it is placed where the FALSE-ALARM cost
#: was measured to be nil, and the single box then clears it by 1.22x rather than by
#: rounding. On 705 frames of empty corridor (``neg_prone`` + ``neg_standing``,
#: 2026-08-24) the full gate fires on **0 of 705** frames at this floor. Drop to 0.05 and
#: that becomes 20% of frames; the cliff is real and it is just below here.
DEFAULT_STATIC_CONFIDENCE = 0.10

#: Fraction of the frame a static candidate must fill. THE AREA GATE IS THE ONE WITH
#: MARGIN, and it is doing most of the work — on 216 peer-absent corridor frames it cuts
#: the fire rate from 46.3% to 27.8% where the aspect band cuts only 46.3% to 43.5%.
#:
#: It is also the gate that can be argued from physics rather than fitted: this pipeline
#: has no depth sensor and ranges by a size prior it must be TOLD, so a small distant
#: blob is both the least trustworthy detection and the least urgent one. Requiring 4% of
#: the frame keeps the decision to the near field where a wrong answer costs a collision.
#: The one box observed fills 11.0%, i.e. 2.75x this floor.
STATIC_MIN_AREA_FRAC = 0.04

#: Upper area bound. A box larger than this is the detector describing the corridor, not
#: an object in it: the 0.1475 ``person`` covering 81% of the Lite3 frame is the worked
#: example, and mapping it as a landmark would put a 0.6 m disc on top of the robot.
STATIC_MAX_AREA_FRAC = 0.60

#: Lower aspect bound (height/width). Below this the box is a horizontal slab — a
#: skirting rail, a window mullion, the wall-wide ``aeroplane`` at aspect 0.19 in the
#: Lite3 frame — not a floor-standing object. The UPPER bound is not a separate number:
#: it is :data:`PERSON_ASPECT_MIN`, because the thing this tier must never do is quietly
#: reclassify a person as scenery. See :func:`static_shaped`.
STATIC_ASPECT_MIN = 0.6

#: Box aspect (height/width) at or above which a detection is treated as PERSON-SHAPED
#: and stops the robot, whatever VOC called it. See
#: :meth:`RangedDetection.person_shaped` for why shape decides this and the label
#: does not. 2.0 is the geometric midpoint of the two populations it separates: the peer
#: corpus tops out at 0.99 over 1,159 unclipped boxes, and a standing adult is
#: ``PERSON_HEIGHT_M / PERSON_WIDTH_M`` = 3.40. Raising it towards 3.4 starts letting
#: crouching or partly-occluded people through to the policy; lowering it towards 1.0
#: starts holding for the peer this exists to pass.
PERSON_ASPECT_MIN = 2.0


# Preprocessing constants baked into the published MobileNet-SSD weights.
_SSD_SCALE = 1.0 / 127.5
_SSD_MEAN = 127.5

# Distance reported when a box is clipped on BOTH axes, i.e. the person fills the
# frame. At that point no span is measurable; this is a stand-in that is inside every
# sane stop radius, so the planner halts rather than trusting a number it cannot get.
FILLS_FRAME_RANGE_M = 0.8

# Deliberately below the customary 0.5, because the two errors are not symmetric: a
# false positive costs a needless stop, a miss costs walking into someone. Two
# measurements back it. On hard footage (pedestrians ~45 px tall in the network's
# squashed input) dropping 0.5 -> 0.4 took detections from 0.61 to 1.32 per frame. In
# the robot's own workplace — 139 frames of an empty office — the detector produced
# ZERO false positives all the way down to 0.2. 0.4 rather than 0.2 because that
# false-positive figure comes from one static scene and should not be over-read.
DEFAULT_CONFIDENCE = 0.4


@dataclass(frozen=True)
class Detection:
    """One detected object, in FULL-RESOLUTION image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str

    @property
    def width_px(self) -> float:
        return self.x2 - self.x1

    @property
    def height_px(self) -> float:
        return self.y2 - self.y1

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def clipped(self, width: int, height: int, margin_px: float = 2.0
                ) -> tuple[bool, bool]:
        """``(vertically_clipped, horizontally_clipped)`` against the frame edges.

        ``margin_px`` absorbs the box regression landing a pixel or two inside the
        edge on an object that genuinely runs off it.
        """
        vertical = self.y1 <= margin_px or self.y2 >= height - margin_px
        horizontal = self.x1 <= margin_px or self.x2 >= width - margin_px
        return vertical, horizontal


@dataclass(frozen=True)
class SizePrior:
    """How big the tracked thing is — the only reason a mono camera can range it.

    Defaults describe a standing adult. Override it to track something else (a
    hand-slid box standing in for a person when no volunteer is around); the whole
    metric scale is proportional to :attr:`height_m`, so it must be measured, not
    assumed.
    """

    height_m: float = PERSON_HEIGHT_M
    width_m: float = PERSON_WIDTH_M

    @classmethod
    def of_height(cls, height_m: float, width_m: float | None = None) -> SizePrior:
        """A prior for an object of known height, with an optional measured width.

        ⚠️ PASS ``width_m`` FOR ANYTHING THAT IS NOT PERSON-SHAPED. Without it the
        width is inferred from a standing adult's aspect ratio, and the old docstring
        here claimed that "only matters when the box is vertically clipped, which for
        anything much smaller than a person happens only when it is nearly touching the
        lens". Both halves of that are wrong for a quadruped, and the error runs in the
        dangerous direction.

        Measured on the 2026-08-24 peer corpus: **39% of 1,903 boxes touch a frame
        edge**, not a rare event, because a wide low robot fills the frame sideways long
        before a person would. And the inferred width for a 0.514 m peer is
        ``0.514 * 0.50/1.70 = 0.151 m`` against a real ~0.31 m head-on. Ranging a
        clipped box by a width prior 2x too small reports the object 2x too NEAR:
        segments of that corpus come out at 0.09-0.14 m, i.e. inside the robot's own
        footprint, which the planner reads as an unavoidable collision and stops for.
        """
        return cls(height_m=height_m,
                   width_m=(width_m if width_m is not None
                            else height_m * (PERSON_WIDTH_M / PERSON_HEIGHT_M)))


#: The default prior. A module-level singleton rather than a `SizePrior()` call in
#: each signature: the instance is frozen so sharing it is safe, and naming it makes
#: "these all default to a person" explicit at every call site.
PERSON_PRIOR = SizePrior()


def object_fit_range(camera: FisheyeCamera, prior: SizePrior = PERSON_PRIOR) -> float:
    """Closest distance at which the object still fits in frame, in metres.

    Below this one of its ends leaves the image, so a height-derived range is only an
    upper bound. Derived from the model rather than measured once and hard-coded, so it
    stays correct after calibration changes the focal length.

    BOTH ENDS CAN BE THE BINDING ONE, and which it is depends on how the object stands
    relative to the lens. For something taller than the camera — a person against the
    Go2's 0.32 m mount — the crown is what leaves first. For something SHORTER, the top
    never leaves at all and the base does: a 0.30 m bin loses its foot below 0.72 m while its
    rim stays comfortably in view. Taking only the top (``height_m - camera.height_m``)
    returns 0.0 for every such object, and ``estimate_range`` caps a width-derived range
    at this value — so a bin would be reported at ZERO metres, i.e. inside the robot,
    and the planner would hold for the rest of the run against an obstacle it was
    standing on. Take whichever end leaves first.

    Assumes a level camera and a target near the image centre-line; it is a bound to
    switch estimators on, not a precise threshold.

    REFUSES when the camera states no lens height. There is no safe fallback: `inf` would
    disable the cap this function exists to impose, and 0.0 would report every object as
    standing inside the robot. `FisheyeCamera.height_m` used to default to the Go2's
    0.32 m, which is how a Lite3 would have inherited it.
    """
    lens_height_m = camera.require_height_m("the fit distance cannot be computed")
    half_vfov = (camera.height / 2.0) / camera.focal_px
    if half_vfov <= 0.0 or half_vfov >= math.pi / 2.0:
        return math.inf
    tan_half_vfov = math.tan(half_vfov)
    top_leaves = max(0.0, prior.height_m - lens_height_m) / tan_half_vfov
    base_leaves = max(0.0, lens_height_m) / tan_half_vfov
    return max(top_leaves, base_leaves)


def estimate_range(detection: Detection, camera: FisheyeCamera,
                   prior: SizePrior = PERSON_PRIOR) -> tuple[float, str]:
    """Monocular range to ``detection`` in metres, plus which prior produced it.

    Returns ``(range_m, source)`` with ``source`` one of ``"height"``, ``"width"`` or
    ``"frame-fill"`` so callers can weight the measurement (the tracker does).
    """
    vertical_clip, horizontal_clip = detection.clipped(camera.width, camera.height)
    cx, cy = detection.centre

    if not vertical_clip:
        span = camera.range_from_span((cx, detection.y1), (cx, detection.y2),
                                      prior.height_m)
        return span, "height"

    if not horizontal_clip:
        span = camera.range_from_span((detection.x1, cy), (detection.x2, cy),
                                      prior.width_m)
        # A clipped-height box cannot belong to something far away, whatever the width
        # says — cap it at the fit distance so a narrow (profile) reading cannot be
        # inflated into a false "they're 6 m off".
        fit = object_fit_range(camera, prior)
        if span > fit:
            # THE CAP BOUND, so what comes back is the cap, not the measurement, and the
            # caller has to be able to tell. Reported as its own source because the two
            # deserve different trust: a width-prior range is a weak measurement, a
            # capped one is a CONSTANT that cannot move however the robot does.
            #
            # Live run of 2026-08-19: approaching a bin, the width span read 0.748-0.907 m
            # across thirteen frames while the fit range is 0.719 m, so every single one
            # was capped and the reported range was 0.719 m to three decimals for five
            # seconds — while the bearing tracked correctly from +13 to +25 deg. The
            # planner's gap sat at 0.20-0.24 m against a 0.25 m hard gap, vetoed every
            # tick, and the robot deadlocked against a number that could not move.
            return fit, "width-capped"
        return span, "width"

    return FILLS_FRAME_RANGE_M, "frame-fill"


@dataclass(frozen=True)
class RangedDetection:
    """A detection with its metric range attached.

    Exists so range, source and box travel together. Carrying them as three parallel
    lists invites the bug where a detection with no usable range is dropped from the
    range list but not the box list, silently shifting every later box's label onto
    the wrong person.
    """

    detection: Detection
    range_m: float
    bearing_rad: float
    source: str

    @property
    def label(self) -> str:
        return self.detection.label

    def person_shaped(self, width: int, height: int) -> bool:
        """Whether this must STOP the robot, judged on shape rather than on the VOC label.

        WHY THE VOC LABEL CANNOT DECIDE THIS. ``mappo_bridge.HOLD_LABELS`` stops the
        robot dead for a ``person`` and routes everything else to the policy. That rule
        needs the classifier to tell a person from a robot, and this classifier cannot:
        on 12 consecutive live frames the Go2 Wheel was labelled ``person`` every single
        time, and across the 2026-08-24 corpus the same peer came back as ``motorbike``
        613 times, ``chair`` 372, ``aeroplane`` 200 and ``person`` 109. The labels are
        noise; the boxes are good. So the split was resting on the one thing the
        detector does badly, and it failed in BOTH directions — the peer was held like a
        person, and a person read as ``motorbike`` was silently handed to the policy.

        WHAT DECIDES IT INSTEAD. Box aspect, which is scale-free, so it needs no range.
        That matters because there IS no independent range on this robot: the
        ground-plane intersection is unusable (``camera_model`` documents a 2 deg trunk
        wobble swinging a 3 m estimate from 2.3 m to 4.4 m, and a trotting Go2 does
        more), and ranging by a size prior and then inferring size from that range just
        returns the prior. Aspect sidesteps the circle entirely.

        THE MARGIN, measured on 1,159 unclipped boxes of the peer corpus: median 0.78,
        p99 0.92, **max 0.99**. The repo's own standing-adult prior is
        ``1.70 / 0.50 = 3.40``. :data:`PERSON_ASPECT_MIN` sits at 2.0, roughly the
        geometric mean of the two, so it clears the peer's worst observed case by 2x and
        sits 1.7x below a person.

        ⚠️ VERTICAL CLIPPING HOLDS; HORIZONTAL CLIPPING DOES NOT, AND THE ASYMMETRY IS
        THE WHOLE POINT. A person whose head leaves the frame gives a SHORTER box, so
        the aspect falls and they start to look like a quadruped — the dangerous
        direction, at close range, where being wrong costs the most. That branch must
        fail to ``person``, and it does.

        Horizontal clipping does the opposite. Cutting width RAISES ``height/width``, so
        a partly-out-of-frame object drifts towards the person-shaped verdict on its own.
        The rule is already conservative there and needs no separate branch: a peer at
        0.8 m is 1129 px wide against 829 px tall (aspect 0.73), and has to lose roughly
        two thirds of its width before the aspect reaches 2.0 and it holds. Refusing on
        horizontal clipping as well was the first cut of this rule, and it cost the
        close-range window for nothing: measured on the first live run, the peer clipped
        the right edge as the robot swerved past it, flipped to ``person_shaped`` and
        froze the robot beside it. 39% of the 2026-08-24 corpus is clipped on SOME edge;
        only the vertical share is genuinely unclassifiable.

        The consequence to understand before relying on this: a peer that is vertically
        cut off — closer than about 0.61 m, where its 0.514 m fills the 1080-pixel frame
        — holds the robot rather than being swerved around. That is the intended floor,
        not a bug.
        """
        vertical, _horizontal = self.detection.clipped(width, height)
        if vertical:
            return True
        if self.detection.width_px <= 0.0:
            return True
        return self.detection.height_px / self.detection.width_px >= PERSON_ASPECT_MIN


def static_shaped(detection: Detection, width: int, height: int) -> bool:
    """Whether ``detection`` may be offered to the STATIC map, judged on shape and size.

    THE MIRROR OF :meth:`RangedDetection.person_shaped`, AND DELIBERATELY ITS COMPLEMENT.
    That method decides what must STOP the robot and answers on box aspect because the
    VOC label cannot be trusted — the same peer came back ``motorbike`` 613 times,
    ``chair`` 372 and ``person`` 109 across one corpus. This function inherits the whole
    of that argument. It is asking the opposite question of the same unreliable
    classifier, so it must be at least as sceptical.

    WHY THIS RUNS BEFORE RANGING, unlike ``person_shaped`` which runs after. Ranging
    needs a size prior, and the prior for a static prop is not the person prior — the
    caller must supply a measured one. Handing an ungated detection to
    :func:`range_detections` with a prop's prior is how a person 0.8 m away gets reported
    as a 0.3 m object 6 m off. So the gate is on :class:`Detection`, before a prior has
    been chosen.

    ⚠️ THE PERSON REFUSAL IS THE SAFETY-CRITICAL HALF, and it is not symmetry for its own
    sake. A landmark in :mod:`static_map` is planned against with ``person_shaped=False``
    hard-coded, so anything that reaches the map is a thing the robot will route AROUND
    at policy speed and never HOLD for. Route a person in there and the one behaviour
    this robot ships — give way to people — is silently disabled for them. That is not
    hypothetical: rendered on frame ``peer_cross5_005`` of the 2026-08-24 cross-day set,
    a person's legs at close range come back as ``chair`` at 0.248, in-band on both
    aspect and area. Refusing person-shaped boxes here is what keeps them with the
    tracker, and it costs 26.4% -> 19.0% of the peer-absent fire rate as a bonus.

    The refusal reuses ``person_shaped``'s two tests unchanged, for the reasons argued
    there rather than re-argued here: an aspect at or above :data:`PERSON_ASPECT_MIN`, and
    ANY vertical clipping — a person whose head has left the frame gives a shorter box
    whose aspect falls toward furniture, which is the dangerous direction at exactly the
    range where being wrong costs the most.
    """
    box_width = detection.width_px
    box_height = detection.height_px
    if box_width <= 0.0 or box_height <= 0.0:
        return False
    vertical, _horizontal = detection.clipped(width, height)
    if vertical:
        return False
    aspect = box_height / box_width
    if aspect >= PERSON_ASPECT_MIN or aspect < STATIC_ASPECT_MIN:
        return False
    frame_area = float(width) * float(height)
    if frame_area <= 0.0:
        return False
    area_fraction = (box_width * box_height) / frame_area
    return STATIC_MIN_AREA_FRAC <= area_fraction <= STATIC_MAX_AREA_FRAC


#: Matches the ONE ``confidence_threshold`` the published MobileNet-SSD prototxt carries,
#: inside its ``detection_output_param``. Anchored on the whole assignment so a stray
#: number elsewhere in the 1,900-line file cannot be rewritten by accident.
_CONFIDENCE_FLOOR_RE = re.compile(r"(confidence_threshold:\s*)([0-9]*\.?[0-9]+)")


def prototxt_with_floor(text: str, floor: float) -> str:
    """The prototxt ``text`` with its ``DetectionOutput`` score floor lowered to ``floor``.

    WHY THIS IS NECESSARY AT ALL, and why passing a lower ``confidence=`` to
    :class:`PersonDetector` is not enough. ``DetectionOutput`` is a network LAYER. It
    applies ``confidence_threshold`` itself, during ``forward()``, before any Python sees
    a row — so the Python-side ``score < self._confidence`` test can only ever discard
    what the layer already let through. The published weights ship
    ``confidence_threshold: 0.25``, and that number is why the cardboard box in the one
    Lite3 frame available is invisible to this pipeline: the network scores it 0.1221 and
    then deletes it. **The box is suppressed by a floor, not missed by a model.**

    That is the whole reason this route is cheap. The alternative already tried and
    measured to fail was colour: a brown box against a wooden background merges into one
    contour, and the workaround in production is a green panel taped to the prop. Nothing
    here needs new weights, a retrain, a second inference pass or a panel — it needs the
    network to stop throwing away a number it has already computed.

    Returns the text unchanged when ``floor`` is not below what is already there; the
    floor is never RAISED, because a caller asking for a permissive static tier must not
    be able to make the person tier blinder as a side effect.

    Raises ``ValueError`` unless exactly one threshold is present. A prototxt with two
    would leave one of them silently authoritative.
    """
    matches = _CONFIDENCE_FLOOR_RE.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one confidence_threshold in the prototxt, found "
            f"{len(matches)} — refusing to guess which one gates DetectionOutput")
    existing = float(matches[0][1])
    if floor >= existing:
        return text
    return _CONFIDENCE_FLOOR_RE.sub(lambda m: f"{m.group(1)}{floor:g}", text, count=1)


def range_detections(detections: list[Detection], camera: FisheyeCamera,
                     prior: SizePrior = PERSON_PRIOR) -> list[RangedDetection]:
    """Locate each detection in polar body coordinates, dropping unusable ones."""
    ranged = []
    for detection in detections:
        range_m, source = estimate_range(detection, camera, prior)
        if not math.isfinite(range_m):
            continue
        centre_x, centre_y = detection.centre
        bearing, _ = camera.bearing_elevation(centre_x, centre_y)
        ranged.append(RangedDetection(detection=detection, range_m=range_m,
                                      bearing_rad=float(bearing), source=source))
    return ranged


class PersonDetector:
    """MobileNet-SSD person detector over ``cv2.dnn`` on CPU.

    Args:
        model_dir: holds ``MobileNetSSD_deploy.prototxt`` and ``.caffemodel``.
        input_size: square network input. 300 is the trained size and most accurate;
            224 is ~1.7x faster on this Jetson at some cost in small-object recall.
            People close enough to matter are large in frame, so 300 is the default
            and the speed knob is there if the control loop needs it.
        confidence: minimum score to report. See :data:`DEFAULT_CONFIDENCE` for why
            the default leans lower than the usual 0.5.
        classes: which VOC labels count as obstacles.
        static_confidence: floor for the second, STATIC tier. ``None`` — the default —
            leaves the detector exactly as it was: one tier, the published prototxt
            loaded unmodified, ``detect_tiered`` returning an empty second list.
        static_classes: which VOC labels a static candidate may carry. ``person`` is
            rejected outright; see :data:`STATIC_CLASSES`.
    """

    def __init__(self, model_dir: str | Path, input_size: int = 300,
                 confidence: float = DEFAULT_CONFIDENCE,
                 classes: tuple[str, ...] = DYNAMIC_CLASSES,
                 static_confidence: float | None = None,
                 static_classes: tuple[str, ...] = STATIC_CLASSES) -> None:
        model_dir = Path(model_dir)
        prototxt = model_dir / "MobileNetSSD_deploy.prototxt"
        weights = model_dir / "MobileNetSSD_deploy.caffemodel"
        for path in (prototxt, weights):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found — fetch the MobileNet-SSD model into "
                    f"{model_dir} (see README.md)")
        self._input_size = int(input_size)
        self._confidence = float(confidence)
        self._classes = frozenset(classes)
        self._static_confidence = (None if static_confidence is None
                                   else float(static_confidence))
        self._static_classes = frozenset(static_classes)
        unknown = self._classes.union(self._static_classes).difference(VOC_CLASSES)
        if unknown:
            raise ValueError(f"not VOC classes: {sorted(unknown)}")
        if "person" in self._static_classes:
            # The static map plans every landmark with `person_shaped=False`, so a person
            # routed there is a person the robot will never hold for. `static_shaped`
            # refuses person-SHAPED boxes, but a label that says `person` outright is a
            # configuration error, not a shape question, and must fail at construction
            # rather than being silently filtered later.
            raise ValueError(
                "'person' cannot be a static class: a landmark is planned with "
                "person_shaped=False and would never hold the robot")
        self._net = self._load_net(prototxt, weights)

    @property
    def static_tier(self) -> bool:
        """Whether a second, sub-threshold tier is enabled. Off unless asked for."""
        return self._static_confidence is not None

    def _load_net(self, prototxt: Path, weights: Path):
        """Load the net, lowering the prototxt's own score floor if a static tier is on.

        The patched prototxt is a TEMPORARY FILE, written and unlinked inside this call.
        ``cv2.dnn.readNetFromCaffe`` takes paths rather than text but has fully parsed the
        file by the time it returns, so nothing outlives the load. It matters that the
        published file is not edited in place: the model directory is shared with the GOAL
        detector, which constructs its own :class:`PersonDetector` at a different
        threshold, and a rewrite on disk would silently retune that one too — and would
        persist into every later run from the same machine.
        """
        if not self.static_tier:
            return cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
        patched = prototxt_with_floor(prototxt.read_text(), self._static_confidence)
        descriptor, path = tempfile.mkstemp(suffix=".prototxt")
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(patched)
            return cv2.dnn.readNetFromCaffe(path, str(weights))
        finally:
            # Best effort: a leaked temp file is untidy, but raising here would replace a
            # working detector with a crash on the way out of a successful load.
            with contextlib.suppress(OSError):
                os.unlink(path)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Mover detections in ``image``, in that image's own pixel coordinates.

        The network squashes its input to a square, which is what MobileNet-SSD was
        trained on, so the aspect distortion is expected rather than a bug — and the
        normalised boxes it returns map straight back onto the original frame.
        """
        return self.detect_tiered(image)[0]

    def detect_tiered(self, image: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        """``(movers, static_candidates)`` from ONE forward pass.

        The second list is empty unless a ``static_confidence`` was given, so the default
        construction of this class behaves exactly as it did before the tier existed.

        ONE PASS, NOT TWO, and that is the point of putting the tier here rather than in a
        second detector. Inference is 131 ms at 300x300 on this robot's four Cortex-A78
        cores with no CUDA — it is the entire perception budget, and the control loop
        already consumes a belief that is ~160 ms old. A second pass would halve the
        perception rate to buy detections the first pass has ALREADY COMPUTED and thrown
        away. The two lists are two readings of the same ``forward()`` output.

        The tiers cannot overlap: ``person`` may not be a static class (rejected in
        ``__init__``) and the mover classes ship as ``("person",)``, so no detection can
        be both a tracked mover and a mapped landmark. If a caller widens ``classes``
        into the furniture set they get both, and that is their business — the shape
        gates still apply independently.
        """
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, _SSD_SCALE,
                                     (self._input_size, self._input_size), _SSD_MEAN)
        self._net.setInput(blob)
        raw = self._net.forward()

        floor = (self._confidence if not self.static_tier
                 else min(self._confidence, self._static_confidence))
        movers: list[Detection] = []
        statics: list[Detection] = []
        for _, class_id, score, x1, y1, x2, y2 in raw[0, 0]:
            if score < floor:
                continue
            label = VOC_CLASSES[int(class_id)] if int(class_id) < len(VOC_CLASSES) else "?"
            is_mover = score >= self._confidence and label in self._classes
            is_static = (self.static_tier and score >= self._static_confidence
                         and label in self._static_classes)
            if not is_mover and not is_static:
                continue
            # SSD box regression can land slightly outside the image. Clamp before
            # the model sees these: the fisheye projection extrapolated past the
            # frame edge would report an angle the lens never imaged, inflating the
            # subtended span and so under-reading the range.
            detection = Detection(
                x1=min(max(float(x1) * width, 0.0), width - 1.0),
                y1=min(max(float(y1) * height, 0.0), height - 1.0),
                x2=min(max(float(x2) * width, 0.0), width - 1.0),
                y2=min(max(float(y2) * height, 0.0), height - 1.0),
                score=float(score), label=label)
            if is_mover:
                movers.append(detection)
            # Clamped FIRST, then shape-gated. `static_shaped` asks whether the box
            # touches a frame edge, and an unclamped box that ran off the image would
            # answer "no" on coordinates the lens never imaged — losing exactly the
            # vertical-clip refusal that keeps close-range people out of the map.
            if is_static and static_shaped(detection, width, height):
                statics.append(detection)
        return movers, statics
