# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Where the robot is trying to get to — an RGB beacon, a detected object, or a waypoint.

Three interchangeable sources behind one interface:

  * :class:`ArucoGoal` — a printed marker the camera fixes on. Its known side length
    turns into a range through the same angular-size maths the person ranger uses, so
    the goal is a real metric point rather than "somewhere over there".
  * :class:`DetectedObjectGoal` — a piece of ordinary furniture the object detector
    recognises, e.g. a chair. No prop to print, but it needs a measured height and it
    only works for the twenty classes the detector knows.
  * :class:`OdomWaypoint` — an offset from wherever the robot started, dead-reckoned
    from ``rt/sportmodestate``. Needs no props, drifts, cannot self-correct.

THE GOAL IS LATCHED IN ODOM, and that is the important part. The camera sees ~120°, so
the moment the robot swerves around a person the marker leaves the frame — and a
controller that steers straight at a live pixel measurement would lose the goal exactly
when it is manoeuvring. Each sighting instead writes a point into the odom frame and
the planner drives at *that*; a sighting refreshes it, and losing sight of it costs
nothing but the slow accumulation of odometry drift. It also means the robot can round
a person and re-acquire, rather than aborting.

RANGING A SQUARE VIEWED OFF-AXIS. A marker rotated away from the camera foreshortens,
which would read as "further away". The two edge pairs are measured separately and the
LONGER pair wins: rotate a square about its vertical axis and the vertical edges keep
their true length while the horizontal ones shrink, so the longer pair is always the
one closest to unforeshortened.

OpenCV renamed the whole ArUco API at 4.7. This robot runs 4.2, so the 4.2 spelling is
primary and :func:`aruco_detector` shims the modern one for anyone running this
off-robot.
"""

from __future__ import annotations

import abc
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from camera_model import FisheyeCamera
from person_detector import Detection, SizePrior, estimate_range

DEFAULT_DICTIONARY = "DICT_4X4_50"
DEFAULT_MARKER_ID = 0
# Side length of the printed marker's black square, metres. A 4x4 marker at 0.20 m is
# detectable to roughly 6 m on this 1080p sensor and fits on one sheet of A4.
DEFAULT_MARKER_SIZE_M = 0.20


def aruco_detector(dictionary_name: str):
    """Return ``detect(gray) -> (corners, ids)`` for either OpenCV ArUco API.

    4.2 has ``Dictionary_get`` / ``DetectorParameters_create`` / ``detectMarkers``;
    4.7+ replaced them with ``getPredefinedDictionary`` / ``DetectorParameters`` /
    ``ArucoDetector``. The robot is on 4.2.
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("this OpenCV build has no aruco module (needs opencv-contrib)")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary {dictionary_name!r}")

    if hasattr(cv2.aruco, "Dictionary_get"):        # OpenCV <= 4.6
        dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        params = cv2.aruco.DetectorParameters_create()

        def detect(gray):
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
            return corners, ids
        return detect

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)  # OpenCV >= 4.7
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    def detect(gray):
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids
    return detect


def write_marker(path: str, marker_id: int = DEFAULT_MARKER_ID,
                 dictionary_name: str = DEFAULT_DICTIONARY, pixels: int = 800) -> None:
    """Render a printable marker PNG. Print it so the BLACK SQUARE measures the
    ``--marker-size`` handed to the navigator — the range scales linearly with it."""
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    if hasattr(cv2.aruco, "Dictionary_get"):
        dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        image = cv2.aruco.drawMarker(dictionary, marker_id, pixels)
    else:
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)
    # A quiet zone is required for detection; the printed sheet's white margin usually
    # supplies it, but bordering the PNG means a full-bleed print still works.
    border = pixels // 8
    image = cv2.copyMakeBorder(image, border, border, border, border,
                               cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, image)


@dataclass(frozen=True)
class GoalFix:
    """One observation of the goal, in both robot-relative and odom terms."""

    bearing_rad: float
    range_m: float
    x: float
    y: float


class GoalSource(abc.ABC):
    """A goal the navigator can drive at, in odom coordinates."""

    @abc.abstractmethod
    def update(self, image: np.ndarray | None,
               pose: tuple[float, float, float]) -> GoalFix | None:
        """Fold in one frame. Returns a fix if the goal was observed, else ``None``."""

    @abc.abstractmethod
    def goal_xy(self) -> tuple[float, float] | None:
        """Latched goal in odom, or ``None`` if never acquired."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable identity, for logs and the overlay."""


class ArucoGoal(GoalSource):
    """Drive at a printed ArUco marker, latching each sighting into odom.

    Args:
        camera: model expressed in the SAME pixel coordinates as the images passed to
            :meth:`update`.
        marker_size_m: side of the printed black square.
        detect_scale: images are downscaled by this before detection. Marker detection
            on a 1080p frame costs ~29 ms, which competes with the person detector for
            the same four cores; halving the image cuts that ~4x and still resolves a
            0.20 m marker at several metres.
    """

    def __init__(self, camera: FisheyeCamera, marker_id: int = DEFAULT_MARKER_ID,
                 marker_size_m: float = DEFAULT_MARKER_SIZE_M,
                 dictionary_name: str = DEFAULT_DICTIONARY,
                 detect_scale: float = 0.5) -> None:
        if marker_size_m <= 0.0:
            raise ValueError(f"marker_size_m must be positive, got {marker_size_m}")
        self._camera = camera
        self._marker_id = marker_id
        self._marker_size_m = marker_size_m
        self._detect = aruco_detector(dictionary_name)
        self._detect_scale = detect_scale
        self._goal: tuple[float, float] | None = None
        self._sightings = 0

    @property
    def description(self) -> str:
        return (f"ArUco id={self._marker_id} ({self._marker_size_m * 100:.0f} cm), "
                f"{self._sightings} sightings")

    # NOTE: there is deliberately no `last_fix` accessor. One existed, documented "for
    # the overlay", and nothing ever read it — the overlay is handed the CURRENT frame's
    # fix so its "GOAL 2.3m @ +12deg" label marks an actual sighting, while the
    # persistent latched goal is drawn in the plan-view inset from `goal_xy()`. Holding
    # the last fix and drawing it every frame would render a stale bearing as if it were
    # live, which is worse than the label flickering.

    def update(self, image: np.ndarray | None,
               pose: tuple[float, float, float]) -> GoalFix | None:
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._detect_scale != 1.0:
            gray = cv2.resize(gray, None, fx=self._detect_scale, fy=self._detect_scale,
                              interpolation=cv2.INTER_AREA)
        corners, ids = self._detect(gray)
        if ids is None:
            return None

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) != self._marker_id:
                continue
            # Back to full-resolution pixels so the camera model applies.
            points = marker_corners.reshape(4, 2) / self._detect_scale
            fix = self._fix_from_corners(points, pose)
            if not math.isfinite(fix.range_m):
                # A degenerate span carries no usable range, and latching it puts the
                # goal at infinity: the planner's goal cost becomes inf/inf = NaN, so
                # argmin picks an arbitrary command while still reporting reason=goal,
                # and `distance <= arrive_tolerance` can never be true again. Same rule
                # the person ranger applies (person_detector.range_detections).
                continue
            self._goal = (fix.x, fix.y)
            self._sightings += 1
            return fix
        return None

    def _fix_from_corners(self, points: np.ndarray,
                          pose: tuple[float, float, float]) -> GoalFix:
        # detectMarkers returns corners clockwise from top-left, so edges are
        # (0,1) & (2,3) — one opposite pair — and (1,2) & (3,0) — the other.
        pair_a = 0.5 * (self._camera.angle_between(points[0], points[1])
                        + self._camera.angle_between(points[2], points[3]))
        pair_b = 0.5 * (self._camera.angle_between(points[1], points[2])
                        + self._camera.angle_between(points[3], points[0]))
        subtended = float(max(pair_a, pair_b))   # the less-foreshortened pair
        # Sub-pixel spans are unmeasurable, and gating on a float epsilon instead would
        # let a zero-span sighting through as a finite range of millions of metres —
        # see FisheyeCamera.range_from_span, whose floor this shares.
        range_m = (math.inf if subtended <= self._camera.pixel_angle_rad
                   else self._marker_size_m / (2.0 * math.tan(subtended / 2.0)))

        centre = points.mean(axis=0)
        bearing, _ = self._camera.bearing_elevation(centre[0], centre[1])
        bearing = float(bearing)

        robot_x, robot_y, robot_yaw = pose
        world_bearing = robot_yaw + bearing
        return GoalFix(bearing_rad=bearing, range_m=range_m,
                       x=robot_x + range_m * math.cos(world_bearing),
                       y=robot_y + range_m * math.sin(world_bearing))

    def goal_xy(self) -> tuple[float, float] | None:
        return self._goal


#: Default fraction of the frame the goal detector looks at, per axis. See
#: :class:`DetectedObjectGoal` for why cropping is what makes this work at all.
DEFAULT_GOAL_CROP = 0.5

#: Default seconds between goal-detection passes once the goal has been acquired.
#: Sized from the measured cost, not picked round: the goal pass is 69 ms on this
#: Jetson against 124 ms for everything else in a cycle, so refreshing at 1 Hz raised
#: the mean perception age enough to trip the navigator's 0.6 s staleness guard several
#: times a run. A latched goal only needs re-measuring as fast as the odometry under it
#: drifts, which over three seconds of walking is centimetres.
DEFAULT_GOAL_REFRESH_S = 3.0

#: Default detector input size for the goal pass. Smaller than the obstacle pass's 300
#: on purpose — the crop has already multiplied the target's effective resolution, so
#: 224 on a half crop resolves the goal better than 300 on the full frame ever did, at
#: 69 ms instead of 130. Measured on the staged chair: 0.82 at 224 against 0.93 at 300,
#: both far above the 0.5 latch threshold, and NOTHING AT ALL on the uncropped frame.
DEFAULT_GOAL_INPUT_SIZE = 224

#: Widest the goal crop is ever allowed to open to. NOT 1.0, and the difference is the
#: whole reason a ceiling exists: on the staged chair the uncropped frame scored **nothing
#: at all** where a crop scored 0.82-0.93, so "no crop" is not a degraded goal pass, it is
#: no goal pass. The crop widens toward this and stops.
MAX_GOAL_CROP = 0.9

#: How much taller than the target the crop must be. The crop's job at range is effective
#: resolution; its job up close is simply not to cut the target in half. 1.4 leaves 40%
#: headroom for the box, the height prior being a few per cent off, and the goal sitting
#: off-centre.
GOAL_CROP_HEADROOM = 1.4


class DetectedObjectGoal(GoalSource):
    """Drive at the best-scoring instance of a detector class — a chair, say — via a crop.

    BEST-SCORING, NOT NEAREST, and the distinction matters with two chairs in shot: the
    robot will latch a confident chair 4 m away over a doubtful one at 2 m, and plan a
    path straight through the near one. That is the right trade for a *goal* — a
    mis-latched goal walks the robot somewhere it was never asked to go, while the near
    chair is an obstacle problem, and obstacle classes are the planner's job — but it is
    a real behaviour, not an accident. Score also re-decides on every refresh pass, so
    with two comparable candidates the goal can hop between them; stage one goal object,
    or raise ``min_score`` until only one clears it.

    CROPPING IS THE WHOLE TRICK, and it is worth being precise about why, because the
    obvious diagnosis is wrong. MobileNet-SSD squashes its input to 300x300 whatever
    came in, so a chair 240 px wide in a 1920-wide frame is **37 px** to the network,
    and small objects are exactly what SSD gives up on first. Run on the staged scene at
    full frame with all twenty classes and confidence dropped to 0.15, ``chair`` does
    not fire once. Run the identical detector on a half-size centre crop of the same
    frames and it fires on **8 of 8 at 0.98**. The chair was never too dark, too far, or
    too occluded — the bin in front of it changes nothing. It was too few pixels.

    The cost is a second inference at ~131 ms, which the 10 Hz control loop cannot pay
    every cycle. It does not have to: a goal is *latched in odom* (see the module
    docstring), so a chair that has not moved needs re-measuring about as often as the
    odometry drifts. :data:`DEFAULT_GOAL_REFRESH_S` throttles the pass once a goal is
    held, while an unacquired goal is searched for on every frame — the run cannot start
    until it is found, so that is where the budget belongs.

    The crop is a TRANSLATION, not a rescale: boxes are shifted back by the crop origin
    and the full-resolution camera model then applies unchanged, with no second focal
    length to keep in step.

    Args:
        camera: model in the same pixel coordinates as the images passed to
            :meth:`update` — i.e. the full frame, not the crop.
        detector: anything with ``detect(image) -> list[Detection]``, already filtered
            to the goal class. A separate instance from the obstacle detector, because
            the two want different classes and different confidences.
        label: which detector class is the goal.
        height_m: the object's real height, floor to top. The range scales linearly on
            it, so it is a tape-measure number.
        width_m: the object's real width, used only once the box is cut off by the
            frame edge. Optional because it is a fallback, but worth supplying: left to
            ``SizePrior.of_height`` it is inferred from a PERSON's aspect ratio, which
            for a 1.07 m chair claims 0.31 m across instead of ~0.62 m and halves the
            range at close quarters. That is the safe direction for a goal — the robot
            stops short rather than walking into it — but it is still wrong.
        crop: fraction of each axis to keep, centred. 1.0 disables cropping.
        refresh_s: minimum seconds between passes once a goal is held.
        min_score: detections below this are ignored. Higher than the obstacle
            threshold on purpose — the two errors are not symmetric. Missing a goal for
            one frame costs a frame; latching a false one walks the robot at a wall.
        clock: seconds source, injected so the throttle is testable without sleeping.
    """

    def __init__(self, camera: FisheyeCamera, detector, label: str = "chair",
                 height_m: float = 1.0, width_m: float | None = None,
                 crop: float = DEFAULT_GOAL_CROP,
                 refresh_s: float = DEFAULT_GOAL_REFRESH_S, min_score: float = 0.5,
                 clock=None) -> None:
        if height_m <= 0.0:
            raise ValueError(f"height_m must be positive, got {height_m}")
        if width_m is not None and width_m <= 0.0:
            raise ValueError(f"width_m must be positive, got {width_m}")
        if not 0.0 < crop <= 1.0:
            raise ValueError(f"crop must be in (0, 1], got {crop}")
        self._camera = camera
        self._detector = detector
        self._label = label
        self._prior = (SizePrior.of_height(height_m) if width_m is None
                       else SizePrior(height_m=height_m, width_m=width_m))
        self._crop = crop
        self._refresh_s = refresh_s
        self._min_score = min_score
        self._clock = clock if clock is not None else time.monotonic
        self._goal: tuple[float, float] | None = None
        self._sightings = 0
        self._passes = 0
        self._last_pass: float | None = None
        #: The crop the last pass actually used. Recorded because it is no longer the
        #: value the caller passed — it widens with range — and a goal that jumps is
        #: otherwise impossible to tell from a crop that moved under it.
        self.last_crop = crop

    @property
    def description(self) -> str:
        return (f"detected {self._label} ({self._prior.height_m:.3f} m tall), "
                f"{self._sightings} sightings in {self._passes} passes")

    @property
    def sightings(self) -> int:
        return self._sightings

    def goal_xy(self) -> tuple[float, float] | None:
        return self._goal

    def _crop_for(self, pose: tuple, frame_height: int) -> float:
        """The crop to use this pass, widened if the goal has grown too big for it.

        A FIXED crop is wrong at both ends of an approach and only one end is obvious.
        Far away the target is a handful of pixels and the crop is what makes it
        detectable at all. Up close the target outgrows the window and the crop CUTS IT
        IN HALF — measured on the office chair at ~1.9 m with the 0.5 default: the crop
        clips the backrest, every pass is dropped, and the robot never stands up, with a
        goal in plain view and no log line saying why.

        So the crop is a floor, not a value. It stays at whatever the caller asked for
        while the goal is small, and opens only as far as it must to keep the target
        inside it with :data:`GOAL_CROP_HEADROOM` to spare, never past
        :data:`MAX_GOAL_CROP`.

        Returns the caller's crop unchanged until a goal is latched, because the range
        this reasons about is the latched goal's own. Before that, acquisition is the
        only job and the full-frame retry in :meth:`update` already covers the case where
        the crop is what is hiding the target.
        """
        if self._goal is None:
            return self._crop
        range_m = math.hypot(self._goal[0] - pose[0], self._goal[1] - pose[1])
        if not math.isfinite(range_m) or range_m <= 0.0:
            return self._crop
        target_px = self._prior.height_m * self._camera.focal_px / range_m
        needed = target_px * GOAL_CROP_HEADROOM / frame_height
        return min(MAX_GOAL_CROP, max(self._crop, needed))

    def _due(self, now: float) -> bool:
        """Whether to spend an inference this frame."""
        if self._goal is None or self._last_pass is None:
            return True          # not acquired yet: the run is blocked until it is
        return now - self._last_pass >= self._refresh_s

    def update(self, image: np.ndarray | None,
               pose: tuple[float, float, float]) -> GoalFix | None:
        if image is None:
            return None
        now = self._clock()
        if not self._due(now):
            return None
        self._last_pass = now
        self._passes += 1

        height, width = image.shape[:2]
        crop = self._crop_for(pose, height)
        self.last_crop = crop
        crop_w, crop_h = int(width * crop), int(height * crop)
        x0, y0 = (width - crop_w) // 2, (height - crop_h) // 2

        best = self._best_in(image[y0:y0 + crop_h, x0:x0 + crop_w], crop_w, crop_h)
        if best is None and self._goal is None and self._crop < 1.0:
            # NOTHING IS LATCHED YET, so the edge guard's own bargain does not apply.
            # It trades a skipped frame for a range it can trust, and that is right once
            # a goal is held. Before the first fix there is nothing to fall back on:
            # `_due()` returns True forever until acquisition, so a "skipped" frame does
            # not cost a frame, it costs the RUN. Measured on the office chair at ~1.9 m:
            # the crop clips the backrest, every pass is dropped, and the robot never
            # stands up — a goal in plain view, 0 sightings in 6 passes, and no log line
            # saying why.
            #
            # The full frame is the correct retry rather than a looser guard: a box
            # against the REAL frame edge is what `Detection.clipped` is for, and
            # `estimate_range` already falls back to the width prior for it. So the
            # range stays honest; only the window changes.
            x0 = y0 = 0
            best = self._best_in(image, width, height)
        if best is None:
            return None

        full_frame = Detection(x1=best.x1 + x0, y1=best.y1 + y0,
                               x2=best.x2 + x0, y2=best.y2 + y0,
                               score=best.score, label=best.label)
        range_m, _ = estimate_range(full_frame, self._camera, self._prior)
        if not math.isfinite(range_m) or range_m <= 0.0:
            # Same rule as ArucoGoal: an unusable range latched as a goal puts it at
            # infinity, the planner's goal cost becomes inf/inf = NaN, and `distance <=
            # arrive_tolerance` can never be true again.
            return None

        centre_x, centre_y = full_frame.centre
        bearing, _ = self._camera.bearing_elevation(centre_x, centre_y)
        bearing = float(bearing)
        robot_x, robot_y, robot_yaw = pose
        world_bearing = robot_yaw + bearing
        self._goal = (robot_x + range_m * math.cos(world_bearing),
                      robot_y + range_m * math.sin(world_bearing))
        self._sightings += 1
        return GoalFix(bearing_rad=bearing, range_m=range_m,
                       x=self._goal[0], y=self._goal[1])

    def _best_in(self, window: np.ndarray, width: int,
                 height: int) -> Detection | None:
        """Highest-scoring unclipped detection of the goal label inside ``window``.

        ``width``/``height`` are the WINDOW's, not the frame's — the edge test has to be
        against the boundary the box was actually cut by.
        """
        best: Detection | None = None
        for detection in self._detector.detect(window):
            if detection.label != self._label or detection.score < self._min_score:
                continue
            # A box touching the window's edge is cut off by it, and a truncated box is
            # short, which reads as FAR. Prefer a whole box; see `update` for what
            # happens when there isn't one.
            if self._touches_edge(detection, width, height):
                continue
            if best is None or detection.score > best.score:
                best = detection
        return best

    @staticmethod
    def _touches_edge(detection: Detection, width: int, height: int,
                      margin_px: float = 2.0) -> bool:
        return (detection.x1 <= margin_px or detection.y1 <= margin_px
                or detection.x2 >= width - margin_px
                or detection.y2 >= height - margin_px)


class OdomWaypoint(GoalSource):
    """Drive to a fixed offset from the pose at construction, dead-reckoned.

    The offset is interpreted in the robot's BODY frame at construction time, so
    ``forward_m=3`` means three metres along whatever way the robot was facing when
    the run started — not three metres along the odom x-axis.
    """

    def __init__(self, start_pose: tuple[float, float, float],
                 forward_m: float, lateral_m: float = 0.0) -> None:
        x, y, yaw = start_pose
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        self._goal = (x + forward_m * cos_y - lateral_m * sin_y,
                      y + forward_m * sin_y + lateral_m * cos_y)
        self._offset = (forward_m, lateral_m)

    @property
    def description(self) -> str:
        return (f"odom waypoint {self._offset[0]:+.2f} m fwd, "
                f"{self._offset[1]:+.2f} m left of start")

    def update(self, image: np.ndarray | None,
               pose: tuple[float, float, float]) -> GoalFix | None:
        """A dead-reckoned waypoint is never *observed*, so this always returns None.

        The method exists so the navigator can hold either source without caring
        which; there is nothing for a frame to contribute here.
        """
        return None

    def goal_xy(self) -> tuple[float, float] | None:
        return self._goal


def main() -> None:
    """Write a printable marker PNG."""
    import argparse

    ap = argparse.ArgumentParser(description="Generate a printable ArUco goal marker.")
    ap.add_argument("--out", default="goal_marker.png", help="output PNG path")
    ap.add_argument("--id", type=int, default=DEFAULT_MARKER_ID, help="marker id")
    ap.add_argument("--dictionary", default=DEFAULT_DICTIONARY, help="ArUco dictionary")
    ap.add_argument("--pixels", type=int, default=800, help="marker size in pixels")
    args = ap.parse_args()

    write_marker(args.out, marker_id=args.id, dictionary_name=args.dictionary,
                 pixels=args.pixels)
    print(f"wrote {args.out} (id={args.id}, {args.dictionary})")
    print("Print it, MEASURE the black square, and pass that to --marker-size.")


if __name__ == "__main__":
    main()
