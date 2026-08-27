#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the ArUco goal source — synthetic corners, no camera.

The marker detector itself is OpenCV's and not ours to test. What is ours is what
happens to the corners afterwards: the foreshortening rule that picks an edge pair,
the undoing of ``detect_scale``, the latch into odom — and the refusal to latch a
sighting that carries no usable range.

That last one is the reason this file exists. An infinite range propagates silently:
it makes the planner's goal cost ``inf/inf`` = NaN, so ``argmin`` returns an arbitrary
command that still reports ``reason=goal``, and ``distance <= arrive_tolerance`` can
never be true again — a run that steers on nothing and can only end on the clock.

Run: ``python3 test_goal.py``
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_model import FisheyeCamera
from goal import MAX_GOAL_CROP, ArucoGoal, DetectedObjectGoal, OdomWaypoint
from person_detector import Detection

WIDTH, HEIGHT = 1920, 1080
CAMERA = FisheyeCamera.from_hfov(WIDTH, HEIGHT, 85.27)
BLANK = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
MARKER_M = 0.20
ORIGIN = (0.0, 0.0, 0.0)


def _square_corners(distance_m: float, marker_m: float = MARKER_M) -> np.ndarray:
    """Image corners of a marker facing the camera square-on at ``distance_m``.

    Clockwise from top-left, which is the order ``cv2.aruco.detectMarkers`` returns
    and the order the edge-pair rule in ``_fix_from_corners`` assumes.
    """
    half = marker_m / 2.0
    return np.array([CAMERA.project([distance_m, +half, +half]),    # top-left
                     CAMERA.project([distance_m, -half, +half]),    # top-right
                     CAMERA.project([distance_m, -half, -half]),    # bottom-right
                     CAMERA.project([distance_m, +half, -half])])   # bottom-left


def _goal_seeing(corners: np.ndarray, marker_id: int = 0,
                 detect_scale: float = 1.0) -> ArucoGoal:
    """An ArucoGoal whose detector always reports ``corners`` as ``marker_id``."""
    goal = ArucoGoal(CAMERA, marker_id=marker_id, marker_size_m=MARKER_M,
                     detect_scale=detect_scale)
    goal._detect = lambda gray: ([corners.reshape(1, 4, 2)], np.array([[marker_id]]))
    return goal


def test_a_square_marker_ranges_to_its_true_distance():
    goal = _goal_seeing(_square_corners(3.0))
    fix = goal.update(BLANK, ORIGIN)
    assert fix is not None
    assert abs(fix.range_m - 3.0) < 0.03, fix.range_m
    assert abs(fix.bearing_rad) < math.radians(0.5), fix.bearing_rad


def test_the_goal_is_latched_into_odom():
    """A sighting is stored as a world point, not as a live bearing."""
    goal = _goal_seeing(_square_corners(3.0))
    goal.update(BLANK, (1.0, 2.0, math.pi / 2.0))     # robot at (1,2) facing +y
    x, y = goal.goal_xy()
    assert abs(x - 1.0) < 0.05 and abs(y - 5.0) < 0.05, (x, y)


def test_detect_scale_is_undone_before_the_camera_model_is_applied():
    """Halved-resolution corners must range the same as full-resolution ones."""
    full = _goal_seeing(_square_corners(3.0), detect_scale=1.0)
    half = _goal_seeing(_square_corners(3.0) * 0.5, detect_scale=0.5)
    assert abs(full.update(BLANK, ORIGIN).range_m
               - half.update(BLANK, ORIGIN).range_m) < 0.02


def test_a_degenerate_marker_does_not_latch_an_infinite_goal():
    """The regression: a zero-span sighting must be discarded, not latched."""
    goal = _goal_seeing(np.full((4, 2), 500.0))       # all four corners coincident
    fix = goal.update(BLANK, ORIGIN)
    assert fix is None, fix
    assert goal.goal_xy() is None, goal.goal_xy()


def test_a_degenerate_sighting_does_not_discard_a_good_one():
    """Losing the range on one frame must not unlatch the goal already held."""
    good = _square_corners(3.0)
    goal = _goal_seeing(good)
    goal.update(BLANK, ORIGIN)
    latched = goal.goal_xy()

    goal._detect = lambda gray: ([np.full((1, 4, 2), 500.0)], np.array([[0]]))
    assert goal.update(BLANK, ORIGIN) is None
    assert goal.goal_xy() == latched


def test_a_different_marker_id_is_ignored():
    goal = _goal_seeing(_square_corners(3.0), marker_id=7)
    goal._marker_id = 0                               # now looking for a different id
    assert goal.update(BLANK, ORIGIN) is None
    assert goal.goal_xy() is None


def test_waypoint_offset_is_in_the_body_frame_at_construction():
    """`--waypoint 3 0` means 3 m along the way the robot was facing, not along +x."""
    waypoint = OdomWaypoint((1.0, 2.0, math.pi / 2.0), 3.0, 0.0)
    x, y = waypoint.goal_xy()
    assert abs(x - 1.0) < 1e-9 and abs(y - 5.0) < 1e-9, (x, y)
    assert waypoint.update(BLANK, ORIGIN) is None     # never observed, by design


# ── DetectedObjectGoal ──────────────────────────────────────────────────────
class _ScriptedDetector:
    """Returns canned boxes in the CROP's pixel coordinates, and counts its passes.

    Standing in for MobileNet-SSD so the latch, the crop arithmetic and the throttle
    can be tested without a 23 MB caffemodel or a 131 ms inference.
    """

    def __init__(self, boxes):
        self._boxes = list(boxes)
        self.passes = 0

    def detect(self, image):
        self.passes += 1
        return self._boxes


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


#: The staged chair as the detector actually saw it: a 2x centre crop of a 1920x1080
#: frame puts the origin at (480, 270), and the measured full-frame box
#: (677,303)-(923,730) is therefore (197,33)-(443,460) inside it. 1.0668 m of chair over
#: 427 px is 3.22 m.
#:
#: Note how little headroom y1=33 leaves. At the default crop the staged chair clears
#: the top edge by 33 px of 540, so a goal any taller, nearer, or higher in frame would
#: be refused by the crop-edge guard and the run would never acquire it. That is the
#: knob --goal-crop exists for.
CHAIR_IN_CROP = Detection(x1=197.0, y1=33.0, x2=443.0, y2=460.0, score=0.98,
                          label="chair")
CHAIR_HEIGHT_M = 1.0668


def _chair_goal(detector=None, **kwargs):
    kwargs.setdefault("height_m", CHAIR_HEIGHT_M)
    kwargs.setdefault("clock", _Clock())
    return DetectedObjectGoal(CAMERA, detector or _ScriptedDetector([CHAIR_IN_CROP]),
                              label="chair", **kwargs)


def test_a_detected_chair_ranges_to_its_measured_distance():
    """Pins the crop arithmetic against numbers taken off the robot's own footage.

    A crop is a TRANSLATION, so adding the origin back lets the full-resolution camera
    model apply unchanged. Get that wrong and the range is still plausible, just wrong.
    """
    goal = _chair_goal()
    fix = goal.update(BLANK, ORIGIN)
    assert fix is not None
    assert abs(fix.range_m - 3.22) < 0.05, fix.range_m
    assert abs(math.degrees(fix.bearing_rad) - 7.1) < 0.3, fix.bearing_rad


def test_a_detected_goal_is_latched_into_odom():
    goal = _chair_goal()
    goal.update(BLANK, (1.0, 2.0, math.radians(90.0)))
    latched = goal.goal_xy()
    # 3.22 m at 7.1 deg left of a robot facing +y, from (1, 2).
    assert abs(latched[0] - (1.0 - 3.22 * math.sin(math.radians(7.1)))) < 0.05
    assert abs(latched[1] - (2.0 + 3.22 * math.cos(math.radians(7.1)))) < 0.05


def test_a_box_touching_the_crop_edge_is_refused():
    """Crop-edge truncation is invisible to Detection.clipped, which measures against
    the FULL frame — and a short box reads FAR, which walks the robot past its goal."""
    edge = Detection(x1=0.0, y1=33.0, x2=443.0, y2=460.0, score=0.98, label="chair")
    goal = _chair_goal(_ScriptedDetector([edge]))
    assert goal.update(BLANK, ORIGIN) is None
    assert goal.goal_xy() is None


class _WindowAwareDetector:
    """A goal that is whole in the full frame and clipped by the 0.5 crop.

    Not a contrivance — it is simply what a NEAR object does. The crop exists to give a
    DISTANT goal more pixels, and the very same crop cuts the top off a close one.
    """

    def __init__(self, in_crop: Detection, in_full: Detection) -> None:
        self._in_crop, self._in_full = in_crop, in_full
        self.windows: list[tuple[int, int]] = []

    def detect(self, image):
        height, width = image.shape[:2]
        self.windows.append((width, height))
        return [self._in_full] if (width, height) == (WIDTH, HEIGHT) else [self._in_crop]


def test_an_unlatched_goal_falls_back_to_the_full_frame_when_the_crop_clips_it():
    """The bug that stopped a live run with the chair filling the frame: 0 sightings in
    6 passes, the robot never stood up, and nothing said why.

    The edge guard trades a skipped frame for a range it can trust. That bargain is
    right once a goal is latched and there is something to fall back on. Before the
    first fix there is nothing — ``_due()`` keeps returning True until acquisition — so
    a "skipped" frame does not cost a frame, it costs the run.
    """
    clipped = Detection(x1=197.0, y1=0.0, x2=443.0, y2=460.0, score=0.98, label="chair")
    whole = Detection(x1=677.0, y1=303.0, x2=923.0, y2=763.0, score=0.98, label="chair")
    detector = _WindowAwareDetector(clipped, whole)
    goal = _chair_goal(detector)

    fix = goal.update(BLANK, ORIGIN)
    assert fix is not None, "a goal in plain view must not be dropped before acquisition"
    assert goal.goal_xy() is not None
    assert detector.windows == [(WIDTH // 2, HEIGHT // 2), (WIDTH, HEIGHT)], \
        "the crop is tried first and the FULL frame is the retry, in that order"


def test_a_latched_goal_still_refuses_a_crop_clipped_box():
    """The fallback must not weaken the guard once there IS something to protect. Here
    a truncated box would read as far and walk the robot past its target, and the held
    goal makes skipping the frame genuinely free — which is the case the guard is for.
    """
    clipped = Detection(x1=197.0, y1=0.0, x2=443.0, y2=460.0, score=0.98, label="chair")
    whole = Detection(x1=677.0, y1=303.0, x2=923.0, y2=763.0, score=0.98, label="chair")
    goal = _chair_goal(_ScriptedDetector([CHAIR_IN_CROP]))
    assert goal.update(BLANK, ORIGIN) is not None, "latch a goal first"
    latched = goal.goal_xy()

    goal._detector = _WindowAwareDetector(clipped, whole)
    goal._last_pass = None                      # let the throttle allow another pass
    assert goal.update(BLANK, ORIGIN) is None, "a latched goal keeps the strict guard"
    assert goal.goal_xy() == latched, "and keeps the fix it already had"


def test_a_low_scoring_detection_is_refused():
    weak = Detection(x1=197.0, y1=33.0, x2=443.0, y2=460.0, score=0.2, label="chair")
    goal = _chair_goal(_ScriptedDetector([weak]), min_score=0.5)
    assert goal.update(BLANK, ORIGIN) is None


def test_another_class_is_ignored():
    other = Detection(x1=197.0, y1=33.0, x2=443.0, y2=460.0, score=0.98, label="sofa")
    assert _chair_goal(_ScriptedDetector([other])).update(BLANK, ORIGIN) is None


def test_the_highest_scoring_candidate_wins():
    near = Detection(x1=100.0, y1=100.0, x2=200.0, y2=300.0, score=0.6, label="chair")
    goal = _chair_goal(_ScriptedDetector([near, CHAIR_IN_CROP]))
    fix = goal.update(BLANK, ORIGIN)
    assert abs(fix.range_m - 3.22) < 0.05, fix.range_m


def test_an_unacquired_goal_is_searched_for_every_frame():
    """The run is blocked until the goal is found, so that is where the budget goes.

    TWO inferences per frame, not one: the crop, then the full-frame retry that keeps a
    near goal from being dropped by the edge guard. That doubles the acquisition cost —
    about 131 ms becomes 262 ms — and it is the right place to spend it, because until
    a goal is latched the robot does not stand up at all. The moment one IS latched the
    throttle takes over and the cost goes back to one inference every `refresh_s`.
    """
    detector = _ScriptedDetector([])
    goal = _chair_goal(detector)
    for _ in range(5):
        goal.update(BLANK, ORIGIN)
    assert detector.passes == 10
    assert "0 sightings in 5 passes" in goal.description, "5 passes, 10 inferences"


def test_an_acquired_goal_is_refreshed_only_on_the_throttle():
    """A second inference costs ~131 ms, which a 10 Hz control loop cannot pay every
    cycle. It does not have to: the goal is latched in odom and a chair does not move."""
    clock = _Clock()
    detector = _ScriptedDetector([CHAIR_IN_CROP])
    goal = _chair_goal(detector, refresh_s=1.0, clock=clock)
    goal.update(BLANK, ORIGIN)
    assert detector.passes == 1
    # 0.125 s steps, not 0.1: a tenth is not representable in binary, so adding it ten
    # times lands on 0.9999999999999999 and the throttle misses by one ULP. The test
    # would then be measuring float accumulation rather than the throttle.
    for _ in range(16):                       # 2 s of ticks
        clock.now += 0.125
        goal.update(BLANK, ORIGIN)
    assert detector.passes == 3, detector.passes


def test_the_width_prior_is_the_real_one_when_given():
    """SizePrior.of_height infers width from a PERSON's aspect ratio: 0.31 m for a
    1.07 m chair instead of ~0.62 m."""
    inferred = _chair_goal()
    explicit = _chair_goal(width_m=0.62)
    assert inferred._prior.width_m < 0.4
    assert explicit._prior.width_m == 0.62


def test_a_nonsense_goal_is_rejected_at_construction():
    for bad in ({"height_m": 0.0}, {"width_m": -1.0}, {"crop": 0.0},
                {"crop": 1.5}):
        try:
            _chair_goal(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_no_image_is_not_a_sighting():
    goal = _chair_goal()
    assert goal.update(None, ORIGIN) is None


# --- the crop widens as the goal is approached ------------------------------------

def test_the_crop_stays_where_the_caller_put_it_while_the_goal_is_far():
    """A crop is what makes a distant goal detectable at all — 0.82 cropped against
    NOTHING on the full frame. Widening it early would give that up for nothing."""
    goal = _chair_goal(crop=0.8)
    goal._goal = (4.0, 0.0)
    assert goal._crop_for((0.0, 0.0, 0.0), HEIGHT) == 0.8


def test_the_crop_opens_before_it_can_cut_the_goal_in_half():
    """The failure this exists to stop, measured on the office chair at ~1.9 m: the crop
    clips the backrest, every pass is dropped, the robot never stands up, and no log line
    says why. At 2.0 m a 1.067 m chair is 64% of the frame; 0.8 of the frame with 1.4x
    headroom no longer contains it, so the window has to open."""
    goal = _chair_goal(crop=0.8)
    goal._goal = (2.0, 0.0)
    wide = goal._crop_for((0.0, 0.0, 0.0), HEIGHT)
    assert wide > 0.8
    target_px = CHAIR_HEIGHT_M * CAMERA.focal_px / 2.0
    assert wide * HEIGHT >= target_px, "the crop must contain the target it is cropping to"


def test_the_crop_never_opens_all_the_way_to_the_full_frame():
    """MAX_GOAL_CROP is not 1.0 on purpose. Uncropped, this chair scored NOTHING AT ALL,
    so a crop that opens the whole way does not degrade the goal pass, it removes it.
    Standing almost on top of the goal must not be the case that blinds it."""
    goal = _chair_goal(crop=0.8)
    goal._goal = (0.4, 0.0)
    assert goal._crop_for((0.0, 0.0, 0.0), HEIGHT) == MAX_GOAL_CROP
    assert MAX_GOAL_CROP < 1.0


def test_an_unlatched_goal_keeps_the_callers_crop():
    """Nothing is latched, so there is no range to reason from. Acquisition is the only
    job, and the full-frame retry already covers a crop that is hiding the target."""
    goal = _chair_goal(crop=0.6)
    assert goal.goal_xy() is None
    assert goal._crop_for((0.0, 0.0, 0.0), HEIGHT) == 0.6


def test_the_crop_only_ever_widens_never_narrows():
    """It is a floor, not a value. A caller who asked for a wide window because their
    goal is wide must not have it quietly tightened when the robot gets close."""
    goal = _chair_goal(crop=0.85)
    for distance in (6.0, 4.0, 2.0, 1.0, 0.5):
        goal._goal = (distance, 0.0)
        assert goal._crop_for((0.0, 0.0, 0.0), HEIGHT) >= 0.85


# ── the range scale, against a recorded run ─────────────────────────────────
#: Both live runs of `evidence/2026-08-25-peer-runs/` drove at a printed ArUco marker,
#: and a printed marker's size is known BY MANUFACTURE — so the range the stack computed
#: for it carries no unaudited prior. `ArucoGoal` latches
#: `pose + range*(cos, sin)(yaw + bearing)` on each sighting and the telemetry writes the
#: latched pair, so every raw fix is recoverable from the committed files.
_RUNS = Path(os.path.abspath(__file__)).parents[4] / "evidence" / "2026-08-25-peer-runs"


def _marker_scale(name: str) -> tuple:
    """``(k, residual_rms_m, n)`` for one recorded run.

    Reported range is ``k`` times the true one exactly when
    ``pose_i + (range_i / k) * bearing_i = P`` holds for a fixed ``P``, which is a
    three-parameter linear least squares in ``(P_x, P_y, 1/k)``. Written out here rather
    than imported from `evidence/` so the assertion below does not depend on a directory
    nothing else in the tree imports from.
    """
    fixes, previous = [], None
    with open(_RUNS / f"{name}-run-telemetry.jsonl", encoding="utf-8") as handle:
        for line in handle:
            tick = json.loads(line)
            if tick.get("type") != "tick" or not tick.get("goal"):
                continue
            latched = (tick["goal"]["x"], tick["goal"]["y"])
            if latched == previous:
                continue
            previous = latched
            pose = tick["pose"]
            offset = (latched[0] - pose["x"], latched[1] - pose["y"])
            fixes.append((pose["x"], pose["y"], math.hypot(*offset),
                          math.atan2(offset[1], offset[0])))
    design, target = [], []
    for px, py, range_m, bearing in fixes:
        design.append([1.0, 0.0, -range_m * math.cos(bearing)])
        target.append(px)
        design.append([0.0, 1.0, -range_m * math.sin(bearing)])
        target.append(py)
    solution, *_ = np.linalg.lstsq(np.array(design), np.array(target), rcond=None)
    residual = np.array(design) @ solution - np.array(target)
    rms = float(np.sqrt((residual ** 2).reshape(-1, 2).sum(axis=1).mean()))
    return 1.0 / float(solution[2]), rms, len(fixes)


def test_the_recorded_marker_ranges_agree_with_the_odometry_at_unit_scale():
    """Issue #35 in one assertion: the range scale is not off by 20-30% in either
    direction.

    ⚠️ WHAT WOULD MAKE THIS FAIL, stated because it is narrower than it looks. It is a
    REGRESSION PIN on two committed recordings, so a change to `focal_px` today does not
    move it — the recordings were written with the focal length of the day. What it does
    catch is a change to this fit, and what it is FOR is that the number is now written
    down somewhere a future recalibration has to be re-checked against: re-run
    `evidence/2026-08-26-range-scale-audit/scale_audit.py` on a NEW recording and this is
    the value it has to reproduce.

    ⚠️ AND IT IS A RATIO. k = 1.0 says the camera scale and the leg odometry agree, not
    that either is separately right. A tape is still the only thing that excludes both
    being wrong by the same factor.
    """
    for name, expected in (("hero", 1.092), ("contrast", 1.020)):
        scale, rms, n = _marker_scale(name)
        assert n > 40, (name, n)
        assert abs(scale - expected) < 0.002, (name, scale, expected)
        assert rms < 0.06, (name, rms)
        # The claims under test on #35 — "the map reads 25% LONG" from landmark drift and
        # "the map reads 24% SHORT" from the operator — predict 1.25 and 0.81.
        assert 0.90 < scale < 1.15, (name, scale)


def test_the_marker_fit_is_not_reproducing_odometry_by_construction():
    """Ask what would make the fit report 1.0 when it should not.

    It cannot come from the geometry: scaling every recorded range by a known factor has
    to move ``k`` by exactly that factor, or the estimator is reading the odometry back
    to itself rather than measuring anything. This is the check that the 30-of-35
    landmark-drift argument on #35 could not make, and it is why that argument was
    confounded — a Kalman filter converging toward the robot produces the same signature
    with no scale error at all.
    """
    fixes, previous = [], None
    with open(_RUNS / "hero-run-telemetry.jsonl", encoding="utf-8") as handle:
        for line in handle:
            tick = json.loads(line)
            if tick.get("type") != "tick" or not tick.get("goal"):
                continue
            latched = (tick["goal"]["x"], tick["goal"]["y"])
            if latched == previous:
                continue
            previous = latched
            fixes.append((tick["pose"], latched))

    def scale_with(factor: float) -> float:
        design, target = [], []
        for pose, latched in fixes:
            offset = (latched[0] - pose["x"], latched[1] - pose["y"])
            range_m = math.hypot(*offset) * factor
            bearing = math.atan2(offset[1], offset[0])
            design.append([1.0, 0.0, -range_m * math.cos(bearing)])
            target.append(pose["x"])
            design.append([0.0, 1.0, -range_m * math.sin(bearing)])
            target.append(pose["y"])
        solution, *_ = np.linalg.lstsq(np.array(design), np.array(target), rcond=None)
        return 1.0 / float(solution[2])

    base = scale_with(1.0)
    for factor in (0.75, 1.25):
        assert abs(scale_with(factor) - base * factor) < 1e-6, factor

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"goal: {len(tests)}/{len(tests)} passed")
