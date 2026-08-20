# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remember where the things that do not move are, in the odom frame.

A STATIC OBSTACLE MUST BE MAPPED, NOT TRACKED, and that is the whole reason this module
exists next to ``tracker.py`` rather than inside it. Feeding a bin to the constant-
velocity tracker is wrong three separate ways, and each one bites in the manoeuvre this
was built for:

  * **It invents velocity.** The filter's whole job is to differentiate position. Range
    from a size prior carries ~18% noise, so a bin sitting still at 2.15 m jitters by
    tens of centimetres between frames and the filter reads that as real motion — then
    the planner rolls it forward two and a half seconds and swerves around a phantom.
  * **It forgets.** Tracks are pruned at ``COAST_TIMEOUT_S`` (3 s). Rounding an obstacle
    is exactly when it leaves a 120-degree field of view, so a tracked bin is deleted
    at the moment the robot is beside it and least able to afford forgetting it.
  * **It re-learns.** Each re-sighting after a prune spawns a NEW track that needs
    ``CONFIRM_HITS`` before the planner will believe it, so the robot alternates between
    avoiding the bin and ignoring it.

A landmark instead accumulates. It is a fixed point with a shrinking covariance, so
repeated sightings make it *sharper* rather than moving it, and it survives leaving the
frame because nothing about a bin depends on being looked at.

FUSION IS AN INFORMATION FILTER, which is just a Kalman update for a state that never
changes: covariances add in inverse. The measurement covariance is anisotropic and
rotates as the robot walks around the object — the range axis is ~10x noisier than the
bearing axis — so successive sightings from different angles genuinely triangulate it.
That is a real property, not a decoration: it is why walking past the bin sharpens its
position instead of smearing it.

WHAT STOPS A PHANTOM BECOMING PERMANENT. A *confirmed* landmark expires on
**disagreement**, never on time: one the camera is looking straight at, in range and
unoccluded, that repeatedly fails to appear is deleted. That is the same evidence rule
``tracker.py`` applies to a person. An *unconfirmed* one expires on time as well
(:data:`UNCONFIRMED_TIMEOUT_S`) — otherwise a single false blob out at 50 degrees, which
the visibility test can never penalise because it is never looked at again, would sit in
the map for the whole run waiting for an unrelated later blob to fuse it into existence.

AND THE ROBOT DRIFTS EVEN WHEN THE BIN DOES NOT. The map is expressed in odom, and odom
is not an inertial frame — it accumulates error as the robot walks. A pure information
filter would freeze the mean (after N sightings the covariance is ~R/N and new evidence
barely moves it), so once drift exceeded the association gate the observations would stop
matching and spawn a *second* landmark on top of the first. :data:`DRIFT_SIGMA_M_PER_S`
re-inflates the covariance with age, which is what lets a landmark follow the frame it is
expressed in rather than fighting it.

Pure numpy; no robot and no OpenCV. ``python3 test_static_map.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from geometry import angular_shadow, hidden_by, wrap_pi
from tracker import Observation

#: Sightings before the planner is allowed to act on a landmark. One flash of the right
#: colour must not stop the robot; two consecutive agreeing ones is evidence.
CONFIRM_SIGHTINGS = 2

#: In-view, in-range, unoccluded misses before a landmark is deleted. Deliberately
#: larger than the tracker's MAX_MISSES: a bin that is briefly mis-segmented (someone
#: steps half in front of it) should not be forgotten, because unlike a person it
#: cannot have walked away.
#:
#: RAISED 8 -> 25 after a live run of 2026-08-19 stalled on the consequence. At 5-7 Hz
#: perception, 8 misses is a 1.1-1.6 s dropout, and the colour detector loses a bin for
#: longer than that as the robot closes on it: at 0.71 m a 0.3 m bin fills 51% of the
#: frame and is clipped by its bottom edge, which puts the blob under `min_fill` and
#: outside the aspect band. :meth:`is_visible` still calls it visible — it is, it just
#: cannot be segmented — so misses accrue on a landmark the detector was never going to
#: match. The landmark is then deleted after 32 sightings of converged evidence, the very
#: next detection spawns a fresh one 0.216 m away with an unconverged 0.518 m radius, the
#: gap between the two bins collapses from 0.23 m to 0.06 m, and the run holds and stalls.
#: 25 covers a ~4 s dropout, which is longer than any observed close-range loss.
#:
#: This is the blunt half of the fix. The precise half is not to score a miss the
#: detector could not have won — see :meth:`is_visible` — and is tracked separately.
#: What is bought here is that a landmark's *identity*, and therefore its converged
#: covariance, survives the approach.
MAX_MISSES = 25

#: Association distance, metres, on top of the landmark's own radius. Euclidean and
#: FIXED, deliberately unlike the tracker's Mahalanobis gate: a landmark's covariance
#: collapses as sightings accumulate, so a Mahalanobis gate collapses with it and any
#: systematic bias — a size prior 5% off, say — would stop associating and spawn a
#: duplicate landmark every frame. A physical distance cannot do that. The two props in
#: the staged scene are ~2 m apart, so this is not close to merging them.
ASSOCIATION_GATE_M = 0.75

#: Floor on a landmark's reported position sigma, metres. The information filter drives
#: the covariance toward zero given enough sightings, which would claim a precision the
#: odometry underneath it does not have — the goal latch has the same problem and the
#: same answer. This is the drift the robot accumulates walking a few metres.
POSITION_SIGMA_FLOOR_M = 0.08

#: Odometry drift injected into a landmark's covariance per second, metres. Without it
#: the fused mean FREEZES: a pure information update has no process noise, so after N
#: sightings the covariance is ~R/N and later observations cannot move the estimate. The
#: landmark then stops following the drifting frame it is expressed in, and once the
#: error exceeds the association gate every observation spawns a duplicate instead of
#: correcting the original — two overlapping bins spanning most of the lane, one of them
#: a ghost that takes MAX_MISSES to die. The floor above widens the RADIUS; only this
#: un-freezes the POSITION.
DRIFT_SIGMA_M_PER_S = 0.02

#: Position sigma, metres, beyond which a landmark is no longer planned against. It is
#: kept and can re-converge — it is simply not evidence yet.
#:
#: Measured on the failed live run: a landmark reached sigma 0.52 m, so the planner saw a
#: 0.69 m disc, and four of them between the robot and its goal. A landmark that uncertain
#: has stopped carrying DIRECTION as well as position — it blocks a bearing the robot
#: could have taken as hard as one it could not — so planning against it is worse than
#: ignoring it.
#:
#: Sized from BOTH sides rather than picked round. A legitimately acquired landmark is
#: already at 0.27 m after its first two sightings of the staged bin at 2.15 m, and falls
#: below 0.10 m within ten; the run's ghosts sat at 0.52 m. Set it at 0.30 and a real
#: landmark one metre further out would fail to confirm and the robot would walk at it.
MAX_PLANNING_SIGMA_M = 0.40

#: Most landmarks of one label the map will plan against, best-evidenced first. ONE bin
#: cannot be in four places, and on the failed run it was: odometry that reads
#: "stationary" while the robot is physically dragged projects every sighting to a
#: different odom point, and each becomes its own landmark. The duplicates then box the
#: robot in. This does not fix the odometry — nothing here can — it bounds the damage,
#: and the stall abort in visual_nav is what actually catches the cause.
MAX_LANDMARKS_PER_LABEL = 2

#: Seconds an UNCONFIRMED landmark survives without being seen again. Confirmed ones
#: have no such limit — a bin that has been seen properly has no reason to stop existing
#: because nobody looked. This bounds the other case: a one-frame colour false positive
#: outside the field of view can never be penalised by the visibility test, so without a
#: clock it would live for the whole run.
UNCONFIRMED_TIMEOUT_S = 2.0


@dataclass
class Landmark:
    """A thing that does not move, in odom coordinates."""

    landmark_id: int
    mean: np.ndarray                       # [x, y]
    covariance: np.ndarray                 # 2x2
    radius_m: float
    label: str = "bin"
    sightings: int = 1
    misses: int = 0
    last_seen: float = 0.0

    @property
    def x(self) -> float:
        return float(self.mean[0])

    @property
    def y(self) -> float:
        return float(self.mean[1])

    @property
    def confirmed(self) -> bool:
        """Whether the planner should treat this as a real obstacle."""
        return self.sightings >= CONFIRM_SIGHTINGS

    @property
    def position_sigma(self) -> float:
        """Scalar 1-sigma position uncertainty, floored at the odometry's own drift."""
        trace = float(self.covariance[0, 0] + self.covariance[1, 1])
        return max(math.sqrt(max(trace, 0.0) / 2.0), POSITION_SIGMA_FLOOR_M)

    @property
    def planning_radius_m(self) -> float:
        """Radius the planner must clear: the object, widened by how unsure we are."""
        return self.radius_m + self.position_sigma


class StaticObstacleMap:
    """Landmarks for things that do not move, fused across sightings in odom.

    Deliberately shaped like :class:`tracker.ObstacleTracker` — same constructor style,
    same ``landmarks``/``confirmed`` split, same observe-then-prune order — because the
    two are read side by side whenever anyone is working out why the planner saw what it
    saw, and gratuitous differences between them cost more than they save.

    Args:
        radii: per-label plan-view radius in metres. A bin's footprint is a fifth of a
            person's, and inflating it to person-size is what turns a passable gap into
            a local minimum for the planner.
        default_radius_m: used for a label ``radii`` does not name.
        fov_rad/max_range_m: the same visibility envelope the tracker uses, here only to
            decide whether a landmark that failed to appear counts as evidence of
            absence. Defaults match ``ObstacleTracker``.
    """

    def __init__(self, radii: dict | None = None, default_radius_m: float = 0.25,
                 fov_rad: float = math.radians(120.0),
                 max_range_m: float = 6.0) -> None:
        self.radii = dict(radii or {})
        self.default_radius_m = default_radius_m
        self.fov_rad = fov_rad
        self.max_range_m = max_range_m
        self._landmarks: list[Landmark] = []
        self._next_id = 1

    # ── Queries ─────────────────────────────────────────────────────────────
    @property
    def landmarks(self) -> list[Landmark]:
        """Every landmark, confirmed or not."""
        return list(self._landmarks)

    def confirmed(self) -> list[Landmark]:
        """Landmarks stable enough, and certain enough, to plan against.

        Three gates, and the last two exist because of a live run that failed on exactly
        this: enough sightings, a position the map is still sure of
        (:data:`MAX_PLANNING_SIGMA_M`), and no more than
        :data:`MAX_LANDMARKS_PER_LABEL` of any one kind — best-evidenced first, since a
        duplicate spawned by bad odometry has few sightings and a wide covariance while
        the real one has many and a tight one.
        """
        candidates = [lm for lm in self._landmarks
                      if lm.confirmed and lm.position_sigma <= MAX_PLANNING_SIGMA_M]
        by_label: dict = {}
        for landmark in candidates:
            by_label.setdefault(landmark.label, []).append(landmark)
        kept = []
        for landmarks in by_label.values():
            landmarks.sort(key=lambda lm: (-lm.sightings, lm.position_sigma))
            kept.extend(landmarks[:MAX_LANDMARKS_PER_LABEL])
        # Map order, not ranking order, so the overlay and the logs stay stable.
        keep = {id(lm) for lm in kept}
        return [lm for lm in self._landmarks if id(lm) in keep]

    def radius_for(self, label: str) -> float:
        return float(self.radii.get(label, self.default_radius_m))

    # ── Fusion ──────────────────────────────────────────────────────────────
    def observe(self, observations: list, now: float, robot_x: float, robot_y: float,
                robot_yaw: float, occluders: tuple = ()) -> None:
        """Fold one perception cycle's static observations into the map.

        ``observations`` are :class:`tracker.Observation` — the same objects the mover
        tracker consumes, so the perception worker builds them once.

        ``occluders`` are :func:`geometry.angular_shadow` triples for things that can
        stand in front of a landmark — in practice the tracked PEOPLE. Occlusion runs
        both ways: this map already hides people behind the bin for the tracker, and
        without the reverse a volunteer walking between the robot and the bin blanks the
        colour blob for a second or two and the landmark is pruned — removing the only
        static obstacle from the planner at the exact moment the robot is committed to a
        swerve around it.
        """
        pairs = self._associate(observations)
        matched_landmarks = {lm_index for lm_index, _ in pairs}
        matched_obs = {obs_index for _, obs_index in pairs}

        for lm_index, obs_index in pairs:
            self._fuse(self._landmarks[lm_index], observations[obs_index], now)

        # Score misses BEFORE spawning, so a landmark created by this call is not
        # immediately marked as missed by it — the same ordering tracker.update uses.
        for lm_index, landmark in enumerate(self._landmarks):
            if lm_index in matched_landmarks:
                continue
            if self.is_visible(landmark, robot_x, robot_y, robot_yaw, occluders):
                landmark.misses += 1

        for obs_index, obs in enumerate(observations):
            if obs_index not in matched_obs:
                self._spawn(obs, now)

        self._prune(now)

    def _associate(self, observations: list) -> list[tuple[int, int]]:
        """Greedy nearest-neighbour association, ``(landmark_index, obs_index)`` pairs.

        Same greedy shape as ``ObstacleTracker._associate``, but gated on plain EUCLIDEAN
        distance rather than Mahalanobis. That difference is deliberate, and it is the
        one place a landmark must not behave like a track: a landmark's covariance
        collapses as sightings accumulate, so a Mahalanobis gate collapses with it, and
        any systematic bias — a size prior a few per cent off, say — would then stop
        associating and spawn a fresh landmark every single frame. A physical distance
        cannot do that. See :data:`ASSOCIATION_GATE_M`.
        """
        candidates = []
        for lm_index, landmark in enumerate(self._landmarks):
            for obs_index, obs in enumerate(observations):
                if obs.label != landmark.label:
                    continue
                distance = math.hypot(obs.x - landmark.x, obs.y - landmark.y)
                if distance <= landmark.radius_m + ASSOCIATION_GATE_M:
                    candidates.append((distance, lm_index, obs_index))

        # Nearest first, so when two observations fall inside one landmark's gate the
        # closer one claims it rather than whichever happened to be listed first.
        candidates.sort()
        used_landmarks: set = set()
        used_obs: set = set()
        assignments: list[tuple[int, int]] = []
        for _, lm_index, obs_index in candidates:
            if lm_index in used_landmarks or obs_index in used_obs:
                continue
            used_landmarks.add(lm_index)
            used_obs.add(obs_index)
            assignments.append((lm_index, obs_index))
        return assignments

    def _prune(self, now: float) -> None:
        """Drop landmarks that lost their evidence.

        Two rules, because the two cases are not alike. A CONFIRMED landmark goes only on
        disagreement — repeatedly looked at and not found — since a thing that does not
        move has no reason to stop existing because nobody looked. An UNCONFIRMED one
        also goes on time: it may sit outside the field of view where the visibility test
        can never penalise it, so a clock is the only thing that can retire it.
        """
        self._landmarks = [
            lm for lm in self._landmarks
            if lm.misses <= MAX_MISSES
            and (lm.confirmed or now - lm.last_seen <= UNCONFIRMED_TIMEOUT_S)
        ]

    def _fuse(self, landmark: Landmark, obs: Observation, now: float) -> None:
        """Information-filter update of a state that does not change.

        ``P' = (P^-1 + R^-1)^-1`` and ``x' = P'(P^-1 x + R^-1 z)``. Written in
        information form rather than as a Kalman gain because that is what it is —
        there is no transition to carry.

        There IS process noise, though, and leaving it out was a bug: the landmark does
        not move but the ODOM FRAME IT IS EXPRESSED IN does. See
        :data:`DRIFT_SIGMA_M_PER_S`.
        """
        landmark.covariance = landmark.covariance + self._drift(now - landmark.last_seen)
        prior_information = np.linalg.inv(landmark.covariance)
        obs_information = np.linalg.inv(obs.covariance())
        covariance = np.linalg.inv(prior_information + obs_information)
        landmark.mean = covariance @ (prior_information @ landmark.mean
                                      + obs_information @ np.array([obs.x, obs.y]))
        landmark.covariance = covariance
        landmark.sightings += 1
        landmark.misses = 0
        landmark.last_seen = now

    @staticmethod
    def _drift(elapsed_s: float) -> np.ndarray:
        """Isotropic covariance the odom frame accumulates over ``elapsed_s``."""
        variance = (DRIFT_SIGMA_M_PER_S * max(0.0, elapsed_s)) ** 2
        return np.eye(2) * variance

    def _spawn(self, obs: Observation, now: float) -> None:
        self._landmarks.append(Landmark(
            landmark_id=self._next_id,
            mean=np.array([obs.x, obs.y], dtype=float),
            covariance=obs.covariance(),
            radius_m=self.radius_for(obs.label),
            label=obs.label,
            last_seen=now))
        self._next_id += 1

    # ── Visibility ──────────────────────────────────────────────────────────
    def is_visible(self, landmark: Landmark, robot_x: float, robot_y: float,
                   robot_yaw: float, occluders: tuple = ()) -> bool:
        """Whether the camera could currently see ``landmark``.

        Same envelope as ``ObstacleTracker.is_visible``, and used for the same purpose:
        only a landmark that SHOULD have been seen may be penalised for being absent —
        which means range, field of view AND nothing standing in front of it.
        """
        dx, dy = landmark.x - robot_x, landmark.y - robot_y
        distance = math.hypot(dx, dy)
        if distance > self.max_range_m:
            return False
        bearing = float(wrap_pi(math.atan2(dy, dx) - robot_yaw))
        if abs(bearing) > self.fov_rad / 2.0:
            return False
        return not hidden_by(bearing, distance, occluders)

    def occluders(self, robot_x: float, robot_y: float,
                  robot_yaw: float) -> list:
        """Angular shadows the confirmed landmarks cast, as seen from the robot.

        Returns ``(bearing_rad, half_angle_rad, range_m)`` per landmark, bearing
        relative to the robot's nose. ``tracker.ObstacleTracker`` uses these to tell
        "this person is gone" from "this person is behind the bin" — without them a
        target that vanishes INSIDE the field of view is read as absence, which is the
        one direction that is unsafe.

        The half-angle uses the planning radius, not the physical one, so a landmark
        whose own position is uncertain casts a correspondingly wider shadow.
        """
        shadows = []
        for landmark in self._landmarks:
            if not landmark.confirmed:
                continue
            shadow = angular_shadow(robot_x, robot_y, robot_yaw,
                                    landmark.x, landmark.y, landmark.planning_radius_m)
            if shadow is not None:
                shadows.append(shadow)
        return shadows
