# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Track moving obstacles in the ODOMETRY frame, with velocity, from bearing+range.

A detector gives you where a person is *now*. Dodging one needs where they will BE,
and that means velocity — which is the whole reason this module exists rather than
feeding boxes straight to the planner.

TRACKING IN THE ODOM FRAME IS THE POINT. Measurements arrive in the robot's body
frame, but the robot is walking and turning while it measures. Differencing body-frame
positions would credit the robot's own motion to the obstacle: turn left at 0.8 rad/s
and every stationary person in view acquires a metre per second of phantom sideways
velocity, which is exactly the signal the planner is about to swerve on. Converting
each measurement to the estimator's fixed odom frame first (using the pose sampled at
SHUTTER time, not at processing time) makes the estimated velocities real
world velocities, and ego-motion cancels for free.

A constant-velocity Kalman filter per track then supplies two things a raw detection
cannot: a smoothed velocity to extrapolate, and a covariance that grows while the
target is unobserved — so a stale track widens rather than silently going stale.

COASTING PEOPLE OUT OF VIEW. This camera sees ~120°, so someone the robot has just
swerved around is now beside it and invisible. Deleting tracks on consecutive misses
would forget them within a fraction of a second and let the planner turn straight back
into them. Instead a miss only counts against a track that the model says SHOULD be
visible (:meth:`ObstacleTracker.is_visible`); a track outside the field of view coasts
on its last velocity, its covariance inflating, until :data:`COAST_TIMEOUT_S`.

Measurement noise is anisotropic and range-dependent — the size-prior range is far
noisier than the bearing — so ``R`` is built in the sensor frame and rotated into odom
rather than being a single scalar. Pure numpy; no robot required.
Tests: ``python3 test_tracker.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from geometry import hidden_by, wrap_pi

# ── Noise model ─────────────────────────────────────────────────────────────
# Range from a size prior carries a roughly PROPORTIONAL error: the box-height error
# and the +-10% spread in adult height both scale with distance. The floor keeps the
# sigma from collapsing to nothing on a very close target.
RANGE_SIGMA_FRACTION = 0.18
RANGE_SIGMA_FLOOR_M = 0.15
# Bearing is the accurate axis — a box centre is good to a few pixels, and the fisheye
# model converts that to well under a degree. 1.5° carries the calibration error too.
BEARING_SIGMA_RAD = math.radians(1.5)
# A width-prior range (see person_detector) is much weaker than a height-prior one:
# body yaw swings the apparent width nearly 2:1. Inflate its range sigma so the filter
# leans on the good measurements.
WIDTH_SOURCE_SIGMA_SCALE = 2.5

# Pedestrian acceleration driving the constant-velocity process noise. A person can
# change direction hard; this is what lets the filter follow it instead of lagging.
PROCESS_ACCEL_SIGMA = 1.0      # m/s^2

# ── Track lifecycle ─────────────────────────────────────────────────────────
CONFIRM_HITS = 2               # detections before a track is planned against
MAX_MISSES = 4                 # in-view misses before deletion
COAST_TIMEOUT_S = 3.0          # unobserved (incl. out-of-view) before deletion
GATE_CHI2 = 9.21               # 99% for 2 DOF — the association gate
MAX_TRACK_SPEED = 3.0          # m/s; clamp so one bad range cannot fling a track


@dataclass(frozen=True)
class Observation:
    """One detection expressed as a metric measurement in the odom frame."""

    x: float                   # odom-frame position
    y: float
    sigma_along: float         # 1-sigma along the line of sight (range error)
    sigma_cross: float         # 1-sigma perpendicular to it (bearing error)
    world_bearing: float       # line-of-sight direction in odom, for rotating R
    label: str = "person"

    @classmethod
    def from_bearing_range(cls, bearing_rad: float, range_m: float,
                           robot_x: float, robot_y: float, robot_yaw: float,
                           label: str = "person",
                           range_sigma_scale: float = 1.0) -> Observation:
        """Build from a body-frame measurement plus the frame-associated robot pose.

        ``bearing_rad`` is positive to the robot's left. ``range_sigma_scale``
        downweights a weaker range source (see :data:`WIDTH_SOURCE_SIGMA_SCALE`).
        """
        world_bearing = robot_yaw + bearing_rad
        return cls(
            x=robot_x + range_m * math.cos(world_bearing),
            y=robot_y + range_m * math.sin(world_bearing),
            sigma_along=(RANGE_SIGMA_FRACTION * range_m + RANGE_SIGMA_FLOOR_M)
            * range_sigma_scale,
            sigma_cross=max(range_m, 0.1) * BEARING_SIGMA_RAD,
            world_bearing=world_bearing,
            label=label,
        )

    def covariance(self) -> np.ndarray:
        """2x2 measurement covariance in the odom frame.

        Built diagonal in the sensor frame (along/cross line of sight) and rotated,
        because the two axes differ by an order of magnitude and a scalar sigma would
        either trust the range too much or throw away the good bearing.
        """
        c, s = math.cos(self.world_bearing), math.sin(self.world_bearing)
        rot = np.array([[c, -s], [s, c]])
        return rot @ np.diag([self.sigma_along ** 2, self.sigma_cross ** 2]) @ rot.T


def observation_from(bearing_rad: float, range_m: float, source: str, label: str,
                     pose: tuple[float, float, float]) -> Observation:
    """Build an :class:`Observation` from a ranged detection and the robot's pose.

    Takes primitives rather than a detection object so this module stays pure numpy —
    importing the detector would drag OpenCV into every test of the filter. The
    ``source`` string is the detector's account of WHICH size prior produced the
    range, and is the only thing that decides how much the filter trusts it.
    """
    return Observation.from_bearing_range(
        bearing_rad=bearing_rad, range_m=range_m,
        robot_x=pose[0], robot_y=pose[1], robot_yaw=pose[2], label=label,
        range_sigma_scale=1.0 if source == "height" else WIDTH_SOURCE_SIGMA_SCALE)


@dataclass
class Track:
    """A tracked mover: constant-velocity state in the odom frame."""

    track_id: int
    state: np.ndarray                       # [x, y, vx, vy]
    covariance: np.ndarray                  # 4x4
    label: str = "person"
    hits: int = 1
    misses: int = 0
    last_seen: float = 0.0                  # monotonic seconds

    @property
    def position(self) -> np.ndarray:
        return self.state[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:]

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.state[2:]))

    @property
    def confirmed(self) -> bool:
        """Whether the planner should treat this as a real obstacle."""
        return self.hits >= CONFIRM_HITS

    @property
    def position_sigma(self) -> float:
        """Scalar 1-sigma position uncertainty — grows while the track coasts.

        The planner adds this to the obstacle radius, so an unobserved person is
        automatically given a wider berth than a freshly measured one.
        """
        return float(math.sqrt(max(self.covariance[0, 0] + self.covariance[1, 1], 0.0) / 2.0))


class ObstacleTracker:
    """Multi-target constant-velocity tracker over bearing+range observations.

    Args:
        fov_rad: the camera's usable horizontal field of view. Used only to decide
            whether a missing track is *absent* or merely *out of shot*.
        max_range_m: beyond this the detector is not trusted to see people, so a
            missing far-away track likewise does not count as a miss. This is a
            property of the DETECTOR, not of the camera: MobileNet-SSD squashes the
            frame to 300x300, where a person at range ``d`` is about ``433/d`` pixels
            tall, and measured recall falls away below ~45 px — i.e. past ~9.6 m. The
            default is set well inside that, so people at the edge of detectability
            coast on their last estimate through a flickering detection rather than
            being repeatedly created and destroyed. The coast timeout still bounds
            how long they survive unseen.
    """

    def __init__(self, fov_rad: float = math.radians(120.0),
                 max_range_m: float = 6.0) -> None:
        self._fov_rad = fov_rad
        self._max_range_m = max_range_m
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        """All live tracks, confirmed or not."""
        return list(self._tracks)

    def confirmed_tracks(self) -> list[Track]:
        """Tracks stable enough to plan against."""
        return [t for t in self._tracks if t.confirmed]

    # ── Filter steps ────────────────────────────────────────────────────────
    def predict(self, dt: float) -> None:
        """Advance every track by ``dt`` seconds under constant velocity."""
        if dt <= 0.0:
            return
        transition = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # Discrete white-noise acceleration: the covariance a random accel of
        # PROCESS_ACCEL_SIGMA injects over dt.
        q = PROCESS_ACCEL_SIGMA ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        process = q * np.array([
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ])
        for track in self._tracks:
            track.state = transition @ track.state
            track.covariance = transition @ track.covariance @ transition.T + process

    def update(self, observations: list[Observation], now: float,
               robot_x: float, robot_y: float, robot_yaw: float,
               occluders: tuple = ()) -> None:
        """Associate ``observations`` to tracks, correct, spawn, and prune.

        The robot pose is needed only for the visibility test that decides whether an
        unmatched track was absent or merely out of shot. ``occluders`` are the angular
        shadows of known static obstacles — see :meth:`is_visible`.
        """
        pairs = self._associate(observations)
        matched_tracks = {track_index for track_index, _ in pairs}
        matched_obs = {obs_index for _, obs_index in pairs}

        for track_index, obs_index in pairs:
            self._correct(self._tracks[track_index], observations[obs_index], now)

        # Score misses BEFORE spawning, so a track created by this very call is not
        # immediately marked as having been missed by it.
        for index, track in enumerate(self._tracks):
            if index in matched_tracks:
                continue
            # Only penalise a track the camera should have seen. Out of shot — or
            # behind something — is not evidence of absence; see the module docstring.
            if self.is_visible(track, robot_x, robot_y, robot_yaw, occluders):
                track.misses += 1

        for index, obs in enumerate(observations):
            if index not in matched_obs:
                self._spawn(obs, now)

        self._prune(now)

    def _associate(self, observations: list[Observation]
                   ) -> list[tuple[int, int]]:
        """Greedy nearest-neighbour association under a Mahalanobis gate.

        Returns ``(track_index, observation_index)`` pairs.

        Greedy rather than Hungarian: with the handful of people that fit in one
        room, and a gate tight enough that two candidates rarely contend for the same
        track, the two agree — and greedy keeps this dependency-free. A crowded scene
        would want ``scipy.optimize.linear_sum_assignment`` here instead.
        """
        pairs = []
        for track_index, track in enumerate(self._tracks):
            for obs_index, obs in enumerate(observations):
                if obs.label != track.label:
                    continue
                innovation = np.array([obs.x, obs.y]) - track.position
                innovation_cov = track.covariance[:2, :2] + obs.covariance()
                try:
                    distance = float(innovation @ np.linalg.solve(innovation_cov,
                                                                  innovation))
                except np.linalg.LinAlgError:
                    continue
                if distance <= GATE_CHI2:
                    pairs.append((distance, track_index, obs_index))

        pairs.sort()
        used_tracks: set = set()
        used_obs: set = set()
        assignments: list[tuple[int, int]] = []
        for _, track_index, obs_index in pairs:
            if track_index in used_tracks or obs_index in used_obs:
                continue
            used_tracks.add(track_index)
            used_obs.add(obs_index)
            assignments.append((track_index, obs_index))
        return assignments

    def _correct(self, track: Track, obs: Observation, now: float) -> None:
        measurement_matrix = np.zeros((2, 4))
        measurement_matrix[0, 0] = 1.0
        measurement_matrix[1, 1] = 1.0

        innovation = np.array([obs.x, obs.y]) - measurement_matrix @ track.state
        innovation_cov = (measurement_matrix @ track.covariance @ measurement_matrix.T
                          + obs.covariance())
        gain = track.covariance @ measurement_matrix.T @ np.linalg.inv(innovation_cov)
        track.state = track.state + gain @ innovation
        identity = np.eye(4)
        factor = identity - gain @ measurement_matrix
        # Joseph form: stays symmetric positive-definite under the repeated updates a
        # long run accumulates, where the short form can drift negative.
        track.covariance = (factor @ track.covariance @ factor.T
                            + gain @ obs.covariance() @ gain.T)

        speed = float(np.hypot(track.state[2], track.state[3]))
        if speed > MAX_TRACK_SPEED:
            track.state[2:] *= MAX_TRACK_SPEED / speed

        track.hits += 1
        track.misses = 0
        track.last_seen = now

    def _spawn(self, obs: Observation, now: float) -> None:
        covariance = np.zeros((4, 4))
        covariance[:2, :2] = obs.covariance()
        # No velocity information from one detection: start at rest but say so loudly,
        # so the first correction moves the estimate rather than being averaged away.
        covariance[2, 2] = covariance[3, 3] = MAX_TRACK_SPEED ** 2
        self._tracks.append(Track(
            track_id=self._next_id,
            state=np.array([obs.x, obs.y, 0.0, 0.0]),
            covariance=covariance,
            label=obs.label,
            last_seen=now,
        ))
        self._next_id += 1

    def _prune(self, now: float) -> None:
        self._tracks = [t for t in self._tracks
                        if t.misses <= MAX_MISSES
                        and now - t.last_seen <= COAST_TIMEOUT_S]

    # ── Visibility ──────────────────────────────────────────────────────────
    def is_visible(self, track: Track, robot_x: float, robot_y: float,
                   robot_yaw: float, occluders: tuple = ()) -> bool:
        """Whether ``track`` is where the camera could currently see it.

        ``occluders`` are ``(bearing_rad, half_angle_rad, range_m)`` shadows cast by
        known static obstacles, as ``static_map.StaticObstacleMap.occluders`` returns
        them. A track further away than an occluder and inside its angular shadow is
        HIDDEN, not absent.

        WHY THIS MATTERS MORE THAN THE FIELD-OF-VIEW TEST IT SITS NEXT TO. Without it
        the two ways of losing sight of someone get opposite treatment, and the
        asymmetry runs the wrong way: a person who walks out of the cone is credited
        with 3 s of coast, while a person who steps behind a bin INSIDE the cone is
        judged "visible, therefore absent" and deleted in about half a second. That was
        observed live — a volunteer stepped into a doorway, the track was deleted, the
        robot read the lane as clear, accelerated, and they reappeared 0.19 m inside the
        hard gap. An occluder is precisely a thing people walk behind, so building one
        into the scene without this makes the failure routine rather than rare.
        """
        dx, dy = track.state[0] - robot_x, track.state[1] - robot_y
        distance = math.hypot(dx, dy)
        if distance > self._max_range_m:
            return False
        bearing = float(wrap_pi(math.atan2(dy, dx) - robot_yaw))
        if abs(bearing) > self._fov_rad / 2.0:
            return False
        return not hidden_by(bearing, distance, occluders)
