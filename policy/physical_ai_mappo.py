# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the delivered MAPPO actor once per robot control step.

Authored by Sagar Surendran and delivered as ``physicalai_mappo_go2.zip``. The deltas
applied on the way into this repository are listed in ``PROVENANCE.md``; each one is
marked ``CORRECTION`` in the code below with the reason it was needed. Do not "tidy"
those markers away — they are what stops the next re-vendor from silently reverting
them.

The shape of the thing: the policy is a **holonomic, non-rotating** VMAS agent. Its
observation is ``[x, y, vx, vy, x-gx, y-gy, *12 lidar]`` in a **run-local frame** fixed
at the first ``reset_run=True`` step, every term divided by
:attr:`Config.meters_per_vmas_unit`. Because the trained agent never rotates, the 12 rays
sit at FIXED angles in that run-local frame rather than turning with the robot's nose,
and yaw is used only to convert the resulting action back into a body-frame command.
``lidar`` is PROXIMITY, ``range_max - range``: bigger means closer.

All of that was verified against the weights rather than read off this file — a
goal-bearing sweep tracks the goal to within 14.5 degrees, and a ring of obstacles
produces strong evasion where clear space does not. See ``integration/replay_mappo.py``.

``python3 test_physical_ai_mappo.py`` for the behaviour suite; ``python3 basic_test.py``
for the one-inference smoke check the installer runs on the target machine.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_RAYS = 12
OBS_DIM = 18

#: Velocity frames :func:`MappoController._local_state` can convert from. Split into two
#: tuples because the conversion differs: an odom velocity is rotated by the frame the run
#: reset into, a body one by how far the robot has turned since.
ODOM_FRAMES = ("odom", "world")
BODY_FRAMES = ("body", "robot")
VELOCITY_FRAMES = ODOM_FRAMES + BODY_FRAMES

#: How far two detections may be apart and still be treated as the same object, when
#: neither carries an ``object_id``. Named because it is a real limit a caller has to know
#: about: without ids, two objects that pass this close merge into one and stay merged.
ASSOCIATION_RADIUS_M = 0.45

#: Smoothing applied to a re-observed obstacle's mapped position. Low, because the
#: obstacles this policy is given are STATIC — a large step would be tracking measurement
#: noise, and a mapped landmark that jitters moves the range vector on every tick.
POSITION_SMOOTHING = 0.35

#: Slack allowed on ``timestamp_s`` before it is called a broken clock. A coarse
#: ``time.monotonic()`` can read a few milliseconds backwards across a process boundary;
#: 50 ms covers that with room to spare. It cannot hide the failure this guards against —
#: a wall-clock timestamp is out by about 1.8e9 s, ten orders of magnitude past it.
CLOCK_TOLERANCE_S = 0.05


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class StationaryObject:
    """A STATIONARY object, in the robot's body frame.

    Stationary means ``kind == "static"`` as the control stack reports it — a mapped
    landmark — and NOT "a track whose velocity happens to read zero". The division of
    labour is that movers stay with the existing stop/wait/resume logic and reach this
    controller as :attr:`RobotInput.external_hold`, while stationary objects become the
    lidar-like input the policy is trained to path around. A person who has STOPPED has a
    bin's velocity and a person's claim on the lane, so a speed threshold cannot make this
    call; ``integration/mappo_bridge.is_stationary`` reads the explicit field.
    """

    distance_m: float
    bearing_rad: float          # +left / CCW from robot forward
    radius_m: float = 0.15
    object_id: str | None = None


@dataclass
class RobotInput:
    """Input required for one MAPPO inference/control step."""

    x_m: float
    y_m: float
    yaw_rad: float
    vx_mps: float
    vy_mps: float
    goal_x_m: float
    goal_y_m: float
    #: CORRECTION (was ``"odom"``). ``None`` means "whatever the deployment config says",
    #: which for this stack is ``"body"``: ``measured`` comes from the Go2's own estimator
    #: (``SportModeState_.velocity``), which is body-frame. The default was the one field
    #: where being wrong is invisible — the two frames agree EXACTLY while the robot faces
    #: its start heading and diverge only as it turns, so a bench test at yaw 0 cannot
    #: tell them apart. Confirmed by @spsagar13, AIDP-567.
    velocity_frame: str | None = None
    external_hold: bool = False     # existing dynamic-obstacle/safety stop
    stationary_objects: Sequence[StationaryObject] = field(default_factory=tuple)
    reset_run: bool = False
    #: MUST be a ``time.monotonic()`` reading or ``None``. It is compared against this
    #: process's own ``time.monotonic()``; the two clocks share no epoch with a wall clock.
    timestamp_s: float | None = None


@dataclass
class ActionOutput:
    status: str
    action_x: float
    action_y: float
    vx_mps: float
    vy_mps: float
    vyaw_radps: float
    #: How old the input was, in seconds, by the controller's own clock. Reported rather
    #: than only thresholded so a caller can log the distribution and see a clock problem
    #: coming before it trips :data:`STOP_CLOCK_ERROR`.
    age_s: float = 0.0


#: The policy is driving.
COMMAND = "COMMAND"
#: ``timestamp_s`` is older than :attr:`Config.stale_input_timeout_s`.
STOP_STALE_INPUT = "STOP_STALE_INPUT"
#: CORRECTION (new). ``timestamp_s`` is in this process's FUTURE by more than
#: :data:`CLOCK_TOLERANCE_S`, so it is not a ``time.monotonic()`` reading and the age is
#: meaningless. Distinct from :data:`STOP_STALE_INPUT` because the fix is different — one
#: is a slow producer, the other is a wiring mistake — and because the failure it replaces
#: was silent: a wall-clock stamp gives an age of about -1.8e9 s, which is under any
#: threshold, so the staleness gate could never fire and the controller kept driving on a
#: frozen world model. It failed OPEN. @spsagar13 agreed to this guard in AIDP-567.
STOP_CLOCK_ERROR = "STOP_CLOCK_ERROR"
#: The existing moving-object/safety logic is in charge this tick.
STOP_EXTERNAL_HOLD = "STOP_EXTERNAL_HOLD"
#: Within :attr:`Config.goal_stop_distance_m` of the goal.
STOP_GOAL_REACHED = "STOP_GOAL_REACHED"


@dataclass
class Config:
    """Deployment configuration. Everything here is a property of THIS robot and room.

    The two numbers the policy itself constrains — :attr:`lidar_range_vmas` and the
    observation width — are checked against the checkpoint's own metadata at load, so a
    config that disagrees with training fails loudly instead of feeding the network an
    observation it never saw.
    """

    model_path: str = "models/mappo_actor_3agent_1910000.npz"
    #: CALIBRATION, not a model requirement — confirmed by @spsagar13 in AIDP-567. It is
    #: the single most consequential number here: it sets the sensing horizon, at
    #: ``lidar_range_vmas * meters_per_vmas_unit``. 2.5 matches the ROBOT to the trained
    #: agent (the live runs' 0.25 m planner radius / the trained 0.10 VMAS agent radius)
    #: rather than the room to the trained spawn region, which is where the delivered 1.5
    #: came from. See ``integration/replay_mappo.py --config`` and issue #4.
    meters_per_vmas_unit: float = 2.5
    lidar_range_vmas: float = 0.35
    #: CORRECTION (new). The frame ``vx_mps``/``vy_mps`` arrive in when a caller does not
    #: say. Body for this stack; see :attr:`RobotInput.velocity_frame`.
    velocity_frame: str = "body"
    max_vx_mps: float = 0.35
    max_vy_mps: float = 0.20
    #: CALIBRATION, and it is a SPEED knob rather than a safety one — the safety envelope
    #: is the control stack's ``Limits``, which ``mappo_drive`` clamps to and which
    #: ``--derate`` scales. It first moved from the delivered 0.30 to 0.60 for the run
    #: budget, then to 1.0 after hardware showed that 0.60 commands only 0.21 m/s — below
    #: this Go2's gait floor. At 1.0 the robot completed the first policy-driven walk;
    #: the control stack's ``Limits`` remains the safety envelope.
    command_scale: float = 1.0
    goal_stop_distance_m: float = 0.20
    stale_input_timeout_s: float = 0.75
    static_obstacle_ttl_s: float = 120.0

    def __post_init__(self) -> None:
        """Reject a config that cannot produce a usable observation.

        Without this, a zero or negative scale divides the whole observation into
        infinities and the failure surfaces as ``"observation must contain 18 finite
        values"`` from inside the actor, several frames from the config that caused it.
        """
        if self.velocity_frame.lower() not in VELOCITY_FRAMES:
            raise ValueError(f"velocity_frame must be one of {VELOCITY_FRAMES}, "
                             f"not {self.velocity_frame!r}")
        for name in ("meters_per_vmas_unit", "lidar_range_vmas", "max_vx_mps",
                     "max_vy_mps", "stale_input_timeout_s", "static_obstacle_ttl_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, not {value!r}")
        if not 0.0 <= self.command_scale <= 1.0:
            raise ValueError(f"command_scale must be in [0, 1], not {self.command_scale!r}")
        if not math.isfinite(self.goal_stop_distance_m) or self.goal_stop_distance_m < 0.0:
            raise ValueError("goal_stop_distance_m must be finite and non-negative, "
                             f"not {self.goal_stop_distance_m!r}")

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Read a config file, naming any key this class does not know.

        ``cls(**json.loads(...))`` on its own reports an unknown key as a bare
        ``TypeError`` from the constructor, which does not say which file it came from —
        and a MISSPELLED key is worse than an unknown one, because the field it was meant
        to set silently keeps its default. Both are the same mistake and both are caught
        here.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"{path}: unknown config key(s) {unknown}. Known keys: "
                             f"{sorted(cls.__dataclass_fields__)}")
        return cls(**data)

    @property
    def lidar_range_m(self) -> float:
        """The sensing horizon, to the obstacle's SURFACE."""
        return self.lidar_range_vmas * self.meters_per_vmas_unit


class _Actor:
    """The delivered network: 18 -> 256 -> 256 -> 4, tanh throughout.

    The 4 raw outputs are a ``TanhNormal``'s ``loc`` and ``scale``; the deterministic
    action the deployment uses is ``tanh(loc)``, i.e. the first two. That is not a guess —
    the checkpoint states it in :attr:`metadata`.
    """

    def __init__(self, path: str | Path):
        d = np.load(path, allow_pickle=False)
        self.w1 = d["W1"].astype(np.float32, copy=False)
        self.b1 = d["b1"].astype(np.float32, copy=False)
        self.w2 = d["W2"].astype(np.float32, copy=False)
        self.b2 = d["b2"].astype(np.float32, copy=False)
        self.w3 = d["W3"].astype(np.float32, copy=False)
        self.b3 = d["b3"].astype(np.float32, copy=False)
        if self.w1.shape != (256, 18) or self.w2.shape != (256, 256) or self.w3.shape != (4, 256):
            raise ValueError(
                f"Unexpected model shape in {path}: W1 {self.w1.shape}, W2 "
                f"{self.w2.shape}, W3 {self.w3.shape}; expected (256, 18), (256, 256), "
                f"(4, 256)")
        #: CORRECTION (new). The checkpoint ships a ``metadata_json`` array recording what
        #: it was trained with, and nothing read it. It is the only in-band statement of
        #: the training constants, so it is the right thing to validate the config
        #: against — see :meth:`MappoController._check_against_checkpoint`. Empty for a
        #: checkpoint that predates the field, which is why every use below is a lookup
        #: with a default rather than an index.
        self.metadata: dict = {}
        if "metadata_json" in d.files:
            self.metadata = json.loads(str(d["metadata_json"]))

    def act(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (OBS_DIM,) or not np.all(np.isfinite(observation)):
            raise ValueError(f"MAPPO observation must be {OBS_DIM} finite values, got "
                             f"shape {observation.shape}")
        x = np.tanh(self.w1 @ observation + self.b1)
        x = np.tanh(self.w2 @ x + self.b2)
        raw = self.w3 @ x + self.b3
        return np.tanh(raw[:2]).astype(np.float32)


@dataclass
class _Obstacle:
    x: float
    y: float
    radius: float
    last_seen: float
    object_id: str | None


class MappoController:
    """Stateful per-robot controller. Call :meth:`step` once per robot control cycle."""

    def __init__(self, config_path: str | Path = "config.json"):
        config_path = Path(config_path).resolve()
        self.cfg = Config.load(config_path)
        model_path = Path(self.cfg.model_path)
        if not model_path.is_absolute():
            model_path = config_path.parent / model_path
        self.actor = _Actor(model_path)
        self._check_against_checkpoint(model_path)
        self._origin: tuple[float, float, float] | None = None
        self._obstacles: list[_Obstacle] = []
        #: The 18 values handed to the network on the last :meth:`step`, or ``None``
        #: before the first. An attribute rather than a field of :class:`ActionOutput`
        #: so that printing a result stays readable. It is what a shadow run logs and
        #: what an off-robot analysis needs: an action alone cannot be checked, because
        #: the input that produced it is the half that can be wrong.
        self.last_observation: np.ndarray | None = None

    def _check_against_checkpoint(self, model_path: Path) -> None:
        """Fail if the config claims something the checkpoint was not trained with.

        ``lidar_range_vmas`` is the one config field the POLICY constrains rather than the
        room: it is the range the observation's proximity convention is measured against,
        so a config that disagrees hands the network a lidar vector on a different scale
        from the one it learned. Nothing downstream notices — the values stay finite and
        in range, the robot simply steers wrongly. ``meters_per_vmas_unit`` is deliberately
        NOT checked: it is a calibration and it is meant to be swept.
        """
        meta = self.actor.metadata
        if not meta:
            return                      # a checkpoint from before the field existed
        input_dim = meta.get("actor_input_dim")
        if input_dim is not None and input_dim != OBS_DIM:
            raise ValueError(f"{model_path}: checkpoint expects a {input_dim}-value "
                             f"observation, this adapter builds {OBS_DIM}")
        layout = meta.get("observation_layout")
        if layout is not None:
            rays = sum(1 for name in layout if name.startswith("lidar"))
            if rays != N_RAYS:
                raise ValueError(f"{model_path}: checkpoint has {rays} lidar features, "
                                 f"this adapter casts {N_RAYS} rays")
        trained_range = meta.get("training_lidar_range_vmas")
        if trained_range is not None and not math.isclose(
                trained_range, self.cfg.lidar_range_vmas, rel_tol=1e-6):
            raise ValueError(
                f"config lidar_range_vmas={self.cfg.lidar_range_vmas} but "
                f"{model_path.name} was trained with {trained_range}. The proximity "
                f"convention is measured against this range, so the two must agree. "
                f"To change the SENSING DISTANCE, change meters_per_vmas_unit "
                f"(currently {self.cfg.meters_per_vmas_unit}, giving a "
                f"{self.cfg.lidar_range_m:.3f} m horizon).")

    @property
    def agent_radius_m(self) -> float | None:
        """The trained agent's radius in metres at the configured scale, or ``None``.

        This is what :attr:`Config.meters_per_vmas_unit` is calibrated against: set the
        scale so this matches the radius the control stack plans with, and the policy's
        idea of how much room it needs matches the robot's. ``None`` for a checkpoint that
        does not record its training radius.
        """
        radius = self.actor.metadata.get("training_agent_radius_vmas")
        return None if radius is None else radius * self.cfg.meters_per_vmas_unit

    def _reset(self, inp: RobotInput) -> None:
        self._origin = (inp.x_m, inp.y_m, inp.yaw_rad)
        self._obstacles.clear()

    def _to_local_point(self, x: float, y: float) -> tuple[float, float]:
        assert self._origin is not None
        ox, oy, oyaw = self._origin
        dx, dy = x - ox, y - oy
        c, s = math.cos(oyaw), math.sin(oyaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _local_state(self, inp: RobotInput
                     ) -> tuple[float, float, float, float, float, float, float]:
        assert self._origin is not None
        x, y = self._to_local_point(inp.x_m, inp.y_m)
        gx, gy = self._to_local_point(inp.goal_x_m, inp.goal_y_m)
        yaw = _wrap(inp.yaw_rad - self._origin[2])

        # CORRECTION: fall back to the deployment config rather than to "odom".
        frame = (inp.velocity_frame or self.cfg.velocity_frame).lower()
        if frame in ODOM_FRAMES:
            # An odom velocity is already in a non-rotating frame; only the fixed offset
            # between odom and the run-local frame applies.
            c, s = math.cos(self._origin[2]), math.sin(self._origin[2])
            vx = c * inp.vx_mps + s * inp.vy_mps
            vy = -s * inp.vx_mps + c * inp.vy_mps
        elif frame in BODY_FRAMES:
            # A body velocity turns with the robot, so it is rotated by how far the robot
            # has turned SINCE the reset, not by the reset heading.
            c, s = math.cos(yaw), math.sin(yaw)
            vx = c * inp.vx_mps - s * inp.vy_mps
            vy = s * inp.vx_mps + c * inp.vy_mps
        else:
            raise ValueError(f"velocity_frame must be one of {VELOCITY_FRAMES}, "
                             f"not {frame!r}")
        return x, y, yaw, vx, vy, gx, gy

    def _update_obstacles(self, inp: RobotInput, x: float, y: float, yaw: float,
                          now: float) -> None:
        cy, sy = math.cos(yaw), math.sin(yaw)

        for d in inp.stationary_objects:
            if not math.isfinite(d.distance_m) or d.distance_m <= 0 or \
                    not math.isfinite(d.bearing_rad):
                continue
            bx = d.distance_m * math.cos(d.bearing_rad)
            by = d.distance_m * math.sin(d.bearing_rad)
            wx = x + cy * bx - sy * by
            wy = y + sy * bx + cy * by
            radius = max(0.0, float(d.radius_m))

            match = None
            #: Whether this detection was matched to a mapped obstacle by IDENTITY. It
            #: decides how the radius combines below, and the two cases are genuinely
            #: different: one is the producer re-measuring an object it has named, the
            #: other is two unnamed detections being declared the same thing.
            matched_by_id = False
            if d.object_id is not None:
                match = next((o for o in self._obstacles if o.object_id == d.object_id),
                             None)
                matched_by_id = match is not None
            if match is None:
                # CORRECTION: the positional fallback used to consider EVERY mapped
                # obstacle, including ones already carrying a different identity — so two
                # objects 0.2 m apart merged even when the producer had told them apart.
                # That defeated the whole point of adding `id` to the telemetry, and it
                # did it silently: the merged disc takes the larger radius, so the range
                # vector still looks plausible. Only an obstacle that is anonymous, or is
                # this same one, may absorb a detection by position.
                eligible = [o for o in self._obstacles
                            if o.object_id is None or o.object_id == d.object_id]
                if eligible:
                    candidate = min(eligible,
                                    key=lambda o: math.hypot(o.x - wx, o.y - wy))
                    if math.hypot(candidate.x - wx,
                                  candidate.y - wy) <= ASSOCIATION_RADIUS_M:
                        match = candidate

            if match is None:
                self._obstacles.append(_Obstacle(wx, wy, radius, now, d.object_id))
            else:
                match.x = (1 - POSITION_SMOOTHING) * match.x + POSITION_SMOOTHING * wx
                match.y = (1 - POSITION_SMOOTHING) * match.y + POSITION_SMOOTHING * wy
                # CORRECTION (new). A re-observation of an object the producer has
                # NAMED takes the radius it reports now; only an anonymous merge keeps
                # the larger of the two. The delivered code took `max` unconditionally,
                # which turns the mapped radius into a high-water mark that can never
                # come down — and the control stack's radius is
                # `radius_m + position_sigma`, an estimate that STARTS large and
                # converges. So the policy was permanently planning against the map's
                # least certain moment. Measured over the four two-bin runs of
                # 2026-08-18: every landmark converged to 0.230 m in telemetry while the
                # controller held 0.379-0.472 m for the whole run, which shrank the
                # aperture between two bins from 0.91-0.97 m to 0.45-0.60 m and its
                # angular width, which is what the ray fan has to resolve, by 3.4x.
                # Invisible because an over-large disc produces a completely plausible
                # range vector; it just reports a gap the robot cannot fit through.
                match.radius = radius if matched_by_id else max(match.radius, radius)
                match.last_seen = now
                if match.object_id is None:
                    match.object_id = d.object_id

    def _expire(self, now: float) -> None:
        """Drop obstacles not seen for :attr:`Config.static_obstacle_ttl_s`."""
        ttl = self.cfg.static_obstacle_ttl_s
        self._obstacles = [o for o in self._obstacles if now - o.last_seen <= ttl]

    @staticmethod
    def _ray_circle(px: float, py: float, angle: float, o: _Obstacle) -> float:
        """Distance from ``(px, py)`` along ``angle`` to a disc, or ``inf`` if it misses.

        Zero when the point is INSIDE the disc, which is the conservative reading and the
        one the control stack's own caster uses: reporting the exit distance there would
        tell the policy it has clear space while it is already in the obstacle.
        """
        dx, dy = math.cos(angle), math.sin(angle)
        cx, cy = o.x - px, o.y - py
        proj = cx * dx + cy * dy
        perp2 = cx * cx + cy * cy - proj * proj
        if perp2 > o.radius * o.radius:
            return float("inf")
        half = math.sqrt(max(o.radius * o.radius - perp2, 0.0))
        t0, t1 = proj - half, proj + half
        if t1 < 0:
            return float("inf")
        return max(0.0, t0)

    def _ranges(self, x: float, y: float) -> np.ndarray:
        """The 12-ray fan, in metres, clamped at the horizon.

        The rays are at FIXED angles in the run-local frame and do not turn with the
        robot, because the trained VMAS agent does not rotate — ``state.rot`` stays zero
        for the whole navigation task, so the simulator's rays are in its world frame.
        Confirmed by @spsagar13 in AIDP-567.
        """
        max_r = self.cfg.lidar_range_m
        ranges = np.full(N_RAYS, max_r, dtype=np.float32)
        for i in range(N_RAYS):
            angle = 2 * math.pi * i / N_RAYS
            for o in self._obstacles:
                ranges[i] = min(ranges[i], self._ray_circle(x, y, angle, o))
        return ranges

    def _observation(self, x: float, y: float, vx: float, vy: float, gx: float,
                     gy: float, ranges_m: np.ndarray) -> np.ndarray:
        s = self.cfg.meters_per_vmas_unit
        ranges_vmas = np.clip(ranges_m / s, 0.0, self.cfg.lidar_range_vmas)
        lidar = self.cfg.lidar_range_vmas - ranges_vmas
        return np.concatenate([
            np.asarray([x / s, y / s, vx / s, vy / s, (x - gx) / s, (y - gy) / s],
                       dtype=np.float32),
            lidar.astype(np.float32),
        ])

    def step(self, inp: RobotInput) -> ActionOutput:
        now = time.monotonic()
        sample_time = now if inp.timestamp_s is None else inp.timestamp_s
        age_s = now - sample_time
        if self._origin is None or inp.reset_run:
            self._reset(inp)

        x, y, yaw, vx, vy, gx, gy = self._local_state(inp)
        self._expire(sample_time)
        self._update_obstacles(inp, x, y, yaw, sample_time)
        ranges = self._ranges(x, y)
        obs = self._observation(x, y, vx, vy, gx, gy, ranges)
        self.last_observation = obs
        action = self.actor.act(obs)

        c, s = math.cos(yaw), math.sin(yaw)
        body_x = c * float(action[0]) + s * float(action[1])
        body_y = -s * float(action[0]) + c * float(action[1])
        cmd_vx = float(np.clip(body_x, -1.0, 1.0) * self.cfg.max_vx_mps
                       * self.cfg.command_scale)
        cmd_vy = float(np.clip(body_y, -1.0, 1.0) * self.cfg.max_vy_mps
                       * self.cfg.command_scale)
        cmd_wz = 0.0

        # Precedence, most severe first. The clock check is FIRST because it decides
        # whether `age_s` means anything at all, and the staleness check below is the one
        # it would otherwise silently disable. External hold outranks goal-reached because
        # both stop the robot and the caller should be told which authority did it.
        status = COMMAND
        if age_s < -CLOCK_TOLERANCE_S:
            status = STOP_CLOCK_ERROR
        elif age_s > self.cfg.stale_input_timeout_s:
            status = STOP_STALE_INPUT
        elif inp.external_hold:
            status = STOP_EXTERNAL_HOLD
        elif math.hypot(gx - x, gy - y) <= self.cfg.goal_stop_distance_m:
            status = STOP_GOAL_REACHED
        if status != COMMAND:
            cmd_vx = cmd_vy = cmd_wz = 0.0

        return ActionOutput(
            status=status,
            action_x=float(action[0]),
            action_y=float(action[1]),
            vx_mps=cmd_vx,
            vy_mps=cmd_vy,
            vyaw_radps=cmd_wz,
            age_s=age_s,
        )
