# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One place that turns a telemetry tick into a velocity command via the MAPPO policy.

Three callers need exactly this loop and they must not each have their own copy, because
the whole value of the closed-loop simulation is that **the loop it exercises is the loop
that runs on the robot**:

  * ``closed_loop_sim.py`` — the policy's own actions move a simulated robot (issue #5)
  * ``mappo_shadow.py``    — a live run, policy logged beside the planner, legs untouched
  * ``mappo_drive.py``     — a live run, the policy driving under the planner's veto

``replay_mappo.py`` deliberately does NOT use it: it builds its own two controllers so it
can run a paired ablated control, and keeping it independent means the two agree by
measurement rather than by construction.

## The heading servo

The delivered policy outputs a 2-D force and no yaw, because the trained VMAS agent is
holonomic and never rotates. Sent straight to a Go2 that is a real consequence: the robot
crabs, sideways at 0.20 m/s against 0.35 forward, and — the part that actually matters —
**never turns its head**. Perception is a forward 85-degree cone with no rear view, so a
robot that holds its start heading for the whole run only ever sees the wedge it began
in. Every bearing outside that wedge reports clear, which is the optimistic direction.

:class:`HeadingServo` closes that gap outside the policy. It is invisible to the policy —
the observation carries no yaw, and the rays are fixed in the run-local frame — so this
adds a degree of freedom rather than perturbing the one that was trained.

**It is off unless a caller asks for it, and that is issue #16.** The servo's first law
steered on the direction of the COMMAND, and on 2026-08-17 it saturated ``wz`` and put
the robot into a cubicle panel or a cabinet on three runs out of four. That law is still
here as :data:`TRAVEL`, so the recorded runs stay reproducible; :data:`GOAL` is the one
that is not known to do this, and no robot has yet been driven with it. ``mappo_drive.py``
therefore defaults to no servo at all — see :class:`HeadingServo` for both laws and for
the measurements that separate them.

Needs the policy package's numpy. ``python3 test_mappo_policy.py``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

from mappo_bridge import robot_input
from observation import wrap_pi

#: The vendored policy package.
DEFAULT_PACKAGE = Path(__file__).resolve().parent.parent / "policy"

#: How far ahead the veto checks a proposed command, for
#: :meth:`avoidance.DynamicWindowPlanner.is_feasible`. ``None`` means the planner's own
#: horizon, which is the right answer: shortening it was swept in ``closed_loop_sim.py
#: --veto-horizon`` and cost collisions without even firing less often, because a veto
#: that intervenes late lets the robot get closer, where it then has to intervene more.
VETO_HORIZON_S: float | None = None


def load_policy(package: Path = DEFAULT_PACKAGE):
    """``(MappoController, RobotInput, StationaryObject)`` from the policy package.

    Imported by path rather than installed: the package is a checkpoint plus an adapter
    that the policy owner ships as a directory, and pip has no part in it.
    """
    package = Path(package).expanduser().resolve()
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    try:
        from physical_ai_mappo import MappoController, RobotInput, StationaryObject
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the policy package from {package}: {exc}\n"
            f"pass --package <dir containing physical_ai_mappo.py>") from exc
    return MappoController, RobotInput, StationaryObject


#: Steer on the bearing from the robot to the GOAL, taken in odom. Nothing the policy
#: emits is in this loop, so yawing moves the robot and leaves the reference where it is.
GOAL = "goal"

#: Steer on the direction of the body-frame velocity COMMAND. Issue #16's law. Kept so
#: the 2026-08-17 runs remain reproducible; see :class:`HeadingServo` before selecting it.
TRAVEL = "travel"

SERVO_MODES = (GOAL, TRAVEL)


def goal_bearing_body(x_m: float, y_m: float, yaw_rad: float,
                      goal_x_m: float, goal_y_m: float) -> float:
    """Bearing to the goal in the BODY frame, ``+`` to the left.

    Takes the odom scalars rather than a tick because :meth:`PolicyRunner.step` has
    already validated them — ``mappo_bridge.robot_input`` drops a tick whose pose or goal
    is missing or non-finite, so by the time this is reached every argument is a real
    number. ``replay_mappo._goal_bearing`` and ``mappo_shadow._goal_bearing`` compute the
    same quantity from a raw tick, for logging rather than for control.
    """
    return wrap_pi(math.atan2(goal_y_m - y_m, goal_x_m - x_m) - yaw_rad)


@dataclass(frozen=True)
class HeadingServo:
    """Yaw rate that turns the nose towards a heading the policy does not command.

    Proportional with a deadband, and that is deliberate rather than lazy: the input is
    already smooth (it comes from a saturated tanh, not from a range estimate), and the
    thing a robot with a rolling-shutter camera cannot afford is a yaw rate that changes
    every tick. The deadband is what stops a heading error of a couple of degrees from
    turning into a permanent small twitch.

    ## Two laws, and why :data:`TRAVEL` is not one you should pick

    :data:`TRAVEL` steers on ``atan2(body_vy, body_vx)`` — the direction of the command
    the policy just produced. **``vx`` changes sign, and when it does that reference
    jumps to near 180 degrees.** Over 30 closed-loop episodes the policy's command was
    backwards on 367 of 6123 moving ticks (6.0%), and every one of those 367 is a heading
    error past 90 degrees: roughly once a second, at 10 Hz, this law demands more than a
    quarter turn. :attr:`min_speed_mps` cannot catch it, because it gates on the
    magnitude ``hypot(vx, vy)`` and a backwards command is not a slow one.

    Once ``wz`` saturates it does not recover on its own. ``closed_loop_sim.py`` seed 19
    holds the yaw rate at its ceiling for **16.2 continuous seconds** and turns the robot
    through **179.8 degrees**; on hardware on 2026-08-17 it was 2.8 seconds, 34-54
    degrees, and three collisions (issue #16). What sustains the saturation past the
    first tick is not established here — two plausible accounts, that spinning starves
    the obstacle map and that the anisotropic ``max_vx``/``max_vy`` envelope is the
    coupling, were both tested and neither holds. What IS established is that this law's
    reference is a function of the robot's own yaw and the other law's is not, and that
    the behaviour goes with it.

    :data:`GOAL` steers on the bearing to the goal, taken in odom. Turning the robot
    moves its heading and leaves that bearing alone, so the error falls instead of
    running; it is the quantity ``avoidance.py`` scores as ``heading_cost``, and the
    planner is the controller that has not hit anything. Same 30 episodes:

    | law | longest saturated ``wz`` | peak yaw | collisions |
    | --- | --- | --- | --- |
    | ``TRAVEL`` | 16.2 s | 179.8 deg | 6 / 30 |
    | ``GOAL`` | **1.7 s** | **74.9 deg** | 11 / 30 |

    **The collision column is stated because it does not support the change, and it does
    not settle the question either way.** That arena is an open 3 x 3 m square holding one
    disc; it has no walls, and a wall is what the robot hit. A law that yaws through 180
    degrees cannot be charged for it by a simulator with nothing to yaw into. The
    saturation and excursion columns are what issue #16 defines the defect as, and those
    are the ones this changes.

    Both laws serve the reason a servo exists: the policy commands **no yaw**, so an
    unassisted robot holds its start heading and its 85-degree camera never looks
    anywhere new. :data:`GOAL` still sweeps it, towards where the robot is going.
    """

    #: rad/s per rad of heading error.
    gain: float = 1.2
    #: Ceiling on the commanded yaw rate. Well under the planner's own 0.70 rad/s: this
    #: turn is a convenience, and a fast one smears the frames the policy is being fed.
    max_wz: float = 0.40
    #: Heading errors smaller than this command no yaw at all.
    deadband_rad: float = math.radians(8.0)
    #: Below this commanded speed the robot is not going anywhere, so the servo holds
    #: still rather than turning on the strength of a command that is basically noise.
    #: It gates on SPEED and therefore says nothing about direction — see the class
    #: docstring, where that is the whole of :data:`TRAVEL`'s problem.
    min_speed_mps: float = 0.02
    #: Which heading to close the loop on, :data:`GOAL` or :data:`TRAVEL`.
    mode: str = GOAL

    def __post_init__(self) -> None:
        if self.mode not in SERVO_MODES:
            raise ValueError(f"mode must be one of {SERVO_MODES}, not {self.mode!r}")

    def yaw_rate(self, body_vx: float, body_vy: float,
                 goal_bearing_rad: float | None = None) -> float:
        """Yaw rate, rad/s, for a body-frame velocity command.

        ``goal_bearing_rad`` is :func:`goal_bearing_body`, and is required in
        :data:`GOAL` mode. ``None`` means the caller has no goal this tick, which leaves
        nothing to face: a robot that has lost its goal should not also start turning.
        """
        if math.hypot(body_vx, body_vy) < self.min_speed_mps:
            return 0.0
        if self.mode == GOAL:
            if goal_bearing_rad is None:
                return 0.0
            error = wrap_pi(goal_bearing_rad)
        else:
            error = wrap_pi(math.atan2(body_vy, body_vx))
        if abs(error) < self.deadband_rad:
            return 0.0
        return max(-self.max_wz, min(self.max_wz, self.gain * error))


@dataclass(frozen=True)
class PolicyStep:
    """What the policy did with one tick, and enough context to argue about it later."""

    status: str
    #: Body-frame command, after the heading servo has supplied the yaw the policy does
    #: not. Zero on any non-``COMMAND`` status.
    vx_mps: float
    vy_mps: float
    wz_radps: float
    #: The raw action in the run-local frame, reported even when the command is zeroed —
    #: a held tick with no recorded intent is indistinguishable from a policy that had
    #: nothing to say.
    action_x: float
    action_y: float
    #: Heading, in the BODY frame, that the policy was steering for. ``None`` when the
    #: action is too small to have a direction.
    intent_bearing_rad: float | None
    #: Age of the input by the controller's clock.
    age_s: float
    #: The 18 values the network saw, as a tuple. Kept because an action on its own
    #: cannot be checked: the input that produced it is the half that can be wrong.
    observation: tuple


class PolicyRunner:
    """The MAPPO policy, driven one telemetry tick at a time.

    Stateful, one per robot: the controller holds the run-local frame and the obstacle
    map, and this holds the reset bookkeeping around it.
    """

    def __init__(self, package: Path = DEFAULT_PACKAGE, config: Path | None = None,
                 servo: HeadingServo | None = None):
        controller_cls, self._input_cls, self._object_cls = load_policy(package)
        self.controller = controller_cls(config or Path(package) / "config.json")
        #: ``None`` disables the servo, leaving ``wz`` at the policy's own zero. Off is
        #: the honest default for a SHADOW run, where nothing should imply the recorded
        #: command is what the robot would have done — and since issue #16 it is the
        #: default for the DRIVE path too, which now builds a servo only when an operator
        #: names one. ``closed_loop_sim.PolicyController`` is the exception and reads
        #: ``None`` as "build the default servo", so it cannot currently simulate the
        #: configuration the robot ships with; that is its own to fix.
        self.servo = servo
        self._started = False
        #: Yaw at the last reset. The controller holds this too, privately, and this copy
        #: exists only to turn the run-local action back into a body-frame INTENT for the
        #: log. The command cannot stand in for it: the envelope scales x by 0.35 and y by
        #: 0.20, so a 45-degree intent leaves as a 30-degree command.
        self._origin_yaw = 0.0

    @property
    def config(self):
        return self.controller.cfg

    def reset(self) -> None:
        """Make the next tick the first of a new run, re-fixing the run-local frame."""
        self._started = False

    def step(self, tick: dict, monotonic_s: float | None = None) -> PolicyStep | None:
        """One tick through the bridge and the policy, or ``None`` if it has no goal.

        ``None`` is not an error: the robot searching for its goal is a real part of the
        episode. Handing the policy a goal at the origin instead — which is what a
        zero-filled default would do — makes it drive there.
        """
        reset = not self._started
        kwargs = robot_input(tick, reset_run=reset, monotonic_s=monotonic_s)
        if kwargs is None:
            return None
        if reset:
            self._origin_yaw = kwargs["yaw_rad"]
        self._started = True
        kwargs["stationary_objects"] = [self._object_cls(**o)
                                        for o in kwargs["stationary_objects"]]
        out = self.controller.step(self._input_cls(**kwargs))

        # `vx_mps`/`vy_mps` are already body-frame and already zeroed on a stop status,
        # so the servo sees zero speed and holds — no extra case needed here.
        #
        # The goal bearing is taken from the ODOM pose and goal. Deriving it from
        # anything the policy's action has touched would put the robot's own yaw back on
        # both sides of the loop, which is the whole of issue #16.
        wz = 0.0 if self.servo is None else self.servo.yaw_rate(
            out.vx_mps, out.vy_mps,
            goal_bearing_body(kwargs["x_m"], kwargs["y_m"], kwargs["yaw_rad"],
                              kwargs["goal_x_m"], kwargs["goal_y_m"]))

        # The action rotated out of the run-local frame into the body frame. This mirrors
        # two lines inside the policy package, deliberately: the intent is wanted even on
        # a stop status, when the command it would otherwise be read off has been zeroed.
        yaw_local = wrap_pi(kwargs["yaw_rad"] - self._origin_yaw)
        cos_yaw, sin_yaw = math.cos(yaw_local), math.sin(yaw_local)
        body_x = cos_yaw * out.action_x + sin_yaw * out.action_y
        body_y = -sin_yaw * out.action_x + cos_yaw * out.action_y
        intent = None if (body_x, body_y) == (0.0, 0.0) else math.atan2(body_y, body_x)

        observation = self.controller.last_observation
        return PolicyStep(
            status=out.status,
            vx_mps=out.vx_mps, vy_mps=out.vy_mps, wz_radps=wz,
            action_x=out.action_x, action_y=out.action_y,
            intent_bearing_rad=intent,
            age_s=out.age_s,
            observation=() if observation is None else tuple(float(v)
                                                             for v in observation),
        )


def tick_from_state(t: float, pose: tuple, goal: tuple | None, obstacles: list,
                    measured: tuple = (0.0, 0.0, 0.0),
                    reason: str | None = None, stale: bool = False,
                    peer_link: dict | None = None) -> dict:
    """A telemetry-shaped tick built from live or simulated state.

    The point of routing through the tick shape rather than calling
    :func:`physical_ai_mappo.RobotInput` directly is that the mapping — the frames, the
    static/tracked split, the hold semantics — then happens in ``mappo_bridge`` for the
    simulator and the robot alike. A simulator with its own private mapping tests the
    simulator.

    ``obstacles`` are dicts or anything with ``x``, ``y``, ``radius_m``, ``kind``,
    ``object_id``, ``vx`` and ``vy`` attributes, i.e. ``avoidance.Obstacle``.

    ``peer_link`` carries ``{"lost": bool, "reason": str}`` when a mesh peer source is
    running, and is absent otherwise — which is every recorded run and the whole of
    ``closed_loop_sim``. ``mappo_bridge.external_hold`` is what reads it; it is threaded
    through here rather than handed to the policy directly so that the simulator and the
    robot go on sharing one mapping, which is the reason this function exists at all.
    """
    def _record(obstacle) -> dict:
        if isinstance(obstacle, dict):
            return obstacle
        return {"x": obstacle.x, "y": obstacle.y, "radius_m": obstacle.radius_m,
                "kind": obstacle.kind, "id": obstacle.object_id,
                "vx": obstacle.vx, "vy": obstacle.vy,
                "label": getattr(obstacle, "label", "")}

    return {
        "type": "tick", "t": round(t, 3),
        "pose": {"x": pose[0], "y": pose[1], "yaw": pose[2]},
        "goal": None if goal is None else {"x": goal[0], "y": goal[1]},
        "measured": {"vx": measured[0], "vy": measured[1], "wz": measured[2]},
        "command": None if reason is None else {"reason": reason},
        "obstacles": [_record(o) for o in obstacles],
        "perception": {"stale": stale},
        "peer_link": dict(peer_link or {}),
    }
