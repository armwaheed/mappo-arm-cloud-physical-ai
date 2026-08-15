#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Close the loop on the MAPPO policy before it is allowed to drive legs. Issue #5.

``replay_mappo.py`` is open-loop: the shipped planner drove the path and the policy's
actions were recorded and thrown away, so the policy never met the states its own actions
produce. That pins the mapping and measures the horizon. It says **nothing** about whether
the policy navigates, and a policy that steers correctly for one tick and diverges over
fifty looks identical in it.

This closes the loop: action -> actuator -> pose -> what the camera can now see -> the
next observation, through the same ``mappo_bridge`` mapping the robot uses.

## Three controllers on identical scenarios

Because "the policy arrived 8 times out of 10" is not a result on its own — the question
is whether it beats what already works.

| | |
| --- | --- |
| ``planner`` | the shipped dynamic-window planner. The incumbent, and it walks today. |
| ``policy`` | the MAPPO checkpoint, driving alone. |
| ``supervised`` | the policy driving under the planner's veto — what ``mappo_drive.py`` runs. |

Every one of them is paired with an **ablated control**: the same seed, the same noise
draws, the obstacles removed. That is not decoration. This checkpoint carries a 6-16
degree heading bias with nothing in the scene at all, so an arrival rate with obstacles
means nothing without the arrival rate without them.

## What is modelled, and what is a stated assumption

Every number below is fitted to the recorded run rather than chosen, and the derivation
sits beside the constant it produced.

* **⚠️ A 0.45 actuator gain — the robot delivers under half of what it is told to.** Fitted
  against the POSE, which is ground truth, not against ``measured``. It is the single
  most consequential thing in this file and it was nearly missed: a naive
  ``measured - command`` fit charges the whole shortfall to noise and reports 0.17 m/s of
  jitter instead. The two models differ enormously — with noise-not-gain, a robot parked
  and commanding zero random-walks to a 2.6 m goal, which the first version of this
  simulation duly recorded as the shipped planner "arriving" twice in ten runs.
* **0.07 m/s of residual noise** per axis, after the gain, and none at all on a commanded
  stop. Commands are rate-limited first, by the vendored ``Limits`` accelerations.
* **No velocity LAG, and that is a measurement rather than an omission.** A first-order
  lag fitted to the same run improves the residual only from 0.145 to 0.132 m/s at a 2 s
  time constant — the data does not identify one. Inventing a plausible tau would have
  made this look more principled and been less true. ``--lag`` tests the assumption.
* **Perception latency**, 0.31 s (measured median 0.326 s, p90 0.476 s): an obstacle is
  detected from where the robot was a third of a second ago. Static detections then
  persist, because the odom map is what persists them on the robot.
* **The camera cone**, 85.27 degrees with no rear view. Everything outside it reads clear.
* **The estimator is treated as exact.** Its own error is 0.041 m/s against pose-derived
  velocity, an order below the actuation error, so it is left out.

## "Collision" here means the PLANNING disc overlapped

Judged at the 0.25 m radius issue #5 asks for, which is the radius the live runs planned
with. The Go2 is 0.70 x 0.31 m, so its actual half-width is about 0.155 m and that disc
carries roughly 0.10 m of margin — a run reported at -0.01 m still had about 9 cm of real
air. Read the ``min clear`` column, not just the count: a metre-deep collision and a
one-centimetre graze are the same tally mark and very much not the same event.

Nothing here models a leg, a gait, or the floor. It is a velocity-commanded planar body,
which is the level the control stack itself works at. It cannot tell you the robot will
not fall over, and passing it is a licence to try on hardware, not a substitute for it.

    python3 closed_loop_sim.py                        # the full matrix
    python3 closed_loop_sim.py --seeds 30 --scale 2.5
    python3 closed_loop_sim.py --command-scale 0.3 0.6 1.0   # 0.3 was the delivered value
    python3 closed_loop_sim.py --controller policy --verbose

Needs the policy package's numpy and the vendored planner. ``python3 test_closed_loop_sim.py``
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mappo_policy import (
    DEFAULT_PACKAGE,
    VETO_HORIZON_S,
    HeadingServo,
    PolicyRunner,
    tick_from_state,
)
from observation import DEFAULT_FOV_RAD, DEFAULT_MAX_RANGE_M, wrap_pi

_STACK = Path(__file__).resolve().parent.parent / "robot-stack" / "unitree" / "go2"
sys.path.insert(0, str(_STACK / "visual_nav"))
from avoidance import (  # noqa: E402
    STATIC_HARD_GAP_M,
    STATIC_SOFT_GAP_M,
    DynamicWindowPlanner,
    Limits,
    Obstacle,
    PlannerConfig,
)

#: ⚠️ THE ROBOT DELIVERS LESS THAN HALF OF WHAT IT IS TOLD TO. Least-squares fit of
#: pose-derived body velocity against the command over the recorded run's 116 standing
#: ticks; the integrated displacement agrees (2.09 m travelled against 4.32 m commanded,
#: 0.48) and the per-tick forward median is 0.59. Yaw fits 0.44 separately.
#:
#: The POSE is the ground truth here, not ``measured``: the estimator's own error is only
#: 0.041 m/s against pose-derived velocity, so the shortfall is real motion, not a bad
#: reading. That distinction is why this is a gain and not the 0.17 m/s of "noise" that a
#: naive ``measured - command`` fit reports.
#:
#: One run, tethered, carrying the 3.15 kg D1 arm, on the derated envelope — so this is a
#: property of that configuration rather than of a Go2. ``--gain 1.0`` is the limit case.
#: The consequence is worth doing in your head before the demo: at the delivered
#: ``command_scale`` of 0.30 the top speed on the floor was 0.35 x 0.30 x 0.45 =
#: 0.047 m/s, i.e. 2.8 m in the whole run budget. A later hardware sweep also showed
#: 0.60 is below the Go2 gait floor, so the shipped value is now 1.0. This fitted 0.45
#: gain remains the conservative recorded-run model; ``--gain 0.70`` covers the measured
#: full-command run.
ACTUATOR_GAIN = 0.45

#: Residual after the gain, per axis. The naive figure is 0.137 m/s, which is what you
#: get by charging the gain deficit to noise.
VELOCITY_NOISE_MPS = 0.07

#: Same fit for yaw: gain 0.44, residual 0.10 rad/s.
YAW_GAIN = 0.45
YAW_NOISE_RADPS = 0.10

#: Measured perception latency, median over the recorded run. p90 is 0.476 s.
PERCEPTION_LATENCY_S = 0.31

#: Deflection past which a tick counts as "swerving", for the reversal count. Matches
#: ``replay_mappo.py`` so the closed-loop and open-loop numbers are comparable.
SWERVE_DEG = 10.0


@dataclass(frozen=True)
class SimObstacle:
    """A physical disc. ``radius_m`` is the real one: the detector reports it unchanged.

    The live stack inflates a radius for position uncertainty and then plans against the
    inflated one. Modelling that here would need an uncertainty model this simulation does
    not have, so it is left out and collision is measured against the true disc — the
    pessimistic direction for the controller under test.
    """

    x: float
    y: float
    radius_m: float
    kind: str = "static"
    object_id: str | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    start: tuple            # (x, y, yaw)
    goal: tuple             # (x, y)
    obstacles: tuple = ()


@dataclass(frozen=True)
class SimConfig:
    control_hz: float = 10.0
    max_run_s: float = 60.0
    #: Reaching the goal. Tighter than the stack's 0.8 m arrive tolerance, which was sized
    #: for a goal found by monocular detection rather than for a known point.
    arrive_tolerance_m: float = 0.30
    #: The radius #5 asks for collisions to be judged at — the one the live runs planned
    #: with, not the trained 0.10 VMAS radius.
    robot_radius_m: float = 0.25
    perception_latency_s: float = PERCEPTION_LATENCY_S
    actuator_gain: float = ACTUATOR_GAIN
    velocity_noise_mps: float = VELOCITY_NOISE_MPS
    yaw_gain: float = YAW_GAIN
    yaw_noise_radps: float = YAW_NOISE_RADPS
    #: First-order actuator lag, seconds. Zero by default; see the module docstring for
    #: why the recorded run does not identify one.
    velocity_lag_s: float = 0.0
    fov_rad: float = DEFAULT_FOV_RAD
    detect_range_m: float = DEFAULT_MAX_RANGE_M
    limits: Limits = field(default_factory=Limits)


@dataclass(frozen=True)
class RunResult:
    controller: str
    scenario: str
    seed: int
    ablated: bool
    outcome: str            # "arrived" | "collision" | "timeout"
    ticks: int
    elapsed_s: float
    path_m: float
    #: Smallest gap between the robot's disc and any obstacle's, over the whole run.
    #: Negative means they overlapped.
    min_clearance_m: float
    final_distance_m: float
    reversals: int
    swerving_ticks: int


# ── Controllers ─────────────────────────────────────────────────────────────
class PlannerController:
    """The shipped dynamic-window planner, as the live runs are flown today."""

    name = "planner"

    def __init__(self, config: SimConfig):
        self.planner = DynamicWindowPlanner(
            limits=config.limits,
            config=PlannerConfig(robot_radius_m=config.robot_radius_m))
        self._last = (0.0, 0.0, 0.0)
        self._reason = "goal"
        self._dt = 1.0 / config.control_hz

    def reset(self) -> None:
        self._last, self._reason = (0.0, 0.0, 0.0), "goal"

    def command(self, t, pose, goal, obstacles, measured):
        planned = self.planner.plan(pose, goal, self._last,
                                    [_to_planner(o) for o in obstacles],
                                    control_dt=self._dt, last_reason=self._reason)
        self._reason = planned.reason
        self._last = (planned.vx, planned.vy, planned.wz)
        return self._last, planned.reason, None


class PolicyController:
    """The MAPPO checkpoint driving on its own, plus the heading servo."""

    name = "policy"

    def __init__(self, config: SimConfig, package: Path = DEFAULT_PACKAGE,
                 policy_config: Path | None = None, servo: HeadingServo | None = None):
        self._runner = PolicyRunner(
            package, policy_config,
            servo=HeadingServo(max_wz=config.limits.max_wz) if servo is None else servo)

    def reset(self) -> None:
        self._runner.reset()

    def command(self, t, pose, goal, obstacles, measured):
        tick = tick_from_state(t, pose, goal, [_to_record(o) for o in obstacles],
                               measured=measured)
        # `monotonic_s=None` lets the controller stamp its own clock. A simulated `t` is
        # not a `time.monotonic()` reading, and passing one in would trip exactly the
        # guard this repository added — correctly.
        step = self._runner.step(tick)
        if step is None:
            return (0.0, 0.0, 0.0), "no-goal", None
        return ((step.vx_mps, step.vy_mps, step.wz_radps),
                step.status.lower(), step.intent_bearing_rad)


class SupervisedController(PolicyController):
    """The policy drives; the planner keeps a veto. What ``mappo_drive.py`` runs.

    The veto is the planner's OWN feasibility test applied to the policy's command:
    roll the proposed velocity forward over the 2.5 s horizon and require it to keep every
    obstacle's hard gap. If it does not, the planner's own command is used instead.

    That is deliberately not "forward the planner's ``hold``". The planner holds for the
    BIN as well as for people — it treats a static obstacle as one with zero velocity and
    has no special case — so forwarding every hold would zero the policy in the one scene
    it exists to solve. Rolling out the actual command distinguishes "the planner would
    rather wait" from "this specific command ends inside the bin", and only the second is
    the policy's problem to be saved from.
    """

    name = "supervised"

    def __init__(self, config: SimConfig, veto_horizon_s: float | None = VETO_HORIZON_S,
                 **kwargs):
        super().__init__(config, **kwargs)
        self._fallback = PlannerController(config)
        self._veto_horizon_s = veto_horizon_s
        #: How often the veto actually took over, counted only over ticks where there was
        #: something to veto. Reported, because "supervised did not collide" means nothing
        #: if the answer is that the planner drove the whole run — the footage would look
        #: like a policy demo and be one of the incumbent. Ticks with an empty scene are
        #: excluded from the denominator: they are not the veto standing aside, they are
        #: nothing to stand aside from, and counting them halves the rate for free.
        self.vetoed = 0
        self.driven = 0

    def reset(self) -> None:
        super().reset()
        self._fallback.reset()

    def command(self, t, pose, goal, obstacles, measured):
        proposed, reason, intent = super().command(t, pose, goal, obstacles, measured)
        # The fallback planner is stepped every tick whether or not it is used, so that
        # its acceleration window and reason hysteresis are continuous when it IS used.
        # A planner handed a cold start mid-run plans from a standstill it is not in.
        backup, backup_reason, _ = self._fallback.command(t, pose, goal, obstacles,
                                                          measured)
        if self._fallback.planner.is_feasible(pose, proposed,
                                              [_to_planner(o) for o in obstacles],
                                              horizon_s=self._veto_horizon_s):
            self.driven += bool(obstacles)
            return proposed, reason, intent
        self.vetoed += 1
        return backup, f"veto-{backup_reason}", intent


CONTROLLERS = {c.name: c for c in (PlannerController, PolicyController,
                                   SupervisedController)}


def _to_planner(obstacle: SimObstacle) -> Obstacle:
    """A planner obstacle, carrying the per-kind gaps ``visual_nav`` gives it.

    Getting this wrong is not subtle and it was wrong first time round: without the
    override a mapped bin inherits a PERSON's 1.20 m soft gap, the clearance term
    outweighs the goal term from 1.4 m away, and the planner's best option becomes
    standing still. It never moved off the start line. The vendored comment says as much
    — "applied to a 0.30 m bin it is actively harmful" — and the live stack sets these
    two fields for exactly this reason.
    """
    static = obstacle.kind == "static"
    return Obstacle(x=obstacle.x, y=obstacle.y, vx=0.0, vy=0.0,
                    radius_m=obstacle.radius_m, kind=obstacle.kind,
                    object_id=obstacle.object_id,
                    soft_gap_m=STATIC_SOFT_GAP_M if static else None,
                    hard_gap_m=STATIC_HARD_GAP_M if static else None)


def _to_record(obstacle: SimObstacle) -> dict:
    return {"x": obstacle.x, "y": obstacle.y, "radius_m": obstacle.radius_m,
            "kind": obstacle.kind, "id": obstacle.object_id, "vx": 0.0, "vy": 0.0}


# ── The world ───────────────────────────────────────────────────────────────
def _visible(pose: tuple, obstacle: SimObstacle, config: SimConfig) -> bool:
    """Whether the camera can see this obstacle from ``pose``.

    Bearing to the obstacle's CENTRE, which is slightly pessimistic at the edge of the
    cone — a disc whose centre is just outside it may still be partly in frame. The
    detector needs a whole silhouette to range from, so the pessimistic reading is the
    right one.
    """
    dx, dy = obstacle.x - pose[0], obstacle.y - pose[1]
    distance = math.hypot(dx, dy)
    if distance > config.detect_range_m:
        return False
    return abs(wrap_pi(math.atan2(dy, dx) - pose[2])) <= config.fov_rad / 2.0


def run_once(scenario: Scenario, controller, config: SimConfig, seed: int,
             ablated: bool = False, verbose: bool = False) -> RunResult:
    """One episode. The controller is reset, so a caller may reuse it across seeds."""
    controller.reset()
    rng = random.Random(seed)
    dt = 1.0 / config.control_hz
    obstacles = () if ablated else scenario.obstacles

    pose = scenario.start
    velocity = (0.0, 0.0, 0.0)
    history: list = [(0.0, pose)]        # (t, pose), for the perception delay
    detected: set = set()                # the odom map: what has been seen at all
    path_m = 0.0
    min_clearance = float("inf")
    deflections: list = []
    outcome = "timeout"
    step = 0

    for step in range(int(config.max_run_s * config.control_hz)):
        t = step * dt

        # What perception can offer: what was in frame `perception_latency_s` ago. A
        # static detection then persists, because the odom map on the robot persists it.
        delayed_pose = _pose_at(history, t - config.perception_latency_s)
        for obstacle in obstacles:
            if obstacle.kind == "static" and obstacle in detected:
                continue
            if _visible(delayed_pose, obstacle, config):
                detected.add(obstacle)
            elif obstacle.kind != "static":
                detected.discard(obstacle)

        command, reason, intent = controller.command(
            t, pose, scenario.goal, sorted(detected, key=lambda o: (o.x, o.y)),
            velocity)

        if intent is not None:
            goal_bearing = wrap_pi(math.atan2(scenario.goal[1] - pose[1],
                                              scenario.goal[0] - pose[0]) - pose[2])
            deflections.append(math.degrees(wrap_pi(intent - goal_bearing)))

        velocity = _actuate(velocity, command, config, dt, rng)
        moved = math.hypot(velocity[0], velocity[1]) * dt
        path_m += moved
        pose = (pose[0] + (velocity[0] * math.cos(pose[2])
                           - velocity[1] * math.sin(pose[2])) * dt,
                pose[1] + (velocity[0] * math.sin(pose[2])
                           + velocity[1] * math.cos(pose[2])) * dt,
                wrap_pi(pose[2] + velocity[2] * dt))
        history.append((t + dt, pose))

        clearance = min((math.hypot(o.x - pose[0], o.y - pose[1]) - o.radius_m
                         - config.robot_radius_m for o in obstacles),
                        default=float("inf"))
        min_clearance = min(min_clearance, clearance)
        distance = math.hypot(scenario.goal[0] - pose[0], scenario.goal[1] - pose[1])
        if verbose:
            print(f"    t={t:5.2f} pose=({pose[0]:+.2f},{pose[1]:+.2f},"
                  f"{math.degrees(pose[2]):+6.1f}deg) v=({velocity[0]:+.2f},"
                  f"{velocity[1]:+.2f},{velocity[2]:+.2f}) {reason:<18} "
                  f"goal={distance:.2f}m clear={clearance:+.2f}m")

        if clearance < 0.0:
            outcome = "collision"
            break
        if distance <= config.arrive_tolerance_m:
            outcome = "arrived"
            break

    swerves = [d for d in deflections if abs(d) > SWERVE_DEG]
    return RunResult(
        controller=controller.name, scenario=scenario.name, seed=seed, ablated=ablated,
        outcome=outcome, ticks=step + 1, elapsed_s=(step + 1) * dt, path_m=path_m,
        min_clearance_m=min_clearance,
        final_distance_m=math.hypot(scenario.goal[0] - pose[0],
                                    scenario.goal[1] - pose[1]),
        reversals=sum(1 for a, b in zip(swerves, swerves[1:]) if a * b < 0.0),
        swerving_ticks=len(swerves))


def _pose_at(history: list, when: float) -> tuple:
    """The pose at ``when``, or the oldest one held. Newest-first scan: the wanted sample
    is always near the end, so this is O(latency / dt) rather than O(run length)."""
    for t, pose in reversed(history):
        if t <= when:
            return pose
    return history[0][1]


def _actuate(velocity: tuple, command: tuple, config: SimConfig, dt: float,
             rng: random.Random) -> tuple:
    """Achieved body velocity: gain, acceleration limit, optional lag, then noise.

    The gain is applied to the COMMAND rather than to the achieved velocity, so a
    commanded stop is still a stop. That matters more than it looks: the first version
    applied a flat 0.17 m/s of noise to the achieved velocity, which turned a robot
    standing still into a random walk — the shipped planner, parked and commanding zero,
    "arrived" at a 2.6 m goal twice in ten runs by drifting there.

    The RNG is drawn from on EVERY tick, including when the command is zero, so an
    ablated run and its paired live run see the same numbers on the same step. Drawing
    only when moving would make the pairing depend on the trajectory, which is the thing
    being compared.
    """
    limits = config.limits
    gains = (config.actuator_gain, config.actuator_gain, config.yaw_gain)
    accels = (limits.accel_x, limits.accel_y, limits.accel_wz)
    noises = [rng.gauss(0.0, config.velocity_noise_mps),
              rng.gauss(0.0, config.velocity_noise_mps),
              rng.gauss(0.0, config.yaw_noise_radps)]

    out = []
    for value, target, gain, accel, noise in zip(velocity, command, gains, accels,
                                                 noises):
        target = target * gain
        if config.velocity_lag_s > 0.0:
            target = value + (target - value) * min(1.0, dt / config.velocity_lag_s)
        step = accel * dt
        moved = max(value - step, min(value + step, target))
        # No noise on a genuine stop. A robot told to hold does not wander off.
        out.append(moved if target == 0.0 and moved == 0.0 else moved + noise)
    return (max(-limits.max_vx, min(limits.max_vx, out[0])),
            max(-limits.max_vy, min(limits.max_vy, out[1])),
            max(-limits.max_wz, min(limits.max_wz, out[2])))


# ── Scenarios ───────────────────────────────────────────────────────────────
def scenarios(rng: random.Random, count: int) -> list:
    """``count`` randomised versions of the staged scene, plus the recorded one.

    The obstacle sits roughly on the straight line to the goal and is jittered rather
    than placed, because a prop set chosen in advance proves nothing about a prop
    somebody else puts down. The arena is the 3 x 3 m demo square.
    """
    out = [Scenario("recorded", start=(0.0, 0.0, 0.0), goal=(2.6, 0.0),
                    obstacles=(SimObstacle(1.3, 0.0, 0.23, "static", "landmark-1"),))]
    for index in range(count - 1):
        heading = rng.uniform(-0.3, 0.3)
        distance = rng.uniform(2.2, 2.9)
        goal = (distance * math.cos(heading), distance * math.sin(heading))
        along = rng.uniform(0.40, 0.65)
        across = rng.uniform(-0.35, 0.35)
        out.append(Scenario(
            f"jittered-{index}", start=(0.0, 0.0, 0.0), goal=goal,
            obstacles=(SimObstacle(goal[0] * along - across * math.sin(heading),
                                   goal[1] * along + across * math.cos(heading),
                                   rng.uniform(0.18, 0.30), "static", "landmark-1"),)))
    return out


def summarise(results: list) -> None:
    """One row per controller, live and ablated side by side."""
    print()
    print(f"{'controller':<12} {'scene':<9} {'arrived':>9} {'collided':>9} "
          f"{'timeout':>8} {'path':>7} {'min clear':>10} {'reversals':>10}")
    for name in CONTROLLERS:
        for ablated in (False, True):
            rows = [r for r in results if r.controller == name and r.ablated == ablated]
            if not rows:
                continue
            arrived = [r for r in rows if r.outcome == "arrived"]
            clearances = [r.min_clearance_m for r in rows
                          if math.isfinite(r.min_clearance_m)]
            print(f"{name:<12} {'ablated' if ablated else 'with obs':<9} "
                  f"{len(arrived):>4}/{len(rows):<4} "
                  f"{sum(1 for r in rows if r.outcome == 'collision'):>9} "
                  f"{sum(1 for r in rows if r.outcome == 'timeout'):>8} "
                  f"{(sum(r.path_m for r in arrived) / len(arrived) if arrived else 0):>6.2f}m "
                  f"{(min(clearances) if clearances else float('nan')):>+9.2f}m "
                  f"{sum(r.reversals for r in rows):>4} / "
                  f"{sum(r.swerving_ticks for r in rows):<4}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10,
                        help="episodes per controller per scale (issue #5 asks for >= 10)")
    parser.add_argument("--scale", type=float, nargs="+", default=[1.5, 2.5],
                        metavar="M_PER_UNIT",
                        help="meters_per_vmas_unit values to report (>= 2, per issue #5)")
    parser.add_argument("--controller", choices=sorted(CONTROLLERS), nargs="+",
                        default=sorted(CONTROLLERS))
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--lag", type=float, default=0.0, metavar="SECONDS",
                        help="first-order actuator lag; the recorded run does not "
                             "identify one, so this is for testing the assumption")
    parser.add_argument("--gain", type=float, default=ACTUATOR_GAIN,
                        help=f"fraction of the commanded velocity the robot delivers "
                             f"(measured {ACTUATOR_GAIN}; 1.0 is the limit case)")
    parser.add_argument("--noise", type=float, default=VELOCITY_NOISE_MPS,
                        metavar="M_PER_S", help="per-axis actuation noise after the gain")
    parser.add_argument("--command-scale", type=float, nargs="+", default=[1.0],
                        metavar="FRACTION",
                        help="policy command_scale values to report. Defaults to the "
                             "shipped 1.0; 0.6 commands 0.21 m/s, which failed to "
                             "sustain this Go2's gait in five hardware runs")
    parser.add_argument("--veto-horizon", type=float, nargs="+",
                        default=[PlannerConfig().horizon_s], metavar="SECONDS",
                        help="how far ahead the veto checks a proposed command. Defaults "
                             "to the planner's own horizon, which the sweep says is the "
                             "right answer — shorter costs collisions and does not even "
                             "veto less often")
    parser.add_argument("--verbose", action="store_true", help="print every tick")
    args = parser.parse_args(argv)

    from replay_mappo import derived_config

    config = SimConfig(velocity_lag_s=args.lag, actuator_gain=args.gain,
                       yaw_gain=args.gain, velocity_noise_mps=args.noise)
    scenes = scenarios(random.Random(20260813), args.seeds)
    reachable = (config.limits.max_vx * args.gain * config.max_run_s)
    failures = 0
    for scale, command_scale, veto_horizon in itertools.product(
            args.scale, args.command_scale, args.veto_horizon):
        print()
        print("=" * 78)
        print(f"meters_per_vmas_unit {scale}   horizon {0.35 * scale:.3f} m   "
              f"trained agent radius {0.10 * scale:.3f} m")
        # Printed every time because it is the number that explained most of the
        # timeouts: a run budget the robot cannot physically cross reads as a
        # navigation failure, and there is nothing in a summary table to say so.
        print(f"command_scale {command_scale}   veto horizon {veto_horizon} s   "
              f"top speed on the floor "
              f"{config.limits.max_vx * command_scale * args.gain:.3f} m/s   "
              f"reachable in {config.max_run_s:.0f} s: "
              f"{reachable * command_scale:.2f} m")
        print("=" * 78)
        with derived_config(args.package / "config.json",
                            meters_per_vmas_unit=scale,
                            command_scale=command_scale) as policy_config:
            results = []
            built: dict = {}
            for name in args.controller:
                factory = CONTROLLERS[name]
                kwargs: dict = {}
                if factory is not PlannerController:
                    kwargs = {"package": args.package, "policy_config": policy_config}
                if factory is SupervisedController:
                    kwargs["veto_horizon_s"] = veto_horizon
                controller = built[name] = factory(config, **kwargs)
                for seed, scene in enumerate(scenes):
                    for ablated in (False, True):
                        show = args.verbose and not ablated
                        if show:
                            print(f"  {name} / {scene.name}")
                        results.append(run_once(scene, controller, config, seed,
                                                ablated=ablated, verbose=show))
            summarise(results)
            for name, controller in built.items():
                if isinstance(controller, SupervisedController):
                    total = controller.driven + controller.vetoed
                    share = 100.0 * controller.vetoed / total if total else 0.0
                    print(f"\n  {name}: the veto took over on {controller.vetoed} of "
                          f"{total} ticks that had an obstacle ({share:.0f}%). A high "
                          f"number means the PLANNER drove the part of the run the demo "
                          f"exists to show.")
            failures += sum(1 for r in results if r.outcome == "collision")
    print()
    print(f"{failures} collision(s) across the whole matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
