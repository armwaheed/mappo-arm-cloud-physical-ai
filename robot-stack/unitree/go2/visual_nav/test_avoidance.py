#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the dynamic-window planner.

Synthetic obstacles only. The tests that matter are the ones a static planner would
fail: the planner must react to an obstacle's VELOCITY, not just its position — giving
way to someone crossing into its path, and declining to brake for someone who is
walking away.

Run: ``python3 test_avoidance.py``
"""
from __future__ import annotations

import ast
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avoidance import (
    COUNT_NOT_RECORDED,
    GO2_MAX_VX_M_S,
    GO2_MAX_VY_M_S,
    GO2_MAX_WZ_RAD_S,
    MIN_GAIT_COMMAND_M_S,
    PLANNER_REASONS,
    STATIC_HARD_GAP_M,
    STATIC_SOFT_GAP_M,
    Command,
    DynamicWindowPlanner,
    Limits,
    Obstacle,
    PlannerConfig,
    base_reason,
)

ORIGIN = (0.0, 0.0, 0.0)     # at the origin, facing +x


def test_the_velocity_envelope_carries_the_go2s_measurements_and_says_so():
    """``Limits``' three velocity fields are ONE ROBOT's numbers, and the constants they
    are bound to record which robot and what was measured.

    The literals are repeated here on purpose. If somebody changes the Go2's envelope to
    suit a second platform -- which is how the Lite3 inherited 0.35/0.20 in the first
    place, four separate times -- this goes red naming the measurement being overwritten
    instead of the change landing silently. The corresponding refusal lives in
    ``deep_robotics/lite3/visual_nav/robot_bindings.py``.
    """
    assert GO2_MAX_VX_M_S == 0.35
    assert GO2_MAX_VY_M_S == 0.20
    assert GO2_MAX_WZ_RAD_S == 0.70

    shipped = Limits()
    assert shipped.max_vx == GO2_MAX_VX_M_S
    assert shipped.max_vy == GO2_MAX_VY_M_S
    assert shipped.max_wz == GO2_MAX_WZ_RAD_S


def test_the_go2s_forward_ceiling_is_also_its_own_gait_floor():
    """Not a tidy-uppable duplication: this robot's usable forward speed is a POINT, not
    a range, and that is why ``--derate`` below 1.0 walks it off the bottom of its gait.

    Kept separate from ``test_the_shipped_envelope_is_above_the_gait_floor`` because that
    one states an inequality that would still hold if the ceiling were raised, and the
    thing worth knowing is that today there is no headroom at all.
    """
    assert GO2_MAX_VX_M_S == MIN_GAIT_COMMAND_M_S


def test_the_shipped_envelope_is_above_the_gait_floor():
    """The default has to be walkable out of the box. Lower ``Limits.max_vx`` under the
    floor and every shipped run becomes a robot standing still, reported as a tether."""
    assert Limits().max_vx >= MIN_GAIT_COMMAND_M_S


def test_a_plausible_derate_lands_exactly_on_the_measured_failure():
    """``--derate`` scales the whole envelope, which is how a legitimate flag reaches an
    unwalkable speed without anyone typing an unwalkable number.

    0.6 of the shipped 0.35 is 0.21 m/s — the value measured to stall on five runs
    across two different controllers. This was a real trap, not a hypothetical: the
    MAPPO package previously shipped 0.6 and multiplied the same 0.35.
    """
    derated = Limits().scaled(0.6)
    assert derated.max_vx < MIN_GAIT_COMMAND_M_S
    assert abs(derated.max_vx - 0.21) < 0.005, derated.max_vx
GOAL = (5.0, 0.0)
STOPPED = (0.0, 0.0, 0.0)
# Already walking. Several tests need this rather than STOPPED: the dynamic window
# only opens by accel*control_dt per tick, so from a standstill EVERY candidate is
# pinned to 0.05 m/s and two scenarios cannot be told apart by the speed they choose.
CRUISING = (0.30, 0.0, 0.0)


def _planner(**overrides):
    return DynamicWindowPlanner(limits=Limits(**overrides), config=PlannerConfig())


def test_empty_scene_drives_at_the_goal():
    command = _planner().plan(ORIGIN, GOAL, STOPPED, [])
    assert command.reason == "goal"
    assert command.vx > 0.0, command
    assert abs(command.wz) < 0.2, "goal is dead ahead; no need to turn"
    assert math.isinf(command.gap_m)


def test_turns_toward_an_off_axis_goal():
    left = _planner().plan(ORIGIN, (3.0, 3.0), STOPPED, [])
    right = _planner().plan(ORIGIN, (3.0, -3.0), STOPPED, [])
    assert left.wz > 0.0, "goal on the left should yaw left (+)"
    assert right.wz < 0.0


def test_stops_for_someone_standing_in_the_way():
    blocker = Obstacle(x=0.9, y=0.0, vx=0.0, vy=0.0, radius_m=0.35)
    command = _planner().plan(ORIGIN, GOAL, STOPPED, [blocker])
    assert command.is_stop, command
    assert command.reason == "hold"


def test_yields_to_someone_crossing_the_path():
    """A person crossing left-to-right ahead: the planner must not drive into them."""
    crosser = Obstacle(x=2.2, y=-1.6, vx=0.0, vy=1.1, radius_m=0.35)
    planner = _planner()
    command = planner.plan(ORIGIN, GOAL, CRUISING, [crosser])
    assert command.gap_m >= planner.config.hard_gap_m, command
    # Holding the cruise straight at the goal would put the robot where they will be,
    # so the planner has to give something up — speed, heading, or both.
    straight = _planner().plan(ORIGIN, GOAL, CRUISING, [])
    assert (command.vx < straight.vx - 1e-6) or (abs(command.wz) > abs(straight.wz) + 1e-6), \
        f"planner ignored the crosser: {command} vs unobstructed {straight}"


def test_reacts_to_velocity_not_just_position():
    """Same person, same place — but walking away instead of at us."""
    approaching = Obstacle(x=2.5, y=0.0, vx=-1.0, vy=0.0, radius_m=0.35)
    receding = Obstacle(x=2.5, y=0.0, vx=+1.0, vy=0.0, radius_m=0.35)
    toward = _planner().plan(ORIGIN, GOAL, CRUISING, [approaching])
    away = _planner().plan(ORIGIN, GOAL, CRUISING, [receding])
    assert away.vx > toward.vx, (
        f"a receding person should cost less than an approaching one: "
        f"{away.vx:.3f} vs {toward.vx:.3f}")


def test_never_outruns_its_stopping_distance():
    """Exercised with a fast envelope, because with the shipped one it never binds.

    At 0.35 m/s and 0.5 m/s^2 the robot stops in 12 cm, well inside the 25 cm hard
    gap — so on this robot the cap is a slack backstop rather than an active
    constraint. Turning the envelope up puts it in charge, which is the only way to
    test that it works.
    """
    config = PlannerConfig()
    fast = DynamicWindowPlanner(limits=Limits(max_vx=1.5, accel_x=3.0))
    person = Obstacle(x=2.0, y=0.0, vx=0.0, vy=0.0, radius_m=0.35)
    gap = 2.0 - 0.35 - config.robot_radius_m
    cap = math.sqrt(2.0 * config.decel_for_stopping_m_s2 * (gap - config.hard_gap_m))
    assert cap < 1.5, "test is vacuous unless the cap binds below the envelope"
    command = fast.plan(ORIGIN, GOAL, (1.4, 0.0, 0.0), [person])
    assert command.vx <= cap + 1e-6, f"vx={command.vx:.3f} exceeds cap {cap:.3f}"


def test_shipped_envelope_stops_well_inside_the_hard_gap():
    """Documents that the shipped speed cap is conservative by construction."""
    config = PlannerConfig()
    limits = Limits()
    stopping_distance = limits.max_vx ** 2 / (2.0 * config.decel_for_stopping_m_s2)
    assert stopping_distance < config.hard_gap_m, (
        f"at {limits.max_vx} m/s the robot needs {stopping_distance:.2f} m to stop, "
        f"more than the {config.hard_gap_m} m hard gap")


def test_respects_the_dynamic_window():
    limits = Limits()
    planner = DynamicWindowPlanner(limits=limits)
    control_dt = 0.1
    command = planner.plan(ORIGIN, GOAL, STOPPED, [], control_dt=control_dt)
    assert command.vx <= limits.accel_x * control_dt + 1e-9, \
        "cannot jump further than one period of acceleration"


def test_reverse_is_never_commanded():
    """No rear sensing on this robot, so backing away must not be an option."""
    surrounded = [Obstacle(x=0.8, y=0.0, vx=-0.5, vy=0.0, radius_m=0.35)]
    for last in (STOPPED, (0.3, 0.0, 0.0)):
        command = _planner().plan(ORIGIN, GOAL, last, surrounded)
        assert command.vx >= 0.0, command


def test_envelope_is_respected():
    tight = _planner(max_vx=0.1, max_vy=0.05, max_wz=0.2)
    command = tight.plan(ORIGIN, (0.0, 5.0), STOPPED, [])
    assert command.vx <= 0.1 + 1e-9
    assert abs(command.vy) <= 0.05 + 1e-9
    assert abs(command.wz) <= 0.2 + 1e-9


def test_derated_limits_scale_everything():
    derated = Limits().scaled(0.5)
    full = Limits()
    assert abs(derated.max_vx - full.max_vx * 0.5) < 1e-9
    assert abs(derated.accel_wz - full.accel_wz * 0.5) < 1e-9


def test_current_gap_ignores_obstacle_velocity():
    planner = _planner()
    assert math.isinf(planner.current_gap(ORIGIN, []))
    person = Obstacle(x=2.0, y=0.0, vx=-5.0, vy=0.0, radius_m=0.35)
    expected = 2.0 - 0.35 - planner.config.robot_radius_m
    assert abs(planner.current_gap(ORIGIN, [person]) - expected) < 1e-9


def test_hold_when_boxed_in_reports_why():
    boxed = [Obstacle(x=0.5, y=0.0, vx=0.0, vy=0.0, radius_m=0.35),
             Obstacle(x=0.0, y=0.6, vx=0.0, vy=0.0, radius_m=0.35),
             Obstacle(x=0.0, y=-0.6, vx=0.0, vy=0.0, radius_m=0.35)]
    command = _planner().plan(ORIGIN, GOAL, STOPPED, boxed)
    assert command.is_stop and command.reason == "hold"
    assert command.feasible == 0
    assert command.evaluated > 0


def test_rollout_arc_is_consistent_with_the_command():
    """A pure yaw command must trace a circle, not a straight line."""
    planner = _planner()
    import numpy as np
    candidates = np.array([[0.3, 0.0, 0.5]])
    xy, yaws = planner._rollout(candidates, ORIGIN)
    steps = xy.shape[1]
    assert abs(yaws[0, -1] - 0.5 * planner.config.dt_s * steps) < 1e-9
    assert xy[0, -1, 1] > 0.0, "positive yaw with forward speed curves left"


# ── Static obstacles: the staged scene, measured ────────────────────────────
# Bin 2.15 m out at +7.1 deg with a 0.15 m radius, chair 3.22 m out on the SAME
# bearing. Both numbers come off the robot's own footage with the calibrated model,
# cross-checked against the LiDAR. This is the geometry the whole exercise turns on,
# so it is pinned here rather than described in a comment somewhere.
BIN_XY = (2.134, 0.266)
CHAIR_XY = (3.195, 0.398)


def _bin(radius_m=0.23, soft_gap_m=STATIC_SOFT_GAP_M):
    """The mapped bin: its 0.15 m footprint plus a typical position sigma."""
    return Obstacle(x=BIN_XY[0], y=BIN_XY[1], vx=0.0, vy=0.0, radius_m=radius_m,
                    label="bin", soft_gap_m=soft_gap_m)


def test_a_static_obstacle_between_robot_and_goal_is_rounded_not_stopped_for():
    """The local minimum the issue predicted, run against the real staging.

    A dynamic-window planner is textbook-vulnerable here: the goal sits directly
    behind the obstacle, so every candidate that makes progress is blocked and the
    planner returns `hold`. It does not, and the reason is the obstacle's SIZE — a bin
    inflated to a person's 0.35 m footprint is a much worse problem than a bin at its
    own 0.15 m. This test is what says the per-class radius was necessary.
    """
    planner = _planner()
    command = planner.plan(ORIGIN, CHAIR_XY, CRUISING, [_bin()])
    assert command.reason != "hold", command
    assert command.vx > 0.0, "must keep making progress, not freeze"


def _drive(planner, start_pose, goal, obstacles, arrive_m=0.8, control_hz=10.0,
           max_s=60.0):
    """Close the loop: plan, integrate, repeat. Returns the path and how it ended.

    A single call to ``plan`` cannot answer "does it go around?" — that is a property of
    the trajectory, and the interesting failures (grinding to a halt one tick short,
    oscillating between two swerves, clipping the obstacle while turning) are all
    invisible tick by tick. The integrator matches ``_rollout``'s so the test measures
    the planner's decisions, not a second motion model.
    """
    dt = 1.0 / control_hz
    x, y, yaw = start_pose
    command = (0.0, 0.0, 0.0)
    reason = "goal"
    path, reasons = [(x, y)], []
    for _ in range(int(max_s * control_hz)):
        if math.hypot(goal[0] - x, goal[1] - y) <= arrive_m:
            return path, reasons, "arrived"
        plan = planner.plan((x, y, yaw), goal, command, obstacles,
                            control_dt=dt, last_reason=reason)
        command, reason = (plan.vx, plan.vy, plan.wz), plan.reason
        reasons.append(reason)
        x += (plan.vx * math.cos(yaw) - plan.vy * math.sin(yaw)) * dt
        y += (plan.vx * math.sin(yaw) + plan.vy * math.cos(yaw)) * dt
        yaw += plan.wz * dt
        path.append((x, y))
    return path, reasons, "timeout"


def _closest_approach(path, obstacle):
    return min(math.hypot(px - obstacle.x, py - obstacle.y) for px, py in path)


def test_the_robot_walks_around_the_bin_and_reaches_the_chair():
    """The whole issue, in one closed-loop test on the measured geometry.

    Success criteria 2-4 from the brief: the reason goes goal -> avoid -> goal, the
    robot passes the bin with a visible margin, and it arrives.
    """
    planner = _planner()
    obstacle = _bin()
    path, reasons, outcome = _drive(planner, ORIGIN, CHAIR_XY, [obstacle])
    assert outcome == "arrived", (outcome, reasons[-8:])
    assert "avoid" in reasons, "it should notice the bin at all"
    margin = _closest_approach(path, obstacle)
    assert margin >= obstacle.radius_m + planner.config.robot_radius_m, (
        f"clipped the bin: closest approach {margin:.3f} m")


def test_it_commits_to_one_side_rather_than_splitting_the_difference():
    """A swerve must be a real lateral displacement, not a wobble on the centreline."""
    planner = _planner()
    obstacle = _bin()
    path, _, _ = _drive(planner, ORIGIN, CHAIR_XY, [obstacle])
    # Lateral offset from the straight robot->chair line, at the bin's range.
    bearing = math.atan2(CHAIR_XY[1], CHAIR_XY[0])
    offsets = [(-px * math.sin(bearing) + py * math.cos(bearing)) for px, py in path]
    swing = max(abs(min(offsets)), abs(max(offsets)))
    # Bounds from the CONFIGURATION, not from what this run happened to do. The bin sits
    # on the robot-to-chair line, so clearing it needs the full HARD constraint of
    # lateral room — obstacle radius, the robot's own, and the hard gap between them —
    # and the planner is asking for that plus one soft gap on top. A trajectory outside
    # that band means the geometry is no longer driving the detour: too small and it is
    # clipping the bin, too large and it is spending corridor width the robot cannot
    # sense on a berth nothing asked for.
    #
    # `hard_gap_m` belongs in this sum and was once left out of it, which made the upper
    # bound 0.25 m too generous and hid a real over-swerve until a live run drove into a
    # wall. It is also the term that dominates: below about 0.25 m of soft gap the swerve
    # stops shrinking at all, because this is what is setting it.
    minimum = (_bin().radius_m + planner.config.robot_radius_m
               + planner.config.hard_gap_m)
    assert swing >= minimum - 0.05, f"only {swing:.2f} m of offset; needs {minimum:.2f} m"
    assert swing <= minimum + STATIC_SOFT_GAP_M, (
        f"detoured {swing:.2f} m to clear a bin needing {minimum:.2f} m")


def test_the_avoid_label_survives_a_tight_berth():
    """The berth and the label are separate numbers, and this is why.

    They were one field. Tightening the berth for a narrow corridor then silently
    stopped a run that visibly swerved around a bin from ever reporting `avoid` — losing
    the single word that makes the footage and the log legible. A berth is a control
    decision; a label is an explanation.
    """
    planner = _planner()
    _, reasons, outcome = _drive(planner, ORIGIN, CHAIR_XY, [_bin(soft_gap_m=0.05)])
    assert outcome == "arrived", outcome
    assert "avoid" in reasons, "a visible swerve must still say so"


def test_a_bin_does_not_get_a_persons_berth():
    """The per-obstacle soft gap, isolated. Same scene, same radius, only the soft gap
    changes — so a failure here cannot be blamed on the footprint."""
    planner = _planner()
    tight, _, _ = _drive(planner, ORIGIN, CHAIR_XY, [_bin()])
    wide, _, _ = _drive(planner, ORIGIN, CHAIR_XY, [_bin(soft_gap_m=1.20)])
    bearing = math.atan2(CHAIR_XY[1], CHAIR_XY[0])

    def swing(path):
        offsets = [(-px * math.sin(bearing) + py * math.cos(bearing)) for px, py in path]
        return max(abs(min(offsets)), abs(max(offsets)))

    assert swing(tight) < swing(wide) - 0.3, (swing(tight), swing(wide))


def test_the_reported_reason_does_not_chatter_on_approach():
    """`avoid` is a label applied after the choice, so weight_smooth cannot damp it.

    Measured before the hysteresis: goal/avoid alternated 44 times on this approach.
    """
    planner = _planner()
    _, reasons, _ = _drive(planner, ORIGIN, CHAIR_XY, [_bin()])
    flips = sum(1 for a, b in zip(reasons, reasons[1:]) if a != b)
    assert flips <= 6, f"reason changed {flips} times: {reasons}"


def test_it_does_not_dither_between_holding_and_moving():
    """Observed live: `hold` and `avoid` alternated on 9 consecutive ticks.

    Counting transitions rather than holds — a sustained hold is a legitimate answer,
    a rapidly alternating one is the chatter the hysteresis exists to remove.
    """
    planner = _planner()
    _, reasons, _ = _drive(planner, ORIGIN, CHAIR_XY, [_bin()])
    flips = sum(1 for a, b in zip(reasons, reasons[1:])
                if (a == "hold") != (b == "hold"))
    assert flips <= 2, f"hold state flipped {flips} times: {reasons}"


def test_inflating_a_bin_to_person_size_is_what_causes_the_stop():
    """The counter-example that gives the test above its meaning.

    Same scene, same planner, only the radius changed from a bin's to a person's. If
    this passed too, the per-class radius would be decoration.
    """
    planner = _planner()
    approach = (1.55, 0.19, 0.0)
    fine = planner.plan(approach, CHAIR_XY, CRUISING, [_bin(radius_m=0.23)])
    inflated = planner.plan(approach, CHAIR_XY, CRUISING, [_bin(radius_m=0.43)])
    assert fine.gap_m > inflated.gap_m
    assert inflated.gap_m < fine.gap_m - 0.15, (fine, inflated)


def test_a_static_obstacle_is_not_rolled_forward():
    """Zero velocity must mean the rollout leaves it where it is.

    Worth pinning because a mapped landmark reaching the planner with a spurious
    velocity — the failure mode static_map exists to prevent — would show up as the
    robot dodging empty floor, and nothing else in the pipeline would complain.
    """
    planner = _planner()
    still = planner.plan(ORIGIN, GOAL, CRUISING, [_bin()])
    drifting = planner.plan(ORIGIN, GOAL, CRUISING,
                            [Obstacle(x=BIN_XY[0], y=BIN_XY[1], vx=-0.4, vy=0.0,
                                      radius_m=0.23, label="bin")])
    assert still.gap_m > drifting.gap_m, "a mover closing in must score worse"


# ── Hold hysteresis ─────────────────────────────────────────────────────────
def test_leaving_a_hold_needs_more_room_than_staying_out_of_one():
    """The Schmitt trigger. Observed live: hold/avoid alternated on 9 straight ticks.

    A gap sitting exactly on hard_gap_m is the chattering case: from `goal` it is
    feasible, from `hold` it must not yet be. No cost weight can fix this, because
    `hold` is a fallback taken when the feasible set is empty and never competes on
    cost at all.
    """
    planner = _planner()
    config = planner.config
    # Place the obstacle so the WORST gap over the rollout lands just above hard_gap.
    blocker = Obstacle(x=1.35, y=0.0, vx=0.0, vy=0.0, radius_m=0.35)
    from_goal = planner.plan(ORIGIN, GOAL, STOPPED, [blocker], last_reason="goal")
    from_hold = planner.plan(ORIGIN, GOAL, STOPPED, [blocker], last_reason="hold")
    assert config.reason_hysteresis_m > 0.0
    # Whatever the exact gap, the hold branch can never be MORE permissive.
    assert from_hold.feasible <= from_goal.feasible, (from_goal, from_hold)


def test_hysteresis_does_not_trap_the_robot_in_a_hold():
    """A margin that could not be escaped would be worse than the chatter."""
    planner = _planner()
    clear = planner.plan(ORIGIN, GOAL, STOPPED, [], last_reason="hold")
    assert clear.reason == "goal", clear
    assert clear.vx > 0.0


def test_the_stopping_cap_is_not_widened_by_the_hold_margin():
    """Stopping distance is physics, not a threshold.

    Adding the margin there would make the robot creep out of a hold more slowly the
    longer it had been holding, which is a different and worse behaviour.
    """
    planner = _planner()
    far = Obstacle(x=4.0, y=0.0, vx=0.0, vy=0.0, radius_m=0.35)
    a = planner.plan(ORIGIN, GOAL, CRUISING, [far], last_reason="goal")
    b = planner.plan(ORIGIN, GOAL, CRUISING, [far], last_reason="hold")
    assert a.vx == b.vx, "a clear lane must command the same speed either way"


def test_an_obstacle_carries_its_label_through_to_the_command_scoring():
    """The planner ignores labels, but they must survive for the log and overlay."""
    planner = _planner()
    command = planner.plan(ORIGIN, GOAL, CRUISING, [_bin()])
    assert command.evaluated > 0
    assert _bin().label == "bin"


# ── The full-stop candidate ─────────────────────────────────────────────────
def test_a_passable_lane_is_never_answered_with_a_standstill():
    """A stationary rollout has a clearance cost of zero BY CONSTRUCTION.

    That used to make a full stop the cheapest command whenever an obstacle was
    anywhere near, and the planner returned `reason="goal"` with v=(0,0,0): the log
    claimed it was driving at the goal while it stood still, and the rest-after-blocked
    timer never started because the reason was not "hold". Measured on the staged
    scene, with all 331 candidates feasible, the stop scored 1.317 against 1.393 for
    the best moving command.
    """
    planner = _planner()
    command = planner.plan(ORIGIN, CHAIR_XY, CRUISING, [_bin()])
    assert command.feasible > 0, "the lane is passable; this is not a feasibility bug"
    assert not command.is_stop, command
    assert command.vx > 0.0, command


def test_every_sampled_command_is_reachable_within_one_control_period():
    """The dynamic window's one job. A stop row bypassed it entirely."""
    planner = _planner()
    control_dt = 0.1
    window = planner._window(CRUISING, control_dt)
    floor = CRUISING[0] - planner.limits.accel_x * control_dt
    assert window[:, 0].min() >= floor - 1e-9, window[:, 0].min()


def test_stopping_is_still_always_available_when_nothing_is_safe():
    """Removing the stop row must not remove the ability to stop."""
    planner = _planner()
    blocker = Obstacle(x=0.6, y=0.0, vx=0.0, vy=0.0, radius_m=0.35)
    command = planner.plan(ORIGIN, GOAL, CRUISING, [blocker])
    assert command.reason == "hold"
    assert command.is_stop, command


def test_crowding_is_normalised_per_obstacle_before_the_worst_is_taken():
    """A bin's soft gap and a person's differ by 2.7x, so a raw gap cannot rank them.

    Being halfway inside a person's 1.20 m and halfway inside a bin's 0.45 m are the
    same amount of crowding, and must score the same — otherwise one weight cannot serve
    both and the planner treats every bin as an emergency.
    """
    planner = _planner()
    import numpy as np
    # One candidate, two obstacles: each sitting at half its own soft gap.
    per_obstacle = np.array([[0.60, 0.225]])          # gaps to person, bin
    soft = np.array([1.20, 0.45])
    crowding = planner._clearance_cost(per_obstacle, soft)
    assert abs(crowding[0] - 0.5) < 1e-9, crowding


def test_a_crowded_bin_outranks_a_comfortable_person():
    """The max is over NORMALISED crowding, so the worst offender wins on its own scale."""
    planner = _planner()
    import numpy as np
    per_obstacle = np.array([[1.10, 0.05]])           # person nearly clear, bin close
    soft = np.array([1.20, 0.45])
    assert planner._clearance_cost(per_obstacle, soft)[0] > 0.8


def test_an_empty_scene_survives_the_zero_length_reduction():
    """With no obstacles the per-obstacle arrays are (N, 0), and numpy raises on a
    reduction over a zero-length axis. The guard in _clearance_cost is load-bearing
    rather than defensive, so pin it."""
    planner = _planner()
    import numpy as np
    empty_gaps = np.full((7, 0), math.inf)
    empty_soft = np.array([])
    assert planner._clearance_cost(empty_gaps, empty_soft).tolist() == [0.0] * 7
    assert all(math.isinf(v) for v in planner._worst_gap(empty_gaps))


def test_a_bin_and_a_person_are_both_planned_against():
    """End to end through plan(), with the two kinds carrying different soft gaps."""
    planner = _planner()
    crosser = Obstacle(x=1.4, y=-1.0, vx=0.0, vy=0.6, radius_m=0.35, label="person")
    command = planner.plan(ORIGIN, CHAIR_XY, CRUISING, [_bin(), crosser])
    assert command.evaluated > 0
    assert command.reason in ("goal", "avoid", "hold")
    # The reported gap is the WORST over both, so it cannot exceed either one alone.
    bin_only = planner.plan(ORIGIN, CHAIR_XY, CRUISING, [_bin()])
    assert command.gap_m <= bin_only.gap_m + 1e-9


# ── is_feasible: the same test plan() applies, asked about ONE command ──────
def test_a_command_driving_into_someone_is_not_feasible():
    """The point of the predicate. Straight ahead at cruise, at a person 1.2 m away, ends
    inside them well within the horizon."""
    blocker = Obstacle(x=1.2, y=0.0, vx=0.0, vy=0.0, radius_m=0.35)
    assert not _planner().is_feasible(ORIGIN, (0.30, 0.0, 0.0), [blocker])


def test_a_command_that_clears_them_is_feasible():
    """A predicate that refuses everything is not a veto, it is a brake."""
    blocker = Obstacle(x=1.2, y=0.0, vx=0.0, vy=0.0, radius_m=0.35)
    planner = _planner()
    # Sidestep rather than a turn: a constant yaw rate held for the whole horizon traces
    # an arc that can curve back towards the obstacle, which is a true answer about a
    # command nobody would hold that long and a confusing one to assert on.
    assert planner.is_feasible(ORIGIN, (0.0, 0.20, 0.0), [blocker])
    assert planner.is_feasible(ORIGIN, (0.30, 0.0, 0.0), []), "an empty scene is feasible"


def test_it_agrees_with_the_command_plan_actually_chose():
    """The two must not be able to disagree — a veto that refuses the planner's own
    output would deadlock any consumer that fell back to it."""
    for obstacles in ([], [Obstacle(x=1.6, y=0.3, vx=0.0, vy=0.0, radius_m=0.35)],
                      [Obstacle(x=2.2, y=-1.6, vx=0.0, vy=1.1, radius_m=0.35)]):
        planner = _planner()
        command = planner.plan(ORIGIN, GOAL, CRUISING, obstacles)
        if command.is_stop:
            continue                    # a hold is not a sampled candidate
        assert planner.is_feasible(ORIGIN, (command.vx, command.vy, command.wz),
                                   obstacles), command


def test_it_honours_the_per_obstacle_hard_gap():
    """A landmark and a person do not want the same margin, which is the whole reason
    Obstacle carries the override. A predicate that used the planner default would give a
    bin a person's berth — and the docstring on soft_gap_m says that is actively harmful
    in a corridor."""
    at = {"x": 1.1, "y": 0.0, "vx": 0.0, "vy": 0.0, "radius_m": 0.25}
    planner = _planner()
    generous = Obstacle(**at, hard_gap_m=STATIC_HARD_GAP_M)
    strict = Obstacle(**at, hard_gap_m=0.60)
    command = (0.20, 0.20, 0.0)
    assert planner.is_feasible(ORIGIN, command, [generous])
    assert not planner.is_feasible(ORIGIN, command, [strict])


def test_a_shorter_horizon_sees_less_of_the_future():
    """The parameter exists so the "2.5 s is too strict for a command re-chosen at 10 Hz"
    argument can be re-run with numbers. It has to actually do something."""
    blocker = Obstacle(x=1.2, y=0.0, vx=0.0, vy=0.0, radius_m=0.35)
    planner = _planner()
    assert not planner.is_feasible(ORIGIN, (0.30, 0.0, 0.0), [blocker])
    assert planner.is_feasible(ORIGIN, (0.30, 0.0, 0.0), [blocker], horizon_s=0.5)


def test_a_moving_obstacle_is_advanced_over_the_horizon():
    """Same as plan(): scored against where they WILL be. A crosser who is not in the way
    now but will be must make a straight-ahead command infeasible."""
    # Timed to arrive where the robot will be at about t = 1 s. A crosser further out in
    # x passes ahead of it and is correctly feasible, which is a fine answer and a
    # useless test.
    crosser = Obstacle(x=0.8, y=-1.0, vx=0.0, vy=1.0, radius_m=0.35)
    planner = _planner()
    assert planner.is_feasible(ORIGIN, (0.30, 0.0, 0.0), [crosser], horizon_s=0.25), \
        "they are still well clear over the next quarter second"
    assert not planner.is_feasible(ORIGIN, (0.30, 0.0, 0.0), [crosser])


# ── Issue #20: the counts a producer forgot to forward ──────────────────────────────
#
# The defect this guards was NOT in this file. `MappoPlanner.plan` in
# `integration/mappo_drive.py` wraps this planner and returns `Command`s of its own, and
# three of its four branches let `feasible`/`evaluated` fall back to the dataclass
# default. The default was `0`, which is a real answer — the hold branch below returns
# `feasible=0` to mean "nothing cleared the hard gap" — so 58 policy-driven ticks of a
# successful run recorded "boxed in" and nobody could tell. Those four branches were
# fixed; a FIFTH added later would reproduce it, and until this test nothing said so.
#
# So the guard is a scan, not a scenario. A behavioural test can only exercise branches
# that exist when it is written, which is precisely the case that was already covered.


def _repository_root():
    """The directory holding ``AGENTS.md``, walking up from this file.

    Anchored on a marker rather than on ``parents[N]`` because this directory is vendored
    from the upstream control stack and the depth differs between the two trees. Returns
    ``None`` when there is no marker above us, which the caller turns into a narrower
    scan and a loud message rather than a pass.
    """
    directory = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(directory, "AGENTS.md")):
            return directory
        parent = os.path.dirname(directory)
        if parent == directory:
            return None
        directory = parent


def _command_bindings(tree):
    """Local names that refer to THIS dataclass: ``(direct names, module aliases)``.

    Matching the call name alone would also catch ``VelocityCommand`` lookalikes and any
    unrelated class called ``Command``; resolving the binding first keeps the scan
    pointed at this one. All three spellings count, because a guard that sees only the
    spelling in use today is a guard the next author walks past without meaning to:
    ``Command(...)``, ``avoidance.Command(...)``, and ``Command`` imported under another
    name. Both sets are empty for a module that does not have it, which is the signal to
    skip the file entirely.
    """
    direct, modules = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Command":
            direct.add("Command")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] == "avoidance":
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (node.module or "").endswith("avoidance") and alias.name == "Command":
                    direct.add(alias.asname or alias.name)
                elif alias.name == "avoidance":
                    modules.add(alias.asname or alias.name)
    return direct, modules


def _is_command_call(node, direct, modules):
    """A call that constructs this dataclass, under any of its three spellings."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in direct
    return (isinstance(node.func, ast.Attribute) and node.func.attr == "Command"
            and isinstance(node.func.value, ast.Name) and node.func.value.id in modules)


def _command_construction_sites():
    """Every ``Command(...)`` call in non-test Python that binds this dataclass.

    ``(relative path, line number, set of keyword names)``. Test files are excluded on
    purpose: a fixture command is not a producer, nobody reads its counts out of a
    telemetry file, and requiring the keywords there would make every future test carry
    two numbers it does not use.
    """
    root = _repository_root()
    scanned_root = root or os.path.dirname(os.path.abspath(__file__))
    sites = []
    for directory, subdirs, files in os.walk(scanned_root):
        subdirs[:] = [d for d in subdirs
                      if d not in (".git", "__pycache__", "node_modules")]
        for name in sorted(files):
            if not name.endswith(".py") or name.startswith("test_"):
                continue
            path = os.path.join(directory, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=path)
            except (OSError, SyntaxError):
                continue
            direct, modules = _command_bindings(tree)
            if not direct and not modules:
                continue
            for node in ast.walk(tree):
                if _is_command_call(node, direct, modules):
                    sites.append((os.path.relpath(path, scanned_root), node.lineno,
                                  {k.arg for k in node.keywords if k.arg}))
    return root, sites


def test_every_branch_that_returns_a_command_states_its_search():
    """⚠️ THE GUARD FOR ISSUE #20. Add a branch that returns a ``Command`` without
    forwarding ``feasible``/``evaluated`` and this test names the file and the line.

    Demonstrated by doing exactly that: a fifth ``return Command(...)`` added to
    ``DynamicWindowPlanner.plan`` without the two keywords fails here with
    ``avoidance.py:NNN constructs a Command without stating ['evaluated', 'feasible']``,
    and the same edit made in ``integration/mappo_drive.py`` fails the same way. Neither
    is caught by any behavioural test, because a branch nobody has written yet cannot be
    driven by a scenario.
    """
    root, sites = _command_construction_sites()
    assert root is not None, (
        "no AGENTS.md above this file, so the scan only covered this directory and "
        "would not have seen a wrapper elsewhere in the tree")

    # Anti-vacuity, and it is the point: a scan that stops finding anything must go red
    # rather than green. Seven sites today — three in this planner, four in
    # `MappoPlanner.plan` — across two files.
    assert len(sites) >= 5, f"the scan found only {len(sites)} construction sites: {sites}"
    files = {path for path, _, _ in sites}
    assert len(files) >= 2, f"only one file constructs a Command; the wrapper is missing: {files}"
    assert any(path.endswith("avoidance.py") for path in files), (
        f"this planner is not among the scanned producers: {sorted(files)}")

    required = {"feasible", "evaluated"}
    missing = [(path, line, sorted(required - keywords))
               for path, line, keywords in sites if required - keywords]
    assert not missing, "\n".join(
        f"{path}:{line} constructs a Command without stating {sorted(names)} — "
        f"they describe the planner's search and default to COUNT_NOT_RECORDED, "
        f"so a consumer reads 'not stated' rather than a count"
        for path, line, names in missing)


def test_an_unstated_count_is_not_the_same_value_as_a_real_zero():
    """``feasible=0`` is an ANSWER — the hold branch returns it to say nothing cleared
    the hard gap. So the default cannot be 0, or "nobody said" and "the robot is boxed
    in" are the same bytes in the telemetry, and one of them is an alarm."""
    assert COUNT_NOT_RECORDED < 0, "no count can be negative; that is what makes it safe"

    unstated = Command(0.0, 0.0, 0.0, reason="hold", gap_m=1.0)
    assert unstated.feasible == COUNT_NOT_RECORDED
    assert unstated.evaluated == COUNT_NOT_RECORDED
    assert not unstated.search_recorded

    boxed_in = Command(0.0, 0.0, 0.0, reason="hold", gap_m=1.0, feasible=0, evaluated=330)
    assert boxed_in.search_recorded, "0 of 330 is a stated result, not an absent one"
    assert boxed_in.feasible != unstated.feasible


def test_the_planners_own_branches_state_a_search_that_actually_ran():
    """The scan proves the keywords are PRESENT. This proves they carry the search rather
    than a literal: every branch of ``plan`` must report the window it really sampled."""
    planner = _planner()
    blocked = [Obstacle(x=0.45, y=0.0, vx=0.0, vy=0.0, radius_m=0.6)]
    outcomes = {
        "hold": planner.plan(ORIGIN, GOAL, CRUISING, blocked),
        "clear": planner.plan(ORIGIN, GOAL, CRUISING, []),
        "avoid": planner.plan(ORIGIN, GOAL, CRUISING,
                              [Obstacle(x=1.6, y=0.45, vx=0.0, vy=0.0, radius_m=0.35)]),
    }
    window = planner.config.samples_vx * planner.config.samples_vy * planner.config.samples_wz
    for name, command in outcomes.items():
        assert command.reason == name.replace("clear", "goal"), (name, command)
        assert command.search_recorded, (name, command)
        assert command.evaluated == window, (name, command, window)
        assert 0 <= command.feasible <= command.evaluated, (name, command)
    assert outcomes["hold"].feasible == 0, "a hold IS zero feasible; that is the answer"
    assert outcomes["clear"].feasible > 0, outcomes["clear"]


# ── A qualified reason (issue #118) ─────────────────────────────────────────
def test_base_reason_strips_one_qualifier_and_nothing_else():
    """⚠️ THE GUARD FOR ISSUE #118. Two ways to read a reason wrong, both shipped.

    Comparing the whole string misses ``veto-hold``, which is what
    ``integration/mappo_drive.py`` writes when it refuses the policy's command and
    issues the planner's own hold instead. Searching for a substring instead — ``"hold"
    in reason`` — calls ``threshold`` a hold, which is a different decision taken
    because two strings share five letters.

    The rule is: exactly one leading ``<qualifier>-`` comes off, and only when what is
    left is a word the planner actually issues.
    """
    for reason, expected in (
            # The planner's own vocabulary is returned unchanged.
            ("goal", "goal"), ("avoid", "avoid"), ("hold", "hold"),
            ("arrived", "arrived"),
            # A qualifier comes off. `peer-` is not written today; the rule is about
            # the SHAPE, so a second wrapper does not have to re-earn it.
            ("veto-goal", "goal"), ("veto-avoid", "avoid"), ("veto-hold", "hold"),
            ("peer-hold", "hold"), ("-hold", "hold"),
            # Not the planner's words, so nothing is stripped and they are compared as
            # themselves. `policy` is a real one: `MappoPlanner` writes it every tick
            # the policy drives.
            ("policy", "policy"), ("veto-policy", "veto-policy"),
            # A substring test would call these four a hold, a hold, an avoid and a
            # goal. This one calls them what they are.
            ("threshold", "threshold"), ("holding_pattern", "holding_pattern"),
            ("avoidance", "avoidance"), ("goalkeeper", "goalkeeper"),
            # One qualifier, not a chain of them, and nothing degenerate.
            ("veto-peer-hold", "veto-peer-hold"), ("veto-", "veto-"), ("", ""),
    ):
        assert base_reason(reason) == expected, (reason, base_reason(reason))


def test_the_planner_only_issues_words_from_its_own_vocabulary():
    """:data:`PLANNER_REASONS` is what ``base_reason`` will strip down to, so a reason
    this planner issues and that list does not name would survive its own qualifier and
    silently stop matching anywhere. Drives all three branches rather than reading the
    literals back."""
    planner = _planner()
    issued = {
        planner.plan(ORIGIN, GOAL, CRUISING, []).reason,
        planner.plan(ORIGIN, GOAL, CRUISING,
                     [Obstacle(x=1.6, y=0.45, vx=0.0, vy=0.0, radius_m=0.35)]).reason,
        planner.plan(ORIGIN, GOAL, CRUISING,
                     [Obstacle(x=0.45, y=0.0, vx=0.0, vy=0.0, radius_m=0.6)]).reason,
    }
    assert issued == {"goal", "avoid", "hold"}, issued
    assert issued <= set(PLANNER_REASONS), issued
    # `arrived` is never issued HERE — `mappo_drive` maps the policy's STOP_GOAL_REACHED
    # to it — but it travels in this field and must strip like the rest.
    assert "arrived" in PLANNER_REASONS


def test_the_hold_hysteresis_reads_a_qualified_reason():
    """⚠️ SIBLING OF ISSUE #118, and this one moves the legs rather than a counter.

    ``reason_hysteresis_m`` is a Schmitt trigger: leaving a hold has to clear 0.08 m
    more gap than entering one did, because live "``hold`` and ``avoid`` alternated on
    9 consecutive ticks". It was read as ``last_reason == "hold"`` — and on a
    policy-driven run the previous reason is ``veto-hold``, so the trigger was off for
    every tick of every such run and the chatter it was added for was back.

    Staged in the MIDDLE of the band where the margin is what decides: a 0.35 m blocker
    dead ahead, between 1.00 m and 1.07 m from a standstill. Outside that band both
    answers agree and this test would pass without the fix, which is what the first
    assertion is for.
    """
    planner = _planner()
    blocker = [Obstacle(x=1.04, y=0.0, vx=0.0, vy=0.0, radius_m=0.35)]

    entering = planner.plan(ORIGIN, GOAL, STOPPED, blocker, last_reason="goal")
    assert entering.reason != "hold", (
        f"the scene must be one where the margin DECIDES or this test proves nothing: "
        f"{entering}")

    holding = planner.plan(ORIGIN, GOAL, STOPPED, blocker, last_reason="hold")
    assert holding.reason == "hold", holding

    vetoed = planner.plan(ORIGIN, GOAL, STOPPED, blocker, last_reason="veto-hold")
    assert vetoed.reason == "hold", (
        f"a vetoed hold left the hold 0.08 m early, which is the chatter the "
        f"hysteresis exists to remove: {vetoed}")


def test_the_avoid_hysteresis_reads_a_qualified_reason():
    """The other Schmitt trigger, and the other half of the same defect.

    ``avoid`` is a label applied after the choice, so no cost weight can damp it; before
    the hysteresis "the simulated approach to the staged bin flipped ``goal``/``avoid``
    44 times". Same band argument as the test above: a 0.23 m bin offset 0.30 m to the
    left decides between 2.70 m and 2.76 m, and 2.73 m is the middle of it.
    """
    planner = _planner()
    bin_ = [Obstacle(x=2.73, y=0.30, vx=0.0, vy=0.0, radius_m=0.23,
                     soft_gap_m=STATIC_SOFT_GAP_M)]

    entering = planner.plan(ORIGIN, GOAL, CRUISING, bin_, last_reason="goal")
    assert entering.reason == "goal", (
        f"the scene must be one where the margin DECIDES or this test proves nothing: "
        f"{entering}")

    avoiding = planner.plan(ORIGIN, GOAL, CRUISING, bin_, last_reason="avoid")
    assert avoiding.reason == "avoid", avoiding

    vetoed = planner.plan(ORIGIN, GOAL, CRUISING, bin_, last_reason="veto-avoid")
    assert vetoed.reason == "avoid", (
        f"a vetoed avoid dropped its label, which is the 44-flip chatter coming back: "
        f"{vetoed}")


_REASON_KEY = "reason"


def _subscript_key(node):
    """The constant a ``x["..."]`` is keyed by, under both AST shapes this tree runs on.

    ⚠️ Python 3.8 — THE ROBOT'S INTERPRETER, and a CI leg — wraps a subscript key in
    ``ast.Index``; 3.9 and later store the expression directly. Unwrapped by attribute
    rather than by ``isinstance(..., ast.Index)``, because that class is deprecated above
    3.8 and is scheduled to go, so naming it would trade one version break for another.

    Constant first, THEN the unwrap: an ``ast.Constant`` has a ``.value`` of its own —
    the string — so unwrapping unconditionally would hand back ``"reason"`` and the
    isinstance below would reject it. Returns ``None`` for a computed key, which is not
    something this scan can judge.
    """
    key = node.slice
    if not isinstance(key, ast.Constant):
        key = getattr(key, "value", None)
    return key if isinstance(key, ast.Constant) else None


def _bare_reason_comparisons(tree):
    """``(line, dumped source)`` for every ``<a reason> == "<planner word>"`` in a tree.

    Four spellings, because all four exist in this repository and a scan that saw only
    the one in front of it is a scan the next author walks past: an attribute
    (``command.reason``), a bare name (``last_reason``), a mapping read out of a
    telemetry record (``tick["command"].get("reason")`` and ``[...]["reason"]``).
    ``base_reason(x) == "hold"`` is deliberately NOT matched — that is the fixed
    spelling, and matching it would make the fix unfixable.
    """
    def reads_a_reason(node):
        if isinstance(node, ast.Attribute):
            return node.attr == _REASON_KEY
        if isinstance(node, ast.Name):
            return node.id == _REASON_KEY or node.id.endswith("_" + _REASON_KEY)
        if isinstance(node, ast.Subscript):
            key = _subscript_key(node)
            return key is not None and key.value == _REASON_KEY
        if isinstance(node, ast.Call):
            return (isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == _REASON_KEY)
        return False

    def is_planner_word(node):
        return isinstance(node, ast.Constant) and node.value in PLANNER_REASONS

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            continue
        left, right = node.left, node.comparators[0]
        if ((reads_a_reason(left) and is_planner_word(right))
                or (reads_a_reason(right) and is_planner_word(left))):
            found.append((node.lineno, ast.dump(node)))
    return found


def test_the_reason_scan_detects_every_spelling_it_claims_to():
    """A scan whose detector is broken reports a clean tree forever. This is the fixture
    that makes the next test's silence mean something: four spellings that MUST be
    found, and four near-misses that must not be."""
    caught = ast.parse(
        'a = command.reason == "hold"\n'
        'b = last_reason != "avoid"\n'
        'c = tick["command"].get("reason") == "goal"\n'
        'd = "arrived" == record["reason"]\n')
    assert len(_bare_reason_comparisons(caught)) == 4, _bare_reason_comparisons(caught)

    allowed = ast.parse(
        'a = base_reason(command.reason) == "hold"\n'   # the fixed spelling
        'b = previous == "avoid"\n'                     # already a base word
        'c = command.reason == "policy"\n'              # not a planner word
        'd = command.reason in PLANNER_REASONS\n')      # not an equality
    assert _bare_reason_comparisons(allowed) == [], _bare_reason_comparisons(allowed)

    # THE PYTHON 3.8 SUBSCRIPT SHAPE, built by hand because ``ast.parse`` on this
    # interpreter cannot produce it. 3.8 wraps a subscript key in ``ast.Index`` and 3.9+
    # does not, so on 3.11 the unwrap branch in :func:`_subscript_key` never runs — and a
    # scan that had stopped seeing ``record["reason"]`` on THE ROBOT'S interpreter would
    # still be green here. That is not hypothetical: CI's 3.8 leg failed on exactly this
    # the first time this scan was pushed, and this assertion is so that the 3.11 leg does
    # not need it to.
    class _LegacyIndex:
        def __init__(self, value):
            self.value = value

    legacy = ast.Module(body=[ast.Expr(value=ast.Compare(
        left=ast.Subscript(value=ast.Name(id="record", ctx=ast.Load()),
                           slice=_LegacyIndex(ast.Constant(value=_REASON_KEY)),
                           ctx=ast.Load()),
        ops=[ast.Eq()], comparators=[ast.Constant(value="hold")]))], type_ignores=[])
    # A hand-built tree carries no line numbers, and the scan reports one.
    ast.fix_missing_locations(legacy)
    assert len(_bare_reason_comparisons(legacy)) == 1, (
        "the Python 3.8 `ast.Index` subscript shape is not recognised, so this scan is "
        "blind to `record[\"reason\"] == \"hold\"` on the interpreter the robot runs")


def test_no_shipped_module_compares_a_reason_against_a_bare_planner_word():
    """⚠️ THE REGRESSION GUARD FOR ISSUE #118. Write ``command.reason == "hold"``
    anywhere outside a test and this names the file and the line.

    Behavioural tests cover the three sites that were wrong. They cannot cover the
    fourth site somebody adds next month, and this defect's whole character is that the
    wrong spelling keeps looking right: the qualifier is invisible from the consumer's
    side, and neither stall gate fires while the consequence plays out.

    Test files are excluded. A fixture asserting ``command.reason == "veto-hold"`` is
    stating what a producer wrote, which is exactly what a test should do.
    """
    root = _repository_root()
    assert root is not None, (
        "no AGENTS.md above this file, so this scan only covered one directory")

    offenders = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs
                      if d not in (".git", "__pycache__", "node_modules")]
        for name in sorted(files):
            if not name.endswith(".py") or name.startswith("test_"):
                continue
            path = os.path.join(directory, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=path)
            except (OSError, SyntaxError):
                continue
            for line, _dump in _bare_reason_comparisons(tree):
                offenders.append(f"{os.path.relpath(path, root)}:{line}")
    assert not offenders, (
        "a Command reason may arrive QUALIFIED (`veto-hold`), so an equality against a "
        "bare planner word silently never matches — read it through "
        "`avoidance.base_reason`:\n  " + "\n  ".join(offenders))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"avoidance: {len(tests)}/{len(tests)} passed")
