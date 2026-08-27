#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the turn-drive execution supervisor.

The numbers are the measured ones, not round ones: the bin's planning radius is the
0.33 m upper bound measured on 2026-08-26 (the nominal 0.20 m the profile used to carry
is the value these tests exist to reject), the robot radius is the 0.40 m loaded
half-diagonal, and the speeds are LITE3-A's measured primitives (0.5362 m/s forward,
0.8566 rad/s yaw). A test written against rounder numbers would pass against a geometry
the demo floor never had.

Needs the vendored planner (numpy). Run: ``python3 test_turn_drive_supervisor.py``
"""
from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "robot-stack", "unitree", "go2",
                                "visual_nav"))

from avoidance import Obstacle

from turn_drive_supervisor import (
    HEADING_TOLERANCE_RAD,
    TurnDriveSupervisor,
    _point_to_segment_m,
)

ROBOT_RADIUS_M = 0.40
BIN_RADIUS_M = 0.33                 # measured 2026-08-26; the nominal 0.20 m is the bug
FORWARD_M_S = 0.5362                # LITE3-A measured forward primitive
YAW_RAD_S = 0.8566                  # LITE3-A measured yaw primitive


def _supervisor(**kwargs) -> TurnDriveSupervisor:
    return TurnDriveSupervisor(robot_radius_m=ROBOT_RADIUS_M,
                               drive_speed_m_s=FORWARD_M_S,
                               turn_rate_rad_s=YAW_RAD_S, **kwargs)


def _bin(x=1.5, y=0.0, radius_m=BIN_RADIUS_M) -> Obstacle:
    return Obstacle(x=x, y=y, vx=0.0, vy=0.0, radius_m=radius_m, label="bin",
                    kind="static", object_id="landmark-1")


def _execute(supervisor, pose, goal, obstacles, max_ticks=400):
    """Run the supervisor as the only controller, with ideal kinematics.

    The axis transport is sign-only, so the simulation executes exactly what the
    command says at the command's own speed — no scaling, no lag — because what is
    under test is the GEOMETRY the supervisor commits to, not a gait model. Returns
    ``(poses, commands)``.
    """
    poses, commands = [pose], []
    for _ in range(max_ticks):
        command = supervisor.command(pose, goal, obstacles)
        if command is None:
            return poses, commands
        commands.append(command)
        x, y, yaw = pose
        dt = 0.1
        yaw += command.wz * dt
        x += command.vx * math.cos(yaw) * dt
        y += command.vx * math.sin(yaw) * dt
        pose = (x, y, yaw)
        poses.append(pose)
    return poses, commands


def test_the_measured_radius_blocks_a_line_the_nominal_radius_did_not():
    """Regression for the 0.20 m profile radius. The bin sits 0.80 m off the
    robot→goal line: clear of a 0.20 m disc (0.20 + 0.40 + 0.14 = 0.74 < 0.80) and
    blocking the measured 0.33 m one (0.87 > 0.80). The old configuration planned a
    straight line through 5 cm of air that was not there."""
    supervisor = _supervisor()
    offset = _bin(y=0.80)
    assert supervisor.blocker((0.0, 0.0, 0.0), (3.0, 0.0), [offset]) is offset
    nominal = _bin(y=0.80, radius_m=0.20)
    assert supervisor.blocker((0.0, 0.0, 0.0), (3.0, 0.0), [nominal]) is None, \
        "the nominal radius must NOT block this line — that is the defect"


def test_no_static_blocker_leaves_the_policy_in_charge():
    """Absent or off-line obstacles must produce NO supervisor output. A supervisor
    that acts on an empty map is how a missing detection would masquerade as a
    successful avoidance."""
    supervisor = _supervisor()
    assert supervisor.command((0.0, 0.0, 0.0), (3.0, 0.0), []) is None
    beside = _bin(y=1.50)     # 1.50 m off the line; the required gap is 0.87 m
    assert supervisor.command((0.0, 0.0, 0.0), (3.0, 0.0), [beside]) is None
    behind = _bin(x=-1.0)     # off the segment ends entirely
    assert supervisor.command((0.0, 0.0, 0.0), (3.0, 0.0), [behind]) is None


def test_a_tracked_person_on_the_line_is_not_a_detour_trigger():
    """The supervisor exists for the obstacle that will still be there next tick. A
    mover on the goal line belongs to the shared veto, which mappo_drive applies to
    the supervisor's output; routing around a person plans a detour whose target has
    walked away."""
    person = Obstacle(x=1.5, y=0.0, vx=0.0, vy=0.0, radius_m=0.35, label="person",
                      kind="tracked", object_id="track-1")
    supervisor = _supervisor()
    assert supervisor.blocker((0.0, 0.0, 0.0), (3.0, 0.0), [person]) is None
    assert supervisor.command((0.0, 0.0, 0.0), (3.0, 0.0), [person]) is None


def test_the_detour_never_requests_an_unmeasured_motion():
    """Every command over a full executed detour is a pure turn or a pure drive. The
    Lite3 profile has no lateral and no reverse primitive, so a single tick combining
    them — or asking for strafe — is a command the robot silently cannot perform."""
    supervisor = _supervisor()
    _poses, commands = _execute(supervisor, (0.0, 0.0, 0.0), (3.0, 0.0), [_bin()])
    assert commands, "the blocked line must produce supervisor commands"
    for command in commands:
        assert command.vy == 0.0
        assert not (command.vx != 0.0 and command.wz != 0.0), command
        assert command.phase in ("turn", "drive")


def test_the_executed_detour_clears_the_obstacle_and_hands_back():
    """Run the detour end to end: the robot's path must keep the required gap from the
    bin at every tick, and the supervisor must hand control back (return ``None``)
    once the line to the goal is clear — that hand-back is what lets MAPPO finish the
    approach it was trained for."""
    supervisor = _supervisor()
    obstacle = _bin()
    poses, commands = _execute(supervisor, (0.0, 0.0, 0.0), (3.0, 0.0), [obstacle])
    assert commands, "no detour was produced at all"
    required = supervisor._required_gap_m(obstacle)
    for pose in poses:
        distance = math.hypot(obstacle.x - pose[0], obstacle.y - pose[1])
        assert distance >= required - 1e-6, \
            f"path came within {distance:.3f} m of the bin centre (required {required})"
    assert supervisor.blocker(poses[-1], (3.0, 0.0), [obstacle]) is None, \
        "the detour ended with the line to the goal still blocked"
    # And it must have TURNED at least once — a detour that never turns is the
    # straight line this module exists to prevent.
    assert any(command.phase == "turn" for command in commands)
    assert any(command.phase == "drive" for command in commands)


def test_the_turn_half_of_the_band_cannot_deadlock_against_the_veto():
    """Pointing AT the blocker side of the leg must never produce a drive command:
    that drive's 2.5 s rollout enters the required gap, the shared veto refuses it,
    and a supervisor that reissued it next tick has parked the robot facing the bin."""
    supervisor = _supervisor()
    obstacle = _bin()
    waypoint, side = supervisor._waypoint((0.0, 0.0, 0.0), (3.0, 0.0), obstacle,
                                          [obstacle])
    bearing = math.atan2(waypoint[1], waypoint[0])
    # One degree SHORT of the leg bearing — inside the old symmetric band, toward the
    # bin. This must be a turn, and the turn must be AWAY from the blocker.
    inward = bearing - math.radians(1.0) * (1.0 if side == "left" else -1.0)
    command = supervisor.command((0.0, 0.0, inward), (3.0, 0.0), [obstacle])
    assert command is not None and command.phase == "turn", command


def test_a_detour_through_a_second_obstacle_is_refused_not_invented():
    """Two bins walling off both sides: no two-segment detour exists, and the honest
    answer is ``None`` — the planner's own veto then holds the robot, which is the
    safe state. Inventing a line through a gap that is not there is the failure."""
    supervisor = _supervisor()
    walls = [_bin(y=0.0), _bin(y=1.60), _bin(y=-1.60)]
    # Each side's corner is the tangent corner of the central bin; the flanking bins
    # at ±1.60 m sit inside the central one's detour corridor (corner ~1.1 m off the
    # line), so both legs fail clearance against a flanker.
    assert supervisor.command((0.0, 0.0, 0.0), (3.0, 0.0), walls) is None


def test_point_to_segment_refuses_projection_off_the_ends():
    assert _point_to_segment_m(-1.0, 0.1, 0.0, 0.0, 3.0, 0.0) is None
    assert _point_to_segment_m(4.0, 0.1, 0.0, 0.0, 3.0, 0.0) is None
    assert abs(_point_to_segment_m(1.5, 0.8, 0.0, 0.0, 3.0, 0.0) - 0.8) < 1e-9


def test_a_drive_phase_command_faces_the_leg_within_tolerance():
    """Drive begins only with the nose within the tolerance of the leg bearing AND on
    the outward half of it; both bounds exist because each is a distinct way to enter
    the gap."""
    supervisor = _supervisor()
    obstacle = _bin()
    waypoint, side = supervisor._waypoint((0.0, 0.0, 0.0), (3.0, 0.0), obstacle,
                                          [obstacle])
    bearing = math.atan2(waypoint[1], waypoint[0])
    outward_sign = 1.0 if side == "left" else -1.0
    # Nose exactly on the leg bearing: drive.
    on = supervisor.command((0.0, 0.0, bearing), (3.0, 0.0), [obstacle])
    assert on is not None and on.phase == "drive", on
    # Nose at the outward edge of the band: still drive.
    edge = bearing + outward_sign * (HEADING_TOLERANCE_RAD - math.radians(1.0))
    at_edge = supervisor.command((0.0, 0.0, edge), (3.0, 0.0), [obstacle])
    assert at_edge is not None and at_edge.phase == "drive", at_edge
    # Nose one degree PAST the band: turn.
    past = bearing + outward_sign * (HEADING_TOLERANCE_RAD + math.radians(1.0))
    beyond = supervisor.command((0.0, 0.0, past), (3.0, 0.0), [obstacle])
    assert beyond is not None and beyond.phase == "turn", beyond


def test_the_leg_switch_waits_for_a_clear_line_to_the_goal():
    """The arrival disc reaches 0.30 m short of the corner, and the goal-ward line
    from there still cuts the blocker's clearance circle — switching on distance
    alone commands a drive the shared veto must refuse, parking the robot beside
    the bin (reproduced 2026-08-27: 0.102 m of free space against a 0.12 m hard
    gap). Nose exactly on the GOAL bearing from inside the disc must therefore
    produce a TURN back toward the waypoint, not a drive at the goal."""
    supervisor = _supervisor()
    obstacle = _bin()
    waypoint, side = supervisor._waypoint((0.0, 0.0, 0.0), (3.0, 0.0), obstacle,
                                          [obstacle])
    leg = math.hypot(*waypoint)
    along = 1.0 - 0.20 / leg
    pose = (waypoint[0] * along, waypoint[1] * along)
    assert not supervisor._leg_clear(pose, (3.0, 0.0), [obstacle]), \
        "the test pose must still have a blocked goal line — else it tests nothing"
    goal_bearing = math.atan2(0.0 - pose[1], 3.0 - pose[0])
    command = supervisor.command((pose[0], pose[1], goal_bearing), (3.0, 0.0),
                                 [obstacle])
    assert command is not None and command.phase == "turn", command
    assert command.wz * (1.0 if side == "left" else -1.0) > 0.0, \
        "the turn must be back toward the waypoint, not further toward the goal"
    # Positive control: AT the corner the line to the goal is the tangent, which
    # clears — the blocker test then finds nothing on the line and the supervisor
    # hands control back to the policy (``None``), which is the designed exit.
    corner_goal_bearing = math.atan2(0.0 - waypoint[1], 3.0 - waypoint[0])
    at_corner = supervisor.command((waypoint[0], waypoint[1], corner_goal_bearing),
                                   (3.0, 0.0), [obstacle])
    assert at_corner is None, at_corner


def test_every_drive_of_the_executed_detour_passes_the_shared_veto():
    """The supervisor and the veto must never deadlock. Run the detour end to end
    under ideal kinematics and roll every drive-phase command through the planner's
    own ``is_feasible`` — the check ``mappo_drive`` applies before the command
    leaves. The bin carries the static hard gap because that is how ``visual_nav``
    builds landmarks; the veto's default is a PERSON's gap, and a landmark without
    the override is a configuration defect, not a scene this module must survive."""
    from avoidance import (
        STATIC_HARD_GAP_M,
        DynamicWindowPlanner,
        Limits,
        PlannerConfig,
    )
    supervisor = _supervisor()
    obstacle = Obstacle(x=1.5, y=0.0, vx=0.0, vy=0.0, radius_m=BIN_RADIUS_M,
                        label="bin", kind="static", object_id="landmark-1",
                        hard_gap_m=STATIC_HARD_GAP_M)
    planner = DynamicWindowPlanner(limits=Limits(),
                                   config=PlannerConfig(robot_radius_m=ROBOT_RADIUS_M))
    pose, ticks, drove = (0.0, 0.0, 0.0), 0, 0
    while ticks < 600:
        command = supervisor.command(pose, (3.0, 0.0), [obstacle])
        if command is None:
            break
        if command.phase == "drive":
            drove += 1
            assert planner.is_feasible(
                pose, (command.vx, command.vy, command.wz), [obstacle]), \
                f"veto refused a drive at {tuple(round(v, 3) for v in pose)}"
        x, y, yaw = pose
        yaw += command.wz * 0.1
        x += command.vx * math.cos(yaw) * 0.1
        y += command.vx * math.sin(yaw) * 0.1
        pose, ticks = (x, y, yaw), ticks + 1
    else:
        raise AssertionError("the detour never handed control back")
    assert drove > 0, "a detour with no drive phase is not the path under test"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"turn_drive_supervisor: {len(tests)}/{len(tests)} passed")
