#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay the marker-fixed shadow through the turn-drive chain, tick for tick.

``shadow-marker-fixed-20260827T023516Z.jsonl`` is the real scene: the corrected green
marker holding the measured bin in the static map for 186 of 188 ticks, a chair goal
3.46 m out, and a person wandering in from 6.987 s. The run itself drove NOTHING
(``live=false``) and predates the execution supervisor, so its 24 ``veto-hold`` ticks
are the question this script answers: **what does the chain do to those ticks now?**

Every tick is re-planned through the real checkpoint and the real ``MappoPlanner``
with ``--execution-supervisor turn-drive`` engaged, at the run's own configuration
(robot radius 0.40 m, envelope 0.55 / 0 / 0.90, policy scale 4.0 — the value the
scale gate computes for this radius and this checkpoint). The recorded command is
never fed back; the comparison is recorded-vs-replanned, same scene, same policy.

Run: ``python3 replay_with_supervisor.py``
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO / "integration"))
sys.path.insert(0, str(_REPO / "robot-stack" / "unitree" / "go2" / "visual_nav"))

from avoidance import STATIC_HARD_GAP_M, Limits, Obstacle, PlannerConfig
from mappo_drive import MappoPlanner
from mappo_policy import PolicyRunner
from replay_mappo import derived_config
from telemetry_reader import read_run
from turn_drive_supervisor import TurnDriveSupervisor

SHADOW = _HERE / "shadow-marker-fixed-20260827T023516Z.jsonl"

#: The run's own header values, restated so the replay's configuration is visible
#: here rather than inherited from a file the reader has to open.
ROBOT_RADIUS_M = 0.40
ENVELOPE = (0.55, 0.0, 0.90)          # max_vx / max_vy / max_wz, from the header
POLICY_SCALE_M_PER_UNIT = 4.0         # 0.40 m / 0.1 VMAS — what the scale gate asks


class _StationaryLoco:
    """The shadow moved nothing, so the measured velocity is zero throughout."""

    def velocity(self):
        return (0.0, 0.0, 0.0)


def _obstacle(record: dict) -> Obstacle:
    """Rebuild the planner's obstacle from its telemetry record.

    The static hard gap is reattached here because the telemetry carries the radius
    but not the gap override — ``visual_nav`` sets it at production time, and the
    replay must ask the veto the same question the run did.
    """
    static = record["kind"] == "static"
    return Obstacle(x=record["x"], y=record["y"], vx=record["vx"],
                    vy=record["vy"], radius_m=record["radius_m"],
                    label=record.get("label", ""),
                    person_shaped=record.get("person_shaped", not static),
                    kind=record["kind"], object_id=record.get("id"),
                    hard_gap_m=STATIC_HARD_GAP_M if static else None)


def main() -> int:
    run = read_run(SHADOW)
    planner_config = run.header["planner"]
    assert math.isclose(planner_config["robot_radius_m"], ROBOT_RADIUS_M)

    package = _REPO / "policy"
    with derived_config(package / "config.json",
                        meters_per_vmas_unit=POLICY_SCALE_M_PER_UNIT) as config:
        runner = PolicyRunner(package, config=config)
        supervisor = TurnDriveSupervisor(robot_radius_m=ROBOT_RADIUS_M,
                                         drive_speed_m_s=ENVELOPE[0],
                                         turn_rate_rad_s=ENVELOPE[2])
        with contextlib.redirect_stdout(io.StringIO()):
            planner = MappoPlanner(
                Limits(max_vx=ENVELOPE[0], max_vy=ENVELOPE[1], max_wz=ENVELOPE[2]),
                PlannerConfig(robot_radius_m=ROBOT_RADIUS_M), runner,
                supervised=True, execution_supervisor=supervisor)
        planner.attach(_StationaryLoco())

        recorded, replayed = Counter(), Counter()
        supervisor_vetoed = []      # (t, person present) for each refused override
        early_holds = []            # policy vetoed while the supervisor declined
        last = (0.0, 0.0, 0.0)
        for tick in run.ticks:
            command_record = tick.get("command")
            if tick.get("goal") is None or tick["perception"]["stale"] \
                    or command_record is None:
                continue            # the planner was not consulted on this tick
            pose = (tick["pose"]["x"], tick["pose"]["y"], tick["pose"]["yaw"])
            goal = (tick["goal"]["x"], tick["goal"]["y"])
            obstacles = [_obstacle(o) for o in tick["obstacles"]]
            command = planner.plan(pose, goal, last, obstacles)
            last = (command.vx, command.vy, command.wz)
            recorded[command_record.get("reason", "none")] += 1
            replayed[command.reason] += 1
            decision = planner.decision_record() or {}
            person = any(o.kind == "tracked" for o in obstacles)
            if command.reason.startswith("veto-"):
                if decision.get("supervisor", {}).get("vetoed"):
                    supervisor_vetoed.append((tick["t"], person))
                else:
                    statics = [o for o in obstacles if o.kind == "static"]
                    early_holds.append((tick["t"], max(
                        (o.radius_m for o in statics), default=0.0)))

    print(f"shadow: {SHADOW.name}")
    print(f"ticks re-planned: {sum(replayed.values())}")
    print(f"recorded reasons : {dict(recorded)}")
    print(f"replayed reasons : {dict(replayed)}")
    print(f"supervisor commands the veto refused: {len(supervisor_vetoed)}, "
          f"every one with a person on the scene: "
          f"{all(p for _t, p in supervisor_vetoed)}")
    print(f"policy vetoes while the supervisor declined (inflated fresh landmark): "
          f"{[(round(t, 1), round(r, 3)) for t, r in early_holds]}")
    print(f"planner counts: {dict(planner.counts)}")

    # The claims this replay exists to make, as checks rather than prose:
    assert planner.counts["turn_drive"] > 0, \
        "the supervisor never engaged on the real scene"
    assert supervisor_vetoed and all(p for _t, p in supervisor_vetoed), \
        "a supervisor command was refused with NO mover on the scene — the veto " \
        "and the supervisor disagree on geometry, which is the deadlock"
    assert all(r > ROBOT_RADIUS_M for _t, r in early_holds), \
        "a supervisor-declined hold without an inflated landmark is unexplained"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
