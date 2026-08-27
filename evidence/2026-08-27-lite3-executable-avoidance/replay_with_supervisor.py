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

SECOND PASS, ADDED WHEN #150 WAS REBASED PAST #149. The first pass builds
``Limits(...)`` and inherits its default transport, which is ``PROPORTIONAL`` — the
GO2's. Read as evidence about a Lite3 it is silent on the property #149 changed: that
this robot's legs receive a primitive's measured speed rather than the number the
planner typed. So the whole replay is run a second time under
``SignOnlyAxisTransport``, and the two reason streams are asserted EQUAL. They are,
tick for tick, because every command the supervisor emits is a pure turn or a pure
drive and that transport names both exactly. The forward speed is swept rather than
picked: no Lite3 has run ``axis_primitive_probe.py``, so there is no measured number
to use, and a claim that held only at one invented value would be a claim about the
value.

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
sys.path.insert(0, str(_REPO / "robot-stack"))

from avoidance import PROPORTIONAL, STATIC_HARD_GAP_M, Limits, Obstacle, PlannerConfig
from deep_robotics.lite3.locomotion.lite3_axis_locomotion import (
    AXIS_PROFILE_SCHEMA,
    AxisProfile,
    SignOnlyAxisTransport,
)
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


def _lite3_a_transport(forward_m_s: float) -> SignOnlyAxisTransport:
    """LITE3-A's shape, from `DEPLOYMENT-SOP.md`: forward and both yaw primitives
    evidenced, lateral and reverse `null`, `measured_rad_s` absent everywhere.

    ⚠️ ``forward_m_s`` IS NOT A MEASUREMENT OF ANY LITE3. Nothing has run
    ``axis_primitive_probe.py`` on one, which is why a live run refuses without the
    field. It is swept by the caller for that reason.
    """
    import json
    import tempfile

    data = {
        "schema": AXIS_PROFILE_SCHEMA,
        "input_deadband": {"linear_m_s": 0.05, "yaw_rad_s": 0.10},
        "allowed_gait_states": [0],
        "measured_m_s": {"forward_positive": forward_m_s},
        "measured_rad_s": {},
        "evidence": {"forward_positive": "lite3-a-forward",
                     "yaw_positive": "lite3-a-yaw-positive",
                     "yaw_negative": "lite3-a-yaw-negative"},
        "primitives": {"forward_positive": 32767, "forward_negative": None,
                       "lateral_positive": None, "lateral_negative": None,
                       "yaw_positive": 16000, "yaw_negative": -16000},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "lite3-a.json"
        path.write_text(json.dumps(data))
        return SignOnlyAxisTransport(AxisProfile.load(path))


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


def _replay(run, runner, transport):
    """One pass over the shadow, under one transport model. Returns the counters.

    A FRESH planner per pass: the counts, the acceleration window and both Schmitt
    triggers are per-run state, and reusing one would make the second pass a
    continuation of the first rather than a repeat of it.
    """
    supervisor = TurnDriveSupervisor(robot_radius_m=ROBOT_RADIUS_M,
                                     drive_speed_m_s=ENVELOPE[0],
                                     turn_rate_rad_s=ENVELOPE[2])
    with contextlib.redirect_stdout(io.StringIO()):
        planner = MappoPlanner(
            Limits(max_vx=ENVELOPE[0], max_vy=ENVELOPE[1], max_wz=ENVELOPE[2],
                   transport=transport),
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

    return recorded, replayed, supervisor_vetoed, early_holds, planner


def main() -> int:
    run = read_run(SHADOW)
    planner_config = run.header["planner"]
    assert math.isclose(planner_config["robot_radius_m"], ROBOT_RADIUS_M)

    package = _REPO / "policy"
    with derived_config(package / "config.json",
                        meters_per_vmas_unit=POLICY_SCALE_M_PER_UNIT) as config:
        runner = PolicyRunner(package, config=config)
        recorded, replayed, supervisor_vetoed, early_holds, planner = _replay(
            run, runner, PROPORTIONAL)

        # THE SECOND PASS: the same 186 ticks under the transport this robot has.
        # Swept, because nothing has measured its forward primitive. The claim is
        # that the reason stream does not move — see the module docstring.
        sign_only = {}
        for forward_m_s in (0.12, 0.30, 0.55):
            _rec, rep, vetoed, _early, plan_2 = _replay(
                run, runner, _lite3_a_transport(forward_m_s))
            sign_only[forward_m_s] = (rep, len(vetoed),
                                      plan_2.counts["transport_refused"])

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
    print()
    print("under SignOnlyAxisTransport — the transport this robot actually has:")
    for forward_m_s, (rep, vetoed, refused) in sign_only.items():
        print(f"  forward primitive {forward_m_s:.2f} m/s: reasons {dict(rep)}, "
              f"supervisor commands refused {vetoed}, transport refusals {refused}")

    # The claims this replay exists to make, as checks rather than prose:
    assert planner.counts["turn_drive"] > 0, \
        "the supervisor never engaged on the real scene"
    assert supervisor_vetoed and all(p for _t, p in supervisor_vetoed), \
        "a supervisor command was refused with NO mover on the scene — the veto " \
        "and the supervisor disagree on geometry, which is the deadlock"
    assert all(r > ROBOT_RADIUS_M for _t, r in early_holds), \
        "a supervisor-declined hold without an inflated landmark is unexplained"

    # ⚠️ AND THE ONE THE REBASE ADDED. #149 made every rollout a question about the
    # velocity the LEGS receive; #150 re-expresses avoidance as the turns and drives
    # these legs were measured to perform. If the second is executable at all, the
    # first must not refuse it — so the reason stream under the real transport has to
    # be the one above, tick for tick, at every forward speed the probe might return.
    for forward_m_s, (rep, vetoed, refused) in sign_only.items():
        assert rep == replayed, (
            f"the sign-only transport changed the decision stream at "
            f"{forward_m_s} m/s: {dict(rep)} vs {dict(replayed)} — the planner's "
            f"refusal is pre-empting the execution supervisor")
        assert vetoed == len(supervisor_vetoed), (forward_m_s, vetoed)
        assert refused == 0, (
            f"the transport refused {refused} ticks at {forward_m_s} m/s; on this "
            f"scene the supervisor has an answer for every one of them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
