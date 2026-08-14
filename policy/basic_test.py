#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke check: this machine can load the checkpoint and run one inference.

THIS IS AN INSTALL CHECK, NOT A TEST OF THE POLICY. It answers "is numpy here, does the
npz load, does one forward pass produce a finite action" — which is exactly what you want
to know on a robot you have just deployed to, and nothing more. It would pass with the
weights replaced by noise.

The behaviour is pinned by ``test_physical_ai_mappo.py``, and the mapping from real
telemetry by ``integration/test_mappo_bridge.py`` and ``integration/replay_mappo.py``.
Kept under its delivered name because ``deploy/install.sh`` and the package README both
call it.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from physical_ai_mappo import COMMAND, MappoController, RobotInput, StationaryObject

controller = MappoController(ROOT / "config.json")

# The calibration, printed rather than assumed: on a robot, "which config did I actually
# deploy" is the question this check is really being run to answer.
radius = controller.agent_radius_m
print(f"checkpoint       {Path(controller.cfg.model_path).name}")
print(f"trained on       {controller.actor.metadata.get('training_frames', '?')} frames, "
      f"{controller.actor.metadata.get('training_n_agents', '?')} agents")
print(f"scale            {controller.cfg.meters_per_vmas_unit} m per VMAS unit")
print(f"sensing horizon  {controller.cfg.lidar_range_m:.3f} m to the obstacle surface")
print(f"agent radius     {'unknown' if radius is None else f'{radius:.3f} m'} at that scale")
print(f"velocity frame   {controller.cfg.velocity_frame}")

# One dry-run inference step. No robot SDK is called and nothing moves.
result = controller.step(RobotInput(
    x_m=0.0,
    y_m=0.0,
    yaw_rad=0.0,
    vx_mps=0.0,
    vy_mps=0.0,
    goal_x_m=2.0,
    goal_y_m=0.0,
    stationary_objects=[
        StationaryObject(
            object_id="box-1",
            distance_m=0.45,
            bearing_rad=math.radians(15.0),
            radius_m=0.10,
        )
    ],
    external_hold=False,
    reset_run=True,
))

print(result)
assert result.status == COMMAND, result.status
assert -1.0 <= result.action_x <= 1.0
assert -1.0 <= result.action_y <= 1.0
assert math.isfinite(result.vx_mps) and math.isfinite(result.vy_mps)
print("PASS")
