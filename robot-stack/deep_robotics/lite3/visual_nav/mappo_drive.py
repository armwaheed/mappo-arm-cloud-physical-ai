#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the supervised MAPPO drive path on a calibrated Lite3 Venture."""

from __future__ import annotations

import sys
from pathlib import Path

_ROBOT_STACK = Path(__file__).resolve().parents[3]
_REPOSITORY = _ROBOT_STACK.parent
_COMMON = _ROBOT_STACK / "unitree" / "go2" / "visual_nav"
sys.path.insert(0, str(_ROBOT_STACK))
sys.path.insert(0, str(_COMMON))
sys.path.insert(0, str(_REPOSITORY / "integration"))

from deep_robotics.lite3.visual_nav.robot_bindings import Lite3Bindings

import mappo_drive as common_drive


def main(argv=None) -> int:
    return common_drive.main(argv=argv, bindings=Lite3Bindings())


if __name__ == "__main__":
    raise SystemExit(main())
