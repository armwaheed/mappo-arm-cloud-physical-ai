#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the Lite3 Venture camera through the shared focal-length fitter."""

from __future__ import annotations

import sys
from pathlib import Path

_ROBOT_STACK = Path(__file__).resolve().parents[3]
_COMMON = _ROBOT_STACK / "unitree" / "go2" / "visual_nav"
sys.path.insert(0, str(_ROBOT_STACK))
sys.path.insert(0, str(_COMMON))

from deep_robotics.lite3.visual_nav.robot_bindings import Lite3Bindings

import calibrate_camera as common_calibration


def main(argv=None) -> None:
    common_calibration.main(argv=argv, bindings=Lite3Bindings())


if __name__ == "__main__":
    main()
