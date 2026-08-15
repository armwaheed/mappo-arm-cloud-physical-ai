#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the shared RGB-only visual navigator on a Deep Robotics Lite3 Venture."""

from __future__ import annotations

import sys
from pathlib import Path

_ROBOT_STACK = Path(__file__).resolve().parents[3]
_COMMON = _ROBOT_STACK / "unitree" / "go2" / "visual_nav"
sys.path.insert(0, str(_ROBOT_STACK))
sys.path.insert(0, str(_COMMON))

import visual_nav
from deep_robotics.lite3.visual_nav.robot_bindings import Lite3Bindings


def main(argv=None) -> None:
    visual_nav.main(argv=argv, bindings=Lite3Bindings())


if __name__ == "__main__":
    main()
