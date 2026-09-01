#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Draw — or re-check — the bearing error an equidistant model makes on the Lite3's lens.

The figure is derived from the COMMITTED rectification profile rather than from a notebook,
so the curve in the paper and the intrinsics the robot actually runs cannot drift apart.
``--check`` re-derives the three errors A19 quotes and exits non-zero if any has moved,
because a stale figure should be a failing command rather than a picture nobody re-ran.

    python3 make_lens_error_chart.py --check
    python3 make_lens_error_chart.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROFILE = (Path(__file__).resolve().parents[2] / "robot-stack" / "deep_robotics" / "lite3"
           / "visual_nav" / "lite3_front_camera_rectify_20260901.json")

#: The numbers Appendix A19 states, and therefore the ones a reader will check.
QUOTED = {30: 3.1, 40: 8.1, 50: 18.3}

#: Worst-case bearing error measured after rectification, over the same 26 views spanning
#: -51 to +28 degrees. Quoted in A19 beside the curve.
MEASURED_AFTER_DEG = 0.18


def error_at(degrees: float) -> float:
    """How far off an equidistant reading of a rectilinear lens is, at a true bearing.

    A rectilinear ray at theta lands at ``f*tan(theta)``. Read back through
    ``radius = f*theta`` — the shared model — that radius is reported as ``tan(theta)``
    radians. The focal length CANCELS, which is the whole point: this error is a property
    of the two projections, not of any calibration, so no choice of focal length reduces it.
    """
    theta = math.radians(degrees)
    return math.degrees(math.tan(theta)) - degrees


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="re-derive the quoted errors and fail if any has moved")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent
                        / "lite3-equidistant-bearing-error.png")
    args = parser.parse_args(argv)

    profile = json.loads(PROFILE.read_text())
    hfov = math.degrees(2.0 * math.atan(
        (profile["width"] / 2.0) / profile["camera_matrix"][0][0]))

    if args.check:
        bad = [(deg, want, error_at(deg)) for deg, want in QUOTED.items()
               if abs(error_at(deg) - want) > 0.05]
        for deg, want, got in bad:
            print(f"  MOVED: at {deg} deg the paper says +{want} deg, derived +{got:.2f}")
        if bad:
            return 1
        print(f"lens error chart: {len(QUOTED)} quoted values re-derived, all agree "
              f"(rectilinear HFOV {hfov:.1f} deg from {PROFILE.name})")
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    span = np.linspace(0.0, 55.0, 400)
    fig, axes = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    axes.plot(span, [error_at(v) for v in span], lw=2.4, color="#c1121f",
              label="equidistant model on this lens")
    axes.axhline(0.0, lw=1.6, color="#2a9d8f",
                 label=f"after rectification (measured: {MEASURED_AFTER_DEG} deg worst)")
    for deg in QUOTED:
        axes.plot([deg], [error_at(deg)], "o", color="#c1121f", ms=6)
        axes.annotate(f"+{error_at(deg):.1f}$\\degree$ at {deg}$\\degree$",
                      (deg, error_at(deg)), textcoords="offset points",
                      xytext=(-96, 4), fontsize=9, color="#c1121f")
    axes.axvspan(36, 42, color="#c1121f", alpha=0.08)
    axes.annotate("the demo box sat here", (39, 1.0), ha="center", fontsize=9,
                  color="#6c757d")
    axes.set_xlabel("true bearing off the optical axis (degrees)")
    axes.set_ylabel("bearing error (degrees)")
    axes.set_title(f"A {hfov:.0f}$\\degree$ rectilinear lens read through an "
                   f"equidistant model", fontsize=11)
    axes.grid(alpha=0.25)
    axes.legend(loc="upper left", fontsize=9)
    axes.set_xlim(0, 55)
    axes.set_ylim(-1, 22)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
