#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""How much lateral velocity does this Go2 deliver WHILE WALKING FORWARD?

    python3 lateral_floor_probe.py --live            # ~8 s, ~2.8 m forward

THE DISAGREEMENT THIS SETTLES. The lateral gait floor was measured on 2026-08-19 as a
PURE strafe from standstill: 0.15 m/s does not walk this robot and 0.20 does, three
repeats of three. Every design decision since has treated 0.20 as a hard floor on the
lateral axis, which is what produces the rule that a command must be at least
``atan(0.20/0.35)`` = 29.7 degrees off the nose before any sideways travel happens.

The live run of 2026-08-25 disagrees with that. Walking forward at 0.35 m/s the policy
commanded a mean ``vy`` of 0.130 — below the supposed floor — and the odometry reported a
mean measured ``vy`` of 0.054. Not zero. If lateral is delivered PROPORTIONALLY once the
gait is running, there is no floor on a diagonal at all and the 29.7-degree argument
dissolves, along with every conclusion built on it.

Those two measurements are not comparable: one is a standing start, the other is a robot
already in gait. This probe measures the second case directly.

METHOD, and why it is shaped this way.

* **Forward speed is held at the FORWARD floor** (0.35 m/s) for every segment, so the
  gait is running throughout and the only variable is ``vy``. Dropping ``vx`` would
  confound "lateral does not execute" with "the robot stopped walking".
* **Zero-lateral controls are interleaved between every treatment**, not recorded once at
  the start. Three of this project's gait conclusions were overturned by trial-order
  confounds — a warm robot walks differently from a cold one — and a contemporaneous
  control is the only thing that separates the treatment from the drift.
* **Displacement is measured from POSE, not from the velocity estimate.** The reported
  velocity on this unit has been caught reading 0.17 m/s of noise as signal; integrating
  pose over a whole segment averages that away.
* **Body frame, per segment.** The world-frame delta is rotated by the segment's mean yaw,
  so a robot that drifts in heading does not have its forward travel counted as lateral.

WHAT A RESULT LOOKS LIKE. If the floor is real, every ``vy`` below 0.20 delivers the same
lateral travel as the zero control and the plot is a step. If lateral is proportional,
delivered rises with commanded from the smallest setting and the plot is a line through
the origin. Anything non-monotonic is a confound, not a finding — stop and look for it.

⚠️ THE ROBOT CRABS SIDEWAYS INTO SPACE IT CANNOT SEE. The camera is an 85-degree forward
cone and there is no lateral sensing at all. Clear both sides of the lane, not just ahead.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Forward speed for every segment: the measured FORWARD gait floor, so the gait is
#: running the whole time and ``vy`` is the only thing that changes.
FORWARD_M_S = 0.35

#: Lateral commands to test, each preceded by a zero-lateral control. 0.20 is the pure
#: strafe floor, 0.05 is a quarter of it — well under anything previously believed to
#: move the robot at all.
LATERAL_STEPS = (0.20, 0.15, 0.10, 0.05)

#: Seconds per segment. At 0.35 m/s this is 0.35 m of forward travel each, so the whole
#: sequence is about 2.8 m — sized to a corridor rather than to statistical comfort.
SEGMENT_S = 1.0

#: Control period. Matches the stack's 10 Hz so the vendor gait sees the same command
#: cadence it does on a real run.
TICK_S = 0.1


def _body_delta(start, end) -> tuple[float, float]:
    """World displacement rotated into the body frame the segment was flown in.

    Uses the MEAN of the start and end yaw rather than either endpoint: over a segment
    that turns, either endpoint attributes part of the arc to the wrong axis, and the
    mean is the first-order correction.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    yaw = math.atan2(math.sin(start[2]) + math.sin(end[2]),
                     math.cos(start[2]) + math.cos(end[2]))
    return (dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw))


def _segment(loco, vx: float, vy: float, duration: float) -> dict:
    start = loco.pose()
    start_t = time.monotonic()
    while time.monotonic() - start_t < duration:
        loco.set_velocity(vx, vy, 0.0)
        time.sleep(TICK_S)
    end, elapsed = loco.pose(), time.monotonic() - start_t
    forward, lateral = _body_delta((start.x, start.y, start.yaw),
                                   (end.x, end.y, end.yaw))
    return {"commanded_vy": vy, "elapsed_s": elapsed,
            "forward_m": forward, "lateral_m": lateral,
            "forward_mps": forward / elapsed, "lateral_mps": lateral / elapsed,
            "yaw_change_deg": math.degrees(end.yaw - start.yaw)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="DANGER: actually walk. Without this nothing moves.")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--segment", type=float, default=SEGMENT_S)
    ap.add_argument("--steps", type=float, nargs="+", default=list(LATERAL_STEPS))
    args = ap.parse_args(argv)

    plan = []
    for vy in args.steps:
        plan.append(0.0)      # contemporaneous control, before every treatment
        plan.append(vy)
    forward_m = len(plan) * args.segment * FORWARD_M_S
    print(f"[probe] {len(plan)} segments x {args.segment}s at vx={FORWARD_M_S}")
    print(f"[probe] about {forward_m:.1f} m forward, plus lateral drift")
    if not args.live:
        print("[probe] DRY: pass --live to walk")
        return 0

    from locomotion.go2_locomotion import Go2Locomotion

    from safety import lie_down, stand_up

    loco = Go2Locomotion(iface=args.iface)
    loco.connect()
    rows = []
    try:
        # `safety.stand_up`, NOT `loco.stand()`. Getting up needs TWO calls — recover()
        # is RecoveryStand and is what lifts a prone robot, stand() is BalanceStand and
        # is what puts it in the mode that accepts Move. The first version of this probe
        # called only stand(), the robot stayed prone, every Move was silently ignored,
        # and the run reported 0.000 m/s on EVERY axis including forward — which reads
        # as "the floor is real and total" rather than as "nothing moved". That function
        # exists precisely because two callers had already drifted into private copies
        # of this sequence; this was the third.
        stand_up(loco)
        for vy in plan:
            row = _segment(loco, FORWARD_M_S, vy, args.segment)
            rows.append(row)
            print(f"  vy={vy:+.2f} -> forward {row['forward_mps']:+.3f} m/s  "
                  f"lateral {row['lateral_mps']:+.3f} m/s  "
                  f"yaw {row['yaw_change_deg']:+.1f} deg")
    finally:
        lie_down(loco)

    # A run in which the robot never walked reports 0.000 lateral at every setting,
    # which reads exactly like "the floor is real and total". Refuse to tabulate it.
    walked = [r for r in rows if abs(r["forward_mps"]) > 0.05]
    if len(walked) < len(rows):
        print()
        print(f"⚠️  REFUSING TO REPORT: {len(rows) - len(walked)} of {len(rows)} segments "
              f"had no forward motion, so the legs were not walking and a zero lateral "
              f"means nothing. Check the stand sequence and the motion mode.")
        return 1

    print()
    print(f"{'commanded vy':>13} {'lateral m/s':>12} {'vs control':>11} {'delivered':>10}")
    controls = [r["lateral_mps"] for r in rows if r["commanded_vy"] == 0.0]
    baseline = sum(controls) / len(controls) if controls else 0.0
    print(f"{'0.00 (control)':>13} {baseline:>12.3f} {'-':>11} {'-':>10}")
    for row in rows:
        vy = row["commanded_vy"]
        if vy == 0.0:
            continue
        net = row["lateral_mps"] - baseline
        print(f"{vy:>13.2f} {row['lateral_mps']:>12.3f} {net:>+11.3f} "
              f"{net / vy * 100:>9.0f}%")
    print()
    print("A STEP at 0.20 means the floor is real. A LINE through the origin means")
    print("lateral is proportional and there is no floor on a diagonal. Anything")
    print("non-monotonic is a confound — find it before believing the numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
