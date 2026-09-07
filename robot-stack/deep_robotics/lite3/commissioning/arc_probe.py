#!/usr/bin/env python3
"""Measure the arc the Lite3 actually traces when forward and yaw fire TOGETHER.

Both axes are individually evidenced -- forward_positive 0.536 m/s, yaw +/-0.857 rad/s --
but they were measured IN ISOLATION, one axis nonzero at a time. `executed_velocity`
assumes they sum independently and nothing has ever checked that on this robot. If the
vendor gait blends or attenuates a combined command, the predicted 0.63 m turn radius is
wrong, and `--heading-servo goal` would be steering with a number nobody verified.

Refuses rather than defaults, like every other probe here: preflight, force-control gate,
and an operator-stated clear radius with no default.
"""
import argparse, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from deep_robotics.lite3.commissioning import robot_link
from deep_robotics.lite3.commissioning.measurement import (
    Refusal, brief, refuse_unmeasured, require_positive_finite, run_main)

TICK_S = 0.05


def plan_radius_m(fwd_m_s: float, yaw_rad_s: float) -> float:
    """The radius the two measured speeds PREDICT, which is the thing being tested."""
    return fwd_m_s / yaw_rad_s if yaw_rad_s else math.inf


def build_parser() -> argparse.ArgumentParser:
    """Separate from :func:`main` so a test can read the flags without running anything."""
    p = argparse.ArgumentParser(description=__doc__)
    robot_link.add_link_arguments(p, moving=True)
    p.add_argument("--seconds", type=float, default=None,
                   help="how long to hold forward+yaw. No default: it sets how far the "
                        "robot travels and only you know the room")
    p.add_argument("--clear-radius-metres", type=float, default=None,
                   help="clear space in EVERY direction. No default. The robot arcs, so "
                        "this is not a lane -- it sweeps a curve on one side")
    p.add_argument("--yaw-sign", type=int, choices=(1, -1), default=None,
                   help="+1 turns the robot LEFT (navigator +yaw), -1 RIGHT")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    robot_link.require_sign_only_transport(
        args, measures="the arc a combined forward+yaw command actually traces",
        instead="one axis at a time, with axis_primitive_probe.py")
    robot_link.require_walked_transport(args)
    refuse_unmeasured(**{"--seconds": args.seconds,
                         "--clear-radius-metres": args.clear_radius_metres,
                         "--yaw-sign": args.yaw_sign})
    require_positive_finite(**{"--seconds": args.seconds,
                               "--clear-radius-metres": args.clear_radius_metres})

    # ⚠️ THE PROFILE IS READ BEFORE ANY SOCKET, and the dry run returns before one is
    # opened. `axis_primitive_probe` has the same shape for the same reason: a run that
    # prints its plan must not have touched the robot to do it.
    profile = robot_link.load_axis_profile(args)
    fwd = profile.measured_speeds.get("forward_positive")
    yaw = profile.measured_speeds.get(
        "yaw_negative" if args.yaw_sign > 0 else "yaw_positive")
    if profile is None:
        raise Refusal("this probe needs the sign-only axis transport and its profile")
    if fwd is None or yaw is None:
        raise Refusal("forward and yaw must BOTH be measured before their combination "
                      "can be checked against a prediction")
    predicted = plan_radius_m(fwd, yaw)
    travel = fwd * args.seconds
    sweep = math.degrees(yaw * args.seconds)
    side = "LEFT" if args.yaw_sign > 0 else "RIGHT"

    brief("Lite3 combined forward+yaw arc",
          does=(f"holds vx>0 and yaw {side} together for {args.seconds:.1f}s. Predicted: "
                f"{travel:.2f} m of travel, {sweep:.0f} deg of turn, radius "
                f"{predicted:.2f} m -- ALL from two speeds measured SEPARATELY."),
          needs=[f"{args.clear_radius_metres:.1f} m clear in EVERY direction -- this arcs",
                 "the robot STANDING in force-control 6",
                 "your hand on the emergency stop"],
          means="the measured radius versus the predicted one. If they differ, "
                "`executed_velocity` is wrong about this robot and --heading-servo goal "
                "would steer on a number that does not hold.",
          moves=True)
    if travel > args.clear_radius_metres:
        raise Refusal(
            f"this would travel {travel:.2f} m along the arc and you have stated "
            f"{args.clear_radius_metres:.1f} m. Lower --seconds.")
    if not args.live:
        print("[arc] DRY RUN. Nothing opened, nothing commanded.")
        return 0

    link = robot_link.connect(args)
    loco = link.locomotion
    robot_link.preflight(link, args)
    loco.prepare_motion()
    start = loco.pose()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            loco.set_velocity(0.30, 0.0, args.yaw_sign * 0.5)
            time.sleep(TICK_S)
    finally:
        loco.stop()
        time.sleep(0.4)
        end = loco.pose()
        loco.shutdown()
    dx, dy = end.x - start.x, end.y - start.y
    chord = math.hypot(dx, dy)
    dyaw = math.atan2(math.sin(end.yaw - start.yaw), math.cos(end.yaw - start.yaw))
    print(f"\n[arc] chord {chord:.3f} m   yaw change {math.degrees(dyaw):+.1f} deg")
    if abs(dyaw) > 1e-3:
        measured = chord / (2.0 * math.sin(abs(dyaw) / 2.0))
        print(f"[arc] MEASURED radius {measured:.3f} m   PREDICTED {predicted:.3f} m "
              f"({(measured / predicted - 1) * 100:+.0f}%)")
        print(f"[arc] measured yaw rate {dyaw / args.seconds:+.3f} rad/s "
              f"(isolated measurement was {yaw:.3f})")
    else:
        print("[arc] the robot did not turn. A combined command may not blend on this "
              "firmware -- that is a result, and it refutes the prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(main, "arc"))
