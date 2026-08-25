"""Falsify the two gait-floor claims the code currently rests on.

SWEEP A — is the forward floor really 0.35, and is it per-tick or sustained?
    `MIN_GAIT_COMMAND_M_S = 0.35` is a lowest-OBSERVED value. Upstream issue #32 records
    54/54 ticks below it while the robot walked ~3 m, mean commanded 0.295, and a stall at
    a SUSTAINED 0.137 m/s for 4.1 s. If that holds, the floor is lower than 0.35 and the
    failure depends on duration, not on any single tick. This holds each speed steady and
    measures travel.

SWEEP B — does the floor ellipse hold at 45 degrees?
    Two endpoints are measured (0.35 forward, 0.20 lateral) and the curve between them is
    an INTERPOLATION nobody has tested. The ellipse predicts
        floor(theta) = 1 / hypot(cos(theta)/0.35, sin(theta)/0.20)
    which at 45 deg is 0.246 m/s. Commanding exactly that at each bearing asks the robot
    the question directly: does it walk, or not?

SAFETY, and none of it is optional:
  * `SportClient.Move` PERSISTS until the next command — there is no dead-man. Every exit
    path stops the robot, in a `finally`, and the robot is put prone after.
  * vx is never negative. The Go2 has no rear-facing sensing on this unit, and a
    substituted planner driving blind backwards is an open upstream defect (#30).
  * The D1 arm is latched while PRONE, before sport mode is selected.
  * Motors are checked before standing and reported after every trial.
  * `--dry` runs the whole sequence including timing and reporting but commands no motion,
    so the script can be validated before anything moves.
"""
import argparse, json, math, sys, time

sys.path.insert(0, "/home/unitree/robotics-connect/unitree/go2/visual_nav")
sys.path.insert(0, "/home/unitree/robotics-connect")
from safety import MOTOR_TEMP_WARN_C, ArmStowMonitor, HealthMonitor, latch_arm, lie_down, stand_up
from unitree.go2.locomotion.go2_locomotion import Go2Locomotion

FORWARD_FLOOR, LATERAL_FLOOR = 0.35, 0.20
SPEEDS = [0.10, 0.137, 0.175, 0.20, 0.25, 0.295, 0.35]
BEARINGS_DEG = [0.0, 22.5, 45.0, 67.5, 90.0]

#: Sweep C — find the TRUE floor at the bearings where the ellipse failed, by raising the
#: speed until the robot walks. Note these deliberately run PAST `max_vy` = 0.20: the
#: envelope is a software limit, and the question is what the hardware does. If the floor
#: at 90 degrees turns out to be above 0.20, then the shipped envelope's lateral region is
#: empty — the floor would exceed the ceiling and a pure strafe could never be walked.
#: Nothing here should be copied into Limits; it is a measurement, not a new envelope.
SWEEP_C_BEARINGS_DEG = [90.0, 67.5]
SWEEP_C_SPEEDS = [0.20, 0.25, 0.30, 0.35, 0.40]

#: Sweep D — REPEATABILITY, forward only. Sweep A walked at 0.100 and stalled at 0.137,
#: which is non-monotonic in commanded speed and therefore cannot be a property of the
#: gait. Either it is trial-to-trial noise or it is position, and a single pass down a
#: corridor cannot separate them because every trial lands somewhere new. Repeating the
#: same two speeds interleaved keeps both explanations in play and lets the numbers choose.
SWEEP_D_SPEEDS = [0.100, 0.137]
SWEEP_D_REPEATS = 3
STAND_CEILING_C = 50.0
REFRESH_HZ = 10.0


def ellipse_floor(theta):
    return 1.0 / math.hypot(math.cos(theta) / FORWARD_FLOOR, math.sin(theta) / LATERAL_FLOOR)


def trial(loco, vx, vy, seconds, dry):
    """Hold one velocity, return measured travel and mean estimator speed."""
    start = loco.pose()
    samples = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not dry:
            loco.set_velocity(vx, vy, 0.0)
        samples.append(math.hypot(*loco.velocity()[:2]))
        time.sleep(1.0 / REFRESH_HZ)
    if not dry:
        loco.stop()
    time.sleep(0.6)                      # let it settle before reading the end pose
    end = loco.pose()
    travel = math.hypot(end.x - start.x, end.y - start.y)
    return travel, (sum(samples) / len(samples) if samples else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true", help="no motion; validates the sequence")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds per trial")
    ap.add_argument("--sweep", choices=("a", "b", "c", "d", "z", "both"), default="both")
    ap.add_argument("--out", default="/home/unitree/peercap/gait_sweep.json")
    args = ap.parse_args()

    loco = Go2Locomotion(iface="eth0")
    loco.connect()
    health = HealthMonitor(); health.start()
    before = health.latest()
    if before is not None:
        print("motors %.1fC (ceiling %.0fC, warn %.0fC)"
              % (before.max_motor_temp_c, STAND_CEILING_C, MOTOR_TEMP_WARN_C), flush=True)
        if before.max_motor_temp_c >= STAND_CEILING_C and not args.dry:
            raise SystemExit("REFUSING: motors at or above the ceiling — let it cool")

    arm = ArmStowMonitor(); arm.start()
    latch = latch_arm(arm, iface="eth0")
    if not latch.held:
        raise SystemExit("REFUSING: the D1 latch did not take — hand-pose it flat")
    print("latch drift %.2f deg — HELD" % latch.drift_deg, flush=True)

    results = {"dry": args.dry, "hold_s": args.hold, "a": [], "b": []}
    try:
        if not args.dry:
            # Selecting a mode can itself make the robot shift — done deliberately,
            # once, with the area confirmed clear, and before anything is commanded.
            print("mode now: %r" % loco.ensure_sport_mode("normal"), flush=True)
        stand_up(loco)
        print("standing", flush=True)

        if args.sweep in ("a", "both"):
            print("\nSWEEP A — sustained forward speed", flush=True)
            print("%8s %10s %10s %8s" % ("cmd", "travel_m", "est_mps", "walked"), flush=True)
            for v in SPEEDS:
                travel, est = trial(loco, v, 0.0, args.hold, args.dry)
                walked = travel > 0.05
                results["a"].append({"cmd_vx": v, "travel_m": travel, "est_mps": est,
                                     "walked": walked})
                print("%8.3f %10.3f %10.3f %8s" % (v, travel, est, walked), flush=True)

        if args.sweep in ("b", "both"):
            print("\nSWEEP B — ellipse floor at each bearing", flush=True)
            print("%8s %8s %8s %10s %8s" % ("deg", "speed", "vx", "travel_m", "walked"),
                  flush=True)
            for deg in BEARINGS_DEG:
                th = math.radians(deg)
                speed = ellipse_floor(th)
                vx, vy = speed * math.cos(th), speed * math.sin(th)
                travel, est = trial(loco, vx, vy, args.hold, args.dry)
                walked = travel > 0.05
                results["b"].append({"bearing_deg": deg, "speed": speed, "vx": vx, "vy": vy,
                                     "travel_m": travel, "est_mps": est, "walked": walked})
                print("%8.1f %8.3f %8.3f %10.3f %8s"
                      % (deg, speed, vx, travel, walked), flush=True)
        if args.sweep == "c":
            print("\nSWEEP C — raise the speed until it walks, at the failing bearings",
                  flush=True)
            print("%8s %8s %8s %8s %10s %8s"
                  % ("deg", "speed", "vx", "vy", "travel_m", "walked"), flush=True)
            results["c"] = []
            for deg in SWEEP_C_BEARINGS_DEG:
                th = math.radians(deg)
                for speed in SWEEP_C_SPEEDS:
                    vx, vy = speed * math.cos(th), speed * math.sin(th)
                    travel, est = trial(loco, vx, vy, args.hold, args.dry)
                    walked = travel > 0.05
                    results["c"].append({"bearing_deg": deg, "speed": speed, "vx": vx,
                                         "vy": vy, "travel_m": travel, "est_mps": est,
                                         "walked": walked})
                    print("%8.1f %8.3f %8.3f %8.3f %10.3f %8s"
                          % (deg, speed, vx, vy, travel, walked), flush=True)
                    if walked:
                        print("         ^ floor at %.1f deg is at or below %.2f m/s"
                              % (deg, speed), flush=True)
                        break
        if args.sweep == "d":
            print("\nSWEEP D — repeatability of the two low forward speeds", flush=True)
            print("%8s %6s %10s %10s %8s"
                  % ("cmd", "rep", "travel_m", "est_mps", "walked"), flush=True)
            results["d"] = []
            for rep in range(SWEEP_D_REPEATS):
                for v in SWEEP_D_SPEEDS:      # interleaved, so drift hits both equally
                    travel, est = trial(loco, v, 0.0, args.hold, args.dry)
                    walked = travel > 0.05
                    results["d"].append({"cmd_vx": v, "rep": rep, "travel_m": travel,
                                         "est_mps": est, "walked": walked})
                    print("%8.3f %6d %10.3f %10.3f %8s"
                          % (v, rep, travel, est, walked), flush=True)
        if args.sweep == "z":
            # THE NULL CONTROL, and it should have been trial zero of every sweep.
            # Commands ZERO velocity, so any travel this records is not walking: it is
            # the robot settling out of stand_up(), plus estimator drift. Every "walk"
            # measured today at low speed was the FIRST trial after standing and landed
            # at 0.114-0.127 m, which is exactly the magnitude a settle would produce.
            # Without this row there is no way to tell a slow walk from standing up.
            print("\nSWEEP Z — zero command. Any travel here is NOT walking.", flush=True)
            print("%8s %6s %10s %10s" % ("cmd", "rep", "travel_m", "est_mps"), flush=True)
            results["z"] = []
            for rep in range(4):
                travel, est = trial(loco, 0.0, 0.0, args.hold, args.dry)
                results["z"].append({"rep": rep, "travel_m": travel, "est_mps": est})
                print("%8.3f %6d %10.3f %10.3f" % (0.0, rep, travel, est), flush=True)
    finally:
        loco.stop()                      # Move persists; this is not optional
        lie_down(loco)
        after = health.latest()
        if before is not None and after is not None:
            print("\nmotors %.1fC -> %.1fC" % (before.max_motor_temp_c,
                                               after.max_motor_temp_c), flush=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print("PRONE. results -> %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
