"""Stand, capture, lie down — in ONE process, so the legs are loaded only while filming.

Standing is a cost on this robot, not a default: the D1 arm loads the hind legs
continuously and they heat up. The earlier protocol stood the robot, then ran a separate
capture, then a separate lie-down — each re-connecting DDS and re-latching the arm while
the legs were already bearing the load. This does the latch PRONE, stands, films, and
puts it straight back down.

The lie-down is in a `finally`: a capture that raises must not leave the robot standing.
"""
import json, sys, time
import cv2

sys.path.insert(0, "/home/unitree/robotics-connect/unitree/go2/visual_nav")
sys.path.insert(0, "/home/unitree/robotics-connect")
from camera import Go2Camera
from safety import (
    MOTOR_TEMP_WARN_C,
    ArmStowMonitor,
    HealthMonitor,
    latch_arm,
    lie_down,
    stand_up,
)
from unitree.go2.locomotion.go2_locomotion import Go2Locomotion

tag, seconds = sys.argv[1], float(sys.argv[2])
out = "/home/unitree/peercap"

# Stand only when the legs can afford it. The D1 loads the hind legs continuously, so
# a capture session heats them in a way an unloaded Go2 would not, and 'they feel hot'
# is not a number. This ceiling sits below the stack's own warning so a session stops
# climbing before anything downstream starts complaining.
STAND_CEILING_C = 50.0

loco = Go2Locomotion(iface="eth0")
loco.connect()
health = HealthMonitor(); health.start()
before = health.latest()
if before is not None:
    print("motors %.1fC before standing (ceiling %.0fC, stack warns at %.0fC)"
          % (before.max_motor_temp_c, STAND_CEILING_C, MOTOR_TEMP_WARN_C),
          flush=True)
    if before.max_motor_temp_c >= STAND_CEILING_C:
        raise SystemExit("REFUSING TO STAND: motor %s at %.1fC is at or above the "
                         "%.0fC ceiling. Leave it prone to cool."
                         % (before.hottest_motor, before.max_motor_temp_c,
                            STAND_CEILING_C))
arm = ArmStowMonitor(); arm.start()
latch = latch_arm(arm, iface="eth0")
if not latch.held:
    raise SystemExit("REFUSING: the D1 latch did not take")
print(f"latch drift {latch.drift_deg:.2f} deg — HELD", flush=True)

saved = 0
stood = time.time()
try:
    stand_up(loco)
    print("standing", flush=True)
    with Go2Camera() as cam, open(f"{out}/{tag}.jsonl", "w") as manifest:
        last, t0 = -1, time.time()
        while time.time() - t0 < seconds:
            f = cam.latest()
            if f is not None and f.seq != last:
                last = f.seq
                name = f"{tag}_{saved:04d}.jpg"
                cv2.imwrite(f"{out}/{name}", f.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                manifest.write(json.dumps({"image": name, "seq": f.seq,
                                           "t": round(f.capture_time, 3)}) + "\n")
                saved += 1
            time.sleep(0.05)
finally:
    lie_down(loco)
    after = health.latest()
    trend = ""
    if before is not None and after is not None:
        trend = " motors %.1fC -> %.1fC," % (before.max_motor_temp_c,
                                             after.max_motor_temp_c)
    print("DONE %s: %d frames,%s %.0fs standing — PRONE"
          % (tag, saved, trend, time.time() - stood), flush=True)
