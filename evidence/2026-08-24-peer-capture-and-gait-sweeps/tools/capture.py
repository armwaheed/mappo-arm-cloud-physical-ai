"""Capture frames from the Go2 front camera for detector training data.

Saves every NEW frame the camera publishes, plus a manifest line each. Deliberately
dumb: no detection, no labelling, no filtering. The operator moves the peer; the
value is in the coverage, and anything discarded here cannot be recovered later.
"""
import json, sys, time
import cv2

sys.path.insert(0, "/home/unitree/robotics-connect/unitree/go2/visual_nav")
from camera import Go2Camera

tag = sys.argv[1]
seconds = float(sys.argv[2])
out = "/home/unitree/peercap"

with Go2Camera() as cam, open(f"{out}/{tag}.jsonl", "w") as manifest:
    last, saved, t0 = -1, 0, time.time()
    while time.time() - t0 < seconds:
        f = cam.latest()
        if f is not None and f.seq != last:
            last = f.seq
            name = f"{tag}_{saved:04d}.jpg"
            cv2.imwrite(f"{out}/{name}", f.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest.write(json.dumps({"image": name, "seq": f.seq,
                                       "t": round(f.capture_time, 3)}) + "\n")
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved} frames, {time.time()-t0:.0f}s", flush=True)
        time.sleep(0.08)
    print(f"DONE {tag}: {saved} frames in {time.time()-t0:.0f}s", flush=True)
