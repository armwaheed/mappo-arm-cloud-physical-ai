"""Can MAPPO avoid a MOVING peer at all, given perfect information?

The falsifier for every sensing route at once. Mesh pose-sharing, a fine-tuned detector and
a LiDAR differ only in how accurately they report where the peer is. This hands the policy
the peer's EXACT position every tick — better than any of them could manage — and asks
whether the policy avoids it. If it cannot with perfect data, no amount of sensing helps.

The known risk it measures: the delivered checkpoint's 18-value observation is
``[x, y, vx, vy, x-gx, y-gy, *12 lidar]``. There is no channel for an OBSTACLE's velocity,
so a moving peer enters as an instantaneous disc wherever it happens to be. This quantifies
what that costs across crossing speeds.

Paired ablated control: every run is repeated with the peer removed, same seed and same
policy state, so deflection is attributed to the peer rather than to the policy wandering —
the same discipline replay_mappo uses.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "integration"))

from mappo_policy import HeadingServo, PolicyRunner, tick_from_state   # noqa: E402

DT = 0.1                     # 10 Hz control, matching the stack
MAX_VX, MAX_VY = 0.35, 0.20  # the shipped envelope
DELIVERY = 0.45              # this Go2 delivers ~0.45 of commanded; measured
PEER_R, ROBOT_R = 0.35, 0.25 # Go2 Wheel half-extent; the robot's planning radius


class Peer:
    """A disc crossing the robot's path left to right at constant speed."""
    def __init__(self, speed, cross_x=2.0, start_y=-1.6):
        self.speed, self.x, self.y = speed, cross_x, start_y
    def step(self):
        self.y += self.speed * DT


def run(peer_speed, with_peer=True, goal=(4.0, 0.0), max_s=30.0):
    runner = PolicyRunner(servo=HeadingServo(max_wz=0.7))
    runner.reset()
    x, y, yaw = 0.0, 0.0, 0.0
    vx = vy = 0.0
    peer = Peer(peer_speed)
    closest, headings, collided = float("inf"), [], False
    for step in range(int(max_s / DT)):
        t = step * DT
        obstacles = []
        if with_peer:
            obstacles.append({"x": peer.x, "y": peer.y, "radius_m": PEER_R,
                              "kind": "static", "id": "peer",
                              "vx": 0.0, "vy": peer.speed, "label": "peer"})
        tick = tick_from_state(t, (x, y, yaw), goal, obstacles,
                               measured=(vx, vy, 0.0), reason="policy")
        out = runner.step(tick)   # let the policy stamp its own monotonic clock
        if out is None or out.status != "COMMAND":
            cvx = cvy = cwz = 0.0
        else:
            cvx = max(0.0, min(MAX_VX, out.vx_mps))
            cvy = max(-MAX_VY, min(MAX_VY, out.vy_mps))
            cwz = out.wz_radps
        # Body-frame command -> odom motion, at the measured delivery ratio.
        vx, vy = cvx * DELIVERY, cvy * DELIVERY
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * DT
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * DT
        yaw += cwz * DT
        headings.append(math.atan2(cvy, cvx) if math.hypot(cvx, cvy) > 0.02 else 0.0)
        if with_peer:
            peer.step()
            gap = math.hypot(x - peer.x, y - peer.y) - PEER_R - ROBOT_R
            closest = min(closest, gap)
            collided |= gap <= 0.0
        if math.hypot(goal[0] - x, goal[1] - y) < 0.8:
            return {"arrived": True, "closest": closest, "collided": collided,
                    "headings": headings, "t": t}
    return {"arrived": False, "closest": closest, "collided": collided,
            "headings": headings, "t": max_s}


print("MAPPO vs a crossing peer, with the peer's EXACT position every tick.")
print("Paired ablated control (peer removed) attributes the deflection.\n")
print(f"{'peer m/s':>9} {'arrived':>8} {'collided':>9} {'closest gap':>12} "
      f"{'mean deflection':>16}")
for speed in (0.0, 0.10, 0.20, 0.35, 0.50, 0.70):
    live = run(speed, with_peer=True)
    blind = run(speed, with_peer=False)
    n = min(len(live["headings"]), len(blind["headings"]))
    deflect = (sum(abs(live["headings"][i] - blind["headings"][i]) for i in range(n)) / n
               if n else 0.0)
    print(f"{speed:>9.2f} {str(live['arrived']):>8} {str(live['collided']):>9} "
          f"{live['closest']:>11.3f}m {math.degrees(deflect):>15.1f}deg")
