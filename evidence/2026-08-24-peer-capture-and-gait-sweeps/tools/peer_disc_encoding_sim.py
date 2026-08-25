"""Can the policy avoid a moving peer if the DISC encodes where it is going?

The checkpoint's 18-value observation has no obstacle-velocity channel, so a crossing peer
enters as an instantaneous disc and — measured — the policy responds by stopping, with a
peak lateral command of 0.108 m/s that is below the gait floor and would not execute.

But the policy does not need a velocity channel to react to motion if the DISC ITSELF
carries it. Inflating an obstacle along its direction of travel by `speed * horizon` makes
the ray cast report where the peer WILL be. The planner's own rollout already reasons this
way in `_gaps`; this asks whether the policy can use the same idea through the only input
it has.

Two variants, because the cheap one may be enough:
  * ISOTROPIC — grow the radius by `speed * horizon`. Trivial, but it also inflates
    backwards, so a peer that has already crossed keeps pushing the robot away.
  * SWEPT — place the disc at the peer's PREDICTED position, radius grown by half the
    travel. Directional, and it stops mattering once the peer is past.

Paired ablated control on every run, same as the baseline measurement.
"""
import math, sys
S = "/private/tmp/claude-501/-Users-wahbro01-workspaces-git/ae5beebd-3312-48c6-92c7-3538b392af3f/scratchpad"
sys.path.insert(0, f"{S}/wt-peersim/integration")
from mappo_policy import PolicyRunner, HeadingServo, tick_from_state

DT, MAX_VX, MAX_VY, DELIVERY = 0.1, 0.35, 0.20, 0.45
PEER_R, ROBOT_R = 0.40, 0.25          # 0.40 = half-DIAGONAL, per NavConfig.robot_radius_m
LATERAL_FLOOR = 0.20


def encode(peer_x, peer_y, speed, mode, horizon):
    """The disc the policy is shown."""
    if mode == "none" or speed == 0.0:
        return peer_x, peer_y, PEER_R
    if mode == "isotropic":
        return peer_x, peer_y, PEER_R + speed * horizon
    travel = speed * horizon                       # swept
    return peer_x, peer_y + travel / 2.0, PEER_R + travel / 2.0


def run(peer_speed, mode, horizon, with_peer=True):
    r = PolicyRunner(servo=HeadingServo(max_wz=0.7)); r.reset()
    x = y = yaw = vx = vy = 0.0
    py = -1.6
    closest, collided, headings, lat = float("inf"), False, [], 0.0
    for step in range(300):
        obs = []
        if with_peer:
            ox, oy, orad = encode(2.0, py, peer_speed, mode, horizon)
            obs.append({"x": ox, "y": oy, "radius_m": orad, "kind": "static",
                        "id": "peer", "vx": 0.0, "vy": peer_speed, "label": "peer"})
        out = r.step(tick_from_state(step * DT, (x, y, yaw), (4.0, 0.0), obs,
                                     measured=(vx, vy, 0.0), reason="policy"))
        if out is None or out.status != "COMMAND":
            cvx = cvy = cwz = 0.0
        else:
            cvx = max(0.0, min(MAX_VX, out.vx_mps))
            cvy = max(-MAX_VY, min(MAX_VY, out.vy_mps))
            cwz = out.wz_radps
        lat = max(lat, abs(cvy))
        vx, vy = cvx * DELIVERY, cvy * DELIVERY
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * DT
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * DT
        yaw += cwz * DT
        headings.append(math.atan2(cvy, cvx) if math.hypot(cvx, cvy) > 0.02 else 0.0)
        if with_peer:
            # Clearance is measured against the TRUE disc, never the inflated one.
            closest = min(closest, math.hypot(x - 2.0, y - py) - PEER_R - ROBOT_R)
            collided |= closest <= 0.0
            py += peer_speed * DT
        if math.hypot(4.0 - x, -y) < 0.8:
            return dict(arrived=True, closest=closest, collided=collided,
                        headings=headings, lat=lat)
    return dict(arrived=False, closest=closest, collided=collided,
                headings=headings, lat=lat)


print("Clearance is measured against the TRUE peer disc in every row; only what the")
print("policy is SHOWN changes.\n")
for mode, horizon in (("none", 0.0), ("isotropic", 1.5), ("swept", 1.5), ("swept", 2.5)):
    tag = mode if mode == "none" else f"{mode} h={horizon}"
    print(f"=== {tag} ===")
    print(f"{'peer m/s':>9} {'arrived':>8} {'hit':>5} {'closest':>9} "
          f"{'peak |vy|':>10} {'walkable':>9} {'deflect':>9}")
    for speed in (0.10, 0.20, 0.35, 0.50):
        live = run(speed, mode, horizon)
        blind = run(speed, mode, horizon, with_peer=False)
        n = min(len(live["headings"]), len(blind["headings"]))
        d = (sum(abs(live["headings"][i] - blind["headings"][i]) for i in range(n)) / n
             if n else 0.0)
        print(f"{speed:>9.2f} {str(live['arrived']):>8} {str(live['collided']):>5} "
              f"{live['closest']:>8.3f}m {live['lat']:>10.3f} "
              f"{'yes' if live['lat'] >= LATERAL_FLOOR else 'NO':>9} "
              f"{math.degrees(d):>8.1f}d")
    print()
