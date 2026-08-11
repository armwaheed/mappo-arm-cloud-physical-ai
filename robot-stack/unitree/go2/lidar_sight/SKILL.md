---
name: unitree-go2-sense-lidar
description: >-
  Perceive with the Unitree Go2's built-in L1 dome LiDAR. Subscribes to the onboard cloud
  (rt/utlidar/cloud_deskewed) and returns clean body-frame geometry: self-returns dropped and — the
  headline step — CROPPED TO THE ROOM so an open doorway doesn't leak corridor returns into the map.
  Produces a top-down occupancy grid and feeds lib/navigation for A* planning. Unlike the G1 (no onboard
  SLAM), the Go2 also publishes its own LiDAR-SLAM odom + height/grid maps (rt/uslam/*, rt/utlidar/*),
  so prefer those for production mapping. Use to detect obstacles/free space, crop scans to a room, or
  fill a descriptor's LiDAR sensor block.
metadata:
  tags: [unitree-go2, lidar, perception, point-cloud, room-crop, occupancy, navigation, slam]
---

# Unitree Go2 — LiDAR sight (onboard L1 → body frame, cropped to the room)

Turn the Go2's onboard LiDAR cloud into clean body-frame geometry. The cloud topic, the room-crop, the
occupancy grid, and the SLAM-reuse guidance are in **[`README.md`](README.md)** — this skill is the agent
entry point. Module: [`go2_lidar_sight.py`](go2_lidar_sight.py); planning in
`lib/navigation.py` (arm-dc-robotkit `lib/navigation.py`).

## When to use

- You need **obstacles / free space** in the robot frame to plan a path (feed the cropped cloud to the
  `Navigator`).
- You are scanning **indoors with an open door** and must **crop the scan to the room** so corridor
  returns don't pollute the map (the headline reason this module exists).
- You're filling a descriptor's LiDAR `sensor` block (mount / FOV / occlusions / reference scans).

## How to use

```python
import sys; sys.path.insert(0, "unitree/go2/lidar_sight")
from go2_lidar_sight import Go2LidarSight, RoomBounds

# set bounds to the WALL lines; give the doorway edge the wall's coordinate, not the far return's
room = RoomBounds(x_min=-2.0, x_max=3.0, y_min=-2.0, y_max=2.0, door_note="south doorway on +x")
sight = Go2LidarSight(iface="eth0", bounds=room); sight.connect()
cloud = sight.cropped_cloud()          # (N,3), self-returns dropped + room-cropped

import sys; sys.path.insert(0, "lib")
from navigation import Navigator
Navigator().navigate(loco, goal_xy=(2.0, 0.0), cloud_source=sight.cropped_cloud)
```

The geometry (`crop_to_room`, `drop_self_returns`, `room_occupancy`) is pure numpy and unit-tested
(`python3 test_go2_lidar_sight.py`, incl. a synthetic doorway-leak case). `_capture_scene.py` saves the
top-down + side scans and a range histogram into `images/` for the descriptor's reference media.

## Note on the room crop

An open door defeats a naive occupancy grid: the dome sees straight through it and paints phantom free
space + stray obstacles beyond the walls. Always `crop_to_room(...)` first, bounding the doorway edge at
the **wall line**. Capture the CROPPED cloud for any reference scan.
