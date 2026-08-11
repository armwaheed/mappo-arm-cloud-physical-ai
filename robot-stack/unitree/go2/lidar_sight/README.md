# Unitree Go2 — LiDAR sight (`lidar_sight`)

Perceive with the Go2's built-in **L1 4D dome LiDAR**: acquire the onboard cloud, clean it, and — the
headline step — **crop it to the room** so an open doorway doesn't pollute the map. Module:
[`go2_lidar_sight.py`](go2_lidar_sight.py) → `Go2LidarSight` + pure geometry helpers.

## Where the data comes from

The Go2 runs its **own** LiDAR-SLAM, so — unlike the G1, which needs an off-board SLAM — the cloud, a
robot-frame height map, and an odom/pose estimate arrive on the bus:

| Topic | Contents |
|---|---|
| `rt/utlidar/cloud_deskewed` | motion-compensated point cloud (this module's input) |
| `rt/utlidar/height_map` | robot-frame height map |
| `rt/utlidar/robot_odom` / `robot_pose` | LiDAR odometry / pose |
| `rt/uslam/*`, `rt/api/slam_operate` | the onboard graph-SLAM map + relocalization service |

For production mapping/relocalization, prefer the robot's **onboard SLAM** (`rt/uslam/*`). This module is
for turning a single cloud into obstacles/free space for a quick planned traverse.

## The room crop — the reason this module exists

The dome LiDAR sees **straight through an open door**. A raw scan taken in a room with a doorway is
polluted by returns from the corridor/next room — far, sparse points that paint phantom free space and
stray obstacles outside the walls, wrecking any occupancy grid. So always crop first:

```python
from go2_lidar_sight import RoomBounds, crop_to_room, drop_self_returns
room = RoomBounds(x_min=-2.0, x_max=3.0, y_min=-2.0, y_max=2.0, door_note="south doorway on +x")
clean = crop_to_room(drop_self_returns(raw_cloud), room)   # doorway leak removed
```

Set the bounds to the **wall lines**; give the doorway edge the wall's coordinate, not the far return's.
In the robotics-lab test scene the Go2 stands facing the open **south** door, so that edge is the one to
clamp to the wall — otherwise scans run down the corridor beyond it. Capture the **cropped** cloud for
any reference scan (see `images/`).

## API

| Function / method | What it does |
|---|---|
| `Go2LidarSight(iface, bounds).connect()` | subscribe to `rt/utlidar/cloud_deskewed` |
| `.get_cloud()` | latest raw cloud, `(N,3)` XYZ |
| `.cropped_cloud()` | self-returns dropped + cropped to `bounds` |
| `crop_to_room(cloud, bounds)` | clip to the room footprint (pure) |
| `drop_self_returns(cloud, r)` | drop dome self-returns near the origin (pure) |
| `room_occupancy(cloud, bounds, cell_m)` | top-down boolean occupancy grid (pure) |
| `pointcloud2_to_xyz(msg)` | decode a `sensor_msgs/PointCloud2` by field offsets (pure) |

The pure helpers are unit-tested (`test_go2_lidar_sight.py`, 7 cases incl. a synthetic doorway-leak). The
`PointCloud2` IDL is resolved from `unitree_sdk2py` / a Go2 ROS2 bridge, or injected via `cloud_type=`.

## Reference media

`_capture_scene.py` saves, into [`images/`](images/): a **top-down** (XY) scan, a **side** (XZ) scan, and
a **range histogram** — before/after the room crop, so the doorway leak is visible. These are the
"what a calibrated Go2 scan looks like" references the descriptor's LiDAR `calibration.reference_media`
points at. They must be captured **on the robot** (pending — the descriptor marks the LiDAR envelope
`PENDING` until then).

## Try it

```bash
# read-only: raw vs room-cropped point counts (shows how much the doorway leaks)
python3 unitree/go2/lidar_sight/go2_lidar_sight.py --iface eth0 --room -2 3 -2 2 --seconds 6

# capture reference scans + histogram into images/ (needs matplotlib on the robot)
python3 unitree/go2/lidar_sight/_capture_scene.py --iface eth0 --room -2 3 -2 2
```
