# Go2 test-scene reference media

The reference photos of the environment the Go2 stack was brought up in — a cluttered **robotics lab**
(the shared "what the test scene looks like" standard, mirroring `unitree/g1/images/`).

| Image | What it shows |
|---|---|
| `go2_test_scene_overview.jpg` | The Go2 (with the D1 arm stowed on its back) standing mid-room, facing the open **south** door. Walls N/E/W; standing desk, flight cases, and cabinets to the north; a humanoid + wheeled gantry to the west; battery packs + cabling to the east; patterned carpet. |
| `go2_walk_scene_southbound.jpg` | The Go2's SSW walk direction toward the south doorway (framed by the frosted-glass door) — the path a `walk_forward` follows, and why LiDAR scans must be [cropped to the room](../lidar_sight/README.md) so the open door doesn't leak corridor returns. |

Use these to understand the scene, the obstacle layout for navigation, and the room extent for the
LiDAR crop. They are **not** a substitute for on-robot sensor calibration — the descriptor keeps the
LiDAR/RGB-D poses `PENDING` until captured with `lidar_sight` / `depth_camera_sight` on the robot.

_Metadata stripped; the room is referred to only as "the robotics lab."_
