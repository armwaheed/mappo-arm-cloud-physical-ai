# lidar_sight — reference scans (slot)

Captured by [`../_capture_scene.py`](../_capture_scene.py) **on the robot** (read-only). Expected files —
the descriptor's LiDAR `calibration.reference_media` points here:

| File | What it is |
|---|---|
| `scene_topdown.jpg` | Top-down (XY) scan, RAW vs ROOM-CROPPED side by side — the open **south** door's leak shown removed. |
| `lidar_near_xz.jpg` | Side (XZ) scan of the cropped cloud (floor → ceiling). |
| `range_hist.jpg` | Point-range histogram, raw vs cropped (the leak is the far tail the crop removes). |

**PENDING an on-robot capture.** Always capture the **cropped** cloud — a scan that leaks through the
open door is not a useful reference (see [`../README.md`](../README.md)).
