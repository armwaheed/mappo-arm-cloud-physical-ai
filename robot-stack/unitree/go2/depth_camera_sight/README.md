# Unitree Go2 — Camera / depth sight (`depth_camera_sight`)

The Go2's forward vision. **Characterization is PENDING an on-robot capture** — this module is documented
but intentionally code-light until the cameras are calibrated on the robot (the descriptor marks the
`front_camera` / `vio_depth` sensors `PENDING`). It's the RGB-D counterpart to
[`lidar_sight`](../lidar_sight/), which is the Go2's primary perception.

## The two forward cameras

| Sensor | What it is | How it's read |
|---|---|---|
| **Front camera** | Built-in forward wide-angle (fisheye) camera | Unitree VideoHub over DDS/WebRTC (`rt/api/videohub/request`, `frontvideostream`) — undistort before use |
| **VIO depth** | Add-on Intel RealSense feeding the SVO visual-odometry `Odometer_service` | ROS1 `realsense2_camera` on the nav Jetson; VIO odometry on `rt/uslam/*` / `lio_sam_ros2/mapping/odometry` |

## To characterize (on the robot)

Mirror the G1's [`depth_camera_sight`](https://github.com/arm/arm-dc-unitree-g1/blob/main/unitree/g1/depth_camera_sight/) method:

1. Capture RGB (VideoClient) + depth (RealSense) frames; save into [`images/`](images/) as the
   descriptor's `calibration.reference_media`.
2. Calibrate the **down-tilt** by a floor-plane SVD fit (as the G1 does for its head cam), and the mount
   `xyz_m` relative to the head link.
3. Note the fisheye distortion / undistort parameters for the front camera.
4. Write the measured `pose` / `fov` / `calibration` back into the `front_camera` and `vio_depth` sensor
   entries and flip them from `PENDING`.

Until then, use [`lidar_sight`](../lidar_sight/) — the L1 dome LiDAR is the Go2's primary, room-cropped
perception, and it is built and tested.

## Reference media

Expected in [`images/`](images/) after an on-robot capture: an **RGB** frame, a **depth** frame, and a
scene close-up — the "what a calibrated Go2 RGB-D view looks like" standard. See the
[test-scene overview](../images/) for the environment.
