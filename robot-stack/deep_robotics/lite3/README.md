<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Robotics Lite3 Venture: RGB-only demo port

This is the Lite3 Venture binding for the same visual navigator and MAPPO integration
used on the Go2. It assumes the two event robots have one forward RGB camera and **no
LiDAR**. Nothing in this path starts a LiDAR node or consumes a point cloud.

The code is offline-tested. It has not moved either event robot. Hardware commissioning
is tracked in [issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13)
and still has to supply the four measurements in the table below and the two health
topics; live mode fails closed when any of them is absent.

| platform item | implementation | hardware evidence |
| --- | --- | --- |
| high-level locomotion | vendor high-level UDP (default), or `Lite3_ROS` `/cmd_vel` + `/leg_odom2` | not yet run on either Venture |
| RGB capture | explicit V4L2 index, RTSP URI, or GStreamer pipeline | endpoint not yet supplied |
| gait floor | required as `--gait-floor` | not measured |
| actuator gain | required as `--actuator-gain` | not measured |
| loaded planning radius | required as `--robot-radius` | not measured |
| focal length / HFOV | Lite3-tagged calibration JSON required live | not measured |
| battery | read from the UDP state stream | not yet read on either Venture |
| motor temperatures | absent from the high-level interface | vendor question, still open |

## Which of the three vendor interfaces this uses, and why

Deep Robotics expose the same high-level gait controller two ways, and a third,
lower-level interface that this repository does not use.

**The high-level UDP interface — the default here.** The motion host accepts a 20-byte
`{int32 cmd_code, int32 size, int32 type, double data}` frame on port 43893, with code
`320` carrying forward velocity, `325` lateral and `321` yaw. It streams pose, body
velocity, IMU, joints, handle state and battery back to the single address configured in
`~/jy_exe/conf/network.toml`, on port 43897. `robot-stack/deep_robotics/lite3/locomotion/
lite3_udp_locomotion.py` speaks exactly that, and `--locomotion-transport udp` selects it.

**The `Lite3_ROS` bridge — the same thing, wrapped.** The official
[Lite3_ROS](https://github.com/DeepRoboticsLab/Lite3_ROS) `transfer` package exposes that
controller through a `geometry_msgs/msg/Twist` command and publishes pose plus measured
body velocity as `nav_msgs/msg/Odometry`. Its documented convention matches this stack:
+x forward, +y left, and +yaw left. Reading `Jetson2Motion.cpp`, its whole command path is
the three `sendto` calls above; it adds no heartbeat, no timer and no periodic
transmission. It runs on a *perception* host — the executable is named `jetson2motion` and
its `target_ip` defaults to the motion host — so it needs a ROS 2 Foxy runtime on a
computer these two Ventures may not have. `--locomotion-transport ros2` selects it where
a unit does ship it.

The official [Lite3_MotionSDK](https://github.com/DeepRoboticsLab/Lite3_MotionSDK) is a
different layer. Its example resets all 12 joints and gains low-level control; a caller
must provide joint targets, gains, balance, and a gait. This repository does not contain
a Lite3 balance controller, so using that SDK as a velocity-walk client would remove the
vendor controller that keeps the robot upright. The SDK is therefore intentionally not
used by this port.

The public ROS bridge requires the operator to stand the robot and select its high-level
navigation/AUTO mode. Its README describes an app transition, but
[Lite3_ROS issue #2](https://github.com/DeepRoboticsLab/Lite3_ROS/issues/2) reports that
some lower-spec units lack that app button and need firmware-specific `/simple_cmd`
codes plus a heartbeat. Those numeric codes are not a stable public API, so this binding
does not send them. Commissioning must get the approved sequence for each event robot;
`--operator-ready` records that the external transition succeeded. There is no
documented high-level stand-down verb, so cleanup publishes repeated zero velocities and
the operator returns the robot to manual/prone through the approved vendor interface.

## Read the robot before installing anything on it

`192.168.1.120` is the **motion host**, and it is not supposed to have ROS 2, a
`Lite3_ROS` checkout or outbound DNS. Start with
[`commissioning/`](commissioning/README.md): it decodes the state the robot already
transmits, in stdlib Python, and cannot command a leg. One capture with the operator
driving on the vendor remote supplies the gait floor, the actuator gain, the mode
transitions and the angular-velocity unit.

## Bring up the read-only side first

Read [`../../SAFETY.md`](../../SAFETY.md) before any hardware session. The commands in
this section inspect topics or CLI wiring and do not move a leg.

On the default UDP transport there are no topics to confirm and no robot core to
deploy — the commissioning probe is the read-only check, and it reports the frame rates
this stack depends on. For `--locomotion-transport ros2`, use the official `ros2-foxy`
Lite3_ROS branch on the perception computer, deploy the shared Device Connect robot core
so `arm_dc_robotkit.ros2_twist_locomotion` is importable, and confirm the documented
topics:

```bash
ros2 topic info /cmd_vel
ros2 topic echo --once /leg_odom2
ros2 topic hz /leg_odom2

cd robot-stack/deep_robotics/lite3/visual_nav
python3 lite3_visual_nav.py --help
python3 calibrate_camera.py --help
python3 mappo_drive.py --help
```

The Lite3 CLIs deliberately contain no D1 arm, latch, or Go2 motion-mode flags.

The camera source is explicit because neither the public bridge nor the beta perception
manual defines one endpoint for every Venture image. Examples are `--camera-source 0`
for V4L2, `--camera-source rtsp://HOST/PATH` for RTSP, or a pipeline plus
`--camera-gstreamer`. OpenCV timestamps a network frame when it is decoded, not when its
shutter fired. Measure end-to-end frame age on the installed endpoint; an RTSP decoder
queue can otherwise make fresh-looking frames old.

## The health feed

**Battery needs nothing extra any more.** `RobotState.battery_level` is in the high-level
UDP stream. The `Lite3_ROS` bridge drops that field, which is why the ROS path needed a
companion publisher for something the robot was already reporting; on the default UDP
transport the health monitor reads it from the same link locomotion uses. That path
imports no ROS at all.

**Motor temperatures are genuinely absent.** The high-level interface does not carry them
in any form. The low-level `Lite3_MotionSDK` reports them, but taking low-level control
merely to read a temperature would remove the vendor controller that keeps the robot
upright, so this port does not do that. The supported publisher is a vendor question and
is still open.

Until it is answered there are two ways to run:

| | motor temperatures | what runs |
| --- | --- | --- |
| default | required | dry navigation only; `--live` refuses |
| `--accept-no-motor-temperatures` | unmonitored | `--live` runs, bounded and recorded |

The override is an explicit operator decision, not a way to make the gate pass:

- battery and the two-second staleness gate stay enforced;
- `--max-seconds` is capped at 120 s;
- the pre-flight prints a banner and `warning_reason()` repeats it every tick;
- telemetry records `motor_temperatures_monitored: false`, so a recording cannot later be
  mistaken for a monitored run.

**It bounds one run and nothing more.** Heat builds across back-to-back runs and no
software here can see that. Let the robot cool between runs. Do not replace any of this
with constants.

For `--locomotion-transport ros2`, the two companion topics are still the path:
`sensor_msgs/msg/BatteryState` on `/battery_state` with `percentage` in the ROS standard
0..1 range, and `std_msgs/msg/Float64MultiArray` on `/motor_temperatures` with exactly 12
Celsius values. Both names are configurable.

## Measure and calibrate this robot

Do this independently for both event robots and keep the results with their robot ID.
None of the Go2 numbers are defaults here.

1. Measure the loaded plan-view planning radius, including the leg envelope and anything
   mounted for the event. The MAPPO scale is
   `robot_radius_m / 0.10`, where 0.10 is the checkpoint's trained VMAS agent radius.
2. With a clear lane and the app emergency stop held, find the lowest commanded forward
   speed that produces a sustained gait rather than a shuffle. Record that conservative
   working value as `--gait-floor`.
3. At the exact command envelope intended for the demo, divide mean measured forward
   velocity from `/leg_odom2` by mean commanded velocity. Record the ratio as
   `--actuator-gain`; use it when choosing `--max-seconds`. Do not interpolate across the
   gait floor.
4. Calibrate the installed RGB camera. A stationary marker calibration does not move the
   robot:

   ```bash
   python3 calibrate_camera.py --camera-source 0 \
       --marker MEASURED_CAMERA_TO_MARKER_M --out lite3_front_camera.json
   ```

   The odometry-based spin fit avoids a range measurement, but it moves the robot. Run
   it only after the Lite3 yaw deadband, health feed, clear area, and tether/remote plan
   are known:

   ```bash
   python3 calibrate_camera.py --camera-source 0 --spin --spin-target marker \
       --spin-rate MEASURED_WORKING_YAW_RAD_S --live --operator-ready \
       --record lite3-camera-calibration.mp4 --out lite3_front_camera.json
   ```

The spin fit uses changes in odometry pose yaw, not the bridge's instantaneous angular
velocity field. During commissioning, still compare that field with pose-yaw change over
time: the public bridge copies its vendor `rpy_vel` value into a ROS field without a unit
conversion, while the corresponding low-level SDK documents angular rate in degrees/s.
Do not use measured yaw rate downstream until that unit is confirmed on the installed
firmware.

## Run the common stack

First run perception and planning without `--live`; it can publish no non-zero velocity:

```bash
python3 lite3_visual_nav.py --camera-source 0 \
    --calibration lite3_front_camera.json --waypoint 2.0 0.0 \
    --record lite3-dry.mp4 --telemetry lite3-dry.jsonl
```

Only after the measurements and the repository's simulation/shadow ladder are complete,
the live visual navigator takes all platform values explicitly:

```bash
python3 lite3_visual_nav.py --camera-source 0 --live --operator-ready \
    --calibration lite3_front_camera.json --gait-floor MEASURED_GAIT_M_S \
    --actuator-gain MEASURED_GAIN --robot-radius MEASURED_RADIUS_M \
    --waypoint 2.0 0.0 --record lite3-live.mp4 --telemetry lite3-live.jsonl
```

The policy drive uses the same flags plus the radius-derived policy scale. Keep the
planner veto on; raw mode remains unsuitable:

```bash
python3 mappo_drive.py --camera-source 0 --live --operator-ready \
    --calibration lite3_front_camera.json --gait-floor MEASURED_GAIT_M_S \
    --actuator-gain MEASURED_GAIN --robot-radius MEASURED_RADIUS_M \
    --policy-scale RADIUS_DIVIDED_BY_0_10 --policy-mode supervised \
    --waypoint 2.0 0.0 --record lite3-drive.mp4 --telemetry lite3-drive.jsonl
```

The policy, telemetry schema, perception, tracking, static map, planner veto, and
closed-loop simulator are shared. Peer detection is not: a second Lite3 is still
invisible unless each robot is given an explicit visual marker or colour profile.

## Offline checks

```bash
cd robot-stack/deep_robotics/lite3/commissioning
python3 test_lite3_state_probe.py && ruff check .

cd ../locomotion
for test in test_*.py; do python3 "$test"; done
ruff check .

cd ../visual_nav
for test in test_*.py; do python3 "$test"; done
ruff check .
```

The attached vendor beta manual remains linked from GitHub issue #12 rather than copied
here; its LiDAR and depth-camera sections do not describe these two Venture robots.
