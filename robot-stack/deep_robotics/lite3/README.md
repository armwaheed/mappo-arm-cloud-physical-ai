<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Robotics Lite3 Venture: RGB-only demo port

This is the Lite3 Venture binding for the same visual navigator and MAPPO integration
used on the Go2. It assumes the two event robots have one forward RGB camera and **no
LiDAR**. Nothing in this path starts a LiDAR node or consumes a point cloud.

The visual-navigation and MAPPO path is offline-tested. A separate, bounded vendor
high-level locomotion proof moved one event robot on 2026-08-24; it does **not** authorize
the visual-navigation or MAPPO path to move a robot. Hardware commissioning is tracked in
[issue #13](https://github.com/armwaheed/mappo-arm-cloud-physical-ai/issues/13) and still
has to supply the four measurements in the table below and the two health topics; live mode
fails closed when any of them is absent.

| platform item | implementation | hardware evidence |
| --- | --- | --- |
| high-level locomotion | bounded vendor moving-mode axis UDP, legacy complex UDP, or `Lite3_ROS` `/cmd_vel` + `/leg_odom2` | one bounded axis-forward proof moved 0.401 m and stopped cleanly; generic velocity mapping remains unverified |
| RGB capture | explicit V4L2 index, RTSP URI, or GStreamer pipeline | endpoint not yet supplied |
| gait floor | required as `--gait-floor` | not measured |
| actuator gain | required as `--actuator-gain` | not measured |
| loaded planning radius | required as `--robot-radius` | not measured |
| focal length / HFOV | Lite3-tagged calibration JSON required live | not measured |
| battery | documented legacy `RobotState` UDP field | 21% after the 2026-08-21 vendor-service restart |
| motor temperatures | absent from the high-level interface | vendor question, still open |

## Which vendor interfaces this uses, and why

Deep Robotics expose the same high-level gait controller two ways, and a third,
lower-level interface that this repository does not use.

**The legacy complex-velocity UDP interface — the default offline binding.** The motion host accepts a 20-byte
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

### Bounded direct velocity proof

[`locomotion/lite3_velocity_udp.py`](locomotion/lite3_velocity_udp.py) is a manual,
bounded reproduction of the three outbound frames in `Lite3_ROS` `Jetson2Motion.cpp`; it
does not import ROS or decode state. On the event motion host, the vendor packed C++
structure measures 20 bytes, matching its explicit little-endian Python encoding:
`<int32 code, int32 size=8, int32 type=1, double data>`. It accepts only codes 320, 325
and 321, and sends the yaw field negated as the bridge does.

The first live command is deliberately narrower than the vendor example: only forward
`0.00..0.10 m/s`, no lateral or yaw component, at most one second, and every exit sends
zero-velocity triplets repeatedly for two seconds. Even a zero-only live check needs
`--operator-ready`; a non-zero command also needs the operator to confirm the documented
external-control mode, a clear lane and the emergency stop in hand.

On 2026-08-21, a zero-only stream at 10 Hz was passively captured at
`192.168.1.120:43893`: 12 packets from the development host, all 20-byte 320/325/321
zero triplets. The robot remained still. The current Venture App exposes **AI Motion
Mode**, while the public `Lite3_ROS` bridge documents **Auto Mode**. No public source
maps those labels, so this tool must not send a non-zero velocity until the vendor confirms
the applicable external-control transition. The current motion-host image also lacks the
ROS 2 and SDK service packages required for the official Venture SDK Mode path, so SDK Mode
cannot currently provide an alternative control transition.

#### AI Motion Control Mode gate

The high-level sender is prepared but must remain at dry-run or zero velocity until a Deep
Robotics engineer has legitimately enabled **AI Motion Control Mode** on this specific
Venture. Do not bypass, alter, or reverse-engineer that activation. Before one non-zero
programmatic command:

1. Record a fresh 10-second `RobotState` baseline after the vendor unlock. Compare only
   documented fields against the pre-unlock capture; a particular undocumented mode bit is
   not a requirement.
2. Have the operator prove a small forward walk and stop through the official app/controller.
   If it cannot walk there, stop: this repository is not the first fault to investigate.
3. While the robot is standing under its onboard controller, capture one 10 Hz zero triplet
   stream arriving at UDP `43893` and confirm fresh `RobotState`, valid battery and
   `error_state == 0`.
4. Obtain a separate, immediate operator authorization for one `vx=0.10`, `vy=0`,
   `wz=0`, 10 Hz, 1-second run. It must finish with at least two seconds of zero triplets.
   Record command and measured body velocity, error state, and the operator's visual result.

The AI Motion label is not assumed to be equivalent to legacy Auto Mode. On 2026-08-24, after
vendor-confirmed AI Motion unlock and an operator-confirmed official-App walk, a single
`vx=0.10`, 10 Hz, 1-second high-level pulse was captured arriving at `43893` as the documented
20-byte 320/325/321 sequence, including zero-velocity cleanup. The robot remained stable but
did not visibly move; six seconds of valid `RobotState` showed zero world-pose delta and
`error_state == 0`. No higher speed or repeat was attempted.

The vendor motion-host communication guide explains this A/B result: its complex floating-point
velocity commands (`0x0140`, `0x0145`, `0x0141`, equivalent to 320/325/321) must be sent in
**autonomous mode**, while a robot in AI state cannot switch into autonomous mode. In moving
mode or AI state, the guide instead specifies a separate simple axis-command interface: at
least 20 Hz, 250 ms command timeout, and zero axis value to stop.

### Verified manual-moving axis path

On 2026-08-24, the vendor reference control script and communication guide were used for one
bounded, separately authorized hardware proof:

1. Strict host captures confirmed one little-endian 12-byte manual-mode command
   `0x21010C02` followed by one moving-mode command `0x21010D06`; neither command contains an
   axis or velocity value.
2. A non-actuating health check from `192.168.1.103:20001` confirmed the documented
   `0x21040001` heartbeat and repeated `0x21010130 = 0` axis packets arriving at motion-host
   UDP port 43893. The robot remained healthy: 2,000 `RobotState` frames in 10 s at 200.1 Hz,
   `error_state = 0`.
3. One nominal-20-Hz `+32767` forward-axis pulse was sent for at most one second, with the
   heartbeat and zero-axis cleanup. The strict capture contained the initial heartbeat, 19
   `+32767` axis packets about 50--55 ms apart, then the first two zero-axis cleanup packets.
   Its fixed packet count intentionally did not capture the entire cleanup interval.
4. During the synchronized passive telemetry capture, firmware `goal_vel_forward` rose from
   0 to at most 0.5012, peak measured body-x velocity was 0.7289 m/s, and world-plane
   displacement at the recording boundary was 0.4011 m. `error_state` stayed 0 across all
   3,999 captured `RobotState` frames; the operator observed forward motion, a stop, and a
   stable robot. A following 10-second passive capture showed no forward command,
   `robot_motion_state = 0`, and `error_state = 0` in 2,000/2,000 frames.

`+32767` is the vendor reference script's full-scale axis value, not a documented metres-per-
second setting. The telemetry values above are one observed response, not a calibration for
the generic navigator. The synchronized capture ended before its handle-state telemetry
reported the zero cleanup, so it does not establish an exact firmware stop latency; it only
establishes that the sender began cleanup and that the robot was stopped in the follow-up
capture.

[`locomotion/lite3_control_mode_udp.py`](locomotion/lite3_control_mode_udp.py) restricts mode
selection to documented 12-byte commands.
[`locomotion/lite3_axis_udp.py`](locomotion/lite3_axis_udp.py) restricts the axis sender to
the documented forward full-scale or zero value, uses the vendor local port 20001, defaults to
dry-run, requires `--operator-ready` for live use, and sends zero axis values in `finally`. Its
periodic scheduler was corrected after this proof so send overhead does not accumulate into the
nominal 20 Hz period; that correction has offline tests but was not physically repeated just to
measure cadence. The same offline tests also prove that a failed zero packet does not abort the
cleanup interval: later zero packets are still attempted, the failure is surfaced, and the socket
is closed.

This proof does not permit a repeat, a higher value, a lateral/yaw command, or an autonomous
visual-navigation run. Motor temperatures remain unavailable; every future leg-moving test
requires new immediate operator authorization, a clear lane, a remote/e-stop in hand, and a
fresh health capture.

### Profile-gated simple-axis navigation transport

The vendor V1.0.8 Motion Host Communication Interface defines the moving/AI axis contract:

| axis | code | vendor positive direction | dead zone |
| --- | ---: | --- | ---: |
| forward/back | `0x21010130` | forward | `[-6553, +6553]` |
| lateral | `0x21010131` | right | `[-12553, +12553]` |
| yaw | `0x21010135` | right turn | `[-9553, +9553]` |

It also specifies a minimum 20 Hz axis cadence, a 250 ms axis timeout, and zero axis values as
the stop command. The supplied vendor reference GUI repeats held axes at 20 Hz, sends heartbeat
at 2 Hz, and sends a zero value when an input is released.

[`locomotion/lite3_axis_locomotion.py`](locomotion/lite3_axis_locomotion.py) is the
repository's offline-tested transport for that interface. It is deliberately not a generic
raw-axis CLI: a live run needs an explicit local axis profile. Every nonzero profile primitive
must carry an evidence reference, must sit outside the corresponding vendor dead zone, and must
cover every enabled navigation direction. No nonzero primitive is shipped by default; see
[`locomotion/lite3_axis_profile.example.json`](locomotion/lite3_axis_profile.example.json).

The shared navigator convention is positive-left lateral/yaw, while the vendor raw axes are
positive-right. The transport performs that inversion once at the boundary. It starts a
profile-gated independent 20 Hz axis stream only after fresh `RobotState` and a nonzero
navigation command, emits 4 Hz heartbeat, zeros all axes after a 150 ms command TTL, and streams
zeros on stop, failure, and shutdown. It neither changes control/moving mode nor falls back to
legacy 320/325/321 velocity commands.

#### ⚠️ The mapping is sign-only: commanded magnitude is discarded

A profile holds one evidenced raw value per direction, so `map_velocity` reads the *sign* of the
command and emits that primitive at full magnitude. It never scales. The consequence is that
`--derate` and `--max-vx` do not reach the wire on this transport — every setting below emits the
same raw axis value:

| `--derate` | commanded `vx` | forward axis emitted |
| ---: | ---: | ---: |
| 1.0 | 0.300 m/s | `+32767` |
| 0.6 | 0.180 m/s | `+32767` |
| 0.3 | 0.090 m/s | `+32767` |
| 0.2 | 0.060 m/s | `+32767` |

That is the deliberate half of the design: this transport will not invent a raw value it has no
physical evidence for, and interpolating between an evidenced `+32767` and an unevidenced zero is
inventing one. The envelope is enforced somewhere else instead. A profile may declare the speed
each primitive was measured to produce:

```json
"measured_m_s":   { "forward_positive": 0.729, "lateral_negative": 0.31 },
"measured_rad_s": { "yaw_positive": 0.55 }
```

`--live` preflight then **refuses the run** when a declared speed exceeds `--max-vx × --derate`
(or the `--max-vy` / `--max-wz` equivalent), so `--derate 0.2` against a primitive measured at
0.729 m/s stops at the gate rather than walking 3.6× faster than the safety veto planned for.
A primitive with no declared speed prints an unverified-envelope warning: it is not a claim that
the envelope holds, only a record that nobody checked. Both fields are optional and both land in
the run's telemetry alongside the profile's SHA-256.

**The linear deadband applies to the vector, not to each axis.** `input_deadband.linear_m_s`
gates `hypot(vx, vy)`, and the bearing is then snapped to the nearest of the eight `(forward,
lateral)` sign pairs the mapping can express. A per-axis deadband instead drops the smaller
component and rotates the command: at the shipped 0.05 m/s deadband, `(0.049, 0.051)` m/s is a
46° command that clears only the lateral gate, so forward zeroes and the robot leaves as a
full-scale 90° strafe — the same class of failure as the Go2 gait floor in #70. Snapping does
not make a diagonal accurate; the executed bearing of one is set by the two primitives' measured
speeds, not by the command. It does stop the direction depending on which of two independent
thresholds a component happened to fall under. Yaw keeps its own scalar deadband — the vendor
gives it its own dead zone, and it is not part of the linear vector.

The counter-evidence, stated plainly: gating the vector rather than each axis makes *more*
commands execute, not fewer. `(0.040, 0.040)` m/s — 0.057 m/s at 45° — used to fall under both
per-axis gates and emit nothing; it now emits both primitives at full scale. Because the mapping
is sign-only, `input_deadband.linear_m_s` is not a small-command filter: it is **the commanded
magnitude at which a full-speed primitive fires**. Set it from that, not from what looks like a
negligible velocity.

Nonzero axes also require `error_state=0`, documented force-control `basic_state=6`,
`policy_state=0`, a profile-allowed documented gait state, and `motion_state` 0 or 1. The
operator establishes manual/moving state; the MAPPO process refuses an unexpected state rather
than switching it.

This transport requires verified host-local `RobotState` before it can support the existing
MAPPO/planner implementation. Camera-only shadow detection is useful evidence but is not an
odometry replacement: the shared RGB path projects detections into world coordinates, latches
the goal and static map there, consumes measured body velocity in MAPPO input, and uses actual
displacement for its stall gate.

### Evidence-backed custom static profiles

`--static-profile PATH.json` selects one custom static colour profile and is mutually exclusive
with the shipped `--static-prop bin` profile. A custom profile must contain schema
`colour-profile/v1`, finite HSV/shape thresholds, known panel dimensions used for monocular
ranging, the whole obstacle's conservative radius, and non-empty evidence references. Its file
hash and evidence are written to telemetry; its local path is not. Geometry overrides are
refused, so a reviewed profile cannot be silently changed by command-line flags.

This is appropriate when a known panel is attached to an otherwise hard-to-segment obstacle. The
panel must be large, saturated, stable, and visibly distinct from all background objects in
actual Lite3 RTSP imagery. Its visual dimensions drive range, while its profile radius must
cover the complete physical obstacle plus documented measurement uncertainty.

## Read the robot before installing anything on it

Bringing up a **new** robot end to end — robot-side staging, commissioning, calibration,
scene layout, live run, evidence recovery — follows
[`DEPLOYMENT-SOP.md`](DEPLOYMENT-SOP.md), once per robot.

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

**Battery is available only while the stream matches the documented contract.** Public
`Lite3_ROS` identifies `code=2305` as `RobotState`, including `battery_level`. Before the
2026-08-21 vendor-service restart, this Venture emitted that code in an incompatible
212-byte packet, which the decoder correctly rejected. The restarted service emits the
documented 220-byte layout and reports 21% battery. The decoder must continue to reject
any future layout mismatch rather than guessing offsets; charge the robot before a live
run because 21% is only one point above the 20% abort threshold.

**Motor temperatures are genuinely absent.** The high-level interface does not carry them
in any form. The low-level `Lite3_MotionSDK` reports them, but taking low-level control
merely to read a temperature would remove the vendor controller that keeps the robot
upright, so this port does not do that. The supported publisher is a vendor question and
is still open.

Until it is answered there are two ways to run:

| | motor temperatures | what runs |
| --- | --- | --- |
| default | required | dry navigation only; `--live` refuses |
| `--accept-no-motor-temperatures` | unmonitored | needs a verified fresh battery; a charged run is bounded and recorded |

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

   Static fits write a **provisional** Lite3 calibration. They may be used for shadow/map
   evidence, but `--live` refuses them. Independently validate focal length, optical-centre
   height, and mount pitch against a second known-distance observation before producing a
   reviewed `calibration_status=validated` artifact.

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

### Record live perception without motion

[`visual_nav/lite3_vision_shadow.py`](visual_nav/lite3_vision_shadow.py) is the first
on-robot deployment rung. It opens only the supplied camera source and runs the existing
MobileNet-SSD detector; it imports no locomotion, UDP, ROS, or vendor-control module and has
no `--live` option. Its JSONL is credential-safe: it records only the camera source kind, frame
metadata, pixel-space boxes, and inference timing.

On the staged motion host, `/dev/video0` is already owned by the vendor GStreamer publisher.
Consume its existing local RTSP output rather than competing for the V4L2 device:

```bash
release=$HOME/mappo-lite3-stage/releases/mappo-arm-cloud-physical-ai-lite3-20260825
export PYTHONPATH=$HOME/mappo-lite3-stage/python
python3 "$release/robot-stack/deep_robotics/lite3/visual_nav/lite3_vision_shadow.py" \
    --camera-source rtsp://127.0.0.1:8554/test \
    --model-dir "$HOME/mappo-lite3-stage/models/mobilenet-ssd" \
    --classes person,chair --seconds 60 \
    --output "$HOME/mappo-lite3-stage/evidence/vision-shadow.jsonl"
```

This confirms camera/model/target perception only. It does not provide calibrated range,
odometry, planner output, obstacle avoidance, or locomotion authorization.

On 2026-08-25 this command ran on the event motion host against the existing local RTSP
publisher: 10 1280x720 frames, zero camera read errors, and 77.5 ms mean MobileNet-SSD
inference. It detected no `person` or `chair` in that short sample; that is a scene observation,
not evidence that the camera or model failed. The deployed AArch64 source passed 284
non-actuating tests: policy 33, integration 144, Lite3 locomotion 45, visual navigation 44, and
commissioning 18.

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
