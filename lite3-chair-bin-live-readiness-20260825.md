# Lite3 Chair-and-Box MAPPO Demo: Current Status and Live Readiness

**Status date:** 2026-08-25
**Purpose:** Record what is deployed, what has been measured, what is still gated, and what must
be true before a `--live` MAPPO chair-and-box demonstration can start.

## Executive status

The Lite3 motion host can now run the repository's non-actuating MAPPO stack locally:

```text
vendor UVC camera
-> vendor local RTSP publisher
-> OpenCV / MobileNet-SSD
-> host-local RobotState
-> shared visual-navigation / MAPPO code
```

The host-local telemetry loopback is active and independently verified. The existing MAPPO
architecture can therefore obtain the pose and measured body velocity it requires without a
MacBook during the demo.

The system is **not yet approved to execute `--live`**. The remaining gates are real safety and
evidence requirements, not missing package installation:

1. the latest source is not yet staged as a host v4 release;
2. the available green-panel camera calibration is explicitly **provisional**;
3. the current vendor telemetry reports `robot_basic_state=98`, not the documented
   force-control state `6` required by the new nonzero-axis gate;
4. only a bounded full-forward raw-axis value has physical evidence; yaw primitives needed for
   chair navigation have not been commissioned;
5. loaded robot radius, gait floor, actuator gain, and thermal operating procedure are not yet
   evidence-backed.

No new physical movement was performed while preparing this status report.

## Host deployment

### Runtime and camera

The motion host has an isolated, user-owned Python runtime with:

| component | status |
| --- | --- |
| Python | system Python 3.8.10 |
| NumPy | 1.24.4 |
| OpenCV | `opencv-python-headless` 4.10.0.84 |
| detector | MobileNet-SSD Caffe model, MIT-licensed upstream |
| camera input | `rtsp://127.0.0.1:8554/test` |
| actual RTSP frame size | 1280x720 BGR |

The deployment consumes the existing vendor RTSP publisher. It does not take over `/dev/video0`
or change the vendor GStreamer process.

### Host-local RobotState loopback

On 2026-08-25, a separately authorized maintenance operation:

1. captured a healthy Mac-side baseline;
2. backed up the original vendor telemetry configuration;
3. changed only the telemetry destination to host loopback;
4. restarted `jy_exe.service` once;
5. verified state reception and retained loopback after success.

The active configuration is now:

```toml
ip = '127.0.0.1'
target_port = 43897
local_port = 43893
```

Current configuration SHA-256:

```text
055cd0a94b895603f81f8bacc24ee315ee47a2353940adcd4da28264aaa96590
```

The backed-up Mac-directed configuration remains on the host with SHA-256:

```text
427fdc72b4baa3f03cfadcd68fc232295e95098301c696a3bc94c5a870d0ee05
```

Maintenance validation result:

| measure | result |
| --- | ---: |
| `RobotState` frames | 2,691 |
| state rate | 200.0 Hz |
| minimum battery | 88.0% |
| observed `error_state` | 0 |

A separate 10-second passive post-check observed 2,000 valid `RobotState` frames at 200.0 Hz,
with `error_state=0` throughout. The profile-gated axis transport then connected only to read
pose, velocity, and battery, and shut down without calling `set_velocity()` or creating an axis
stream.

## Vendor motion protocol

The supplied V1.0.8 interface and customer reference GUI were reviewed and hashed:

| source | SHA-256 |
| --- | --- |
| `绝影Lite3运动主机通讯接口_V1.0.8.md` | `627e3132a2b87788396e7113d84613f24c2d3010b2d297672fddf434b487181b` |
| `Reference_code_Lite3_New_All_Control_v2_0_158.py` | `ec95ecf82fd1aa6e651fde5890ef25f3962aa9ca7e5314dc62e1c09dc788d973` |

The supported moving-mode simple-axis contract is:

| component | code | vendor positive direction | dead zone |
| --- | ---: | --- | ---: |
| forward/back | `0x21010130` | forward | `[-6553, +6553]` |
| lateral | `0x21010131` | right | `[-12553, +12553]` |
| yaw | `0x21010135` | right turn | `[-9553, +9553]` |

Other documented constraints:

```text
axis update rate: >=20 Hz
axis watchdog:    250 ms
heartbeat:        >=2 Hz
zero axis:        stop
moving mode:      0x21010D06
manual mode:      0x21010C02
```

The shared navigation convention is positive-left lateral/yaw; vendor axes use positive-right.
The new transport inverts lateral/yaw exactly once at the vendor boundary.

The historical physical proof establishes only one bounded primitive:

```text
forward raw axis = +32767
duration         <=1 s
world displacement = 0.401080 m
peak body-x velocity = 0.728896 m/s
error state = 0
```

It does **not** establish reverse, lateral, yaw, reduced amplitude, or continuous MAPPO-safe
profiles.

## Scene evidence

### Supplied measurements

| object | measurement |
| --- | --- |
| black chair | height 1.00 m; width 0.71 m |
| brown MSI cardboard box | length 0.32 m; width 0.10 m; height 0.54 m |
| box conservative half-diagonal | about 0.168 m |
| green panel | width 0.10 m; height 0.05 m |
| camera optical-centre height | approximately 0.40 m |
| green-panel centre height | approximately 0.60 m |
| camera optical-centre to green-panel plane | 0.68 m |

The supplied phone photographs and 22.055-second phone video document placement but are not
Lite3 camera calibration frames.

### Actual Lite3 RTSP results

The chair and box were captured through the real Lite3 RTSP stream. Chair detection was evaluated
on three 1280x720 Lite3 frames:

| goal input | centre crop | chair confidence scores |
| --- | ---: | --- |
| 224 | 0.5 | unstable / absent |
| 224 | 0.7 | 0.9744, 0.9787, 0.9749 |
| 300 | 0.5 | 0.9528, 0.9696, 0.9734 |
| 300 | 0.7 | 0.7569, 0.9120, 0.9605 |
| 300 | 1.0 | 0.7763, 0.8404, 0.8296 |

The current shadow candidate is therefore:

```text
goal class: chair
goal input size: 300
goal crop: 0.5
goal confidence: 0.50
goal height: 1.00 m
goal width: 0.71 m
```

The brown cardboard box cannot use the existing blue `--static-prop bin` profile. Its colour
overlaps the wooden background, producing one merged contour rather than an isolated obstacle.

A 10x5 cm high-saturation green panel attached to the box was then captured by the real Lite3
camera. The panel was stable across three frames:

```text
box: (622,326) to (692,360)
shape score: 0.842 to 0.844
```

The candidate green profile:

```text
H=75..90, S>=200, V>=70
min area=400 px, fill>=0.55, aspect=1.3..2.6
visual panel dimensions=0.10 x 0.05 m
planner box radius=0.20 m (conservative)
```

It found exactly the green panel in all three Lite3 frames while excluding the green exit sign.

### Provisional camera calibration

Using the known green-panel dimensions and 0.68 m distance, the static fit produced:

| metric | value |
| --- | ---: |
| horizontal focal estimate | 476.76 px |
| vertical focal estimate | 462.50 px |
| provisional focal | 469.63 px |
| horizontal/vertical disagreement | 3.03% |
| fitted panel range | 0.6905 m |
| measured panel distance | 0.6800 m |

The provisional calibration has:

```text
focal_px = 469.6297
pitch_rad = -0.2625193  (-15.04 degrees; positive is nose-down)
height_m = 0.40
calibration_status = provisional
```

It is intentionally refused by `--live`. A second independently measured static observation is
required before a reviewed `calibration_status=validated` artifact can be created.

## Code and test state

Implemented but not yet used for motion:

- profile-gated simple-axis transport;
- 20 Hz axis streamer and 4 Hz heartbeat;
- 150 ms application command TTL;
- zero-all-axes on stale command, error, shutdown, and cleanup failure;
- no fallback to legacy `320/325/321` packets;
- no automatic mode/gait/AI transition;
- documented `RobotState` gate before nonzero axes:
  `error=0`, `basic=6`, `policy=0`, allowed gait, and motion state 0/1;
- custom evidence-backed static colour profile loader;
- provisional calibration rejection for `--live`;
- telemetry profile SHA-256/evidence provenance.

The latest full local regression run passed:

| suite | count |
| --- | ---: |
| policy | 33 |
| integration | 144 |
| Go2 visual navigation | 267 |
| Lite3 locomotion | 56 |
| Lite3 visual navigation | 51 |
| Lite3 commissioning | 25 |
| **total** | **576** |

Every required directory was also Ruff-clean.

## What still blocks `--live`

1. **Host v4 staging:** the host v3 release does not contain the newest scene/profile/mode gates.
2. **Validated calibration:** the green-panel calibration is provisional because mount-height and
   pitch measurements are approximate and only one distance was used.
3. **Vendor mode state:** latest host telemetry reports `basic_state=98`; the axis transport
   refuses nonzero motion until the operator establishes and proves the documented
   force-control/moving state (`basic_state=6`, profile-allowed gait, `policy_state=0`,
   `error_state=0`).
4. **Axis primitive evidence:** full-forward is the only historical primitive. The first
   chair-and-box MAPPO run requires evidence-backed yaw positive/negative primitives. Lateral
   remains disabled unless separately commissioned.
5. **Robot geometry/dynamics:** loaded robot radius, gait floor, and actuator gain must be
   measured on this Lite3 rather than copied from Go2.
6. **Thermal operations:** no motor-temperature stream is available. A bounded run limit,
   cooldown policy, and remote/e-stop operator must be explicitly set.
7. **Scene safety:** chair, box, lane clearance, cable slack, and all unmodelled objects must be
   confirmed immediately before movement.

## Current release artifacts

The current source patch prepared before this report:

```text
base commit: 87b0f7b905be9c51c488eee99a2db184e011ef08
patch:       lite3-mappo-chair-bin-shadow-20260825.diff
SHA-256:     167548168e440e25c63d4c916ccdd5816bdd228361806080a5e7c5a593f374ba
```

This report itself must be included in the final v4 patch, so the patch checksum above will
change when the final export is regenerated.

## Next safe sequence

1. Stage v4, including profile and calibration shadow support.
2. Run host-local, non-live chair-and-box shadow with the provisional calibration and green
   profile; inspect pose, chair goal acquisition, panel static-map stability, policy proposal,
   and planner veto.
3. Capture a second no-motion green-panel observation at a different measured distance; validate
   focal/pitch consistency and generate a reviewed live calibration artifact.
4. Commission yaw-right and yaw-left primitives under separate immediate walking authorization.
5. Measure loaded radius, gait floor, and actuator gain.
6. Obtain a fresh `basic_state=6` manual/moving capture, final thermal/run-limit agreement, and
   lane/e-stop confirmation.
7. Only then request and execute one bounded supervised `--live` MAPPO chair-and-box run.
