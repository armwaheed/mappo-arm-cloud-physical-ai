<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dashboard local trial, and the first hardware contact — 2026-08-26

Two runs, deliberately reported apart, because they prove different things and one of them
is much weaker than it looks.

1. **A local trial** on one MacBook: a checkpoint server, two drivers and the dashboard,
   over a real D2D mesh, with no robot involved.
2. **A read-only hardware step** against the Go2 at `192.168.123.18`. **No robot moved, and
   motion was never enabled.**

The 2026-08-21 bring-up run closed with *"the hardware run is still outstanding, and the
first one should be `--platform go2` with motion disabled — `get_status` and `list_models`
only."* That is exactly what run 2 is.

---

## Run 1 — the local trial

| | |
| --- | --- |
| `model_server.py` | serving `policy/models` on `127.0.0.1:8800` |
| `robot_driver.py --platform sim` | `mappo-sim`, motion enabled, bench double |
| `robot_driver.py --platform go2 --simulate` | `mappo-go2-sim`, motion enabled, Go2 **rules** |
| `server.py --port 8080` | the dashboard |

Python 3.11.13, `device-connect-edge` / `device-connect-agent-tools` **0.2.5 from PyPI**,
`eclipse-zenoh` 1.10.0. D2D, multicast scouting, no broker and no registry.

### What was exercised, and the result

| path | result |
| --- | --- |
| D2D discovery | both drivers found by presence announcement; **18 functions** enumerated |
| Fleet, grouped by platform | `STOP ALL (2)`, `Stop go2 (1)`, `Stop sim (1)`, per-row stops |
| `STOP ALL` | `stopped: [mappo-sim, mappo-go2-sim]`, `failed: []`, `matched: 2` |
| Cloud AI browse | `list_cloud_models(index_url=…)` → 1 object, `kind: server`, `served_by: mappo-model-server` |
| Cloud AI download | 268 063 bytes, `sha256 7327f724…ca11` — **byte-identical to the server's index** |
| Checkpoint swap | `select_model` → `active` changed, `previous: baseline_zeros.npz`, "next run" |
| Unload | `delete_model` freed 289 432 bytes |
| Delete the armed one | **refused**, naming the config that would dangle |
| Camera | MJPEG `multipart/x-mixed-replace`, **36 frames in 6 s** (6.0 fps, the default), 480×360, labelled `SYNTHETIC` in the pixels |
| SSE | 52 events / 8 types over the run |
| SSE replay | a page connecting **afterwards** got **79 events spanning 115 s in its first 4 s** — a 4 s connection cannot have watched 115 s happen, so the ring buffer really is replayed |

### The refusals, all reached through the page's own API

| asked for | answer |
| --- | --- |
| `file:///etc/passwd` | `uses scheme 'file'; only s3://, https:// and http:// are supported` |
| `…/models/../../etc/passwd` | `'passwd' must end in .npz` |
| an index URL on a dead port | `could not be reached: [Errno 61] Connection refused` |
| `os.system` as a function | **HTTP 403**, never reached the mesh |
| `strafe_left 0.15` on the **simulated go2** | refused, quoting *its own measured* 0.200 lateral floor |
| `strafe_left 0.15` on the **bench double** | accepted — the double's floors are zero, which is the documented difference |
| `turn_left` on the simulated go2 | allowed, and `motion_completed` carried `warning=no yaw gait floor has been measured` |

`dashboard-bench-double.png` is the page mid-run: camera live, drawer open, the log in
start-then-complete order.

### What run 1 does NOT prove

`delivered_fraction` is **1.0** in every bench-double row. That is precisely the number a
real robot does not produce, and it is the tell that nothing here touched a gait.

---

## Run 2 — the Go2 at 192.168.123.18, read-only

**Reachability first, because the brief said it was unreachable.** It was, and then it was
not: the ethernet cable was reconnected between the brief being written and this run. At the
time of the run the MacBook held `192.168.123.50` on `en10` alongside the corporate
`10.118.105.133` on `en0`, `route get` resolved through `en10`, and ping ran 1.3 ms. The
original diagnosis — *"a network-segment problem, not a robot-side one"* — was correct for
the state it was made in.

### ⛔ The robot cannot run this dashboard's device, and that is the finding

```
Go2 Jetson  ·  Linux 5.10.104-tegra aarch64  ·  Python 3.8.10 and 3.9.5
                                                 NO Python >= 3.11 anywhere on the machine
```

`device-connect-edge` requires Python ≥ 3.11. **So `robot_driver.py` cannot run on this
robot at all**, and the runbook's `python3 robot_driver.py --platform go2` on the robot is
not executable here today. This is the same wall `drive_bridge.py` was built for, met from
the other side: the 3.8 half runs, the Device Connect half cannot.

What *does* run on the robot is the bridge — and this is its **first execution on hardware**.
Byte-identical to this branch's copy (`md5 250c1bfc…`), under the robot's own SDK
environment, launched with the robot's own documented `setup_env.sh` and `PYTHONPATH`:

```
$ python3 drive_bridge.py status --platform go2 --iface eth0
[Go2Locomotion] odom live  mode='mcf'  pose=(+2.80, +2.05, +175.8°)
{"ok": true, "platform": "go2", "pose": {"x": 2.797, "y": 2.048, "yaw": 3.068},
 "velocity": [2.7e-08, -1.0e-09, -0.0011], "mode": "mcf", "error": null}
```

| measurement | value |
| --- | --- |
| `status` wall time, 3 consecutive runs | **1.98 s, 1.93 s, 1.95 s** |
| what that time is | the cold SDK import and DDS discovery, paid **per invocation** |
| against `STATE_INTERVAL_S = 5.0` | **~39 %** of every poll period, with a DDS client on the bus |
| pose repeatability across the 3 runs | ±6 × 10⁻⁵ m |
| CPU / GPU thermal | 49.5 °C / 45.9 °C |

`robot_driver.py` carried the comment *"a status read is milliseconds"*. It is not; this PR
replaces the claim with the measurement.

### 🔴 `mode='mcf'`, not a sport mode

The robot answered `mode='mcf'` and `Go2Locomotion` warned that `Move` commands may be
ignored. **Nothing was commanded**, so this is a reading and not a fault — but it means a
motion run from this dashboard would have to call `ensure_sport_mode()` first, and an
operator who skipped that would see a robot that accepts every command and never steps. That
is the exact failure shape this repository's gait-floor work exists to make visible.

### The dashboard, against the real robot

With no `--allow-motion`, the driver was run **on the workstation** with `--bridge-python`
pointed at a wrapper that executes the robot's own byte-identical bridge over SSH.

⚠️ **This is a workaround and not a supported deployment.** It is written down because it is
the only shape that can work while the robot has no Python ≥ 3.11, and because it is exactly
the kind of wrapper that can make a documented command look like it ran when it did not. Its
limits are real and are listed below.

| RPC | result |
| --- | --- |
| `get_status` | real odometry: `x 2.7970 y 2.0479 yaw 3.0672`, `mode mcf`, `motion_enabled false` |
| `get_capabilities` | `platform go2`, **`simulated: false`**, floors 0.35 / 0.20 / `null` |
| `list_models` | answered — but see the limit below |
| `walk_forward` | **refused**: *"this device was started without --allow-motion"* |
| `watch_camera` | reply optimistic (`watching: true`); the truth arrived one poll later in `capabilities.camera.error`: `could not start the Go2 camera: No module named 'unitree_sdk2py'` |

**What this shape cannot do, stated plainly.** The driver process is on the laptop, so
`list_models`, `free_bytes` and `download_model` all act on the **laptop's** package
directory, not the robot's — the checkpoint panel is answering about the wrong machine. The
camera is the same story and fails honestly rather than silently, because `camera_source.py`
runs in the driver and the laptop cannot import the Go2 SDK. Only `get_status`,
`get_capabilities` and the motion gate genuinely reach the robot.

`dashboard-live-go2.png` is the fleet with `mappo-go2-live` in it, badged `MOTION DISABLED`,
real pose on the row, motion pad greyed, and two `motion_refused` lines in the drawer.

### Nothing moved

First and last readings of the whole hardware session, about 40 minutes apart:

| | x | y | yaw |
| --- | --- | --- | --- |
| first reading | 2.7970569 | 2.0479622 | 3.0679421 |
| last reading | 2.7970042 | 2.0479519 | 3.0697386 |
| delta | −5.3 × 10⁻⁵ m | −1.0 × 10⁻⁵ m | +1.8 × 10⁻³ rad (0.103°) |

**Planar displacement 5.4 × 10⁻⁵ m** — 54 micrometres, over forty minutes, which is estimator
drift on a stationary robot and not locomotion. The yaw term is the larger one and is the
same drift; nothing commanded it.

`enable_lease` defaults to `False`, so no control authority was taken, and `command_status`
calls only `pose()`, `velocity()` and `current_mode()`. Nothing was written to the robot's
filesystem: the bridge that ran is the robot's **own** copy, verified byte-identical to this
branch's rather than uploaded. Afterwards its eight recorded run files were still present and
`/tmp` held nothing from this session.

---

## 🔴 The defect the hardware step found: the event stream trusts a robot's clock

```
MacBook:  2026-08-26T20:44:04Z
Go2:      1970-01-16T19:20:58Z        RTC time: 1970-01-01   System clock synchronized: no
```

The Go2 has no working RTC and NTP had not synchronised — its clock is **56 years** behind.

`server._in_time_order` sorts a drained batch by **`ts`, stamped by the emitting device**,
with a string compare. So once `robot_driver.py` runs *on* a robot like this one — which is
the documented deployment — every event it emits sorts before every event from a
correctly-clocked robot, permanently, regardless of when it happened. The operator's drawer
would show one robot's history pinned away from the timeline it belongs in.

**The 2026-08-21 run could not have caught this, and neither could its test.** All three
drivers ran on one workstation and therefore shared one clock, and
`test_a_batch_is_reordered_by_the_devices_timestamp` uses a single date throughout. An
evidence set whose devices all share the condition under test cannot fail that test.

This is already solved once in this repository, in the other direction:
`peer_link`/`integration/peer_source.py` refuse to put a peer's timestamp on the wire and
send a **duration** instead, for exactly this reason. The event stream still trusts one.

`test_two_robots_with_unsynchronised_clocks_cannot_be_interleaved` pins the current
behaviour with the measured 1970/2026 pair so a change to the key is deliberate. **It is a
characterisation test, not an endorsement** — no fix is proposed here, because choosing
between arrival order, a receive-side stamp and a duration is a design decision that wants
its own issue.

---

## Files

| | |
| --- | --- |
| `local-trial-run.txt` | Run 1, every call and every refusal through the dashboard's HTTP API. |
| `event-stream.txt` | The SSE capture, one line per event, including the late-page replay. |
| `model-server-startup.log` | The checkpoint server, its digests, and its loopback warning. |
| `robot-bridge-status.txt` | Run 2: three timed `status` calls on the robot, plus thermals. |
| `robot-over-the-mesh.txt` | Run 2: the read-only RPCs, and the motion refusal. |
| `dashboard-bench-double.png` | Run 1 mid-run: camera live, drawer open. |
| `dashboard-live-go2.png` | Run 2: the real robot in the fleet, motion disabled. |

## Test counts

143 tests across `dashboard/`, all passing, `ruff check .` clean.

| suite | tests | |
| --- | --- | --- |
| `test_camera_source.py` | 11 | |
| `test_cloud_models.py` | 13 | |
| `test_drive_bridge.py` | 28 | |
| `test_model_server.py` | **14** | new |
| `test_model_store.py` | 16 | |
| `test_peer_link.py` | 14 | |
| `test_robot_driver.py` | **35** | +2 |
| `test_server.py` | **27** | +1 |

⚠️ **`test_robot_driver.py`'s 35 tests are not in `.github/test-inventory.tsv`**, which
counts `dashboard` as 125. `measure-suites.sh` skips that file because it imports
`device_connect_edge`, on the stated grounds that the package "is not on PyPI before
launch". **That is no longer true** — `device-connect-edge` and
`device-connect-agent-tools` are both on PyPI at 0.2.5 and were installed with plain
`pip install`. The skip list lives in `.github/`, which this PR does not own; the inventory
is regenerated, so it still reads 125 and CI still agrees with it.

## Mutations run

Seven guards were removed one at a time and the suite watched to fail, with the patch
verified to have applied first — a mutation that does not apply is a vacuous "survived", and
one of these was, on the first attempt, because of a quote mismatch.

| guard removed | caught by |
| --- | --- |
| `resolve()`'s containment check | `test_a_symlink_pointing_out_of_the_directory_is_not_served` |
| `resolve()`'s name gate | `test_a_badly_named_file_is_not_fetchable_even_though_it_is_right_there` |
| `entries()`'s name gate | `test_a_name_the_store_would_refuse_is_never_advertised` |
| the digest memo's size/mtime key | `test_the_digest_follows_the_bytes_and_not_the_filename` |
| relative URLs, replaced by a hard-coded bind address | `test_addresses_are_relative_so_the_server_need_not_know_its_own_name` |
| `list_http_index`'s `urljoin` against the fetched URL | same test |
| `download_model` made to arm what it downloads | `test_a_checkpoint_is_browsed_downloaded_and_armed_from_a_real_model_server` |

The second row is why that test exists: the first mutation run **survived**, because the
containment check alone already refused every traversal the tests spelled. The name gate was
only load-bearing for a badly-named file that really is in the directory, and nothing
covered it.
