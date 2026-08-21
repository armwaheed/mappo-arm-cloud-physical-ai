<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Device Connect dashboard — bring-up run, 2026-08-21

What this run proves, and — more usefully — what it does not.

## What ran

Three `robot_driver.py` processes and one `server.py`, on one workstation, over a **real
Device Connect D2D mesh** (Zenoh peer mode, multicast scouting, no broker and no registry).
Nothing is mocked at the SDK boundary: the dashboard discovered the drivers by presence
announcement, called them with `invoke_device`, and read their events off the mesh.

| device | platform | motion |
| --- | --- | --- |
| `mappo-sim` | bench double | enabled |
| `mappo-go2` | Go2 | enabled |
| `mappo-lite3` | Lite3 Venture | **disabled** |

The Go2 and Lite3 drivers answer everything that does not need the robot — capabilities,
gait-floor refusals, the motion gate, checkpoints — and fail at the SDK import for anything
that does, which is the correct behaviour off-robot and is visible in the log.

## Files

| | |
| --- | --- |
| `end-to-end-run.txt` | All four capabilities exercised through the dashboard's HTTP API, including every refusal. |
| `platform-asymmetry.txt` | The same page against three platforms; `get_capabilities()` is what differs. |
| `event-stream-before-ordering-fix.txt` | The capture that found the event-ordering defect. Kept because it is the evidence. |
| `driver-startup.log` | A driver joining the mesh and emitting its first `robot_state`. |

## The numbers

75 tests, all passing, plus `ruff check .` clean in `dashboard/`:

| suite | tests |
| --- | --- |
| `test_model_store.py` | 16 |
| `test_cloud_models.py` | 13 |
| `test_drive_bridge.py` | 17 |
| `test_robot_driver.py` | 15 |
| `test_server.py` | 14 |

Five guards were mutation-tested — the test was watched to fail with the guard removed,
rather than assumed to:

| guard removed | caught by |
| --- | --- |
| the `finally` that stops the robot in `run_nudge` | `test_the_robot_is_stopped_even_when_the_nudge_raises` — "the robot was not stopped: stops=0" |
| `get_capabilities` renamed back to `capabilities` | `test_no_rpc_shadows_a_base_class_member` — "['capabilities'] shadow members of DeviceDriver" |
| the `--allow-motion` gate | `test_motion_is_refused_and_the_worker_is_never_started` |
| the armed-checkpoint delete guard | `test_installing_over_the_armed_checkpoint_is_refused` |
| the driver-side reverse cap | `test_the_reverse_cap_is_applied_by_the_driver_too_not_only_by_the_worker` |

## Two defects this run found

**A device that announced nothing and could not be found.** `DeviceDriver.capabilities` is
a property on the SDK's base class, read by the runtime to build a device's presence
announcement. An `@rpc()` named `capabilities` overrides it; the announcement is then never
published and the device never appears on the mesh. There is **no exception and no log
line** — the driver logs "D2D presence announcer started", runs perfectly, and is invisible.
It presents as a network problem. Found by subscribing to the raw presence subject and
seeing one device announce and the other not. `test_no_rpc_shadows_a_base_class_member` now
fails on any RPC that collides with a base-class member, of which there are about thirty.

**Completions arriving before their own starts.** A `Subscription` keeps one inbox per
subject and each event *name* is its own subject, so a drained batch comes out grouped by
event type rather than by time. The operator's log showed three `motion_completed` lines
above the three `motion_started` lines that caused them. `event-stream-before-ordering-fix.txt`
is that capture. Batches are now sorted by the device's own `ts` before fan-out.

That timestamp has **one-second resolution**, so this fixes ordering across seconds and
cannot fix it within one; the sort is stable, so events sharing a timestamp keep arrival
order. After the fix, twelve motion events over two runs of three commands each were in
strict start-then-complete order, with zero inversions.

## What this run does NOT prove

**No robot moved.** The bench double delivers exactly what it is commanded — its
`delivered_fraction` is 1.0 in every row above, which is precisely the number a real robot
does not produce. This Go2 delivers about 0.45 of what it is commanded and the Lite3 about
0.74 forward and 0.27 laterally. Nothing here tests gait, balance, the DDS path, the ROS
bridge, the lease, or the SDK import in the robot's own Python 3.8 environment.

So this run establishes that the mesh, the RPC schemas, the event stream, the refusals, the
checkpoint store, the cloud fetch and the page all work end to end. **The hardware run is
still outstanding**, and the first one should be `--platform go2` with motion disabled —
`get_status` and `list_models` only — before `--allow-motion` is passed on a real robot.
