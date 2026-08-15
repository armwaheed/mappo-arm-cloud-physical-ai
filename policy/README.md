<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The MAPPO policy package

The delivered 3-agent MAPPO navigation actor and the adapter that runs it once per robot
control step. Authored by Sagar Surendran ([@spsagar13](https://github.com/spsagar13));
**[`PROVENANCE.md`](PROVENANCE.md) lists every change made on the way into this
repository**, and why each one was needed.

```text
physical_ai_mappo.py               inference + robot-state adapter
config.json                        deployment configuration
models/mappo_actor_3agent_1910000.npz   the checkpoint, 262 KiB, committed
basic_test.py                      install smoke check — one inference, no robot
test_physical_ai_mappo.py          the behaviour suite (30 tests)
requirements.txt                   numpy, and nothing else
```

## Install and check

`../deploy/install.sh` does this as part of bringing the whole demo up on the robot. By
hand, on a workstation:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 basic_test.py              # "PASS", plus the calibration it is running with
python3 test_physical_ai_mappo.py  # 31/31
```

`basic_test.py` is an **install** check: numpy is here, the npz loads, one forward pass
produces a finite action. It would pass with the weights replaced by noise. What pins the
behaviour is the suite beside it, and what pins the mapping from real telemetry is
`integration/`.

## Call once per robot control step

```python
from physical_ai_mappo import COMMAND, MappoController, RobotInput, StationaryObject

controller = MappoController("config.json")

out = controller.step(RobotInput(
    x_m=robot_x, y_m=robot_y, yaw_rad=robot_yaw,      # odom pose
    vx_mps=robot_vx, vy_mps=robot_vy,                 # BODY frame — see below
    goal_x_m=goal_x, goal_y_m=goal_y,                 # odom
    external_hold=a_MOVER_is_blocking_the_lane,
    stationary_objects=[StationaryObject(distance_m=d, bearing_rad=b,
                                         radius_m=r, object_id=obstacle_id)],
    timestamp_s=time.monotonic(),                     # or None to let it stamp its own
    reset_run=first_step_of_this_run,
))

if out.status == COMMAND:
    robot.Move(out.vx_mps, out.vy_mps, out.vyaw_radps)
else:
    robot.StopMove()
```

Do not hand-roll this against live telemetry: `integration/mappo_bridge.py` is the mapping
from one telemetry tick to these arguments, and three of the mappings are not the obvious
ones. `integration/mappo_policy.py` wraps the whole loop.

### The four arguments that are easy to get wrong

| argument | the obvious answer | why it is wrong here |
| --- | --- | --- |
| the velocity frame | odom, like the pose | `measured` is the Go2 estimator's **body**-frame velocity. Omit `velocity_frame` and the config's `"body"` applies. |
| `external_hold` | the planner's `reason == "hold"` | the planner also holds for the **bin**, and forwarding that zeroes the policy in the one scene it exists for. It means *a MOVER is blocking the lane*. |
| `stationary_objects` | anything not moving | anything the stack reports as `kind: "static"`. A person who has **stopped** has a bin's velocity and a person's claim on the lane; the split is a field, not a speed threshold. |
| `timestamp_s` | the wall clock | it is compared against `time.monotonic()`. An epoch value is now refused outright (`STOP_CLOCK_ERROR`) rather than silently disabling the staleness gate. |

### Statuses

| | |
| --- | --- |
| `COMMAND` | the policy is driving |
| `STOP_CLOCK_ERROR` | `timestamp_s` is not a `time.monotonic()` reading |
| `STOP_STALE_INPUT` | the input is older than `stale_input_timeout_s` |
| `STOP_EXTERNAL_HOLD` | the existing moving-object/safety logic is in charge |
| `STOP_GOAL_REACHED` | within `goal_stop_distance_m` |

Every non-`COMMAND` status zeroes the command but still reports `action_x`/`action_y`, so
a shadow run can record what the policy wanted while something else drove.

## How it works, in one paragraph

The trained agent is a **holonomic, non-rotating** VMAS agent. `reset_run=True` fixes a
**run-local frame** at the robot's current pose and heading; everything the network sees
is in that frame, divided by `meters_per_vmas_unit`. Because the trained agent never
rotates — `state.rot` stays zero for the whole navigation task — its 12 rays sit at
**fixed angles in that frame** and do not turn with the robot's nose. `lidar` is
`range_max - range`, i.e. **proximity**: bigger means closer, and clear space reads zero.
Yaw is used only to turn the resulting action back into a body-frame command.

None of that was read off the code. It was checked against the weights: a goal-bearing
sweep tracks the goal to within 14.5°, and a ring of obstacles produces strong evasion
where clear space does not.

## The two numbers that decide the demo

```text
sensing horizon = lidar_range_vmas x meters_per_vmas_unit = 0.35 x 2.5 = 0.875 m
trained agent radius at that scale = 0.10 x 2.5 = 0.25 m
```

`lidar_range_vmas` belongs to the **policy** and is checked against the checkpoint at
load — change it and the config is refused. `meters_per_vmas_unit` belongs to the **robot
and the room** and is meant to be swept:

```bash
cd ../integration && python3 replay_mappo.py ../evidence/sample_telemetry.jsonl \
    --scale 1.0 1.5 2.0 2.5 3.0 4.0
```

2.5 is set so the trained agent's radius equals the radius the planner is actually run
with. **That coupling is live**: the vendored planner's default `--robot-radius` is 0.40 m
and the recorded runs passed 0.25. Run it at the default and this scale is stale — see
`deploy/README.md`, which pins the flag.

## ⚠️ What this checkpoint will not do

Measured, reproducible, and worth knowing before it drives anything:

- **The steering response is a cliff, not a ramp.** 0.1° mean deflection while the
  obstacle is outside the horizon, **103°** once inside — and that magnitude is saturated
  at every scale from 1.5 to 4.0 m/unit. Raising the scale buys *warning*, never
  proportionality. Softening the cliff needs a retrain with a larger `lidar_range`.
- **It commands no yaw.** The action is a 2-D force, so a policy-driven robot crabs and
  never turns — and the camera is a forward 85° cone with no rear view, so it also never
  looks anywhere new. `integration/mappo_policy.py` adds an optional heading servo for
  exactly this.
- **Episodes were 100 steps.** Ten seconds at the stack's 10 Hz. A 60 s demo run is six
  times longer than anything it saw.
- **It is a 3-agent shared actor, and peers are invisible to this stack.** Another
  quadruped is neither a detector class nor a colour profile.
