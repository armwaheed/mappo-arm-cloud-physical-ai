<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Provenance of `policy/`

`policy/` is the MAPPO deployment package **authored by Sagar Surendran**
([@spsagar13](https://github.com/spsagar13)), vendored here so the demo runs from a clean
clone and so every replay in an issue quotes a checkpoint anyone can load. It is his
deliverable; this repository holds a copy, not the original.

| | |
| --- | --- |
| Delivered as | `physicalai_mappo_go2.zip` |
| Delivered | 2026-08-12 |
| Vendored | 2026-08-13 |
| Checkpoint | `models/mappo_actor_3agent_1910000.npz`, 268 063 bytes |
| Checkpoint SHA-256 | `7327f72401adfdfa1931a516e85aeee62b5bee0e06e976c13600515ca2d2ca11` |
| Trained from | `3_agent_m9g48xl_checkpoint_1910000.pt` (per the npz's own `metadata_json`) |

The checkpoint is committed. At 262 KiB it is a text file's worth of weights, and a demo
whose model lives in someone's Downloads folder is a demo that runs once.

## What the checkpoint says about itself

The npz carries a `metadata_json` array. Nothing in the delivered adapter read it; the
vendored one validates the config against it, because these are the only in-band
statement of what the network was trained with:

| | |
| --- | --- |
| `actor_input_dim` | 18 — `[x, y, vx, vy, x-gx, y-gy, *12 lidar]` |
| `actor_hidden_dims` | `[256, 256]`, tanh |
| `actor_raw_output_dim` | 4 — a `TanhNormal`'s `loc` and `scale`; deployment uses `tanh(loc)` |
| `training_lidar_range_vmas` | 0.35 |
| `training_agent_radius_vmas` | 0.1 |
| `training_n_agents` | 3, shared parameters |
| `training_max_steps` | **100** |
| `training_frames` | 1 910 000 |

`training_max_steps: 100` is worth reading twice. At the stack's 10 Hz control rate an
episode the policy was trained on is **ten seconds long**. The recorded demo run is 60 s,
which is six times anything it saw in training. Nothing enforces an episode length here
and nothing should — it is recorded because "the run went on longer than the policy has
ever been asked to act for" is a plausible reading of a late failure, and it is not
otherwise written down anywhere.

## Deltas

Every change to the delivered files is listed here and marked `CORRECTION` in the source.
The first three were agreed with @spsagar13 in AIDP-567; the fourth and fifth were found
while writing tests for the first three. **All five were silent failures** — nothing
raised, nothing logged, and the delivered `basic_test.py` passed with every one of them
in place.

| # | file | delivered | vendored | why it was invisible |
| --- | --- | --- | --- | --- |
| 1 | `physical_ai_mappo.py`, `config.json` | `velocity_frame` defaults to `"odom"` | a config field, defaulting to `"body"` | `measured` is the Go2 estimator's body-frame velocity. The two frames agree **exactly** at the start heading and diverge only in a turn, so a bench test at yaw 0 cannot tell them apart. |
| 2 | `physical_ai_mappo.py` | a negative age is treated as fresh | `STOP_CLOCK_ERROR`, command zeroed | `timestamp_s` is compared against `time.monotonic()`. A wall clock gives an age of ≈ −1.8e9 s, under every threshold, so `STOP_STALE_INPUT` **could never fire**. It failed open. |
| 3 | `physical_ai_mappo.py` | config is trusted | config validated against `metadata_json` | `lidar_range_vmas` is the range the proximity convention is measured against. Disagree with training and every value stays finite and in range; the robot just steers wrongly. |
| 4 | `physical_ai_mappo.py` | positional association ran on **every** obstacle | only on ones that are anonymous or already this id | Two objects 0.2 m apart merged even when the producer had told them apart — which is exactly what `id` was added to the telemetry to prevent. The merged disc takes the larger radius, so the range vector still looks plausible. |
| 5 | `physical_ai_mappo.py` | `Config.load` is `cls(**json.load(...))` | unknown keys named, values range-checked | A **misspelled** key is worse than an unknown one: the field it was meant to set keeps its default and the file reads as though it took effect. |

Two further changes are calibration and packaging rather than corrections:

- **`meters_per_vmas_unit` 1.5 → 2.5.** Confirmed by @spsagar13 as a calibration
  parameter, not a model requirement. 1.5 matched the *room* to the trained spawn region;
  2.5 matches the *robot* to the trained agent (0.25 m planner radius ÷ 0.10 VMAS agent
  radius). See issue #4 and `integration/replay_mappo.py --scale`.
- **`command_scale` 0.30 → 0.60.** Not a model property at all — it is how much of the
  robot's envelope the policy is allowed to ask for. Raised because **this Go2 delivers
  about 0.45 of the velocity it is commanded** (fitted against the pose over the recorded
  run: 2.09 m travelled against 4.32 m commanded), so 0.30 is 0.047 m/s on the floor —
  **2.8 m in the entire 60 s run budget**, less than the 3 m arena is wide. It is a speed
  knob and not a safety one: `mappo_drive` clamps every command to the control stack's
  own `Limits`, which is what `--derate` scales.
- **`basic_test.py` kept, `test_physical_ai_mappo.py` added.** The delivered file is one
  inference and two range assertions; it is a fine *install* check and it is what
  `deploy/install.sh` runs on the target machine, so it keeps its name and its `PASS`.
  It is not a test of the policy — it would pass with the weights replaced by noise — and
  the README no longer implies otherwise. The 30-test suite beside it fails if any delta
  above is reverted.

`README.md` is rewritten rather than diffed: the delivered one documents the delivered
behaviour, and leaving `velocity_frame="odom"` in a copy-pasteable example is how a
correction gets undone.

## If a new checkpoint arrives

1. Drop the `.npz` into `models/` and update the table above, **including the SHA-256**.
2. `cd policy && python3 test_physical_ai_mappo.py` — the config-versus-checkpoint check
   is the one that catches a checkpoint trained with a different lidar range, and it
   reports it as a config error, which is what it is.
3. `cd integration && python3 replay_mappo.py ../evidence/sample_telemetry.jsonl --scale 1.5 2.5` —
   the same recorded run through the new weights, against the paired ablated control.
   Compare the table with the one in issue #4 before believing anything about it.
4. Re-run the closed-loop sim (`integration/closed_loop_sim.py`). A new checkpoint has
   not cleared issue #5's gate just because the old one did.

## If the adapter is re-delivered

Diff it against this copy and re-apply the five deltas — they are marked `CORRECTION` in
the source for exactly this reason. A whole-file overwrite silently reverts all five and
**no test in `integration/` fails**, because they are failures in the policy package's own
behaviour rather than in the mapping into it. Run `policy/test_physical_ai_mappo.py`.
