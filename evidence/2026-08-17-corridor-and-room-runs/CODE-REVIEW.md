<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code review, 2026-08-17

A software-engineering pass over `integration/`, `policy/` and `robot-stack/`, done after
the live runs so the findings could be checked against real telemetry rather than argued
from the source. Everything below was reproduced; nothing is a hunch.

Ordered by consequence. The two `control_dt` and heading-servo items are covered in the
run README and the issues, and are only cross-referenced here.

---

## A. Correctness

### A1. `feasible` and `evaluated` are silently zero on every policy-driven tick

`MappoPlanner.plan` forwards those two counters on the veto branch and drops them on the
other two:

```python
# veto branch — forwards them
return Command(planned.vx, planned.vy, planned.wz, reason=f"veto-{planned.reason}",
               gap_m=planned.gap_m, feasible=planned.feasible, evaluated=planned.evaluated)

# policy branch — does not
return Command(proposed[0], proposed[1], proposed[2], reason="policy", gap_m=planned.gap_m)
```

`Command` defaults both to `0`, and `telemetry.py` writes them unconditionally. So the
recorded evidence says:

```
run5:  reason=policy       feasible=0    evaluated=0     x58
run1:  reason=veto-avoid   feasible=330  evaluated=330   x5
       reason=policy       feasible=0    evaluated=0     x39
```

**`feasible=0 evaluated=0` reads as "the planner sampled nothing and found nothing
feasible" — i.e. completely boxed in — when the truth is that it evaluated 330 candidates
and most cleared.** The incumbent is computed on *every* tick by design (for acceleration
and hysteresis continuity), so the numbers exist and are simply thrown away.

This is the same class of defect the repo has already been bitten by twice: a substituted
default that is indistinguishable from a real measurement. It corrupts exactly the ticks
the demo cares about most.

**Fix:** pass `feasible=planned.feasible, evaluated=planned.evaluated` on all three
branches. One line each.

### A2. Two consumers raise `KeyError` on a `command` dict this repo itself produces

`mappo_policy.tick_from_state` builds `"command": {"reason": reason}` — no `vx`/`vy`/`wz`.
Two readers assume the full shape:

```python
# observation.observation_from_tick
command = tick.get("command") or {"vx": 0.0, "vy": 0.0, "wz": 0.0}
...
speed=(command["vx"], command["vy"], command["wz"])      # KeyError

# telemetry_reader.Run.moving_ticks
if t.get("command") and any(abs(t["command"][k]) > 0.0 for k in ("vx", "vy", "wz"))
```

Reproduced:

```
tick['command'] as this repo builds it: {'reason': 'goal'}
  observation.observation_from_tick: KeyError: 'vx'
  telemetry_reader.Run.moving_ticks: KeyError: 'vx'
```

The `or {...}` fallback looks like it handles this and does not: it only fires when
`command` is absent or empty, never when it is present-but-partial, which is the case that
actually occurs. Not currently reachable on the drive path — `robot_input` uses `.get()`
throughout — but it is a live trap for the next consumer, and `moving_ticks` is public API
on the reader.

**Fix:** `command.get("vx", 0.0)` in both, or give `tick_from_state` the full command shape.

### A3. `replay_mappo`'s "could it see one" split is measured from the wrong source

`replay()` decides visibility from the *telemetry's* obstacle list at that tick:

```python
surface = _nearest_surface_m(tick)          # from tick["obstacles"]
"visible": surface <= horizon_m,
```

but the policy steers on **its own internal map**, which retains obstacles for
`static_obstacle_ttl_s = 120.0` seconds. The two are different data sources and they
diverge whenever a landmark leaves the telemetry — a re-association, a new `landmark-N`,
a map entry that stops being `confirmed()` — while the controller still remembers it.

This is the mechanism behind the anomaly filed as issue #17: the replay attributes
**13.8° mean / 67.7° max** of obstacle-caused deflection to the 43 run-5 ticks it labels
"could not see one". Under the label's own logic that must be zero. The max being *larger*
on the unseen ticks than the seen ones is the tell.

**Consequence: the headline "34.8° on the ticks it could see one" is not trustworthy as a
split**, even though the overall `obstacle_effect_deg` (which compares two controllers with
identical histories) remains sound. The success claim survives; the breakdown does not.

**Fix:** derive `visible` from the controller's own map, not from the tick. Failing that,
rename the field to say it is the telemetry's view and stop drawing conclusions from the
split.

**Confirmed by measurement, and it turned out to be two problems.** Replaying run 5 back
through the controller and inspecting `_obstacles` per tick:

```
ticks the replay labels UNSEEN : 43
  ... of those, with something INSIDE the policy's own fan:   16
      their policy-fan nearest range:  0.484 m to 0.864 m

map size vs tick obstacle count:
      tick objects:  min 0, max 2
      policy map  :  min 0, max 4
```

So the label is wrong (this finding, issue #17) **and** the policy is genuinely carrying
twice as many objects as perception ever reported and steering on them at 0.484 m — which
is a robot-behaviour defect belonging to the policy package, filed separately as issue #19.
`static_obstacle_ttl_s` is 120 s against 20 s runs, so `_expire()` never fires and the map
is append-mostly for the whole episode; there is no disagreement-based expiry to catch a
ghost, and the map is in an odometry-derived frame so stale entries wander rather than
merely persist. 4-for-2 is the same shape and the same count as the control stack's own
odometry-induced landmark duplication.

Run 5's headline number is unaffected — `obstacle_effect_deg` compares two controllers with
identical histories, so ghosts cancel — but the seen/unseen split must not be quoted.

**FIXED.** `policy_sight()` now reads visibility out of `controller.last_observation`,
which is public, is already consumed by `mappo_policy`, and *is* the policy's perception by
definition. The corrected run-5 numbers, and the reason to believe them:

```
ticks the POLICY could see one 31/70 (44%)     <- was 15, measured from the telemetry
  ... on the 31 ticks it could see one: max  67.7, mean  35.9
  ... on the 39 ticks it could not:     max   0.0, mean   0.0     <- was 67.7 / 13.8
```

**The unseen row is now exactly zero, which is the proof.** An obstacle absent from the
observation cannot change the action, so the live and ablated controllers must agree
bit-for-bit there. Any non-zero value in that row is a defect in the tool or the
controller, and the summary now says so in place. The 13.8° that used to sit there was the
symptom this finding is about.

The tool also gained the ghost detector as a by-product — the gap between the two
visibility measures is the #19 signal, so it is reported rather than collapsed:

```
⚠️  16/70 ticks had an obstacle inside the policy's fan that the telemetry did not
    report ... Closest such ghost: 0.484 m. Issue #19.
```

`replay_mappo.py` also had **no tests at all** despite producing the numbers this project
quotes as evidence. It now has six.

### A4. `_record_refusal` promises "never raises" and can

```python
"""Append one refusal record. Never raises — a failed log must not mask the refusal."""
...
except OSError as exc:
```

`json.dumps` raises `TypeError`, not `OSError`, on anything unserialisable — a `Path`, a
numpy scalar. Today's only caller passes floats, so it is safe *by luck of the call site*
rather than by the guarantee the docstring makes. If it ever fires, the traceback replaces
the carefully written refusal message that exists specifically so a demo-day operator knows
why the run did not start.

**Fix:** catch `(OSError, TypeError, ValueError)`, or serialise defensively.

---

## B. Dead code and comments that describe something else

### B1. `integration/observation.py` presents itself as the policy's adapter; nothing uses it

The module docstring opens:

> *"Turn one telemetry tick into the observation a MAPPO policy expects. … `range_vector`
> is the adapter."*

It is not. The shipped checkpoint's observation is built **inside
`policy/physical_ai_mappo.py`**, which has its own ray caster. Across the whole repository,
`observation_from_tick`, `range_vector`, `reliable_range_m` and the `Observation` class are
referenced **only by `test_observation.py`**. The live path imports exactly two helpers
from this file — `wrap_pi` and `to_body_frame`.

So roughly 150 lines of ray-versus-disc geometry, plus a long and genuinely interesting
docstring about ray-fan blind spots, sit outside the code that runs. Two concrete ways this
misleads:

- `DEFAULT_RAYS = 16`, while the delivered checkpoint uses **12**. A reader checking the
  fan against `reliable_range_m` computes the wrong number.
- The `Observation` layout is `[goal polar, 16 ranges, last command]` = 21 values. The
  policy's is `[x, y, vx, vy, x-gx, y-gy, *12 lidar]` = 18. They are not the same
  observation and the file does not say so.

**Fix:** either state plainly at the top that this is analysis machinery and the shipped
adapter lives in the policy package, or move `wrap_pi`/`to_body_frame` into a small
`geometry` module and let the rest be an explicitly-named analysis tool. The ray-fan
analysis is worth keeping — it just is not on the path it claims to be on.

### B2. ~~The runbook's rung-2 command names a venv that does not exist~~ — WITHDRAWN, I was wrong

**This finding was incorrect and is retracted.** `deploy/install.sh:33` reads
`ENV_DIR="${ENV_DIR:-$HOME/robotics-connect-go2}"`, so `~/robotics-connect-go2` is exactly
what a default install creates and exactly what the runbook documents. The docs are right.

What I actually observed was the lab Go2's deploy manifest recording
`env_dir /home/unitree/robotics-connect-envs/armwaheed` with `created_env 0` — that
machine was installed with `--env-dir` pointed at a pre-existing per-researcher venv. I
inferred a documentation defect from a single non-default deployment without reading the
installer, and asserted it in four places before checking.

There is a real, smaller improvement underneath it, and it has been made: the runbook now
says the path is the installer's *default* rather than a guarantee, and points at
`~/.mappo-go2-deploy.manifest`, which records the `env_dir` an install actually used.

---

## C. Structure and consistency

### C1. Five directories of `robot-stack/` are not linted at all

`ruff.toml` exists in `integration/`, `policy/`, `robot-stack/unitree/go2/visual_nav/` and
the two Lite3 directories. There is **no root config**, so these have nothing above them:

| directory | `.py` files |
| --- | --- |
| `unitree/go2/controller` | 2 |
| `unitree/go2/d1_arm` | 5 |
| `unitree/go2/deploy` | 2 |
| `unitree/go2/lidar_sight` | 3 |
| `unitree/go2/locomotion` | 1 |

Even ruff's *default* rule set finds dead imports there, which means no one has ever run it
in those directories:

- `d1_arm/d1_arm.py:43` — `import math`, never used
- `lidar_sight/go2_lidar_sight.py:31,35` — `import sys` and `from pathlib import Path`,
  neither used

**Fix:** one root `ruff.toml`, or a config per directory matching the existing pattern.
Belongs upstream, since these are vendored files.

### C2. Three functions are well past the project's own complexity thresholds

Running the repo's documented SE pass
(`--select E722,E731,C901,PLR0911,PLR0912,PLR0913,PLR0915`):

| location | finding |
| --- | --- |
| `visual_nav.py:508` `run` | C901 **18**, 18 branches, **96 statements** |
| `visual_nav.py:1038` `main` | C901 **13**, 71 statements |
| `replay.py:60` `main` | C901 **20**, 20 branches, **104 statements** |
| `telemetry.py:128` | **15 arguments** |
| `visual_nav.py:387` | 12 arguments |

`run()` is the one that matters: it is the control loop, it is where the `control_dt` bug
lived, and at 96 statements it is where the next one will live too. The loop body mixes
health checks, perception consumption, staleness handling, goal search, arrival, planning,
rest-when-blocked and telemetry in a single `while True`.

No finding in `E722` (bare except) or `E731` (assigned lambda) anywhere — that part is
clean.

### C3. `robot-stack/` has diverged *forward* from upstream, which inverts the vendoring contract

`PROVENANCE.md` says *"Upstream is the source of truth."* It currently is not. These exist
only in this repository:

- `visual_nav/robot_bindings.py` (172 lines)
- `visual_nav/lifecycle.py` + `test_lifecycle.py`
- ~307 lines of `visual_nav.py` restructuring, plus `calibrate_camera.py` and `camera.py`
  changes

They arrived with the Lite3 port and were never upstreamed. The danger is specific and
silent: `PROVENANCE.md`'s own re-vendor recipe is a whole-tree `rsync` from upstream, which
would **revert all of it, and no test would fail** — the tests would be reverted too.

**Fix:** upstream the bindings refactor, or record it explicitly in `PROVENANCE.md`'s
deliberately-different list so the re-vendor recipe skips it. Right now it is in neither.

---

## D. Efficiency

### D1. Every tick pays for the planner twice, on a loop already at a third of its design rate

`MappoPlanner.plan` runs the full DWA rollout (`evaluated=330` candidates × a 2.5 s
horizon) and then, when supervised, `is_feasible` rolls out once more. The full plan is
deliberate and documented — the incumbent's acceleration window and reason hysteresis must
stay continuous — and that reasoning is sound. But the cost lands on a loop measured at
**2.78–3.58 Hz against a nominal 10**, and the policy path is roughly half the rate of the
planner path (5.1 Hz on run 0).

Not a bug, and not obviously the dominant term — the DWA rollout may well cost the same in
both paths. **It has never been profiled**, which is the actual gap. Filed as issue #18
with the specific measurement to take.

`is_feasible` also re-derives a rollout the plan just computed for a superset of commands;
whether the proposed velocity can reuse that work is worth a look once the profile says
where the time goes.

---

## What is good, and worth not regressing

Stated because a review that lists only faults misrepresents the codebase.

- **The comments earn their place.** Most explain *why*, name the measurement behind a
  constant, and record what was already tried and falsified. `MOVER_SPEED_MPS`,
  `VELOCITY_FRAME` and `_ray_hits_disc`'s "inside the disc is zero" note are all things a
  reader would otherwise get wrong.
- **`BridgeReport.merge` rejects unknown counter names** rather than dropping them, so a
  mistyped counter fails loudly instead of reading as "no problems found".
- **`Config.load` names unknown keys**, which catches the misspelling that would otherwise
  silently keep a default.
- **The paired ablated control in `replay_mappo`** is the right instrument for the question
  and is why "MAPPO worked" is a measurement rather than an impression — A3 notwithstanding.
- **`mappo_shadow` holds no locomotion client and opens no DDS channel**, and says so at the
  top. That is a safety argument a reader can verify in ten seconds.
- The refusal path writes `~/.mappo-refusals.jsonl` because a refused run leaves no
  telemetry. That is the kind of thing normally learned the hard way.
