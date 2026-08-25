<!--
Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cold-robot protocol: settling the Go2 gait floor in one 20-minute session

Written to be run by somebody who was not there on 2026-08-24. Everything it needs is
here or in `tools/gait_sweep.py --sweep p --plan`; you do not need to read the four sweeps
that came before, though `README.md` beside this file says which of their conclusions
survived.

**The tool refuses rather than degrades.** If the robot is too warm, the arm has crept, the
legs are not listening or the robot has wandered off its mark, the run stops, parks the
robot prone, and prints what to do. There is no mode in which it produces a plausible file
from a session that could not answer the question.

---

## 1. Why this is worth a cold robot

The gait floor is the last unmeasured link in a chain that otherwise works.

A disc-inflation change now in progress makes the MAPPO policy genuinely swerve around a
moving peer: simulated clearance goes from 0.194 m to 0.505 m at the worst crossing speed.
But the policy's **peak lateral command in every configuration is 0.085–0.182 m/s**, and
three live runs have already issued `v = (+0.000, -0.150)` and stood still inside their own
hard gap with an escape available. The scaler that exists to fix that,
`integration/mappo_drive.py::_at_least_walking_pace`, raises a small command to a floor —
and the floor it is given is a number hardware contradicts. `MIN_GAIT_COMMAND_M_S = 0.35`
is not a floor: 0.100 m/s produced 0.114 m of travel.

So: measure the floor, on the axis the swerve needs, in the motion state the swerve
happens in, and the whole chain lands. Get it wrong and the sim number is the only number
there is.

---

## 2. Established — do not spend budget re-deriving these

| fact | evidence |
| --- | --- |
| `MIN_GAIT_COMMAND_M_S = 0.35` is **not** a floor | 0.100 m/s → 0.114 m of travel (`sweeps/sweep_a.json`) |
| Forward commands **≥ 0.25 m/s walked in every trial**, whatever the order | 514 telemetry samples, 88–95%, plus every staged trial |
| The **same** low command walks or stalls depending on motion state | every low-speed success was the first trial after standing (0.114, 0.120 forward; 0.127 lateral); every repeat gave 0.004–0.013 m |
| The odometry is trustworthy | null control, standing, zero command: 0.001 / 0.000 / 0.000 / 0.000 m. Anything past ~0.01 m is real motion |
| Motion mode `'mcf'` obeys `Move` | the mode stayed `'mcf'` all session and every `Move` was obeyed, up to 0.707 m in 3 s. `Go2Locomotion`'s warning is misleading |

**Open**: the starting floor. The sustaining floor. Whether surface matters. Whether the
floor's shape across bearings is an ellipse.

---

## 3. The design

### 3.1 Holding position fixed when walking is the thing being measured

The circularity is real: you cannot hold a robot still and measure whether it walks. The
way out is that **the outcome is its own position controller**, on one axis and not the
other.

* **A stall does not move the robot.** 0.004–0.017 m, which is the null control. Near the
  floor — the only place the answer lives — most trials stall, so most trials cost nothing
  in floor space. The design is self-centring in exactly the regime it is measuring.
* **A success moves the robot about 0.13 m** in a 2 s trial in the band under test, and
  `vy` may be signed. **Alternating the strafe direction makes the robot oscillate about
  the mark rather than walk away from it.** Beyond 0.15 m of drift the sign stops
  alternating and points home. This is why the primary grid is LATERAL: `vx` may never be
  negative on this unit (no rear sensing), so forward travel is one-way and cannot be
  undone by the next trial.
* **Two hard gates** stand behind the controller, because a mechanism that usually works is
  not the same as a confound made impossible. No trial starts more than **0.50 m** from the
  mark — the run refuses and tells the operator to walk the robot back — and any single
  trial that travels **0.80 m** is truncated mid-hold and recorded as truncated. The
  position budget cannot be enforced only between trials if one trial can spend all of it.
* **Offsets are measured in the mark's own frame**, not in odom, so a strafe is never
  mistaken for a forward drift when the mark's heading is not zero.
* **The tape is a second instrument.** The gate above is odometric. Put a tape cross on the
  floor and eyeball it at each break: if the tape and the printed `along/across` disagree by
  more than a few centimetres, say so in the session notes — that is odometry drift, and it
  is worth more than the trial it interrupted.

There is exactly **one block that does not hold position, and it cannot**: block D measures
a robot that is already moving, so moving is the precondition, not a confound. It re-marks
after each of its two trials, and the scheduled break before block E puts the robot back on
the tape so that the one comparison that needs position held gets it.

The whole session therefore needs **1.8 m ahead of the tape and 1.0 m each side** — against
the ~4 m × 1 m the corridor was asked for on 2026-08-24, and most of the 1.8 m is block D.

### 3.2 Varying motion state while position is fixed

Three states, and the third is the one the previous session never named:

| state | how it is entered | what it models |
| --- | --- | --- |
| `FRESH` | `stand_up()` (or `BalanceStand` alone — block B decides), then the command within ~0.2 s | the state every low-speed success on 2026-08-24 happened to be in |
| `REST` | standing, stopped, **verified** below 0.02 m/s for three consecutive samples | a robot restarting after a hold |
| `ROLLING` | a forward primer at 0.300 m/s, then the command steps down without ever passing through zero | a robot mid-swerve, which is the demo case |

State is **verified from telemetry at the instant the command goes out**, not assumed. A
"REST" trial issued while the robot was still coasting is recorded `state_verified: false`
and excluded from every table. The 2026-08-24 script verified nothing.

`FRESH` has to be re-enterable at an arbitrary point in the trial order or the whole design
collapses back into "first trial after standing", so **block B measures which call restores
it** before anything depends on the answer. If `BalanceStand` alone is enough, a refresh
costs 1 s; if only the full `stand_up()` works, 3 s and the optional blocks get dropped.

### 3.3 The four confounds, made impossible rather than unlikely

**1 — Position changed between trials.** Section 3.1. Enforced by the sign controller, a
0.50 m refusal gate, a 0.80 m mid-trial truncation, and a per-trial `along_m` / `across_m`
column so any residual effect is visible rather than argued about.

**2 — No null control.** A zero-command null runs **after every single commanded trial**,
generated by the plan rather than remembered by an operator; `test_gait_sweep.py` fails if
one goes missing. The null is very nearly free — a `REST` trial had to wait for the robot to
settle anyway, so commanding zero through that interval costs one extra pose read. It buys a
noise measurement at the same position, temperature and battery state as the trial it
controls, instead of one taken at the end of the day after the third reversal. The
classifier's threshold is then set from **this session's own nulls**, four times the worst
one seen so far.

Three of those nulls are run in the `FRESH` state, which is the only kind that can refute
"the first-trial walks are the robot settling out of `stand_up()`". On 2026-08-24 that
refutation existed by luck: the null control happened to be run first.

**And the null is only half of the control.** A robot whose legs were never enabled, or that
is pressed against a wall, or whose arm is blocking, records a **perfect** null — and then
records 0.000 m for every real command too. So each end of the session carries an
**anchor**: 0.300 m/s forward for 2 s, a command that walked in every trial of every
session. If the opening anchor travels less than 0.15 m the run refuses immediately and
nothing is measured. If the closing anchor fails, every trial since the opening one is
suspect and the session says so in its summary. This is the half of the lesson 2026-08-24
did not learn, and it is also what would have caught the wall-contact runs that poisoned the
retrospective corpus.

**3 — Dry runs.** `--dry` is **refused** for sweep P. A dry protocol run writes a
full-looking results file in which nothing moved, which is the exact shape of the confound
that made two dry runs supply 198 of 301 from-rest samples in the retrospective analysis.
Rehearse with `--plan`, which produces no results file at all. A run is confirmed live by
three things in order: the `latch drift … HELD` line, the `mode now:` line, and — the only
one that actually proves it — **the opening anchor's travel**.

**4 — Non-monotonic results treated as data.** The tool computes it. After every block, and
again at the end, any cell where a **bigger** command produced motion **less** often than a
smaller one is printed as `*** NON-MONOTONIC` with the instruction not to read a floor off
that block. Comparison is within a state, never pooled across states — a starting floor
above a sustaining floor is precisely a case where `FRESH` walks at a speed `REST` does
not, and pooling would report that real effect as a confound.

### 3.4 Three nuisances that rise monotonically, and were never separated

Motor temperature, D1 arm sway and battery charge all rise or fall monotonically through a
session, and on 2026-08-24 **all three were perfectly confounded with "first trial after
standing"**. Cold legs walking at 0.100 m/s fits that day's data exactly as well as the
state hypothesis does, and nobody named it.

* Both passes of block C run the states **and** the commands in reversed order, so a
  monotone drift cancels to first order rather than being regressed out afterwards.
* Every trial records `temp_c`, `hottest_motor`, `soc_pct`, `arm_sway_deg` and
  `arm_reach_m`, so what could not be cancelled can at least be plotted.
* **Block E is the limit case**: the identical cell block B ran cold is repeated at the
  session's peak temperature. It is the cheapest discriminator available and it costs four
  trials, so its budget is reserved before anything else may spend.

### 3.5 What this design cannot answer

Stated up front so nobody reads more into the results than is there.

1. **The shape of the floor across bearings.** Only 0° and 90° are measured under
   controlled state, plus the two intermediate bearings inside block D's staircase, once
   each. Everything between the axes remains interpolation. The ellipse is not tested.
2. **Surface.** It is **held constant**, not varied — every trial happens on one patch of
   floor. That is strictly better than 2026-08-24, where it varied uncontrolled and
   position could not be separated from trial order, but the result is *the floor on that
   patch*. Testing the metal strip's other side means a second block on a second mark, and
   the thermal budget does not have room for it. Stage on the surface the demo lane uses,
   and write down which one it was.
3. **The sustaining floor as a threshold.** Block D is two descents, not a distribution. It
   can say "the gait survived to this step" or "it died here"; it cannot put a number on
   the boundary, and step order is confounded with elapsed time — which is exactly why the
   cliff trial exists, and the cliff is n = 1.
4. **Duration beyond ~8 s.** The longest hold in the session is 4 s at the final step,
   matched to issue #31's recorded 4.1 s stall. Whether a gait that survives 4 s survives 30
   is not asked.
5. **Yaw.** Every trial commands `wz = 0`. A real swerve may include yaw, and yaw may itself
   change the gait state.
6. **Closed-loop commanding.** Every number here comes from a **held** command; the
   navigator issues a new one every 100 ms. Optional block F is a first probe at this and
   nothing more — if a dithered command behaves differently from a held one, the whole
   held-command paradigm needs re-reading before its floor is written into
   `_at_least_walking_pace`.
7. **Mechanism.** This measures behaviour. If `BalanceStand` turns out to re-arm the gait,
   that is an operational fact about the vendor controller, not an explanation of it.
8. **Battery band and payload.** One arm pose, one battery. A 40% battery may answer
   differently.

---

## 4. Staging

Do all of this **before** the robot stands up. Standing is the budget.

**People.** Two. One on the terminal, one on the remote with a thumb on it, standing beside
the lane and not in it. Nobody walks through the lane once the run starts.

**The robot.**

1. It must be **cold**. Not "cooled down a bit" — the previous session ended at 48 °C
   against a 50 °C ceiling and cooling while prone is real but slow. The whole plan needs a
   start temperature at or below **34 °C**; blocks A–E alone need **38 °C**. The tool reads
   the temperature itself, prints the budget, and drops the optional blocks if it must —
   and if even A–E will not fit it says so **before anything stands up**, with the
   shortfall in seconds. Every 1 °C you wait for buys 16 s. Waiting is cheaper than a
   session that stops in the middle of block D having already spent its budget on C.
2. **Hand-pose the D1 flat along the spine and as low as it will go**, supporting its
   weight. Never lift the robot by the arm. Do this immediately before the run, not twenty
   minutes earlier: the arm creeps out of its 3.0° gate about every 20 minutes and then
   refuses the latch.
3. Leave it **prone** on the start line. The tool does the latch, the mode and the stand.

**The floor.**

* Pick the surface the demo lane actually uses. Write down which side of the metal strip it
  is, and put the mark **at least 0.7 m from the strip** so that the 0.50 m position budget
  cannot carry the robot across it.
* Tape a cross for **the mark**. Tape a second line **0.35 m behind it** — the start line.
  The opening anchor walks the robot from the start line onto the mark.
* Clear **1.8 m ahead of the mark** and **1.0 m each side**. The forward requirement is
  almost entirely block D; blocks A–C alone need 0.7 m.

**The tool.**

```bash
# workstation -> robot (the address is in your local ~/.robot-creds; never in this repo)
scp evidence/2026-08-24-peer-capture-and-gait-sweeps/tools/gait_sweep.py \
    unitree@<go2>:~/peercap/tools/

# on the robot: read the schedule. This touches nothing.
cd ~/peercap/tools && python3 gait_sweep.py --sweep p --plan
```

`--plan` prints all 64 trials, the per-block cost in standing seconds and in degrees, and —
if you pass `--start-temp` — which blocks fit. Read it once. It is the same list the run
executes.

---

## 5. The run

```bash
python3 gait_sweep.py --sweep p --out ~/peercap/gait_p_$(date +%H%M).jsonl
```

Nothing else. No flags. `--out` must not already exist: sweep P writes each trial as it
finishes, so an interrupted session keeps what it measured, and overwriting one is how a
resume destroys the data it was resuming from.

Total: **~250 s of standing** across 64 trials, plus two prone breaks and the setup — under
ten minutes of the twenty. The rest of the window is deliberately left as **re-run budget
for one non-monotonic block**. The binding constraint is the thermal budget, not the clock.

### Say this, in this order

| when | say |
| --- | --- |
| before you press Enter | "Lane clear? Anyone crossing in the next ten minutes?" — then wait for the answer |
| at `mode now:` | "Selecting the mode can make it shift. Standing now." |
| at the opening anchor | "This one walks about a third of a metre onto the mark. It is the only trial that proves the legs are listening." |
| at each `SCHEDULED BREAK` | read out what the tool printed, do it, then say what you did before pressing Enter |
| at a `***` line | read it aloud. It means something the design did not hold still has moved |
| at the end | "Prone. Results at …" |

### Block by block

| block | trials | ~cost | what it asks | what a result would falsify |
| --- | --- | --- | --- | --- |
| **A** | 3 | 14 s | Are the legs listening, and is the odometry honest? | Anchor under 0.15 m ⇒ **nothing measured afterwards means anything**; the run refuses. Check, in order: sport mode, arm latch, whether the robot is against something, battery |
| **B** | 8 | 28 s | Which call re-enters the state a low command can start in? | `BalanceStand` alone works ⇒ the state is re-armable in 1 s, and **scaling is not the only fix available to `_at_least_walking_pace`** — re-arming the gait before a small command is a second one. Only `stand_up()` works ⇒ the state cannot be entered mid-run, and scaling is the only lever. Nothing works and the anchor passed ⇒ the lateral floor is above 0.150 everywhere, which is itself a result |
| **C** | 26 | 101 s | **The primary result.** Lateral, 0.085 / 0.150 / 0.200 m/s × `FRESH` / `REST`, twice each, counterbalanced | `FRESH` and `REST` identical at every command ⇒ **the state hypothesis of issue #42 is dead**, and the floor is one number, read off the monotone column. They differ ⇒ two floors, and the `REST` column is the operational one. Every cell 0/2 at ≤ 0.200 ⇒ **the lateral starting floor exceeds `max_vy` and the envelope's lateral region is empty**: no scaler can execute a pure strafe, and the swerve has to keep a forward component or `max_vy` has to be raised on evidence |
| **D** | 4 | 21 s | The **sustaining** floor. A forward primer at 0.300 swung into a pure strafe at (0.000, 0.150) — the exact command three live runs could not execute — held for 4 s, matching issue #31's 4.1 s stall | The last step sustains ⇒ **the sustaining lateral floor is at or below 0.150 and the swerve is executable while the robot is already moving**, which in the disc-inflation scenario it is. This is the single result that unblocks the chain. It dies ⇒ compare the cliff: cliff sustains ⇒ what ran out was duration; cliff also dies ⇒ 0.150 lateral cannot be sustained at all |
| **E** | 5 | 20 s | Block B's cell repeated **hot**, plus the closing anchor. Reserved; never dropped | It walks ⇒ the state effect survives a session of heating and creep, and temperature is not the explanation. It stalls ⇒ "first trial after standing" was at least partly "coldest trial", and **the session has not settled the floor** — report that, not the grid |
| **F** | 6 | 21 s | *Optional.* Does a command dithered ±0.03 at 10 Hz behave like a held one? | Different ⇒ every number in this investigation describes a held command the navigator never issues |
| **G** | 12 | 45 s | *Optional.* The forward grid, 0.100 / 0.150 / 0.200 × two states | Brackets the gap issue #42 leaves between 0.137 (stalled 4/4) and 0.250 (always walked) |

Each trial prints one line: id, state, axis, command, net travel, tail speed, class, motor
temperature, arm sway. **`gait` / `shuffle` / `no-gait`** is a three-way verdict, not the
binary `travel > 0.05` that 2026-08-24 used: a shuffle is real motion that dies before the
hold ends, which is the difference between a swerve that completes and one that stops
halfway across the lane. Neither threshold is load-bearing — every 10 Hz sample is in the
results file, so both can be moved afterwards without going near the robot again.

---

## 6. When it refuses

Every refusal parks the robot prone first. `Move` persists until the next command, and
there is no dead-man.

**The D1 has left its stow gate.** Expected at least once — it happened twice on
2026-08-24, at 3.3° and 3.6°. Only you can re-pose it.

1. The robot is already prone. Hand-pose the arm flat along the spine, low, supporting its
   weight.
2. Resume with the block list the refusal printed, **plus block A**, and a new output file:
   ```bash
   python3 gait_sweep.py --sweep p --blocks A,C,D,E --out ~/peercap/gait_p_resume.jsonl
   ```
   Block A goes back in every time. It re-latches, re-anchors and re-marks, and it costs
   14 s. A resumed session without an anchor is a session you cannot read.
3. If the refresh branch had already been decided, pass it: `--refresh balance` or
   `--refresh full`. Otherwise block B decides again, which costs 28 s.

**Motors at the 50 °C ceiling.** Stop. Prone cooling is too slow to matter inside the
window, so the remaining blocks belong to a second session. The block order means the
primary result is already banked: A, B and C are the first 143 s.

**The robot is past its position budget.** Walk it back onto the tape, facing the same way,
and resume as above. Do not nudge it and carry on — position is the confound this design
exists to remove.

**A `*** NON-MONOTONIC` line.** Something the design did not hold still moved. If there is
budget, re-run that block (`--blocks A,C` and a new output file). If there is not, report it
as unresolved. Do not read a floor off it. This is not a cautious formality: a bigger
command working less often appeared four separate times on 2026-08-24 and was a confound
every single time.

**The closing anchor failed.** Everything since the opening anchor is suspect. Do not report
the grid.

**Anything else, or you just want it to stop.** Press **Ctrl-C once and wait**. The handler
sends a zero velocity immediately and the run unwinds to a `finally` that stops and lies the
robot down. If you press it repeatedly you may interrupt the parking itself — use the
remote's damp, and only when the robot is low or supported.

---

## 7. After

1. Pull the results off the robot: `scp unitree@<go2>:~/peercap/gait_p_*.jsonl .` A robot's
   home directory is not storage, and this one has been found reflashed before.
2. Re-run `test_gait_sweep.py` if you changed anything, and re-run the classifier over the
   saved samples if you want different thresholds — no robot needed.
3. **Write the continuation comment on issue #42** with the two tables the session printed,
   the anchors, the null control's worst net, and the temperature span. That comment is the
   handover; the next session is a fresh context.
4. The number that goes into `_at_least_walking_pace` is **not** the `FRESH` one. A run's
   commands arrive at a robot that is either at rest after a hold or already moving, and the
   robot is never freshly stood up mid-run. Take the larger of the `REST` starting floor and
   the block-D sustaining floor — and if the `REST` lateral floor turns out to sit above
   `max_vy`, say so loudly, because then the fix is not a bigger number in the scaler at
   all.
