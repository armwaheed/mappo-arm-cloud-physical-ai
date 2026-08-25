"""Falsify the two gait-floor claims the code currently rests on.

SWEEP A — is the forward floor really 0.35, and is it per-tick or sustained?
    `MIN_GAIT_COMMAND_M_S = 0.35` is a lowest-OBSERVED value. Upstream issue #32 records
    54/54 ticks below it while the robot walked ~3 m, mean commanded 0.295, and a stall at
    a SUSTAINED 0.137 m/s for 4.1 s. If that holds, the floor is lower than 0.35 and the
    failure depends on duration, not on any single tick. This holds each speed steady and
    measures travel.

SWEEP B — does the floor ellipse hold at 45 degrees?
    Two endpoints are measured (0.35 forward, 0.20 lateral) and the curve between them is
    an INTERPOLATION nobody has tested. The ellipse predicts
        floor(theta) = 1 / hypot(cos(theta)/0.35, sin(theta)/0.20)
    which at 45 deg is 0.246 m/s. Commanding exactly that at each bearing asks the robot
    the question directly: does it walk, or not?

SWEEP P — the DESIGNED experiment, and the only mode here that can settle the floor.
    A-D each produced a confident conclusion the data could not support, because four
    things varied together across a sweep: POSITION (a forward trial carries the robot
    0.5-0.75 m), TRIAL ORDER, MOTOR TEMPERATURE and D1 ARM SWAY. Every low-speed success
    on 2026-08-24 was the first trial after standing -- which is also the coldest trial,
    the least-crept, and the one nearest the start of the corridor. P separates them:
    position is held fixed and gated, the motion state is verified from telemetry at the
    instant the command goes out, a zero-command null control runs between EVERY trial,
    and the state order is counterbalanced so a monotone drift cannot masquerade as state.

    Read `PROTOCOL.md` beside this file before running it. `--plan` prints the whole
    schedule and its thermal arithmetic without touching the robot. `--dry` is REFUSED for
    sweep P: a dry protocol run writes a full-looking results file in which nothing moved,
    and that is precisely the confound that poisoned the retrospective corpus -- two dry
    runs supplied 198 of 301 from-rest samples there, all of them non-moving.

SAFETY, and none of it is optional:
  * `SportClient.Move` PERSISTS until the next command -- there is no dead-man. Every exit
    path stops the robot, in a `finally`, and the robot is put prone after.
  * vx is never negative. The Go2 has no rear-facing sensing on this unit, and a
    substituted planner driving blind backwards is an open upstream defect (#30).
  * The D1 arm is latched while PRONE, before sport mode is selected.
  * Motors are checked before standing and reported after every trial.
  * `--dry` runs the whole sequence including timing and reporting but commands no motion,
    so the script can be validated before anything moves. Sweeps A-D and Z only.
"""
import argparse
import contextlib
import json
import math
import os
import signal
import sys
import time

sys.path.insert(0, "/home/unitree/robotics-connect/unitree/go2/visual_nav")
sys.path.insert(0, "/home/unitree/robotics-connect")

FORWARD_FLOOR, LATERAL_FLOOR = 0.35, 0.20
SPEEDS = [0.10, 0.137, 0.175, 0.20, 0.25, 0.295, 0.35]
BEARINGS_DEG = [0.0, 22.5, 45.0, 67.5, 90.0]

#: Sweep C — find the TRUE floor at the bearings where the ellipse failed, by raising the
#: speed until the robot walks. Note these deliberately run PAST `max_vy` = 0.20: the
#: envelope is a software limit, and the question is what the hardware does. If the floor
#: at 90 degrees turns out to be above 0.20, then the shipped envelope's lateral region is
#: empty — the floor would exceed the ceiling and a pure strafe could never be walked.
#: Nothing here should be copied into Limits; it is a measurement, not a new envelope.
SWEEP_C_BEARINGS_DEG = [90.0, 67.5]
SWEEP_C_SPEEDS = [0.20, 0.25, 0.30, 0.35, 0.40]

#: Sweep D — REPEATABILITY, forward only. Sweep A walked at 0.100 and stalled at 0.137,
#: which is non-monotonic in commanded speed and therefore cannot be a property of the
#: gait. Either it is trial-to-trial noise or it is position, and a single pass down a
#: corridor cannot separate them because every trial lands somewhere new. Repeating the
#: same two speeds interleaved keeps both explanations in play and lets the numbers choose.
SWEEP_D_SPEEDS = [0.100, 0.137]
SWEEP_D_REPEATS = 3
STAND_CEILING_C = 50.0
REFRESH_HZ = 10.0


def ellipse_floor(theta):
    return 1.0 / math.hypot(math.cos(theta) / FORWARD_FLOOR,
                            math.sin(theta) / LATERAL_FLOOR)


def trial(loco, vx, vy, seconds, dry):
    """Hold one velocity, return measured travel and mean estimator speed."""
    start = loco.pose()
    samples = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not dry:
            loco.set_velocity(vx, vy, 0.0)
        samples.append(math.hypot(*loco.velocity()[:2]))
        time.sleep(1.0 / REFRESH_HZ)
    if not dry:
        loco.stop()
    time.sleep(0.6)                      # let it settle before reading the end pose
    end = loco.pose()
    travel = math.hypot(end.x - start.x, end.y - start.y)
    return travel, (sum(samples) / len(samples) if samples else 0.0)


# ── SWEEP P — the designed experiment ────────────────────────────────────────
#
# Everything from here to the runner is pure scheduling, classification and budget
# arithmetic. It imports nothing from the robot, so `--plan` runs on a workstation and
# `test_gait_sweep.py` can falsify it before the cold window opens.

#: Standard hold for a test trial AND for the null that controls it. The two are equal on
#: purpose: a null of a different length is not a like-for-like control. 2.0 s is the
#: shortest hold that still separates the two regimes measured on 2026-08-24 — a stall
#: travelled 0.004-0.017 m in 2.5-3.0 s (~0.006 m/s of net displacement) and the slowest
#: real walk 0.114 m in 3.0 s (~0.038 m/s) — while bounding what one trial costs in floor.
HOLD_S = 2.0

#: Settle before reading the end pose. The value sweeps A-D used, kept so their travel
#: numbers and P's stay comparable.
SETTLE_S = 0.6

#: Lateral commands. Not a uniform grid: these are the three numbers the rest of the chain
#: turns on. 0.085 is the MAPPO policy's smallest peak lateral command across the
#: disc-inflation configurations, 0.150 is the command three live runs issued as
#: v=(+0.000,-0.150) and failed to execute, and 0.200 is `Limits.max_vy` and the only
#: lateral speed ever measured walking (sweep_c, one trial, from a fresh stand).
LATERAL_COMMANDS = [0.085, 0.150, 0.200]

#: Forward commands for the optional block G. Brackets the gap issue #42 leaves open
#: between 0.137, which stalled 4 of 4, and 0.250, which walked in every trial.
FORWARD_COMMANDS = [0.100, 0.150, 0.200]

#: The ANCHOR: a positive control at a speed that walked in every trial of every session
#: (0.295 walked 0.539 m in 3.0 s; 0.350 walked 0.707 m). One at each end of the session
#: brackets everything between them.
#:
#: The null control alone is not enough, and 2026-08-24 only learned half the lesson. A
#: robot whose legs were never enabled, or that is pressed against a wall, or whose arm is
#: blocking, records a PERFECT null — 0.000 m — and then records 0.000 m for every real
#: command too. The null says the odometry is not lying. Only the anchor says the legs are
#: listening, and nothing measured after a failed anchor means anything.
ANCHOR_MPS = 0.300
ANCHOR_HOLD_S = 2.0
ANCHOR_MIN_TRAVEL_M = 0.15

#: Block D — the SUSTAINING floor, as (vx, vy, seconds). The command descends and swings
#: from forward to a pure strafe without ever passing through zero, so every step after
#: the first arrives at a robot that is already moving.
#:
#: THE PRIMER IS FORWARD, and that is not a detail. The only commands established to walk
#: this robot in every trial of every session are forward and at or above 0.25 m/s; the
#: lateral axis is the thing under test and has exactly one recorded success. Priming
#: laterally would mean the ROLLING state was unreachable on precisely the trials that
#: needed it, and the staircase would void itself. The swing from (0.300, 0.000) to
#: (0.000, 0.150) is also the shape of the manoeuvre the chain needs: a robot already
#: under way, turning a forward command into a strafe.
#:
#: The last step is 4.0 s because that is how long issue #31's recorded stall sustained
#: (0.137 m/s for 4.1 s), and (0.000, 0.150) is exactly the command three live runs issued
#: and could not execute. The cliff drops straight from the primer to that step: if the
#: staircase dies on its last step and the cliff sustains at the same command, what ran
#: out was elapsed time and not command.
STAIRCASE_STEPS = [(0.300, 0.000, 1.5), (0.150, 0.150, 1.5),
                   (0.050, 0.150, 1.5), (0.000, 0.150, 4.0)]
CLIFF_STEPS = [(0.300, 0.000, 1.5), (0.000, 0.150, 4.0)]

#: Dither for the optional block F. Every number in this investigation was measured with a
#: HELD command; the navigator issues a new one every 100 ms. If a command that wobbles by
#: +-0.03 m/s behaves differently from a held one, the whole held-command paradigm needs
#: re-reading before its floor is written into `_at_least_walking_pace`.
DITHER_MPS = 0.03

#: Motor heating measured on 2026-08-24: about 1 C per 16 s of standing, arm-loaded. Used
#: only to PLAN. The session re-estimates it from its own first blocks, and the gate that
#: actually stops a trial is the measured temperature, never the estimate.
ASSUMED_HEAT_C_PER_S = 1.0 / 16.0

#: How far the robot may sit from its mark before the next trial is refused. Two lateral
#: successes in the same direction is about 0.26 m, so this allows a run of them and still
#: stops well inside the ~1 m of lateral room the corridor offers.
POSITION_BUDGET_M = 0.50

#: Beyond this offset the lateral sign stops alternating and points back at the mark.
HOMEWARD_M = 0.15

#: A single trial is truncated if it carries the robot this far. The position budget
#: cannot be enforced only between trials if one trial can spend all of it. Sized for
#: block D, whose staircase deliberately travels: everything else stays under 0.15 m.
MAX_TRIAL_TRAVEL_M = 0.80

#: A REST trial is only a REST trial if the robot is actually still when the command goes
#: out. Verified, not assumed: 0.02 m/s is twice the ~0.010 m/s the null control measured
#: as estimator noise while standing.
REST_TOL_MPS = 0.02
REST_TIMEOUT_S = 3.0

#: A ROLLING segment is only rolling if the primer actually moved the robot.
ROLLING_MIN_MPS = 0.05

#: Classification. `MOVED_RATE_MPS` is net displacement over hold time, so it does not
#: depend on how long the hold was: 0.015 m/s sits ~35x above the null control's net rate
#: (0.001 m in 2.5 s) and ~2x below the slowest motion ever produced by a command (0.078 m
#: in 3.0 s). `TAIL_GAIT_MPS` splits a gait that KEPT GOING from the "shuffles a few steps
#: and stops" regime, using the estimator's own speed over the last part of the hold; it
#: sits between the standing noise floor (0.010) and the slowest sustained walk (0.049
#: whole-trial mean). Neither threshold is load-bearing — every sample is written to the
#: results file, so both can be moved afterwards without going near the robot again.
MOVED_RATE_MPS = 0.015
TAIL_FRACTION = 0.4
TAIL_GAIT_MPS = 0.04

#: Scheduled breaks: the robot is parked PRONE and the operator does something before the
#: block starts. Prone time costs no thermal budget at all, which is what makes these
#: nearly free — and each one removes a failure that ended a session on 2026-08-24.
#:
#: The D1 creeps out of its 3.0 deg stow gate roughly every 20 minutes and then refuses
#: the latch; it did so twice that day. Scheduling the re-pose converts a random
#: mid-session refusal into a planned one.
#:
#: The break before E is a POSITION reset, and block E is the reason block D is allowed to
#: travel. E repeats block B's cell at the session's peak temperature, and that comparison
#: is worthless if the two ran a metre apart — which is exactly the mistake this design
#: exists to remove.
PAUSE_BEFORE = {
    "D": "Re-pose the D1 arm, and confirm the lane AHEAD is clear: block D is the one\n"
         "block that deliberately travels, up to about 1.2 m forward and 0.4 m sideways.",
    "E": "Put the robot back on the tape, facing the same way, and re-pose the arm if\n"
         "sway has grown. Block E repeats block B's cell hot, and it has to repeat it in\n"
         "the same PLACE or it is measuring position again.",
}

LAT, FWD, VEC = "lat", "fwd", "vec"


def _steps(axis, cmd, hold_s):
    """One step, as ``(vx, vy, seconds)`` in the body frame.

    ``vy`` is always written POSITIVE here. Which way the robot strafes is chosen at
    issue time by :meth:`Session.sign_for`, because it depends on where the robot has
    drifted to — that is the whole fixed-position mechanism, and it cannot be planned.
    """
    if axis == FWD:
        return [(float(cmd), 0.0, float(hold_s))]
    if axis == LAT:
        return [(0.0, float(cmd), float(hold_s))]
    return [(0.0, 0.0, float(hold_s))]


def step_speed(step):
    return math.hypot(step[0], step[1])


def _test(block, state, axis, cmd, hold_s=HOLD_S, dither=0.0, note=""):
    return {"block": block, "kind": "test", "state": state, "axis": axis,
            "steps": _steps(axis, cmd, hold_s), "dither": dither, "note": note}


def _null(block, state="REST", note=""):
    return {"block": block, "kind": "null", "state": state, "axis": None,
            "steps": _steps(None, 0.0, HOLD_S), "dither": 0.0, "note": note}


def _anchor(block, note=""):
    return {"block": block, "kind": "anchor", "state": "FRESH", "axis": FWD,
            "steps": _steps(FWD, ANCHOR_MPS, ANCHOR_HOLD_S), "dither": 0.0, "note": note}


def _with_nulls(trials, block):
    """A zero-command null after every single commanded trial.

    The null is very nearly free, which is why there is no excuse for the version of this
    experiment that omitted it. A REST trial needs the robot to have stopped and settled
    before the next command anyway; commanding zero through that interval and reading the
    pose either side costs one extra pose read. What it buys is a noise measurement taken
    at the same position, the same motor temperature and the same battery state as the
    trial it controls — instead of one run once, at the end of the day, after the third
    reversal.
    """
    out = []
    for item in trials:
        out.append(item)
        out.append(_null(block, note=f"controls {item['note']}" if item["note"] else ""))
    return out


def plan_a():
    """Block A — is anything listening, and is the odometry telling the truth?"""
    return [
        _anchor("A", "opening anchor: walks the robot from the start line onto the mark"),
        _null("A", "REST", "negative control, interpretable now the anchor has passed"),
        _null("A", "FRESH", "refresh then ZERO: the settling control, in FRESH"),
    ]


def plan_b():
    """Block B — which call restores the state in which a low command starts a gait?

    The whole design turns on being able to RE-ENTER the state in which 0.100 forward and
    0.200 lateral walked on 2026-08-24, at an arbitrary point in the trial order. If
    `BalanceStand` alone restores it a refresh costs ~1 s and the grid can afford twelve of
    them; if only the full `stand_up()` sequence does, a refresh costs ~3 s and the grid
    shrinks. If NEITHER does — if the state can be entered only once per stand — then the
    state hypothesis is not testable inside a 20 minute window, and the session should say
    so and stop rather than spend its budget measuring something else.
    """
    trials = [
        _test("B", "FRESH", LAT, 0.150, note="FRESH via full stand_up()"),
        _test("B", "REST", LAT, 0.150, note="REST reference"),
        _test("B", "FRESH-BAL", LAT, 0.150, note="FRESH via BalanceStand only"),
        _test("B", "FRESH-BAL", LAT, 0.150, note="FRESH via BalanceStand only, repeat"),
    ]
    return _with_nulls(trials, "B")


def plan_c():
    """Block C — the lateral state grid. The primary result of the session.

    Lateral rather than forward because that is where the chain is stuck: the MAPPO
    policy's peak lateral command is 0.085-0.182 m/s in every disc-inflation
    configuration, three live runs issued v=(+0.000,-0.150) and stood still, and
    `_at_least_walking_pace` scales toward a floor no measurement supports. Lateral is also
    the only axis that can be held at a fixed position for free: `vy` may be signed, so
    alternating the sign makes the robot oscillate about the mark instead of walking away
    from it. `vx` may not be signed (no rear sensing), which is why the forward grid is
    optional and last.

    Two passes, with the state order AND the command order reversed in the second. Motor
    temperature, arm sway and battery charge all move monotonically through a session, and
    all three were perfectly confounded with "first trial after standing" in the
    2026-08-24 data. Reversing the order makes each of them cancel to first order instead
    of being argued about afterwards.
    """
    out = []
    for index in range(2):
        states = ("FRESH", "REST") if index == 0 else ("REST", "FRESH")
        commands = LATERAL_COMMANDS if index == 0 else list(reversed(LATERAL_COMMANDS))
        tag = f"pass {index + 1}"
        out.append(_null("C", "FRESH", f"refresh then ZERO: settling control, {tag}"))
        for cmd in commands:
            for state in states:
                out.append(_test("C", state, LAT, cmd, note=tag))
                out.append(_null("C", "REST", f"controls {cmd:.3f} {state} {tag}"))
    return out


def plan_d():
    """Block D — the SUSTAINING floor, which nothing has ever held long enough to see.

    A descending staircase issued as one continuous command stream. No segment after the
    first arrives at a stationary robot, so this is the ROLLING state the corridor
    telemetry answers well above 0.25 m/s and not at all below it. The last step is held
    for 6 s, the longest hold in the session, because the question is whether a gait DECAYS
    and a 2 s hold cannot see that.

    The cliff repeats that last step without the intermediate ones. A staircase that dies
    on its fourth step and a cliff that sustains at the same speed say the same thing: what
    ran out was time, not command. The cliff also takes the opposite sign, which walks the
    robot back toward the mark.
    """
    return [
        {"block": "D", "kind": "staircase", "state": "ROLLING", "axis": VEC,
         "steps": list(STAIRCASE_STEPS), "dither": 0.0,
         "note": "swerve staircase: forward primer swung into a pure strafe"},
        _null("D", "REST", "controls the staircase"),
        {"block": "D", "kind": "staircase", "state": "ROLLING", "axis": VEC,
         "steps": list(CLIFF_STEPS), "dither": 0.0,
         "note": "cliff: same last step, none of the intermediate ones"},
        _null("D", "REST", "controls the cliff"),
    ]


def plan_e():
    """Block E — the hot repeat and the closing anchor. Reserved, and never skipped.

    Blocks B and C run 0.150 lateral from a fresh stand at the START of the session, when
    the motors are coldest and the arm has crept least. E repeats the identical cell at the
    session's peak temperature. This is the cheapest discriminator between the two live
    explanations of the 2026-08-24 data, and it costs four trials:

      * it walks -> the state effect survives a session of heating and creep, and
        temperature is not the explanation;
      * it stalls -> "first trial after standing" was at least partly "coldest trial", the
        per-trial temperature column says how much, and the session has NOT settled the
        floor. Report that, rather than the grid.

    The closing anchor is the other half of the bracket. If it fails, every trial since the
    last passing anchor is unreadable.
    """
    trials = [
        _test("E", "FRESH", LAT, 0.150, note="hot repeat of B1"),
        _test("E", "REST", LAT, 0.150, note="hot repeat of the REST reference"),
    ]
    return [*_with_nulls(trials, "E"), _anchor("E", "closing anchor")]


def plan_f():
    """Block F — optional. Does a command that WOBBLES behave like a held one?"""
    trials = [
        _test("F", "REST", LAT, 0.150, note="held, the control for the dither"),
        _test("F", "REST", LAT, 0.150, dither=DITHER_MPS, note="dithered +-0.03 at 10 Hz"),
        _test("F", "FRESH", LAT, 0.150, dither=DITHER_MPS, note="dithered, from fresh"),
    ]
    return _with_nulls(trials, "F")


def plan_g():
    """Block G — optional. The forward grid, one pass, states alternated.

    Last because forward travel is one-way: `vx` may never be negative on this unit, so a
    forward success cannot be undone by the next trial and the position budget pays for
    every one of them.
    """
    out = []
    for index, cmd in enumerate(FORWARD_COMMANDS):
        states = ("FRESH", "REST") if index % 2 == 0 else ("REST", "FRESH")
        for state in states:
            out.append(_test("G", state, FWD, cmd, note="single pass"))
            out.append(_null("G", "REST", f"controls {cmd:.3f} {state}"))
    return out


#: Ordered, with the value decreasing down the list and the two protected blocks first and
#: last. A session that runs out of thermal budget loses the tail, not the answer.
BLOCK_PLANS = [("A", plan_a), ("B", plan_b), ("C", plan_c), ("D", plan_d),
               ("E", plan_e), ("F", plan_f), ("G", plan_g)]

#: Blocks dropped when the budget is short, in this order.
OPTIONAL_BLOCKS = ("G", "F")

#: Block E's cost is reserved before anything else is allowed to spend.
RESERVED_BLOCK = "E"


def build_plan(selected=None):
    """The full schedule, as an ordered list of trial dicts with ids assigned."""
    plan = []
    for name, builder in BLOCK_PLANS:
        if selected and name not in selected:
            continue
        for index, item in enumerate(builder()):
            copy = dict(item)
            copy["id"] = f"{name}{index + 1:02d}"
            plan.append(copy)
    return plan


def trial_seconds(item):
    """Standing seconds one trial costs: the holds, the settle, and the state entry."""
    hold = sum(step[2] for step in item["steps"])
    entry = 0.0
    if item["state"] == "FRESH":
        entry = 3.0
    elif item["state"] == "FRESH-BAL":
        entry = 1.0
    elif item["state"] == "REST":
        entry = 0.5
    return hold + SETTLE_S + entry


def plan_seconds(plan, block=None):
    return sum(trial_seconds(i) for i in plan if block is None or i["block"] == block)


def stand_budget_s(temp_c, rate_c_per_s=ASSUMED_HEAT_C_PER_S, ceiling_c=STAND_CEILING_C):
    """Seconds of standing the thermal ceiling allows from `temp_c`. Never negative."""
    if rate_c_per_s <= 0.0:
        return float("inf")
    return max(0.0, (ceiling_c - temp_c) / rate_c_per_s)


def fit_blocks(plan, budget_s):
    """Which blocks fit, dropping the optional tail first. Returns (kept, dropped)."""
    kept = [name for name, _ in BLOCK_PLANS if any(i["block"] == name for i in plan)]
    dropped = []
    for name in OPTIONAL_BLOCKS:
        if name not in kept:
            continue
        if sum(plan_seconds(plan, b) for b in kept) <= budget_s:
            break
        kept.remove(name)
        dropped.append(name)
    return kept, dropped


def budget_shortfall(plan, kept, budget_s):
    """Seconds by which the REQUIRED blocks overrun the thermal budget, or 0.0.

    `fit_blocks` can only drop the optional tail, so a warm robot leaves a plan that
    still does not fit — and the run would then discover that halfway through block D,
    having already spent the budget on C. Saying it up front is the difference between
    waiting twenty minutes for the motors to cool and getting a truncated session.
    """
    return max(0.0, sum(plan_seconds(plan, name) for name in kept) - budget_s)


def classify(samples, steps, null_ceiling_m):
    """Per-segment outcome for one trial. Pure: it takes the recorded trace, nothing else.

    `net_m` is straight-line displacement and `path_m` the integrated path, and both are
    reported because they disagree in the case that matters: a robot that steps out and
    rocks back has a small `net_m` and a large `path_m`, and calling that a stall is how
    "it did not move" gets written down for something that plainly did.
    """
    out = []
    for index, step in enumerate(steps):
        cmd = round(step_speed(step), 4)
        block = [s for s in samples if s["seg"] == index]
        if len(block) < 2:
            out.append({"seg": index, "cmd": cmd, "cmd_vx": step[0], "cmd_vy": step[1],
                        "samples": len(block), "class": "no-data"})
            continue
        net = math.hypot(block[-1]["x"] - block[0]["x"], block[-1]["y"] - block[0]["y"])
        path = 0.0
        for prev, cur in zip(block, block[1:]):
            path += math.hypot(cur["x"] - prev["x"], cur["y"] - prev["y"])
        span = max(block[-1]["t"] - block[0]["t"], 1e-6)
        tail_from = block[-1]["t"] - TAIL_FRACTION * span
        tail = [math.hypot(s["mvx"], s["mvy"]) for s in block if s["t"] >= tail_from]
        tail_mps = sum(tail) / len(tail) if tail else 0.0
        peak = max(math.hypot(s["mvx"], s["mvy"]) for s in block)
        moved = net > max(null_ceiling_m, MOVED_RATE_MPS * span)
        if not moved:
            verdict = "no-gait"
        elif tail_mps >= TAIL_GAIT_MPS:
            verdict = "gait"
        else:
            verdict = "shuffle"
        out.append({"seg": index, "cmd": cmd, "cmd_vx": step[0], "cmd_vy": step[1],
                    "samples": len(block),
                    "seconds": round(span, 2), "net_m": round(net, 4),
                    "path_m": round(path, 4), "tail_mps": round(tail_mps, 4),
                    "peak_mps": round(peak, 4), "class": verdict})
    return out


def walked(record):
    """True if any segment of the trial produced motion — a gait or a shuffle."""
    return any(s.get("class") in ("gait", "shuffle") for s in record.get("segments", []))


def anchor_ok(record):
    """Whether a positive control actually walked the robot.

    Judged by its own 0.15 m rule and NOT by :func:`walked`, which asks a much weaker
    question. A robot leaning into a wall can produce a shuffle; the anchor exists to say
    the legs are executing a command that has walked 0.5-0.7 m in every session, so
    anything short of a third of the expected travel is a failure. Two rules for one
    verdict is how a refusal and a summary end up disagreeing about the same trial.
    """
    segments = record.get("segments") or [{}]
    return segments[0].get("net_m", 0.0) >= ANCHOR_MIN_TRAVEL_M


def monotonicity_violations(records):
    """Cells where a BIGGER command produced motion LESS often than a smaller one.

    Confound 4 of 2026-08-24, made mechanical. It appeared four separate times that day —
    0.100 walking while 0.137 stalled, 45 degrees walking twice and stalling once above the
    bearing that had already failed — and every time it was an uncontrolled variable
    leaking through, not a property of the gait. A bigger command working less often is the
    signature of something the design did not hold still, and the correct response is to
    re-run the block, never to write down a floor.
    """
    rates = {}
    for record in records:
        if record.get("kind") != "test" or record.get("axis") is None:
            continue
        if not record.get("state_verified", True):
            continue
        key = (record["axis"], record["state"], round(step_speed(record["steps"][0]), 4))
        hit, total = rates.get(key, (0, 0))
        rates[key] = (hit + (1 if walked(record) else 0), total + 1)
    out = []
    for axis, state in sorted({(a, s) for a, s, _ in rates}):
        levels = sorted(cmd for a, s, cmd in rates if (a, s) == (axis, state))
        for low, high in zip(levels, levels[1:]):
            lo_hit, lo_n = rates[(axis, state, low)]
            hi_hit, hi_n = rates[(axis, state, high)]
            if lo_n and hi_n and lo_hit / lo_n > hi_hit / hi_n:
                out.append({"axis": axis, "state": state, "lower": low, "higher": high,
                            "lower_rate": f"{lo_hit}/{lo_n}",
                            "higher_rate": f"{hi_hit}/{hi_n}"})
    return out


def rate_table(records, axis):
    """``{(command, state): [moved, trials]}`` for the end-of-session print."""
    table = {}
    for record in records:
        if record.get("kind") != "test" or record.get("axis") != axis:
            continue
        if not record.get("state_verified", True):
            continue
        cell = table.setdefault((round(step_speed(record["steps"][0]), 4),
                                 record["state"]), [0, 0])
        cell[0] += 1 if walked(record) else 0
        cell[1] += 1
    return table


# ── SWEEP P runner ───────────────────────────────────────────────────────────

class Aborted(Exception):
    """Raised to unwind to the parking `finally` without a traceback."""


class Session:
    """Runs one protocol plan. Owns the mark, the budgets and the results file."""

    def __init__(self, loco, health, arm, args, plan, helpers):
        self.loco = loco
        self.health = health
        self.arm = arm
        self.args = args
        self.plan = plan
        #: ``stand_up``, ``lie_down`` and ``latch_arm``, injected so that the pure logic
        #: above stays importable off the robot.
        self.helpers = helpers
        self.records = []
        self.mark = None
        self.mark_yaw = 0.0
        self.stand_s = 0.0
        self.start_temp_c = None
        self.last_sign = 1.0
        self.refresh = args.refresh
        self.null_ceiling_m = 0.01
        self.stop_flag = False
        #: Trials are written as they finish, not at the end. A session killed by a
        #: dropped SSH link or a second Ctrl-C keeps everything it had measured; the
        #: handle is closed by `run_protocol`'s `finally`.
        self.handle = open(args.out, "w")  # noqa: SIM115

    # -- helpers ---------------------------------------------------------
    def emit(self, record):
        self.records.append(record)
        self.handle.write(json.dumps(record) + "\n")
        self.handle.flush()

    def say(self, text):
        print(text, flush=True)

    def health_now(self):
        latest = self.health.latest() if self.health is not None else None
        if latest is None:
            return {"temp_c": None, "hottest_motor": None, "soc_pct": None}
        return {"temp_c": round(latest.max_motor_temp_c, 1),
                "hottest_motor": latest.hottest_motor,
                "soc_pct": round(latest.battery_soc_pct, 1)}

    def arm_now(self):
        sway = self.arm.sway_deg() if self.arm is not None else None
        reach = self.arm.reach_m() if self.arm is not None else None
        return {"arm_sway_deg": None if sway is None else round(sway, 2),
                "arm_reach_m": None if reach is None else round(reach, 3)}

    def offsets(self):
        """``(along, across)`` metres from the mark, in the mark's own frame."""
        if self.mark is None:
            return 0.0, 0.0
        pose = self.loco.pose()
        dx, dy = pose.x - self.mark[0], pose.y - self.mark[1]
        cos_y, sin_y = math.cos(self.mark_yaw), math.sin(self.mark_yaw)
        return cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy

    def set_mark(self):
        pose = self.loco.pose()
        self.mark = (pose.x, pose.y)
        self.mark_yaw = pose.yaw
        self.say(f"mark set at odom ({pose.x:.3f}, {pose.y:.3f}) yaw {pose.yaw:.3f} rad")

    def refuse(self, reason):
        self.say("\nREFUSING: " + reason)
        raise Aborted(reason)

    def remaining_blocks(self):
        done = {r["block"] for r in self.records}
        names = [n for n, _ in BLOCK_PLANS if any(i["block"] == n for i in self.plan)]
        pending = [n for n in names if n not in done]
        return ",".join(pending) if pending else RESERVED_BLOCK

    # -- gates -----------------------------------------------------------
    def check_gates(self):
        """Everything that must be true before another command goes out."""
        if self.stop_flag:
            self.refuse("interrupted by the operator")
        reason = self.health.abort_reason() if self.health is not None else None
        if reason is not None:
            self.refuse(f"the stack's own health gate says: {reason}")
        latest = self.health.latest() if self.health is not None else None
        if latest is not None and latest.max_motor_temp_c >= STAND_CEILING_C:
            self.refuse(
                f"motor {latest.hottest_motor} at {latest.max_motor_temp_c:.1f}C is at "
                f"the self-imposed\n  {STAND_CEILING_C:.0f}C stand ceiling. Park it prone "
                f"and stop. Cooling while prone is real\n  but slow, so the remaining "
                f"blocks belong to a second session, not to this one.")
        blocking = self.arm.blocking_reason() if self.arm is not None else None
        if blocking is not None:
            self.refuse(
                f"the D1 arm has left its stow gate: {blocking}\n"
                f"  The arm is a 3.15 kg lever over the hind legs, and an off-centre arm\n"
                f"  unbalances the vendor gait controller — so every trial after this "
                f"point\n  would be measuring the arm and not the floor. Hand-pose it "
                f"flat along the\n  spine, then resume with:\n"
                f"    --sweep p --blocks {self.remaining_blocks()} "
                f"--refresh {self.refresh} --out <a NEW file>")
        along, across = self.offsets()
        drift = math.hypot(along, across)
        if drift > POSITION_BUDGET_M:
            self.refuse(
                f"the robot is {drift:.2f} m from its mark (budget {POSITION_BUDGET_M:.2f}"
                f" m; along {along:+.2f}, across {across:+.2f}).\n"
                f"  Position is the confound this design exists to remove, so no trial "
                f"runs from\n  here. Walk it back onto the tape and resume with "
                f"--blocks {self.remaining_blocks()}.")

    # -- state entry -----------------------------------------------------
    def enter_state(self, item):
        """Put the robot into the trial's state, and say whether it verifiably got there."""
        state = item["state"]
        if state == "FRESH":
            # Block B is the experiment that DECIDES which call restores this state, so it
            # is never subject to the answer it is producing.
            if self.refresh == "balance" and item["block"] != "B":
                self.loco.stand()
                time.sleep(1.0)
                self.stand_s += 1.0
                return True, "BalanceStand() (block B said that is enough)"
            self.loco.recover()
            time.sleep(2.0)
            self.loco.stand()
            time.sleep(1.0)
            self.stand_s += 3.0
            return True, "stand_up()"
        if state == "FRESH-BAL":
            self.loco.stand()
            time.sleep(1.0)
            self.stand_s += 1.0
            return True, "BalanceStand()"
        if state == "REST":
            deadline = time.time() + REST_TIMEOUT_S
            still = 0
            while time.time() < deadline:
                if math.hypot(*self.loco.velocity()[:2]) < REST_TOL_MPS:
                    still += 1
                    if still >= 3:
                        return True, "measured still"
                else:
                    still = 0
                time.sleep(1.0 / REFRESH_HZ)
            return False, (f"never settled below {REST_TOL_MPS:.3f} m/s in "
                           f"{REST_TIMEOUT_S:.1f}s")
        return True, state

    def sign_for(self, axis):
        """Which way to strafe. Homeward when the robot has drifted, alternating otherwise.

        This is the whole fixed-position mechanism on the lateral axis. A stall does not
        move the robot at all, so the trials that matter most cost nothing in floor space;
        a success moves it ~0.13 m, and the next trial takes it back.
        """
        if axis not in (LAT, VEC):
            return 1.0
        _, across = self.offsets()
        if abs(across) > HOMEWARD_M:
            return -1.0 if across > 0 else 1.0
        self.last_sign = -self.last_sign
        return self.last_sign

    # -- driving ---------------------------------------------------------
    def drive(self, item, sign):
        """Hold each segment in turn and return the sampled trace.

        The command is re-issued every tick because `Move` has no dead-man: a segment that
        simply stopped writing would leave the last velocity latched on the robot.
        """
        samples = []
        start = self.loco.pose()
        t0 = time.time()
        truncated = None
        for index, (step_vx, step_vy, seconds) in enumerate(item["steps"]):
            magnitude = math.hypot(step_vx, step_vy)
            deadline = time.time() + seconds
            tick = 0
            while time.time() < deadline:
                # The dither scales the whole vector, so it changes the SPEED and never
                # the bearing: a wobble that moved the direction would be testing
                # something else. `sign` only ever touches vy — vx is never negative,
                # because there is no rear-facing sensing on this unit.
                wobble = item["dither"] * (1.0 if tick % 2 == 0 else -1.0)
                scale = max(0.0, magnitude + wobble) / magnitude if magnitude else 0.0
                vx, vy = max(0.0, step_vx * scale), sign * step_vy * scale
                self.loco.set_velocity(vx, vy, 0.0)
                pose = self.loco.pose()
                mvx, mvy, _ = self.loco.velocity()
                samples.append({"t": round(time.time() - t0, 3), "seg": index,
                                "cvx": round(vx, 4), "cvy": round(vy, 4),
                                "x": round(pose.x, 4), "y": round(pose.y, 4),
                                "yaw": round(pose.yaw, 4),
                                "mvx": round(mvx, 4), "mvy": round(mvy, 4)})
                if math.hypot(pose.x - start.x, pose.y - start.y) > MAX_TRIAL_TRAVEL_M:
                    truncated = f"travelled past {MAX_TRIAL_TRAVEL_M:.2f} m"
                    break
                if self.stop_flag:
                    truncated = "operator interrupt"
                    break
                tick += 1
                time.sleep(1.0 / REFRESH_HZ)
            if truncated:
                break
        self.loco.stop()
        time.sleep(SETTLE_S)
        self.stand_s += time.time() - t0
        return samples, truncated

    def run_trial(self, item):
        self.check_gates()
        state_ok, how = self.enter_state(item)
        sign = self.sign_for(item["axis"])
        along_before, across_before = self.offsets()
        samples, truncated = self.drive(item, sign)
        if item["state"] == "ROLLING":
            primer = [math.hypot(s["mvx"], s["mvy"]) for s in samples if s["seg"] == 0]
            peak = max(primer) if primer else 0.0
            state_ok = peak >= ROLLING_MIN_MPS
            how = f"primer peaked at {peak:.3f} m/s"
        record = {"id": item["id"], "block": item["block"], "kind": item["kind"],
                  "state": item["state"], "axis": item["axis"], "sign": sign,
                  "steps": item["steps"], "dither": item["dither"], "note": item["note"],
                  "state_verified": state_ok, "state_how": how, "truncated": truncated,
                  "segments": classify(samples, item["steps"], self.null_ceiling_m),
                  "along_m": round(along_before, 3), "across_m": round(across_before, 3),
                  "stand_s": round(self.stand_s, 1), "wall": round(time.time(), 3),
                  "samples": samples}
        record.update(self.health_now())
        record.update(self.arm_now())
        self.emit(record)
        self.report_trial(record)
        if item["kind"] == "null":
            self.retune_null_ceiling()
        if item["kind"] == "staircase":
            # Block D is the one block that does not hold position, and it cannot: the
            # ROLLING state is DEFINED by the robot already moving, and its question is
            # duration rather than place. Re-marking keeps the position gate meaningful
            # for the trial after it, and the break before block E puts the robot back on
            # the tape for the comparison that does need position held.
            self.set_mark()
        return record

    def retune_null_ceiling(self):
        """The classifier's floor is THIS session's nulls, not a number from another day.

        Four times the worst null measured so far, with 0.01 m as a backstop so a run of
        unusually quiet nulls cannot make the threshold absurdly tight.
        """
        nets = [s["net_m"] for r in self.records if r["kind"] == "null"
                for s in r["segments"] if "net_m" in s]
        if nets:
            self.null_ceiling_m = max(0.01, 4.0 * max(nets))

    def report_trial(self, record):
        head = record["segments"][0] if record["segments"] else {}
        flags = ""
        if not record["state_verified"]:
            flags += "  STATE-UNVERIFIED"
        if record["truncated"]:
            flags += f"  TRUNCATED({record['truncated']})"
        temp = record["temp_c"]
        self.say(f"{record['id']:<5} {record['state']:<10} {record['axis'] or '-':<4} "
                 f"{step_speed(record['steps'][0]):6.3f}  "
                 f"net {head.get('net_m', 0.0):6.3f} m  "
                 f"tail {head.get('tail_mps', 0.0):5.3f} m/s  "
                 f"{head.get('class', '-'):<8}  "
                 f"{'--' if temp is None else format(temp, '.1f')}C  "
                 f"sway {record['arm_sway_deg']}{flags}")
        for extra in record["segments"][1:]:
            self.say(f"      step {extra['cmd']:.3f} for {extra.get('seconds', 0.0):.1f}s"
                     f"  net {extra.get('net_m', 0.0):6.3f} m"
                     f"  tail {extra.get('tail_mps', 0.0):5.3f} m/s"
                     f"  {extra.get('class', '-')}")

    # -- blocks ----------------------------------------------------------
    def run_block(self, name, items, reserve_s):
        cost = sum(trial_seconds(i) for i in items)
        latest = self.health.latest() if self.health is not None else None
        if latest is not None and name != RESERVED_BLOCK:
            rate = self.heat_rate()
            left = stand_budget_s(latest.max_motor_temp_c, rate)
            if cost + reserve_s > left:
                self.say(f"\nSKIPPING BLOCK {name}: it needs {cost:.0f}s of standing, "
                         f"{reserve_s:.0f}s is reserved\nfor block {RESERVED_BLOCK}, and "
                         f"{STAND_CEILING_C - latest.max_motor_temp_c:.1f}C of headroom at "
                         f"{rate:.3f} C/s leaves {left:.0f}s.")
                return False
        if name in PAUSE_BEFORE and not self.args.no_pause:
            self.scheduled_break(name)
        self.say(f"\n=== BLOCK {name} — {len(items)} trials, ~{cost:.0f}s of standing ===")
        for item in items:
            self.run_trial(item)
        self.block_verdict(name)
        return True

    def scheduled_break(self, name):
        """Park prone, let the operator act, re-latch, and stand again.

        Prone time is thermally free, so a break costs nothing but wall clock — and each
        one removes a failure that ended a session on 2026-08-24.
        """
        self.say(f"\n--- SCHEDULED BREAK before block {name} ---")
        self.loco.stop()
        self.helpers["lie_down"](self.loco)
        self.say(f"The robot is PRONE. {PAUSE_BEFORE[name]}\n"
                 f"Arm sway is {self.arm_now()['arm_sway_deg']} deg; the gate is 3.0 deg. "
                 f"Support the arm's\nweight when you move it, and never lift the robot "
                 f"by the arm.")
        try:
            input("Press Enter when that is done — or Ctrl-C then Enter to stop here: ")
        except (EOFError, KeyboardInterrupt):
            self.refuse("the operator stopped at the scheduled break")
        # A SIGINT during `input()` runs the handler and, under PEP 475, the read is
        # RETRIED rather than raising — so a Ctrl-C at a break would otherwise be
        # swallowed and the robot would stand back up. The handler sets the flag; this is
        # where it is honoured.
        if self.stop_flag:
            self.refuse("interrupted by the operator during the scheduled break")
        latch = self.helpers["latch_arm"](self.arm, iface="eth0")
        self.say(str(latch))
        if not latch.held:
            self.refuse("the D1 latch did not take after the break")
        self.helpers["stand_up"](self.loco)
        self.stand_s += 3.0
        self.set_mark()

    def block_verdict(self, name):
        mine = [r for r in self.records if r["block"] == name]
        if name == "A":
            self.verdict_a(mine)
        if name == "B":
            self.branch_on_refresh(mine)
        if name == "E":
            anchor = next((r for r in mine if r["kind"] == "anchor"), None)
            if anchor is not None and not anchor_ok(anchor):
                self.say("\n*** THE CLOSING ANCHOR FAILED. Every trial since the opening\n"
                         "    anchor is suspect: the legs may have stopped listening at "
                         "any point.\n    Do not report this session's grid.")
        violations = monotonicity_violations(mine)
        if violations:
            self.say(f"\n*** NON-MONOTONIC IN BLOCK {name} — a confound, not a finding.")
            for bad in violations:
                self.say(f"    {bad['axis']} {bad['state']}: {bad['lower']} moved "
                         f"{bad['lower_rate']} but {bad['higher']} moved "
                         f"{bad['higher_rate']}")
            self.say("    Do NOT read a floor off this block. Something the design did not"
                     " hold\n    still moved. Re-run the block if there is budget; "
                     "otherwise report it\n    as unresolved.")

    def verdict_a(self, mine):
        anchor = next((r for r in mine if r["kind"] == "anchor"), None)
        net = (anchor["segments"] or [{}])[0].get("net_m", 0.0) if anchor else 0.0
        if anchor is not None and not anchor_ok(anchor):
            self.refuse(
                f"THE OPENING ANCHOR DID NOT MOVE THE ROBOT.\n"
                f"  {ANCHOR_MPS:.2f} m/s for {ANCHOR_HOLD_S:.1f}s travelled {net:.3f} m, "
                f"and anything under\n  {ANCHOR_MIN_TRAVEL_M:.2f} m means the legs are not "
                f"listening. Check, in this order: the\n  sport mode, the arm latch, "
                f"whether the robot is against something, and the\n  battery. A null "
                f"control passes PERFECTLY on a robot that cannot move, so\n  nothing "
                f"measured after this point would mean anything.")
        self.set_mark()

    def branch_on_refresh(self, mine):
        """Block B decides how the rest of the session enters the FRESH state."""
        tests = [r for r in mine if r["kind"] == "test"]
        full = [r for r in tests if r["state"] == "FRESH"]
        bal = [r for r in tests if r["state"] == "FRESH-BAL"]
        rest = [r for r in tests if r["state"] == "REST"]
        full_ok = any(walked(r) for r in full)
        bal_ok = sum(1 for r in bal if walked(r))
        rest_ok = any(walked(r) for r in rest)
        self.say(f"\nBLOCK B VERDICT: stand_up() "
                 f"{'moved' if full_ok else 'did NOT move'} the robot, BalanceStand-only "
                 f"{bal_ok} of {len(bal)}, REST {'moved' if rest_ok else 'did NOT move'}")
        if self.args.refresh != "auto":
            self.say(f"  --refresh {self.args.refresh} was given, so the branch is not "
                     f"applied.")
            return
        if bal and bal_ok == len(bal):
            self.refresh = "balance"
            self.say("  BalanceStand alone restores it. FRESH costs 1s and the grid runs "
                     "in full.\n  Worth carrying downstream: if RE-ARMING the gait is what "
                     "lets a low command\n  start, then scaling the command up is not the "
                     "only fix available to\n  _at_least_walking_pace.")
        elif full_ok:
            self.refresh = "full"
            self.say("  Only the full stand_up() restores it. FRESH costs 3s, so the "
                     "optional\n  blocks will probably be dropped for budget.")
        elif not rest_ok:
            self.refuse(
                "0.150 lateral did not move the robot in ANY state, a fresh stand "
                "included.\n  Either the lateral floor is above 0.150 everywhere — which "
                "is a real result,\n  and blocks C and D will bracket it — or the staging "
                "is wrong. The block A\n  anchor decides which: if it walked, this is a "
                "result. Resume with\n  --blocks C,D,E --refresh full.")
        else:
            self.refresh = "full"
            self.say("  Inconclusive; falling back to the full stand_up() refresh.")

    def heat_rate(self):
        """C per standing second, measured from this session where it can be, else assumed."""
        latest = self.health.latest() if self.health is not None else None
        if latest is None or self.start_temp_c is None or self.stand_s < 30.0:
            return ASSUMED_HEAT_C_PER_S
        risen = latest.max_motor_temp_c - self.start_temp_c
        if risen < 0.5:
            return ASSUMED_HEAT_C_PER_S
        return risen / self.stand_s

    # -- summary ---------------------------------------------------------
    def summarise(self):
        self.say("\n=== SESSION SUMMARY ===")
        nulls = [s["net_m"] for r in self.records if r["kind"] == "null"
                 for s in r["segments"] if "net_m" in s]
        if nulls:
            self.say(f"null control: {len(nulls)} trials, worst net {max(nulls):.4f} m, "
                     f"mean {sum(nulls) / len(nulls):.4f} m")
        for anchor in [r for r in self.records if r["kind"] == "anchor"]:
            verdict = "LEGS LISTENING" if anchor_ok(anchor) else "DEAD"
            self.say(f"anchor {anchor['id']}: net "
                     f"{(anchor['segments'] or [{}])[0].get('net_m', 0.0):.3f} m "
                     f"(needs {ANCHOR_MIN_TRAVEL_M:.2f}) -> {verdict}")
        for axis, label in ((LAT, "LATERAL"), (FWD, "FORWARD")):
            self.say_table(axis, label)
        violations = monotonicity_violations(self.records)
        if violations:
            self.say("\nNON-MONOTONIC CELLS REMAIN — do not report a floor from this "
                     "session:")
            for bad in violations:
                self.say(f"  {bad['axis']} {bad['state']}: {bad['lower']} "
                         f"{bad['lower_rate']} vs {bad['higher']} {bad['higher_rate']}")
        latest = self.health.latest() if self.health is not None else None
        if latest is not None and self.start_temp_c is not None:
            self.say(f"\nmotors {self.start_temp_c:.1f}C -> {latest.max_motor_temp_c:.1f}C "
                     f"over {self.stand_s:.0f}s of standing ({self.heat_rate():.3f} C/s)")
        self.say(f"results -> {self.args.out}")

    def say_table(self, axis, label):
        table = rate_table(self.records, axis)
        if not table:
            return
        states = sorted({state for _, state in table})
        self.say(f"\n{label} — trials that produced motion, by command and state")
        self.say("      cmd  " + "".join(f"{s:<14}" for s in states))
        for cmd in sorted({c for c, _ in table}):
            row = f"  {cmd:7.3f}  "
            for state in states:
                hit, total = table.get((cmd, state), (0, 0))
                row += f"{f'{hit}/{total}' if total else '-':<14}"
            self.say(row)


def print_plan(plan, temp_c):
    """The whole schedule and its arithmetic, without touching the robot."""
    total = plan_seconds(plan)
    print(f"SWEEP P — {len(plan)} trials, ~{total:.0f}s of standing")
    print(f"{'id':<6} {'block':<5} {'kind':<10} {'state':<10} {'axis':<5} "
          f"{'vx,vy per step':<44} note")
    for item in plan:
        commands = "/".join(f"{vx:.2f},{vy:.2f}" for vx, vy, _ in item["steps"])
        print(f"{item['id']:<6} {item['block']:<5} {item['kind']:<10} "
              f"{item['state']:<10} {item['axis'] or '-':<5} {commands:<44} "
              f"{item['note']}")
    print("\nper block:")
    for name, _ in BLOCK_PLANS:
        seconds = plan_seconds(plan, name)
        if seconds:
            print(f"  {name}  {seconds:5.0f}s  "
                  f"{seconds * ASSUMED_HEAT_C_PER_S:+.1f} C at {ASSUMED_HEAT_C_PER_S:.3f}"
                  f" C/s")
    print(f"\nreserved for block {RESERVED_BLOCK}: "
          f"{plan_seconds(plan, RESERVED_BLOCK):.0f}s")
    if temp_c is None:
        coldest = STAND_CEILING_C - total * ASSUMED_HEAT_C_PER_S
        print(f"Pass --start-temp to see which blocks fit. The whole plan needs a start\n"
              f"temperature at or below {coldest:.0f} C.")
        return
    budget = stand_budget_s(temp_c)
    kept, dropped = fit_blocks(plan, budget)
    print(f"at {temp_c:.1f} C the {STAND_CEILING_C:.0f} C ceiling allows {budget:.0f}s "
          f"of standing")
    print(f"  blocks that fit: {','.join(kept)}")
    if dropped:
        print(f"  dropped for budget: {','.join(dropped)}")
    short = budget_shortfall(plan, kept, budget)
    if short:
        print(f"  *** STILL {short:.0f}s OVER. The required blocks A-E cannot all run at "
              f"{temp_c:.1f} C.\n      Let it cool: every 1 C below that buys "
              f"{1.0 / ASSUMED_HEAT_C_PER_S:.0f}s. The run will stop where the\n      "
              f"budget stops, and the block order means A, B and C go first.")


def run_protocol(args):
    """Sweep P. The robot modules are imported here so `--plan` never needs them."""
    from safety import (
        MOTOR_TEMP_WARN_C,
        ArmStowMonitor,
        HealthMonitor,
        latch_arm,
        lie_down,
        stand_up,
    )
    from unitree.go2.locomotion.go2_locomotion import Go2Locomotion

    plan = build_plan([b.strip() for b in args.blocks.split(",")] if args.blocks else None)
    if not plan:
        raise SystemExit("no blocks selected")

    loco = Go2Locomotion(iface="eth0")
    loco.connect()
    health = HealthMonitor()
    health.start()
    before = health.latest()
    if before is None:
        raise SystemExit("REFUSING: no rt/lowstate — a session cannot be budgeted blind")
    print(f"motors {before.max_motor_temp_c:.1f}C (ceiling {STAND_CEILING_C:.0f}C, warn "
          f"{MOTOR_TEMP_WARN_C:.0f}C)", flush=True)
    if before.max_motor_temp_c >= STAND_CEILING_C:
        raise SystemExit("REFUSING: motors at or above the ceiling — let it cool")

    budget = stand_budget_s(before.max_motor_temp_c)
    kept, dropped = fit_blocks(plan, budget)
    print(f"thermal budget {budget:.0f}s of standing; the plan needs "
          f"{plan_seconds(plan):.0f}s", flush=True)
    if dropped:
        print(f"dropping for budget: {','.join(dropped)}", flush=True)
    short = budget_shortfall(plan, kept, budget)
    if short:
        print(f"*** WARNING: the required blocks still overrun by {short:.0f}s. This "
              f"session will\n    stop where the budget stops. A, B and C run first, so "
              f"the primary result is\n    banked before D and E are at risk. Cooling "
              f"first buys {1.0 / ASSUMED_HEAT_C_PER_S:.0f}s per degree.", flush=True)
    plan = [i for i in plan if i["block"] in kept]

    arm = ArmStowMonitor()
    arm.start()
    latch = latch_arm(arm, iface="eth0")
    if not latch.held:
        raise SystemExit("REFUSING: the D1 latch did not take — hand-pose it flat")
    print(f"latch drift {latch.drift_deg:.2f} deg — HELD", flush=True)

    helpers = {"lie_down": lie_down, "stand_up": stand_up, "latch_arm": latch_arm}
    session = Session(loco, health, arm, args, plan, helpers)
    session.start_temp_c = before.max_motor_temp_c

    def on_signal(_signum, _frame):
        # Move persists until the next command, so get the zero out before unwinding —
        # a Ctrl-C that only raises leaves the last velocity latched on the robot.
        session.stop_flag = True
        with contextlib.suppress(Exception):
            loco.stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    reserve = plan_seconds(plan, RESERVED_BLOCK)
    try:
        print(f"mode now: {loco.ensure_sport_mode('normal')!r}", flush=True)
        stand_up(loco)
        session.stand_s += 3.0
        session.set_mark()
        print("standing", flush=True)
        for name in kept:
            items = [i for i in plan if i["block"] == name]
            if items:
                session.run_block(name, items, 0.0 if name == RESERVED_BLOCK else reserve)
    except Aborted:
        pass
    finally:
        loco.stop()                      # Move persists; this is not optional
        lie_down(loco)
        session.summarise()
        session.handle.close()
        print(f"PRONE. results -> {args.out}", flush=True)


def run_sweeps(args):
    """Sweeps A-D and Z, exactly as they ran on 2026-08-24."""
    from safety import (
        MOTOR_TEMP_WARN_C,
        ArmStowMonitor,
        HealthMonitor,
        latch_arm,
        lie_down,
        stand_up,
    )
    from unitree.go2.locomotion.go2_locomotion import Go2Locomotion

    loco = Go2Locomotion(iface="eth0")
    loco.connect()
    health = HealthMonitor()
    health.start()
    before = health.latest()
    if before is not None:
        print(f"motors {before.max_motor_temp_c:.1f}C (ceiling {STAND_CEILING_C:.0f}C, "
              f"warn {MOTOR_TEMP_WARN_C:.0f}C)", flush=True)
        if before.max_motor_temp_c >= STAND_CEILING_C and not args.dry:
            raise SystemExit("REFUSING: motors at or above the ceiling — let it cool")

    arm = ArmStowMonitor()
    arm.start()
    latch = latch_arm(arm, iface="eth0")
    if not latch.held:
        raise SystemExit("REFUSING: the D1 latch did not take — hand-pose it flat")
    print(f"latch drift {latch.drift_deg:.2f} deg — HELD", flush=True)

    results = {"dry": args.dry, "hold_s": args.hold, "a": [], "b": []}
    try:
        if not args.dry:
            # Selecting a mode can itself make the robot shift — done deliberately,
            # once, with the area confirmed clear, and before anything is commanded.
            print(f"mode now: {loco.ensure_sport_mode('normal')!r}", flush=True)
        stand_up(loco)
        print("standing", flush=True)

        if args.sweep in ("a", "both"):
            print("\nSWEEP A — sustained forward speed", flush=True)
            print(f"{'cmd':>8} {'travel_m':>10} {'est_mps':>10} {'walked':>8}", flush=True)
            for v in SPEEDS:
                travel, est = trial(loco, v, 0.0, args.hold, args.dry)
                is_walk = travel > 0.05
                results["a"].append({"cmd_vx": v, "travel_m": travel, "est_mps": est,
                                     "walked": is_walk})
                print(f"{v:8.3f} {travel:10.3f} {est:10.3f} {is_walk!s:>8}", flush=True)

        if args.sweep in ("b", "both"):
            print("\nSWEEP B — ellipse floor at each bearing", flush=True)
            print(f"{'deg':>8} {'speed':>8} {'vx':>8} {'travel_m':>10} {'walked':>8}",
                  flush=True)
            for deg in BEARINGS_DEG:
                th = math.radians(deg)
                speed = ellipse_floor(th)
                vx, vy = speed * math.cos(th), speed * math.sin(th)
                travel, est = trial(loco, vx, vy, args.hold, args.dry)
                is_walk = travel > 0.05
                results["b"].append({"bearing_deg": deg, "speed": speed, "vx": vx,
                                     "vy": vy, "travel_m": travel, "est_mps": est,
                                     "walked": is_walk})
                print(f"{deg:8.1f} {speed:8.3f} {vx:8.3f} {travel:10.3f} {is_walk!s:>8}",
                      flush=True)
        if args.sweep == "c":
            print("\nSWEEP C — raise the speed until it walks, at the failing bearings",
                  flush=True)
            print(f"{'deg':>8} {'speed':>8} {'vx':>8} {'vy':>8} {'travel_m':>10} "
                  f"{'walked':>8}", flush=True)
            results["c"] = []
            for deg in SWEEP_C_BEARINGS_DEG:
                th = math.radians(deg)
                for speed in SWEEP_C_SPEEDS:
                    vx, vy = speed * math.cos(th), speed * math.sin(th)
                    travel, est = trial(loco, vx, vy, args.hold, args.dry)
                    is_walk = travel > 0.05
                    results["c"].append({"bearing_deg": deg, "speed": speed, "vx": vx,
                                         "vy": vy, "travel_m": travel, "est_mps": est,
                                         "walked": is_walk})
                    print(f"{deg:8.1f} {speed:8.3f} {vx:8.3f} {vy:8.3f} {travel:10.3f} "
                          f"{is_walk!s:>8}", flush=True)
                    if is_walk:
                        print(f"         ^ floor at {deg:.1f} deg is at or below "
                              f"{speed:.2f} m/s", flush=True)
                        break
        if args.sweep == "d":
            print("\nSWEEP D — repeatability of the two low forward speeds", flush=True)
            print(f"{'cmd':>8} {'rep':>6} {'travel_m':>10} {'est_mps':>10} {'walked':>8}",
                  flush=True)
            results["d"] = []
            for rep in range(SWEEP_D_REPEATS):
                for v in SWEEP_D_SPEEDS:      # interleaved, so drift hits both equally
                    travel, est = trial(loco, v, 0.0, args.hold, args.dry)
                    is_walk = travel > 0.05
                    results["d"].append({"cmd_vx": v, "rep": rep, "travel_m": travel,
                                         "est_mps": est, "walked": is_walk})
                    print(f"{v:8.3f} {rep:6d} {travel:10.3f} {est:10.3f} {is_walk!s:>8}",
                          flush=True)
        if args.sweep == "z":
            # THE NULL CONTROL, and it should have been trial zero of every sweep.
            # Commands ZERO velocity, so any travel this records is not walking: it is
            # the robot settling out of stand_up(), plus estimator drift. Every "walk"
            # measured today at low speed was the FIRST trial after standing and landed
            # at 0.114-0.127 m, which is exactly the magnitude a settle would produce.
            # Without this row there is no way to tell a slow walk from standing up.
            #
            # It is only HALF the control, and sweep P supplies the other half. A robot
            # whose legs were never enabled records this same perfect 0.000 m — so a null
            # that passes proves the odometry is honest and proves nothing at all about
            # whether anything is listening. That takes a positive control.
            print("\nSWEEP Z — zero command. Any travel here is NOT walking.", flush=True)
            print(f"{'cmd':>8} {'rep':>6} {'travel_m':>10} {'est_mps':>10}", flush=True)
            results["z"] = []
            for rep in range(4):
                travel, est = trial(loco, 0.0, 0.0, args.hold, args.dry)
                results["z"].append({"rep": rep, "travel_m": travel, "est_mps": est})
                print(f"{0.0:8.3f} {rep:6d} {travel:10.3f} {est:10.3f}", flush=True)
    finally:
        loco.stop()                      # Move persists; this is not optional
        lie_down(loco)
        after = health.latest()
        if before is not None and after is not None:
            print(f"\nmotors {before.max_motor_temp_c:.1f}C -> "
                  f"{after.max_motor_temp_c:.1f}C", flush=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"PRONE. results -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true", help="no motion; validates the sequence")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds per trial")
    ap.add_argument("--sweep", choices=("a", "b", "c", "d", "z", "p", "both"),
                    default="both")
    ap.add_argument("--out", default=None,
                    help="results file; defaults per sweep (sweep p writes JSONL)")
    ap.add_argument("--plan", action="store_true",
                    help="sweep p: print the schedule and the budget, touch nothing")
    ap.add_argument("--start-temp", type=float, default=None,
                    help="sweep p --plan: hottest motor in C, to size the session")
    ap.add_argument("--blocks", default="",
                    help="sweep p: comma-separated blocks to run, e.g. C,D,E to resume")
    ap.add_argument("--refresh", choices=("auto", "full", "balance"), default="auto",
                    help="sweep p: how to re-enter the FRESH state; auto asks block B")
    ap.add_argument("--no-pause", action="store_true",
                    help="sweep p: skip the scheduled arm break (rehearsal only)")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("/home/unitree/peercap/gait_protocol.jsonl" if args.sweep == "p"
                    else "/home/unitree/peercap/gait_sweep.json")

    if args.sweep != "p":
        run_sweeps(args)
        return

    blocks = [b.strip() for b in args.blocks.split(",")] if args.blocks else None
    if args.plan:
        print_plan(build_plan(blocks), args.start_temp)
        return
    if args.dry:
        raise SystemExit(
            "REFUSING: --dry is not available for sweep p. A dry protocol run writes a\n"
            "full-looking results file in which nothing moved, which is exactly the shape "
            "of\nthe confound that poisoned the retrospective corpus. Rehearse with "
            "--plan,\nwhich produces no results file at all.")
    if os.path.exists(args.out):
        raise SystemExit(
            f"REFUSING: {args.out} already exists. Sweep p writes incrementally so that "
            f"an\ninterrupted session keeps its trials, and overwriting one is how a "
            f"resume\ndestroys the data it was resuming from. Pass a new --out.")
    run_protocol(args)


if __name__ == "__main__":
    main()
