#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the ego-motion expansion gate.

Synthetic approaches only — the corpus that set the constants is a PARKED robot and
cannot exercise the signal at all (see ``expansion.py``). Every test here walks a
robot toward a claimed obstacle position and asks what the gate concludes.

The ones that matter, each named for the failure it prevents:

  * **A ghost is dropped, a real obstacle is not.** The whole point.
  * **Nothing is dropped for growing too FAST.** The gate is one-sided; an obstacle
    nearer than reported, or coming at the robot, must never raise a verdict.
  * **A parked robot decides nothing.** With no ego-motion the prediction is flat, the
    observed rate is noise, and a gate that ruled on it would be reading a coin. This
    is the condition the entire measured corpus was taken under.
  * **A frame-fill track is never dropped.** ``FILLS_FRAME_RANGE_M`` is a constant, so
    its observed rate is exactly zero however hard the robot closes — the arithmetic
    of a total ghost, on a box that by definition fills the frame.
  * **A ranging-source switch does not read as motion.** Measured at a median of 103%
    between consecutive frames on a parked robot; a fit spanning one measures the
    estimator changing prior.

Run: ``python3 test_expansion.py``
"""
from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expansion import (
    CONSISTENT,
    INCONSISTENT,
    LOG_RANGE_SIGMA,
    REJECT_SIGMAS,
    TAU_HORIZON_S,
    UNRESOLVED,
    ExpansionConsistency,
    _slope,
)

DT = 0.143            # the perception thread's ~7 Hz period
GAIT = 0.35           # MIN_GAIT_COMMAND_M_S — the slowest the Go2 will actually walk


def _approach(gate, *, claimed_m, true_m, speed=GAIT, steps=30, dt=DT,
              source="height", track_id=1, target_speed=0.0, sigma=0.0,
              seed=None, start_t=100.0):
    """Walk the robot down the +x axis toward an obstacle and feed the gate.

    ``claimed_m`` is where the size prior puts the obstacle at t=0; ``true_m`` is where
    it really is. The reported range is the true range divided by the scale error, which
    is exactly what a wrong size prior does: ``R_reported = (prior/true_size) * R_true``,
    a constant factor on a series that is itself real.

    ``target_speed`` moves the obstacle along +x (positive = retreating). ``sigma``
    adds log-normal noise of that fractional size to every reported range.
    """
    rng = random.Random(seed)
    scale = claimed_m / true_m          # R_reported / R_true
    verdict = None
    for i in range(steps):
        t = start_t + i * dt
        robot_x = speed * i * dt
        obstacle_x = true_m + target_speed * i * dt
        true_range = obstacle_x - robot_x
        reported = scale * true_range
        if sigma:
            reported *= math.exp(rng.gauss(0.0, sigma))
        verdict = gate.observe(track_id, time_s=t, range_m=reported, source=source,
                               odom_xy=(robot_x + reported, 0.0),
                               robot_xy=(robot_x, 0.0))
    return verdict


def _walk_past(*, claimed_m, true_m, speed, lateral_m=0.0, steps=60, dt=DT):
    """Walk the robot along +x past an obstacle offset ``lateral_m`` to the side.

    Returns ``[(sample_index, state, true_tau_s), ...]`` for the whole walk. Unlike
    ``_approach`` this deliberately continues past the closest approach to the CLAIMED
    position, which is where the prediction's V-shape lives.

    ``true_tau_s`` is the real time-to-contact, computed from the geometry the harness
    knows and the gate does not. It is what makes the reversal invariant testable: a
    verdict that flips back once the object is genuinely inside the horizon has not
    reversed itself, it has re-decided a question whose answer changed.
    """
    gate = ExpansionConsistency()
    scale = claimed_m / true_m
    states = []
    for i in range(steps):
        robot_x = speed * i * dt
        dx, dy = true_m - robot_x, lateral_m
        distance = math.hypot(dx, dy)
        reported = scale * distance
        if reported <= 1e-6:
            break
        v = gate.observe(1, time_s=100.0 + i * dt, range_m=reported, source="height",
                         odom_xy=(robot_x + reported * dx / distance,
                                  reported * dy / distance),
                         robot_xy=(robot_x, 0.0))
        closing = speed * dx / distance          # rate of change of the true range
        tau = distance / closing if closing > 0 else float("inf")
        states.append((i, v.state, tau))
    return states


# ── The two outcomes the gate exists to separate ────────────────────────────
def test_a_real_obstacle_where_it_says_it_is_survives():
    """The baseline that everything else is a deviation from. A correct size prior
    means reported == true, the range falls exactly as odometry demands, and the gate
    must say so rather than shrugging."""
    v = _approach(ExpansionConsistency(), claimed_m=2.0, true_m=2.0)
    assert v.state == CONSISTENT, f"{v.state}: {v.reason}"
    assert abs(v.scale - 1.0) < 0.02, f"scale {v.scale:.3f} should be 1.0"


def test_a_box_on_a_far_structure_is_dropped():
    """The false alarm this module is for: a person-height prior on a doorway or a
    cabinet run reports 2.0 m for something at 6.0 m, the planner's 2.5 s horizon
    reaches it, and the robot stops for a wall it was never going to touch."""
    v = _approach(ExpansionConsistency(), claimed_m=2.0, true_m=6.0)
    assert v.state == INCONSISTENT, f"{v.state}: {v.reason}"
    assert "further off than reported" in v.reason


def test_the_scale_it_reports_is_a_bound_and_errs_away_from_one():
    """`scale` estimates R_reported/R_true, but from two straight-line fits to curved
    series, so it is exact only for small travel. What must hold is the DIRECTION of
    the error: always away from 1.0, so that ``1/scale`` under-states rather than
    over-states how far off a rejected track is. A message that over-stated it would be
    a claim the measurement does not support."""
    for claimed, true in ((2.0, 3.0), (1.5, 4.5), (3.0, 4.5), (2.0, 2.4)):
        k = claimed / true
        for steps in (8, 14, 20):
            v = _approach(ExpansionConsistency(), claimed_m=claimed, true_m=true,
                          steps=steps)
            if v.scale is None:
                continue
            assert v.scale <= k + 0.01, (
                f"claimed {claimed} true {true} steps {steps}: scale {v.scale:.3f} "
                f"is above the true {k:.3f} — the error runs the unsafe way")
            assert v.scale > 0.0, v.scale


def test_a_correct_prior_reports_a_scale_of_exactly_one():
    """The one point where the ratio of fits is not approximate: with the prior right,
    the observed and predicted series are the SAME series, so the ratio is 1 whatever
    the curvature. That is also why the drop condition — the difference, not the ratio
    — carries no bias at the decision boundary."""
    checked = 0
    for steps in (8, 14, 20):
        v = _approach(ExpansionConsistency(), claimed_m=2.0, true_m=2.0, steps=steps)
        if v.scale is None:          # no ego-motion power yet; nothing to check
            continue
        assert abs(v.scale - 1.0) < 1e-9, f"{steps} samples: scale {v.scale}"
        checked += 1
    assert checked, "no window resolved at all — the test proved nothing"


def test_more_travel_never_un_drops_a_ghost_that_is_still_far():
    """⚠️ THE REGRESSION THIS FILE EXISTS FOR, and the invariant took two goes to state.

    The prediction integrates ``ln|anchor - robot|``, which falls to the closest
    approach and then RISES. Let a window span that turn and the fitted prediction
    flattens; a ghost correctly rejected earlier in the walk comes back CONSISTENT
    later. More evidence, weaker conclusion.

    But "never flips back" is TOO STRONG and asserting it found a false positive: a
    ghost 10 m off that the robot keeps walking toward genuinely comes inside
    TAU_HORIZON_S, and re-deciding is then correct. The invariant is restricted to the
    stretch where the object is still comfortably beyond the horizon, which is exactly
    where the original bug was — MEASURED there at a true 10.1 s of contact, with the
    slope's sigma rising from 0.0084 to 0.0098 as a trim shortened the window. The
    noise term moved; the signal did not.

    The claimed ranges here are small against ``speed * TAU_HORIZON_S`` on purpose, so
    the robot walks THROUGH the claimed position while the real object is still far.
    That is the only geometry in which the V-shape and the horizon are separable."""
    ever_dropped = 0
    for claimed, true, speed, lateral in ((0.8, 10.0, 0.35, 0.0), (0.8, 10.0, 0.35, 1.5),
                                          (1.0, 10.0, 0.35, 0.0), (0.8, 15.0, 0.35, 0.4),
                                          (1.2, 12.0, 0.35, 0.0), (1.0, 15.0, 0.35, 0.3),
                                          (1.5, 20.0, 0.5, 0.0), (0.8, 10.0, 0.5, 0.0),
                                          (1.5, 20.0, 0.5, 0.8), (2.0, 30.0, 0.8, 1.0)):
        states = _walk_past(claimed_m=claimed, true_m=true, speed=speed,
                            lateral_m=lateral, steps=60)
        first = next((i for i, s, _ in states if s == INCONSISTENT), None)
        if first is None:
            continue
        ever_dropped += 1
        back = [i for i, s, tau in states
                if i > first and s == CONSISTENT and tau > 1.2 * TAU_HORIZON_S]
        assert not back, (
            f"claimed {claimed} true {true} v {speed} lateral {lateral}: dropped at "
            f"sample {first} then called CONSISTENT again at {back[:5]} while contact "
            f"was still more than {1.2 * TAU_HORIZON_S:.1f}s away")
    assert ever_dropped >= 4, (
        f"only {ever_dropped} of the cases ever dropped anything — the invariant was "
        f"asserted over almost nothing")


# ── One-sidedness: the safety property ──────────────────────────────────────
def test_an_obstacle_nearer_than_reported_is_never_dropped():
    """The other half of the size-prior error, and the dangerous half: a 1.70 m person
    prior on a 0.40 m peer robot reports 2.05 m for something much closer (MEASURED on
    the p1b clip of the 2026-08-24 corpus, which reports 2.05 m for a peer filling the
    frame). Its range collapses FASTER than odometry demands, which is the opposite
    sign of a ghost, and the gate must not discard it.

    ⚠️ The obvious version of this test — a 4x error at 0.48 m — passes for the wrong
    reason: the robot walks through the claimed position inside one window and the gate
    abstains, so it never exercises the one-sidedness at all. A two-sided reject
    survived it. These cases all RESOLVE."""
    resolved = 0
    for claimed, true in ((2.0, 1.0), (3.0, 1.5), (4.0, 2.5), (3.0, 2.0)):
        v = _approach(ExpansionConsistency(), claimed_m=claimed, true_m=true, steps=20)
        assert v.state != INCONSISTENT, (
            f"{claimed} m claimed, {true} m true: dropped something closing too FAST "
            f"— {v.reason}")
        resolved += v.state == CONSISTENT
    assert resolved >= 3, f"only {resolved}/4 resolved; the one-sidedness went untested"


def test_an_obstacle_walking_into_the_robot_is_never_dropped():
    """A peer closing head-on makes the range fall faster still. Expansion is a filter
    and not an avoidance signal, but it must at minimum not delete the thing it would
    have been asked to avoid."""
    v = _approach(ExpansionConsistency(), claimed_m=3.0, true_m=3.0,
                  target_speed=-0.4, steps=20)
    assert v.state == CONSISTENT, f"{v.state}: {v.reason}"
    assert v.observed_rate < v.predicted_rate, "it IS closing faster than ego-motion"


def test_a_far_approaching_obstacle_is_kept_where_a_two_sided_test_would_drop_it():
    """⚠️ The one region where one-sidedness is the ONLY thing standing between a real
    approaching obstacle and the bin, so it is the region a two-sided test has to be
    caught in. Everywhere else the contact-horizon clause happens to save it: a target
    closing fast has a rate far below the horizon and is kept whatever the difference
    test says. Here it does not.

    A correctly-ranged obstacle 7 m off, walking in at 0.2 m/s while the robot walks at
    0.35: the range falls at -0.079/s where ego-motion alone demands -0.050/s, a
    difference well past 4 sigma, and -0.079/s is still on the safe side of the 8 s
    horizon. Symmetric rejection discards it. It is a peer robot coming at the robot."""
    v = _approach(ExpansionConsistency(), claimed_m=7.0, true_m=7.0,
                  target_speed=-0.2, steps=30)
    assert v.state == CONSISTENT, f"{v.state}: {v.reason}"
    margin = REJECT_SIGMAS * v.sigma_rate
    assert v.predicted_rate - v.observed_rate > margin, (
        "the difference is inside the margin, so a two-sided test would not have "
        "fired here either and this proves nothing")
    assert v.observed_rate - margin > -1.0 / TAU_HORIZON_S, (
        "the horizon clause would have saved it anyway, so this proves nothing")


def test_a_barely_slower_approach_is_kept_not_dropped():
    """The safety half of the drop condition. An obstacle CAN grow more slowly than
    odometry demands and still be closing on the robot — a target retreating at
    0.05 m/s against a 0.35 m/s walk is still closed on at 0.30 m/s. Being inconsistent
    is not enough; the observed rate must also put contact beyond TAU_HORIZON_S."""
    gate = ExpansionConsistency()
    v = _approach(gate, claimed_m=2.0, true_m=2.0, target_speed=0.05, steps=20)
    assert v.state == CONSISTENT, f"{v.state}: {v.reason}"
    assert v.observed_rate > v.predicted_rate, "it IS growing more slowly"
    assert "still closing" in v.reason


# ── The latch: a rejection may only be contradicted, never merely forgotten ─
def test_a_rejection_survives_the_window_getting_shorter():
    """⚠️ THE EXACT REVERSAL. Both halves of the drop condition carry
    ``REJECT_SIGMAS * sigma``, and sigma depends on how many samples the window
    currently holds. Traced on a ghost at 10x its reported range, walked at the gait
    floor: held INCONSISTENT for 60 consecutive frames, then the window went 12 samples
    to 11, 4 sigma went 0.0725 to 0.0827, the contact-horizon comparison crossed, and
    the verdict became CONSISTENT. Nothing about the world had changed.

    Asserted on the frame either side of that step, not on the whole walk, so it cannot
    pass because some other guard happened to abstain first."""
    states = _walk_past(claimed_m=1.0, true_m=10.0, speed=GAIT, steps=76)
    by_index = {i: s for i, s, _ in states}
    assert by_index[69] == INCONSISTENT, f"sample 69 was {by_index[69]}"
    assert by_index[70] == INCONSISTENT, (
        f"sample 70 was {by_index[70]} — the window lost one sample and the "
        f"rejection with it")
    assert by_index[75] == INCONSISTENT, f"sample 75 was {by_index[75]}"


def test_the_range_starting_to_close_properly_releases_a_rejection():
    """The one thing that MUST clear a rejection, or the latch is a trapdoor. The gate
    cannot separate a mis-scaled ghost from a real obstacle that was retreating, so the
    obstacle turning round and coming back has to restore it — and on positive
    evidence, not on the fit simply going quiet."""
    gate = ExpansionConsistency()
    # A ghost: reported 2.0 m, really 8.0 m. Rejected.
    _approach(gate, claimed_m=2.0, true_m=8.0, steps=30)
    assert gate.verdict(1).rejected, gate.verdict(1).reason
    # Now its range starts falling exactly as ego-motion demands, from where it is.
    start_x, start_r = GAIT * 30 * DT, 2.0 * (8.0 - GAIT * 29 * DT) / 8.0
    v = None
    for i in range(30):
        robot_x = start_x + GAIT * i * DT
        remaining = start_r - GAIT * i * DT
        v = gate.observe(1, time_s=100.0 + (30 + i) * DT, range_m=remaining,
                         source="height", odom_xy=(robot_x + remaining, 0.0),
                         robot_xy=(robot_x, 0.0))
    assert v.state == CONSISTENT, f"the latch never released: {v.reason}"
    assert not gate.rejects([1]), "still in rejects() after positive evidence"


def test_discarding_the_window_discards_the_rejection():
    """A latched rejection is an opinion about a particular run of measurements. Once a
    ranging-source switch throws those away — the 103% step — carrying the verdict over
    would be judging the new source by the old one's evidence."""
    for source, gap in (("width", DT), ("height", 500.0)):
        gate = ExpansionConsistency()
        _approach(gate, claimed_m=2.0, true_m=8.0, steps=30)
        assert gate.verdict(1).rejected
        robot_x = GAIT * 30 * DT
        v = gate.observe(1, time_s=100.0 + 29 * DT + gap, range_m=1.6, source=source,
                         odom_xy=(robot_x + 1.6, 0.0), robot_xy=(robot_x, 0.0))
        assert v.state == UNRESOLVED, f"{source}/{gap}: {v.state} — {v.reason}"
        assert not gate.rejects([1]), f"{source}/{gap}: rejection outlived its window"


def test_the_latch_is_what_holds_a_rejection_through_a_shrinking_window():
    """Names the mechanism rather than only its effect, so a refactor that silently
    removes the latch and passes the reversal test by luck cannot. The rejection that
    survives sample 70 of the walk above is the LATCHED one, and it says so."""
    states = []
    gate = ExpansionConsistency()
    for i in range(76):
        robot_x = GAIT * i * DT
        remaining = (10.0 - robot_x) / 10.0
        v = gate.observe(1, time_s=100.0 + i * DT, range_m=remaining, source="height",
                         odom_xy=(robot_x + remaining, 0.0), robot_xy=(robot_x, 0.0))
        states.append((i, v.state, v.reason))
    latched = [i for i, s, r in states if s == INCONSISTENT and "earlier evidence" in r]
    fresh = [i for i, s, r in states if s == INCONSISTENT and "earlier evidence" not in r]
    assert fresh, "nothing was ever rejected on its own evidence"
    assert latched, "the latch never carried a rejection — the reversal test is luck"
    assert min(latched) > max(fresh) - len(fresh), "the latch fired before any rejection"


def test_stopping_mid_approach_re_opens_the_question():
    """Documented rather than defended. The latch releases on a RESOLVED verdict whose
    rate difference is inside the margin, and a window half-filled with parked samples
    can produce one: both rates are diluted toward zero and they agree. So a robot that
    halts mid-approach hands the track back and has to earn the rejection again when it
    moves off.

    That is the conservative direction — it restores an obstacle rather than discarding
    one — and a parked robot is not about to reach anything. It is here so the
    behaviour is a decision and not a surprise."""
    gate = ExpansionConsistency()
    _approach(gate, claimed_m=2.0, true_m=8.0, steps=30)
    assert gate.verdict(1).rejected
    robot_x, remaining = GAIT * 29 * DT, 2.0 * (8.0 - GAIT * 29 * DT) / 8.0
    v = None
    for i in range(35):     # parked, long enough that no moving sample survives
        v = gate.observe(1, time_s=100.0 + (30 + i) * DT, range_m=remaining,
                         source="height", odom_xy=(robot_x + remaining, 0.0),
                         robot_xy=(robot_x, 0.0))
    assert v.state == UNRESOLVED, f"{v.state}: {v.reason}"
    assert not gate.rejects([1]), "the rejection outlived the evidence for it"


# ── Abstention: the states where deciding would be reading noise ────────────
def test_a_parked_robot_resolves_nothing():
    """The condition of every frame in the 2026-08-24 corpus — measured net camera
    motion of at most 14.7 px at 480-wide over a whole clip. With no ego-motion the
    prediction is flat, so a total ghost and a real obstacle produce the identical
    series and any verdict is the noise's."""
    for true_m in (2.0, 6.0, 20.0):
        v = _approach(ExpansionConsistency(), claimed_m=2.0, true_m=true_m, speed=0.0)
        assert v.state == UNRESOLVED, f"true {true_m} m: {v.state} — {v.reason}"
        assert "too little to resolve" in v.reason


def test_a_small_size_prior_error_is_not_worth_a_drop():
    """Being wrong is not the test — being wrong enough to matter is. A track reported
    at 5.0 m that is really at 6.0 m is a 20% prior error and still an obstacle the
    robot will reach; the gate must leave it alone rather than congratulate itself."""
    for claimed, true in ((5.0, 6.0), (3.0, 3.6), (2.0, 2.4)):
        v = _approach(ExpansionConsistency(), claimed_m=claimed, true_m=true)
        assert v.state != INCONSISTENT, f"{claimed}->{true}: {v.reason}"


def test_a_slow_walk_cannot_close_a_short_window():
    """Below some speed the whole exercise is a parked robot with extra steps. At
    0.05 m/s a 30-sample window covers 0.21 m, which against a 3.0 m claim moves the
    prediction by less than the noise moves the observation."""
    v = _approach(ExpansionConsistency(), claimed_m=3.0, true_m=12.0, speed=0.05)
    assert v.state == UNRESOLVED, f"{v.state}: {v.reason}"


def test_what_it_actually_decides_is_a_true_range_and_that_scales_with_speed():
    """The gate looks like a test of the size prior and behaves like a test of TRUE
    range: a track is dropped when ego-motion shows the object is really further off
    than the robot can reach inside TAU_HORIZON_S. So the true range at which drops
    begin has to scale with how fast the robot is walking, and it does — measured at
    the 30-sample window, roughly 4.1 m at 0.35 m/s, 5.8 m at 0.5 and 9.3 m at 0.8.

    Monotone on purpose. Note the REACH IN CLAIMED RANGE IS NOT monotone in speed and
    a test asserting that it was is what found this: walking faster sharpens the
    ego-motion signal and pushes the horizon out at the same time, and past about
    0.5 m/s the second wins. That is the horizon doing its job, not a defect."""
    onsets = []
    for speed in (0.35, 0.5, 0.8):
        first = None
        for claimed in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
            for ratio in [1.0 + 0.05 * i for i in range(140)]:
                v = _approach(ExpansionConsistency(), claimed_m=claimed,
                              true_m=claimed * ratio, speed=speed)
                if v.rejected:
                    true_m = claimed * ratio
                    first = true_m if first is None else min(first, true_m)
                    break
        assert first is not None, f"nothing dropped at all at {speed} m/s"
        onsets.append(round(first, 2))
    assert onsets == sorted(onsets), f"drop onset not monotonic in speed: {onsets}"
    assert onsets[-1] > 1.5 * onsets[0], f"speed barely moved the onset: {onsets}"


def test_arriving_at_the_claimed_position_resolves_nothing():
    """The other half of the conditioning guard, and the one the walked-past test
    cannot reach. A robot that arrives ON the claimed point drives the predicted range
    toward zero, and ``ln`` of it toward minus infinity: the last two samples then carry
    the whole fit. Measured with the floor removed, a correctly-ranged 2.0 m obstacle
    walked up to reported a rate of -0.626/s where the honest window gives -0.23/s.
    Abstaining is the only defensible answer at that point — and the obstacle is
    underfoot, which is the planner's problem and not this gate's."""
    gate = ExpansionConsistency()
    v = None
    for i in range(60):
        robot_x = GAIT * i * DT
        remaining = 2.0 - robot_x
        if remaining <= 1e-9:
            break
        v = gate.observe(1, time_s=100.0 + i * DT, range_m=remaining, source="height",
                         odom_xy=(2.0, 0.0), robot_xy=(robot_x, 0.0))
    assert v.state == UNRESOLVED, f"{v.state}: {v.reason} (rate {v.observed_rate})"


def test_a_window_too_short_to_resolve_the_horizon_says_so():
    """The gate has two power tests and they fail for different reasons, so they have
    to give different accounts of themselves. This one fires when the fit is precise
    enough about the ego-motion but not about the contact horizon — a short window at
    close range, where 4 sigma of slope noise is itself worth more than 1/TAU."""
    gate = ExpansionConsistency(min_samples=8)
    v = None
    for i in range(8):
        robot_x = 0.8 * i * DT
        v = gate.observe(1, time_s=100.0 + i * DT, range_m=6.0 - 0.2 * i,
                         source="height", odom_xy=(robot_x + 6.0 - 0.2 * i, 0.0),
                         robot_xy=(robot_x, 0.0))
    assert v.state == UNRESOLVED, f"{v.state}: {v.reason}"
    assert "contact horizon" in v.reason, v.reason


def test_too_few_samples_resolves_nothing():
    """MIN_SAMPLES is not a sensitivity knob — the power test does that — but a fit
    over three points has a confidence this noise model cannot support."""
    v = _approach(ExpansionConsistency(), claimed_m=2.0, true_m=6.0, steps=5)
    assert v.state == UNRESOLVED and "samples" in v.reason, v.reason


# ── The two estimator quirks that would have fired the gate on nothing ──────
def test_a_frame_fill_track_is_never_dropped():
    """`estimate_range` returns a CONSTANT 0.8 m when a box is clipped on both axes.
    Its log-range rate is exactly zero however hard the robot is closing, which is the
    arithmetic of a total ghost — so without this clause the gate would delete every
    obstacle close enough to fill the frame, i.e. every one that matters most."""
    gate = ExpansionConsistency()
    v = None
    for i in range(30):
        v = gate.observe(1, time_s=100.0 + i * DT, range_m=0.8, source="frame-fill",
                         odom_xy=(GAIT * i * DT + 0.8, 0.0),
                         robot_xy=(GAIT * i * DT, 0.0))
    assert v.state == UNRESOLVED, f"{v.state}: {v.reason}"
    assert "constant" in v.reason


def test_a_width_capped_track_is_never_dropped():
    """The same trap through the other constant. `width-capped` returns the fit range
    whenever a width span exceeds it, and a live run of 2026-08-19 held it at 0.719 m
    to three decimals for five seconds while the robot moved."""
    gate = ExpansionConsistency()
    v = None
    for i in range(30):
        v = gate.observe(1, time_s=100.0 + i * DT, range_m=0.719,
                         source="width-capped",
                         odom_xy=(GAIT * i * DT + 0.719, 0.0),
                         robot_xy=(GAIT * i * DT, 0.0))
    assert v.state == UNRESOLVED, f"{v.state}: {v.reason}"


def test_a_source_switch_restarts_the_window():
    """MEASURED: switching between the height prior, the width prior and the frame-fill
    constant moves the reported range by a median of 103% between consecutive frames —
    on a PARKED robot — and it happened on 208 of 1,471 frame pairs (14.1%) of the
    2026-08-24 corpus. A window spanning one would be fitting the estimator's change of
    mind and calling it the world."""
    gate = ExpansionConsistency()
    steps = 20
    _approach(gate, claimed_m=2.0, true_m=2.0, steps=steps, source="height")
    settled = gate.verdict(1)
    assert settled.state == CONSISTENT, settled.reason
    # THE VERY NEXT FRAME, not a later one. Putting the switched reading at an
    # arbitrary later time makes the max-gap reset clear the window first, and the
    # source clause is then never reached — a version of this test that did exactly
    # that passed with the source check deleted.
    next_t = 100.0 + steps * DT
    robot_x = GAIT * steps * DT
    # A single width-prior reading at 2.8x the range: the measured size of the step.
    v = gate.observe(1, time_s=next_t, range_m=5.6, source="width",
                     odom_xy=(robot_x + 5.6, 0.0), robot_xy=(robot_x, 0.0))
    assert v.state == UNRESOLVED and v.samples == 1, (
        f"the switch left {v.samples} samples in the window: {v.reason}")


def test_a_gap_restarts_the_window():
    """A track that coasted out of view and came back was not measured across the gap,
    and the samples either side are not one continuous approach."""
    gate = ExpansionConsistency()
    _approach(gate, claimed_m=2.0, true_m=2.0, steps=20)
    v = gate.observe(1, time_s=500.0, range_m=2.0, source="height",
                     odom_xy=(2.0, 0.0), robot_xy=(0.0, 0.0))
    assert v.samples == 1, f"a 400 s gap left {v.samples} samples in the window"


def test_time_running_backwards_restarts_the_window():
    """Perception results are consumed in sequence order, but a replay or a re-seeked
    log can hand this an older capture time. A negative dt in the fit would invert the
    sign of the observed rate — a closing obstacle read as a receding one, which is the
    drop direction."""
    gate = ExpansionConsistency()
    _approach(gate, claimed_m=2.0, true_m=2.0, steps=20, start_t=100.0)
    v = gate.observe(1, time_s=90.0, range_m=2.0, source="height",
                     odom_xy=(2.0, 0.0), robot_xy=(0.0, 0.0))
    assert v.samples == 1, f"a backwards timestamp left {v.samples} samples"


# ── The noise model, against the measured floor ─────────────────────────────
def test_the_measured_noise_floor_does_not_fire_the_gate():
    """The false-drop rate, driven at the MEASURED per-sample scatter of ln R (3.07%,
    source held, 1,133 frame pairs). The robot walks at the gait floor toward an
    obstacle that is exactly where it says it is; every rejection is a false one.

    The corpus itself bounds this at under 1 window in 595 at +4 sigma. This is the
    same statement against the same sigma with the ego-motion term switched on."""
    gate = ExpansionConsistency()
    rejects = 0
    trials = 400
    for seed in range(trials):
        gate.forget(1)
        v = _approach(gate, claimed_m=2.0, true_m=2.0, sigma=LOG_RANGE_SIGMA,
                      seed=seed, steps=30)
        rejects += v.rejected
    assert rejects <= trials // 100, (
        f"{rejects}/{trials} false drops at the measured noise floor")


def test_a_large_scale_error_is_caught_through_the_noise():
    """The other side of the same experiment: a ghost at 3x its reported range, with
    the measured noise on top, has to be caught most of the time or the gate is an
    expensive no-op."""
    gate = ExpansionConsistency()
    caught = 0
    trials = 200
    for seed in range(trials):
        gate.forget(1)
        v = _approach(gate, claimed_m=2.0, true_m=6.0, sigma=LOG_RANGE_SIGMA,
                      seed=seed, steps=30)
        caught += v.rejected
    assert caught >= int(0.9 * trials), f"caught only {caught}/{trials}"


def test_smaller_scale_errors_are_caught_less_often():
    """Detection has to fall away smoothly as the error shrinks toward 1.0, and reach
    zero when there is no error at all. A gate that caught a 1.05x error as readily as
    a 3x one would be responding to something other than the scale."""
    gate = ExpansionConsistency()
    rates = []
    for factor in (1.0, 1.2, 1.5, 2.0, 3.0):
        caught = 0
        for seed in range(120):
            gate.forget(1)
            v = _approach(gate, claimed_m=2.0, true_m=2.0 * factor,
                          sigma=LOG_RANGE_SIGMA, seed=seed, steps=30)
            caught += v.rejected
        rates.append(caught)
    assert rates == sorted(rates), f"detection not monotonic in scale error: {rates}"
    assert rates[0] == 0, f"{rates[0]}/120 drops with no scale error at all"


def test_a_longer_window_resolves_a_smaller_error():
    """The slope's sigma falls as the fit's time baseline grows, so more samples must
    buy sensitivity. If it did not, the window length would be a free parameter and the
    measured sigma would not be reaching the decision."""
    caught = []
    for steps in (12, 16, 20):
        gate = ExpansionConsistency()
        hits = 0
        for seed in range(120):
            gate.forget(1)
            v = _approach(gate, claimed_m=2.0, true_m=6.0, sigma=LOG_RANGE_SIGMA,
                          seed=seed, steps=steps)
            hits += v.rejected
        caught.append(hits)
    # 120 trials, so a 6-count band absorbs the binomial noise without absorbing the
    # effect, which is measured at 39 -> 109 -> 120.
    assert all(b >= a - 6 for a, b in zip(caught, caught[1:])), (
        f"sensitivity fell as the window grew: {caught}")
    assert caught[-1] > caught[0] + 20, f"a longer window bought nothing: {caught}"


def test_the_reported_sigma_matches_the_fit():
    """`sigma_rate` is what the whole decision is scaled by, so it has to be the real
    standard error of the slope and not a stand-in. Against a straight-line fit over
    n evenly spaced samples the closed form is sigma * sqrt(12 / (n^3 - n)) / dt."""
    gate = ExpansionConsistency()
    n = 30
    v = _approach(gate, claimed_m=2.0, true_m=2.0, steps=n)
    closed_form = LOG_RANGE_SIGMA * math.sqrt(12.0 / (n ** 3 - n)) / DT
    assert abs(v.sigma_rate - closed_form) < 1e-9 * max(1.0, closed_form), (
        f"sigma_rate {v.sigma_rate:.6f} against closed form {closed_form:.6f}")


def test_slope_recovers_a_known_line():
    """The estimator under everything else, checked against arithmetic rather than
    against itself. Irregular sample times on purpose: the camera's frame interval
    ranges over 50-142 ms and a uniform-spacing shortcut would be wrong by the ratio."""
    times = [0.0, 0.05, 0.19, 0.24, 0.38, 0.41]
    slope, sxx = _slope(times, [3.0 + 2.5 * t for t in times])
    assert abs(slope - 2.5) < 1e-9, slope
    mean = sum(times) / len(times)
    assert abs(sxx - sum((t - mean) ** 2 for t in times)) < 1e-12


# ── Bookkeeping ─────────────────────────────────────────────────────────────
def test_a_forgotten_track_leaves_nothing_behind():
    """Track ids are handed out by a counter that could be reused by a future
    implementation, and a window inherited from a deleted track would be a fit across
    two different objects."""
    gate = ExpansionConsistency()
    _approach(gate, claimed_m=2.0, true_m=6.0, track_id=7)
    assert gate.verdict(7).rejected
    gate.forget(7)
    assert gate.verdict(7).state == UNRESOLVED
    assert gate.verdict(7).samples == 0


def test_retain_sweeps_out_everything_the_tracker_deleted():
    """The tracker prunes by rebuilding its list, so there is no per-track deletion
    hook. Sweeping against the survivors is what stops the gate growing without bound
    over a long run."""
    gate = ExpansionConsistency()
    for track_id in (1, 2, 3):
        _approach(gate, claimed_m=2.0, true_m=6.0, track_id=track_id)
    gate.retain([2])
    assert gate.verdict(2).rejected
    assert gate.verdict(1).state == UNRESOLVED
    assert gate.verdict(3).state == UNRESOLVED


def test_rejects_names_only_the_inconsistent():
    gate = ExpansionConsistency()
    _approach(gate, claimed_m=2.0, true_m=6.0, track_id=1)     # ghost
    _approach(gate, claimed_m=2.0, true_m=2.0, track_id=2)     # real
    _approach(gate, claimed_m=2.0, true_m=2.0, track_id=3, speed=0.0)  # parked
    assert gate.rejects([1, 2, 3, 99]) == {1}


def test_an_unusable_range_is_refused_not_logged():
    """`range_detections` drops non-finite ranges, but this gate is fed from the
    tracker and a zero would take the logarithm to -inf and poison every later fit."""
    gate = ExpansionConsistency()
    _approach(gate, claimed_m=2.0, true_m=2.0, steps=20)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        v = gate.observe(1, time_s=300.0, range_m=bad, source="height",
                         odom_xy=(2.0, 0.0), robot_xy=(0.0, 0.0))
        assert v.state == UNRESOLVED and v.samples == 0, f"{bad} was accepted"


def test_a_window_shorter_than_the_minimum_is_refused_at_construction():
    """A gate whose window can never reach its own minimum resolves nothing, forever,
    and looks exactly like a gate that is simply cautious."""
    try:
        ExpansionConsistency(window_samples=6, min_samples=12)
    except ValueError as exc:
        assert "could never resolve" in str(exc)
    else:
        raise AssertionError("a window below min_samples was accepted")


def test_the_reject_threshold_is_the_measured_one():
    """REJECT_SIGMAS is 4.0 from the measured tail of the parked null, not 3.0 from a
    normal table: the one-sided exceedance at +3 sigma measured 0.29-0.50% against the
    0.13% a Gaussian gives. Pinned so a later 'tidy-up' to the customary value is a
    test failure rather than a silent tripling of the false-drop rate."""
    assert REJECT_SIGMAS == 4.0
    assert abs(LOG_RANGE_SIGMA - 0.031) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"expansion: {len(tests)}/{len(tests)} passed")
