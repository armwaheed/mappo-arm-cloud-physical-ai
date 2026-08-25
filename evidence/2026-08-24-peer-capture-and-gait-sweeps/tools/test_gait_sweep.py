#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Falsify sweep P's scheduling and classification BEFORE the cold window opens.

The session this schedules gets about 250 seconds of standing before the motors reach
their ceiling, and there is no second attempt on the same afternoon. So everything in
`gait_sweep.py` that can be wrong without the robot noticing is tested here: the
counterbalancing, the null-after-every-trial rule, the three-way outcome classifier
against the actual numbers from 2026-08-24, the non-monotonicity detector against the
actual inversion it missed that day, the thermal budget, and the sign rule that is the
whole fixed-position mechanism.

Nothing here touches DDS. `gait_sweep` puts two robot paths on `sys.path` at import time
and imports the robot modules inside functions, so importing it on a workstation is safe.

Run: ``python3 test_gait_sweep.py``
"""
from __future__ import annotations

import os
import sys

# The sys.path line goes in BEFORE the sibling import and must stay there. `ruff --fix`
# sorts imports into contiguous blocks and has been observed in this repository hoisting
# a sibling import above the line that makes it importable, turning a passing suite into
# ModuleNotFoundError with nobody touching a test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gait_sweep as gs


# ── The schedule ────────────────────────────────────────────────────────────
def test_every_commanded_trial_is_followed_by_a_null():
    """Confound 2 of 2026-08-24, made structural.

    The null control did not exist until the third reversal, and until it did there was no
    way to tell a slow walk from the robot settling. Here it is not a habit an operator has
    to remember: the plan is built with one after every command, so a session without them
    is not reachable.
    """
    plan = gs.build_plan()
    for index, item in enumerate(plan[:-1]):
        if item["kind"] in ("test", "staircase") and gs.step_speed(item["steps"][0]):
            following = plan[index + 1]
            assert following["kind"] == "null", (item["id"], following["id"])
    assert sum(1 for i in plan if i["kind"] == "null") >= 25, "nulls got dropped"


def test_the_settling_control_is_run_in_the_fresh_state_too():
    """Conclusion 4 of 2026-08-24 was "the first-trial walks are the robot settling".

    It was overturned by a null control that happened to be the first trial after standing.
    That was luck, not design: a null in the REST state cannot refute it. Sweep P runs a
    refresh-then-zero trial three times, at three different motor temperatures.
    """
    fresh_nulls = [i for i in gs.build_plan()
                   if i["kind"] == "null" and i["state"] == "FRESH"]
    assert len(fresh_nulls) >= 3, fresh_nulls


def test_the_state_order_reverses_between_the_two_passes_of_block_c():
    """Motor temperature, arm sway and battery charge all rise monotonically through a
    session, and all three were perfectly confounded with "first trial after standing" on
    2026-08-24. If both passes ran the states in the same order, they still would be."""
    tests = [i for i in gs.build_plan(["C"]) if i["kind"] == "test"]
    first_pass = [i["state"] for i in tests if i["note"].endswith("1")]
    second_pass = [i["state"] for i in tests if i["note"].endswith("2")]
    assert first_pass[0] != second_pass[0], (first_pass, second_pass)
    assert first_pass[0] == second_pass[1], "the pairs are swapped, not shuffled"


def test_the_command_order_also_reverses():
    tests = [i for i in gs.build_plan(["C"]) if i["kind"] == "test"]
    first = [gs.step_speed(i["steps"][0]) for i in tests if i["note"].endswith("1")]
    second = [gs.step_speed(i["steps"][0]) for i in tests if i["note"].endswith("2")]
    assert first[0] == min(gs.LATERAL_COMMANDS)
    assert second[0] == max(gs.LATERAL_COMMANDS)


def test_no_trial_anywhere_commands_a_negative_forward_speed():
    """`vx` may never be negative on this unit: there is no rear-facing sensing. Nor is
    `vy` ever planned negative — which way to strafe is a runtime decision, so a minus
    sign in the plan would silently fight the one that holds the position."""
    for item in gs.build_plan():
        for vx, vy, seconds in item["steps"]:
            assert vx >= 0.0, item
            assert vy >= 0.0, item
            assert seconds > 0.0, item


def test_the_block_d_primer_is_forward_and_known_good():
    """The ROLLING state has to be REACHED before it can be measured, and the only
    commands established to walk this robot in every trial are forward and >= 0.25 m/s.
    A lateral primer would void the staircase on exactly the trials that needed it."""
    for item in gs.build_plan(["D"]):
        if item["kind"] != "staircase":
            continue
        primer_vx, primer_vy, _ = item["steps"][0]
        assert primer_vy == 0.0 and primer_vx >= 0.25, item
        last_vx, last_vy, seconds = item["steps"][-1]
        assert (last_vx, last_vy) == (0.0, 0.150), "the live stall command, exactly"
        assert seconds >= 4.0, "issue #31's stall sustained 4.1 s; a shorter hold cannot"


def test_every_trial_id_is_unique():
    ids = [i["id"] for i in gs.build_plan()]
    assert len(ids) == len(set(ids))


# ── The budget ──────────────────────────────────────────────────────────────
def test_the_whole_plan_fits_a_cold_robot_and_not_a_warm_one():
    """~1 C per 16 s of standing, a 50 C self-imposed ceiling. The session that produced
    the confounds ENDED at 48 C, so "cold" is a staging requirement and not a nicety."""
    plan = gs.build_plan()
    needed = gs.plan_seconds(plan)
    assert gs.stand_budget_s(32.0) > needed, needed
    assert gs.stand_budget_s(46.0) < needed, needed


def test_a_short_budget_drops_the_optional_tail_and_never_block_e():
    """Block E is the hot repeat that discriminates the state hypothesis from the
    temperature one, plus the closing anchor. A session that drops it has measured a grid
    it cannot interpret, so it is reserved before anything else may spend."""
    plan = gs.build_plan()
    kept, dropped = gs.fit_blocks(plan, 150.0)
    assert "E" in kept, kept
    assert "G" in dropped and "F" in dropped, dropped
    assert kept[:5] == ["A", "B", "C", "D", "E"]


def test_a_warm_robot_is_told_the_required_blocks_do_not_fit():
    """`fit_blocks` can only drop the optional tail, so at 41 C it still returns A-E — and
    a run that discovers the shortfall halfway through block D has already spent the
    budget on C. The shortfall is reported before anything stands up."""
    plan = gs.build_plan()
    budget = gs.stand_budget_s(41.0)
    kept, _ = gs.fit_blocks(plan, budget)
    assert kept == ["A", "B", "C", "D", "E"], kept
    assert gs.budget_shortfall(plan, kept, budget) > 0.0
    cold = gs.stand_budget_s(32.0)
    assert gs.budget_shortfall(plan, gs.fit_blocks(plan, cold)[0], cold) == 0.0


def test_a_generous_budget_drops_nothing():
    kept, dropped = gs.fit_blocks(gs.build_plan(), 10_000.0)
    assert dropped == []
    assert len(kept) == len(gs.BLOCK_PLANS)


def test_the_budget_is_zero_rather_than_negative_past_the_ceiling():
    assert gs.stand_budget_s(gs.STAND_CEILING_C + 5.0) == 0.0


# ── The classifier, against the numbers actually recorded ───────────────────
def _trace(net_m, seconds, tail_mps, samples=20):
    """A synthetic straight-line trace: `net_m` of travel over `seconds`, ending at
    `tail_mps` on the estimator."""
    out = []
    for index in range(samples):
        fraction = index / (samples - 1.0)
        out.append({"t": fraction * seconds, "seg": 0,
                    "x": fraction * net_m, "y": 0.0, "yaw": 0.0,
                    "mvx": tail_mps, "mvy": 0.0})
    return out


def test_the_null_control_of_2026_08_24_classifies_as_no_gait():
    """0.001 m over 2.5 s, estimator noise ~0.010 m/s. If this ever reads as motion the
    whole session is unreadable, because every other verdict is relative to it."""
    verdict = gs.classify(_trace(0.001, 2.5, 0.010), [(0.0, 0.0, 2.5)], 0.01)
    assert verdict[0]["class"] == "no-gait", verdict


def test_the_stalls_of_2026_08_24_classify_as_no_gait():
    """0.004-0.017 m of travel, which the day called "no". The worst of them, 0.017 m in
    3.0 s, is the one that decides whether the threshold is in the right place."""
    for net, seconds in ((0.004, 2.5), (0.013, 2.5), (0.0176, 3.0)):
        verdict = gs.classify(_trace(net, seconds, 0.021), [(0.15, 0.0, seconds)], 0.01)
        assert verdict[0]["class"] == "no-gait", (net, verdict)


def test_the_walks_of_2026_08_24_classify_as_motion():
    """0.114 m forward at 0.100 m/s and 0.127 m lateral at 0.200 — the first-trial
    successes the whole investigation is about."""
    for net, seconds, tail in ((0.114, 3.0, 0.049), (0.127, 2.0, 0.072)):
        verdict = gs.classify(_trace(net, seconds, tail), [(0.10, 0.0, seconds)], 0.01)
        assert verdict[0]["class"] == "gait", (net, verdict)


def test_a_shuffle_is_not_reported_as_a_gait():
    """0.078-0.088 m of travel that dies before the hold ends — issue #42 calls it "real
    motion, not a sustained gait". The binary `travel > 0.05` used on 2026-08-24 could not
    see the difference, and it is the difference between a swerve that completes and one
    that stops halfway across the lane."""
    verdict = gs.classify(_trace(0.078, 3.0, 0.005), [(0.175, 0.0, 3.0)], 0.01)
    assert verdict[0]["class"] == "shuffle", verdict
    assert gs.walked({"segments": verdict}), "a shuffle still counts as having moved"


def test_a_step_out_and_back_is_not_recorded_as_a_stall():
    """Net displacement alone says 0.000 m for a robot that stepped 0.20 m and rocked
    back. `path_m` is recorded beside it so that case is visible in the data."""
    out = [{"t": 0.0, "seg": 0, "x": 0.0, "y": 0.0, "yaw": 0.0, "mvx": 0.0, "mvy": 0.0},
           {"t": 1.0, "seg": 0, "x": 0.2, "y": 0.0, "yaw": 0.0, "mvx": 0.2, "mvy": 0.0},
           {"t": 2.0, "seg": 0, "x": 0.0, "y": 0.0, "yaw": 0.0, "mvx": 0.0, "mvy": 0.0}]
    verdict = gs.classify(out, [(0.15, 0.0, 2.0)], 0.01)
    assert verdict[0]["net_m"] == 0.0
    assert verdict[0]["path_m"] == 0.4, verdict


def test_each_staircase_step_is_classified_separately():
    """The sustaining floor is "which step did the gait die on", so a single verdict for
    the whole trial would answer a different question."""
    samples = []
    for seg, (moving, base) in enumerate(((True, 0.0), (True, 0.4), (False, 0.8))):
        for index in range(10):
            fraction = index / 9.0
            samples.append({"t": seg * 1.5 + fraction * 1.5, "seg": seg,
                            "x": base + (0.2 * fraction if moving else 0.0), "y": 0.0,
                            "yaw": 0.0, "mvx": 0.12 if moving else 0.0, "mvy": 0.0})
    verdict = gs.classify(samples, [(0.25, 0.0, 1.5), (0.20, 0.0, 1.5), (0.085, 0.0, 1.5)], 0.01)
    assert [s["class"] for s in verdict] == ["gait", "gait", "no-gait"], verdict


def test_a_segment_with_no_samples_is_reported_rather_than_scored():
    verdict = gs.classify([], [(0.15, 0.0, 2.0)], 0.01)
    assert verdict[0]["class"] == "no-data"
    assert not gs.walked({"segments": verdict})


# ── The non-monotonicity detector ───────────────────────────────────────────
def _record(cmd, axis, state, moved):
    return {"kind": "test", "axis": axis, "state": state, "steps": [(cmd, 0.0, 2.0)],
            "state_verified": True,
            "segments": [{"class": "gait" if moved else "no-gait"}]}


def test_it_catches_the_inversion_that_2026_08_24_wrote_down_as_a_finding():
    """0.100 m/s walked and 0.137 m/s, commanded next, did not. That is the shape of a
    confound, and it appeared four separate times that day."""
    bad = gs.monotonicity_violations([_record(0.100, "fwd", "REST", True),
                                      _record(0.137, "fwd", "REST", False)])
    assert len(bad) == 1, bad
    assert bad[0]["lower"] == 0.100 and bad[0]["higher"] == 0.137


def test_it_stays_quiet_on_a_monotone_result():
    fine = gs.monotonicity_violations([_record(0.100, "lat", "REST", False),
                                       _record(0.150, "lat", "REST", False),
                                       _record(0.200, "lat", "REST", True)])
    assert fine == [], fine


def test_states_are_compared_within_themselves_and_not_pooled():
    """A starting floor above a sustaining floor is exactly a case where FRESH walks at a
    speed REST does not. Pooling the states would report that real effect as a confound and
    send the operator off to re-run a block that was fine."""
    records = [_record(0.100, "lat", "FRESH", True), _record(0.150, "lat", "REST", False)]
    assert gs.monotonicity_violations(records) == []


def test_a_trial_whose_state_could_not_be_verified_is_excluded():
    """A "REST" trial issued while the robot was still coasting is not a REST trial, and
    counting it is how an uncontrolled variable gets back into the table."""
    records = [_record(0.100, "lat", "REST", True), _record(0.150, "lat", "REST", False)]
    records[1]["state_verified"] = False
    assert gs.monotonicity_violations(records) == []


# ── The fixed-position mechanism ────────────────────────────────────────────
class _Pose:
    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw


class _Loco:
    def __init__(self):
        self.at = _Pose(0.0, 0.0, 0.0)

    def pose(self):
        return self.at


class _Args:
    refresh = "auto"
    no_pause = True

    def __init__(self, out):
        self.out = out


def _session(loco):
    # The results file goes to /dev/null: what is under test here is the geometry, and a
    # test that leaves JSONL scattered through the temp directory is its own small mess.
    session = gs.Session(loco, None, None, _Args(os.devnull), [], {})
    session.set_mark()
    return session


def test_the_lateral_sign_alternates_while_the_robot_is_near_its_mark():
    """The whole fixed-position argument on the lateral axis. A stall does not move the
    robot at all, so the trials that matter most cost nothing in floor space; a success
    moves it ~0.13 m and the next trial takes it straight back."""
    session = _session(_Loco())
    signs = [session.sign_for(gs.LAT) for _ in range(4)]
    assert signs in ([-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]), signs


def test_the_lateral_sign_points_home_once_the_robot_has_drifted():
    session = _session(_Loco())
    session.loco.at = _Pose(0.0, 0.30, 0.0)
    assert session.sign_for(gs.LAT) == -1.0
    session.loco.at = _Pose(0.0, -0.30, 0.0)
    assert session.sign_for(gs.LAT) == 1.0


def test_the_forward_axis_is_never_signed():
    session = _session(_Loco())
    session.loco.at = _Pose(-2.0, 0.0, 0.0)
    assert session.sign_for(gs.FWD) == 1.0, "no rear sensing: vx is never negative"


def test_the_offset_is_measured_in_the_marks_own_frame():
    """The robot is marked wherever it happens to be pointing, so an odom-frame offset
    would call a strafe a forward drift the moment the mark's yaw was not zero."""
    loco = _Loco()
    loco.at = _Pose(0.0, 0.0, 1.5707963)         # marked facing +y
    session = _session(loco)
    loco.at = _Pose(0.0, 0.4, 1.5707963)         # then 0.4 m further along its nose
    along, across = session.offsets()
    assert abs(along - 0.4) < 1e-6, (along, across)
    assert abs(across) < 1e-6, (along, across)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"gait_sweep: {len(tests)}/{len(tests)} passed")
