#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the replay tool's own measurements.

``replay_mappo.py`` produces the numbers this project quotes as evidence, and it had no
tests at all until issue #17 showed one of those numbers was measuring the wrong object.
A tool that generates evidence needs its measurements pinned at least as tightly as the
code it measures.

The first section uses a stub controller rather than the real checkpoint:
:func:`policy_sight` is pure arithmetic over an observation vector, and a test whose
expected value depends on what a set of weights happens to produce is not a test of the
arithmetic.

The second section is the opposite on purpose, and it is the only place here that loads
the checkpoint. Its subject is the GHOST COUNT — how often the policy's retained obstacle
map holds something the telemetry's own obstacle list does not — which is a property of
the real weights, the real config and the real recorded runs together. No stub can
produce it, and issue #19 has now twice watched that number move without a suite noticing.

Run: ``python3 test_replay_mappo.py``
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_mappo import (
    OBSERVATION_STATE_WIDTH,
    derived_config,
    policy_sight,
    replay,
)
from telemetry_reader import read_run


@dataclass(frozen=True)
class _Config:
    lidar_range_vmas: float = 0.35
    meters_per_vmas_unit: float = 2.5

    @property
    def lidar_range_m(self) -> float:
        return self.lidar_range_vmas * self.meters_per_vmas_unit


class _Controller:
    """Only the two attributes :func:`policy_sight` is allowed to read."""

    def __init__(self, observation) -> None:
        self.last_observation = observation
        self.cfg = _Config()


def _observation(lidar, state=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
    """A full observation: six state entries then the fan, as the checkpoint records."""
    return list(state) + list(lidar)


# ── Visibility comes from the observation, not from anywhere else ───────────
def test_an_empty_fan_is_not_visible():
    visible, surface = policy_sight(_Controller(_observation([0.0] * 12)))
    assert visible is False
    assert surface == float("inf")


def test_a_controller_that_has_not_stepped_is_not_visible():
    """``last_observation`` is ``None`` before the first step and on a tick with no goal.
    Returning "visible" there would invent an obstacle out of missing data."""
    visible, surface = policy_sight(_Controller(None))
    assert visible is False
    assert surface == float("inf")


def test_a_positive_ray_is_visible_and_inverts_the_proximity_encoding():
    """``lidar`` is PROXIMITY — ``lidar_range_vmas - range_vmas`` — so the LARGEST entry
    is the NEAREST thing, and the surface range is the encoding read backwards.

    0.15 of proximity against a 0.35 range leaves 0.20 VMAS of range, which at 2.5 m per
    unit is 0.50 m. Getting the sign of this backwards yields a plausible number rather
    than an error, which is why it is pinned.
    """
    fan = [0.0] * 12
    fan[3] = 0.15
    visible, surface = policy_sight(_Controller(_observation(fan)))
    assert visible is True
    assert math.isclose(surface, 0.50, abs_tol=1e-9), surface


def test_the_nearest_ray_wins_not_the_first_or_the_last():
    fan = [0.0] * 12
    fan[1], fan[7] = 0.05, 0.25          # 0.25 is the closer one
    _, surface = policy_sight(_Controller(_observation(fan)))
    assert math.isclose(surface, (0.35 - 0.25) * 2.5, abs_tol=1e-9), surface


def test_the_state_entries_are_not_read_as_rays():
    """The fan starts at :data:`OBSERVATION_STATE_WIDTH`. Slicing at the wrong offset
    would read the robot's own position — routinely much larger than any proximity — as
    the nearest obstacle, and report a confident, wrong range instead of failing."""
    state = (99.0, 99.0, 0.0, 0.0, 99.0, 99.0)
    visible, surface = policy_sight(_Controller(_observation([0.0] * 12, state=state)))
    assert visible is False, "position entries must not count as a detection"
    assert surface == float("inf")


def test_the_state_width_matches_the_checkpoints_recorded_layout():
    """``[x, y, vx, vy, x-gx, y-gy, *lidar]`` — six state entries. Pinned so a checkpoint
    with a different layout fails here rather than silently mis-slicing."""
    assert OBSERVATION_STATE_WIDTH == 6


# ── The ghost count, against the recorded corpus ────────────────────────────
#
# A "ghost" is a tick where the POLICY's own fan reported something inside its horizon
# while the telemetry's obstacle list did not. ``replay_mappo`` has printed that number
# since issue #17, and issue #19 exists because of it — but nothing has ever held it, and
# it has already moved twice unattended: 37 ghost ticks across the six runs of 2026-08-17
# on 2026-08-18, then 15 on 2026-08-25, from a radius-latch correction that was not aimed
# at it. An unpinned number that moves by accident can go back by accident.
#
# These replays are cheap — the whole table below runs two controllers over 24 files in
# about a third of a second — so the pin covers every recorded run rather than the six
# the issue happens to quote.

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
POLICY_PACKAGE = Path(__file__).resolve().parent.parent / "policy"

#: ``run -> (ticks replayed, ghost ticks, closest ghost in metres or None)``.
#:
#: MEASURED, not chosen: every line is what ``replay_mappo.py`` prints for that file on
#: ``main`` today. A number here changing is not automatically a regression — six of these
#: runs carry ghosts and driving them to zero is the point of issue #19 — but it must be a
#: change someone MEANT, and the ``closest`` column is the one that matters for the robot:
#: 0.000 m means the policy believed it was standing inside an obstacle.
#:
#: ``sample_telemetry.jsonl`` is a byte-identical copy of ``live_run_telemetry.jsonl``, so
#: its 5 ghosts are the same 5 counted twice in the total below. Both are listed because
#: both are files a reader will replay.
GHOST_BASELINE = {
    "2026-08-14-first-policy-driven-walk/hero-run-telemetry.jsonl": (58, 0, None),
    "2026-08-17-corridor-and-room-runs/dryrun-corridor-scene-check.jsonl": (195, 0, None),
    "2026-08-17-corridor-and-room-runs/dryrun-room-scene-check.jsonl": (145, 0, None),
    "2026-08-17-corridor-and-room-runs/run0-planner-baseline-corridor.jsonl": (82, 0, None),
    "2026-08-17-corridor-and-room-runs/run1-mappo-corridor-wall-contact.jsonl": (72, 0, None),
    "2026-08-17-corridor-and-room-runs/run2-maxvy010-gait-floor-stall.jsonl": (29, 0, None),
    "2026-08-17-corridor-and-room-runs/run3-control-dt-fix-corridor.jsonl": (61, 0, None),
    "2026-08-17-corridor-and-room-runs/run4-room-cabinet-contact.jsonl": (38, 0, None),
    "2026-08-17-corridor-and-room-runs/run5-room-policy-driven-success.jsonl": (70, 15, 0.515),
    "2026-08-18-swerve-width-and-veto-precedence/runA-scale2.0-veto-on-servo-on.jsonl":
        (64, 0, None),
    "2026-08-18-swerve-width-and-veto-precedence/runB-scale2.0-veto-off-servo-on.jsonl":
        (78, 0, None),
    "2026-08-18-swerve-width-and-veto-precedence/runC-scale2.0-veto-on-servo-off.jsonl":
        (112, 0, None),
    "2026-08-18-threading-two-bins/run1-gait-floor-stall-veto-crawl.jsonl": (59, 1, 0.853),
    "2026-08-18-threading-two-bins/run10-forward-clamp-pure-strafe.jsonl": (46, 0, None),
    "2026-08-18-threading-two-bins/run11-SUCCESS-threaded-the-gap.jsonl": (79, 0, None),
    "2026-08-18-threading-two-bins/run13-person-track-covers-goal.jsonl": (212, 0, None),
    "2026-08-18-threading-two-bins/run14-ray0-blocked-policy-retreats.jsonl": (50, 0, None),
    "2026-08-18-threading-two-bins/run15-aimed-8deg-off-corridor.jsonl": (42, 0, None),
    "2026-08-18-threading-two-bins/run7-veto-shortened-policy-stall.jsonl": (49, 15, 0.409),
    "2026-08-18-threading-two-bins/run9-full-speed-reverse-hazard.jsonl": (45, 0, None),
    # Added with the run itself (issue #26). Zero ghosts of 86 ticks, which is what a
    # run with one mapped bin and one tracked person in a clear lane should look like —
    # the freeze it records is a VELOCITY failure, not a perception one, and this line is
    # the assertion that the two do not get confused later.
    "2026-08-27-gait-floor-freeze/run-20260827T012702Z-00652ea.jsonl": (86, 0, None),
    "2026-08-25-peer-runs/contrast-run-telemetry.jsonl": (91, 44, 0.000),
    "2026-08-25-peer-runs/hero-run-telemetry.jsonl": (59, 32, 0.000),
    "live_run_telemetry.jsonl": (122, 5, 0.720),
    "sample_telemetry.jsonl": (122, 5, 0.720),
}

#: Every ghost tick in the corpus. Held separately from the table so that moving one run's
#: ghosts into another run's does not read as "no change".
CORPUS_GHOST_TICKS = 117

RUN5 = "2026-08-17-corridor-and-room-runs/run5-room-policy-driven-success.jsonl"
PEER_RUN = "2026-08-25-peer-runs/contrast-run-telemetry.jsonl"


def _ghosts(name: str, **config_overrides) -> tuple:
    """``(rows, ghost rows)`` for one recorded run, through the real checkpoint."""
    run = read_run(EVIDENCE / name)
    if config_overrides:
        with derived_config(POLICY_PACKAGE / "config.json", **config_overrides) as path:
            result = replay(run, POLICY_PACKAGE, config=path)
    else:
        result = replay(run, POLICY_PACKAGE)
    rows = result["rows"]
    return rows, [row for row in rows if row["remembered_only"]]


def test_the_recorded_corpus_ghost_counts_have_not_moved():
    """Every committed telemetry file, against :data:`GHOST_BASELINE`.

    What makes this fail: any change to the retained map, the association radius, the disc
    radius, the horizon or the scale that moves how often the policy's fan and the
    telemetry disagree — in either direction. That is deliberate. An improvement has to be
    re-measured into the table by hand, which is the step that was missing when 37 became
    15 and nobody noticed until a triage pass three weeks later.

    It also fails on a recorded run that is NOT in the table. A baseline that covers only
    the files someone remembered is how the peer runs of 2026-08-25 came to carry 44 and 32
    ghost ticks, at 0.000 m, while the issue was still quoting the six runs of 2026-08-17.
    """
    streams = set()
    for path in sorted(EVIDENCE.rglob("*.jsonl")):
        try:
            read_run(path)
        except Exception:
            continue        # leg-encoder logs and the like: not a telemetry stream
        streams.add(str(path.relative_to(EVIDENCE)))
    assert streams == set(GHOST_BASELINE), \
        f"not baselined: {sorted(streams - set(GHOST_BASELINE))}; " \
        f"gone: {sorted(set(GHOST_BASELINE) - streams)}"

    measured = {}
    for name in GHOST_BASELINE:
        rows, ghosts = _ghosts(name)
        closest = min((row["policy_surface_m"] for row in ghosts), default=None)
        measured[name] = (len(rows), len(ghosts),
                          None if closest is None else round(closest, 3))

    for name, expected in GHOST_BASELINE.items():
        ticks, count, closest = measured[name]
        assert (ticks, count) == expected[:2], f"{name}: {(ticks, count)} != {expected[:2]}"
        if expected[2] is None:
            assert closest is None, f"{name}: expected no ghost, closest {closest}"
        else:
            assert closest is not None and abs(closest - expected[2]) < 0.005, \
                f"{name}: closest ghost {closest} != {expected[2]}"

    total = sum(row[1] for row in measured.values())
    assert total == CORPUS_GHOST_TICKS, f"corpus ghost total moved to {total}"


def test_run5s_ghosts_are_one_landmark_the_producer_withheld_not_one_it_lost():
    """The mechanism behind run 5's 15, pinned so a wrong fix cannot look like a right one.

    The issue body names three suspects — the 120 s TTL, the append path, and odometry
    moving a stale entry. None of them produces this number. The 15 ghost ticks are one
    contiguous window in which the telemetry publishes ``landmark-2`` and ``landmark-5``
    and withholds ``landmark-4``; ``landmark-4`` returns on the very next tick with its x,
    y and radius IDENTICAL TO THE LAST BIT, so it was never re-observed and never re-fused
    — the producer's ``static_map.confirmed()`` caps its published set at
    ``MAX_LANDMARKS_PER_LABEL = 2`` per label, and for 4.35 s the nearest bin was not in
    the best-evidenced two. The policy's retained copy was RIGHT: when the telemetry starts
    publishing it again it agrees with the remembered disc to 24 mm.

    So a fix that retires the remembered landmark makes this number better and the robot
    worse — it deletes a real obstacle half a metre away to make a metric read zero. Adding
    the issue's disagreement-based expiry takes run 5 to 0 ghosts and takes this test red,
    which is the point of it.
    """
    rows, ghosts = _ghosts(RUN5)
    assert len(ghosts) == 15, len(ghosts)

    index = {row["t"]: position for position, row in enumerate(rows)}
    first, last = index[ghosts[0]["t"]], index[ghosts[-1]["t"]]
    assert last - first + 1 == len(ghosts), "the ghost ticks are one contiguous window"

    ticks = {tick["t"]: tick for tick in read_run(EVIDENCE / RUN5).ticks
             if tick.get("t") is not None}
    published = {row["t"]: {obstacle["id"]: obstacle
                            for obstacle in (ticks[row["t"]].get("obstacles") or [])}
                 for row in rows}

    for ghost in ghosts:
        listed = published[ghost["t"]]
        assert set(listed) == {"landmark-2", "landmark-5"}, (ghost["t"], sorted(listed))

    before = published[rows[first - 1]["t"]]["landmark-4"]
    after = published[rows[last + 1]["t"]]["landmark-4"]
    assert (before["x"], before["y"], before["radius_m"]) == \
           (after["x"], after["y"], after["radius_m"]), \
        "landmark-4 was withheld, not lost: a re-acquired landmark would have moved"

    withheld_s = rows[last + 1]["t"] - rows[first - 1]["t"]
    assert 4.3 < withheld_s < 4.4, withheld_s

    # The remembered disc against the producer's own estimate, on the tick the producer
    # starts publishing it again. THIS is the number that says the memory was right, and
    # 24 mm is far tighter than the 0.05 m gate, so the gate is not what is passing.
    returning = rows[last + 1]
    disagreement = abs(returning["policy_surface_m"] - returning["nearest_surface_m"])
    assert disagreement < 0.05, (disagreement, returning)


def test_the_ghost_count_cannot_be_moved_by_the_obstacle_ttl():
    """``static_obstacle_ttl_s`` is inert in replay, so the ghost count cannot audit it.

    Issue #19's proposed loop is "change the TTL, re-run ``replay_mappo.py`` over the
    recorded runs, and the ghost count is the acceptance criterion". That loop does not
    exist yet. ``replay_mappo`` calls ``robot_input`` without ``monotonic_s``, so
    ``RobotInput.timestamp_s`` is ``None``, so ``MappoController.step`` stamps the sample
    from ITS OWN ``time.monotonic()`` — the replaying workstation's clock, not the run's.
    Seventy ticks of a 20 s robot run replay in a tenth of a second, so no entry is ever
    old enough to expire at ANY setting: 0.5 s and 120 s give the same count, on the run
    with the most ghosts and on a peer run with a different mechanism.

    ``mappo_drive.py`` DOES pass ``monotonic_s=time.monotonic()``, so the TTL is live on
    the robot. It is only the offline instrument that cannot see it, which is why "set a
    shorter TTL and re-run the six files" would have reported success without testing
    anything.

    What would make this fail: teaching the replay to stamp each tick with the run's own
    timeline. That is a good change — and when it lands, this test is the reminder that
    :data:`GHOST_BASELINE` was measured under a TTL that did nothing and has to be
    re-measured.
    """
    for name in (RUN5, PEER_RUN):
        _, shipped = _ghosts(name)
        _, impatient = _ghosts(name, static_obstacle_ttl_s=0.5)
        assert len(shipped) == len(impatient), \
            f"{name}: {len(shipped)} ghosts at the shipped TTL, {len(impatient)} at 0.5 s"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"replay_mappo: {len(tests)}/{len(tests)} passed")
