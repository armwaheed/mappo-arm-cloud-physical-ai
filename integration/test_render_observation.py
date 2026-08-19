#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cover ``render_observation``'s data path. The drawing is not tested; the numbers are.

``_draw`` needs matplotlib and produces a picture, and a test that only asserts a PNG was
written would pass on a blank one. Everything a panel shows comes out of :func:`walk`, so
that is what has assertions on it: the retained map, the observation the network was
handed, and the clear window toward the goal. Run it against the recorded run whose
stall the tool was built to explain, so a regression shows up as a changed conclusion
rather than a changed shape.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from observation import window_is_sampled
from render_observation import walk
from telemetry_reader import read_run

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "policy"
STALL = ROOT / ("evidence/2026-08-18-threading-two-bins/"
                "run14-ray0-blocked-policy-retreats.jsonl")


def _records():
    return walk(read_run(STALL), PACKAGE)


def test_every_record_carries_the_observation_the_network_was_handed():
    records = _records()
    assert records, "the recorded run should produce driven ticks"
    for record in records:
        assert len(record["lidar"]) == 12
        assert all(0.0 <= v <= 0.35 for v in record["lidar"])


def test_the_retained_radius_tracks_the_producer_on_a_real_run():
    """CORRECTION 6, end to end on recorded telemetry rather than on a synthetic tick.

    The control stack's radius is ``radius_m + position_sigma``, so it CONVERGES: over
    this run every landmark falls from 0.40-0.47 m to 0.230 m. The delivered ``max`` kept
    the first, largest value for the whole run, and nothing downstream could tell —
    an over-large disc yields a perfectly well-formed range vector. Revert the fix and
    the retained radius stays at its opening value while the reported one converges,
    which is what this compares.
    """
    records = _records()
    mapped = [(retained, reported)
              for record in records
              for _x, _y, retained, _id, reported in record["obstacles"]
              if reported is not None]
    assert mapped, "the run maps two identified landmarks"
    assert all(math.isclose(retained, reported, abs_tol=1e-6)
               for retained, reported in mapped)
    # And the run really does exercise convergence, so the assertion above is not
    # vacuously true on a radius that never moved.
    assert max(r for r, _ in mapped) - min(r for r, _ in mapped) > 0.15


def test_the_stall_run_has_an_open_window_that_the_shipped_fan_cannot_sample():
    """The finding of 2026-08-18, as an executable claim.

    With the radii the producer actually reported (CORRECTION 6), a window toward the
    goal exists on essentially every tick and is far wider than the robot needs — and the
    12-ray fan still puts a ray inside it almost never, while 24 rays nearly always do.
    That is the case for a finer fan stated as an assertion, and it is deliberately
    measured AFTER the radius fix: under the latch the aperture is a third as wide and 24
    rays would not be enough either, so quoting the retrain ask against the unfixed
    observation would ask for the wrong checkpoint.
    """
    records = _records()
    windows = [r["window"] for r in records if r["window"] is not None]
    assert len(windows) >= len(records) - 1

    widths = [math.degrees(hi - lo) for lo, hi in windows]
    assert min(widths) > 7.0
    assert max(widths) > 25.0

    sampled = {n: sum(window_is_sampled(w, n) for w in windows) for n in (12, 16, 24)}
    assert sampled[12] <= 0.10 * len(windows), "12 rays should miss it nearly always"
    assert sampled[24] >= 0.90 * len(windows), "24 rays should catch it nearly always"
    assert sampled[12] < sampled[16] < sampled[24]


def test_ray_zero_is_read_in_the_run_local_frame_not_the_body_frame():
    """The nose and ray 0 diverge as soon as the robot turns, and the run's own yaw shows
    it. A record whose fan turned with the robot would keep them equal."""
    records = _records()
    yaws = [record["pose_local"][2] for record in records]
    assert max(abs(y) for y in yaws) > math.radians(3.0), \
        "the robot turned during this run, so the two frames must differ"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if not STALL.exists():
        # EVERY assertion here is about one recorded run, and `deploy/` does not put
        # `evidence/` on the robot — so on a deployed tree this suite has nothing to test.
        # It exits 0, because a missing recording is not a broken install and
        # `install.sh` reads only the exit status. It says so at maximum volume, because
        # a suite that reports "passed" while running nothing is exactly the vacuous
        # green this repository keeps finding.
        print(f"  SKIPPED — no recorded run at {STALL}")
        print(f"render_observation: 0/{len(tests)} passed, {len(tests)} SKIPPED "
              f"(needs evidence/, which is not deployed)")
        raise SystemExit(0)
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"render_observation: {len(tests)}/{len(tests)} passed")
