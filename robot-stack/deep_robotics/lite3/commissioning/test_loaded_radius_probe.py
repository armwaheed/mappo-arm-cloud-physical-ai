#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Lite3 loaded-radius recorder.

The arithmetic is one line, so these are mostly guard tests. The one that matters is
:func:`test_the_radius_is_the_corner_not_the_longest_extent`: the planner treats the robot
as a disc, and a disc sized to the longest single extent leaves all four corners of the
robot outside it. That is the difference between clearing a door frame and clipping it.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))

from deep_robotics.lite3.commissioning.loaded_radius_probe import (
    ASYMMETRY_RATIO,
    check_symmetry,
    circumscribing_radius,
    main,
    policy_scale,
)
from deep_robotics.lite3.commissioning.measurement import (
    POLICY_AGENT_RADIUS_M,
    Refusal,
    read_record,
    run_main,
)

_CONTEXT = ["--robot-id", "LITE3-A", "--firmware", "V1.0.8", "--payload", "mast"]
_EXTENTS = ["--front", "0.42", "--back", "0.38", "--left", "0.24", "--right", "0.24"]


def _quiet(callable_):
    with contextlib.redirect_stdout(io.StringIO()):
        return callable_()


def test_the_radius_is_the_corner_not_the_longest_extent():
    radius = circumscribing_radius(0.42, 0.38, 0.24, 0.24)
    assert abs(radius - math.hypot(0.42, 0.24)) < 1e-12
    assert radius > 0.42, "a disc sized to the longest extent leaves the corners outside"


def test_the_radius_uses_the_larger_of_each_opposing_pair():
    assert abs(circumscribing_radius(0.10, 0.42, 0.24, 0.10)
               - circumscribing_radius(0.42, 0.10, 0.10, 0.24)) < 1e-12


def test_policy_scale_divides_by_the_trained_agent_radius():
    assert abs(policy_scale(0.484) - 4.84) < 1e-9
    assert abs(policy_scale(POLICY_AGENT_RADIUS_M) - 1.0) < 1e-12


def test_a_lopsided_pair_is_refused_because_the_tape_was_probably_on_the_wrong_origin():
    try:
        check_symmetry(0.60, 0.20, 0.24, 0.24, confirmed=False)
    except Refusal as refusal:
        assert "turns about" in str(refusal)
    else:
        raise AssertionError("a 3x front/back split must be questioned")


def test_the_symmetry_check_can_be_overridden_deliberately():
    check_symmetry(0.60, 0.20, 0.24, 0.24, confirmed=True)


def test_the_symmetry_check_passes_a_normally_proportioned_robot():
    check_symmetry(0.42, 0.38, 0.24, 0.24, confirmed=False)
    check_symmetry(0.42, 0.22, 0.24, 0.24, confirmed=False)      # just inside the ratio


def test_the_symmetry_threshold_is_the_thing_that_decides():
    inside = 0.24 * (ASYMMETRY_RATIO - 0.1)
    outside = 0.24 * (ASYMMETRY_RATIO + 0.1)
    check_symmetry(inside, 0.24, 0.24, 0.24, confirmed=False)
    try:
        check_symmetry(outside, 0.24, 0.24, 0.24, confirmed=False)
    except Refusal:
        return
    raise AssertionError("the ratio guard never fires")


def test_an_unconfirmed_stance_is_refused_because_a_prone_outline_is_the_wrong_geometry():
    code = _quiet(lambda: run_main(lambda: main([*_CONTEXT, *_EXTENTS]),
                                   "radius", printer=lambda _line: None))
    assert code == 2


def test_an_unmeasured_extent_is_refused_rather_than_defaulted():
    code = _quiet(lambda: run_main(
        lambda: main([*_CONTEXT, "--front", "0.42", "--stance-confirmed"]),
        "radius", printer=lambda _line: None))
    assert code == 2


def test_a_negative_or_zero_extent_is_refused():
    for value in ("0", "-0.3"):
        code = _quiet(lambda value=value: run_main(
            lambda: main([*_CONTEXT, "--front", value, "--back", "0.38", "--left", "0.24",
                                     "--right", "0.24", "--stance-confirmed"]),
            "radius", printer=lambda _line: None))
        assert code == 2, value


def test_a_complete_measurement_writes_a_provisional_artefact_with_the_payload_on_it():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "radius.json"
        code = _quiet(lambda: main([*_CONTEXT, *_EXTENTS,
                                    "--stance-confirmed", "--artefact", str(path)]))
        assert code == 0
        record = read_record(path)
        assert record.provenance == "provisional"
        assert record.context["payload"] == "mast"
        measurement = record.measurements["loaded_radius"]
        assert abs(measurement["radius_m"] - math.hypot(0.42, 0.24)) < 1e-9
        assert measurement["stance_confirmed"] is True


def test_nothing_in_this_module_opens_a_socket():
    """Structural: the radius is a tape measure and must stay one."""
    import ast

    source = (_HERE / "loaded_radius_probe.py").read_text(encoding="utf-8")
    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.Attribute)}
    assert "socket" not in names and "socket" not in attributes
    for forbidden in ("connect", "sendto", "set_velocity"):
        assert forbidden not in attributes, forbidden


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"loaded_radius_probe: {len(tests)}/{len(tests)} passed")
