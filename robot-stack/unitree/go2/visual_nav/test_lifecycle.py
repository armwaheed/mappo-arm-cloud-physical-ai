#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for cleanup ordering on safety-critical exit paths."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lifecycle import run_cleanup


def _failure():
    raise RuntimeError("camera teardown failed")


def test_a_cleanup_failure_does_not_skip_the_locomotion_stop():
    calls = []
    try:
        run_cleanup("test", [
            ("camera", _failure),
            ("locomotion", lambda: calls.append("locomotion stopped")),
        ])
    except RuntimeError as exc:
        assert "camera teardown" in str(exc)
    else:
        raise AssertionError("a cleanup failure disappeared")
    assert calls == ["locomotion stopped"]


def test_a_cleanup_failure_does_not_replace_the_exception_that_ended_the_run():
    calls = []
    try:
        try:
            raise ValueError("detector failed")
        finally:
            run_cleanup("test", [
                ("camera", _failure),
                ("locomotion", lambda: calls.append("locomotion stopped")),
            ])
    except ValueError as exc:
        assert "detector failed" in str(exc)
    else:
        raise AssertionError("cleanup replaced or swallowed the run failure")
    assert calls == ["locomotion stopped"]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"lifecycle: {len(tests)}/{len(tests)} passed")
