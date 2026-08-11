#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Go2 controller — struct parsing + the abort latch. No DDS.

Feeds synthetic ``wireless_remote`` bytes through the reader, so the arm-then-latch
logic is verified without a robot.

Run: ``python3 test_go2_remote.py``
"""
from __future__ import annotations

import os
import struct
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_remote import Go2Remote, parse_buttons, parse_sticks, button_names  # noqa: E402


def _remote(mask=0, lx=0.0, rx=0.0, ry=0.0, ly=0.0):
    """A 40-byte xRockerBtnDataStruct with the given button mask + stick axes."""
    buf = bytearray(40)
    struct.pack_into("<H", buf, 2, mask)
    struct.pack_into("<fff", buf, 4, lx, rx, ry)
    struct.pack_into("<f", buf, 20, ly)
    return SimpleNamespace(wireless_remote=bytes(buf))


def test_parse_buttons_reads_mask_at_offset_2():
    assert parse_buttons(_remote(mask=0b101).wireless_remote) == 0b101


def test_button_names_bit_order():
    # per the KeyMap tuple: bit 0 = R1, bit 2 = start, bit 4 = R2
    assert button_names(0b101) == ["R1", "start"]
    assert button_names(0b10001) == ["R1", "R2"]


def test_parse_sticks_offsets():
    s = parse_sticks(_remote(lx=0.1, rx=-0.2, ry=0.3, ly=-0.4).wireless_remote)
    assert abs(s["lx"] - 0.1) < 1e-6 and abs(s["rx"] + 0.2) < 1e-6
    assert abs(s["ry"] - 0.3) < 1e-6 and abs(s["ly"] + 0.4) < 1e-6  # ly is at offset 20, not 16


def test_latch_arms_then_trips():
    r = Go2Remote(init_dds=False)
    assert not r.armed() and not r.aborted()
    r._on_state(_remote(mask=0))       # buttons released → arm
    assert r.armed() and not r.aborted()
    r._on_state(_remote(mask=0b1))     # a press → latch abort
    assert r.aborted()


def test_button_held_at_start_does_not_trip():
    r = Go2Remote(init_dds=False)
    r._on_state(_remote(mask=0b10))    # held at startup, before arming
    assert not r.aborted(), "must not latch until it has seen a clean release first"
    r._on_state(_remote(mask=0))       # release → arm
    r._on_state(_remote(mask=0b10))    # now a press latches
    assert r.aborted()


def test_reset_clears_and_redisarms():
    r = Go2Remote(init_dds=False)
    r._on_state(_remote(mask=0))
    r._on_state(_remote(mask=0b1))
    assert r.aborted()
    r.reset()
    assert not r.aborted() and not r.armed()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"go2_remote: {len(tests)}/{len(tests)} passed")
