#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the Go2 MHS driver's RPC plumbing. No fabric, no robot, no SDK.

Stubs the SDK-env worker (`drive_blocking`) and drives every RPC through the async
`_drive` path — the regression guard for the keyword-only `gripper` argument, which a
direct `drive_blocking` call does not exercise (run_in_executor forwards positionally).

Run: ``python3 test_go2_agent.py``
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for mhs_sidecar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import go2_agent as G  # noqa: E402


def _install_fake(record):
    def fake_drive(subcommand, args, drive_py, drive_script, dds_iface, allow_motion,
                   *, gripper=None, timeout=60.0):
        record.append({"cmd": subcommand, "args": list(args or []), "gripper": gripper,
                       "allow_motion": allow_motion})
        return {"ok": True, "pose": {"x": 1.0, "y": 2.0},
                "sent_angles": list(args or []) + ([gripper] if gripper is not None else [])}
    G.drive_blocking = fake_drive


def _mount_events(driver):
    """Wire @emit to a no-op sink so `arrived` / `blocked` / `arm_moved` can fire offline.

    Under the MHS SDK an @emit with no event callback raises RuntimeError
    ("Driver ... has no event callback"); mount()/set_event_callback wires it. Without the
    SDK installed @emit is an inert pass-through and the driver exposes no such method, so
    this is a guarded no-op."""
    if hasattr(driver, "set_event_callback"):
        driver.set_event_callback(lambda *a, **k: None)


def test_all_rpcs_run_through_async_drive():
    rec = []
    _install_fake(rec)
    d = G.Go2AgentDriver(allow_motion=True)
    _mount_events(d)
    assert asyncio.run(d.get_status())["ok"]
    assert asyncio.run(d.stop())["ok"]
    assert asyncio.run(d.stand())["ok"]
    assert asyncio.run(d.stand_down())["ok"]
    assert asyncio.run(d.walk_forward(1.5))["ok"]
    assert asyncio.run(d.walk_to(2.0, 0.5))["ok"]
    assert asyncio.run(d.arm_reset())["ok"]


def test_gripper_threads_through_keyword_only():
    rec = []
    _install_fake(rec)
    d = G.Go2AgentDriver(allow_motion=True)
    _mount_events(d)
    res = asyncio.run(d.arm_move([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], gripper=0.3))
    assert res["ok"]
    last = rec[-1]
    assert last["cmd"] == "arm-move"
    assert last["gripper"] == 0.3          # the keyword-only arg actually arrived
    assert last["args"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_subcommand_names_are_hyphenated():
    rec = []
    _install_fake(rec)
    d = G.Go2AgentDriver(allow_motion=True)
    _mount_events(d)
    asyncio.run(d.stand_down())
    asyncio.run(d.arm_reset())
    cmds = [r["cmd"] for r in rec]
    assert "stand-down" in cmds and "arm-reset" in cmds  # match go2_drive's argparse choices


def test_default_driver_is_status_only():
    d = G.Go2AgentDriver()  # allow_motion defaults False
    assert d.allow_motion is False
    assert "disabled" in d.identity.description


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"go2_agent: {len(tests)}/{len(tests)} passed")
