#!/usr/bin/env python3
# Copyright (c) 2024-2026, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SDK-env worker: execute ONE Go2 command over DDS, print a JSON result, exit.

Runs in the Unitree SDK env (``unitree_sdk2py`` + CycloneDDS, Python 3.8 on the Go2
Jetson). The MHS driver (``driver/go2/go2_agent.py``, a Python ≥3.10 env)
invokes this as a subprocess — the two-env bridge (see the bootstrap-mhs-env skill),
which the Go2 needs because ``unitree_sdk2py`` runs on the robot's 3.8 while MHS
wants ≥3.10.

It reuses the verified control code in ``unitree/go2`` (``Go2Locomotion``, ``D1Arm``)
— the driver adds no new robot logic. Motion subcommands require ``--allow-motion``
(the driver passes it only when configured to command motion), so a default Go2
MHS device is status-only and cannot move by accident.

Usage (one-shot):
  python go2_drive.py status
  python go2_drive.py stand --allow-motion
  python go2_drive.py walk-forward 1.0 --allow-motion
  python go2_drive.py arm-move 0 -0.5 0.5 0 0 0 --gripper 0.3 --allow-motion
"""

from __future__ import annotations

import argparse
import json
import os
import sys

MOTION_COMMANDS = {"stand", "stand-down", "recover", "walk-forward", "walk-to",
                   "arm-reset", "arm-move", "arm-relax"}


def _stack_dir() -> str:
    """The deployed ``unitree/go2`` dir.

    ``GO2_STACK_DIR`` wins; else the on-robot install path if present; else the path
    relative to this file in the repo (so it works from a checkout too).
    """
    if os.environ.get("GO2_STACK_DIR"):
        return os.environ["GO2_STACK_DIR"]
    on_robot = "/home/unitree/robotics-connect/unitree/go2"
    if os.path.isdir(on_robot):
        return on_robot
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "unitree", "go2"))


def _load_locomotion(iface: str):
    sys.path.insert(0, os.path.join(_stack_dir(), "locomotion"))
    from go2_locomotion import Go2Locomotion  # noqa: E402
    loco = Go2Locomotion(iface=iface)
    loco.connect()
    return loco


def _load_arm(iface: str):
    sys.path.insert(0, os.path.join(_stack_dir(), "d1_arm"))
    from d1_arm import D1Arm  # noqa: E402
    arm = D1Arm(iface=iface)
    arm.connect()
    return arm


def _load_safe_stop():
    """arm_mhs_robotkit.safe_stop.SafeStop — the shared guaranteed-damp wrapper for any
    process that commands motors, so a signal (SIGTERM/SIGINT) or exit stops the robot."""
    from arm_mhs_robotkit.safe_stop import SafeStop
    return SafeStop


def _do_locomotion(cmd: str, nums, iface: str, allow_motion: bool) -> dict:
    loco = _load_locomotion(iface)
    try:
        if cmd == "status":
            p = loco.pose()
            return {"ok": True, "mode": loco.current_mode(),
                    "pose": {"x": p.x, "y": p.y, "yaw": p.yaw}}
        if cmd == "stop":
            loco.stop()
            return {"ok": True, "stopped": True}
        # --- motion: wrap in SafeStop so a signal (e.g. the driver's timeout
        # SIGTERM) or exit stops the robot — the repo rule for any motor process ---
        SafeStop = _load_safe_stop()
        with SafeStop(loco.stop, name="go2-drive"):
            loco.ensure_sport_mode("normal")
            if cmd == "stand":
                loco.stand(); return {"ok": True, "stood": True}
            if cmd == "recover":
                loco.recover(); return {"ok": True, "recovered": True}
            if cmd == "stand-down":
                loco.stand_down(); return {"ok": True, "stood_down": True}
            if cmd == "walk-forward":
                if not nums:
                    return {"ok": False, "error": "walk-forward needs a distance (m)"}
                result = loco.walk_forward(float(nums[0]))
                p = loco.pose()
                return {"ok": result is None, "result": result,
                        "pose": {"x": p.x, "y": p.y, "yaw": p.yaw}}
            if cmd == "walk-to":
                if len(nums) < 2:
                    return {"ok": False, "error": "walk-to needs X and Y (m)"}
                result = loco.walk_to((float(nums[0]), float(nums[1])))
                p = loco.pose()
                return {"ok": result is None, "result": result,
                        "pose": {"x": p.x, "y": p.y, "yaw": p.yaw}}
            return {"ok": False, "error": f"unknown locomotion command {cmd!r}"}
    finally:
        # Safety net: stop after any command that could leave the robot moving.
        # A read-only status must NOT command the robot, so skip the stop for it.
        if cmd != "status":
            try:
                loco.stop()
            except Exception:
                pass
        loco.shutdown()


def _do_arm(cmd: str, nums, gripper, iface: str) -> dict:
    arm = _load_arm(iface)
    try:
        if cmd == "arm-reset":
            arm.reset(); return {"ok": True, "reset": True}
        if cmd == "arm-relax":
            arm.relax(); return {"ok": True, "relaxed": True}
        if cmd == "arm-move":
            if len(nums) != 6:
                return {"ok": False, "error": "arm-move needs 6 joint angles (rad)"}
            sent = arm.move_all(nums, gripper_rad=gripper)
            return {"ok": True, "sent_angles": sent}
        return {"ok": False, "error": f"unknown arm command {cmd!r}"}
    finally:
        arm.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description="Go2 SDK-env command worker (one-shot).")
    ap.add_argument("command", choices=sorted(
        {"status", "stop"} | MOTION_COMMANDS))
    ap.add_argument("nums", nargs="*", type=float, help="numeric args (distances / angles)")
    ap.add_argument("--iface", default=os.environ.get("GO2_DDS_IFACE", "eth0"))
    ap.add_argument("--gripper", type=float, default=None, help="jaw angle (rad) for arm-move")
    ap.add_argument("--allow-motion", action="store_true",
                    help="required for any command that moves the robot")
    args = ap.parse_args()

    if args.command in MOTION_COMMANDS and not args.allow_motion:
        print(json.dumps({"ok": False, "error": "motion disabled — pass --allow-motion"}))
        sys.exit(3)

    try:
        if args.command.startswith("arm-"):
            result = _do_arm(args.command, args.nums, args.gripper, args.iface)
        else:
            result = _do_locomotion(args.command, args.nums, args.iface, args.allow_motion)
    except Exception as exc:  # surface as JSON so the driver can report it
        print(json.dumps({"ok": False, "error": repr(exc)}))
        sys.exit(1)

    print(json.dumps(result))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
